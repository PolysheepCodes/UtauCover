"""Decide which stem holds the main singer, moment by moment, and whether there are real backing vocals.

Karaoke (lead/backing) separation models guess "lead" vs "backing" from how a voice is mixed. On a
song with one singer they often hand whole lines (reverb-heavy, doubled or quieter passages) to
"backing", leaving the lead stem with only a faint residue that pitch trackers read an octave or
more too high. So the main vocal line is assembled from the lead stem where it really holds the
singer, and from the full vocal stem where the split took the singer away.

Harmony tracks are only written when the backing stem has its own part: notes that the backing
stem's own pitch confirms, that aren't the singer's unison, octave or overtones, and that add up to
more than a few seconds.
"""

from __future__ import annotations

import numpy as np

from .pitch import F0Track, _runs, clean_mask

# Intervals (semitones above the singer) where a separation residue or a polyphonic transcriber
# reports the singer's own harmonics as notes: 2nd..8th harmonic, +-1 semitone.
HARMONIC_INTERVALS = (12, 19, 24, 28, 31, 34, 36)


def is_harmonic(d: float, tol: float = 1.0) -> bool:
    return any(abs(d - h) <= tol for h in HARMONIC_INTERVALS)


def main_voice_mask(lead: F0Track, allv: F0Track, backing: F0Track | None = None,
                    moved_db: float = 4.5, lost_db: float = 12.0, alone_rel_db: float = -9.0,
                    loud_rel_db: float = -14.0, smooth_s: float = 0.2, min_run_s: float = 0.15) -> np.ndarray:
    """Frames where the main voice should be taken from the full vocal stem instead of the lead stem.

    * lost:  the lead stem is (nearly) silent while the full mix sings (> lost_db quieter);
    * moved: the lead stem is clearly quieter than the full mix (> moved_db) and the backing stem is
             singing the full mix's melody (same pitch or its octave), i.e. it holds the singer;
    * alone: the lead stem isn't singing, the backing stem is, as loud as the song's main vocal.
    """
    n = min(len(lead.midi), len(allv.midi), len(backing.midi) if backing is not None else 10 ** 9)
    hop = allv.hop_s
    v = allv.voiced[:n]
    if not v.any():
        return np.zeros(n, bool)
    ref = float(np.percentile(allv.rms_db[:n][v], 90))
    k = max(1, int(smooth_s / hop)) | 1
    a = _median(allv.rms_db[:n] - lead.rms_db[:n], k)
    level = _median(allv.rms_db[:n], k)
    loud = level > ref + loud_rel_db
    lost = a > lost_db
    moved = a > moved_db
    if backing is not None:
        dd = np.abs(backing.midi[:n] - allv.midi[:n])
        carries = backing.voiced[:n] & v & ((dd < 1.0) | (np.abs(dd - 12) < 1.0))
        carries = _median(carries.astype(float), k) > 0.5
        moved &= carries
    alone = ~lead.voiced[:n] & (a > moved_db) & (_median(allv.rms_db[:n], 5 * k) > ref + alone_rel_db)
    use = v & loud & (lost | moved | alone)
    m = max(1, int(min_run_s / hop))
    return clean_mask(use, fill_gap=m, min_run=m)


def mask_spans(mask: np.ndarray, hop: float, pad_s: float = 0.03) -> list[tuple[float, float]]:
    return [(max(0.0, a * hop - pad_s), b * hop + pad_s) for v, a, b in _runs(mask) if v]


def fix_octaves(track: F0Track, alt: F0Track, context_s: float = 1.0, tol: float = 1.2,
                min_context: int = 10) -> tuple[F0Track, float]:
    """Where two pitch readings of the same singer disagree by an octave, keep the one that fits the
    melody around it (judged from the frames where both readings agree). Returns (track, seconds changed)."""
    n = min(len(track.midi), len(alt.midi))
    m = track.midi.copy()
    a = alt.midi[:n]
    both = track.voiced[:n] & alt.voiced[:n] & ~np.isnan(m[:n]) & ~np.isnan(a)
    d = np.where(both, m[:n] - a, 0.0)
    agree = both & (np.abs(d) < 1.0)
    octave = both & (np.abs(np.abs(d) - 12) < tol)
    if not octave.any():
        return track, 0.0
    w = int(context_s / track.hop_s)
    idx = np.flatnonzero(agree)
    vals = m[:n][agree]
    changed = 0
    for on, s, e in _runs(clean_mask(octave, fill_gap=5, min_run=1)):
        if not on:
            continue
        lo, hi = np.searchsorted(idx, s - w), np.searchsorted(idx, e + w)
        if hi - lo < min_context:
            continue
        ctx = float(np.median(vals[lo:hi]))
        sel = np.flatnonzero(octave[s:e]) + s
        if abs(float(np.median(a[sel])) - ctx) + 2.0 < abs(float(np.median(m[sel])) - ctx):
            m[sel] = a[sel]
            changed += len(sel)
    if not changed:
        return track, 0.0
    hz = np.where(np.isnan(m), np.nan, 440.0 * 2 ** ((m - 69) / 12))
    return F0Track(track.times, hz, m, track.periodicity, track.rms_db, track.voiced, track.hop_s), changed * track.hop_s


def octave_doubts(notes, readings: list[F0Track], tol: float = 1.0, min_cover: float = 0.5,
                  quorum: int = 1) -> list[tuple[object, int]]:
    """Notes where at least `quorum` other pitch readings of the singer put them an octave away (and
    none agrees with the note). Returns (note, other midi)."""
    out = []
    for n in notes:
        away, agree, other = 0, 0, 0
        for r in readings:
            i, j = r.frame(n.start), r.frame(n.end)
            if j - i < 3:
                continue
            v = r.voiced[i:j] & ~np.isnan(r.midi[i:j])
            if v.mean() < min_cover:
                continue
            d = float(np.median(r.midi[i:j][v])) - n.pitch
            if abs(abs(d) - 12) <= tol:
                away += 1
                other = int(round(n.pitch + (12 if d > 0 else -12)))
            elif abs(d) <= tol:
                agree += 1
        if away >= quorum and (quorum == 1 or agree == 0):
            out.append((n, other))
    return out


def backing_evidence(poly_notes, backing: F0Track | None, lead_notes, allv: F0Track | None = None,
                     min_support: float = 0.5, min_level_db: float = -12.0) -> float:
    """Seconds of backing singing that is clearly its own part (see module docstring), measured against the
    final lead notes. The backing stem's own pitch must confirm the note (in any octave: its pitch reading
    can slip an octave too). With no separate backing stem (backing is None), every note that isn't the
    singer's unison/octave/overtone counts."""
    lead = sorted(lead_notes, key=lambda n: n.start)
    starts = np.array([n.start for n in lead])
    total = 0.0
    for p in poly_notes:
        if backing is not None:
            hop = backing.hop_s
            i, j = int(p.start / hop), int(p.end / hop) + 1
            bv = backing.voiced[i:j]
            if bv.sum() < 3:
                continue
            dd = np.abs(backing.midi[i:j][bv] - p.pitch)
            support = float(np.mean((dd < 1.0) | (np.abs(dd - 12) < 1.0))) * float(bv.mean())
            if support < min_support:
                continue
            if allv is not None:
                lvl = float(np.median(backing.rms_db[i:j]) - np.median(allv.rms_db[i:j]))
                if lvl < min_level_db:
                    continue
        mid = 0.5 * (p.start + p.end)
        k = int(np.searchsorted(starts, mid)) - 1
        if 0 <= k < len(lead) and lead[k].end > mid:
            d = p.pitch - lead[k].pitch
            if abs(d) < 1.0 or abs(d + 12) < 1.0 or is_harmonic(d):
                continue
        total += p.end - p.start
    return total


def _median(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    from scipy.ndimage import median_filter
    return median_filter(x, size=k, mode="nearest")
