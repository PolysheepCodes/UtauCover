"""Turn a pitch track into discrete notes.

Each voiced region is cut into piecewise-constant pitch segments with dynamic programming.
The cost of a segment is its squared pitch deviation plus a fixed penalty and a penalty that
grows for short segments, so vibrato and scoops don't shatter a held note while real note
changes (even fast ones) still split. Lyric unit start times can force extra boundaries so
repeated notes on the same pitch ("la la la") are separated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pitch import F0Track, _runs


@dataclass
class Note:
    start: float
    end: float
    pitch: float              # fractional MIDI, tuning-corrected
    lyric: str = ""
    unit: int = -1            # index of the lyric unit this note belongs to (-1 = none)
    is_unit_start: bool = False
    forced: bool = False      # start boundary came from a lyric timing
    contour: np.ndarray | None = field(default=None, repr=False)

    @property
    def tone(self) -> int:
        return int(round(self.pitch))

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class SegParams:
    lam: float = 2.0          # per-note penalty (semitone^2 * frames)
    mu: float = 40.0          # short-note penalty numerator (frames)
    min_frames: int = 5       # shortest segment the DP may create (50 ms)
    min_note_s: float = 0.10  # shorter notes are merged into a neighbour (transition blips)
    glide_s: float = 0.20     # short segments that are pure slides get merged
    split_search_s: float = 0.08
    dip_db: float = 9.0       # a loudness dip this deep (vs. both sides) is a syllable boundary
    dip_window_s: float = 0.15
    min_peak_rel_db: float = -20.0  # notes this far below the nearby singing (±8 s) are bleed
    bleed_window_s: float = 8.0
    quiet_group_db: float = -13.0   # whole stretches (between 1.5 s silences) this far below the song's
    group_gap_s: float = 1.5        # vocal level are leakage, e.g. instruments in an instrumental break


def smooth_contour(m: np.ndarray) -> np.ndarray:
    from scipy.ndimage import median_filter
    c = median_filter(m, size=5, mode="nearest")
    # octave / spike errors: compare against a wider median
    wide = median_filter(c, size=21, mode="nearest")
    bad = np.abs(c - wide) > 5.0
    c[bad] = wide[bad]
    return c


def _pc_segments(c: np.ndarray, p: SegParams) -> list[tuple[int, int]]:
    n = len(c)
    lmin = p.min_frames
    if n < 2 * lmin:
        return [(0, n)]
    S1 = np.concatenate([[0.0], np.cumsum(c)])
    S2 = np.concatenate([[0.0], np.cumsum(c * c)])
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    back = np.zeros(n + 1, dtype=np.int64)
    for j in range(lmin, n + 1):
        i = np.arange(0, j - lmin + 1)
        L = (j - i).astype(np.float64)
        cost = (S2[j] - S2[i]) - (S1[j] - S1[i]) ** 2 / L + p.lam + p.mu / L
        tot = best[i] + cost
        k = int(np.argmin(tot))
        best[j] = tot[k]
        back[j] = i[k]
    segs = []
    j = n
    while j > 0:
        i = int(back[j])
        segs.append((i, j))
        j = i
    return segs[::-1]


def _seg_pitch(c: np.ndarray) -> float:
    n = len(c)
    a, b = int(n * 0.2), max(int(n * 0.8), int(n * 0.2) + 1)
    core = c[a:b] if b - a >= 2 else c
    return float(np.median(core))


def _refine_split(track: F0Track, f: int, lo: int, hi: int, search: int) -> int:
    a, b = max(lo + 1, f - search), min(hi - 1, f + search)
    if b <= a:
        return f
    seg = track.rms_db[a:b]
    return int(a + np.argmin(seg))


def energy_dips(rms_db: np.ndarray, a: int, b: int, depth_db: float, window: int) -> list[int]:
    """Frames inside [a, b) where loudness dips sharply below both neighbours: consonants between
    syllables sung on the same pitch ("ten feet twenty")."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    seg = uniform_filter1d(rms_db[a:b], 3)
    if len(seg) < 2 * window:
        return []
    out = []
    i = window
    while i < len(seg) - window:
        if seg[i] == seg[max(0, i - 2):i + 3].min():
            left = seg[max(0, i - window):i].max()
            right = seg[i + 1:i + 1 + window].max()
            if min(left, right) - seg[i] >= depth_db:
                out.append(a + i)
                i += window // 2
                continue
        i += 1
    return out


def segment_notes(track: F0Track, tuning: float = 0.0, splits: list[float] | None = None,
                  params: SegParams | None = None) -> list[Note]:
    p = params or SegParams()
    hop = track.hop_s
    midi = track.midi - tuning
    split_frames = sorted(int(round(t / hop)) for t in (splits or []))
    search = int(p.split_search_s / hop)
    loud_ref = np.percentile(track.rms_db[track.rms_db > -100], 95) if np.any(track.rms_db > -100) else -20
    notes: list[Note] = []

    for v, a, b in _runs(track.voiced):
        if not v:
            continue
        c = smooth_contour(np.nan_to_num(midi[a:b], nan=np.nanmedian(midi[a:b])))
        # forced boundaries inside this region
        cuts = []
        for f in split_frames:
            if a + p.min_frames <= f <= b - p.min_frames:
                cuts.append(_refine_split(track, f, a, b, search))
        for f in energy_dips(track.rms_db, a, b, p.dip_db, int(p.dip_window_s / hop)):
            if a + p.min_frames <= f <= b - p.min_frames:
                cuts.append(f)
        bounds = sorted(set([a] + cuts + [b]))
        bounds = [x for i, x in enumerate(bounds) if i == 0 or x - bounds[i - 1] >= p.min_frames // 2 + 1]
        if bounds[-1] != b:
            bounds[-1] = b
        region_notes: list[Note] = []
        for k in range(len(bounds) - 1):
            s0, s1 = bounds[k] - a, bounds[k + 1] - a
            sub = c[s0:s1]
            for i, j in _pc_segments(sub, p):
                seg = sub[i:j]
                region_notes.append(Note(start=(a + s0 + i) * hop, end=(a + s0 + j) * hop,
                                         pitch=_seg_pitch(seg), forced=(i == 0 and k > 0),
                                         contour=seg.copy()))
        notes.extend(_cleanup_region(region_notes, p))
    # Drop quiet bleed (instruments leaking into the vocal stem): a note far quieter than the
    # singing around it. Judged locally, so a soft verse is compared with itself, not the chorus.
    peaks = []
    for n in notes:
        i, j = int(n.start / hop), max(int(n.end / hop), int(n.start / hop) + 1)
        peaks.append(float(track.rms_db[i:j].max()))
    starts = np.array([n.start for n in notes])
    keep = []
    for k, n in enumerate(notes):
        lo = np.searchsorted(starts, n.start - p.bleed_window_s)
        hi = np.searchsorted(starts, n.start + p.bleed_window_s)
        local = max(peaks[lo:hi])
        if peaks[k] >= local + p.min_peak_rel_db and peaks[k] >= loud_ref - 40:
            keep.append((n, peaks[k]))
    # Stretches of "singing" that are quiet throughout (median note level far below the song's vocal
    # level) are leakage. Real verses, even soft ones, sit much closer to the song's level.
    out: list[Note] = []
    group: list = []
    for item in keep + [None]:
        if item is not None and (not group or item[0].start - group[-1][0].end <= p.group_gap_s):
            group.append(item)
            continue
        if group and np.median([pk for _, pk in group]) >= loud_ref + p.quiet_group_db:
            out.extend(n for n, _ in group)
        group = [item] if item is not None else []
    return fix_spikes(out)


def fix_spikes(notes: list[Note], jump: float = 8.5, max_dur: float = 0.25, max_run: float = 0.35,
               glued: float = 0.15, rest: float = 0.3, max_len: int = 3) -> list[Note]:
    """Short notes far above or below the notes they sit between are pitch-tracking slips (an overtone,
    a breath or a separation artefact read as pitch), not sung leaps: give them the neighbour's pitch.

    A "run" is one to `max_len` short notes glued together (<= max_run in all); the shortest run that
    qualifies wins. It's a slip when every note in it is >= `jump` semitones away, in the same direction,
    from each glued neighbour. At a phrase edge (one neighbour, a rest on the other side) the neighbour
    must be the longer note and the jump a bit bigger."""
    def glued_to(x: Note, y: Note) -> bool:
        return y.start - x.end <= glued

    def slip(i: int, j: int) -> Note | None:
        run = notes[i:j + 1]
        prv = notes[i - 1] if i > 0 and glued_to(notes[i - 1], run[0]) else None
        nxt = notes[j + 1] if j + 1 < len(notes) and glued_to(run[-1], notes[j + 1]) else None
        anchors = [x for x in (prv, nxt) if x is not None]
        if not anchors:
            return None
        if len(anchors) == 1:
            if prv is not None:
                far = notes[j + 1].start - run[-1].end if j + 1 < len(notes) else 9.0
            else:
                far = run[0].start - notes[i - 1].end if i > 0 else 9.0
            if far < rest or anchors[0].dur < max(r.dur for r in run):
                return None
        need = jump + (0.5 if len(anchors) == 1 else 0.0)
        devs = [r.pitch - x.pitch for r in run for x in anchors]
        if not (all(d >= need for d in devs) or all(d <= -need for d in devs)):
            return None
        mean = float(np.mean([r.pitch for r in run]))
        return min(anchors, key=lambda x: abs(mean - x.pitch))

    i = 0
    while i < len(notes):
        done = False
        for j in range(i, min(i + max_len, len(notes))):
            if notes[j].dur > max_dur or notes[j].end - notes[i].start > max_run or \
                    (j > i and not glued_to(notes[j - 1], notes[j])):
                break
            ref = slip(i, j)
            if ref is not None:
                for r in notes[i:j + 1]:
                    r.pitch, r.contour = ref.pitch, None
                i = j + 1
                done = True
                break
        if not done:
            i += 1
    return notes


def fix_octave_jumps(notes: list[Note], glued: float = 0.15, max_dur: float = 0.8, near: float = 5.0) -> list[Note]:
    """A note that leaps ~an octave away from both neighbours and back (\"win\" 62, \"dow\" 49, \"some\" 59)
    is a transcriber's octave slip: move it the octave that puts it beside its neighbours."""
    for i in range(1, len(notes) - 1):
        n, a, b = notes[i], notes[i - 1], notes[i + 1]
        if n.dur > max_dur or n.start - a.end > glued or b.start - n.end > glued:
            continue
        d1, d2 = n.pitch - a.pitch, n.pitch - b.pitch
        if d1 * d2 <= 0 or min(abs(d1), abs(d2)) < 9:
            continue
        shifted = n.pitch - 12 if d1 > 0 else n.pitch + 12
        if abs(shifted - a.pitch) <= near and abs(shifted - b.pitch) <= near:
            n.pitch, n.contour = shifted, None
    return notes


def split_hidden_pitches(notes: list[Note], track: F0Track, tuning: float = 0.0, min_s: float = 0.18,
                         min_st: float = 1.5, steady_st: float = 0.6) -> list[Note]:
    """Where a note covers a second sung pitch (a steady stretch of the pitch track >= min_s long and
    >= min_st away, read in the note's own octave), give that stretch its own note: the transcriber
    missed a note change there."""
    hop = track.hop_s
    out: list[Note] = []
    for n in notes:
        i, j = int(n.start / hop), int(n.end / hop)
        m = track.midi[i:j] - tuning
        v = track.voiced[i:j] & ~np.isnan(m)
        if v.sum() < int(2 * min_s / hop):
            out.append(n)
            continue
        d = np.where(v, m - n.pitch, 0.0)
        d = d - 12 * np.round(d / 12)
        off = v & (np.abs(d) >= min_st)
        best = None  # longest steady off-pitch run
        k = 0
        while k < len(off):
            if not off[k]:
                k += 1
                continue
            s = k
            while k + 1 < len(off) and off[k + 1] and abs(m[k + 1] - m[k]) < steady_st:
                k += 1
            if (k + 1 - s) * hop >= min_s and (best is None or k + 1 - s > best[1] - best[0]):
                best = (s, k + 1)
            k += 1
        if best is None:
            out.append(n)
            continue
        s, e = best
        alt = n.pitch + float(np.median(d[s:e]))
        cut_a, cut_b = n.start + s * hop, n.start + e * hop
        pieces = []
        if cut_a - n.start >= 0.06:
            pieces.append(Note(n.start, cut_a, n.pitch, forced=n.forced))
        pieces.append(Note(cut_a if pieces else n.start, cut_b if n.end - cut_b >= 0.06 else n.end, alt,
                           forced=bool(pieces) or n.forced))
        if n.end - cut_b >= 0.06:
            pieces.append(Note(cut_b, n.end, n.pitch, forced=True))
        out.extend(pieces)
    return out


def fill_dropouts(notes: list[Note], track: F0Track, tuning: float = 0.0, min_s: float = 0.2,
                  rel_db: float = -20.0, context_s: float = 1.5) -> tuple[list[Note], float]:
    """Safety net for a transcriber that skips a sung stretch: where the pitch track is clearly singing
    (voiced, within rel_db of the song's level) for >= min_s and no note covers it, add notes from the
    built-in segmentation, in the octave of the surrounding notes. Returns (notes, seconds filled)."""
    hop = track.hop_s
    n = len(track.voiced)
    cov = np.zeros(n, bool)
    for x in notes:
        cov[max(0, int(x.start / hop) - 5):int(x.end / hop) + 5] = True
    ref = np.percentile(track.rms_db[track.voiced], 90) if track.voiced.any() else -20.0
    sung = track.voiced & (track.rms_db > ref + rel_db)
    gaps = [(a * hop, b * hop) for v, a, b in _runs(sung & ~cov) if v and (b - a) * hop >= min_s]
    if not gaps:
        return notes, 0.0
    builtin = segment_notes(track, tuning, [])
    added: list[Note] = []
    for a, b in gaps:
        for m in builtin:
            s, e = max(m.start, a - 0.05), min(m.end, b + 0.05)
            if e - s < 0.08:
                continue
            near = [x.pitch for x in notes if abs(0.5 * (x.start + x.end) - 0.5 * (s + e)) <= context_s]
            p = m.pitch
            if near:
                ctx = float(np.median(near))
                while p - ctx > 7:
                    p -= 12
                while ctx - p > 7:
                    p += 12
            added.append(Note(s, e, p))
    out = sorted(notes + added, key=lambda x: x.start)
    mine = {id(x) for x in added}
    for x, y in zip(out, out[1:]):  # no overlaps with the transcriber's own notes
        if y.start < x.end:
            if id(y) in mine:
                y.start = x.end
            else:
                x.end = y.start
    out = [x for x in out if x.end - x.start >= 0.06]
    return out, float(sum(b - a for a, b in gaps))


def _merge(a: Note, b: Note) -> Note:
    contour = np.concatenate([a.contour, b.contour]) if a.contour is not None and b.contour is not None else None
    pitch = _seg_pitch(contour) if contour is not None else (a.pitch * a.dur + b.pitch * b.dur) / (a.dur + b.dur)
    # keep the dominant pitch: whichever part is longer decides when they round differently
    if contour is not None and round(a.pitch) != round(b.pitch):
        pitch = a.pitch if a.dur >= b.dur else b.pitch
    return Note(a.start, b.end, pitch, forced=a.forced, contour=contour)


def _is_glide(n: Note) -> bool:
    c = n.contour
    if c is None or len(c) < 3:
        return False
    rng = np.percentile(c, 90) - np.percentile(c, 10)
    d = np.diff(c)
    mono = max((d > 0).mean(), (d < 0).mean())
    return rng > 0.8 and mono > 0.7


def _cleanup_region(notes: list[Note], p: SegParams) -> list[Note]:
    if not notes:
        return notes
    changed = True
    while changed and len(notes) > 1:
        changed = False
        # 1) same-tone neighbours merge (unless the boundary is lyric-forced)
        out = [notes[0]]
        for n in notes[1:]:
            prev = out[-1]
            if n.tone == prev.tone and not n.forced:
                out[-1] = _merge(prev, n)
                changed = True
            else:
                out.append(n)
        notes = out
        # 2) too-short or slide segments merge into the closer-pitched neighbour
        for idx, n in enumerate(notes):
            short = n.dur < p.min_note_s or (n.dur < p.glide_s and _is_glide(n))
            if not short or len(notes) == 1:
                continue
            left = notes[idx - 1] if idx > 0 else None
            right = notes[idx + 1] if idx + 1 < len(notes) else None
            if n.forced and left is not None:
                # a lyric-forced short note: give it to the right neighbour so the boundary survives
                if right is not None and not right.forced:
                    m = _merge(n, right)
                    m.pitch = right.pitch
                    m.forced = True
                    notes[idx:idx + 2] = [m]
                    changed = True
                    break
                continue
            cand = []
            if left is not None:
                cand.append((abs(left.pitch - n.pitch), "L"))
            if right is not None and not right.forced:
                cand.append((abs(right.pitch - n.pitch), "R"))
            if not cand:
                continue
            side = min(cand)[1]
            if side == "L":
                m = _merge(left, n)
                m.pitch = left.pitch
                notes[idx - 1:idx + 1] = [m]
            else:
                m = _merge(n, right)
                m.pitch = right.pitch
                m.forced = n.forced
                notes[idx:idx + 2] = [m]
            changed = True
            break
    return notes
