"""Backing vocals -> one or more harmony lines.

If the backing stem is basically one voice, it goes through the same CREPE pipeline as the lead.
If it is stacked harmonies, Spotify's basic-pitch (polyphonic) finds the notes and they are
dealt out to lines by pitch rank and continuity (highest harmony = line 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import numpy as np

from .notes import Note


@dataclass
class PolyNote:
    start: float
    end: float
    pitch: int
    amp: float


BP_FPS = 22050 / 256  # basic-pitch frames per second



def basic_pitch_model():
    """basic-pitch's model. Loading the CoreML version (macOS) rewrites files inside the model package, so
    the packaged app (whose own folder must stay untouched) loads it from a copy in its data folder."""
    from basic_pitch import ICASSP_2022_MODEL_PATH as path
    from .paths import APP_DIR, APP_SUPPORT
    path = Path(str(path))
    if APP_DIR is None or not path.is_dir():
        return path
    import shutil
    from . import __version__
    copy = APP_SUPPORT / "cache" / f"basic-pitch-{__version__}" / path.name
    if not copy.exists():
        tmp = copy.with_name(copy.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(path, tmp)
        tmp.rename(copy)
    return copy

def note_activations(path) -> np.ndarray:
    """basic-pitch's polyphonic note posteriorgram (frames x 88 keys from MIDI 21) for a stem."""
    import contextlib
    import io
    import logging
    logging.getLogger("basic_pitch").setLevel(logging.ERROR)
    from basic_pitch.inference import predict
    ICASSP_2022_MODEL_PATH = basic_pitch_model()
    with contextlib.redirect_stdout(io.StringIO()):
        out, _, _ = predict(str(path), ICASSP_2022_MODEL_PATH)
    return out["note"]


def _act(acts: np.ndarray, a: float, b: float, pitch: int) -> float:
    i, j = int(a * BP_FPS), max(int(b * BP_FPS), int(a * BP_FPS) + 1)
    k = pitch - 21
    if not 0 <= k < acts.shape[1] or i >= len(acts):
        return 0.0
    return float(acts[i:j, k].mean())


def refine_with_polyphony(notes, lead, activations: list[np.ndarray], min_act: float = 0.25,
                          search=(-2, -1, 1, 2, -12, 12), own_bonus: float = 0.05):
    """Check each harmony note against a polyphonic transcriber (basic-pitch), which hears several voices
    at once where a monophonic reading of the backing stem can lock onto the wrong one. A note moves to
    the nearby pitch (within 2 semitones, or an octave) where a voice is clearly singing, never onto the
    lead's own pitch or its octave; a note with no voice anywhere near it is dropped.
    Returns (notes, moved, dropped)."""
    lead = sorted(lead, key=lambda m: m.start)
    starts = np.array([m.start for m in lead])
    out, moved, dropped = [], 0, 0
    for n in notes:
        mid = 0.5 * (n.start + n.end)
        k = int(np.searchsorted(starts, mid)) - 1
        lp = lead[k].tone if 0 <= k < len(lead) and lead[k].end > mid else None
        p = int(round(n.pitch))
        cands = [p] + [p + d for d in search]
        cands = [q for q in cands if lp is None or (abs(q - lp) >= 1 and abs(abs(q - lp) - 12) >= 1)]
        if not cands:
            dropped += 1
            continue
        score = {q: max(_act(A, n.start, n.end, q) for A in activations) + (own_bonus if q == p else 0.0)
                 for q in cands}
        best = max(score, key=score.get)
        if score[best] - (own_bonus if best == p else 0.0) < min_act:
            dropped += 1
            continue
        if best != p:
            n.pitch = float(best) + (n.pitch - p)
            moved += 1
        out.append(n)
    return out, moved, dropped


def detect_poly_notes(path, min_len: float = 0.09, progress=None) -> list[PolyNote]:
    import logging
    logging.getLogger("basic_pitch").setLevel(logging.ERROR)
    from basic_pitch.inference import predict
    ICASSP_2022_MODEL_PATH = basic_pitch_model()

    if progress:
        progress("[harmony] finding stacked harmony notes with basic-pitch")
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, events = predict(str(path), ICASSP_2022_MODEL_PATH, onset_threshold=0.45, frame_threshold=0.3,
                               minimum_note_length=min_len * 1000, minimum_frequency=70, maximum_frequency=1300,
                               multiple_pitch_bends=False, melodia_trick=True)
    notes = [PolyNote(float(s), float(e), int(p), float(a)) for s, e, p, a, *_ in events]
    notes.sort(key=lambda n: (n.start, -n.pitch))
    return _drop_overtones(notes)


def _drop_overtones(notes: list[PolyNote]) -> list[PolyNote]:
    """basic-pitch sometimes reports the 2nd/3rd harmonic of a voice as its own note."""
    keep = []
    for n in notes:
        ghost = False
        for m in notes:
            if m is n or m.amp < n.amp * 0.8:
                continue
            ov = min(n.end, m.end) - max(n.start, m.start)
            if ov > 0.7 * (n.end - n.start) and (n.pitch - m.pitch) in (12, 19, 24, 28, 31):
                ghost = True
                break
        if not ghost:
            keep.append(n)
    return keep


# The singer's own harmonics read as notes: 2nd..8th harmonic (octave, octave+5th, two octaves,
# two octaves+3rd, +5th, +7th, three octaves), +-1 semitone.
OVERTONES = {11, 12, 13, 18, 19, 20, 23, 24, 25} | set(range(27, 38))


def remove_lead_duplicates(notes, lead, min_overlap: float = 0.5, tail_s: float = 0.6):
    """Drop backing notes that are really the lead: bleed at the same pitch, overtones of the lead
    an octave (or more) above it, and reverb/echo tails right after a lead note ends."""
    if not lead:
        return notes
    lead = sorted(lead, key=lambda m: m.start)
    starts = np.array([m.start for m in lead])
    out = []
    for n in notes:
        dur = n.end - n.start
        i0 = max(0, int(np.searchsorted(starts, n.start - 10.0)))
        dup = False
        overlapped = False
        for m in lead[i0:]:
            if m.start > n.end:
                break
            ov = min(n.end, m.end) - max(n.start, m.start)
            if ov > 0.3 * dur:
                overlapped = True
            if ov > min_overlap * dur:
                d = round(n.pitch - m.tone)
                if abs(d) <= 1 or d in OVERTONES:
                    dup = True
                    break
        if not dup and not overlapped:
            k = int(np.searchsorted(starts, n.start + 0.05)) - 1
            if k >= 0:
                m = lead[k]
                d = round(n.pitch - m.tone)
                if 0 <= n.start - m.end < tail_s and (abs(d) <= 1 or d in OVERTONES):
                    dup = True
        if not dup:
            out.append(n)
    return out


def remove_quiet(notes: list[PolyNote], path, ref_db: float, rel_db: float = -20.0) -> list[PolyNote]:
    """Drop notes where the backing stem is far quieter than the song's singing: separation
    leakage (instrument bleed, reverb tails) rather than real backing vocals."""
    from .pitch import HOP, SR, load_mono
    y = load_mono(path)
    pad = np.pad(y, (512, 512))
    frames = np.lib.stride_tricks.sliding_window_view(pad, 1024)[::HOP]
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12) + 1e-9)
    hop_s = HOP / SR
    keep = []
    for n in notes:
        i, j = int(n.start / hop_s), max(int(n.end / hop_s), int(n.start / hop_s) + 1)
        seg = rms_db[i:j]
        if len(seg) and np.percentile(seg, 75) >= ref_db + rel_db:
            keep.append(n)
    return keep


def polyphony_profile(notes: list[PolyNote], hop: float = 0.02) -> np.ndarray:
    if not notes:
        return np.zeros(0, int)
    end = max(n.end for n in notes)
    count = np.zeros(int(end / hop) + 2, int)
    for n in notes:
        count[int(n.start / hop):int(n.end / hop)] += 1
    return count


def estimate_lines(notes: list[PolyNote], max_lines: int = 4) -> int:
    prof = polyphony_profile(notes)
    active = prof[prof > 0]
    if len(active) == 0:
        return 0
    # number of voices present for a meaningful share of the sung time
    for k in range(max_lines, 1, -1):
        if (active >= k).mean() > 0.12:
            return k
    return 1


def allocate_lines(notes: list[PolyNote], k: int) -> list[list[PolyNote]]:
    lines: list[list[PolyNote]] = [[] for _ in range(k)]
    for n in sorted(notes, key=lambda x: (x.start, -x.pitch)):
        mid = 0.5 * (n.start + n.end)
        active = sorted([m for m in notes if m.start <= mid < m.end], key=lambda m: -m.pitch)
        rank = active.index(n) if n in active else 0
        free = [i for i in range(k) if not lines[i] or lines[i][-1].end <= n.start + 0.04]
        if not free:
            continue
        if len(active) >= k and rank in free:
            choice = rank
        else:
            def cost(i):
                last = lines[i][-1].pitch if lines[i] else None
                cont = abs(last - n.pitch) if last is not None else 6
                rank_pen = abs(i - rank) * 2.5 if len(active) > 1 else 0
                return cont + rank_pen
            choice = min(free, key=cost)
        if lines[choice] and lines[choice][-1].end > n.start:
            lines[choice][-1].end = n.start  # trim tiny overlaps
        lines[choice].append(n)
    return [ln for ln in lines if ln]


def drop_isolated(notes, gap_s: float = 1.2, min_notes: int = 3, min_dur_s: float = 1.0):
    """A harmony part comes in phrases; a lone note or two with nothing around them is leakage (a word
    of the lead bleeding through, a reverb tail). Keep groups with >= min_notes notes or >= min_dur_s."""
    notes = sorted(notes, key=lambda n: n.start)
    out, group = [], []
    for n in notes + [None]:
        if n is not None and group and n.start - group[-1].end < gap_s:
            group.append(n)
            continue
        if group and (len(group) >= min_notes or sum(x.end - x.start for x in group) >= min_dur_s):
            out.extend(group)
        group = [n] if n is not None else []
    return out


def to_notes(line: list[PolyNote]) -> list[Note]:
    return [Note(n.start, n.end, float(n.pitch)) for n in line if n.end - n.start >= 0.06]
