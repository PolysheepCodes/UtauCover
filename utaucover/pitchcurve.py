"""Copy the original singer's pitch movement (slides, vibrato, drift) into OpenUTAU's
'pitd' (pitch deviation, cents) curve, relative to the notes we placed."""

from __future__ import annotations

import numpy as np

from .notes import Note
from .pitch import F0Track

PORTA_MS = 25.0  # matches the default pitch points we write on each note (x: -25 .. 25 ms)


def _base_pitch(times: np.ndarray, notes: list[Note]) -> np.ndarray:
    """Approximate what OpenUTAU will sing before any curve: note tones plus the short
    default portamento into each note that directly follows another."""
    base = np.full(len(times), np.nan)
    p = PORTA_MS / 1000
    for i, n in enumerate(notes):
        m = (times >= n.start) & (times < n.end)
        base[m] = n.tone
    for i in range(1, len(notes)):
        a, b = notes[i - 1], notes[i]
        if b.start - a.end > 0.002 or a.tone == b.tone:
            continue
        m = (times >= b.start - p) & (times < b.start + p)
        x = (times[m] - (b.start - p)) / (2 * p)
        ease = 0.5 - 0.5 * np.cos(np.pi * x)
        base[m] = a.tone + (b.tone - a.tone) * ease
    return base


def _rdp(xs: np.ndarray, ys: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker keep-mask."""
    keep = np.zeros(len(xs), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(xs) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        xa, ya, xb, yb = xs[a], ys[a], xs[b], ys[b]
        seg = slice(a + 1, b)
        if xb == xa:
            d = np.abs(ys[seg] - ya)
        else:
            d = np.abs(ys[seg] - (ya + (yb - ya) * (xs[seg] - xa) / (xb - xa)))
        k = int(np.argmax(d))
        if d[k] > eps:
            idx = a + 1 + k
            keep[idx] = True
            stack += [(a, idx), (idx, b)]
    return keep


def pitch_deviation_curve(track: F0Track, notes: list[Note], tuning: float, sec_to_tick,
                          interval: int = 5, eps_cents: float = 4.0) -> tuple[list[int], list[int]]:
    """Return (ticks, cents) for a pitd curve. sec_to_tick maps seconds -> absolute ticks."""
    notes = sorted(notes, key=lambda n: n.start)
    if not notes:
        return [], []
    base = _base_pitch(track.times, notes)
    dev = (track.midi - tuning - base) * 100.0
    ok = track.voiced & ~np.isnan(dev)
    dev = np.where(ok, dev, 0.0)
    # A reading an octave (or two) away from the note is the tracker slipping onto an overtone or
    # sub-octave, not a bend: fold it back so it can't drag the note a whole octave.
    far = np.abs(dev) > 900
    dev[far] -= 1200 * np.round(dev[far] / 1200)
    dev = np.clip(dev, -700, 700)
    # resample onto OpenUTAU's 5-tick curve grid
    t0, t1 = notes[0].start, notes[-1].end
    tick0, tick1 = int(sec_to_tick(t0)) // interval * interval, int(sec_to_tick(t1)) + interval
    ticks = np.arange(tick0, tick1, interval)
    # invert sec_to_tick (it is linear)
    a = sec_to_tick(1.0) - sec_to_tick(0.0)
    b = sec_to_tick(0.0)
    secs = (ticks - b) / a
    vals = np.interp(secs, track.times, dev)
    okr = np.interp(secs, track.times, ok.astype(float)) > 0.5
    vals = np.where(okr, vals, 0.0)
    vals = np.round(vals)
    keep = _rdp(ticks.astype(float), vals, eps_cents)
    # keep zero-crossings into/out of unvoiced stretches crisp
    edges = np.flatnonzero(np.diff(okr.astype(int)) != 0)
    keep[edges] = True
    keep[np.clip(edges + 1, 0, len(keep) - 1)] = True
    return [int(x) for x in ticks[keep]], [int(y) for y in vals[keep]]
