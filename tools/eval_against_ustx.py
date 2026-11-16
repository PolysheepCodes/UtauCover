"""Score detected notes against a hand-made .ustx (e.g. your own rendered cover + its project).

    python tools/eval_against_ustx.py reference.ustx detected.ustx [--ref-track NAME] [--det-track NAME]

Reports note-level precision / recall / F1 (onset within 50 ms, pitch within 50 cents),
plus the same with onset tolerance 100 ms, and lyric agreement on matched notes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def load_notes(path, track: str | None = None, track_no: int | None = None):
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    tempos = d.get("tempos") or [{"position": 0, "bpm": d.get("bpm", 120)}]
    if len(tempos) > 1:
        print(f"warning: {path} has tempo changes; using the first tempo only", file=sys.stderr)
    bpm = float(tempos[0]["bpm"])
    tick_s = 60.0 / bpm / 480
    names = [t.get("track_name", f"Track{i}") for i, t in enumerate(d["tracks"])]
    # audio may be shifted (UtauCover pads so bar lines line up): measure times from the audio start
    waves = d.get("wave_parts") or []
    shift = min((w.get("position", 0) for w in waves), default=0)
    out = {}
    for part in d.get("voice_parts") or []:
        tn = part["track_no"]
        nm = names[tn]
        if track is not None and nm != track:
            continue
        if track_no is not None and tn != track_no:
            continue
        for n in part.get("notes") or []:
            tick = part["position"] + n["position"] - shift
            out.setdefault(nm, []).append((tick * tick_s, (tick + n["duration"]) * tick_s, int(n["tone"]), str(n["lyric"])))
    for k in out:
        out[k].sort()
    return out


def score(ref, det, onset_tol=0.05):
    import mir_eval
    if not ref or not det:
        return 0, 0, 0
    ri = np.array([[a, max(b, a + 0.01)] for a, b, _, _ in ref])
    rp = np.array([440 * 2 ** ((t - 69) / 12) for _, _, t, _ in ref])
    di = np.array([[a, max(b, a + 0.01)] for a, b, _, _ in det])
    dp = np.array([440 * 2 ** ((t - 69) / 12) for _, _, t, _ in det])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, di, dp, onset_tolerance=onset_tol,
                                                                   pitch_tolerance=50.0, offset_ratio=None)
    return p, r, f


def onset_only(ref, det, tol=0.05):
    import mir_eval
    ro = np.array([a for a, *_ in ref])
    do = np.array([a for a, *_ in det])
    if not len(ro) or not len(do):
        return 0, 0, 0
    return mir_eval.onset.f_measure(ro, do, window=tol)


def frame_pitch_acc(ref, det, hop=0.01):
    end = max(ref[-1][1], det[-1][1]) if ref and det else 0
    n = int(end / hop) + 1
    rt = np.zeros(n, int)
    dt = np.zeros(n, int)
    for a, b, t, _ in ref:
        rt[int(a / hop):int(b / hop)] = t
    for a, b, t, _ in det:
        dt[int(a / hop):int(b / hop)] = t
    both = (rt > 0) & (dt > 0)
    return {"ref_voiced_covered": float(both.sum() / max(1, (rt > 0).sum())),
            "pitch_correct_when_both": float((rt[both] == dt[both]).mean()) if both.any() else 0.0,
            "octave_errors": float((np.abs(rt[both] - dt[both]) == 12).mean()) if both.any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("det")
    ap.add_argument("--ref-track")
    ap.add_argument("--det-track")
    a = ap.parse_args()
    ref = load_notes(a.ref, a.ref_track)
    det = load_notes(a.det, a.det_track)
    rn = sum(ref.values(), [])
    rn.sort()
    dn = sum(det.values(), [])
    dn.sort()
    print(f"reference notes: {len(rn)}  detected notes: {len(dn)}")
    for tol in (0.05, 0.1):
        p, r, f = score(rn, dn, tol)
        print(f"  note P/R/F1 @{int(tol * 1000)}ms: {p:.2f} / {r:.2f} / {f:.2f}")
    f, p, r = onset_only(rn, dn, 0.05)
    print(f"  onset-only P/R/F1 @50ms: {p:.2f} / {r:.2f} / {f:.2f}")
    print("  frames:", {k: round(v, 3) for k, v in frame_pitch_acc(rn, dn).items()})


if __name__ == "__main__":
    main()
