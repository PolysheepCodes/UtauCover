"""Benchmark UtauCover against hand-charted projects (lives in the repo so a reboot can't wipe it).

    python tools/bench.py build  [flowerman die rubio cats]   # build into ~/Library/Application Support/UtauCover/bench
    python tools/bench.py score                               # score whatever has been built
    python tools/bench.py validate FILE.ustx ...              # load projects with UtauV's own loader

flowerman / die-in-a-fire are the user's own renders + charts (ground truth for notes; flowerman also for
lyrics and harmonies). Rubio's Lament and Here Come the Cats are real songs without a chart: they're built
so they can be listened to and spot-checked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from eval_against_ustx import frame_pitch_acc, load_notes, score  # noqa: E402

BENCH = Path.home() / "Library/Application Support/UtauCover/bench"
COVERS = Path.home() / "Documents/OpenUTAU/Covers"
DL = Path.home() / "Downloads"
SONGS = {
    "flowerman": dict(song=COVERS / "flowerman-googoloid.wav", chart=COVERS / "flowerman-googoloid.ustx",
                      track="main", sheet_from_chart=True,
                      harmony=["higher harmony", "lower harmony", "higherer", "lowerer"]),
    "die": dict(song=COVERS / "die-in-a-fire-googoloid.wav", chart=COVERS / "die-in-a-fire-googoloid.ustx",
                track="Main Melody"),
    "rubio": dict(song=DL / "Rubio’s Lament - Waitress (One Take Lyric Video).mp3",
                  sheet=DL / "Rubio_s Lament lyrics (official).txt"),
    "cats": dict(song=DL / "Here Come The Cats.mp3", sheet=DL / "Here Come the Cats v1/lyrics_detected.txt"),
}


def sheet_from_chart(chart: Path, track: str, out: Path, rest_s: float = 0.9) -> Path:
    """A lyric sheet rebuilt from a chart's word notes (lines break at rests)."""
    lines, cur, last = [], [], None
    for a, b, t, lyric in load_notes(chart)[track]:
        if lyric.startswith("+") or not lyric.strip():
            last = b
            continue
        if last is not None and a - last > rest_s and cur:
            lines.append(" ".join(cur))
            cur = []
        cur.append(lyric.split(" [")[0])
        last = b
    if cur:
        lines.append(" ".join(cur))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build(names: list[str], extra: list[str]):
    BENCH.mkdir(parents=True, exist_ok=True)
    for name in names:
        cfg = SONGS[name]
        cmd = [sys.executable, "-m", "utaucover", "build", str(cfg["song"]), "--out", str(BENCH / name),
               "--singer", "Googoloid English", "--name", name, *extra]
        if cfg.get("sheet_from_chart"):
            cmd += ["--lyrics", str(sheet_from_chart(cfg["chart"], cfg["track"], BENCH / f"{name}_sheet.txt"))]
        elif cfg.get("sheet"):
            cmd += ["--lyrics", str(cfg["sheet"])]
        print(f"building {name} ...", flush=True)
        with open(BENCH / f"{name}.log", "w") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"  done: {(BENCH / name / f'{name}.ustx').exists()}", flush=True)


def _lyric_hits(ref, det):
    norm = lambda s: s.lower().strip(".,!?'\"-").split(" [")[0]  # noqa: E731
    r = [n for n in ref if not n[3].startswith("+")]
    d = [n for n in det if not n[3].startswith("+")]
    on = sum(1 for a, _, _, lyr in r if any(norm(x[3]) == norm(lyr) for x in d if abs(x[0] - a) < 0.15))
    near = sum(1 for a, _, _, lyr in r if any(norm(x[3]) == norm(lyr) for x in d if abs(x[0] - a) < 0.5))
    return on / len(r), near / len(r)


def score_all():
    for name, cfg in SONGS.items():
        f = BENCH / name / f"{name}.ustx"
        if not f.exists():
            continue
        det = load_notes(f)
        lead = det.get("Lead", [])
        res = json.loads((BENCH / name / "analysis/result.json").read_text())
        line = f"{name:10} {res.get('seconds', 0):5.0f}s  lead {len(lead)} notes"
        if cfg.get("chart"):
            ref = load_notes(cfg["chart"])
            fp = frame_pitch_acc(ref[cfg["track"]], lead)
            line += (f" | F1@50 {score(ref[cfg['track']], lead, .05)[2]:.2f} @100 {score(ref[cfg['track']], lead, .1)[2]:.2f}"
                     f" pitch {fp['pitch_correct_when_both']:.0%} oct {fp['octave_errors']:.1%}")
            if cfg.get("sheet_from_chart"):
                on, near = _lyric_hits(ref[cfg["track"]], lead)
                line += f" | words on note {on:.0%}, within .5s {near:.0%}"
            if cfg.get("harmony"):
                rh = sorted(sum([ref[k] for k in cfg["harmony"]], []))
                dh = sorted(sum([v for k, v in det.items() if k != "Lead"], []))
                line += f" | harmony F1@100 {score(rh, dh, .1)[2]:.2f} ({len(dh)} notes)"
        else:
            line += " | " + ", ".join(f"{k}: {len(v)}" for k, v in det.items() if k != "Lead")
        print(line)


_VALIDATE = r'''
import sys, os, glob
from clr_loader import get_coreclr
from pythonnet import set_runtime
app = "/Applications/UtauV.app/Contents/MacOS"
set_runtime(get_coreclr(entry_dll=os.path.join(app, "UtauV.dll"), dotnet_root=app))
import clr
sys.path.append(app); clr.AddReference("OpenUtau.Core")
from OpenUtau.Core.Format import Ustx
for p in sys.argv[1:]:
    try:
        proj = Ustx.Load(p)
        notes = sum(part.notes.Count for part in proj.parts if hasattr(part, "notes"))
        print(f"LOADED OK  {os.path.basename(p)}: {proj.tracks.Count} tracks, {notes} notes")
    except Exception as e:
        print(f"FAILED     {os.path.basename(p)}: {e}")
'''


def validate(paths: list[str]):
    with tempfile.TemporaryDirectory() as home:
        script = Path(home) / "validate.py"
        script.write_text(_VALIDATE)
        subprocess.run([sys.executable, str(script), *paths], env=dict(os.environ, HOME=home))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "build":
        names = [a for a in sys.argv[2:] if not a.startswith("-")] or list(SONGS)
        build(names, [a for a in sys.argv[2:] if a.startswith("-")])
        score_all()
    elif cmd == "validate":
        validate(sys.argv[2:])
    else:
        score_all()
