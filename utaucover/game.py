"""Note transcription with GAME (Generative Adaptive MIDI Extractor, openvpi), the model behind UtauV's
"transcribe audio to notes" tool.

UtauCover doesn't ship GAME: it runs the copy installed in UtauV / OpenUtau (<data folder>/Dependencies/game)
through the app's own code (OpenUtau.Core.Analysis.Game), in a helper process that hosts the app's bundled
.NET runtime. So the notes are the ones UtauV's tool would give. When GAME isn't installed, the built-in
pitch-tracking + segmentation engine is used instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .apps import data_dirs, programs
from .notes import Note
GAME_DOWNLOAD_URL = "https://github.com/openvpi/GAME/releases/tag/oudep"  # what UtauV itself points to
INSTALL_HINT = ("Install GAME in UtauV: download the .oudep package from " + GAME_DOWNLOAD_URL +
                " and drag it onto UtauV's window (or use UtauV's 'transcribe audio to notes' tool, which offers "
                "to download it). Then build again.")
# GAME was trained with a language embedding for these; anything else runs language-neutral.
GAME_LANGS = {"en", "ja", "zh", "yue"}


def find_game() -> tuple[Path, Path] | None:
    """(app code dir, GAME model dir) for an installed app that has GAME, or None.
    UTAUCOVER_NO_GAME=1 pretends it's missing (to try the fallback)."""
    if os.environ.get("UTAUCOVER_NO_GAME"):
        return None
    models = [d / "Dependencies" / "game" for d in data_dirs()]
    models = [m for m in models if (m / "encoder.onnx").exists() and (m / "config.json").exists()]
    if not models:
        return None
    for app in programs():
        core = app.code_dir / "OpenUtau.Core.dll"
        try:
            has_game = b"GameOnnxBackend" in core.read_bytes()
        except OSError:
            continue
        if has_game:
            mine = [m for m in models if m.parent.parent == app.data_dir]  # this app's own models first
            return app.code_dir, (mine or models)[0]
    return None


def game_version(model_dir: Path) -> str:
    try:
        for line in (model_dir / "oudep.yaml").read_text().splitlines():
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return "?"


_HELPER = r'''
import sys, os, json, glob
import numpy as np, soundfile as sf
from clr_loader import get_coreclr
from pythonnet import set_runtime
app_dir, model_dir, wav, lang, chunks_json, steps, bthr, sthr, passes = sys.argv[1:10]
entry = [f for f in glob.glob(os.path.join(app_dir, "*.dll")) if os.path.exists(f[:-4] + ".runtimeconfig.json")]
set_runtime(get_coreclr(entry_dll=entry[0], dotnet_root=app_dir))
import clr, System
sys.path.append(app_dir)
clr.AddReference("OpenUtau.Core")
def _sandbox_app_paths():
    """UtauV saves its settings next to the program running its code (here: UtauCover's Python, inside the
    app). Point its folders at this helper's throwaway folder before anything reads its settings."""
    box = os.environ.get("UTAUCOVER_SANDBOX")
    if not box:
        return
    try:
        from OpenUtau.Core import PathManager
        pm = PathManager.Inst
        for name, value in (("RootPath", box), ("DataPath", os.path.join(box, "data")),
                            ("CachePath", os.path.join(box, "cache"))):
            try:
                setattr(pm, name, value)
            except Exception:
                from System.Reflection import BindingFlags
                f = pm.GetType().GetField(f"<{name}>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
                if f is not None:
                    f.SetValue(pm, value)
    except Exception:
        pass
_sandbox_app_paths()
from System.Runtime.InteropServices import Marshal
from OpenUtau.Core.Analysis import Game, GameOptions
y, sr = sf.read(wav, dtype="float32", always_2d=True)
y = y.mean(axis=1)
if sr != 44100:
    import librosa
    y = librosa.resample(y, orig_sr=sr, target_sr=44100)
    sr = 44100
g = Game(model_dir)
o = GameOptions()
if lang:
    o.LanguageCode = lang
o.SamplingSteps = int(steps); o.BoundaryThreshold = float(bthr); o.ScoreThreshold = float(sthr)
out = []
chunks = json.loads(chunks_json)
for k, (a, b, _ca, _cb) in enumerate(chunks):
    seg = np.ascontiguousarray(y[int(a * sr):int(b * sr)], dtype=np.float32)
    if len(seg) < sr // 4:
        out.append([[]])  # keep results aligned with the chunk list
        continue
    arr = System.Array.CreateInstance(System.Single, len(seg))
    Marshal.Copy(System.IntPtr(seg.ctypes.data), arr, 0, len(seg))
    runs = []
    for _ in range(int(passes)):
        pos, notes = a, []
        for n in g.TranscribeWaveform(arr, o):
            d = float(n.noteDuration)
            if n.noteVoiced:
                notes.append([pos, pos + d, float(n.noteScore)])
            pos += d
        runs.append(notes)
    out.append(runs)
    print(f"@@PROGRESS@@{k + 1}/{len(chunks)}", flush=True)
print("@@JSON@@" + json.dumps(out))
'''


def _sandbox_env(home: str) -> dict:
    """Environment for a helper that runs the app's .NET code: its settings go to a throwaway folder
    (macOS / Linux read them from HOME / XDG_*), and output is UTF-8."""
    env = dict(os.environ, HOME=home, XDG_DATA_HOME=os.path.join(home, "data"),
               XDG_CACHE_HOME=os.path.join(home, "cache"), XDG_CONFIG_HOME=os.path.join(home, "config"),
               UTAUCOVER_SANDBOX=home, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    for k in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(k, None)
    return env


def _no_window() -> dict:
    """On Windows, don't flash a console window for a helper process started from the GUI."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def chunk_spans(voiced: np.ndarray, hop: float, total_s: float, max_len: float = 25.0,
                min_gap: float = 0.3, context: float = 2.0) -> list[tuple[float, float, float, float]]:
    """Cut the song at silences into pieces GAME can take in one go (it works on phrases, not whole songs).

    Returns (start, end, core_start, core_end) per piece. GAME needs a couple of seconds of audio before a
    phrase to hear it properly (a phrase starting 0.3 s into a piece lost most of its notes), so each piece
    gets `context` seconds on both sides, and only the notes in its core (the half-way points to its
    neighbours) are kept."""
    from .pitch import _runs
    regions = [(a * hop, b * hop) for v, a, b in _runs(voiced) if v]
    merged: list[list[float]] = []
    for a, b in regions:
        if merged and a - merged[-1][1] < min_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    groups: list[list[float]] = []
    for a, b in merged:
        if groups and b - groups[-1][0] <= max_len:
            groups[-1][1] = b
        else:
            groups.append([a, b])
    content: list[tuple[float, float]] = []
    for a, b in groups:  # a single phrase longer than max_len is split evenly
        n = max(1, int(np.ceil((b - a) / max_len)))
        step = (b - a) / n
        content += [(a + k * step, a + (k + 1) * step) for k in range(n)]
    out = []
    for k, (a, b) in enumerate(content):
        core_a = 0.0 if k == 0 else 0.5 * (content[k - 1][1] + a)
        core_b = total_s if k == len(content) - 1 else 0.5 * (b + content[k + 1][0])
        out.append((max(0.0, a - context), min(total_s, b + context), core_a, core_b))
    return out


def consensus(runs: list[list[list[float]]]) -> list[list[float]]:
    """GAME samples its answer, so two runs differ (on ~13% of sung frames, mostly by a semitone). Keep the
    first run's note boundaries (stable run to run) and give each note the pitch most runs agree on."""
    if len(runs) == 1:
        return runs[0]
    out = []
    for a, b, p in runs[0]:
        votes = [p]
        for other in runs[1:]:
            best, cover = None, 0.0
            for c, d, q in other:
                ov = min(b, d) - max(a, c)
                if ov > cover:
                    best, cover = q, ov
            if best is not None and cover >= 0.5 * (b - a):
                votes.append(best)
        tones = [round(v) for v in votes]
        top = max(set(tones), key=lambda t: (tones.count(t), -abs(t - p)))
        out.append([a, b, float(np.median([v for v in votes if round(v) == top]))])
    return out


def transcribe(wav: Path, voiced: np.ndarray, hop: float, language: str | None = None, steps: int = 8,
               boundary_threshold: float = 0.2, score_threshold: float = 0.2, passes: int = 3,
               progress=None) -> list[Note]:
    """Notes (start, end, fractional MIDI pitch) for a monophonic vocal, from UtauV's GAME (the consensus
    of `passes` runs)."""
    found = find_game()
    if found is None:
        raise RuntimeError("GAME isn't installed. " + INSTALL_HINT)
    code, model = found
    import soundfile as sf
    total = sf.info(str(wav)).duration
    chunks = chunk_spans(voiced, hop, total)
    lang = language if language in GAME_LANGS else ""
    if progress:
        progress(f"[notes] transcribing with GAME {game_version(model)} ({len(chunks)} phrases, "
                 f"{passes} passes{', language ' + lang if lang else ', language-neutral'})")
    with tempfile.TemporaryDirectory(prefix="utaucover_game_") as home:
        script = Path(home) / "game_helper.py"
        script.write_text(_HELPER, encoding="utf-8")
        env = _sandbox_env(home)
        p = subprocess.Popen([sys.executable, "-I", str(script), str(code), str(model), str(wav), lang,
                              json.dumps(chunks), str(steps), str(boundary_threshold), str(score_threshold),
                              str(max(1, passes))],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                             errors="replace", env=env, **_no_window())
        result = None
        for line in p.stdout:
            if line.startswith("@@JSON@@"):
                result = json.loads(line[8:])
        err = p.stderr.read()
        p.wait()
    if result is None:
        raise RuntimeError(f"GAME failed: {err.strip()[-400:]}")
    notes = [Note(a, b, s) for (_, _, ca, cb), chunk_runs in zip(chunks, result)
             for a, b, s in consensus(chunk_runs) if b - a >= 0.04 and ca <= 0.5 * (a + b) < cb]
    notes.sort(key=lambda n: n.start)
    # chunks overlap by their padding: drop duplicates where two chunks transcribed the same moment
    out: list[Note] = []
    for n in notes:
        if out and n.start < out[-1].end - 0.02:
            if n.end <= out[-1].end:
                continue
            n.start = out[-1].end
        out.append(n)
    return out


def split_at_words(notes: list[Note], times: list[float], rms_db: np.ndarray | None = None, hop: float = 0.01,
                   min_piece: float = 0.08, search: float = 0.04) -> list[Note]:
    """Cut notes where a confident word starts inside them (repeated pitches on different words), at the
    loudness dip nearest that time."""
    times = sorted(times)
    out: list[Note] = []
    for n in notes:
        cuts = []
        for t in times:
            if n.start + min_piece <= t <= n.end - min_piece:
                if rms_db is not None:
                    a, b = int((t - search) / hop), int((t + search) / hop) + 1
                    seg = rms_db[max(0, a):max(0, b)]
                    if len(seg):
                        t = (max(0, a) + int(np.argmin(seg))) * hop
                if not cuts or t - cuts[-1] >= min_piece:
                    if n.start + min_piece <= t <= n.end - min_piece:
                        cuts.append(t)
        edges = [n.start] + cuts + [n.end]
        for k in range(len(edges) - 1):
            out.append(Note(edges[k], edges[k + 1], n.pitch, forced=k > 0))
    return out
