"""`utaucover selftest`: check that this installation can do everything a build needs.

Used by the packaging pipeline on every OS (a bundle that passes has its Python, libraries, native code,
ffmpeg and models' runtimes all working), and useful for "it doesn't work on my PC" reports.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


class Report:
    def __init__(self):
        self.failed = 0

    def check(self, name: str, fn):
        t = time.time()
        try:
            detail = fn()
            print(f"  ok    {name}: {detail if detail is not None else ''}  ({time.time() - t:.1f}s)", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
            tb = traceback.format_exc().strip().splitlines()
            print("        " + "\n        ".join(tb[-6:]), flush=True)
            return False

    def info(self, name: str, fn):
        try:
            print(f"  info  {name}: {fn()}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  info  {name}: ({type(e).__name__}: {e})", flush=True)


def _tone(seconds=2.0, hz=220.0, sr=44100):
    import numpy as np
    t = np.arange(int(seconds * sr)) / sr
    # a sung-ish vowel: fundamental + harmonics, slight vibrato
    f = hz * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    ph = 2 * np.pi * np.cumsum(f) / sr
    y = sum(0.25 / k * np.sin(k * ph) for k in range(1, 8))
    return y.astype("float32"), sr


def run(gui: bool = False) -> int:
    from . import __version__
    from .paths import APP_DIR, APP_SUPPORT, MODEL_DIR, PORTABLE, whisper_dir
    r = Report()
    print(f"UtauCover {__version__} selftest  ({sys.platform}, Python {sys.version.split()[0]})")
    r.info("python", lambda: sys.executable)
    r.info("program folder", lambda: f"{APP_DIR} (portable)" if PORTABLE else str(APP_DIR or "(not packaged)"))
    r.info("data folder", lambda: APP_SUPPORT)

    def writable():
        APP_SUPPORT.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=APP_SUPPORT):
            pass
        return f"models in {MODEL_DIR}, Whisper in {whisper_dir()}"
    r.check("data folder writable", writable)

    def ffmpeg():
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg isn't on PATH")
        out = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True, timeout=30)
        return f"{exe}: {out.stdout.splitlines()[0][:60]}"
    r.check("ffmpeg", ffmpeg)

    for mod in ("numpy", "scipy", "librosa", "soundfile", "yaml", "mido", "pykakasi", "pyphen", "pronouncing",
                "pypinyin", "mcp", "pythonnet", "clr_loader", "onnxruntime", "torch", "torchaudio", "torchcrepe",
                "torchfcpe", "whisper", "stable_whisper", "audio_separator.separator", "basic_pitch", "webview"):
        r.check(f"import {mod}", lambda m=mod: getattr(importlib.import_module(m), "__version__", "") or "")

    def soundfile_mp3():
        import soundfile as sf
        y, sr = _tone(1.0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mp3"
            subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(sr),
                            "-ac", "1", "-i", "-", str(p)], input=y.tobytes(), check=True, timeout=60)
            z, sr2 = sf.read(str(p))
        return f"libsndfile {sf.__libsndfile_version__}: read {len(z) / sr2:.2f}s of mp3"
    r.check("mp3 decode (ffmpeg -> libsndfile)", soundfile_mp3)

    def torch_info():
        import torch
        from .pitch import _device
        return f"torch {torch.__version__}, device {_device()}"
    r.check("torch device", torch_info)

    def ort():
        import onnxruntime as ort
        return ", ".join(ort.get_available_providers())
    r.check("onnxruntime providers", ort)

    def fcpe():
        import numpy as np
        import librosa
        from .pitch import SR, _fcpe_track
        y, sr = _tone(2.0, 220.0)
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        hz, conf = _fcpe_track(y, int(len(y) / SR / 0.01))
        med = float(np.median(hz[(hz > 0) & (conf > 0.5)])) if np.any(hz > 0) else 0.0
        if not 200 < med < 240:
            raise RuntimeError(f"expected ~220 Hz, got {med:.1f}")
        return f"{med:.1f} Hz on a 220 Hz tone"
    r.check("pitch tracking (FCPE)", fcpe)

    def basic_pitch():
        import soundfile as sf
        from basic_pitch.inference import Model, predict
        from .harmony import basic_pitch_model
        ICASSP_2022_MODEL_PATH = basic_pitch_model()
        y, sr = _tone(2.0, 330.0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.wav"
            sf.write(str(p), y, sr)
            model = Model(ICASSP_2022_MODEL_PATH)
            _, _, notes = predict(str(p), model)
        return f"{Path(str(ICASSP_2022_MODEL_PATH)).name}: {len(notes)} note(s)"
    r.check("polyphonic notes (basic-pitch)", basic_pitch)

    def whisper_ok():
        import whisper
        from whisper.tokenizer import get_tokenizer
        tok = get_tokenizer(multilingual=True)
        assert tok.encode(" hello")
        return f"tokenizer ok, {len(whisper.available_models())} model sizes"
    r.check("Whisper", whisper_ok)

    def separator():
        from audio_separator.separator import Separator
        with tempfile.TemporaryDirectory() as d:
            Separator(output_dir=d, model_file_dir=d, log_level=40)
        return "ffmpeg + onnxruntime found"
    r.check("audio-separator", separator)

    def webview_backend():
        import webview  # noqa: F401
        name = {"darwin": "webview.platforms.cocoa", "win32": "webview.platforms.winforms"}.get(
            sys.platform, "webview.platforms.qt")
        importlib.import_module(name)
        return name
    r.check("window backend", webview_backend)

    def apps():
        from .apps import describe
        return describe().replace("\n", " | ")
    r.info("UtauV / OpenUtau", apps)

    def game():
        from .game import find_game, game_version
        g = find_game()
        return f"GAME {game_version(g[1])} at {g[1]}" if g else "not installed (builds use the built-in engine)"
    r.info("GAME", game)

    if gui:
        def window():
            env = dict(os.environ, UTAUCOVER_GUI_SELFTEST="1")
            p = subprocess.run([sys.executable, "-m", "utaucover.gui.app"], capture_output=True, text=True,
                               timeout=180, env=env)
            line = next((l for l in p.stderr.splitlines() if l.startswith("SELFTEST")), None)
            if not line:
                raise RuntimeError((p.stderr or p.stdout)[-800:])
            return line[9:]
        r.check("window", window)

    print("selftest passed" if not r.failed else f"selftest FAILED ({r.failed} problem(s))", flush=True)
    return 1 if r.failed else 0


def mcp_config() -> dict:
    """The MCP server entry for Claude Desktop / Claude Code, pointing at this installation."""
    from .paths import APP_DIR
    if APP_DIR is not None:
        if sys.platform == "darwin":
            cli = APP_DIR.parent / "MacOS" / "utaucover-cli"   # APP_DIR is UtauCover.app/Contents/Resources
        elif sys.platform == "win32":
            cli = APP_DIR / "utaucover-cli.exe"
        else:
            cli = APP_DIR / "utaucover-cli"
        cmd, args = str(cli), ["mcp"]
    else:
        cmd, args = sys.executable, ["-m", "utaucover", "mcp"]
    return {"mcpServers": {"utaucover": {"command": cmd, "args": args}}}
