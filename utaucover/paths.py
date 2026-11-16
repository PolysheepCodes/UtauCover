"""Shared locations and small helpers."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Set by the packaged app's launcher: the folder UtauCover was installed / unzipped to.
APP_DIR = Path(os.environ["UTAUCOVER_APP_DIR"]) if os.environ.get("UTAUCOVER_APP_DIR") else None
# A portable copy (Windows .zip) keeps its data next to the program, so the whole thing lives in one
# folder that can be moved or deleted. It's marked by a portable.txt file in the program folder.
PORTABLE = bool(APP_DIR and (APP_DIR / "portable.txt").exists())


def _default_home() -> Path:
    if PORTABLE:
        return APP_DIR / "data"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "UtauCover"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "UtauCover"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "UtauCover"


APP_SUPPORT = Path(os.environ.get("UTAUCOVER_HOME") or _default_home())
MODEL_DIR = Path(os.environ.get("UTAUCOVER_MODELS") or APP_SUPPORT / "models")
SETTINGS_FILE = APP_SUPPORT / "settings.json"


def whisper_dir() -> Path:
    """Where Whisper models are kept. Whisper's own default (~/.cache/whisper) is shared with other
    tools that use Whisper, so it's used unless this is a portable copy (then: inside its data folder)."""
    if os.environ.get("UTAUCOVER_WHISPER"):
        return Path(os.environ["UTAUCOVER_WHISPER"])
    if PORTABLE:
        return MODEL_DIR / "whisper"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "whisper"


def settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(**changes) -> dict:
    s = settings()
    s.update(changes)
    APP_SUPPORT.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    return s


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".aiff", ".aif", ".wma", ".mp4", ".webm", ".mkv", ".mov"}


def file_fingerprint(path: str | Path, chunk: int = 1 << 20) -> str:
    """Cheap content hash: size + first/last MiB. Good enough to detect a changed song."""
    p = Path(path)
    h = hashlib.sha1()
    size = p.stat().st_size
    h.update(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def safe_name(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in " -_.()[]") else "_" for c in name)
    return keep.strip().strip(".") or "song"
