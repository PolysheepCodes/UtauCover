"""Environment for the packaged app (its launcher sets UTAUCOVER_APP_DIR). Imported first by utaucover."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _unblock_dotnet():
    """Windows marks every file unzipped from a downloaded .zip as "from the internet", and .NET Framework
    then refuses to load the window's .NET libraries (pythonnet, pywebview's WebView2 wrappers): the window
    fails with "Failed to resolve Python.Runtime.Loader.Initialize". Remove that mark from them. (Files put
    in place by the installer aren't marked; a folder UtauCover can't write to is left alone.)"""
    import sysconfig
    site = Path(sysconfig.get_paths()["purelib"])
    for pkg in ("pythonnet", "clr_loader", "webview"):
        root = site / pkg
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            if f.suffix.lower() in (".dll", ".exe"):
                try:
                    os.remove(f"{f}:Zone.Identifier")
                except OSError:  # not marked, or not allowed to change it
                    pass


def _setup():
    # The program folder can be read-only (Program Files, a signed .app): never write into it.
    sys.dont_write_bytecode = True
    if sys.platform == "win32":
        _unblock_dotnet()
    # ffmpeg ships next to the bundled Python (audio-separator and Whisper call it by name)
    exe_dir = str(Path(sys.executable).resolve().parent)
    if exe_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")
    from .paths import APP_SUPPORT
    cache = APP_SUPPORT / "cache"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))  # librosa's compiled functions
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    try:  # HTTPS certificates for downloads, whatever the system provides
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


_setup()
