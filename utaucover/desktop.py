"""Hand files to the desktop: open with the default app, show in Finder / Explorer / the file manager."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

FILE_MANAGER = {"darwin": "Finder", "win32": "Explorer"}.get(sys.platform, "file manager")


def _spawn(cmd, **kw):
    """Start a program without tying it to this process (it keeps running if UtauCover quits)."""
    if sys.platform == "win32":
        kw.setdefault("creationflags", subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kw.setdefault("start_new_session", True)
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, close_fds=True, **kw)


def open_path(path: str | os.PathLike):
    """Open a file or folder with its default app."""
    p = str(path)
    if sys.platform == "darwin":
        subprocess.run(["open", p], check=False)
    elif sys.platform == "win32":
        os.startfile(p)  # noqa: S606
    else:
        _spawn(["xdg-open", p])


def reveal(path: str | os.PathLike):
    """Show a file selected in Finder / Explorer / the file manager."""
    p = Path(path)
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(p)], check=False)
    elif sys.platform == "win32":
        # explorer parses its own command line: the path must be quoted after the comma
        _spawn(f'explorer /select,"{p}"')
    else:
        # most Linux file managers implement the freedesktop "ShowItems" call; else open the folder
        uri = p.resolve().as_uri()
        try:
            r = subprocess.run(["dbus-send", "--session", "--print-reply", "--dest=org.freedesktop.FileManager1",
                                "--type=method_call", "/org/freedesktop/FileManager1",
                                "org.freedesktop.FileManager1.ShowItems", f"array:string:{uri}", "string:"],
                               capture_output=True, timeout=5)
            if r.returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        open_path(p.parent if p.is_file() else p)


def launch(exe: str | os.PathLike, *args: str, cwd: str | os.PathLike | None = None):
    _spawn([str(exe), *map(str, args)], cwd=str(cwd) if cwd else None)
