"""Where the OpenUTAU-family apps (UtauV, OpenUtau) live on this computer: macOS, Windows and Linux.

An installation has
- a program folder (`code_dir`) with the .NET assemblies (OpenUtau.Core.dll ...): UtauCover reads the
  phonemizer list from them and runs GAME through them;
- a data folder (`data_dir`) with Singers, Plugins and Dependencies (GAME's model is Dependencies/game);
- the program that opens a project (`exe`; on macOS the .app bundle).

On Windows and Linux OpenUtau is often an unzipped folder that can be anywhere, so the usual places are
searched (and, on Windows, the uninstall list and the .ustx file association). A folder can also be given
with the UTAUCOVER_UTAU_DIRS environment variable (separated by os.pathsep) or the window's
"OpenUtau folder" setting (settings.json -> "utau_dirs").
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import settings

APP_NAMES = ("UtauV", "OpenUtau")  # UtauV first: when both are installed, projects are made for UtauV
CORE_DLL = "OpenUtau.Core.dll"


@dataclass(frozen=True)
class UtauApp:
    name: str
    code_dir: Path | None   # folder with OpenUtau.Core.dll (None: only its data folder was found)
    data_dir: Path
    exe: Path | None

    def to_dict(self):
        return {"name": self.name, "program": str(self.code_dir or ""), "data": str(self.data_dir),
                "exe": str(self.exe or "")}


_CACHE: list[UtauApp] | None = None


def refresh():
    """Forget what was found (after the user installs an app or changes the folder setting)."""
    global _CACHE
    _CACHE = None


def installed_apps() -> list[UtauApp]:
    global _CACHE
    if _CACHE is None:
        found = _discover()
        seen, out = set(), []
        for a in found:
            key = (str(a.code_dir.resolve()) if a.code_dir else None, str(a.data_dir))
            if key not in seen:
                seen.add(key)
                out.append(a)
        order = {n: i for i, n in enumerate(APP_NAMES)}
        _CACHE = sorted(out, key=lambda a: (a.code_dir is None, order.get(a.name, 9)))
    return list(_CACHE)


def data_dirs() -> list[Path]:
    """Existing data folders (Singers / Plugins / Dependencies) of every app found."""
    seen, out = set(), []
    for d in [a.data_dir for a in installed_apps()] + _standard_data_dirs():
        if d.is_dir() and str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def programs() -> list[UtauApp]:
    """Installations whose program folder was found (needed for phonemizers, GAME and opening projects)."""
    return [a for a in installed_apps() if a.code_dir is not None]


def target_app() -> UtauApp | None:
    """The app projects are made for: UtauV if it's installed, else OpenUtau."""
    apps = installed_apps()
    return apps[0] if apps else None


def target_data_dir() -> Path:
    a = target_app()
    if a:
        return a.data_dir
    std = _standard_data_dirs()
    return std[-1] if std else Path.home() / "OpenUtau"


def describe() -> str:
    apps = installed_apps()
    if not apps:
        return "No UtauV / OpenUtau installation found."
    return "\n".join(f"{a.name}: program {a.code_dir or '(not found)'}; data {a.data_dir}" for a in apps)


# ------------------------------------------------------------------------------------------ discovery
def _user_dirs() -> list[Path]:
    raw = [p for p in os.environ.get("UTAUCOVER_UTAU_DIRS", "").split(os.pathsep) if p.strip()]
    raw += [str(p) for p in settings().get("utau_dirs", []) or [] if p]
    return [Path(p).expanduser() for p in raw]


def _name_for(folder: Path) -> str:
    low = str(folder).lower()
    if "utauv" in low or (folder / "UtauV.dll").exists() or (folder / "UtauV.exe").exists():
        return "UtauV"
    return "OpenUtau"


def _code_dir(folder: Path) -> Path | None:
    """The program folder for something the user or a search pointed at: the folder itself, a macOS
    .app bundle, or a release folder that has the program one level down."""
    if folder.suffix.lower() == ".app":
        folder = folder / "Contents" / "MacOS"
    if (folder / CORE_DLL).exists():
        return folder
    try:
        for sub in folder.iterdir():
            if sub.is_dir() and (sub / CORE_DLL).exists():
                return sub
    except OSError:
        pass
    return None


def _exe(code: Path, name: str) -> Path | None:
    if sys.platform == "darwin":
        app = code.parent.parent
        return app if app.suffix == ".app" else None
    for n in (name, "UtauV", "OpenUtau"):
        f = code / (n + ".exe" if sys.platform == "win32" else n)
        if f.exists():
            return f
    return None


def _standard_data_dirs() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        return [home / "Library" / n for n in APP_NAMES] + \
               [home / "Library" / "Application Support" / n for n in APP_NAMES]
    if sys.platform == "win32":
        docs = _win_documents()
        return [docs / n for n in APP_NAMES] + \
               [Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / n for n in APP_NAMES]
    xdg = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return [xdg / n for n in APP_NAMES]


def _data_dir(code: Path | None, name: str) -> Path:
    """Data folder of an installation. macOS / Linux keep it in the user's Library / ~/.local/share; on
    Windows a portable copy keeps it in its own folder and an installed copy (installed.txt) in Documents."""
    home = Path.home()
    if sys.platform == "darwin":
        cands = [home / "Library" / name, home / "Library" / "Application Support" / name]
    elif sys.platform == "win32":
        cands = []
        if code is not None and not (code / "installed.txt").exists():
            cands.append(code)
        cands += [_win_documents() / name, Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / name]
        if code is not None:
            cands.append(code)
    else:
        xdg = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
        cands = [xdg / name] + ([code] if code is not None else [])
    for d in cands:
        if (d / "Singers").is_dir() or (d / "prefs.json").exists():
            return d
    return cands[0]


def _discover() -> list[UtauApp]:
    found: list[UtauApp] = []
    for folder in _user_dirs() + _search_folders():
        code = _code_dir(folder)
        if code is not None:
            name = _name_for(code if sys.platform != "darwin" else code.parent.parent)
            found.append(UtauApp(name, code, _data_dir(code, name), _exe(code, name)))
        elif (folder / "Singers").is_dir():  # the user pointed at a data folder
            found.append(UtauApp(_name_for(folder), None, folder, None))
    # data folders whose program wasn't found still have singers / plugins / GAME models
    for d in _standard_data_dirs():
        if d.is_dir():
            found.append(UtauApp(d.name, None, d, None))
    with_code = {str(a.data_dir) for a in found if a.code_dir is not None}
    return [a for a in found if a.code_dir is not None or str(a.data_dir) not in with_code]


def _search_folders() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        return [root / f"{n}.app" for root in (Path("/Applications"), home / "Applications")
                for n in ("UtauV", "OpenUtau", "OpenUTAU")]
    if sys.platform == "win32":
        env = os.environ
        local = Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local")
        roots = [local / "Programs", Path(env.get("ProgramFiles") or r"C:\Program Files"),
                 Path(env.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"), local, home,
                 home / "Desktop", home / "Downloads", _win_documents(), Path("C:/")]
        out = _glob_named(roots)
        out += _win_registry_folders()
        return out
    roots = [home, home / "Applications", home / ".local" / "opt", home / ".local" / "share", home / "Desktop",
             home / "Downloads", home / "Programs", home / "opt", Path("/opt"), Path("/usr/lib"),
             Path("/usr/share"), Path("/usr/local/share"), Path("/usr/local/lib")]
    out = _glob_named(roots)
    for exe in ("OpenUtau", "openutau", "UtauV", "utauv"):
        w = shutil.which(exe)
        if w:
            out.append(Path(w).resolve().parent)
    return out


def _glob_named(roots: list[Path]) -> list[Path]:
    """Folders named like the apps (OpenUtau, OpenUtau-win-x64, openutau ...) in the given places."""
    out = []
    for r in roots:
        try:
            for sub in r.iterdir():
                low = sub.name.lower()
                if any(low.startswith(n.lower()) for n in APP_NAMES) and sub.is_dir():
                    out.append(sub)
        except OSError:
            continue
    return out


def _win_documents() -> Path:
    """The Documents folder (it can be redirected, e.g. into OneDrive)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:  # CSIDL_PERSONAL
                return Path(buf.value)
        except Exception:
            pass
    return Path.home() / "Documents"


def _win_registry_folders() -> list[Path]:
    """Install folders from the uninstall list and the program registered to open .ustx files."""
    out: list[Path] = []
    try:
        import winreg
    except ImportError:
        return out

    def add(value):
        p = Path(str(value).split(",")[0].strip().strip('"'))
        if str(p) and p.exists():
            out.append(p if p.is_dir() else p.parent)

    uninstall = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    for hive, path in ((winreg.HKEY_CURRENT_USER, uninstall), (winreg.HKEY_LOCAL_MACHINE, uninstall),
                       (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall")):
        try:
            key = winreg.OpenKey(hive, path)
        except OSError:
            continue
        with key:
            for i in range(winreg.QueryInfoKey(key)[0]):
                try:
                    with winreg.OpenKey(key, winreg.EnumKey(key, i)) as sub:
                        name = str(winreg.QueryValueEx(sub, "DisplayName")[0])
                        if not any(n.lower() in name.lower() for n in APP_NAMES):
                            continue
                        for v in ("InstallLocation", "DisplayIcon"):
                            try:
                                add(winreg.QueryValueEx(sub, v)[0])
                            except OSError:
                                pass
                except OSError:
                    continue
    try:
        prog = winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, ".ustx")
        if prog:
            cmd = winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, prog + r"\shell\open\command")
            if cmd:
                add(cmd.split('"')[1] if cmd.startswith('"') else cmd.split(" ")[0])
    except OSError:
        pass
    return out
