#!/usr/bin/env python3
"""Build UtauCover's downloadable packages.

    python packaging/build.py                      # the packages for this computer's OS
    python packaging/build.py --target windows-x64 # another OS (see "Cross builds")
    python packaging/build.py lock                 # re-resolve packaging/locks/*.txt after changing dependencies

What comes out (in dist/):
    macOS    UtauCover-<v>-macos-arm64.dmg            open it, drag UtauCover to Applications
    Windows  UtauCover-<v>-windows-x64-portable.zip   unzip anywhere, run UtauCover.exe (no install, no admin)
             UtauCover-<v>-windows-x64-setup.exe      installer: per-user by default (no admin), or for all
                                                      users in Program Files (asks for admin). Needs Inno Setup.
    Linux    UtauCover-<v>-linux-x64.tar.gz           extract, run ./UtauCover, or ./install.sh for a menu entry

Each package is self-contained: a relocatable Python (python-build-standalone), every library, ffmpeg and a
native launcher. AI models download on first use into the user's data folder, with progress in the window.

Needs uv (https://docs.astral.sh/uv/). The launcher is compiled with clang on macOS (Xcode command line tools)
and with zig for Windows (fetched automatically through uv, so a Windows exe can be built on any OS).

Cross builds: Windows and Linux packages can be built on any OS (every package they use ships prebuilt wheels;
a check stops the build if any native file isn't built for the target). The macOS app needs a Mac. Release
builds run on each OS in the GitHub Actions pipeline (.github/workflows/build.yml), which also tests them.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "packaging"


def _work_dir() -> Path:
    """Scratch space outside the source folder: a synced folder (iCloud Drive, OneDrive, Dropbox) would
    upload ~2 GB per build and tag files with attributes that break macOS code signing."""
    if os.environ.get("UTAUCOVER_BUILD_DIR"):
        return Path(os.environ["UTAUCOVER_BUILD_DIR"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "UtauCover-build"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "UtauCover-build"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "utaucover-build"


BUILD = _work_dir()
DIST = BUILD / "dist"   # --out changes it
CACHE = BUILD / "cache"

PY_VERSION, PBS_RELEASE = "3.11.18", "20261110"
PBS_BASE = f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_RELEASE}"
MACOS_MIN = "14.0"  # torch / onnxruntime / scipy wheels need macOS 14 (Sonoma)
IMAGEIO_FFMPEG = "0.6.0"

TARGETS = {
    "macos-arm64": dict(os="macos", triple="aarch64-apple-darwin", platform="aarch64-apple-darwin", torch=None),
    "windows-x64": dict(os="windows", triple="x86_64-pc-windows-msvc", platform="x86_64-pc-windows-msvc", torch="cpu"),
    "linux-x64": dict(os="linux", triple="x86_64-unknown-linux-gnu", platform="x86_64-manylinux_2_28", torch="cpu"),
}


def host_target() -> str:
    m = platform.machine().lower()
    if sys.platform == "darwin":
        return "macos-arm64"
    if sys.platform == "win32":
        return "windows-x64"
    return "linux-x64" if m in ("x86_64", "amd64") else f"linux-{m}"


def version() -> str:
    return re.search(r'__version__ = "([^"]+)"', (ROOT / "utaucover" / "__init__.py").read_text()).group(1)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def uv() -> str:
    exe = shutil.which("uv")
    if not exe:
        sys.exit("uv is needed: https://docs.astral.sh/uv/getting-started/installation/")
    return exe


# ------------------------------------------------------------------------------------------- locks
def lock(targets: list[str]):
    """Resolve the exact package versions for each OS (commit the results: builds are reproducible)."""
    for name in targets:
        t = TARGETS[name]
        out = f"packaging/locks/{name}.txt"  # relative: the command is recorded in the file's header
        (ROOT / out).unlink(missing_ok=True)  # uv would otherwise prefer the old pins
        cmd = ["uv", "pip", "compile", "pyproject.toml", "packaging/bundle-requirements.in", "--python-version",
               "3.11", "--python-platform", t["platform"], "--overrides", "packaging/overrides.txt", "-o", out]
        if t["torch"]:
            cmd += ["--torch-backend", t["torch"]]
        run([uv(), *cmd[1:]], cwd=ROOT, env=dict(os.environ, MACOSX_DEPLOYMENT_TARGET=MACOS_MIN))


# ----------------------------------------------------------------------------------------- runtime
def download(url: str, dest: Path, sha256: str | None = None) -> Path:
    if dest.exists() and (not sha256 or _sha(dest) == sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "utaucover-build"})) as r, \
            open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if sha256 and _sha(tmp) != sha256:
        tmp.unlink()
        sys.exit(f"checksum mismatch for {url}")
    tmp.replace(dest)
    return dest


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def python_runtime(name: str, dest: Path):
    """python-build-standalone, unpacked to dest (= the bundle's runtime/ folder)."""
    t = TARGETS[name]
    fname = f"cpython-{PY_VERSION}+{PBS_RELEASE}-{t['triple']}-install_only_stripped.tar.gz"
    sums = download(f"{PBS_BASE}/SHA256SUMS", CACHE / f"pbs-{PBS_RELEASE}-SHA256SUMS")
    sha = next((l.split()[0] for l in sums.read_text().splitlines() if l.strip().endswith(fname)), None)
    if not sha:
        sys.exit(f"{fname} isn't in python-build-standalone {PBS_RELEASE}")
    tar = download(f"{PBS_BASE}/{fname.replace('+', '%2B')}", CACHE / fname, sha)
    log(f"unpacking Python {PY_VERSION} for {name}")
    tmp = dest.parent / "_python_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(tar) as tf:
        # share/ (terminfo, man pages) isn't used, and its case-colliding names break on macOS / Windows disks
        members = [m for m in tf.getmembers() if not m.name.startswith("python/share/")]
        tf.extractall(tmp, members=members, filter="tar" if hasattr(tarfile, "data_filter") else None)
    shutil.rmtree(dest, ignore_errors=True)
    (tmp / "python").rename(dest)
    tmp.rmdir()


def site_packages(name: str, runtime: Path) -> Path:
    if TARGETS[name]["os"] == "windows":
        return runtime / "Lib" / "site-packages"
    return runtime / "lib" / "python3.11" / "site-packages"


def host_python() -> str:
    """A Python 3.11 on this computer, for bytecode compiling and building the wheel."""
    return subprocess.run([uv(), "python", "find", "3.11"], capture_output=True, text=True, check=True).stdout.strip()


def install_packages(name: str, runtime: Path):
    t = TARGETS[name]
    site = site_packages(name, runtime)
    wheel_dir = BUILD / "wheel"
    shutil.rmtree(wheel_dir, ignore_errors=True)
    run([uv(), "build", "--wheel", "--out-dir", wheel_dir, ROOT])
    wheel = next(wheel_dir.glob("utaucover-*.whl"))
    common = ["--target", site, "--python-version", "3.11", "--python-platform", t["platform"], "--no-deps",
              "--no-config"]
    if t["torch"]:
        common += ["--torch-backend", t["torch"]]
    env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET=MACOS_MIN)
    log(f"installing {name} packages from packaging/locks/{name}.txt")
    run([uv(), "pip", "install", *common, "-r", PKG / "locks" / f"{name}.txt"], env=env)
    run([uv(), "pip", "install", *common, "--reinstall-package", "utaucover", wheel], env=env)
    shutil.rmtree(site / "bin", ignore_errors=True)  # console scripts from --target installs point at the build


def add_ffmpeg(name: str, runtime: Path):
    """A static ffmpeg build, next to the bundled python (UtauCover puts that folder on PATH)."""
    t = TARGETS[name]
    tmp = CACHE / f"ffmpeg-{name}"
    if not any(tmp.glob("imageio_ffmpeg/binaries/ffmpeg*")):
        run([uv(), "pip", "install", "--target", tmp, "--python-version", "3.11", "--python-platform",
             t["platform"], "--no-deps", "--no-config", f"imageio-ffmpeg=={IMAGEIO_FFMPEG}"],
            env=dict(os.environ, MACOSX_DEPLOYMENT_TARGET=MACOS_MIN))
    src = next(p for p in (tmp / "imageio_ffmpeg" / "binaries").iterdir() if p.name.startswith("ffmpeg"))
    dst = runtime / "ffmpeg.exe" if t["os"] == "windows" else runtime / "bin" / "ffmpeg"
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def trim(name: str, runtime: Path):
    """Leave out what an app never uses: Python's test suite / IDLE / Tk, build headers, static libraries."""
    win = TARGETS[name]["os"] == "windows"
    lib = runtime / "Lib" if win else runtime / "lib" / "python3.11"
    site = site_packages(name, runtime)
    gone = [lib / d for d in ("test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3/tests")]
    gone += [site / d for d in ("pip", "torch/include", "torch/share/cmake", "onnx/backend/test",
                                "numpy/_core/tests", "scipy/_lib/tests", "pandas/tests")]
    gone += list(site.glob("pip-*.dist-info"))
    if win:
        gone += [runtime / d for d in ("tcl", "Scripts", "include", "libs")]
        gone += list((runtime / "DLLs").glob("_tkinter*")) + list((runtime / "DLLs").glob("t[ck]l*.dll"))
        gone += list((site / "torch" / "lib").glob("*.lib"))
    else:
        gone += list((runtime / "lib").glob("tcl*")) + list((runtime / "lib").glob("tk*")) + \
                list((runtime / "lib").glob("itcl*")) + list((runtime / "lib").glob("thread*")) + \
                list((runtime / "lib").glob("libtcl*")) + list((runtime / "lib").glob("libtk*")) + \
                [runtime / "share", runtime / "lib" / "pkgconfig"]
        gone += list(lib.glob("lib-dynload/_tkinter*"))
        for b in (runtime / "bin").glob("*"):
            if b.name.startswith(("idle", "pydoc", "2to3", "pip", "python3-config", "python3.11-config")):
                gone.append(b)
    gone += [p for p in site.rglob("*.a")]
    # packages' own test suites and onnxruntime's model-conversion tools: never imported by the app, and
    # their deep paths are the ones that come close to Windows' 260-character limit
    gone += [d for d in site.rglob("tests") if d.is_dir()] + [site / "onnxruntime" / "tools",
                                                              site / "pkg_resources" / "tests"]
    for p in gone:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists() or p.is_symlink():
            p.unlink()
    for pyc in runtime.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)


def compile_bytecode(name: str, runtime: Path):
    """Precompile .pyc files (the app never writes into its own folder, and first start is faster)."""
    lib = runtime / "Lib" if TARGETS[name]["os"] == "windows" else runtime / "lib" / "python3.11"
    log("compiling bytecode")
    subprocess.run([host_python(), "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "unchecked-hash",
                    "-x", r"[/\\](test|tests)[/\\].*_py2|lib2to3[/\\]tests|bad_?syntax", str(lib)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------------------- launchers
def zig() -> list[str]:
    return [uv(), "run", "--no-project", "--with", "ziglang", "python", "-m", "ziglang"]


def build_launchers(name: str, runtime: Path, out_dir: Path):
    t = TARGETS[name]
    src = PKG / "launcher" / "launcher.c"
    if t["os"] == "macos":
        inc = runtime / "include" / "python3.11"
        for gui, exe in ((1, "UtauCover"), (0, "utaucover-cli")):
            run(["clang", "-O2", "-arch", "arm64", f"-mmacosx-version-min={MACOS_MIN}", f"-DUC_GUI={gui}",
                 "-I", inc, src, "-framework", "CoreFoundation", "-o", out_dir / exe])
    elif t["os"] == "windows":
        inc = runtime / "include"
        rc = BUILD / name / "launcher.rc"
        ver = version().split(".") + ["0"] * 4
        v4 = ",".join(ver[:4])
        rc.write_text(RC_TEMPLATE.format(icon=str(PKG / "windows" / "UtauCover.ico").replace("\\", "\\\\"),
                                         v4=v4, version=version()), encoding="utf-8")
        res = rc.with_suffix(".res")  # compiled on its own: zig cc's built-in .rc step is flaky
        run([*zig(), "rc", rc, res])
        for gui, exe in ((1, "UtauCover.exe"), (0, "utaucover-cli.exe")):
            run([*zig(), "cc", "-target", "x86_64-windows-gnu", "-O2", f"-DUC_GUI={gui}", "-I", inc, src, res,
                 "-o", out_dir / exe, f"-Wl,--subsystem,{'windows' if gui else 'console'}", "-lshell32", "-luser32"])
        for junk in out_dir.glob("*.pdb"):
            junk.unlink()
    else:
        for exe, module in (("UtauCover", "utaucover.gui"), ("utaucover-cli", "utaucover")):
            f = out_dir / exe
            f.write_text(LINUX_LAUNCHER.format(module=module), encoding="utf-8")
            f.chmod(0o755)


RC_TEMPLATE = """1 ICON "{icon}"
1 VERSIONINFO
 FILEVERSION {v4}
 PRODUCTVERSION {v4}
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "FileDescription", "UtauCover"
      VALUE "ProductName", "UtauCover"
      VALUE "FileVersion", "{version}"
      VALUE "ProductVersion", "{version}"
      VALUE "OriginalFilename", "UtauCover.exe"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"""

LINUX_LAUNCHER = """#!/bin/sh
# UtauCover launcher: runs the Python bundled in this folder. Works from anywhere, through symlinks too.
HERE="$(dirname "$(readlink -f "$0")")"
export UTAUCOVER_APP_DIR="$HERE"
export QT_API=pyside6
# The window only shows UtauCover's own page, and Chromium's sandbox fails where unprivileged user
# namespaces are restricted (Ubuntu 24.04+), so Qt WebEngine runs without it.
export QTWEBENGINE_DISABLE_SANDBOX=1
unset PYTHONHOME PYTHONPATH
exec "$HERE/runtime/bin/python3.11" -I -B -X utf8 -m {module} "$@"
"""


# ------------------------------------------------------------------------------------------- checks
MAGIC = {"macos": (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf"), "linux": (b"\x7fELF",),
         "windows": (b"MZ",)}
NATIVE_EXT = {"macos": (".so", ".dylib"), "linux": (".so",), "windows": (".pyd", ".dll", ".exe")}
FOREIGN_OK = ("samplerate/_samplerate_data/",)  # ships every OS's library; picks its own at run time


def verify_binaries(name: str, root: Path):
    """Every native library must be built for the target OS (catches a source-only package that got
    compiled for the build computer instead)."""
    os_ = TARGETS[name]["os"]
    bad = []
    for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink():
            continue
        n = p.name.lower()
        native = n.endswith(NATIVE_EXT[os_]) or (os_ == "linux" and ".so." in n)
        foreign = any(n.endswith(e) for o, exts in NATIVE_EXT.items() if o != os_ for e in exts if e in (".so", ".dylib", ".pyd"))
        if not (native or foreign):
            continue
        rel = p.relative_to(root).as_posix()
        if any(k in rel for k in FOREIGN_OK):
            continue
        with open(p, "rb") as f:
            head = f.read(4)
        if not head.startswith(MAGIC[os_]):
            bad.append(rel)
    if bad:
        sys.exit(f"{len(bad)} native file(s) aren't built for {name} (cross-built from source?):\n  " +
                 "\n  ".join(bad[:20]) + "\nBuild this target on its own OS (see the CI pipeline).")
    log(f"native libraries check: all built for {name}")


def check_path_lengths(app: Path, budget: int = 170):
    """Windows Explorer can't extract a zip entry whose full path passes 260 characters. Paths inside the
    package stay under 170, so it unzips into any folder up to ~90 characters (C:\\Users\\<name>\\Downloads\\
    UtauCover-<v>-windows-x64-portable\\ is ~70) and installs under AppData\\Local\\Programs or Program Files."""
    longest = max((p for p in app.rglob("*")), key=lambda p: len(str(p.relative_to(app.parent))))
    n = len(str(longest.relative_to(app.parent)))
    log(f"longest path inside the package: {n} characters")
    if n > budget:
        sys.exit(f"{longest.relative_to(app.parent)} is {n} characters: too long to unzip with Explorer")


def check_glibc(root: Path, limit=(2, 28)):
    """Linux: no native file may need a newer glibc than the oldest distributions we support (2.28)."""
    tool = ["readelf", "-V", "-W"] if shutil.which("readelf") else ["objdump", "-T"] if shutil.which("objdump") else None
    if tool is None:
        log("neither readelf nor objdump found: skipping the glibc check")
        return
    worst, too_new = (0, 0), []
    for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink() or not (p.name.endswith(".so") or ".so." in p.name or
                                                       os.access(p, os.X_OK)):
            continue
        with open(p, "rb") as f:
            if f.read(4) != b"\x7fELF":
                continue
        out = subprocess.run([*tool, str(p)], capture_output=True, text=True).stdout
        vers = [tuple(map(int, m)) for m in re.findall(r"GLIBC_(\d+)\.(\d+)", out)]
        if vers:
            v = max(vers)
            worst = max(worst, v)
            if v > limit:
                too_new.append(f"{p.relative_to(root)} needs glibc {v[0]}.{v[1]}")
    if too_new:
        sys.exit("native files need a newer glibc than 2.28:\n  " + "\n  ".join(too_new[:20]))
    log(f"glibc check: newest needed is {worst[0]}.{worst[1]} (limit {limit[0]}.{limit[1]})")


def selftest(cli: Path, gui: bool):
    log("running the bundle's selftest")
    run([cli, "selftest", *(["--gui"] if gui else [])])


# ------------------------------------------------------------------------------------------ notices
def write_notices(folder: Path):
    (folder / "THIRD-PARTY-NOTICES.txt").write_text(NOTICES, encoding="utf-8")


NOTICES = """UtauCover bundles third-party software, each under its own license:

- Python (python-build-standalone): Python Software Foundation License. https://github.com/astral-sh/python-build-standalone
- ffmpeg (static build from imageio-ffmpeg): GPL v3. Source: https://ffmpeg.org/download.html
- PyTorch, torchaudio: BSD-3-Clause. ONNX Runtime: MIT. NumPy, SciPy, librosa: BSD/ISC.
- openai-whisper: MIT. stable-ts: MIT. audio-separator: MIT. basic-pitch: Apache-2.0. torchfcpe: MIT.
- pywebview: BSD-3-Clause. pythonnet: MIT. PySide6 / Qt (Linux build): LGPL v3, https://www.qt.io/licensing
- The full list of Python packages and versions is in packaging/locks/ of UtauCover's source.

AI models (stem separation, Whisper, GAME) are downloaded on first use from their authors and are covered
by their own licenses. GAME is run from your UtauV / OpenUtau installation and isn't included.
"""


# ---------------------------------------------------------------------------------------- per OS
def build_macos(name: str, test: bool, gui_test: bool) -> list[Path]:
    v = version()
    work = BUILD / name
    app = work / "UtauCover.app"
    shutil.rmtree(work, ignore_errors=True)
    (app / "Contents" / "MacOS").mkdir(parents=True)
    res = app / "Contents" / "Resources"
    res.mkdir()
    runtime = res / "runtime"
    python_runtime(name, runtime)
    install_packages(name, runtime)
    add_ffmpeg(name, runtime)
    trim(name, runtime)
    build_launchers(name, runtime, app / "Contents" / "MacOS")
    compile_bytecode(name, runtime)
    shutil.copy2(PKG / "macos" / "UtauCover.icns", res / "UtauCover.icns")
    write_notices(res)
    (app / "Contents" / "Info.plist").write_text(
        (PKG / "macos" / "Info.plist").read_text().replace("@VERSION@", v).replace("@MACOS_MIN@", MACOS_MIN))
    verify_binaries(name, app)
    sign_macos(app)
    if test:
        selftest(app / "Contents" / "MacOS" / "utaucover-cli", gui_test)
        sign_macos(app)  # in case the test run touched anything (it shouldn't)
    DIST.mkdir(exist_ok=True)
    dmg = DIST / f"UtauCover-{v}-{name}.dmg"
    stage = work / "dmg"
    stage.mkdir()
    shutil.move(str(app), stage / "UtauCover.app")
    (stage / "Applications").symlink_to("/Applications")
    dmg.unlink(missing_ok=True)
    log("making the disk image")
    run(["hdiutil", "create", "-volname", f"UtauCover {v}", "-srcfolder", stage, "-fs", "HFS+", "-format", "ULMO",
         "-ov", dmg])
    notarize(dmg)
    return [dmg]


def sign_macos(app: Path):
    """Sign every binary inside-out. With MACOS_SIGN_IDENTITY (a "Developer ID Application" certificate)
    the app can be notarized; without, it's ad-hoc signed (runs on this Mac; other Macs need right-click >
    Open, or System Settings > Privacy & Security > Open Anyway)."""
    ident = os.environ.get("MACOS_SIGN_IDENTITY", "-")
    ent = PKG / "macos" / "entitlements.plist"
    opts = ["--force", "--timestamp", "--options", "runtime", "--entitlements", ent] if ident != "-" else ["--force"]
    log(f"signing ({'ad-hoc' if ident == '-' else ident})")
    subprocess.run(["xattr", "-cr", str(app)], check=False)  # Finder info / quarantine flags break signing
    machos = []
    for p in app.rglob("*"):
        if p.is_file() and not p.is_symlink():
            with open(p, "rb") as f:
                if f.read(4) in MAGIC["macos"]:
                    machos.append(p)
    # deepest first, the app's own launchers last
    machos.sort(key=lambda p: (p.parent.name == "MacOS", -len(p.parts)))
    for i in range(0, len(machos), 200):
        r = subprocess.run(["codesign", "-s", ident, *map(str, opts), *map(str, machos[i:i + 200])],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit("codesign failed:\n" + "\n".join(l for l in r.stderr.splitlines()
                                                     if "replacing existing signature" not in l))
    log(f"signed {len(machos)} binaries")
    run(["codesign", "-s", ident, *opts, app])
    run(["codesign", "--verify", "--deep", "--strict", app])


def notarize(dmg: Path):
    profile = os.environ.get("MACOS_NOTARY_PROFILE")  # xcrun notarytool store-credentials <profile>
    if not profile or os.environ.get("MACOS_SIGN_IDENTITY", "-") == "-":
        return
    run(["codesign", "-s", os.environ["MACOS_SIGN_IDENTITY"], "--timestamp", dmg])
    run(["xcrun", "notarytool", "submit", dmg, "--keychain-profile", profile, "--wait"])
    run(["xcrun", "stapler", "staple", dmg])


def build_windows(name: str, test: bool, gui_test: bool) -> list[Path]:
    v = version()
    work = BUILD / name
    shutil.rmtree(work, ignore_errors=True)
    app = work / "UtauCover"
    app.mkdir(parents=True)
    runtime = app / "runtime"
    python_runtime(name, runtime)
    install_packages(name, runtime)
    add_ffmpeg(name, runtime)
    build_launchers(name, runtime, app)  # needs runtime/include and runtime/libs, so before trim()
    trim(name, runtime)
    compile_bytecode(name, runtime)
    shutil.copy2(PKG / "windows" / "UtauCover.ico", app / "UtauCover.ico")
    write_notices(app)
    (app / "README.txt").write_text(WINDOWS_README.format(version=v), encoding="utf-8")
    verify_binaries(name, app)
    check_path_lengths(app)
    if test:
        selftest(app / "utaucover-cli.exe", gui_test)
    DIST.mkdir(exist_ok=True)
    out = []
    # portable: data stays in the folder (portable.txt), so the whole thing can be moved or deleted
    (app / "portable.txt").write_text(PORTABLE_NOTE, encoding="utf-8")
    z = DIST / f"UtauCover-{v}-{name}-portable.zip"
    log(f"zipping {z.name}")
    zip_folder(app, z)
    out.append(z)
    (app / "portable.txt").unlink()
    iscc = find_iscc()
    if iscc:
        log("building the installer")
        run([iscc, "/Qp", f"/DAppVersion={v}", f"/DSourceDir={app}", f"/DOutputDir={DIST}",
             f"/DOutputName=UtauCover-{v}-{name}-setup", PKG / "windows" / "UtauCover.iss"])
        out.append(DIST / f"UtauCover-{v}-{name}-setup.exe")
    else:
        log("Inno Setup (iscc) not found: skipping the installer (it's built by the CI pipeline on Windows)")
    return out


def find_iscc() -> str | None:
    exe = shutil.which("iscc") or shutil.which("ISCC")
    if exe:
        return exe
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            for p in Path(base).glob("Inno Setup */ISCC.exe"):
                return str(p)
    return None


def zip_folder(folder: Path, dest: Path):
    dest.unlink(missing_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(folder.rglob("*")):
            arc = Path(folder.name) / p.relative_to(folder)
            if p.is_dir():
                continue
            z.write(p, arc.as_posix())


WINDOWS_README = """UtauCover {version}

Double-click UtauCover.exe. Drop a song on the window, build, and open the project in UtauV / OpenUtau.

- Command line (and Claude's MCP server): utaucover-cli.exe --help, utaucover-cli.exe mcp --print-config
- AI models download the first time they're needed (about 3 GB for the default settings).
- GAME, the recommended note transcriber, comes from UtauV / OpenUtau: install it there.
- If Windows SmartScreen says "Windows protected your PC", click "More info" > "Run anyway"
  (the app isn't code-signed).
"""

PORTABLE_NOTE = """This is a portable copy: UtauCover keeps its settings, stem cache and AI models in the "data" folder
next to this file, so the whole folder can be moved or deleted. Delete this file to keep data in
%LOCALAPPDATA%\\UtauCover instead (shared with an installed copy).
"""


def build_linux(name: str, test: bool, gui_test: bool) -> list[Path]:
    v = version()
    work = BUILD / name
    shutil.rmtree(work, ignore_errors=True)
    app = work / "UtauCover"
    (app / "resources").mkdir(parents=True)
    runtime = app / "runtime"
    python_runtime(name, runtime)
    install_packages(name, runtime)
    add_ffmpeg(name, runtime)
    trim(name, runtime)
    compile_bytecode(name, runtime)
    build_launchers(name, runtime, app)
    for f in ("install.sh",):
        shutil.copy2(PKG / "linux" / f, app / f)
        (app / f).chmod(0o755)
    shutil.copy2(PKG / "linux" / "utaucover.png", app / "resources" / "utaucover.png")
    write_notices(app)
    (app / "README.txt").write_text(LINUX_README.format(version=v), encoding="utf-8")
    verify_binaries(name, app)
    check_glibc(app)
    if test:
        selftest(app / "utaucover-cli", gui_test)
    DIST.mkdir(exist_ok=True)
    out = DIST / f"UtauCover-{v}-{name}.tar.gz"
    log(f"packing {out.name}")
    with tarfile.open(out, "w:gz", compresslevel=6) as tf:
        tf.add(app, arcname=f"UtauCover-{v}")
    return [out]


LINUX_README = """UtauCover {version}

Run ./UtauCover from this folder. To add it to the applications menu and put `utaucover` on your PATH:

    ./install.sh            # for you only: ~/.local/share/UtauCover, no root needed
    sudo ./install.sh       # for everyone: /opt/UtauCover
    ./install.sh --uninstall

install.sh also checks for the few system libraries the window needs (most desktops have them) and
offers to install any that are missing.

- Command line (and Claude's MCP server): ./utaucover-cli --help, ./utaucover-cli mcp --print-config
- AI models download the first time they're needed (about 3 GB for the default settings).
- GAME, the recommended note transcriber, comes from OpenUtau: install it there.
"""


# -------------------------------------------------------------------------------------------- main
def main():
    global DIST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="build", choices=["build", "lock"])
    ap.add_argument("--target", action="append", choices=list(TARGETS), help="default: this computer's OS")
    ap.add_argument("--no-test", action="store_true", help="skip the bundle's selftest")
    ap.add_argument("--gui-test", action="store_true", help="also open the window in the selftest")
    ap.add_argument("--out", type=Path, help=f"where the packages go (default: {DIST})")
    a = ap.parse_args()
    if a.out:
        DIST = a.out.expanduser().resolve()
    if a.command == "lock":
        lock(a.target or list(TARGETS))
        return
    targets = a.target or [host_target()]
    made = []
    for name in targets:
        if name not in TARGETS:
            sys.exit(f"no build for {name}")
        native = name == host_target()
        if TARGETS[name]["os"] == "macos" and sys.platform != "darwin":
            sys.exit("the macOS app can only be built on a Mac (signing, disk image)")
        log(f"building UtauCover {version()} for {name}" + ("" if native else " (cross build)"))
        builder = {"macos": build_macos, "windows": build_windows, "linux": build_linux}[TARGETS[name]["os"]]
        made += builder(name, test=native and not a.no_test, gui_test=a.gui_test)
    log(f"done (work folder {BUILD}):")
    for p in made:
        print(f"  {p}  ({p.stat().st_size / 1e6:,.0f} MB)")


if __name__ == "__main__":
    main()
