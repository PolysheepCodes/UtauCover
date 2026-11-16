# Packaging UtauCover

UtauCover ships as a self-contained app on each OS. Nothing has to be installed first: each package carries its
own Python ([python-build-standalone](https://github.com/astral-sh/python-build-standalone)), every library,
a static ffmpeg and a native launcher. AI models are not bundled. They download the first time they're used,
with progress in the window: about 3 GB for the default settings.

| OS | File | How people install it |
|---|---|---|
| macOS 14+ (Apple Silicon) | `UtauCover-<v>-macos-arm64.dmg` | Open it, drag **UtauCover** onto **Applications** |
| Windows 10/11 (x64) | `UtauCover-<v>-windows-x64-portable.zip` | Unzip anywhere, double-click `UtauCover.exe`. No install, no admin. Settings, stem cache and models stay in its `data` folder |
| | `UtauCover-<v>-windows-x64-setup.exe` | Installer. **Just for me** (default) installs to `%LOCALAPPDATA%\Programs\UtauCover` with no admin prompt. **All users** installs to `C:\Program Files\UtauCover` and asks for admin |
| Linux x64 (glibc 2.28+: Ubuntu 20.04+, Debian 10+, Fedora 29+, RHEL 8+) | `UtauCover-<v>-linux-x64.tar.gz` | Extract and run `./UtauCover`, or `./install.sh` for a menu entry and a `utaucover` command |

**Can the Windows installer write to Program Files without admin?** No. Windows only lets administrators write
there. The installer's default **Just for me** mode is the standard no-admin install, the same as VS Code,
Discord and Spotify use. It installs per user in `AppData\Local\Programs` and gets a Start-menu entry and an
uninstaller in Settings > Apps. **All users** goes to Program Files and asks for admin once.

## Building

Everything goes through one script and needs only [uv](https://docs.astral.sh/uv/):

```bash
uv run --no-project --python 3.11 python packaging/build.py                     # this computer's OS
uv run --no-project --python 3.11 python packaging/build.py --target windows-x64 # another OS (see below)
```

- Packages go to `~/Library/Caches/UtauCover-build/dist` (macOS), `%LOCALAPPDATA%\UtauCover-build\dist`
  (Windows), or `~/.cache/utaucover-build/dist` (Linux). Use `--out DIR` to put them somewhere else.
  The work folder is deliberately outside the source tree. A synced folder (iCloud Drive, OneDrive) would
  upload about 2 GB per build, and its file tags break macOS code signing.
- A build for the computer's own OS runs the bundle's `selftest`: every library, ffmpeg, a pitch track, a
  basic-pitch pass, the separator and the window backend. `--gui-test` also opens the window.
- macOS needs the Xcode command line tools (clang, codesign, hdiutil). The Windows launcher is compiled with
  [zig](https://ziglang.org), which uv fetches automatically, so it builds on any OS. The Windows installer
  needs [Inno Setup 6](https://jrsoftware.org/isdl.php), which only runs on Windows. The CI pipeline makes it.

**Cross builds.** The Windows and Linux packages can be built on a Mac (or on each other). Everything they
use ships prebuilt wheels, and the build stops if any native file isn't built for the target OS. For Linux it
also stops if any native file needs a newer glibc than 2.28. A cross build can't run the bundle's selftest,
so release builds go through the pipeline, which builds and tests each OS on that OS. The macOS app needs a
Mac.

### The pipeline (GitHub Actions)

`.github/workflows/build.yml` builds and tests all three on GitHub's own macOS, Windows and Linux machines:

- **Push a tag** (`git tag v0.7.0 && git push origin v0.7.0`): the packages are attached to a draft GitHub
  release. Check it, then publish.
- **Every push to `main`** that changes the app or its packaging, and **Actions → Build packages → Run
  workflow** (which can build just one OS): the packages are kept as downloadable artifacts for 7 days.
- Besides each bundle's selftest (libraries, ffmpeg, a pitch track, basic-pitch, the separator, the window),
  the Windows job tests the zip as a browser download leaves it (every file marked "from the internet") and
  the installer's install → run → uninstall. The Linux job tests `install.sh`.

A run takes about 25–35 minutes (Windows is the slowest). The repository is public, so GitHub Actions
minutes and artifact storage are free.

### Signing

- **macOS**: without a certificate the app is ad-hoc signed. It runs on the Mac that built it. On other Macs
  the first launch needs **System Settings → Privacy & Security → Open Anyway**. With an Apple Developer
  account ($99/year), set `MACOS_SIGN_IDENTITY="Developer ID Application: …"` and `MACOS_NOTARY_PROFILE`
  (from `xcrun notarytool store-credentials`). The build then signs with the hardened runtime and notarizes
  the disk image. In the pipeline, add the secrets listed at the top of `build.yml`.
- **Windows**: the exe isn't code-signed, so SmartScreen shows "Windows protected your PC" until a download
  has a reputation. Users click **More info → Run anyway**. Signing needs a code-signing certificate.

## What's in a package

```
UtauCover.app/Contents/            UtauCover\                     UtauCover-<v>/
  MacOS/UtauCover   (window)         UtauCover.exe     (window)     UtauCover        (window)
  MacOS/utaucover-cli               utaucover-cli.exe              utaucover-cli
  Resources/runtime/ (Python+libs)  runtime\                       runtime/
                                    portable.txt (zip only)        install.sh
```

- The launcher (`launcher/launcher.c`, the shell scripts on Linux) loads the bundled Python and runs
  `utaucover.gui` (the window) or `utaucover` (command line). It sets `UTAUCOVER_APP_DIR`, which tells
  UtauCover it's packaged (see `utaucover/_bundle.py`).
- `utaucover-cli mcp --print-config` prints the entry for Claude Desktop / Claude Code's MCP settings.
- Python's bytecode is precompiled, and the app never writes into its own folder. A signed `.app` or a
  Program Files install stays untouched.
- Dependencies are pinned per OS in `locks/`. After changing `pyproject.toml`, run
  `python packaging/build.py lock` and commit the new lock files. `overrides.txt` leaves out TensorFlow
  (basic-pitch uses its ONNX model instead) and Triton. On Linux it swaps diffq for diffq-fixed, which ships
  wheels, so no package needs compiling.

## Where user data goes

| | macOS | Windows | Linux |
|---|---|---|---|
| Settings, stem cache, separation models | `~/Library/Application Support/UtauCover` | `%LOCALAPPDATA%\UtauCover` (portable zip: its `data` folder) | `~/.local/share/UtauCover` |
| Whisper models | `~/.cache/whisper` | `%USERPROFILE%\.cache\whisper` (portable: `data\models\whisper`) | `~/.cache/whisper` |
| Window log | `…/UtauCover/logs/window.log` | same | same |

UtauCover finds UtauV / OpenUtau automatically in the usual places: `/Applications`; Program Files and
`AppData\Local\Programs`; unzipped copies in Downloads, Desktop or the home folder; the Windows uninstall list
and the `.ustx` file association; `/opt` and `~/.local/share` on Linux. For anywhere else, the window shows a
**Choose its folder…** button. You can also set `UTAUCOVER_UTAU_DIRS` to the folder.
