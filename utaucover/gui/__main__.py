"""The packaged app's window entry point (python -m utaucover.gui).

Output goes to a log file, and if the window can't open the user gets a message box instead of nothing
(a windowed program has no console to print the error to).
"""

from __future__ import annotations

import subprocess
import sys
import traceback


def _tell_user(msg: str):
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg, "UtauCover", 0x10)
        elif sys.platform == "darwin":
            subprocess.run(["osascript", "-e", "on run argv", "-e",
                            'display alert "UtauCover couldn\'t open its window" message (item 1 of argv) as critical',
                            "-e", "end run", msg], check=False)
        else:
            for cmd in (["zenity", "--error", "--title=UtauCover", f"--text={msg}"],
                        ["kdialog", "--title", "UtauCover", "--error", msg],
                        ["notify-send", "UtauCover", msg]):
                try:
                    subprocess.run(cmd, check=False)
                    break
                except OSError:
                    continue
    except Exception:  # noqa: BLE001
        pass


def run():
    from ..paths import APP_SUPPORT
    from .app import _log_to_file, main
    _log_to_file()
    try:
        main()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        sys.stderr.flush()
        hint = ""
        if sys.platform == "win32" and "Python.Runtime" in str(e):
            hint = ("\n\nWindows is blocking UtauCover's .NET files because the .zip came from the internet. "
                    "Right-click the downloaded .zip > Properties > tick Unblock > OK, then extract it again "
                    "(or use the installer).")
        elif sys.platform == "win32" and "webview2" in str(e).lower():
            hint = ("\n\nInstall Microsoft Edge WebView2 Runtime (free) from "
                    "https://developer.microsoft.com/microsoft-edge/webview2/ and start UtauCover again.")
        elif sys.platform.startswith("linux") and ("xcb" in str(e).lower() or "qt" in str(e).lower()):
            hint = "\n\nRun ./install.sh --check in UtauCover's folder to find missing system libraries."
        _tell_user(f"{type(e).__name__}: {e}{hint}\n\nDetails: {APP_SUPPORT / 'logs' / 'window.log'}")
        sys.exit(1)


run()
