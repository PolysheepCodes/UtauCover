"""Drag-and-drop desktop window (pywebview) for building cover projects."""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from pathlib import Path

from ..paths import AUDIO_EXTS

HERE = Path(__file__).parent
LYRIC_EXTS = {".txt", ".lrc", ".text", ".md"}


def _dialog(kind: str):
    import webview
    fd = getattr(webview, "FileDialog", None)
    if fd is not None:
        return {"open": fd.OPEN, "folder": fd.FOLDER}[kind]
    return {"open": webview.OPEN_DIALOG, "folder": webview.FOLDER_DIALOG}[kind]


class Api:
    def __init__(self):
        # private (leading underscore): pywebview hands every public attribute to the page and walks into
        # it, and on Windows walking the native window object never ends (the window hung on start)
        self._window: webview.Window | None = None
        self._busy = False

    # ---- helpers called from JS -------------------------------------------------
    def get_singers(self):
        from ..phonemizers import by_class, find_phonemizers
        from ..singers import find_singers
        installed = find_phonemizers()
        out = []
        for s in find_singers():
            p = by_class(s.default_phonemizer, installed)
            out.append({"id": s.id, "name": s.name, "type": s.singer_type, "languages": s.languages,
                        "default_phonemizer": p.tag if p else (s.default_phonemizer or "")})
        return out

    def get_phonemizers(self, refresh=False):
        from ..phonemizers import find_phonemizers
        return [{"cls": p.cls, "tag": p.tag, "name": p.name, "language": p.language or "Any"}
                for p in find_phonemizers(refresh=bool(refresh))]

    def get_models(self, refresh=False):
        from ..models import note_engines, whisper_models
        if refresh:
            from ..apps import refresh as refresh_apps
            refresh_apps()
        return {"engines": note_engines(), "whisper": whisper_models()}

    def get_platform(self):
        from ..apps import programs, target_app
        from ..desktop import FILE_MANAGER
        app = target_app()
        return {"os": sys.platform, "file_manager": FILE_MANAGER, "app": app.name if app else "",
                "programs_found": bool(programs())}

    def choose_utau_dir(self):
        """Let the user point at a UtauV / OpenUtau folder the search didn't find."""
        from .. import apps
        from ..paths import save_settings, settings
        r = self._window.create_file_dialog(_dialog("folder"))
        if not r:
            return None
        folder = Path(r[0] if isinstance(r, (list, tuple)) else r)
        if apps._code_dir(folder) is None and not (folder / "Singers").is_dir():
            return {"ok": False, "message": f"No UtauV / OpenUtau program or Singers folder in {folder.name}"}
        dirs = [d for d in settings().get("utau_dirs", []) if d != str(folder)] + [str(folder)]
        save_settings(utau_dirs=dirs)
        apps.refresh()
        found = apps.programs()
        what = found[0].name if found else "its data folder"
        return {"ok": True, "message": f"Found {what}: loading voicebanks"}

    def open_url(self, url):
        import webbrowser
        if isinstance(url, str) and url.startswith("https://"):
            webbrowser.open(url)

    def get_version(self):
        from .. import __version__
        return __version__

    def browse_song(self):
        exts = ";".join(f"*{e}" for e in sorted(AUDIO_EXTS))
        r = self._window.create_file_dialog(_dialog("open"), file_types=(f"Audio ({exts})", "All files (*.*)"))
        return r[0] if r else None

    def browse_lyrics(self):
        r = self._window.create_file_dialog(_dialog("open"), file_types=("Text (*.txt;*.lrc)", "All files (*.*)"))
        if not r:
            return None
        return {"path": r[0], "text": self.read_text(r[0])}

    def browse_folder(self):
        r = self._window.create_file_dialog(_dialog("folder"))
        return r[0] if r else None

    def read_text(self, path):
        from ..lyrics import read_text
        try:
            return read_text(path)
        except Exception:
            return ""

    def default_out(self, song):
        from ..pipeline import default_out_dir
        return str(default_out_dir(song))

    def open_project(self, path):
        from ..cli import open_in_app
        return open_in_app(path)

    def reveal(self, path):
        from ..desktop import reveal
        reveal(path)

    def open_file(self, path):
        from ..desktop import open_path
        open_path(path)

    # ---- the build ----------------------------------------------------------------
    def build(self, o: dict):
        if self._busy:
            return {"error": "A build is already running."}
        if not o.get("song") or not Path(o["song"]).exists():
            return {"error": "Choose a song first."}
        self._busy = True
        threading.Thread(target=self._build, args=(o,), daemon=True).start()
        return {"ok": True}

    def _js(self, fn: str, *args):
        if self._window:
            self._window.evaluate_js(f"{fn}({', '.join(json.dumps(a, ensure_ascii=False) for a in args)})")

    def _build(self, o: dict):
        from ..pipeline import CoverOptions, build_cover

        def num(k, cast=float):
            v = o.get(k)
            try:
                return cast(v) if v not in (None, "") else None
            except ValueError:
                return None
        try:
            lyrics_mode = o.get("lyricsMode", "auto")
            opts = CoverOptions(
                song=o["song"], out_dir=o.get("outDir") or None,
                lyrics_text=(o.get("lyricsText") or None) if lyrics_mode != "auto" else None,
                language=o.get("language") or None, singer=o.get("singer") or None,
                harmony_singer=o.get("harmonySinger") or None, quality=o.get("quality", "best"),
                phonemizer=o.get("phonemizer") or None, harmony_phonemizer=o.get("harmonyPhonemizer") or None,
                align=o.get("align") or "whisper", respell_foreign=bool(o.get("respell", True)),
                harmonies=bool(o.get("harmonies", True)), backing=o.get("backing") or "auto",
                notes_engine=o.get("notesEngine") or "auto", whisper_model=o.get("whisperModel") or "turbo",
                guide_tracks=bool(o.get("guides", True)),
                instrument_stems=bool(o.get("instStems")), dereverb=bool(o.get("dereverb")),
                pitch_curve=bool(o.get("pitchCurve")), bpm=num("bpm"), snap=num("snap", int) or 0,
                transpose=num("transpose", int) or 0, lyric_mode=o.get("lyricMode") or None,
                fill_lyric=o.get("fillLyric") or None,
                script=o.get("script") or None)
            res = build_cover(opts, progress=lambda m: self._js("onProgress", m))
            lyr = ""
            if res.lyrics_file and Path(res.lyrics_file).exists():
                lyr = Path(res.lyrics_file).read_text(encoding="utf-8")
            self._js("onDone", res.to_dict(), lyr)
        except Exception as e:  # noqa: BLE001
            self._js("onError", f"{type(e).__name__}: {e}", traceback.format_exc(limit=3))
        finally:
            self._busy = False


def _log_to_file():
    """A windowed program has no console (on Windows print() output would vanish and libraries that write to
    stderr would crash). Send output to a log file in UtauCover's data folder instead, so problems can be
    looked into."""
    from ..paths import APP_DIR, APP_SUPPORT
    tty = all(f is not None and f.isatty() for f in (sys.stdout, sys.stderr))
    if (sys.stdout is not None and sys.stderr is not None) and \
            (tty or APP_DIR is None or os.environ.get("UTAUCOVER_GUI_SELFTEST")):
        return
    try:
        (APP_SUPPORT / "logs").mkdir(parents=True, exist_ok=True)
        f = open(APP_SUPPORT / "logs" / "window.log", "w", encoding="utf-8", buffering=1)  # noqa: SIM115
    except OSError:
        f = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    sys.stdout = sys.stderr = f
    try:  # native libraries and helper processes write to file descriptors 1/2: log those too
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
    except OSError:
        pass


def _gui_backend() -> str | None:
    """Linux: the bundled Qt web view (GTK's needs system packages a self-contained app can't rely on)."""
    if sys.platform.startswith("linux"):
        return "qt"
    return None


def _icon() -> str | None:
    """Window icon for Windows / Linux (macOS takes it from the .app bundle)."""
    from ..paths import APP_DIR
    if APP_DIR is None:
        return None
    for name in ("UtauCover.ico", "utaucover.png"):
        for d in (APP_DIR, APP_DIR / "resources"):
            if (d / name).exists() and (name.endswith(".ico") == (sys.platform == "win32")):
                return str(d / name)
    return None


def _selftest_trace(msg: str):
    if os.environ.get("UTAUCOVER_GUI_SELFTEST"):
        print(f"SELFTEST-STEP {msg}", file=sys.stderr, flush=True)


def main():
    if os.environ.get("UTAUCOVER_GUI_SELFTEST"):
        # if the window hangs, show where: every thread's stack, before the test gives up on it
        import faulthandler
        import logging
        faulthandler.dump_traceback_later(int(os.environ.get("UTAUCOVER_GUI_SELFTEST_DUMP", "150")), exit=True)
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    _selftest_trace("importing the web view")
    import webview
    _log_to_file()
    os.environ.setdefault("TQDM_DISABLE", "1")
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    if not os.environ.get("UTAUCOVER_GUI_SELFTEST"):
        logging.getLogger().setLevel(logging.ERROR)

    if _gui_backend() == "qt":
        # Name the Qt application so the desktop matches the window to utaucover.desktop (icon, taskbar)
        from qtpy.QtWidgets import QApplication
        qapp = QApplication.instance() or QApplication(["UtauCover", *sys.argv[1:]])
        qapp.setApplicationName("UtauCover")
        qapp.setDesktopFileName("utaucover")

    api = Api()
    from .. import __version__
    window = webview.create_window(f"UtauCover {__version__}", str(HERE / "index.html"), js_api=api,
                                   width=880, height=900, min_size=(640, 640),
                                   background_color="#16151c")
    api._window = window

    def bind(win):
        _selftest_trace("window created, binding events")
        from webview.dom import DOMEventHandler

        def on_drop(e):
            files = (e.get("dataTransfer") or {}).get("files") or []
            paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
            song = next((p for p in paths if Path(p).suffix.lower() in AUDIO_EXTS), None)
            lyr = next((p for p in paths if Path(p).suffix.lower() in LYRIC_EXTS), None)
            if song:
                api._js("setSong", song, api.default_out(song))
            if lyr:
                api._js("setLyricsFile", lyr, api.read_text(lyr))
            if paths and not song and not lyr:
                api._js("toast", "Drop an audio file (mp3, wav, flac, m4a…) or a lyrics .txt")

        win.dom.document.events.dragover += DOMEventHandler(lambda e: None, True, True, debounce=200)
        win.dom.document.events.drop += DOMEventHandler(on_drop, True, True)

        # a song handed to the program ("Open with", dropped on the app icon / exe, the Linux menu entry)
        given = [a for a in sys.argv[1:] if Path(a).suffix.lower() in AUDIO_EXTS and Path(a).is_file()]
        if given:
            song = str(Path(given[0]).resolve())
            api._js("setSong", song, api.default_out(song))

        if os.environ.get("UTAUCOVER_GUI_SELFTEST"):
            import time
            _selftest_trace("page loaded, checking the form")
            for _ in range(40):
                n = win.evaluate_js("document.querySelectorAll('#singer option').length")
                if n and int(n) > 1:
                    break
                time.sleep(0.25)
            n = n or 1
            for _ in range(40):
                ph = win.evaluate_js("document.querySelectorAll('#phonemizer option').length")
                if ph and int(ph) > 1:
                    break
                time.sleep(0.25)
            win.evaluate_js("setSong('/tmp/example song.mp3', '/tmp/example song UtauCover')")
            btn = win.evaluate_js("document.getElementById('buildBtn').disabled")
            title = win.evaluate_js("document.title")
            win.evaluate_js("document.getElementById('singer').value='Googoloid English release_1.3'; updateSingerHint()")
            hint = win.evaluate_js("document.getElementById('singerHint').textContent")
            backing = win.evaluate_js("[...document.querySelectorAll('#backing option')].map(o => o.value).join(',') + "
                                      "' default=' + document.getElementById('backing').value")
            for _ in range(20):
                eng = win.evaluate_js("[...document.querySelectorAll('#notesEngine option')].map(o => o.value + (o.disabled ? '(off)' : '')).join(',')")
                if eng and "game" in eng:
                    break
                time.sleep(0.25)
            wm = win.evaluate_js("[...document.querySelectorAll('#whisperModel option')].map(o => o.textContent).join(' | ')")
            eng += " selected=" + str(win.evaluate_js("document.getElementById('notesEngine').value")) + \
                " banner=" + ("shown" if not win.evaluate_js("document.getElementById('gameBanner').classList.contains('hidden')") else "hidden")
            banner = "shown" if not win.evaluate_js("document.getElementById('appBanner').classList.contains('hidden')") else "hidden"
            fm = win.evaluate_js("document.getElementById('fileManager').textContent")
            print(f"SELFTEST app_banner={banner} file_manager={fm!r} singers_in_dropdown={int(n) - 1} phonemizers_in_dropdown={int(ph) - 1} engines={eng!r} "
                  f"whisper={wm!r} "
                  f"build_button_enabled={not btn} title={title!r} googoloid_hint={hint!r} backing={backing!r}",
                  file=sys.stderr, flush=True)
            win.destroy()

    kwargs = {}
    if _gui_backend():
        kwargs["gui"] = _gui_backend()
    if _icon():
        kwargs["icon"] = _icon()
    _selftest_trace(f"starting the window ({kwargs or 'default backend'})")
    webview.start(bind, window, **kwargs)


if __name__ == "__main__":
    main()
