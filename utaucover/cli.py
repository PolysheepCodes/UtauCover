"""Command line: `utaucover build song.mp3 --lyrics lyrics.txt --singer Googoloid`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _progress(msg: str):
    if msg.startswith("[stage]"):
        return
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _quiet_libs():
    import logging
    import os
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TQDM_DISABLE", "1")
    for name in ("audio_separator", "basic_pitch", "coremltools", "root", "numba", "matplotlib"):
        logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)


def cmd_build(a):
    from .pipeline import CoverOptions, build_cover
    opts = CoverOptions(
        song=a.song, out_dir=a.out, lyrics_file=a.lyrics, language=a.language, lyric_mode=a.lyric_mode,
        script=a.script, backing_in_parens=not a.no_paren_backing, singer=a.singer, harmony_singer=a.harmony_singer,
        phonemizer=a.phonemizer, harmony_phonemizer=a.harmony_phonemizer, align=a.align,
        respell_foreign=not a.no_respell,
        quality=a.quality, harmonies=not a.no_harmonies, backing=a.backing, max_harmony_lines=a.max_harmonies,
        notes_engine=a.notes,
        instrument_stems=a.stems, six_stems=a.six_stems, dereverb=a.dereverb, pitch_curve=a.pitch_curve,
        bpm=a.bpm, snap=a.snap, transpose=a.transpose, whisper_model=a.whisper_model, lyrics=not a.no_lyrics,
        fill_lyric=a.fill_lyric, legato_gap=0.0 if a.no_legato else 0.3,
        guide_tracks=not a.no_guides, midi=not a.no_midi, project_name=a.name)
    if a.notes == "auto":
        from .game import INSTALL_HINT, find_game
        if find_game() is None:
            print("WARNING: GAME isn't installed, so notes will come from the built-in engine (much less accurate).\n"
                  f"         {INSTALL_HINT}\n", file=sys.stderr)
    res = build_cover(opts, progress=_progress)
    print()
    print(f"Project:  {res.ustx}")
    if res.midi:
        print(f"MIDI:     {res.midi}")
    print(f"Lyrics:   {res.lyrics_file}  (edit and pass back with --lyrics to fix words)")
    print(f"Pronunciation: {res.pronunciation}")
    print(f"          {res.pronunciation_report}")
    print(f"Tempo:    {res.bpm:g} BPM   language: {res.language}")
    for t in res.tracks:
        print(f"  - {t['name']}: {t['notes']} notes  [{t['singer'] or 'no singer set'} · {t.get('phonemizer_tag') or t['phonemizer']}]")
    for n in res.notes:
        print(f"note:    {n}")
    if res.check_file:
        print(f"Check by ear: {res.check_file}")
    for w in res.warnings:
        print(f"warning: {w}")
    if a.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    if a.open:
        print(open_in_app(res.ustx))


def cmd_separate(a):
    from .pipeline import default_out_dir, publish_stems, stem_cache_dir
    from .separate import StemSeparator
    out = Path(a.out) if a.out else default_out_dir(a.song) / "stems"
    sep = StemSeparator(stem_cache_dir(a.song, a.quality), quality=a.quality, progress=_progress)
    try:
        stems = sep.separate(a.song, backing_split=not a.no_backing, dereverb=a.dereverb,
                             instruments=a.stems or a.six_stems, six_stems=a.six_stems)
    finally:
        sep.close()
    stems = publish_stems(stems, out)
    for k, v in stems.items():
        print(f"{k:13} {v}")


def cmd_lyrics(a):
    from .lyrics import LyricEngine
    from .pipeline import _words_to_text
    eng = LyricEngine(a.whisper_model, progress=_progress)
    words, lang = eng.transcribe(a.audio, a.language)
    text = _words_to_text(words, lang)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(text)


def cmd_singers(a):
    from .phonemizers import by_class, find_phonemizers
    from .singers import find_singers
    singers = find_singers()
    if a.json:
        print(json.dumps([s.to_dict() for s in singers], indent=2, ensure_ascii=False))
        return
    if not singers:
        from .apps import data_dirs
        where = ", ".join(str(d / "Singers") for d in data_dirs()) or "any UtauV / OpenUtau data folder"
        print(f"No voicebanks found (looked in {where}). If OpenUtau is in an unusual place, set "
              "UTAUCOVER_UTAU_DIRS to its folder.")
    installed = find_phonemizers()
    for s in singers:
        p = by_class(s.default_phonemizer, installed)
        ph = p.tag if p else ((s.default_phonemizer or "").split(".")[-1] + (" (not installed!)" if s.default_phonemizer else "none declared"))
        langs = ",".join(s.languages) or "?"
        print(f"{s.name}\n    id: {s.id}   type: {s.singer_type}   lang: {langs}   default phonemizer: {ph}")


def cmd_models(a):
    from .models import note_engines, whisper_models
    if a.json:
        print(json.dumps({"engines": note_engines(), "whisper": whisper_models()}, indent=2))
        return
    print("Note engines (--notes):")
    for e in note_engines():
        print(f"  {e['name']:8} {'installed' if e['installed'] else 'missing  '}  {e['label']}: {e['about']}")
    print("Whisper models for lyric timing / transcription (--whisper-model):")
    for w in whisper_models():
        print(f"  {w['name']:8} {w['size_gb']:.1f} GB  {'downloaded' if w['downloaded'] else 'downloads on first use'}  {w['about']}")


def cmd_phonemizers(a):
    from .phonemizers import find_phonemizers
    ps = find_phonemizers(refresh=a.refresh)
    if a.language:
        ps = [p for p in ps if p.lang == a.language.lower() or (a.language.lower() == "none" and not p.lang)]
    if a.json:
        print(json.dumps([p.to_dict() for p in ps], indent=2, ensure_ascii=False))
        return
    print(f"{len(ps)} phonemizers installed (from {', '.join(sorted(set(p.source for p in ps)))}). "
          "Use the TAG (quoted) or class name with --phonemizer.\n")
    cur = None
    for p in ps:
        lg = p.language or "any"
        if lg != cur:
            print(f"[{lg}]")
            cur = lg
        extra = f"  dictionary: {p.dictionary}" if p.dictionary else ""
        print(f"  {p.tag:22} {p.short}{extra}")


def open_in_app(ustx: str) -> str:
    """Open a project in UtauV / OpenUTAU. Returns a message for the user.

    These apps read a project path from the command line at launch. On macOS they ignore
    "open document" requests while already running; in that case the file is shown in
    Finder so it can be dragged onto the app window (or opened with File > Open).
    """
    from . import desktop
    from .apps import programs
    for app in programs():
        if app.exe is None:
            continue
        if sys.platform == "darwin":
            proc = app.exe.stem
            running = subprocess.run(["pgrep", "-x", proc], capture_output=True).returncode == 0
            if running:
                desktop.reveal(ustx)
                return (f"{app.name} is already open, so it can't be handed the file directly. The project is "
                        f"selected in Finder: drag it onto the {app.name} window, or use File > Open.")
            subprocess.run(["open", "-a", str(app.exe), "--args", ustx], check=False)
        else:
            desktop.launch(app.exe, ustx, cwd=app.code_dir)
        return f"Launched {app.name} with {Path(ustx).name}."
    desktop.open_path(ustx)
    return f"Opened {Path(ustx).name} with its default app."


def cmd_mcp(a):
    if a.print_config:
        from .selftest import mcp_config
        print(json.dumps(mcp_config(), indent=2))
        return
    from .mcp_server import main as serve
    serve()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="utaucover", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="song -> stems -> charted OpenUTAU project")
    b.add_argument("song")
    b.add_argument("--lyrics", help="lyric sheet (.txt). Omit to transcribe automatically.")
    b.add_argument("--language", help="Whisper language code (en, ja, zh, ko, es, fr, ...). Auto-detected if omitted.")
    b.add_argument("--singer", help="voicebank for the lead (name or folder; see `utaucover singers`)")
    b.add_argument("--harmony-singer", help="voicebank for harmony tracks (default: same as lead)")
    b.add_argument("--phonemizer", help="phonemizer for the lead, e.g. \"EN ARPA+\" (see `utaucover phonemizers`); "
                                        "default: the voicebank's own")
    b.add_argument("--harmony-phonemizer", help="phonemizer for harmony tracks (default: same as lead)")
    b.add_argument("--align", choices=["whisper", "melody"], default="whisper",
                   help="with --lyrics: time the sheet with Whisper alignment (default) or by melody only (no Whisper)")
    b.add_argument("--no-respell", action="store_true",
                   help="don't write explicit phonemes for lines in another language")
    b.add_argument("--quality", choices=["fast", "balanced", "best"], default="best")
    b.add_argument("--backing", choices=["auto", "none", "split"], default="auto",
                   help="backing vocals: auto = find the singer in both separated stems and only write harmony "
                        "tracks for real backing parts (default); none = one singer, everything is the lead; "
                        "split = trust the lead/backing separation as is")
    b.add_argument("--no-harmonies", action="store_true", help="same as --backing none")
    b.add_argument("--notes", choices=["auto", "game", "builtin"], default="auto",
                   help="note engine: auto (default) = UtauV's GAME transcriber when it's installed, else the "
                        "built-in engine with a warning; game = GAME or fail; builtin = UtauCover's pitch tracking")
    b.add_argument("--max-harmonies", type=int, default=4)
    b.add_argument("--stems", action="store_true", help="also split the instrumental into drums/bass/other")
    b.add_argument("--six-stems", action="store_true", help="drums/bass/guitar/piano/other")
    b.add_argument("--dereverb", action="store_true", help="remove reverb from the lead before analysis")
    b.add_argument("--pitch-curve", action="store_true", help="copy the singer's bends/vibrato into the pitd curve")
    b.add_argument("--bpm", type=float, help="override the detected tempo")
    b.add_argument("--snap", type=int, default=32, choices=[0, 8, 16, 32, 64], help="grid (1/N notes), 0 = off")
    b.add_argument("--transpose", type=int, default=0, help="semitones to shift every note")
    b.add_argument("--lyric-mode", choices=["word", "syllable"], help="word (+ extensions) or one syllable per note")
    b.add_argument("--script", choices=["native", "romaji", "pinyin"], help="Japanese romaji / Chinese pinyin output")
    b.add_argument("--no-paren-backing", action="store_true", help="treat (parentheses) in lyric sheet as lead")
    b.add_argument("--whisper-model", default="turbo",
                   help="Whisper size for lyric timing / transcription: turbo (default), large-v3, medium, small "
                        "(see `utaucover models`)")
    b.add_argument("--no-lyrics", action="store_true", help="notes only; every note gets a placeholder syllable")
    b.add_argument("--fill-lyric", help="syllable for sung notes with no words (default: a / あ / 아 / 啊)")
    b.add_argument("--no-legato", action="store_true", help="keep the small consonant gaps between notes")
    b.add_argument("--no-guides", action="store_true", help="don't add muted original-vocal guide tracks")
    b.add_argument("--no-midi", action="store_true")
    b.add_argument("--name", help="project name")
    b.add_argument("--out", help="output folder (default: '<song> UtauCover' next to the song)")
    b.add_argument("--open", action="store_true", help="open the result in UtauV/OpenUTAU")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("separate", help="only split stems")
    s.add_argument("song")
    s.add_argument("--quality", choices=["fast", "balanced", "best"], default="best")
    s.add_argument("--stems", action="store_true")
    s.add_argument("--six-stems", action="store_true")
    s.add_argument("--no-backing", action="store_true")
    s.add_argument("--dereverb", action="store_true")
    s.add_argument("--out")
    s.set_defaults(func=cmd_separate)

    l = sub.add_parser("lyrics", help="transcribe lyrics from a (vocal) audio file")
    l.add_argument("audio")
    l.add_argument("--language")
    l.add_argument("--whisper-model", default="turbo")
    l.add_argument("--out")
    l.set_defaults(func=cmd_lyrics)

    ph = sub.add_parser("phonemizers", help="list phonemizers installed in UtauV / OpenUTAU")
    ph.add_argument("--language", help="filter by language code (en, ja, zh, ...)")
    ph.add_argument("--refresh", action="store_true", help="re-read the apps (after installing a plugin)")
    ph.add_argument("--json", action="store_true")
    ph.set_defaults(func=cmd_phonemizers)

    mo = sub.add_parser("models", help="list note engines and Whisper models, and which are installed")
    mo.add_argument("--json", action="store_true")
    mo.set_defaults(func=cmd_models)

    g = sub.add_parser("singers", help="list installed voicebanks")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=cmd_singers)

    sub.add_parser("gui", help="open the drag-and-drop window").set_defaults(func=lambda a: __import__("utaucover.gui.app", fromlist=["main"]).main())
    m = sub.add_parser("mcp", help="run the MCP server (stdio)")
    m.add_argument("--print-config", action="store_true",
                   help="print the entry to add to Claude Desktop / Claude Code's MCP settings, then exit")
    m.set_defaults(func=cmd_mcp)

    t = sub.add_parser("selftest", help="check that this installation has everything a build needs")
    t.add_argument("--gui", action="store_true", help="also open the window and check it loads")
    t.set_defaults(func=lambda a: sys.exit(__import__("utaucover.selftest", fromlist=["run"]).run(gui=a.gui)))

    v = sub.add_parser("version", help="print the version")
    v.set_defaults(func=lambda a: print(__import__("utaucover").__version__))

    a = ap.parse_args(argv)
    _quiet_libs()
    try:
        a.func(a)
    except KeyboardInterrupt:
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"error: file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
