"""MCP server: lets Claude (or any MCP client) build OpenUTAU cover projects.

Run with `utaucover-mcp` (stdio). Builds take minutes, so they run as background jobs:
start_cover_build -> get_build_status (long-polls) -> result with the .ustx path.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

mcp = _Server("utaucover", instructions=(
    "Tools for turning a song into an OpenUTAU / UtauV cover project: stem separation, vocal pitch "
    "charting into lead + harmony tracks, and lyrics (from a lyric sheet or auto-transcribed, any language). "
    "Use list_singers to find voicebank names. Builds are background jobs: call start_cover_build, then "
    "get_build_status until state is 'done'. Re-running a build for the same song reuses cached stems, so "
    "fixing lyrics and rebuilding is fast."))

_JOB_LOCK = threading.Lock()   # one heavy job at a time (shared GPU)


@dataclass
class Job:
    id: str
    kind: str
    args: dict
    state: str = "queued"      # queued | running | done | error
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    stage: str = ""


JOBS: dict[str, Job] = {}


def _run_job(job: Job, fn):
    def progress(msg: str):
        if msg.startswith("[stage]"):
            job.stage = msg.split(" ", 1)[1]
        else:
            job.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
    with _JOB_LOCK:
        job.state = "running"
        try:
            job.result = fn(progress)
            job.state = "done"
        except Exception as e:  # noqa: BLE001
            job.error = f"{type(e).__name__}: {e}"
            job.log.append(traceback.format_exc(limit=4))
            job.state = "error"
        finally:
            job.finished = time.time()


def _start(kind: str, args: dict, fn) -> Job:
    job = Job(uuid.uuid4().hex[:8], kind, args)
    JOBS[job.id] = job
    threading.Thread(target=_run_job, args=(job, fn), daemon=True).start()
    return job


def _status(job: Job, tail: int = 15) -> dict:
    return {"job_id": job.id, "kind": job.kind, "state": job.state, "stage": job.stage,
            "elapsed_s": round((job.finished or time.time()) - job.started), "log_tail": job.log[-tail:],
            "result": job.result, "error": job.error}


@mcp.tool()
def list_singers() -> list[dict]:
    """List voicebanks installed for UtauV / OpenUTAU (name, id to use, type, language guess, phonemizer)."""
    from .singers import find_singers
    return [s.to_dict() for s in find_singers()]


@mcp.tool()
def list_phonemizers(language: str | None = None, refresh: bool = False) -> list[dict]:
    """Phonemizers installed in UtauV / OpenUTAU (read from the apps' own plugin metadata).
    Each has a tag (e.g. 'EN C+V'), class name, language and the dictionary file it reads.
    language: optional filter (en, ja, zh, ko, es, ...). refresh: re-read after installing a plugin."""
    from .phonemizers import find_phonemizers
    ps = find_phonemizers(refresh=refresh)
    if language:
        ps = [p for p in ps if p.lang == language.lower()]
    return [p.to_dict() for p in ps]


@mcp.tool()
def list_models() -> dict:
    """Note engines (builtin, game) and Whisper model sizes, with which are installed/downloaded."""
    from .models import note_engines, whisper_models
    return {"engines": note_engines(), "whisper": whisper_models()}


@mcp.tool()
def start_cover_build(song_path: str, lyrics_path: str | None = None, lyrics_text: str | None = None,
                      language: str | None = None, singer: str | None = None, harmony_singer: str | None = None,
                      phonemizer: str | None = None, harmony_phonemizer: str | None = None,
                      align: str = "whisper", respell_foreign: bool = True,
                      quality: str = "best", harmonies: bool = True, backing: str = "auto",
                      notes_engine: str = "auto", whisper_model: str = "turbo",
                      max_harmony_lines: int = 4,
                      pitch_curve: bool = False, instrument_stems: bool = False, dereverb: bool = False,
                      bpm: float | None = None, transpose: int = 0, snap: int = 32,
                      lyric_mode: str | None = None, script: str | None = None,
                      out_dir: str | None = None, project_name: str | None = None) -> dict:
    """Start building an OpenUTAU/UtauV project (.ustx) from a song file. Returns a job id.

    song_path: audio file (mp3/wav/flac/m4a/...). lyrics_path or lyrics_text: optional lyric sheet; text in
    (parentheses) is treated as backing vocals. Without lyrics, Whisper transcribes them.
    language: Whisper code (en, ja, zh, ko, es, fr, de, pt, it, ru, ...), auto-detected if omitted.
    singer / harmony_singer: voicebank name or id from list_singers.
    phonemizer / harmony_phonemizer: tag or class from list_phonemizers (default: the voicebank's own).
    align: with lyrics, 'whisper' (alignment, most accurate) or 'melody' (no Whisper; placed by melody/phrasing).
    respell_foreign: write explicit phonemes for lines in another language (e.g. Spanish on an English bank).
    quality: best (default; cleanest instrumental, ~11 min the first time for a 3-min song) | balanced (~5 min,
    a faint vocal ghost in the instrumental) | fast (~2-3 min); stems are cached, so only the first build pays.
    backing: 'auto' (default) finds the main singer in both separated stems and writes harmony tracks only
    for real backing parts; 'none' = one singer, every vocal is the lead; 'split' = trust the separation.
    harmonies=False is the same as backing='none'.
    notes_engine: 'auto' (default: UtauV's GAME transcriber when installed, else the built-in engine plus a
    warning in the result), 'game' (error if not installed) or 'builtin'. See list_models.
    whisper_model: turbo (default), large-v3, medium, small.
    pitch_curve: copy the original singer's slides/vibrato into the pitd curve.
    lyric_mode: 'word' (words with + extensions, default for alphabetic languages) or 'syllable'.
    script: 'romaji' for Japanese romaji, 'pinyin' for Chinese pinyin (default for zh), else native.
    """
    from .pipeline import CoverOptions, build_cover, default_out_dir
    p = Path(song_path).expanduser()
    if not p.exists():
        return {"error": f"song not found: {p}"}
    from .pipeline import _pick_phonemizer
    from .phonemizers import find_phonemizers
    try:
        for q in (phonemizer, harmony_phonemizer):
            if q:
                _pick_phonemizer(q, None, None, find_phonemizers())
    except ValueError as e:
        return {"error": str(e)}
    if backing not in ("auto", "none", "split"):
        return {"error": "backing must be 'auto', 'none' or 'split'"}
    if notes_engine not in ("auto", "builtin", "game"):
        return {"error": "notes_engine must be 'auto', 'builtin' or 'game'"}
    if notes_engine == "game":
        from .game import find_game
        if find_game() is None:
            from .game import INSTALL_HINT
            return {"error": "GAME isn't installed. " + INSTALL_HINT}
    opts = CoverOptions(song=str(p), out_dir=out_dir, lyrics_file=lyrics_path, lyrics_text=lyrics_text,
                        language=language, singer=singer, harmony_singer=harmony_singer, quality=quality,
                        phonemizer=phonemizer, harmony_phonemizer=harmony_phonemizer, align=align,
                        respell_foreign=respell_foreign,
                        harmonies=harmonies, backing=backing, notes_engine=notes_engine, whisper_model=whisper_model,
                        max_harmony_lines=max_harmony_lines, pitch_curve=pitch_curve,
                        instrument_stems=instrument_stems, dereverb=dereverb, bpm=bpm, transpose=transpose,
                        snap=snap, lyric_mode=lyric_mode, script=script, project_name=project_name)
    job = _start("cover", {"song": str(p)}, lambda prog: build_cover(opts, prog).to_dict())
    return {"job_id": job.id, "out_dir": str(Path(out_dir) if out_dir else default_out_dir(p)),
            "hint": "poll get_build_status(job_id, wait_seconds=60)"}


@mcp.tool()
def start_stem_separation(song_path: str, quality: str = "best", backing_split: bool = True,
                          instrument_stems: bool = False, six_stems: bool = False, out_dir: str | None = None) -> dict:
    """Only split a song into stems (vocals, instrumental, lead, backing, optional drums/bass/other). Returns a job id."""
    from .pipeline import default_out_dir, publish_stems, stem_cache_dir
    from .separate import StemSeparator
    p = Path(song_path).expanduser()
    if not p.exists():
        return {"error": f"song not found: {p}"}
    dest = Path(out_dir) if out_dir else default_out_dir(p) / "stems"

    def work(prog):
        sep = StemSeparator(stem_cache_dir(p, quality), quality=quality, progress=prog)
        try:
            stems = sep.separate(p, backing_split=backing_split, instruments=instrument_stems or six_stems,
                                 six_stems=six_stems)
        finally:
            sep.close()
        return {k: str(v) for k, v in publish_stems(stems, dest).items()}
    job = _start("stems", {"song": str(p)}, work)
    return {"job_id": job.id, "out_dir": str(dest)}


@mcp.tool()
def get_build_status(job_id: str, wait_seconds: int = 30) -> dict:
    """Status of a build/separation job. Waits up to wait_seconds (max 120) for it to finish before answering."""
    job = JOBS.get(job_id)
    if not job:
        return {"error": f"unknown job {job_id}", "known_jobs": list(JOBS)}
    deadline = time.time() + max(0, min(wait_seconds, 120))
    while job.state in ("queued", "running") and time.time() < deadline:
        time.sleep(1)
    return _status(job)


@mcp.tool()
def list_jobs() -> list[dict]:
    """All jobs started in this server session."""
    return [{"job_id": j.id, "kind": j.kind, "state": j.state, "args": j.args} for j in JOBS.values()]


@mcp.tool()
def transcribe_lyrics(audio_path: str, language: str | None = None, whisper_model: str = "turbo") -> dict:
    """Transcribe lyrics from an audio file (ideally an isolated vocal stem). Returns text with one line per phrase."""
    from .lyrics import LyricEngine
    from .pipeline import _words_to_text
    with _JOB_LOCK:
        eng = LyricEngine(whisper_model)
        words, lang = eng.transcribe(Path(audio_path).expanduser(), language)
    return {"language": lang, "lyrics": _words_to_text(words, lang),
            "words": [{"text": w.text, "start": round(w.start, 2), "end": round(w.end, 2)} for w in words[:2000]]}


@mcp.tool()
def read_project(ustx_path: str, max_notes_per_track: int = 200) -> dict:
    """Summarise a .ustx project: tempo, tracks, singer/phonemizer, and each track's notes (time, tone, lyric)."""
    import yaml
    d = yaml.safe_load(Path(ustx_path).expanduser().read_text(encoding="utf-8-sig"))
    bpm = float((d.get("tempos") or [{"bpm": d.get("bpm", 120)}])[0]["bpm"])
    tick_s = 60 / bpm / 480
    tracks = [{"name": t.get("track_name"), "singer": t.get("singer"), "phonemizer": t.get("phonemizer"),
               "notes": []} for t in d.get("tracks", [])]
    for part in d.get("voice_parts") or []:
        for n in part.get("notes") or []:
            tick = part["position"] + n["position"]
            tr = tracks[part["track_no"]]
            if len(tr["notes"]) < max_notes_per_track:
                tr["notes"].append({"t": round(tick * tick_s, 2), "dur": round(n["duration"] * tick_s, 2),
                                    "tone": n["tone"], "lyric": n["lyric"]})
    waves = [{"name": w.get("name"), "track_no": w.get("track_no"), "path": w.get("relative_path")}
             for w in d.get("wave_parts") or []]
    return {"bpm": bpm, "tracks": tracks, "wave_parts": waves}


@mcp.tool()
def open_in_utau(ustx_path: str) -> str:
    """Open a .ustx project in UtauV (or OpenUTAU) on this Mac."""
    from .cli import open_in_app
    p = Path(ustx_path).expanduser()
    if not p.exists():
        return f"not found: {p}"
    return open_in_app(str(p))


def main():
    import io

    import anyio
    from mcp.server.stdio import stdio_server

    # Keep the protocol stream clean: anything libraries print goes to stderr.
    real_out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    os.environ.setdefault("TQDM_DISABLE", "1")
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)

    low = getattr(mcp, "_lowlevel_server", None) or getattr(mcp, "_mcp_server")

    async def run():
        out = anyio.wrap_file(io.TextIOWrapper(real_out, encoding="utf-8", write_through=True))
        async with stdio_server(stdout=out) as (r, w):
            await low.run(r, w, low.create_initialization_options())

    anyio.run(run)


if __name__ == "__main__":
    main()
