"""End-to-end: song file -> stems -> notes + lyrics -> .ustx (+ .mid)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import assign as assign_mod
from .harmony import (allocate_lines, detect_poly_notes, drop_isolated, estimate_lines, note_activations,
                      refine_with_polyphony, remove_lead_duplicates, remove_quiet, to_notes)
from .lyrics import (LyricEngine, Word, filter_heard_words, parse_lyrics, read_text, validate_anchors,
                     words_to_units)
from . import __version__
from . import game as game_mod
from .notes import (Note, SegParams, fill_dropouts, fix_octave_jumps, fix_spikes, segment_notes,
                    split_hidden_pitches)
from .phonemizers import PhonemizerInfo, by_class, find_phonemizers, for_language, resolve_phonemizer
from .pronounce import build_report, guess_text_language, lines_from_words
from .paths import safe_name
from .pitch import F0Track, extract_f0, tuning_offset
from .voices import backing_evidence, fix_octaves, main_voice_mask, mask_spans, octave_doubts
from .pitchcurve import pitch_deviation_curve
from .separate import StemSeparator
from .singers import CLASSIC_BY_LANG, DEFAULT_PHONEMIZER, default_renderer, resolve_singer
from .syllables import default_mode, fill_lyric
from .tempo import TempoInfo, estimate_tempo
from .ustx import RESOLUTION, VOICE_COLORS, UNote, VoiceTrack, WaveTrack, write_ustx

Progress = Callable[[str], None]

STAGES = ["stems", "tempo", "pitch", "lyrics", "notes", "harmony", "write"]


@dataclass
class CoverOptions:
    song: str
    out_dir: str | None = None
    lyrics_file: str | None = None
    lyrics_text: str | None = None
    language: str | None = None          # Whisper code: en, ja, zh, ko, es, ... (auto if None)
    lyric_mode: str | None = None        # "word" | "syllable" (auto by language)
    script: str | None = None            # "native" | "romaji" (ja) | "pinyin" (zh)
    backing_in_parens: bool = True       # lyric-sheet text in (parentheses) = backing vocals
    singer: str | None = None            # voicebank name or folder (see `utaucover singers`)
    harmony_singer: str | None = None
    phonemizer: str | None = None        # tag / class / name (see `utaucover phonemizers`); default: voicebank's
    harmony_phonemizer: str | None = None
    align: str = "whisper"               # lyric sheet timing: "whisper" (alignment) or "melody" (no Whisper)
    respell_foreign: bool = True         # lines in another language -> explicit phonemes when possible
    quality: str = "best"                # best (default: cleanest stems) | balanced | fast
    harmonies: bool = True                # False = same as backing="none"
    backing: str = "auto"                 # auto: find the singer in both stems, harmonies only if real
                                          # none: one singer, all vocals are the lead (no karaoke split)
                                          # split: trust the lead/backing stems as separated
    max_harmony_lines: int = 4
    notes_engine: str = "auto"            # auto: GAME when it's installed in UtauV, else built-in (with a warning)
                                          # game: UtauV's GAME note transcriber (error if not installed)
                                          # builtin: UtauCover's pitch tracking + segmentation
    instrument_stems: bool = False
    six_stems: bool = False
    dereverb: bool = False
    pitch_curve: bool = False
    bpm: float | None = None
    snap: int = 32                        # grid: 16, 32, 64 (1/N notes); 0 = no snapping
    transpose: int = 0
    whisper_model: str = "turbo"
    lyrics: bool = True                   # False: notes only, every note gets a placeholder syllable
    fill_lyric: str | None = None         # syllable for sung notes without words (default "a" / "あ" ...)
    legato_gap: float = 0.3               # close gaps shorter than this between notes (0 = keep gaps)
    vowel_on_note: bool = True            # start word notes at the vowel, after voiced consonants
    guide_tracks: bool = True
    midi: bool = True
    project_name: str | None = None


@dataclass
class CoverResult:
    ustx: str
    out_dir: str
    midi: str | None
    language: str
    bpm: float
    lead_in_s: float
    tracks: list[dict] = field(default_factory=list)
    stems: dict = field(default_factory=dict)
    lyrics_file: str | None = None
    pronunciation_report: str | None = None
    pronunciation: str = ""
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # what the analysis decided (shown, not warnings)
    check_file: str | None = None                    # notes worth checking by ear (octave unsure)
    seconds: float = 0.0
    version: str = ""

    def to_dict(self):
        return asdict(self)


def stem_cache_dir(song: str | Path, quality: str = "best") -> Path:
    """Stems are cached per song (by content) and per quality preset, so new output folders,
    rebuilds and switching presets never throw away a separation that already ran."""
    from .paths import APP_SUPPORT, file_fingerprint
    return APP_SUPPORT / "stems" / f"{safe_name(Path(song).stem)[:40]}-{file_fingerprint(song)}" / quality


def publish_stems(cached: dict, dest: Path) -> dict:
    """Put the stems next to the project (hard links when possible: no extra disk space)."""
    import os
    import shutil
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, src in cached.items():
        target = dest / Path(src).name
        try:
            if target.exists():
                # keep it only if it IS the cached file (same hard link). Stems of one song all have the same
                # size, so size/date checks kept stale stems from another quality preset.
                if target.stat().st_ino == Path(src).stat().st_ino:
                    out[k] = target
                    continue
                target.unlink()
            os.link(src, target)
        except OSError:
            shutil.copy2(src, target)
        out[k] = target
    return out


def default_out_dir(song: str | Path) -> Path:
    song = Path(song)
    return song.parent / f"{safe_name(song.stem)} UtauCover"


def build_cover(opts: CoverOptions, progress: Progress | None = None) -> CoverResult:
    t_start = time.time()
    log = progress or (lambda m: None)
    song = Path(opts.song).expanduser().resolve()
    if not song.exists():
        raise FileNotFoundError(song)
    for q in (opts.phonemizer, opts.harmony_phonemizer):  # fail fast, before minutes of separation
        if q:
            _pick_phonemizer(q, None, None, find_phonemizers())
    if opts.align not in ("whisper", "melody"):
        raise ValueError("align must be 'whisper' or 'melody'")
    backing_mode = (opts.backing or "auto") if opts.harmonies else "none"
    if backing_mode not in ("auto", "none", "split"):
        raise ValueError("backing must be 'auto', 'none' or 'split'")
    engine_name = opts.notes_engine or "auto"
    if engine_name not in ("auto", "builtin", "game"):
        raise ValueError("notes engine must be 'auto', 'builtin' or 'game'")
    if engine_name == "game" and game_mod.find_game() is None:
        raise ValueError("GAME isn't installed. " + game_mod.INSTALL_HINT)
    game_missing = False
    if engine_name == "auto":
        engine_name = "game" if game_mod.find_game() is not None else "builtin"
        game_missing = engine_name == "builtin"
    out = Path(opts.out_dir).expanduser() if opts.out_dir else default_out_dir(song)
    out.mkdir(parents=True, exist_ok=True)
    analysis = out / "analysis"
    analysis.mkdir(exist_ok=True)
    warnings: list[str] = []
    info: list[str] = []
    if game_missing:
        log("[notes] GAME isn't installed in UtauV, so notes come from the built-in engine (much less accurate)")
        warnings.append("GAME isn't installed, so the notes came from the built-in engine, which is much less "
                        "accurate. " + game_mod.INSTALL_HINT)

    # ---------------------------------------------------------------- 1. stems
    log("[stage] stems")
    sep = StemSeparator(stem_cache_dir(song, opts.quality), quality=opts.quality, progress=log)
    try:
        cached = sep.separate(song, backing_split=(backing_mode != "none"), dereverb=opts.dereverb,
                              instruments=opts.instrument_stems or opts.six_stems, six_stems=opts.six_stems)
    finally:
        sep.close()
    stems = publish_stems(cached, out / "stems")
    lead_audio = stems.get("lead", stems["vocals"])
    backing_audio = stems.get("backing")
    split_db = _split_balance_db(stems) if "lead" in stems else None
    if split_db is not None and split_db < -10:
        # Karaoke models are trained on human lead singers; synthesized or same-voice stacks often end
        # up entirely in "backing". Then chart the lead from the full vocal stem instead.
        log(f"[stems] the lead/backing split put almost all singing in 'backing' ({split_db:.0f} dB); "
            "charting the lead from the full vocal stem instead")
        lead_audio = stems["vocals"]
        backing_audio = stems["vocals"]
        info.append("lead/backing separation was unreliable for this song; lead charted from the full vocal stem")
    if backing_mode == "none":
        info.append("one singer: every vocal charted as the lead, no harmony tracks")
    analysis_audio = stems.get("lead_dry", lead_audio) if lead_audio is stems.get("lead") else lead_audio

    # ---------------------------------------------------------------- 2. tempo
    log("[stage] tempo")
    tempo = estimate_tempo(stems["instrumental"], bpm_hint=opts.bpm, progress=log, vocal_path=lead_audio)
    lead_in = tempo.lead_in()
    beat_s = tempo.beat_s

    def sec_to_tick(t: float) -> float:
        return (t + lead_in) / beat_s * RESOLUTION

    # ---------------------------------------------------------------- 3. pitch
    log("[stage] pitch")
    f0 = extract_f0(analysis_audio, progress=log)
    f0_all: F0Track | None = None
    f0_back: F0Track | None = None
    moved_spans: list[tuple[float, float]] = []
    f0_lead_stem = f0
    if "lead" in stems and lead_audio == stems["lead"]:
        f0_all = extract_f0(stems["vocals"])
        if backing_mode == "auto":
            # Karaoke models hand lines of a lone singer to "backing" (and leave a faint residue in the
            # lead stem that reads an octave or more too high). Find the singer in both stems.
            f0_back = extract_f0(stems["backing"])
            spans = mask_spans(main_voice_mask(f0, f0_all, f0_back), f0.hop_s)
        else:
            spans = lost_lead_spans(f0, f0_all)
        total = sum(b - a for a, b in spans)
        if total >= 0.5:
            msg = (f"the lead/backing split gave {total:.0f}s of the main singer to 'backing' "
                   f"({len(spans)} passages); those are taken from the full vocal stem")
            log(f"[voices] {msg}")
            info.append(msg)
            hybrid = out / "stems" / "lead_combined.wav"
            splice_audio(stems["lead"], stems["vocals"], spans, hybrid)
            lead_audio = analysis_audio = hybrid
            f0 = extract_f0(hybrid)
            moved_spans = spans
    if f0_all is not None:
        f0, fixed_s = fix_octaves(f0, f0_all)
        if fixed_s >= 0.1:
            log(f"[pitch] corrected {fixed_s:.1f}s of octave slips (kept the octave that fits the melody)")
    f0.save(analysis / "lead_f0.npz")
    tuning = tuning_offset(f0)
    if abs(tuning) > 0.15:
        log(f"[pitch] recording is tuned {tuning * 100:+.0f} cents from A440; compensating")

    # ---------------------------------------------------------------- 4. voices + lyrics
    log("[stage] lyrics")
    singer = resolve_singer(opts.singer)
    if opts.singer and singer is None:
        warnings.append(f"singer '{opts.singer}' not found among installed voicebanks; track left without a singer")
    h_singer = resolve_singer(opts.harmony_singer) if opts.harmony_singer else singer
    installed = find_phonemizers()
    lead_text, backing_text = None, None
    if opts.lyrics_file or opts.lyrics_text:
        raw = opts.lyrics_text if opts.lyrics_text else read_text(opts.lyrics_file)
        lead_text, backing_text = parse_lyrics(raw, opts.backing_in_parens)
    chosen = _pick_phonemizer(opts.phonemizer, singer, None, installed)
    # Language without Whisper when we can: the user's choice, the lyric sheet, the chosen phonemizer
    # or the voicebank. Whisper only listens for the language when there's nothing else to go on.
    lang = opts.language
    if not lang and lead_text:
        lang = guess_text_language(lead_text, default=(chosen.lang if chosen and chosen.lang else "en"))
        log(f"[lyrics] lyric sheet language: {lang}")
    if not lang and chosen is not None and chosen.lang and opts.phonemizer:
        lang = chosen.lang
    use_whisper = opts.lyrics and not (lead_text and opts.align == "melody")
    engine = LyricEngine(opts.whisper_model, progress=log) if use_whisper else None
    if not lang:
        lang = engine.detect_language(lead_audio) if engine else ((singer.languages or ["en"])[0] if singer else "en")
    lead_ph = _pick_phonemizer(opts.phonemizer, singer, lang, installed)
    harm_ph = _pick_phonemizer(opts.harmony_phonemizer or (opts.phonemizer if h_singer is singer else None),
                               h_singer, lang, installed)
    for ph, who in ((lead_ph, "lead"), (harm_ph, "harmony")):
        if ph is not None and ph.lang and ph.lang != lang:
            warnings.append(f"{who} phonemizer {ph.tag} is for '{ph.lang}' but the lyrics are '{lang}'")
    if lead_ph is not None:
        log(f"[voices] lead: {singer.name if singer else '(no singer)'} with {lead_ph.tag} ({lead_ph.short})")
    raw_game: list[Note] | None = None
    if engine_name == "game":
        raw_game = game_mod.transcribe(analysis_audio, _game_mask(f0), f0.hop_s, lang, progress=log)
        for n in raw_game:
            n.pitch -= tuning
        log(f"[notes] GAME found {len(raw_game)} notes")
        raw_game, filled = fill_dropouts(raw_game, f0, tuning)
        if filled:
            log(f"[notes] GAME skipped {filled:.1f}s of singing; filled it from the pitch track")
        (analysis / "game_notes.json").write_text(json.dumps([[n.start, n.end, n.pitch] for n in raw_game]))
        info.append(f"notes transcribed by UtauV's GAME {game_mod.game_version(game_mod.find_game()[1])}")
    mode = opts.lyric_mode or default_mode(lang)
    script = opts.script or ("pinyin" if lang == "zh" else "native")
    fill = opts.fill_lyric or fill_lyric(lang, script)
    if not opts.lyrics:
        words = []
        log("[lyrics] skipped (notes only)")
    elif lead_text and engine is None:
        log("[lyrics] placing the lyric sheet by melody and phrasing (no Whisper)")
        words = melody_anchor_words(lead_text, lang, f0, [Note(n.start, n.end, n.pitch) for n in raw_game]
                                    if raw_game is not None else segment_notes(f0, tuning, []))
    elif lead_text:
        log("[lyrics] Whisper is only used to time your lyric sheet (no transcription)")
        words = engine.align(lead_audio, lead_text, lang, voiced_at=_voiced_fn(f0), phrase_onsets=_phrase_onsets(f0))
    else:
        heard, _ = engine.transcribe(lead_audio, lang)
        words = filter_heard_words(heard, _voiced_fn(f0), lang)
        if len(heard) - len(words):
            log(f"[lyrics] ignored {len(heard) - len(words)} words heard over silence or in the wrong script")
    _save_words(analysis / "lead_words.json", words)
    detected_lyrics = out / "lyrics_detected.txt"
    detected_lyrics.write_text(_words_to_text(words, lang), encoding="utf-8")
    rough = bool(lead_text)  # timings from forced alignment of a sheet are rougher than transcription
    units = words_to_units(words, lang, mode, script, rough_timing=rough)
    if lead_text and engine is None:
        for u in units:
            u.weight = 0.6
    log(f"[lyrics] {len(words)} words -> {len(units)} lyric units ({mode} mode, {lang})")
    if rough and engine is not None:
        nbad = validate_anchors(units, _voiced_fn(f0), words)
        if nbad:
            log(f"[lyrics] {nbad} words were aligned over silence; placing them by melody and phrasing instead")

    # pronunciation: check every word against the dictionaries the phonemizer will really use
    lines, word_line = lines_from_words(words)
    first_units = [u for u in units if u.first_in_word]
    report, by_word = build_report([u.text for u in first_units], [word_line[u.word] if u.word < len(word_line) else -1
                                                                    for u in first_units],
                                   lines, lead_ph, singer.path if singer else None, lang, respell=opts.respell_foreign)
    unit_index = {id(u): k for k, u in enumerate(units)}
    respell = {unit_index[id(first_units[i])]: rep for i, rep in by_word.items()}  # unit index -> lyric
    report_path = out / "pronunciation_report.txt"
    report_path.write_text(report.text(), encoding="utf-8")
    log(f"[pronounce] {report.summary()}")
    if report.unknown:
        sample = ", ".join(list(report.unknown)[:8])
        warnings.append(f"{len(report.unknown)} word(s) aren't in {lead_ph.tag}'s dictionaries and will be guessed by "
                        f"UtauV: {sample}{'…' if len(report.unknown) > 8 else ''} (see pronunciation_report.txt)")
    for m in report.misplaced:
        warnings.append(f"{m} is in the other app's Plugins folder; UtauV/OpenUTAU only reads its own. "
                        "Copy it over to use it (details in pronunciation_report.txt).")
    if report.foreign_lines and not report.respelled:
        warnings.append(f"{len(report.foreign_lines)} lyric line(s) look like another language than '{lang}' "
                        f"(e.g. \"{report.foreign_lines[0][:50]}\"); this phonemizer will pronounce them as '{lang}'.")

    # ---------------------------------------------------------------- 5. lead notes
    log("[stage] notes")
    # confident word starts may split a held note (repeated pitches on different words)
    splits = [u.start for u in units if (u.first_in_word or mode == "syllable") and u.weight >= 0.8]
    if raw_game is not None:
        g = fix_octave_jumps([Note(n.start, n.end, n.pitch) for n in raw_game])
        n0 = len(g)
        g = split_hidden_pitches(g, f0, tuning)
        if len(g) > n0:
            log(f"[notes] added {len(g) - n0} notes where GAME held one pitch over a sung note change")
        lead_notes = fix_spikes(game_mod.split_at_words(g, splits, f0.rms_db, f0.hop_s))
    else:
        lead_notes = segment_notes(f0, tuning, splits)
    # A lyric sheet is complete (every word is sung), so a word may stretch over a short pause; a
    # transcription can miss whole lines, so there words never cross a rest.
    aparams = (assign_mod.AssignParams(sigma=0.8, timing_cap=4.0, band_s=20.0, inner_rest=1.6, max_span_rest_s=1.0)
               if rough else
               assign_mod.AssignParams(sigma=0.25, timing_cap=12.0, band_s=3.0, inner_rest=2.5, max_span_rest_s=0.45))
    lead_notes, stats = assign_mod.assign_lyrics(lead_notes, units, fill, params=aparams)
    if lead_text and engine is not None:
        # lines that are sung but missing from the sheet (official lyrics often skip ad-libs, intros, outros)
        lead_notes, heard_lines = fill_unlisted_lines(lead_notes, units, engine, lead_audio, lang, mode, script,
                                                      fill, out / "analysis", log, voiced_at=_voiced_fn(f0))
        if heard_lines:
            info.append("sung lines missing from the lyric sheet were transcribed: " +
                        "; ".join(f"\"{t}\" at {int(a // 60)}:{a % 60:04.1f}" for a, t in heard_lines[:6]))
    if engine is not None and lead_text and not backing_text:
        engine.release()  # backing vocals reuse the sheet's words: free its memory before the harmony pass
    if mode == "word":
        lead_notes, nfix = assign_mod.ensure_syllable_notes(lead_notes, units, rms_db=f0.rms_db, hop=f0.hop_s)
        if nfix:
            log(f"[notes] split notes so each syllable has its own note in {nfix} words")
    log(f"[notes] lead: {len(lead_notes)} notes, {stats['orphans']} without lyrics, "
        f"{stats['skipped'] - stats['repaired']} lyric units unplaced")
    if stats["skipped"] - stats["repaired"] > max(3, 0.1 * len(units)):
        warnings.append(f"{stats['skipped'] - stats['repaired']} lyric units could not be placed on lead notes")

    # ---------------------------------------------------------------- 6. harmonies
    log("[stage] harmony")
    harmony_lines: list[list[Note]] = []
    harmony_f0: list[F0Track | None] = []
    reuse_units = False
    if backing_mode != "none" and backing_audio is not None and (backing_audio == lead_audio or
                                                                 _has_content(backing_audio, lead_audio)):
        reuse = units if (lead_text and not backing_text) else None
        reuse_units = reuse is not None
        harmony_lines, harmony_f0 = _analyze_backing(backing_audio, opts, engine, lang, mode, script, fill,
                                                     backing_text, analysis, log, warnings, lead_notes,
                                                     units_override=reuse,
                                                     same_as_lead=(backing_audio == lead_audio),
                                                     vocal_ref_db=float(np.percentile(f0.rms_db[f0.voiced], 90))
                                                     if f0.voiced.any() else -20.0,
                                                     gate=(backing_mode == "auto"), backing_f0=f0_back,
                                                     lead_f0=f0, vocals_f0=f0_all, info=info, engine_name=engine_name,
                                                     moved_spans=moved_spans, vocals_path=stems["vocals"])
    elif backing_mode != "none":
        log("[harmony] backing-vocal stem is (nearly) silent; no harmony tracks")
        info.append("no backing vocals found; no harmony tracks")

    # ---------------------------------------------------------------- 7. write project
    log("[stage] write")
    classic_renderer = default_renderer()

    def track_for(name, notes, color, sg, f0track=None, mute=False, phon=None, respell_map=None):
        ph = phon.cls if phon else (sg.phonemizer_for(lang) if sg else CLASSIC_BY_LANG.get(lang, DEFAULT_PHONEMIZER))
        rend = sg.renderer(classic_renderer) if sg else classic_renderer
        notes = [Note(n.start, n.end, n.pitch, n.lyric, n.unit, n.is_unit_start, n.forced) for n in notes]
        if respell_map and phon is not None and phon.arpabet:
            for n in notes:
                if n.is_unit_start and n.unit in respell_map:
                    n.lyric = respell_map[n.unit]
        if opts.vowel_on_note:
            notes = assign_mod.vowel_onsets(notes, lang)
        notes = assign_mod.min_syllable_length(notes)
        if opts.legato_gap > 0:
            notes = assign_mod.legato(notes, opts.legato_gap)
        unotes = quantize(notes, sec_to_tick, opts.snap, opts.transpose)
        pitd = None
        if opts.pitch_curve and f0track is not None:
            pitd = pitch_deviation_curve(f0track, notes, tuning if f0track is f0 else tuning_offset(f0track), sec_to_tick)
        return VoiceTrack(name=name, notes=unotes, singer=sg.id if sg else None, phonemizer=ph,
                          renderer=rend, color=color, mute=mute, pitd=pitd)

    voice = [track_for("Lead", lead_notes, VOICE_COLORS[0], singer, f0, phon=lead_ph, respell_map=respell)]
    names, rels = _harmony_names(harmony_lines, lead_notes)
    order = sorted(range(len(harmony_lines)), key=lambda k: -rels[k])  # highest harmony first
    harmony_lines = [harmony_lines[k] for k in order]
    harmony_f0 = [(harmony_f0[k] if k < len(harmony_f0) else None) for k in order]
    named = [(None, names[k]) for k in order]
    for i, (line, label) in enumerate(zip(harmony_lines, [x[1] for x in named])):
        voice.append(track_for(label, line, VOICE_COLORS[(i + 1) % len(VOICE_COLORS)], h_singer,
                               harmony_f0[i] if i < len(harmony_f0) else None, phon=harm_ph,
                               respell_map=respell if reuse_units else None))

    wave_pos = int(round(lead_in / beat_s * RESOLUTION))
    waves = [WaveTrack("Instrumental", stems["instrumental"], wave_pos, _dur_ms(stems["instrumental"]))]
    if opts.guide_tracks:
        waves.append(WaveTrack("Guide: all vocals", stems["vocals"], wave_pos, _dur_ms(stems["vocals"]),
                               mute=True, color="Light Blue"))
        if lead_audio != stems["vocals"]:
            waves.append(WaveTrack("Guide: lead vocal", lead_audio, wave_pos, _dur_ms(lead_audio),
                                   mute=True, color="Light Blue"))
        if "backing" in stems and backing_audio is not None and backing_audio != stems["vocals"]:
            guide = stems["backing"]
            if moved_spans:  # the singer's lines now live in the lead guide; don't play them twice
                guide = mute_spans(stems["backing"], moved_spans, out / "stems" / "backing_rest.wav")
            waves.append(WaveTrack("Guide: backing vocals", guide, wave_pos, _dur_ms(guide),
                                   mute=True, color="Light Blue"))
    for k in ("drums", "bass", "guitar", "piano", "other"):
        if k in stems:
            waves.append(WaveTrack(f"Stem: {k}", stems[k], wave_pos, _dur_ms(stems[k]), mute=True, color="Green"))

    name = opts.project_name or song.stem
    comment = (f"Generated by UtauCover {__version__} from {song.name}. {tempo.bpm:g} BPM (detected), language {lang}, "
               f"notes by {'GAME' if raw_game is not None else 'the built-in engine'}, Whisper {opts.whisper_model}, "
               f"audio starts at {lead_in:.3f}s so bar lines match.")
    ustx_path = write_ustx(out / f"{safe_name(name)}.ustx", tempo.bpm, voice, waves, name=name, comment=comment)
    midi_path = None
    if opts.midi:
        from .midi import write_midi
        midi_path = write_midi(out / f"{safe_name(name)}.mid", tempo.bpm, voice)

    readings = []  # independent pitch readings of the singer, to flag notes whose octave is in doubt
    if f0_all is not None:
        readings.append(f0_all)
        if f0_lead_stem is not f0:  # the lead stem, except where it had lost the singer
            lead_ok = f0_lead_stem.voiced.copy()
            for a, b in moved_spans:
                lead_ok[f0_lead_stem.frame(a):f0_lead_stem.frame(b) + 1] = False
            readings.append(F0Track(f0_lead_stem.times, f0_lead_stem.hz, f0_lead_stem.midi,
                                    f0_lead_stem.periodicity, f0_lead_stem.rms_db, lead_ok))
    # GAME rarely slips an octave (0% on the hand-charted tests), so with GAME a note is only flagged
    # when every other reading puts it an octave away
    doubts = (octave_doubts(lead_notes, readings, quorum=2) if len(readings) >= 2 else []) if raw_game is not None \
        else octave_doubts(lead_notes, readings)
    check_path, n_check = write_check_list(out / "check_by_ear.txt", doubts,
                                           sec_to_tick, tempo.beats_per_bar, opts.transpose)
    if check_path:
        info.append(f"{n_check} lead note(s) have an uncertain octave; they're listed in check_by_ear.txt")
    tracks_info = [{"name": t.name, "notes": len(t.notes), "singer": t.singer, "phonemizer": t.phonemizer,
                    "phonemizer_tag": (by_class(t.phonemizer, installed).tag if by_class(t.phonemizer, installed) else "")}
                   for t in voice]
    result = CoverResult(ustx=str(ustx_path), out_dir=str(out), midi=str(midi_path) if midi_path else None,
                         language=lang, bpm=tempo.bpm, lead_in_s=lead_in, tracks=tracks_info,
                         stems={k: str(v) for k, v in stems.items()}, lyrics_file=str(detected_lyrics),
                         pronunciation_report=str(report_path), pronunciation=report.summary(),
                         warnings=warnings, notes=info, check_file=str(check_path) if check_path else None,
                         seconds=round(time.time() - t_start, 1), version=__version__)
    (analysis / "result.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[done] wrote {ustx_path.name} ({sum(t['notes'] for t in tracks_info)} notes on {len(voice)} tracks) "
        f"in {result.seconds:.0f}s")
    return result


# ---------------------------------------------------------------- helpers
def quantize(notes: list[Note], sec_to_tick, snap: int, transpose: int = 0) -> list[UNote]:
    grid = (RESOLUTION * 4) // snap if snap else 1
    out: list[UNote] = []
    for n in sorted(notes, key=lambda n: n.start):
        a = int(round(sec_to_tick(n.start) / grid)) * grid
        b = int(round(sec_to_tick(n.end) / grid)) * grid
        if b - a < grid:
            b = a + grid
        if out and a < out[-1].position + out[-1].duration:
            prev = out[-1]
            if a <= prev.position:  # collapsed onto the previous note's start: push this one later
                a = prev.position + grid
                b = max(b, a + grid)
            prev.duration = a - prev.position
        out.append(UNote(a, b - a, int(n.tone) + transpose, n.lyric or "a"))
    # a note squeezed to zero by a collision would be invalid
    return [u for u in out if u.duration > 0]


def _has_content(backing: Path, lead: Path, rel_db: float = -24.0) -> bool:
    import soundfile as sf
    try:
        b, _ = sf.read(str(backing), dtype="float32", always_2d=True)
        l, _ = sf.read(str(lead), dtype="float32", always_2d=True)
    except Exception:
        return True
    rb = np.sqrt(np.mean(b ** 2) + 1e-12)
    rl = np.sqrt(np.mean(l ** 2) + 1e-12)
    return 20 * np.log10(rb / rl) > rel_db


def _analyze_backing(path: Path, opts: CoverOptions, engine: LyricEngine, lang, mode, script, fill,
                     backing_text, analysis: Path, log, warnings, lead_notes: list[Note],
                     same_as_lead: bool = False, vocal_ref_db: float = -20.0,
                     units_override=None, gate: bool = False, backing_f0: F0Track | None = None,
                     lead_f0: F0Track | None = None, vocals_f0: F0Track | None = None,
                     info: list[str] | None = None, engine_name: str = "builtin",
                     moved_spans: list[tuple[float, float]] | None = None,
                     vocals_path: Path | None = None) -> tuple[list[list[Note]], list]:
    poly = detect_poly_notes(path, progress=log)
    before = len(poly)
    poly = remove_lead_duplicates(poly, lead_notes)
    if before - len(poly):
        log(f"[harmony] dropped {before - len(poly)} notes that double the lead or are its overtones")
    before = len(poly)
    poly = remove_quiet(poly, path, vocal_ref_db)
    if before - len(poly):
        log(f"[harmony] dropped {before - len(poly)} faint notes (separation leakage, not singing)")
    game_line: list[Note] | None = None  # GAME's reading of the backing stem's own line (GAME engine)
    if engine_name == "game" and not same_as_lead:
        if backing_f0 is None:
            backing_f0 = extract_f0(path)
        g = game_mod.transcribe(path, _game_mask(backing_f0), backing_f0.hop_s, lang, progress=log)
        tun = tuning_offset(backing_f0)
        for n in g:
            n.pitch -= tun
        n0 = len(g)
        # where the split handed the singer's own line to "backing", what GAME hears there is the lead
        g = [n for n in g if not any(min(n.end, b) - max(n.start, a) > 0.5 * (n.end - n.start)
                                     for a, b in (moved_spans or []))]
        stages = {"game": [[n.start, n.end, n.pitch] for n in g]}
        game_line = remove_quiet(remove_lead_duplicates(g, lead_notes), path, vocal_ref_db)
        stages["not_lead"] = [[n.start, n.end, n.pitch] for n in game_line]
        acts = _poly_evidence(path, vocals_path)
        game_line, moved, dropped = refine_with_polyphony(fix_octave_jumps(game_line), lead_notes, acts)
        stages["polyphony_checked"] = [[n.start, n.end, n.pitch] for n in game_line]
        # confirmed notes: a pair (or one held >= 0.6 s) is enough to count as a phrase now
        game_line = drop_isolated(game_line, min_notes=2, min_dur_s=0.6)
        stages["final"] = [[n.start, n.end, n.pitch] for n in game_line]
        (analysis / "harmony_stages.json").write_text(json.dumps(stages))
        log(f"[harmony] checked against a polyphonic reading: {moved} notes moved to the pitch actually sung, "
            f"{dropped} with no voice there dropped")
        log(f"[harmony] GAME hears {len(game_line)} backing notes of their own "
            f"({n0 - len(game_line)} were the lead itself, its octave/overtones, faint or stray)")
    if gate and lead_f0 is not None and (poly or game_line):
        # Only chart harmonies when the backing stem really sings a part of its own: notes its own
        # pitch confirms, that aren't the singer's unison/octave/overtones, adding up to a few seconds.
        if backing_f0 is None and not same_as_lead:
            backing_f0 = extract_f0(path)
        if game_line is not None:
            own = float(sum(n.end - n.start for n in game_line))
        else:
            own = backing_evidence(poly, None if same_as_lead else backing_f0, lead_notes, vocals_f0)
        sung = float(lead_f0.voiced.sum() * lead_f0.hop_s)
        need = max(6.0, 0.08 * sung)
        if own < need:
            msg = (f"no backing vocals of their own: only {own:.1f}s of the backing stem sings a separate part "
                   f"(at least {need:.0f}s needed), so no harmony tracks")
            log(f"[harmony] {msg}")
            if info is not None:
                info.append(msg)
            return [], []
        log(f"[harmony] the backing stem sings its own part for {own:.0f}s")
    k = min(estimate_lines(poly, opts.max_harmony_lines), opts.max_harmony_lines)
    if game_line and k <= 2:
        k = 1  # one harmony line on top of the lead: GAME's monophonic reading is the cleaner chart
    if k == 0:
        log("[harmony] no harmony notes found")
        return [], []
    # lyrics for the backing stem
    words: list[Word] = []
    try:
        if units_override is not None:
            log("[harmony] using the lyric sheet's words for the backing vocals (no transcription)")
        elif engine is None:
            pass
        elif backing_text:
            words = engine.align(path, backing_text, lang)
        else:
            heard, _ = engine.transcribe(path, lang)
            bf0 = extract_f0(path)
            words = [w for w in filter_heard_words(heard, _voiced_fn(bf0), lang) if w.prob >= 0.3]
    except Exception as e:  # noqa: BLE001 - harmony lyrics are best-effort
        warnings.append(f"backing lyric recognition failed: {e}")
    _save_words(analysis / "backing_words.json", words)
    units = units_override if units_override is not None else words_to_units(words, lang, mode, script)

    if k == 1 and not same_as_lead:
        log("[harmony] backing vocals are a single line; tracking it like the lead")
        f0 = backing_f0 if backing_f0 is not None else extract_f0(path, progress=log)
        f0.save(analysis / "backing_f0.npz")
        tun = tuning_offset(f0)
        splits = [u.start for u in units if u.first_in_word or mode == "syllable"]
        if game_line is not None:
            notes = fix_spikes(game_mod.split_at_words(game_line, splits, f0.rms_db, f0.hop_s))
        else:
            notes = segment_notes(f0, tun, splits, SegParams(min_note_s=0.1))
            notes, _, _ = refine_with_polyphony(notes, lead_notes, _poly_evidence(path, vocals_path))
        n0 = len(notes)
        notes = remove_quiet(remove_lead_duplicates(notes, lead_notes), path, vocal_ref_db)
        notes = drop_isolated(notes, min_notes=2, min_dur_s=0.6) if game_line is not None else drop_isolated(notes)
        if n0 - len(notes):
            log(f"[harmony] dropped {n0 - len(notes)} backing notes that were lead doubles, faint or stray")
        if len(notes) < 4:
            log("[harmony] no real backing vocals left after filtering")
            return [], []
        notes, _ = assign_mod.assign_lyrics(notes, units, fill, params=assign_mod.AssignParams(skip=0.5), repair=False)
        return [notes], [f0]
    log(f"[harmony] {k} stacked harmony lines detected")
    lines = allocate_lines(poly, k)
    out = []
    for line in lines:
        line_notes = drop_isolated(to_notes(line))
        if not line_notes:
            continue
        notes, _ = assign_mod.assign_lyrics(line_notes, units, fill, params=assign_mod.AssignParams(skip=0.5),
                                            repair=False)
        out.append(notes)
    return out, [None] * len(out)


def lost_lead_spans(lead: F0Track, allv: F0Track, min_span_s: float = 2.0, min_voiced_s: float = 1.2,
                    merge_gap_s: float = 0.6) -> list[tuple[float, float]]:
    """Passages where the full vocal stem is singing but the lead stem is silent, long enough to be
    the main vocal (not a reverb tail or a quick backing echo)."""
    n = min(len(lead.voiced), len(allv.voiced))
    m = allv.voiced[:n] & ~lead.voiced[:n]
    hop = allv.hop_s
    # bridge short silences (where nobody sings) inside such a passage
    quiet = ~allv.voiced[:n] & ~lead.voiced[:n]
    gap = int(merge_gap_s / hop)
    i = 0
    while i < n:
        if not m[i] and quiet[i] and i > 0 and m[i - 1]:
            j = i
            while j < n and quiet[j] and not m[j]:
                j += 1
            if j < n and m[j] and j - i <= gap:
                m[i:j] = True
            i = j
        else:
            i += 1
    ref = np.percentile(allv.rms_db[allv.voiced], 90) if allv.voiced.any() else -20
    spans = []
    for v, a, b in _runs_of(m):
        if not v or (b - a) * hop < min_span_s:
            continue
        sung = allv.voiced[a:b]
        if sung.sum() * hop < min_voiced_s:
            continue
        if np.median(allv.rms_db[a:b][sung]) < ref - 15:
            continue
        spans.append((max(0.0, a * hop - 0.1), b * hop + 0.1))
    return spans


def fill_unlisted_lines(notes: list[Note], units, engine, audio: Path, lang: str, mode: str, script: str, fill: str,
                        workdir: Path, log, voiced_at=None, min_span_s: float = 0.8, min_prob: float = 0.5):
    """After placing a lyric sheet, stretches of notes that got no words (placeholder syllables) are sung
    lines the sheet doesn't have (official lyrics often skip intros, ad-libs and outros). Whisper transcribes
    the whole lead once (it needs the song's context: on a short clip it invents words) and its confident
    words inside each stretch go on those notes; a stretch with none (humming, 'ooh's) keeps its placeholder.
    New lyric units are appended to `units`. Returns (notes, [(start, text)])."""
    import re
    notes = sorted(notes, key=lambda n: n.start)
    runs, cur = [], []
    for k, n in enumerate(notes):
        # wordless notes, joined across short pauses (an unlisted line is often sung in bursts)
        if n.unit < 0 and (not cur or (k == cur[-1] + 1 and n.start - notes[cur[-1]].end < 1.5)):
            cur.append(k)
            continue
        if cur:
            runs.append(cur)
        cur = [k] if n.unit < 0 else []
    if cur:
        runs.append(cur)
    runs = [r for r in runs if notes[r[-1]].end - notes[r[0]].start >= min_span_s and len(r) >= 2]
    if not runs:
        return notes, []
    log(f"[lyrics] {len(runs)} sung stretch(es) have no words in the lyric sheet; listening for them")
    try:
        with _quiet_stderr():
            heard, _ = engine.transcribe(audio, lang, quiet=True)
    except Exception:  # noqa: BLE001 - best effort
        return notes, []
    # (no over-silence filter here: the wordless notes already show singing; confidence weeds out guesses)
    heard_lines = []
    for r in reversed(runs):  # back to front, so replacing a stretch doesn't shift the ones still to do
        lo, hi = notes[r[0]].start - 0.6, notes[r[-1]].end + 0.6
        good = []
        for w in heard:
            t = re.sub(r"(.)\1{2,}", r"\1\1", w.text.strip())  # "waaaaah" -> "waah"
            mid = 0.5 * (w.start + w.end)
            if w.prob >= min_prob and any(ch.isalpha() for ch in t) and lo <= mid <= hi:
                good.append(Word(t, w.start, w.end, w.prob, line_start=not good))
        if not good:
            continue
        new_units = words_to_units(good, lang, mode, script)
        base = len(units)
        part = [Note(n.start, n.end, n.pitch) for n in (notes[k] for k in r)]
        part, _ = assign_mod.assign_lyrics(part, new_units, fill, params=assign_mod.AssignParams(
            sigma=0.25, timing_cap=12.0, band_s=3.0, inner_rest=2.5, max_span_rest_s=0.45), repair=False)
        for n in part:
            if n.unit >= 0:
                n.unit += base
        units.extend(new_units)
        notes[r[0]:r[-1] + 1] = part
        heard_lines.append((good[0].start, " ".join(w.text for w in good)))
        log(f"[lyrics] not in the lyric sheet, transcribed: \"{heard_lines[-1][1]}\" "
            f"at {int(good[0].start // 60)}:{good[0].start % 60:04.1f}")
    return notes, sorted(heard_lines)


class _quiet_stderr:
    """Silence a decoder subprocess's chatter on the terminal (fd-level, so child processes too)."""
    def __enter__(self):
        import os
        import sys
        self._saved = None
        try:
            if sys.stderr is not None:
                sys.stderr.flush()
            self._saved = os.dup(2)
        except OSError:  # no stderr at all (a windowed program on Windows): nothing to silence
            return
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 2)

    def __exit__(self, *exc):
        import os
        if self._saved is None:
            return False
        os.dup2(self._saved, 2)
        os.close(self._saved)
        os.close(self._null)
        return False


_POLY_CACHE: dict[str, np.ndarray] = {}


def _poly_evidence(*paths) -> list[np.ndarray]:
    """basic-pitch note activations for the given stems (cached for the build)."""
    out = []
    for p in paths:
        if p is None:
            continue
        key = f"{p}:{Path(p).stat().st_mtime}"
        if key not in _POLY_CACHE:
            if len(_POLY_CACHE) > 4:
                _POLY_CACHE.clear()
            _POLY_CACHE[key] = note_activations(p)
        out.append(_POLY_CACHE[key])
    return out


def _game_mask(track: F0Track, rel_db: float = -35.0) -> np.ndarray:
    """Where GAME should listen: voiced frames plus anything clearly louder than silence (it decides voicing
    itself; this only picks the phrases to hand it)."""
    loud = track.rms_db > (np.percentile(track.rms_db[track.voiced], 90) + rel_db if track.voiced.any() else -60)
    return track.voiced | loud


def mute_spans(path: Path, spans, out_path: Path, fade_s: float = 0.03) -> Path:
    """Copy of an audio file with the given time spans faded to silence."""
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    f = int(fade_s * sr)
    for a, b in spans:
        i, j = int(a * sr), min(len(y), int(b * sr))
        if j - i <= 2 * f:
            continue
        w = np.ones(j - i, dtype=np.float32)
        w[:f] = np.linspace(0, 1, f)
        w[-f:] = np.linspace(1, 0, f)
        y[i:j] *= (1 - w[:, None])
    sf.write(str(out_path), y, sr, format="WAV", subtype="PCM_16")
    return out_path


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def write_check_list(path: Path, doubts, sec_to_tick, beats_per_bar: int = 4,
                     transpose: int = 0) -> tuple[Path | None, int]:
    """List lead notes whose octave is in doubt (another pitch reading of the singer puts them an
    octave away), with bar/beat so they're quick to find in the editor."""
    if not doubts:
        if path.exists():
            path.unlink()
        return None, 0
    lines = []
    for n, other in sorted(doubts, key=lambda x: x[0].start):
        tick = sec_to_tick(n.start)
        bar = int(tick // (RESOLUTION * beats_per_bar)) + 1
        beat = (tick % (RESOLUTION * beats_per_bar)) / RESOLUTION + 1
        mm, ss = divmod(n.start, 60)
        word = n.lyric if n.lyric and n.lyric != "+" else "(held)"
        lines.append(f"bar {bar:3d} beat {beat:4.2f}  ({int(mm)}:{ss:04.1f})  {word:16} charted "
                     f"{note_name(int(round(n.pitch)) + transpose):4} could be {note_name(other + transpose)}")
    head = ("Lead notes whose octave is in doubt: another pitch reading of the singer puts them an octave away.\n"
            "Pitch trackers can hear a low voice's strong overtone as the note, or the reverse. Play these against\n"
            "the original; if one sounds an octave off, move it 12 semitones.\n\n")
    path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    return path, len(lines)


def _runs_of(mask):
    from .pitch import _runs
    return _runs(mask)


def splice_audio(lead_path: Path, vocals_path: Path, spans, out_path: Path, fade_s: float = 0.03):
    import soundfile as sf
    lead, sr = sf.read(str(lead_path), dtype="float32", always_2d=True)
    allv, sr2 = sf.read(str(vocals_path), dtype="float32", always_2d=True)
    n = min(len(lead), len(allv))
    out = lead[:n].copy()
    f = int(fade_s * sr)
    for a, b in spans:
        i, j = int(a * sr), min(n, int(b * sr))
        if j - i <= 2 * f:
            continue
        w = np.ones(j - i, dtype=np.float32)
        w[:f] = np.linspace(0, 1, f)
        w[-f:] = np.linspace(1, 0, f)
        out[i:j] = out[i:j] * (1 - w[:, None]) + allv[i:j] * w[:, None]
    sf.write(str(out_path), out, sr, format="WAV", subtype="PCM_16")
    return out_path


def _pick_phonemizer(query: str | None, singer, lang: str | None, installed: list[PhonemizerInfo]) -> PhonemizerInfo | None:
    """User's choice > the voicebank's declared default > a sensible installed one for the language."""
    if query:
        p = resolve_phonemizer(query, installed)
        if p is None:
            import difflib
            q = query.lower()
            partial = [x.tag for x in installed if q in (x.tag + " " + x.cls + " " + x.name).lower()]
            if len(partial) > 1:
                raise ValueError(f"phonemizer '{query}' matches several: {', '.join(partial[:10])}. Use the exact tag.")
            close = difflib.get_close_matches(query.upper(), [x.tag for x in installed], n=5, cutoff=0.5)
            raise ValueError(f"phonemizer '{query}' is not installed"
                             + (f"; did you mean: {', '.join(close)}" if close else "; see `utaucover phonemizers`"))
        return p
    if singer is not None and singer.default_phonemizer:
        p = by_class(singer.default_phonemizer, installed)
        if p is not None and (lang is None or not p.lang or p.lang == lang or lang in singer.languages or not singer.languages):
            return p
    if lang is None:
        return None
    cands = for_language(lang, installed)
    if not cands:
        return by_class(DEFAULT_PHONEMIZER, installed)
    want_ds = singer is not None and singer.singer_type == "diffsinger"
    preferred = CLASSIC_BY_LANG.get(lang) if not want_ds else None
    for p in cands:
        if preferred and p.cls == preferred:
            return p
    ds = [p for p in cands if p.is_diffsinger]
    other = [p for p in cands if not p.is_diffsinger and not any(k in p.cls for k in ("Enunu", "Voicevox", "Coeiroink", "Vogen"))]
    pool = (ds or cands) if want_ds else (other or cands)
    return pool[0]


def melody_anchor_words(text: str, lang: str, f0: F0Track, notes: list[Note] | None = None) -> list[Word]:
    """Timing guesses for a lyric sheet without Whisper.

    Each lyric line is matched to a run of consecutive notes by dynamic programming: a line's
    syllable count should match its note count (extra notes = melisma), lines prefer to start after
    a rest, and notes no line claims are allowed (vocalises, leakage). Each line's words are then
    spread over its own notes in proportion to their syllables."""
    from .lyrics import lyric_tokens
    from .syllables import count_syllables
    lines = [lyric_tokens(l, lang) for l in text.splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return []
    if notes is None:
        notes = segment_notes(f0, tuning_offset(f0), [])
    notes = sorted(notes, key=lambda n: n.start)
    N, Lc = len(notes), len(lines)
    if N == 0:
        return [Word(t, 0.0, 0.1, 0.0, k == 0) for l in lines for k, t in enumerate(l)]
    gap = [1e9] + [b.start - a.end for a, b in zip(notes, notes[1:])]
    long_cum = np.concatenate([[0], np.cumsum([g >= 1.0 for g in gap[1:]])])
    syl = [sum(count_syllables(t, lang) for t in l) for l in lines]
    INF = 1e18
    dp = np.full((Lc + 1, N + 1), INF)
    bk = np.zeros((Lc + 1, N + 1, 2), dtype=np.int32)  # (kind: 0 skip note, 1 skip line, 2 take), start
    dp[0, 0] = 0.0
    for j in range(1, N + 1):
        dp[0, j] = dp[0, j - 1] + 0.8
    for i in range(1, Lc + 1):
        s_ = syl[i - 1]
        lo_n, hi_n = max(1, s_ // 2), 3 * s_ + 4
        for j in range(0, N + 1):
            best, kind, kk = dp[i - 1, j] + 8.0 + 2.0 * s_, 1, j
            if j > 0:
                c = dp[i, j - 1] + 0.8
                if c < best:
                    best, kind, kk = c, 0, j - 1
                for k in range(max(0, j - hi_n), j - lo_n + 1):
                    prev = dp[i - 1, k]
                    if prev >= INF:
                        continue
                    n = j - k
                    c = prev + (0.25 * (n - s_) if n >= s_ else 1.2 * (s_ - n))
                    c += 0.0 if gap[k] >= 0.3 else 1.5           # lines start after a breath
                    c += 2.0 * (long_cum[j - 1] - long_cum[k])  # and rarely span long rests
                    if c < best:
                        best, kind, kk = c, 2, k
            dp[i, j] = best
            bk[i, j] = (kind, kk)
    runs: dict[int, tuple[int, int]] = {}
    i, j = Lc, N
    while i > 0 or j > 0:
        kind, k = (bk[i, j] if i > 0 else (0, j - 1))
        if i > 0 and kind == 2:
            runs[i - 1] = (int(k), j)
            i, j = i - 1, int(k)
        elif i > 0 and kind == 1:
            i -= 1
        else:
            j -= 1
    words: list[Word] = []
    last_t = notes[0].start
    for li, toks in enumerate(lines):
        run = notes[runs[li][0]:runs[li][1]] if li in runs else []
        ssum = max(1, sum(count_syllables(t, lang) for t in toks))
        acc = 0
        for k, t in enumerate(toks):
            st = run[min(len(run) - 1, int(acc / ssum * len(run)))].start if run else last_t + 0.3
            words.append(Word(t, float(st), float(st) + 0.2, 0.0, k == 0))
            last_t = st
            acc += count_syllables(t, lang)
    return words


def _foreign_lines(words: list[Word], lang: str) -> list[str]:
    """English only for now: lines whose words are mostly not in the CMU pronouncing dictionary."""
    if lang != "en" or not words:
        return []
    try:
        import pronouncing
    except ImportError:
        return []
    out = []
    for line in _words_to_text(words, lang).splitlines():
        toks = [t.strip(".,!?;:\"()").lower() for t in line.split()]
        toks = [t for t in toks if t]
        if len(toks) < 3:
            continue
        unknown = sum(1 for t in toks if not pronouncing.phones_for_word(t))
        if unknown >= 3 and unknown / len(toks) >= 0.35:
            out.append(line)
    return out


def _phrase_onsets(track: F0Track, min_gap_s: float = 2.0) -> list[float]:
    """Times where singing resumes after at least min_gap_s of silence (and the very first onset)."""
    from .pitch import voiced_regions
    out, last_end = [], -1e9
    for a, b in voiced_regions(track):
        if b - a < 10:  # ignore blips < 100 ms
            continue
        t0, t1 = a * track.hop_s, b * track.hop_s
        if t0 - last_end >= min_gap_s:
            out.append(t0)
        last_end = t1
    return out


def _voiced_fn(track: F0Track):
    def frac(a: float, b: float) -> float:
        i, j = track.frame(max(0.0, a)), track.frame(b) + 1
        seg = track.voiced[i:j]
        return float(seg.mean()) if len(seg) else 0.0
    return frac


def _split_balance_db(stems: dict) -> float | None:
    """Loudness of the 'lead' stem relative to 'backing' over the whole song, in dB."""
    import soundfile as sf
    try:
        l, _ = sf.read(str(stems["lead"]), dtype="float32", always_2d=True)
        b, _ = sf.read(str(stems["backing"]), dtype="float32", always_2d=True)
    except Exception:
        return None
    rl = np.sqrt(np.mean(l ** 2) + 1e-12)
    rb = np.sqrt(np.mean(b ** 2) + 1e-12)
    return float(20 * np.log10(rl / rb))


def _harmony_names(lines: list[list[Note]], lead: list[Note]) -> tuple[list[str], list[float]]:
    """Name harmony tracks by where they sit relative to the lead ('Harmony high', 'Harmony low 2'...).
    Returns (names, median semitones above the lead)."""
    if not lines:
        return [], []

    def rel(line):
        diffs = []
        for n in line:
            mid = 0.5 * (n.start + n.end)
            over = [m for m in lead if m.start <= mid < m.end]
            if over:
                diffs.append(n.pitch - over[0].pitch)
        if diffs:
            return float(np.median(diffs))
        return float(np.median([n.pitch for n in line]) - np.median([m.pitch for m in lead] or [60]))
    rels = [rel(l) for l in lines]
    if len(lines) == 1:
        return ["Harmony"], rels
    highs = sorted([i for i, r in enumerate(rels) if r >= 0], key=lambda i: rels[i])
    lows = sorted([i for i, r in enumerate(rels) if r < 0], key=lambda i: -rels[i])
    names = [""] * len(lines)
    for group, word in ((highs, "high"), (lows, "low")):
        for k, i in enumerate(group):
            names[i] = f"Harmony {word}" + (f" {k + 1}" if len(group) > 1 else "")
    return names, rels


def _dur_ms(path: Path) -> float:
    import soundfile as sf
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate * 1000.0
    except Exception:
        return 0.0


def _save_words(path: Path, words: list[Word]):
    path.write_text(json.dumps([asdict(w) for w in words], ensure_ascii=False, indent=1), encoding="utf-8")


def _words_to_text(words: list[Word], lang: str) -> str:
    if not words:
        return ""
    joiner = "" if lang in ("ja", "zh", "yue", "th") else " "
    lines, cur = [], [words[0].text]
    for a, b in zip(words, words[1:]):
        if b.start - a.end > 0.6:
            lines.append(joiner.join(cur))
            cur = []
        cur.append(b.text)
    lines.append(joiner.join(cur))
    return "\n".join(lines) + "\n"
