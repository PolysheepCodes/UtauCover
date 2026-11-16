"""Lyrics: automatic transcription or alignment of a lyric text file, with word timings.

Uses stable-ts on top of OpenAI Whisper, so any of Whisper's ~99 languages works.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .syllables import (CJK_RE, KANA_RE, SYLLABLE_LANGS, clean_word, count_syllables, default_mode,
                        split_units, to_hiragana)


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0
    line_start: bool = False


@dataclass
class Unit:
    """A piece of lyric that starts a note group: a word (word mode) or a syllable/mora."""
    text: str
    start: float
    end: float
    word: int            # index into the word list
    first_in_word: bool
    syllables: int = 1   # syllables this unit should span (word mode: the word's syllable count)
    line_start: bool = False
    weight: float = 1.0  # trust in the timing (low for shaky alignments)


# ---------------------------------------------------------------- lyric files
SECTION_RE = re.compile(r"^\s*[\[{【].*[\]}】]\s*$")
LABEL_RE = re.compile(r"^\s*(verse|chorus|pre-?chorus|bridge|intro|outro|hook|refrain|interlude|break)\b[^a-z]*$", re.I)
REPEAT_RE = re.compile(r"[\(\[]?\s*[x×]\s*(\d+)\s*[\)\]]?\s*$", re.I)
PAREN_RE = re.compile(r"\(([^)]*)\)|（([^）]*)）")


def read_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp932", "gb18030", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_lyrics(text: str, backing_in_parens: bool = True) -> tuple[str, str]:
    """Split a lyric sheet into (lead_text, backing_text).

    Drops section headers like [Chorus], expands "x2" repeats, and (optionally) treats
    text in parentheses as backing vocals.
    """
    lead_lines, backing_lines = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or SECTION_RE.match(line) or LABEL_RE.match(line):
            continue
        reps = 1
        m = REPEAT_RE.search(line)
        if m and len(line) > len(m.group(0)):
            reps = min(int(m.group(1)), 8)
            line = line[:m.start()].strip()
        bk = []
        if backing_in_parens:
            bk = [a or b for a, b in PAREN_RE.findall(line)]
            line = PAREN_RE.sub(" ", line)
        line = re.sub(r"\[[^\]]*\]", " ", line)  # inline [notes] / phoneme hints
        line = re.sub(r"\s+", " ", line).strip()
        for _ in range(reps):
            if line:
                lead_lines.append(line)
            if bk:
                backing_lines.append(" ".join(bk))
    return "\n".join(lead_lines), "\n".join(backing_lines)


def lyric_tokens(text: str, lang: str) -> list[str]:
    """Split lyric text into the 'words' we time individually.

    Japanese: moras of the contextual hiragana reading of each line. Chinese: characters.
    Everything else: whitespace-separated words.
    """
    toks: list[str] = []
    for line in text.splitlines():
        if lang == "ja" or (KANA_RE.search(line) and lang not in ("zh", "yue")):
            hira = to_hiragana(line)
            cur = []
            for ch in hira:
                if ch in "ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ" and cur:
                    cur[-1] += ch
                elif KANA_RE.match(ch) or ch.isalnum():
                    cur.append(ch)
            toks.extend(cur)
        elif lang in ("zh", "yue"):
            toks.extend(ch for ch in line if CJK_RE.match(ch) or ch.isalnum())
        else:
            toks.extend(w for w in re.split(r"\s+", line) if clean_word(w, lang))
    return toks


# ---------------------------------------------------------------- whisper
def load_16k(audio) -> np.ndarray:
    """Audio as Whisper wants it (16 kHz mono float32), read without spawning ffmpeg: a streaming ffmpeg
    reader killed half-way spills 'Broken pipe' errors onto the terminal."""
    if isinstance(audio, np.ndarray):
        return audio
    import soundfile as sf
    try:
        y, sr = sf.read(str(audio), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        return np.ascontiguousarray(y, dtype=np.float32)
    except Exception:  # formats soundfile can't read (e.g. some m4a): Whisper's own ffmpeg loader
        import whisper
        return whisper.load_audio(str(audio))


class LyricEngine:
    def __init__(self, model: str = "turbo", device: str | None = None, progress=None):
        self.model_name = model
        self.device = device or "cpu"
        self.log = progress or (lambda m: None)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import os

            import stable_whisper
            import whisper
            from .paths import whisper_dir
            root = whisper_dir()
            url = whisper._MODELS.get(self.model_name, "")
            cached = os.path.exists(os.path.join(root, os.path.basename(url)))
            if not cached and url:
                from .download import fetch_whisper
                fetch_whisper(self.model_name, root, self.log)
            self.log(f"[lyrics] loading Whisper '{self.model_name}'")
            self._model = stable_whisper.load_model(self.model_name, device=self.device, download_root=str(root))
        return self._model

    def release(self):
        """Free the Whisper model (a few GB of RAM) once it's no longer needed; it reloads if used again."""
        if self._model is not None:
            self._model = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    def detect_language(self, audio: str | Path) -> str:
        import whisper
        audio_arr = load_16k(audio)
        # use the loudest 30 s window so an instrumental intro doesn't fool detection
        import numpy as np
        sr = 16000
        win = 30 * sr
        if len(audio_arr) > win:
            energy = np.convolve(audio_arr ** 2, np.ones(sr), "valid")[::sr]
            starts = np.arange(len(energy))
            best, best_e = 0, -1
            for s in starts:
                e = energy[s:s + 30].sum()
                if e > best_e:
                    best, best_e = s, e
            audio_arr = audio_arr[best * sr: best * sr + win]
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio_arr), n_mels=self.model.dims.n_mels).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        lang = max(probs, key=probs.get)
        self.log(f"[lyrics] detected language: {lang} ({probs[lang]:.0%})")
        return lang

    def transcribe(self, audio, language: str | None = None, prompt: str | None = None,
                   quiet: bool = False) -> tuple[list[Word], str]:
        """audio: a file path, or a 16 kHz mono float32 numpy array."""
        lang = language or self.detect_language(audio)
        if not quiet:
            self.log(f"[lyrics] transcribing ({lang})")
        res = self.model.transcribe(load_16k(audio), language=lang, word_timestamps=True, regroup=True,
                                    condition_on_previous_text=False, initial_prompt=prompt,
                                    no_speech_threshold=0.7, verbose=None)
        words = _result_words(res)
        words = _drop_hallucinations(words)
        return words, lang

    def _align_once(self, audio, text: str, language: str, offset: float = 0.0) -> list[Word]:
        res = self.model.align(audio, text, language=language, original_split=True, verbose=None)
        if res is None:
            return []
        words = _result_words(res)
        for w in words:
            w.start += offset
            w.end += offset
        return words

    def align(self, audio: str | Path, text: str, language: str, voiced_at=None,
              phrase_onsets: list[float] | None = None) -> list[Word]:
        """Force-align a lyric sheet.

        Whisper's aligner tends to lose sync across long instrumental breaks: it then crams the
        remaining words into zero-length slots. When that happens we re-align the rest of the
        sheet starting where singing resumes (phrase_onsets), repeating if needed.
        """
        self.log(f"[lyrics] aligning lyric text ({language})")
        expected = len(lyric_tokens(text, language))
        words: list[Word] = []
        audio = load_16k(audio)
        try:
            words = self._align_once(audio, text, language)
        except Exception as e:  # stable-ts raises when the text cannot be aligned at all
            self.log(f"[lyrics] direct alignment failed ({e})")
        if words and len(words) >= 0.8 * expected:
            words = self._repair_collapses(audio, words, language, voiced_at, phrase_onsets or [])
            if not _alignment_ok(words):
                self.log("[lyrics] alignment timings are rough for this voice; leaning on the melody's phrasing")
            return words
        self.log("[lyrics] alignment incomplete; matching the lyric sheet against a transcription instead")
        heard, _ = self.transcribe(audio, language, prompt=text[:200])
        return match_text_to_words(text, heard, language)

    def _repair_collapses(self, audio, words: list[Word], language: str, voiced_at, onsets: list[float],
                          max_rounds: int = 4) -> list[Word]:
        audio_arr = None
        for _ in range(max_rounds):
            i0 = _collapse_start(words, voiced_at)
            if i0 is None:
                break
            t_prev = words[i0 - 1].end if i0 > 0 else 0.0
            # Restart where singing resumes, but only skip ahead if what lies in between really is a
            # break (little singing). Otherwise restart right after the last good word.
            later = [o for o in onsets if o >= t_prev - 0.3]
            start = t_prev
            if later and voiced_at is not None and later[0] - t_prev > 1.0:
                if voiced_at(t_prev + 0.3, later[0] - 0.3) < 0.15:
                    start = later[0] - 0.3
            elif later and later[0] - t_prev <= 1.0:
                start = later[0] - 0.3
            start = max(0.0, start)
            rest = words[i0:]
            sub_text = ""
            for k, w in enumerate(rest):
                sub_text += ("\n" if (w.line_start and k) else (" " if k else "")) + w.text
            self.log(f"[lyrics] alignment lost sync at word {i0 + 1} ('{rest[0].text}'); "
                     f"re-aligning the remaining {len(rest)} words from {start:.1f}s")
            if audio_arr is None:
                audio_arr = load_16k(audio)
            try:
                sub = self._align_once(audio_arr[int(start * 16000):], sub_text, language, offset=start)
            except Exception:
                break
            if len(sub) < 0.8 * len(rest) or (sub and _collapse_start(sub, voiced_at) == 0):
                break
            words = words[:i0] + sub
        return words

def _result_words(res) -> list[Word]:
    words = []
    for seg in res.segments:
        first = True
        for w in seg.words:
            txt = w.word.strip()
            if not txt:
                continue
            words.append(Word(txt, float(w.start), float(w.end), float(getattr(w, "probability", 1.0) or 1.0),
                              line_start=first))
            first = False
    return words


def _drop_hallucinations(words: list[Word]) -> list[Word]:
    """Remove classic Whisper filler ("Thank you for watching", subtitles credits) and
    long runs of the same token repeated with near-zero duration."""
    bad = re.compile(r"(subtitles? by|amara\.org|thanks? for watching|字幕|ご視聴ありがとう)", re.I)
    joined = " ".join(w.text for w in words)
    if bad.search(joined):
        words = [w for w in words if not bad.search(w.text)]
    out = []
    for w in words:
        if w.end - w.start < 0.005 and out and out[-1].text == w.text:
            continue
        out.append(w)
    return out


LATIN_LANGS = {"en", "es", "fr", "de", "pt", "it", "nl", "sv", "da", "no", "nb", "fi", "pl", "cs", "sk", "hu",
               "ro", "hr", "sl", "tr", "id", "ms", "tl", "vi", "ca", "gl", "eu", "et", "lv", "lt", "la", "sq", "sw"}


def _script_ok(text: str, lang: str) -> bool:
    """Reject words in a different writing system than the language (e.g. Cyrillic in English)."""
    if "\ufffd" in text:  # Whisper emitted a broken multi-byte token
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    if lang in LATIN_LANGS:
        return all(unicodedata.name(ch, "").startswith("LATIN") for ch in letters)
    if lang == "ja":
        return all(KANA_RE.match(ch) or CJK_RE.match(ch) or ch == "ー" or unicodedata.name(ch, "").startswith("LATIN") for ch in letters)
    return True


def filter_heard_words(words: list[Word], voiced_at, lang: str, min_voiced: float = 0.2) -> list[Word]:
    """Drop Whisper output that nobody sang: words over silence (its classic 'Thank you.' /
    'Subtitles by' hallucinations) and words in the wrong writing system.

    voiced_at(start, end) -> fraction of frames in that span where the vocal is voiced.
    """
    out = []
    for w in words:
        if not _script_ok(w.text, lang):
            continue
        span_end = max(w.end, w.start + 0.1)
        if voiced_at(w.start - 0.05, span_end + 0.05) < min_voiced:
            continue
        out.append(w)
    return out


def _collapse_start(words: list[Word], voiced_at=None, run_len: int = 3) -> int | None:
    """Index where alignment collapsed: the start of a run of words crammed into ~zero length (under 0.11 s
    each; three sung words in a third of a second means the aligner squeezed a repeated line into the wrong
    place) or lying over silence."""
    run = 0
    for i, w in enumerate(words):
        bad = (w.end - w.start) < 0.11
        if not bad and voiced_at is not None:
            bad = voiced_at(w.start - 0.1, max(w.end, w.start + 0.2) + 0.1) < 0.15
        run = run + 1 if bad else 0
        if run >= run_len:
            return i - run_len + 1
    return None


def validate_anchors(units: list["Unit"], voiced_at, words: list[Word] | None = None,
                     min_prob: float = 0.05, jump_s: float = 4.0) -> int:
    """Distrust unit timings that land where nobody is singing (an alignment that lost sync), that Whisper
    itself gave almost no confidence, or that jump far ahead past singing (an aligner that ran out of
    sheet and stretched the last words over later lines). Their weight drops to 0 so note assignment relies
    on order, syllables and phrasing instead. Returns how many were distrusted."""
    bad = 0
    drift = False
    for k, u in enumerate(units):
        w = words[u.word] if words is not None and 0 <= u.word < len(words) else None
        if k > 0 and u.start - units[k - 1].end > jump_s and \
                voiced_at(units[k - 1].end + 0.3, u.start - 0.3) >= 0.3:
            drift = True  # skipped over singing: this word and the shaky ones after it are misplaced
        elif drift and (w is None or w.prob >= 0.3):
            drift = False
        if (u.end - u.start) < 0.03 or voiced_at(u.start - 0.15, max(u.end, u.start + 0.2) + 0.15) < 0.15 \
                or (w is not None and w.prob < min_prob) or drift:
            u.weight = 0.0
            bad += 1
    _estimate_from_neighbours(units)
    return bad


def _estimate_from_neighbours(units: list["Unit"], per_unit_s: float = 0.3, max_units: int = 4,
                              max_span_s: float = 6.0, weight: float = 0.35):
    """A word the aligner couldn't time (zero length, or over silence) still sits between words it did time.
    Give it a rough time from them, so it stays next to its neighbours: otherwise the first word of a sheet
    can be placed on an intro the sheet doesn't cover and stretched over it."""
    trusted = [i for i, u in enumerate(units) if u.weight > 0]
    if not trusted:
        return
    import bisect
    for i, u in enumerate(units):
        if u.weight > 0:
            continue
        k = bisect.bisect_left(trusted, i)
        prv = trusted[k - 1] if k > 0 else None
        nxt = trusted[k] if k < len(trusted) else None
        if prv is not None and nxt is not None and nxt - prv <= max_units + 1 and \
                units[nxt].start - units[prv].end <= max_span_s:
            frac = (i - prv) / (nxt - prv)
            t = units[prv].end + frac * (units[nxt].start - units[prv].end)
        elif nxt is not None and (prv is None or nxt - i <= 2) and nxt - i <= 2:
            t = units[nxt].start - per_unit_s * (nxt - i)
        elif prv is not None and (nxt is None or i - prv <= 2) and i - prv <= 2:
            t = units[prv].end + per_unit_s * (i - prv - 1)
        else:
            continue
        u.start, u.end, u.weight = max(0.0, t), max(0.0, t) + 0.25, weight


def _alignment_ok(words: list[Word]) -> bool:
    if not words:
        return False
    tiny = sum(1 for w in words if w.end - w.start < 0.03)
    lowp = sum(1 for w in words if w.prob < 0.15)
    return tiny / len(words) < 0.25 and lowp / len(words) < 0.5


# ---------------------------------------------------------------- text <-> transcript matching
def _norm_chars(s: str, lang: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    if lang == "ja" or KANA_RE.search(s) or (CJK_RE.search(s) and lang == "ja"):
        s = to_hiragana(s)
    return "".join(ch.lower() for ch in s if unicodedata.category(ch)[0] in "LN")


def match_text_to_words(text: str, heard: list[Word], lang: str) -> list[Word]:
    """Give every word of the lyric sheet a time, using character-level matching against what
    Whisper heard. Works for spaced and unspaced scripts alike. Unmatched words are interpolated."""
    lyric_words = lyric_tokens(text, lang)
    line_starts = set()
    pos = 0
    for line in text.splitlines():
        n = len(lyric_tokens(line, lang))
        if n:
            line_starts.add(pos)
        pos += n
    # transcript characters with times
    t_chars, t_times = [], []
    for w in heard:
        cs = _norm_chars(w.text, lang)
        for k, ch in enumerate(cs):
            t_chars.append(ch)
            t_times.append(w.start + (w.end - w.start) * k / max(1, len(cs)))
    l_chars, owner = [], []
    for i, w in enumerate(lyric_words):
        cs = _norm_chars(w, lang)
        l_chars.extend(cs)
        owner.extend([i] * len(cs))
    starts: list[float | None] = [None] * len(lyric_words)
    ends: list[float | None] = [None] * len(lyric_words)
    if t_chars and l_chars:
        sm = difflib.SequenceMatcher(None, "".join(l_chars), "".join(t_chars), autojunk=False)
        for a, b, size in sm.get_matching_blocks():
            for k in range(size):
                wi = owner[a + k]
                t = t_times[b + k]
                if starts[wi] is None:
                    starts[wi] = t
                ends[wi] = t + 0.08
    # interpolate the gaps
    known = [i for i, s in enumerate(starts) if s is not None]
    out = []
    for i, w in enumerate(lyric_words):
        if starts[i] is None:
            prev = max((k for k in known if k < i), default=None)
            nxt = min((k for k in known if k > i), default=None)
            if prev is not None and nxt is not None:
                frac = (i - prev) / (nxt - prev)
                s = ends[prev] + (starts[nxt] - ends[prev]) * frac
            elif prev is not None:
                s = ends[prev] + 0.3 * (i - prev)
            elif nxt is not None:
                s = max(0.0, starts[nxt] - 0.3 * (nxt - i))
            else:
                s = 0.3 * i
            starts[i], ends[i] = s, s + 0.25
        out.append(Word(w, float(starts[i]), float(max(ends[i], starts[i] + 0.05)), 1.0 if i in known else 0.0,
                        line_start=i in line_starts))
    # enforce monotonic times
    for i in range(1, len(out)):
        if out[i].start < out[i - 1].start:
            out[i].start = out[i - 1].start + 0.01
            out[i].end = max(out[i].end, out[i].start + 0.05)
    return out


# ---------------------------------------------------------------- units
def words_to_units(words: list[Word], lang: str, mode: str | None = None, script: str = "native",
                   rough_timing: bool = False) -> list[Unit]:
    """rough_timing: timestamps came from a shaky alignment; weight them by Whisper's confidence."""
    mode = mode or default_mode(lang)
    units: list[Unit] = []
    for wi, w in enumerate(words):
        cw = clean_word(w.text, lang)
        if not cw:
            continue
        parts = split_units(cw, lang, mode, script)
        if not parts:
            continue
        syl = count_syllables(cw, lang) if mode == "word" else 1
        weight = (0.35 + 0.65 * min(1.0, w.prob * 2)) if rough_timing else 1.0
        dur = max(w.end - w.start, 0.02 * len(parts))
        # distribute in proportion to part length (rough: vowels take most of the time)
        weights = [max(1, len(p)) for p in parts]
        tot = sum(weights)
        t = w.start
        for k, (p, wt) in enumerate(zip(parts, weights)):
            d = dur * wt / tot
            units.append(Unit(p, t, t + d, wi, k == 0, syllables=syl if k == 0 else 1,
                              line_start=w.line_start and k == 0, weight=weight))
            t += d
    return units
