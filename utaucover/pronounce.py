"""Pronunciation checks against the dictionaries the chosen phonemizer will actually use.

OpenUTAU's dictionary-based phonemizers look a word up in
  1. the voicebank folder's dictionary file (e.g. Googoloid's en-cPv.yaml),
  2. the app's Plugins folder dictionary of the same name,
  3. for English ARPAbet phonemizers, the built-in CMU dictionary,
and otherwise *guess* the pronunciation letter by letter. Guessed words are the usual reason a
cover "sounds wrong", so we list them, and for lines in another language (a Spanish verse sung by an
English bank) we write explicit phonemes using UtauV's "word [phonemes]" lyric notation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .apps import data_dirs, target_data_dir
from .phonemizers import PhonemizerInfo


@dataclass
class PronunciationReport:
    phonemizer: str
    dictionaries: list[str] = field(default_factory=list)
    checked: int = 0
    unknown: dict[str, int] = field(default_factory=dict)       # word -> occurrences (will be guessed)
    respelled: dict[str, str] = field(default_factory=dict)     # word -> "[phonemes]"
    foreign_lines: list[str] = field(default_factory=list)
    misplaced: list[str] = field(default_factory=list)          # dictionaries in the other app's folder

    def summary(self) -> str:
        parts = [f"{self.checked} words checked against {', '.join(self.dictionaries) or 'no dictionary'}"]
        if self.unknown:
            parts.append(f"{len(self.unknown)} not in any dictionary (UtauV will guess them)")
        if self.respelled:
            parts.append(f"{len(self.respelled)} respelled with explicit phonemes")
        if self.misplaced:
            parts.append("a dictionary for this phonemizer is in the other app's Plugins folder (see report)")
        return "; ".join(parts)

    def text(self) -> str:
        out = [f"Phonemizer: {self.phonemizer}", f"Dictionaries: {', '.join(self.dictionaries) or '(none)'}",
               f"Words checked: {self.checked}", ""]
        for m in self.misplaced:
            out.append(f"NOTE: {m} exists, but {target_app_dir()}/Plugins does not have it, so")
            out.append(f"      {target_app_dir().name} won't use it. Copy it there to use it.")
            out.append("")
        if self.unknown:
            out.append("Not found in any dictionary. UtauV will guess these; respell them in the lyrics, or")
            out.append("write the phonemes yourself as  word [p h o n e m e s]  (the same notation UtauV uses):")
            for w, n in sorted(self.unknown.items(), key=lambda x: (-x[1], x[0])):
                out.append(f"  {w}" + (f"  (x{n})" if n > 1 else ""))
            out.append("")
        if self.respelled:
            out.append("Written with explicit phonemes (lines in another language):")
            for w, ph in self.respelled.items():
                out.append(f"  {w} -> {ph}")
            out.append("")
        if self.foreign_lines:
            out.append("Lines that look like another language:")
            out.extend(f"  {l}" for l in self.foreign_lines)
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------- dictionaries
def target_app_dir() -> Path:
    """Data folder of the app the projects are for: UtauV if it's installed, else OpenUTAU."""
    return target_data_dir()


def misplaced_dictionaries(phon: PhonemizerInfo) -> list[Path]:
    """Dictionary files sitting in *another* app's Plugins folder, which the target app won't read."""
    name = phon.dictionary
    if not name or phon.is_diffsinger:
        return []
    mine = target_app_dir() / "Plugins" / name
    if mine.exists():
        return []
    return [d / "Plugins" / name for d in data_dirs() if d != target_app_dir() and (d / "Plugins" / name).exists()]


def dictionary_paths(phon: PhonemizerInfo, singer_dir: Path | None) -> list[Path]:
    """Dictionaries the target app will actually read: the voicebank's own, then its Plugins folder."""
    name = phon.dictionary
    if not name:
        return []
    cands = []
    if singer_dir is not None:
        cands.append(Path(singer_dir) / name)
        if phon.is_diffsinger:
            cands += sorted(Path(singer_dir).rglob("dsdict*.yaml"))
    cands.append(target_app_dir() / "Plugins" / name)
    seen, out = set(), []
    for c in cands:
        if c.exists() and c.resolve() not in seen:
            seen.add(c.resolve())
            out.append(c)
    return out


@lru_cache(maxsize=32)
def _graphemes(path: str, mtime: float) -> frozenset:
    txt = Path(path).read_text(encoding="utf-8-sig", errors="ignore")
    return frozenset(m.lower() for m in re.findall(r"grapheme:\s*['\"]?([^\s'\",}]+)", txt))


@lru_cache(maxsize=1)
def _cmudict() -> frozenset:
    try:
        import pronouncing
        pronouncing.init_cmu()
        return frozenset(w.lower() for w in pronouncing.lookup)
    except Exception:
        return frozenset()


def known_words(phon: PhonemizerInfo, singer_dir: Path | None) -> tuple[frozenset | None, list[str]]:
    """Words the phonemizer can pronounce from a dictionary. None means 'not dictionary-based'."""
    paths = dictionary_paths(phon, singer_dir)
    words: set = set()
    names = []
    for p in paths:
        words |= _graphemes(str(p), p.stat().st_mtime)
        names.append(str(p).replace(str(Path.home()), "~"))
    if phon.arpabet and phon.lang == "en":
        words |= _cmudict()
        names.append("CMU dictionary (built in)")
    if not names:
        return None, []
    return frozenset(words), names


# ---------------------------------------------------------------------------------- language of a line
_STOP = {
    "es": {"de", "la", "que", "el", "en", "y", "los", "se", "del", "las", "un", "por", "con", "no", "una", "su",
           "para", "es", "al", "lo", "como", "más", "mas", "pero", "sus", "le", "ya", "este", "sí", "si", "porque",
           "esta", "cuando", "muy", "sin", "sobre", "también", "tampoco", "me", "hasta", "hay", "donde", "dónde",
           "quien", "todo", "nos", "tu", "tú", "te", "mi", "yo", "siempre", "nadie", "nunca", "vas", "eres", "soy",
           "estoy", "amor", "corazón", "vida", "noche", "calle", "muerte", "suerte", "verte", "puedo", "pueda"},
    "fr": {"le", "la", "les", "de", "des", "et", "est", "je", "tu", "il", "elle", "nous", "vous", "un", "une",
           "que", "qui", "pas", "ne", "pour", "dans", "sur", "avec", "mon", "ma", "mes", "ton", "ta", "moi", "toi"},
    "pt": {"de", "que", "não", "nao", "um", "uma", "com", "para", "os", "as", "eu", "você", "voce", "meu", "minha",
           "do", "da", "dos", "das", "em", "no", "na", "por", "mais", "mas", "coração", "amor"},
    "it": {"di", "che", "il", "la", "non", "un", "una", "per", "con", "sono", "mi", "ti", "io", "tu", "lui",
           "lei", "noi", "voi", "del", "della", "amore", "cuore"},
    "de": {"der", "die", "das", "und", "ich", "du", "nicht", "ist", "ein", "eine", "mit", "auf", "für", "dich",
           "mich", "wir", "ihr", "sie", "es", "zu", "den", "dem", "liebe", "herz"},
}


def guess_line_language(line: str, default: str) -> str:
    toks = [t for t in re.findall(r"[^\W\d_]+", line.lower())]
    if not toks:
        return default
    if re.search(r"[぀-ヿ]", line):
        return "ja"
    if re.search(r"[가-힣]", line):
        return "ko"
    if re.search(r"[一-鿿]", line):
        return "zh"
    if re.search(r"[Ѐ-ӿ]", line):
        return "ru"
    scores = {lang: sum(t in s for t in toks) for lang, s in _STOP.items()}
    if re.search(r"[ñ¿¡]", line):
        scores["es"] += 3
    best = max(scores, key=scores.get)
    english = _cmudict()
    unknown = sum(1 for t in toks if english and t not in english)
    if scores[best] >= 2 and (unknown >= 2 or scores[best] >= 0.4 * len(toks)):
        return best
    return default


def guess_text_language(text: str, default: str = "en") -> str:
    """Language of a whole lyric sheet from its script and common words (no audio needed)."""
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[가-힣]", text):
        return "ko"
    if re.search(r"[一-鿿]", text):
        return "zh"
    if re.search(r"[Ѐ-ӿ]", text):
        return "uk" if re.search(r"[іїєґ]", text) else "ru"
    lines = [l for l in text.splitlines() if l.strip()]
    votes: dict[str, int] = {}
    for l in lines:
        g = guess_line_language(l, "en")
        votes[g] = votes.get(g, 0) + 1
    return max(votes, key=votes.get) if votes else default


# ---------------------------------------------------------------------------------- respelling
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


_ES_VOWEL = {"a": "aa", "e": "eh", "i": "iy", "o": "ow", "u": "uw"}


def spanish_to_arpabet(word: str) -> list[str]:
    """Spanish spelling is regular, so simple rules give a good ARPAbet approximation."""
    w = word.lower().replace("ñ", "ny")
    w = _strip_accents(w)
    out: list[str] = []
    i = 0
    while i < len(w):
        c = w[i]
        nxt = w[i + 1] if i + 1 < len(w) else ""
        two = w[i:i + 2]
        if two == "ch":
            out.append("ch"); i += 2; continue
        if two == "ll":
            out.append("y"); i += 2; continue
        if two == "rr":
            out.append("r"); i += 2; continue
        if two == "qu":
            out.append("k"); i += 2; continue
        if two in ("gu",) and w[i + 2:i + 3] in ("e", "i"):
            out.append("g"); i += 2; continue
        if c == "c":
            out.append("s" if nxt in "ei" and nxt else "k"); i += 1; continue
        if c == "g":
            out.append("hh" if nxt in "ei" and nxt else "g"); i += 1; continue
        if c == "j":
            out.append("hh"); i += 1; continue
        if c == "h":
            i += 1; continue
        if c in "bv":
            out.append("b"); i += 1; continue
        if c == "z":
            out.append("s"); i += 1; continue
        if c == "x":
            out += ["k", "s"]; i += 1; continue
        if c == "y":
            out.append("iy" if i == len(w) - 1 else "y"); i += 1; continue
        if c in _ES_VOWEL:
            # unstressed i/u next to another vowel glide into it (ie -> y eh, ue -> w eh)
            if c == "i" and nxt in "aeou" and nxt:
                out.append("y"); i += 1; continue
            if c == "u" and nxt in "aeio" and nxt:
                out.append("w"); i += 1; continue
            out.append(_ES_VOWEL[c]); i += 1; continue
        if c in "dfklmnprst":
            out.append(c); i += 1; continue
        i += 1
    return out


RESPELLERS = {"es": spanish_to_arpabet}


def build_report(words: list[str], word_line: list[int], lines: list[str], phon: PhonemizerInfo | None,
                 singer_dir: Path | None, song_lang: str, respell: bool = True
                 ) -> tuple[PronunciationReport, dict[int, str]]:
    """Check each lyric word (words[i] sung in lines[word_line[i]]).

    Returns the report and {word index -> replacement lyric} for words written with explicit phonemes
    (only words that sit in a line detected as another language, and only for ARPAbet phonemizers)."""
    rep = PronunciationReport(phonemizer=phon.label() if phon else "(none)")
    replacements: dict[int, str] = {}
    if phon is None:
        return rep, replacements
    known, names = known_words(phon, singer_dir)
    rep.dictionaries = names
    rep.misplaced = [str(p).replace(str(Path.home()), "~") for p in misplaced_dictionaries(phon)]
    target = phon.lang or song_lang
    line_lang = [guess_line_language(l, song_lang) for l in lines]
    rep.foreign_lines = [l for l, lg in zip(lines, line_lang) if lg != target]
    # Inside a foreign line, decide word by word (lines can mix languages: "a voice is calling /
    # tampoco en la calle"): a word is foreign if the phonemizer's dictionaries don't know it or it's a
    # function word of that language; lone labels are smoothed by their neighbours.
    foreign_word = [False] * len(words)
    by_line: dict[int, list[int]] = {}
    for i, li in enumerate(word_line):
        by_line.setdefault(li, []).append(i)
    for li, idxs in by_line.items():
        if not (0 <= li < len(line_lang)) or line_lang[li] == target:
            continue
        stops = _STOP.get(line_lang[li], set())
        lab = []
        for i in idxs:
            lw = (words[i] or "").lower()
            in_dict = known is not None and lw in known
            lab.append(lw in stops or not in_dict)
        sm = lab[:]
        for k in range(len(lab)):
            left = lab[k - 1] if k > 0 else None
            right = lab[k + 1] if k + 1 < len(lab) else None
            if left is not None and right is not None and left == right != lab[k]:
                sm[k] = left
        for i, f in zip(idxs, sm):
            foreign_word[i] = f
    for i, w in enumerate(words):
        if not w or w.startswith("+"):
            continue
        rep.checked += 1
        lw = w.lower()
        li = word_line[i] if i < len(word_line) else -1
        src = line_lang[li] if 0 <= li < len(line_lang) else song_lang
        if foreign_word[i] and respell and phon.arpabet and src in RESPELLERS:
            ph = RESPELLERS[src](lw)
            if ph:
                rep.respelled[lw] = "[" + " ".join(ph) + "]"
                replacements[i] = f"{lw} [{' '.join(ph)}]"
                continue
        if known is not None and lw not in known:
            rep.unknown[lw] = rep.unknown.get(lw, 0) + 1
    return rep, replacements


def lines_from_words(words) -> tuple[list[str], list[int]]:
    """Rebuild lyric lines from word objects with .text / .line_start. Returns (lines, line index per word)."""
    lines: list[list[str]] = []
    idx: list[int] = []
    for w in words:
        if w.line_start or not lines:
            lines.append([])
        lines[-1].append(w.text)
        idx.append(len(lines) - 1)
    return [" ".join(l) for l in lines], idx
