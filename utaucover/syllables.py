"""Language-aware splitting of lyric words into the units that go on notes.

Two lyric modes:
  word      one word per note group; extra notes get "+". OpenUTAU's English (and most
            other alphabet-language) phonemizers split the syllables across the "+" notes.
  syllable  one syllable / mora per note: Japanese kana, Chinese characters (or pinyin),
            Korean Hangul blocks, or hyphenated syllables for other languages.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

SYLLABLE_LANGS = {"ja", "zh", "yue", "ko", "vi", "th", "lo", "my", "km"}

SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")
HANGUL_RE = re.compile(r"[가-힣]")
KANA_RE = re.compile(r"[぀-ヿ]")


def default_mode(lang: str) -> str:
    return "syllable" if lang in SYLLABLE_LANGS else "word"


def clean_word(w: str, lang: str) -> str:
    w = unicodedata.normalize("NFKC", w).strip()
    # keep letters, marks, digits, apostrophes inside words, kana prolonged mark
    out = []
    for i, ch in enumerate(w):
        cat = unicodedata.category(ch)
        if cat[0] in "LMN" or ch == "ー":
            out.append(ch)
        elif ch in "'’" and 0 < i < len(w) - 1:
            out.append("'")
    s = "".join(out)
    if lang not in SYLLABLE_LANGS:
        s = s.lower()
    return s


# ---------------- Japanese ----------------
@lru_cache(maxsize=1)
def _kakasi():
    import pykakasi
    return pykakasi.kakasi()


def to_hiragana(text: str) -> str:
    return "".join(item["hira"] for item in _kakasi().convert(text))


def kana_moras(hira: str) -> list[str]:
    moras: list[str] = []
    for ch in hira:
        if ch in SMALL_KANA and moras:
            moras[-1] += ch
        elif ch == "ー":
            moras.append("+")
        elif KANA_RE.match(ch):
            # katakana -> hiragana
            o = ord(ch)
            if 0x30A1 <= o <= 0x30F6:
                ch = chr(o - 0x60)
            moras.append(ch)
        elif ch.isalnum():
            moras.append(ch)
    return moras


def kana_to_romaji(mora: str) -> str:
    if mora == "+":
        return "+"
    if mora in ("っ", "ッ"):
        return "cl"  # sokuon, as romaji UTAU banks name it
    return "".join(i["hepburn"] for i in _kakasi().convert(mora)) or mora


# ---------------- Chinese / Korean ----------------
def zh_syllables(word: str, pinyin: bool) -> list[str]:
    chars = [c for c in word if CJK_RE.match(c) or c.isalnum()]
    if not pinyin:
        return chars
    from pypinyin import lazy_pinyin
    return lazy_pinyin("".join(chars))


# ---------------- alphabetic languages ----------------
PYPHEN_MAP = {"en": "en_US", "de": "de_DE", "pt": "pt_PT", "it": "it_IT", "ru": "ru_RU", "pl": "pl_PL",
              "es": "es", "fr": "fr", "nl": "nl_NL", "sv": "sv", "da": "da_DK", "nb": "nb_NO", "no": "nb_NO",
              "fi": "fi_FI", "cs": "cs_CZ", "sk": "sk_SK", "hu": "hu_HU", "ro": "ro_RO", "uk": "uk_UA",
              "el": "el_GR", "tr": "tr_TR", "id": "id_ID", "ca": "ca", "hr": "hr_HR", "sl": "sl_SI",
              "bg": "bg", "lt": "lt", "lv": "lv", "et": "et_EE", "gl": "gl", "eu": "eu"}


@lru_cache(maxsize=32)
def _pyphen(lang: str):
    import pyphen
    code = PYPHEN_MAP.get(lang, lang)
    try:
        fb = pyphen.language_fallback(code)
        return pyphen.Pyphen(lang=fb) if fb else None
    except Exception:
        return None


VOWELS = set("aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿæœаеёиоуыэюяіїєαεηιουω")


def alpha_syllables(word: str, lang: str) -> list[str]:
    dic = _pyphen(lang)
    if dic is not None:
        parts = dic.inserted(word, hyphen="‧").split("‧")
        parts = [p for p in parts if p]
        if parts:
            return parts
    # fallback: split before each vowel group that follows a consonant
    out, cur, seen_vowel = [], "", False
    for ch in word:
        is_v = ch.lower() in VOWELS
        if is_v and seen_vowel and cur and cur[-1].lower() not in VOWELS:
            # move one consonant to the next syllable
            out.append(cur[:-1])
            cur = cur[-1]
        cur += ch
        seen_vowel = seen_vowel or is_v
    if cur:
        out.append(cur)
    return [o for o in out if o] or [word]


def split_units(word: str, lang: str, mode: str, script: str = "native") -> list[str]:
    """Split a cleaned word into the lyric units placed on consecutive notes.

    script: "native" (kana/hanzi/hangul as-is), "romaji" for Japanese, "pinyin" for Chinese.
    """
    if not word:
        return []
    if mode == "word":
        return [word]
    if lang == "ja" or KANA_RE.search(word):
        moras = kana_moras(to_hiragana(word))
        if script == "romaji":
            moras = [kana_to_romaji(m) for m in moras]
        return moras
    if lang in ("zh", "yue") or CJK_RE.search(word):
        return zh_syllables(word, pinyin=(script == "pinyin"))
    if lang == "ko" or HANGUL_RE.search(word):
        return [c for c in word if HANGUL_RE.match(c) or c.isalnum()]
    return alpha_syllables(word, lang)


def fill_lyric(lang: str, script: str = "native") -> str:
    """Placeholder lyric for sung notes with no recognised words (vocalises, 'ooh's)."""
    if lang == "ja":
        return "a" if script == "romaji" else "あ"
    if lang == "ko":
        return "아"
    if lang in ("zh", "yue"):
        return "a" if script == "pinyin" else "啊"
    return "a"


@lru_cache(maxsize=8192)
def count_syllables(word: str, lang: str) -> int:
    """How many notes a word naturally takes. English uses the CMU dictionary."""
    if lang == "en":
        try:
            import pronouncing
            phones = pronouncing.phones_for_word(word.lower().replace("'", "'"))
            if phones:
                return max(1, pronouncing.syllable_count(phones[0]))
        except Exception:
            pass
        groups = re.findall(r"[aeiouy]+", word.lower())
        n = len(groups)
        if word.lower().endswith("e") and n > 1 and not word.lower().endswith(("le", "ee")):
            n -= 1
        return max(1, n)
    if lang in SYLLABLE_LANGS:
        return 1
    return max(1, len(alpha_syllables(word, lang)))


# ---------------- consonant lead-in (put the vowel on the note, like a hand-made chart) --------------
_VOICED_TH = {"the", "this", "that", "these", "those", "they", "them", "their", "there", "then", "than",
              "though", "thus", "thy", "thee", "thou", "thine", "themselves", "theirs", "therefore"}
_KANA_LONG = set("なにぬねのまみむめもらりるれろわやゆよをん")
_KANA_SHORT = set("がぎぐげござじずぜぞだぢづでどばびぶべぼ")


def onset_consonant_s(text: str, lang: str) -> float:
    """Rough length of a voiced consonant at the start of a sung word or syllable.

    The pitch tracker hears voicing from the first voiced consonant (m, n, l, w, ...), but UTAU
    places a note's consonant *before* the note, so a note starting at that point makes the word
    land early. Shifting the note start by this much puts the vowel on the note instead.
    """
    if not text or text.startswith("+"):
        return 0.0
    # Values are about half a typical consonant: tuned on a hand-charted project (onset F1 0.57 -> 0.63).
    w = text.lower()
    ch = w[0]
    if KANA_RE.match(ch):
        h = to_hiragana(ch) if not ("\u3041" <= ch <= "\u3096") else ch
        return 0.03 if h in _KANA_LONG else 0.015 if h in _KANA_SHORT else 0.0
    if HANGUL_RE.match(ch):
        initial = (ord(ch) - 0xAC00) // 588
        return 0.03 if initial in (2, 5, 6) else 0.0  # ㄴ ㄹ ㅁ
    if lang == "en" and w in _VOICED_TH:
        return 0.025
    if w.startswith("wh"):
        return 0.025
    if lang == "en" and w[:2] in ("kn", "wr", "gn"):  # silent first letter: the n / r is what's voiced
        return 0.035
    if ch in "mnlrwyvzj":
        return 0.035
    if ch in "bdg":
        return 0.015
    return 0.0
