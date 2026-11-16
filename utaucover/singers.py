"""Find installed OpenUTAU / UtauV voicebanks and pick a phonemizer + renderer for them."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .apps import data_dirs

B = "OpenUtau.Plugin.Builtin."
DS = "OpenUtau.Core.DiffSinger."
DEFAULT_PHONEMIZER = "OpenUtau.Core.DefaultPhonemizer"

# Fallback phonemizer per language for classic UTAU banks that don't declare one.
CLASSIC_BY_LANG = {
    "en": B + "ArpasingPhonemizer", "ja": B + "JapanesePresampPhonemizer", "zh": B + "ChineseCVVCPhonemizer",
    "yue": B + "CantoneseCVVCPhonemizer", "ko": B + "KoreanCVPhonemizer", "es": B + "SpanishSyllableBasedPhonemizer",
    "fr": B + "FrenchCMUSphinxPhonemizer", "de": B + "GermanVCCVPhonemizer", "it": B + "ItalianSyllableBasedPhonemizer",
    "pt": B + "BrazilianPortugueseCVCPhonemizer", "ru": B + "RussianCVCPhonemizer", "pl": B + "PolishCVCPhonemizer",
    "vi": B + "VietnameseCVVCPhonemizer", "th": B + "ThaiVCCVPhonemizer", "tr": B + "TurkishCVVCPhonemizer",
    "tl": B + "FilipinoPhonemizer", "la": B + "LatinDiphonePhonemizer",
}
DIFFSINGER_BY_LANG = {
    "en": DS + "DiffSingerEnglishPhonemizer", "ja": DS + "DiffSingerJapanesePhonemizer", "zh": DS + "DiffSingerChinesePhonemizer",
    "yue": DS + "DiffSingerJyutpingPhonemizer", "ko": DS + "DiffSingerKoreanPhonemizer", "es": DS + "DiffSingerSpanishPhonemizer",
    "fr": DS + "DiffSingerFrenchMillfeuillePhonemizer", "de": DS + "DiffSingerGermanPhonemizer", "it": DS + "DiffSingerItalianPhonemizer",
    "pt": DS + "DiffSingerPortuguesePhonemizer", "ru": DS + "DiffSingerRussianPhonemizer", "th": DS + "DiffSingerThaiPhonemizer",
    "fi": DS + "DiffSingerFinnishPhonemizer", "uk": DS + "DiffSingerUkrainianPhonemizer", "tl": DS + "DiffSingerFilipinoPhonemizer",
}
# Guess a bank's language from its phonemizer name.
LANG_HINTS = [("English", "en"), ("Arpasing", "en"), ("EnXSampa", "en"), ("Japanese", "ja"), ("Presamp", "ja"),
              ("Cantonese", "yue"), ("Jyutping", "yue"), ("Chinese", "zh"), ("Mandarin", "zh"), ("Korean", "ko"),
              ("Spanish", "es"), ("French", "fr"), ("German", "de"), ("Italian", "it"), ("Portuguese", "pt"),
              ("Russian", "ru"), ("Polish", "pl"), ("Vietnamese", "vi"), ("Thai", "th"), ("Turkish", "tr"),
              ("Filipino", "tl"), ("Finnish", "fi"), ("Ukrainian", "uk"), ("Latin", "la")]

RENDERERS = {"classic": "WORLDLINE-R", "utau": "WORLDLINE-R", "diffsinger": "DIFFSINGER", "enunu": "ENUNU",
             "vogen": "VOGEN", "voicevox": "VOICEVOX"}


@dataclass
class Singer:
    id: str                    # what goes in the .ustx "singer:" field (folder path relative to Singers/)
    name: str
    path: Path
    singer_type: str = "classic"
    default_phonemizer: str | None = None
    languages: list[str] = field(default_factory=list)

    def phonemizer_for(self, lang: str) -> str:
        if self.default_phonemizer and (not self.languages or lang in self.languages):
            return self.default_phonemizer
        table = DIFFSINGER_BY_LANG if self.singer_type == "diffsinger" else CLASSIC_BY_LANG
        return table.get(lang) or self.default_phonemizer or DEFAULT_PHONEMIZER

    def renderer(self, default_classic: str = "WORLDLINE-R") -> str:
        if self.singer_type in ("classic", "utau"):
            return default_classic
        return RENDERERS.get(self.singer_type, default_classic)

    def to_dict(self):
        return {"id": self.id, "name": self.name, "type": self.singer_type,
                "default_phonemizer": self.default_phonemizer, "languages": self.languages, "path": str(self.path)}


def _singer_roots() -> list[Path]:
    roots = []
    for app in data_dirs():
        s = app / "Singers"
        if s.is_dir():
            roots.append(s)
        prefs = app / "prefs.json"
        if prefs.exists():
            try:
                for extra in json.loads(prefs.read_text(encoding="utf-8-sig")).get("SingerSearchPaths", []) or []:
                    if Path(extra).is_dir():
                        roots.append(Path(extra))
            except Exception:
                pass
    seen, out = set(), []
    for r in roots:
        k = r.resolve()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def default_renderer() -> str:
    for app in data_dirs():
        prefs = app / "prefs.json"
        if prefs.exists():
            try:
                r = json.loads(prefs.read_text(encoding="utf-8-sig")).get("DefaultRenderer")
                if r:
                    return r
            except Exception:
                pass
    return "WORLDLINE-R"


def _read_character(d: Path) -> dict:
    info: dict = {}
    cy = d / "character.yaml"
    if cy.exists():
        try:
            info.update(yaml.safe_load(cy.read_text(encoding="utf-8-sig", errors="ignore")) or {})
        except Exception:
            pass
    ct = d / "character.txt"
    if ct.exists() and "name" not in info:
        for enc in ("utf-8-sig", "cp932", "gbk"):
            try:
                txt = ct.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                txt = ""
        m = re.search(r"^name\s*=\s*(.+)$", txt, re.M)
        if m:
            info["name"] = m.group(1).strip()
    if (d / "dsconfig.yaml").exists() and "singer_type" not in info:
        info["singer_type"] = "diffsinger"
    return info


def find_singers() -> list[Singer]:
    singers: list[Singer] = []
    seen = set()
    for root in _singer_roots():
        for marker in list(root.rglob("character.txt")) + list(root.rglob("character.yaml")) + list(root.rglob("dsconfig.yaml")):
            d = marker.parent
            # a voicebank's sub-folders (pitch banks) can contain character.txt too; keep the top-most
            try:
                rel = d.relative_to(root)
            except ValueError:
                continue
            if any(str(rel).startswith(s + "/") for s in seen) or str(rel) in seen:
                continue
            if any(p.exists() for p in [d.parent / "character.txt", d.parent / "character.yaml"]) and d.parent != root:
                continue
            seen.add(str(rel))
            info = _read_character(d)
            ph = info.get("default_phonemizer")
            langs = []
            for key, code in LANG_HINTS:
                if ph and key.lower() in ph.lower() and code not in langs:
                    langs.append(code)
            label = f"{rel} {info.get('name', '')}"
            if not langs:
                if re.search(r"(\b|_)(en|eng|english)(\b|_)", label, re.I):
                    langs.append("en")
                elif re.search(r"[぀-ヿ]", label):
                    langs.append("ja")
            if not ph and "en" in langs:
                if re.search(r"c\+v|cpv", label, re.I):
                    ph = B + "EnglishCpVPhonemizer"
                elif re.search(r"vccv", label, re.I):
                    ph = B + "EnglishVCCVPhonemizer"
            singers.append(Singer(id=str(rel), name=str(info.get("name") or rel), path=d,
                                  singer_type=str(info.get("singer_type") or "classic").lower(),
                                  default_phonemizer=ph, languages=langs))
    singers.sort(key=lambda s: s.name.lower())
    return singers


def resolve_singer(query: str | None) -> Singer | None:
    if not query:
        return None
    singers = find_singers()
    q = query.lower()
    for s in singers:
        if s.id.lower() == q or s.name.lower() == q:
            return s
    for s in singers:
        if q in s.id.lower() or q in s.name.lower():
            return s
    return None
