"""Discover the phonemizers installed in UtauV / OpenUTAU, and pick one for a voicebank.

The list is read from the apps' own plugin metadata (the same [Phonemizer] attributes the app
uses for its menu) by loading their .NET assemblies in a sandboxed helper process. It's cached
and refreshed when an app or plugin changes. Without .NET hosting available, a built-in table
of known phonemizers is used instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .apps import data_dirs, programs
from .paths import APP_SUPPORT
CACHE = APP_SUPPORT / "phonemizers.json"

# OpenUTAU language tags -> Whisper language codes
LANG_TAGS = {"ZH-YUE": "yue", "FIL": "tl"}

# Dictionary file each phonemizer reads (from the Plugins folder and the voicebank folder).
DICTIONARY_FILES = {
    "EnglishCpVPhonemizer": "en-cPv.yaml", "ArpasingPhonemizer": "arpasing.yaml",
    "ArpasingPlusPhonemizer": "arpasing.yaml", "EnglishVCCVPhonemizer": "envccv.yaml",
    "EnXSampaPhonemizer": "en-xsampa.yaml", "ENtoJAPhonemizer": "en2ja.yaml",
    "SpanishSyllableBasedPhonemizer": "es-syl.yaml", "ItalianSyllableBasedPhonemizer": "it-syl.yaml",
    "GermanVCCVPhonemizer": "german.yaml", "KoreanCVPhonemizer": "kocv.yaml",
    "ChineseCVVPlusPhonemizer": "zhcvv.yaml",
}
# Phonemizers whose phonemes are ARPAbet (CMU dictionary symbols) -> "[aa b k]" respellings work.
ARPABET = {"EnglishCpVPhonemizer", "ArpasingPhonemizer", "ArpasingPlusPhonemizer",
           "DiffSingerEnglishPhonemizer", "DiffSingerARPAPlusEnglishPhonemizer"}

# Used only when the apps' metadata can't be read.
FALLBACK = [
    ("OpenUtau.Core.DefaultPhonemizer", "Default Phonemizer", "DEFAULT", ""),
    ("OpenUtau.Plugin.Builtin.ArpasingPhonemizer", "English Arpasing Phonemizer", "EN ARPA", "EN"),
    ("OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer", "English Arpasing+ Phonemizer", "EN ARPA+", "EN"),
    ("OpenUtau.Plugin.Builtin.EnglishCpVPhonemizer", "English C+V Phonemizer", "EN C+V", "EN"),
    ("OpenUtau.Plugin.Builtin.EnglishVCCVPhonemizer", "English VCCV Phonemizer", "EN VCCV", "EN"),
    ("OpenUtau.Plugin.Builtin.EnXSampaPhonemizer", "English X-SAMPA Phonemizer", "EN X-SAMPA", "EN"),
    ("OpenUtau.Plugin.Builtin.JapanesePresampPhonemizer", "Japanese presamp Phonemizer", "JA VCV & CVVC", "JA"),
    ("OpenUtau.Plugin.Builtin.JapaneseCVVCPhonemizer", "Japanese CVVC Phonemizer", "JA CVVC", "JA"),
    ("OpenUtau.Plugin.Builtin.JapaneseVCVPhonemizer", "Japanese VCV Phonemizer", "JA VCV", "JA"),
    ("OpenUtau.Plugin.Builtin.ChineseCVVCPhonemizer", "Chinese CVVC Phonemizer", "ZH CVVC", "ZH"),
    ("OpenUtau.Plugin.Builtin.KoreanCVPhonemizer", "Korean CV Phonemizer", "KO CV", "KO"),
    ("OpenUtau.Plugin.Builtin.SpanishSyllableBasedPhonemizer", "Spanish Syllable-Based Phonemizer", "ES SYL", "ES"),
    ("OpenUtau.Plugin.Builtin.FrenchCMUSphinxPhonemizer", "French CMUSphinx Phonemizer", "FR SPHINX", "FR"),
    ("OpenUtau.Plugin.Builtin.GermanVCCVPhonemizer", "German VCCV Phonemizer", "DE VCCV", "DE"),
    ("OpenUtau.Plugin.Builtin.ItalianSyllableBasedPhonemizer", "Italian Syllable-Based Phonemizer", "IT SYL", "IT"),
    ("OpenUtau.Plugin.Builtin.BrazilianPortugueseCVCPhonemizer", "Brazilian Portuguese CVC Phonemizer", "PT-BR CVC", "PT"),
    ("OpenUtau.Plugin.Builtin.RussianCVCPhonemizer", "Russian CVC Phonemizer", "RU CVC", "RU"),
]


@dataclass
class PhonemizerInfo:
    cls: str          # full class name, as stored in .ustx files
    name: str
    tag: str          # short label shown in the app (e.g. "EN C+V")
    language: str     # OpenUTAU tag (EN, JA, ZH-YUE...) or "" for language-neutral
    author: str = ""
    source: str = ""  # which app / plugin it came from

    @property
    def short(self) -> str:
        return self.cls.rsplit(".", 1)[-1]

    @property
    def lang(self) -> str:
        """Whisper language code, or '' for language-neutral phonemizers."""
        t = (self.language or "").upper()
        return LANG_TAGS.get(t, t.lower())

    @property
    def is_diffsinger(self) -> bool:
        return ".DiffSinger." in self.cls

    @property
    def dictionary(self) -> str | None:
        if self.is_diffsinger:
            return "dsdict.yaml"
        return DICTIONARY_FILES.get(self.short)

    @property
    def arpabet(self) -> bool:
        return self.short in ARPABET

    def label(self) -> str:
        return f"{self.tag}  ({self.name})"

    def to_dict(self):
        d = asdict(self)
        d.update(short=self.short, lang=self.lang, dictionary=self.dictionary)
        return d


# ------------------------------------------------------------------------------------------ discovery
_HELPER = r'''
import sys, json, glob, os
from clr_loader import get_coreclr
from pythonnet import set_runtime
app_dir, plugin_dirs = sys.argv[1], sys.argv[2:]
entry = [f for f in glob.glob(os.path.join(app_dir, "*.dll")) if os.path.exists(f[:-4] + ".runtimeconfig.json")]
set_runtime(get_coreclr(entry_dll=entry[0], dotnet_root=app_dir))
import clr
sys.path.append(app_dir)
clr.AddReference("OpenUtau.Core")
def _sandbox_app_paths():
    """UtauV saves its settings next to the program running its code (here: UtauCover's Python, inside the
    app). Point its folders at this helper's throwaway folder before anything reads its settings."""
    box = os.environ.get("UTAUCOVER_SANDBOX")
    if not box:
        return
    try:
        from OpenUtau.Core import PathManager
        pm = PathManager.Inst
        for name, value in (("RootPath", box), ("DataPath", os.path.join(box, "data")),
                            ("CachePath", os.path.join(box, "cache"))):
            try:
                setattr(pm, name, value)
            except Exception:
                from System.Reflection import BindingFlags
                f = pm.GetType().GetField(f"<{name}>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)
                if f is not None:
                    f.SetValue(pm, value)
    except Exception:
        pass
_sandbox_app_paths()
import System
from OpenUtau.Api import PhonemizerAttribute, Phonemizer
base = clr.GetClrType(Phonemizer); attr_t = clr.GetClrType(PhonemizerAttribute)
asms = [("app", os.path.join(app_dir, n)) for n in ("OpenUtau.Core.dll", "OpenUtau.Plugin.Builtin.dll")]
for d in plugin_dirs:
    for f in glob.glob(os.path.join(d, "**", "*.dll"), recursive=True):
        asms.append(("plugin:" + os.path.basename(f), f))
out = []
for src, path in asms:
    try:
        asm = System.Reflection.Assembly.LoadFrom(path)
        try:
            types = asm.GetTypes()
        except System.Reflection.ReflectionTypeLoadException as e:
            types = [t for t in e.Types if t is not None]
    except Exception:
        continue
    for t in types:
        try:
            if t.IsAbstract or not base.IsAssignableFrom(t):
                continue
            a = t.GetCustomAttributes(attr_t, False)
            if not a:
                continue
            a = a[0]
            out.append({"cls": t.FullName, "name": a.Name, "tag": a.Tag, "language": a.Language or "",
                        "author": a.Author or "", "source": src})
        except Exception:
            continue
print("@@JSON@@" + json.dumps(out, ensure_ascii=False))
'''


def _plugin_dirs() -> list[Path]:
    return [d / "Plugins" for d in data_dirs() if (d / "Plugins").is_dir()]


def _signature() -> dict:
    sig = {}
    for app in programs():
        for n in ("OpenUtau.Core.dll", "OpenUtau.Plugin.Builtin.dll"):
            f = app.code_dir / n
            if f.exists():
                sig[str(f)] = f.stat().st_mtime
    for d in _plugin_dirs():
        for f in d.rglob("*.dll"):
            sig[str(f)] = f.stat().st_mtime
    return sig


def _reflect(app_dir: Path) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="utaucover_reflect_") as home:
        from .game import _no_window, _sandbox_env
        env = _sandbox_env(home)  # sandbox: the app's code can't touch real settings
        script = Path(home) / "helper.py"
        script.write_text(_HELPER, encoding="utf-8")
        r = subprocess.run([sys.executable, "-I", str(script), str(app_dir), *map(str, _plugin_dirs())],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                           env=env, **_no_window())
    for line in r.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[8:])
    raise RuntimeError(r.stderr[-500:])


def find_phonemizers(refresh: bool = False) -> list[PhonemizerInfo]:
    sig = _signature()
    if not refresh and CACHE.exists():
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            if data.get("signature") == sig:
                return [PhonemizerInfo(**p) for p in data["phonemizers"]]
        except Exception:
            pass
    found: dict[str, dict] = {}
    for app in programs():
        try:
            for p in _reflect(app.code_dir):
                p["source"] = app.name if p["source"] == "app" else p["source"]
                found.setdefault(p["cls"], p)
        except Exception:
            continue
    if not found:
        found = {c: {"cls": c, "name": n, "tag": t, "language": l, "author": "", "source": "built-in table"}
                 for c, n, t, l in FALLBACK}
    out = sorted((PhonemizerInfo(**p) for p in found.values()), key=lambda p: (p.language or "~", p.tag))
    try:
        APP_SUPPORT.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"signature": sig, "phonemizers": [asdict(p) for p in out]}, ensure_ascii=False),
                         encoding="utf-8")
    except Exception:
        pass
    return out


def resolve_phonemizer(query: str | None, installed: list[PhonemizerInfo] | None = None) -> PhonemizerInfo | None:
    """Match by full class name, short class name, tag (e.g. 'EN C+V') or display name."""
    if not query:
        return None
    installed = installed if installed is not None else find_phonemizers()
    q = query.strip().lower()
    for key in (lambda p: p.cls.lower(), lambda p: p.short.lower(), lambda p: p.tag.lower(), lambda p: p.name.lower()):
        for p in installed:
            if key(p) == q:
                return p
    hits = [p for p in installed if q in p.cls.lower() or q in p.tag.lower() or q in p.name.lower()]
    return hits[0] if len(hits) == 1 else None


def for_language(lang: str, installed: list[PhonemizerInfo] | None = None) -> list[PhonemizerInfo]:
    installed = installed if installed is not None else find_phonemizers()
    return [p for p in installed if p.lang == lang]


def by_class(cls: str | None, installed: list[PhonemizerInfo] | None = None) -> PhonemizerInfo | None:
    if not cls:
        return None
    installed = installed if installed is not None else find_phonemizers()
    return next((p for p in installed if p.cls == cls), None)
