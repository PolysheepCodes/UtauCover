"""Write OpenUTAU / UtauV .ustx projects (format version 0.10)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RESOLUTION = 480
TEMPLATE = Path(__file__).parent / "templates" / "expressions_0.10.yaml"
VOICE_COLORS = ["Blue", "Pink", "Purple", "Orange", "Green", "Teal", "Red", "Light Blue", "Light Green", "Yellow"]


@dataclass
class UNote:
    position: int      # absolute tick
    duration: int
    tone: int
    lyric: str


@dataclass
class VoiceTrack:
    name: str
    notes: list[UNote]
    singer: str | None = None
    phonemizer: str = "OpenUtau.Core.DefaultPhonemizer"
    renderer: str = "WORLDLINE-R"
    color: str = "Blue"
    mute: bool = False
    volume: float = 0.0
    pan: float = 0.0
    pitd: tuple[list[int], list[int]] | None = None  # absolute ticks, cents


@dataclass
class WaveTrack:
    name: str
    path: Path           # audio file
    position: int = 0    # tick
    duration_ms: float = 0.0
    mute: bool = False
    volume: float = 0.0
    color: str = "Yellow"


def q(s: str) -> str:
    """YAML double-quoted scalar (always quote lyrics: 'no', 'on', '+' etc. stay strings)."""
    return json.dumps(s, ensure_ascii=False)


def _split_parts(notes: list[UNote], gap_ticks: int, bar_ticks: int) -> list[list[UNote]]:
    parts: list[list[UNote]] = []
    for n in notes:
        if parts and n.position - (parts[-1][-1].position + parts[-1][-1].duration) < gap_ticks:
            parts[-1].append(n)
        else:
            parts.append([n])
    return parts


def write_ustx(path: str | Path, bpm: float, voice: list[VoiceTrack], waves: list[WaveTrack],
               name: str = "Cover", beats_per_bar: int = 4, part_gap_bars: float = 2.0,
               comment: str = "") -> Path:
    path = Path(path)
    bar = RESOLUTION * beats_per_bar
    out: list[str] = []
    w = out.append
    w(f"name: {q(name)}")
    w(f"comment: {q(comment)}")
    w("output_dir: Vocal")
    w("cache_dir: UCache")
    w('ustx_version: "0.10"')
    w(f"bpm: {bpm:g}")
    w(f"beat_per_bar: {beats_per_bar}")
    w("beat_unit: 4")
    w(TEMPLATE.read_text(encoding="utf-8").rstrip("\n"))
    w("time_signatures:")
    w(f"- bar_position: 0\n  beat_per_bar: {beats_per_bar}\n  beat_unit: 4")
    w("tempos:")
    w(f"- position: 0\n  bpm: {bpm:g}")

    w("tracks:")
    for t in voice:
        if t.singer:
            w(f"- singer: {q(t.singer)}")
            w(f"  phonemizer: {t.phonemizer}")
        else:
            w(f"- phonemizer: {t.phonemizer}")
        w("  renderer_settings:")
        w(f"    renderer: {t.renderer}")
        _track_common(w, t.name, t.color, t.mute, t.volume, t.pan)
    for t in waves:
        w("- phonemizer: OpenUtau.Core.DefaultPhonemizer")
        w("  renderer_settings: {}")
        _track_common(w, t.name, t.color, t.mute, t.volume, 0.0)

    w("voice_parts:")
    any_part = False
    for ti, t in enumerate(voice):
        notes = sorted(t.notes, key=lambda n: n.position)
        if not notes:
            continue
        for pi, part_notes in enumerate(_split_parts(notes, int(part_gap_bars * bar), bar)):
            any_part = True
            start = (part_notes[0].position // bar) * bar  # parts start on a bar line
            end = part_notes[-1].position + part_notes[-1].duration
            dur = ((end - start + bar - 1) // bar) * bar
            w(f"- duration: {dur}")
            w(f"  name: {q(f'{t.name} {pi + 1}')}")
            w('  comment: ""')
            w(f"  track_no: {ti}")
            w(f"  position: {start}")
            w("  notes:")
            for n in part_notes:
                w(f"  - position: {n.position - start}")
                w(f"    duration: {n.duration}")
                w(f"    tone: {n.tone}")
                w(f"    lyric: {q(n.lyric)}")
                w("    pitch:")
                w("      data:")
                w("      - {x: -25, y: 0, shape: io}")
                w("      - {x: 25, y: 0, shape: io}")
                w("      snap_first: true")
                w("    vibrato: {length: 0, period: 175, depth: 25, in: 10, out: 10, shift: 0, drift: 0, vol_link: 0}")
                w("    tuning: 0")
                w("    phoneme_expressions: []")
                w("    phoneme_overrides: []")
            curve = _part_curve(t.pitd, start, start + dur) if t.pitd else None
            if curve:
                xs, ys = curve
                w("  curves:")
                w(f"  - xs: [{', '.join(map(str, xs))}]")
                w(f"    ys: [{', '.join(map(str, ys))}]")
                w("    abbr: pitd")
            else:
                w("  curves: []")
            w("  masked_curves: []")
    if not any_part:
        out[-1] = "voice_parts: []"

    if waves:
        w("wave_parts:")
        base = path.parent.resolve()
        for ti, t in enumerate(waves):
            rel = _relpath(Path(t.path).resolve(), base)
            w(f"- name: {q(Path(t.path).name)}")
            w('  comment: ""')
            w(f"  track_no: {len(voice) + ti}")
            w(f"  position: {t.position}")
            w(f"  relative_path: {q(rel)}")
            w(f"  file_duration_ms: {t.duration_ms:.4f}")
            w("  skip: 0\n  trim: 0\n  fadein: 0\n  fadeout: 0")
    else:
        w("wave_parts: []")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def _track_common(w, name, color, mute, volume, pan):
    w(f"  track_name: {q(name)}")
    w(f"  track_color: {color}")
    w(f"  mute: {'true' if mute else 'false'}")
    w("  solo: false")
    w(f"  volume: {volume:g}")
    w(f"  pan: {pan:g}")
    w("  track_expressions: []")
    w("  voice_color_names:")
    w('  - ""')


def _relpath(p: Path, base: Path) -> str:
    import os
    try:
        return os.path.relpath(p, base)
    except ValueError:  # different drive (Windows)
        return str(p)


def _part_curve(pitd, start: int, end: int):
    xs, ys = pitd
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    m = (xs >= start) & (xs < end)
    if not m.any():
        return None
    return [int(x - start) for x in xs[m]], [int(y) for y in ys[m]]
