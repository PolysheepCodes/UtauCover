"""MIDI export (with lyric events) for use in other editors or OpenUTAU's MIDI import."""

from __future__ import annotations

from pathlib import Path

from .ustx import RESOLUTION, VoiceTrack


def write_midi(path: str | Path, bpm: float, tracks: list[VoiceTrack]) -> Path:
    import mido

    mid = mido.MidiFile(ticks_per_beat=RESOLUTION, type=1, charset="utf-8")
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    mid.tracks.append(meta)
    for t in tracks:
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=t.name, time=0))
        events = []
        for n in sorted(t.notes, key=lambda n: n.position):
            events.append((n.position, 1, "lyric", n.lyric))
            events.append((n.position, 2, "on", n.tone))
            events.append((n.position + n.duration, 0, "off", n.tone))
        events.sort(key=lambda e: (e[0], e[1]))
        now = 0
        for tick, _, kind, val in events:
            dt = max(0, tick - now)
            now = max(now, tick)
            if kind == "lyric":
                tr.append(mido.MetaMessage("lyrics", text=val, time=dt))
            elif kind == "on":
                tr.append(mido.Message("note_on", note=max(0, min(127, val)), velocity=100, time=dt))
            else:
                tr.append(mido.Message("note_off", note=max(0, min(127, val)), velocity=0, time=dt))
        mid.tracks.append(tr)
    path = Path(path)
    mid.save(str(path))
    return path
