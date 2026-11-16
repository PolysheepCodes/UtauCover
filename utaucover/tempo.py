"""Tempo + beat-grid estimation so notes land on OpenUTAU's grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TempoInfo:
    bpm: float
    first_downbeat: float  # seconds into the audio of a bar line
    beats_per_bar: int = 4
    confidence: float = 0.0

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    def lead_in(self) -> float:
        """Seconds of silence to put before the audio so its bar lines land on OpenUTAU bar lines."""
        bar = self.beat_s * self.beats_per_bar
        return float((-self.first_downbeat) % bar)


def estimate_tempo(path, bpm_hint: float | None = None, beats_per_bar: int = 4, progress=None,
                   vocal_path=None) -> TempoInfo:
    """Constant-tempo estimate. The BPM is refined on the whole song by phase coherence of the
    onset envelope at the beat, eighth and sixteenth-note rates (robust to accents that move
    between on- and off-beats), then snapped to a whole number when that fits as well."""
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    hop = 256
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    t_on = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop)
    w_on = np.maximum(0.0, onset - np.median(onset))
    wsum = float(np.sum(w_on)) + 1e-9

    def z(b: float, k: int = 1) -> complex:
        return complex(np.sum(w_on * np.exp(2j * np.pi * t_on * k * b / 60.0)))

    def score(b: float) -> float:
        return (abs(z(b, 1)) + abs(z(b, 2)) + abs(z(b, 4))) / wsum

    if bpm_hint:
        bpm = float(bpm_hint)
    else:
        est = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=hop,
                                                         start_bpm=120))[0])
        coarse = np.arange(est - 4.0, est + 4.0, 0.05)
        best = float(coarse[int(np.argmax([score(b) for b in coarse]))])
        fine = np.arange(best - 0.1, best + 0.1, 0.005)
        fs = [score(b) for b in fine]
        best = float(fine[int(np.argmax(fs))])
        top = max(fs)
        rb = float(round(best))
        bpm = rb if score(rb) >= 0.9 * top else round(best, 2)
    period = 60.0 / bpm
    # beat phase: the eighth-note grid is robust; pick which of its two offsets is the beat
    z2 = z(bpm, 2)
    off8 = (np.angle(z2) / (2 * np.pi)) * (period / 2) % (period / 2)
    # Instrumentals often accent off-beats; sung syllables mostly start on the beat, so let the
    # vocal stem (when given) break the on-beat / off-beat tie.
    z1 = z(bpm, 1) / wsum
    if vocal_path is not None:
        try:
            yv, _ = librosa.load(str(vocal_path), sr=sr, mono=True)
            ov = librosa.onset.onset_strength(y=yv, sr=sr, hop_length=hop, aggregate=np.median)
            tv = librosa.frames_to_time(np.arange(len(ov)), sr=sr, hop_length=hop)
            wv = np.maximum(0.0, ov - np.median(ov))
            z1 = z1 + complex(np.sum(wv * np.exp(2j * np.pi * tv * bpm / 60.0))) / (np.sum(wv) + 1e-9)
        except Exception:
            pass
    cands = [off8, off8 + period / 2]
    phase = float(max(cands, key=lambda o: (z1 * np.exp(-2j * np.pi * o / period)).real))
    phase = (phase - 0.015) % period  # onset envelope peaks ~15 ms after the actual attack
    confidence = float(score(bpm) / 3)

    # Downbeat: which beat of the bar gets the kick drum and the chord changes.
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[freqs < 150].sum(axis=0)
    low_on = np.maximum(0, np.diff(low, prepend=low[:1]))
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=sr, hop_length=hop)
    dur = len(y) / sr
    grid = np.arange(phase, dur - period, period)
    frames = np.clip(librosa.time_to_frames(grid, sr=sr, hop_length=hop), 0, S.shape[1] - 1)
    half = max(1, int(period / 2 * sr / hop))
    # chord change at each beat: chroma just after the beat vs just before it
    after = np.stack([chroma[:, f:f + half].mean(axis=1) for f in frames], axis=1)
    before = np.stack([chroma[:, max(0, f - half):max(1, f)].mean(axis=1) for f in frames], axis=1)
    norm = lambda m: m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-9)
    change = 1 - np.sum(norm(after) * norm(before), axis=0)
    kick = low_on[frames]
    kick = kick / (kick.mean() + 1e-9)
    change = change / (change.mean() + 1e-9)
    strength = kick + 1.5 * change
    scores = [strength[k::beats_per_bar].mean() if len(strength[k::beats_per_bar]) else 0 for k in range(beats_per_bar)]
    k = int(np.argmax(scores))
    first_downbeat = phase + k * period
    if progress:
        progress(f"[tempo] {bpm:g} BPM, first bar line at {first_downbeat:.2f}s (fit {confidence:.0%})")
    return TempoInfo(bpm, float(first_downbeat), beats_per_bar, confidence)
