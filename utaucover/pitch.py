"""Frame-level pitch (f0) tracking for a monophonic vocal stem, using CREPE (torchcrepe)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000
HOP_S = 0.01
HOP = int(SR * HOP_S)


@dataclass
class F0Track:
    times: np.ndarray        # frame centre times (s)
    hz: np.ndarray           # f0 in Hz (nan where unvoiced)
    midi: np.ndarray         # f0 as fractional MIDI note (nan where unvoiced)
    periodicity: np.ndarray  # CREPE confidence 0..1
    rms_db: np.ndarray       # frame loudness, dBFS
    voiced: np.ndarray       # bool mask after cleanup
    hop_s: float = HOP_S

    def frame(self, t: float) -> int:
        return int(np.clip(round(t / self.hop_s), 0, len(self.times) - 1))

    def save(self, path):
        np.savez_compressed(path, times=self.times, hz=self.hz, midi=self.midi,
                            periodicity=self.periodicity, rms_db=self.rms_db, voiced=self.voiced)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["times"], d["hz"], d["midi"], d["periodicity"], d["rms_db"], d["voiced"])


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_mono(path, sr=SR) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


_FCPE = None


def _fcpe_track(y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """FCPE (fast context-based pitch estimator): ~100x faster than CREPE-full on a Mac,
    same accuracy on separated vocals. Returns (hz, confidence) on our 10 ms frame grid."""
    global _FCPE
    import torch
    from torchfcpe import spawn_bundled_infer_model
    if _FCPE is None:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            _FCPE = spawn_bundled_infer_model(device="cpu")
    m = _FCPE
    with torch.no_grad():
        wav = torch.from_numpy(y)[None, :, None]
        mel = m.wav2mel(wav, SR)
        latent = m.model.forward(mel)
        conf = latent.max(dim=-1).values[0].numpy().astype(np.float64)
        cents = m.model.latent2cents_local_decoder(latent, threshold=0.0)
        hz = m.model.cent_to_f0(cents)[0, :, 0].numpy().astype(np.float64)
    src = np.linspace(0, 1, len(hz))
    dst = np.linspace(0, 1, n)
    return np.interp(dst, src, hz), np.interp(dst, src, conf)


def _crepe_track(y: np.ndarray, fmin: float, fmax: float, model: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torchcrepe
    audio = torch.from_numpy(y)[None]
    dev = _device()
    try:
        hz, per = torchcrepe.predict(audio, SR, HOP, fmin, fmax, model, decoder=torchcrepe.decode.viterbi,
                                     return_periodicity=True, batch_size=1024, device=dev, pad=True)
    except Exception:
        if dev == "cpu":
            raise
        hz, per = torchcrepe.predict(audio, SR, HOP, fmin, fmax, model, decoder=torchcrepe.decode.viterbi,
                                     return_periodicity=True, batch_size=512, device="cpu", pad=True)
    per = torchcrepe.filter.median(per, 3)
    return hz[0].cpu().numpy().astype(np.float64), per[0].cpu().numpy().astype(np.float64)


# confidence needed to call a frame voiced, per tracker
VOICING = {"fcpe": 0.3, "crepe": 0.45}


def extract_f0(path, fmin: float = 50.0, fmax: float = 1400.0, method: str = "fcpe", model: str = "full",
               voicing_threshold: float | None = None, silence_db: float = -45.0, progress=None) -> F0Track:
    y = load_mono(path)
    if progress:
        progress(f"[pitch] tracking {len(y) / SR:.0f}s of audio with {method.upper()}")
    if method == "crepe":
        hz, per = _crepe_track(y, fmin, fmax, model)
    else:
        hz, per = _fcpe_track(y, len(y) // HOP + 1)
        hz = np.where((hz >= fmin) & (hz <= fmax), hz, 0.0)
    if voicing_threshold is None:
        voicing_threshold = VOICING.get(method, 0.45)

    n = len(hz)
    # loudness per frame, aligned with crepe's centred frames
    pad = np.pad(y, (512, 512))
    frames = np.lib.stride_tricks.sliding_window_view(pad, 1024)[::HOP][:n]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-9)
    if len(rms_db) < n:
        rms_db = np.pad(rms_db, (0, n - len(rms_db)), constant_values=-120)

    # Loudness gate relative to the stem's own level, so quiet mixes still work.
    loud_ref = np.percentile(rms_db[rms_db > -100], 95) if np.any(rms_db > -100) else -20
    gate = max(silence_db, loud_ref - 38)
    voiced = (per >= voicing_threshold) & (rms_db >= gate)
    voiced = clean_mask(voiced, fill_gap=4, min_run=5)

    midi = np.full(n, np.nan)
    ok = voiced & (hz > 0)
    midi[ok] = 69 + 12 * np.log2(hz[ok] / 440.0)
    hz_out = np.where(ok, hz, np.nan)
    times = np.arange(n) * HOP_S
    return F0Track(times, hz_out, midi, per, rms_db, ok)


def clean_mask(mask: np.ndarray, fill_gap: int, min_run: int) -> np.ndarray:
    """Fill unvoiced gaps shorter than fill_gap frames, then drop voiced runs shorter than min_run."""
    m = mask.copy()
    runs = _runs(m)
    for v, a, b in runs:
        if not v and (b - a) < fill_gap and a > 0 and b < len(m):
            m[a:b] = True
    for v, a, b in _runs(m):
        if v and (b - a) < min_run:
            m[a:b] = False
    return m


def _runs(mask: np.ndarray):
    out = []
    if len(mask) == 0:
        return out
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            out.append((bool(mask[start]), start, i))
            start = i
    return out


def voiced_regions(track: F0Track) -> list[tuple[int, int]]:
    return [(a, b) for v, a, b in _runs(track.voiced) if v]


def tuning_offset(track: F0Track) -> float:
    """Global detune of the recording vs A440, in semitones (-0.5..0.5), from stable frames."""
    m = track.midi[track.voiced]
    if len(m) < 50:
        return 0.0
    d = np.diff(track.midi)
    stable = track.voiced[1:] & (np.abs(np.nan_to_num(d, nan=9)) < 0.05)
    vals = track.midi[1:][stable]
    if len(vals) < 50:
        vals = m
    ang = 2 * np.pi * (vals - np.round(vals))
    off = np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi)
    return float(off)
