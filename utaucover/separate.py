"""Stem separation via python-audio-separator (UVR models: RoFormer, MDX, Demucs).

Produces, inside ``stems_dir``:
    vocals.wav        all vocals            (BS-RoFormer)
    instrumental.wav  everything else
    lead.wav          main vocal            (Mel-RoFormer karaoke model, run on vocals.wav)
    backing.wav       backing / harmony vocals
    lead_dry.wav      lead with reverb removed (optional, used only for analysis)
    drums.wav bass.wav other.wav [guitar.wav piano.wav]   (optional, Demucs)

Results are cached: a manifest records the input fingerprint and model used for each stem.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from .paths import MODEL_DIR, file_fingerprint

Progress = Callable[[str], None]


def ensure_pcm16(path: Path) -> bool:
    """Rewrite a WAV as a plain 16-bit PCM WAV if it's anything else. UtauV/OpenUTAU's audio reader rejects
    32-bit and 'extensible' WAVs ("Unsupported source encoding") and the track plays as silence, even though
    Finder plays the file fine. Returns True if the file was converted."""
    import os

    import soundfile as sf
    info = sf.info(str(path))
    if info.format == "WAV" and info.subtype == "PCM_16":
        return False
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    tmp = path.with_name(path.stem + ".pcm16.tmp.wav")
    sf.write(str(tmp), y, sr, format="WAV", subtype="PCM_16")
    os.replace(tmp, path)
    return True

# Quality presets. Rough timings for a 3-minute song on an M4 MacBook Air:
#   fast ~2-3 min, balanced ~5 min, best ~11 min (cached afterwards).
MODELS = {
    "vocals": {
        "best": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "balanced": "UVR-MDX-NET-Voc_FT.onnx",
        "fast": "UVR-MDX-NET-Voc_FT.onnx",
    },
    "karaoke": {
        "best": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "balanced": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "fast": "UVR_MDXNET_KARA_2.onnx",
    },
    "dereverb": {
        "best": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        "balanced": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        "fast": "UVR-DeEcho-DeReverb.pth",
    },
    "instruments": {
        "best": "htdemucs_ft.yaml",
        "balanced": "htdemucs_ft.yaml",
        "fast": "htdemucs.yaml",
        "six": "htdemucs_6s.yaml",
    },
}
QUALITIES = ("fast", "balanced", "best")

# For each job: which separator output (matched by the "(Stem)" tag in the file name) becomes which file.
# Karaoke models call the lead "Vocals" and everything else "Instrumental".
JOBS = {
    "vocals": {"vocals": "vocals", "instrumental": "instrumental", "other": "instrumental"},
    "karaoke": {"vocals": "lead", "instrumental": "backing", "other": "backing"},
    "dereverb": {"noreverb": "lead_dry", "no reverb": "lead_dry", "dry": "lead_dry"},
    "instruments": {"drums": "drums", "bass": "bass", "other": "other", "guitar": "guitar", "piano": "piano"},
}


class StemSeparator:
    def __init__(self, stems_dir: str | Path, quality: str = "best", progress: Progress | None = None):
        self.dir = Path(stems_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.quality = quality
        self.log = progress or (lambda m: None)
        self.manifest_path = self.dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        self._sep = None

    # ---------- cache ----------
    def _cached(self, job: str, src: Path, outputs: list[str], needs_wav_input: bool = False) -> bool:
        entry = self.manifest.get(job)
        if not entry:
            return False
        if entry.get("source") != file_fingerprint(src) or entry.get("model") != self._model(job):
            return False
        if needs_wav_input and entry.get("input") != "safe-wav":
            return False  # made from a compressed file before the decode fix: vocals leaked into the instrumental
        return all((self.dir / f"{o}.wav").exists() for o in outputs)

    def _record(self, job: str, src: Path, outputs: list[str], input_kind: str | None = None):
        self.manifest[job] = {"source": file_fingerprint(src), "model": self._model(job), "outputs": outputs}
        if input_kind:
            self.manifest[job]["input"] = input_kind
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def _model(self, job: str) -> str:
        q = self.quality if self.quality in MODELS[job] else "balanced"
        if job == "instruments" and getattr(self, "six_stems", False):
            q = "six"
        return MODELS[job][q]

    # ---------- running ----------
    def _separator(self):
        if self._sep is None:
            import logging
            from audio_separator.separator import Separator
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self._tmp = Path(tempfile.mkdtemp(prefix="utaucover_sep_"))
            # overlap=2 for RoFormer models: ~4x faster than the library default (8), same quality in practice
            self._sep = Separator(output_dir=str(self._tmp), model_file_dir=str(MODEL_DIR),
                                  output_format="WAV", log_level=logging.WARNING,
                                  mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                                               "batch_size": 1, "overlap": 2, "pitch_shift": 0},
                                  mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25,
                                              "batch_size": 1, "enable_denoise": False},
                                  demucs_params={"segment_size": "Default", "shifts": 1, "overlap": 0.25,
                                                 "segments_enabled": True})
            from .download import report_separator_downloads
            report_separator_downloads(self._sep, self.log)
        return self._sep

    def _run(self, job: str, src: Path, required: list[str]) -> dict[str, Path]:
        mapping = JOBS[job]
        model = self._model(job)
        # Songs are handed to the separator decoded and kept under full scale. Mastered mp3s often decode with
        # peaks above full scale (Here Come the Cats: 1.76x), and MDX models then fail to cancel the vocals out
        # of the instrumental (singing left 7.5 dB under the vocal instead of 27 dB). WAVs in 16/24-bit PCM
        # can't exceed full scale and go in as they are.
        safe = src.suffix.lower() in (".wav", ".wave") and self._pcm(src)
        needs_safe = not safe and model.endswith(".onnx")
        if self._cached(job, src, required, needs_wav_input=needs_safe):
            self.log(f"[stems] {job}: using cached result")
            out = {o: self.dir / f"{o}.wav" for o in self.manifest[job]["outputs"]}
            for f in out.values():  # stems cached by 0.6.1 can be 32-bit: make them playable in UtauV
                ensure_pcm16(f)
            return out
        run_src = src if safe else self._decoded(src)
        compressed = not safe
        have = (MODEL_DIR / model).exists()
        self.log(f"[stems] {job}: running {model}" + ("" if have else " (downloading the model, one time)"))
        sep = self._separator()
        for f in self._tmp.glob("*"):
            f.unlink()
        sep.load_model(model_filename=model)
        sep.separate(str(run_src))
        produced: dict[str, Path] = {}
        for f in self._tmp.glob("*.wav"):
            name = f.name.lower()
            for stem_tag, out in mapping.items():
                if f"({stem_tag})" in name:
                    dest = self.dir / f"{out}.wav"
                    if out in produced:  # e.g. "other" and "instrumental" both mapped; keep first
                        continue
                    shutil.move(str(f), dest)
                    produced[out] = dest
                    break
        for f in produced.values():
            ensure_pcm16(f)
        missing = [o for o in required if o not in produced]
        if missing:
            raise RuntimeError(f"{job} separation did not produce {missing}; got {[f.name for f in self._tmp.iterdir()]}")
        self._record(job, src, sorted(produced), "safe-wav" if compressed else None)
        return produced

    @staticmethod
    def _pcm(path: Path) -> bool:
        try:
            import soundfile as sf
            return sf.info(str(path)).subtype.startswith("PCM")
        except Exception:
            return False

    def _decoded(self, song: Path) -> Path:
        """The song as a 44.1 kHz stereo WAV (kept for the session, outside the separator's output folder)."""
        if getattr(self, "_decoded_path", None) and self._decoded_src == song:
            return self._decoded_path
        import librosa
        import soundfile as sf
        self._dec_dir = Path(tempfile.mkdtemp(prefix="utaucover_dec_"))
        y, sr = librosa.load(str(song), sr=44100, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        peak = float(np.abs(y).max()) if y.size else 0.0
        if peak > 0.98:  # bring overshooting peaks under full scale (a plain gain, no clipping)
            y = y * (0.98 / peak)
            self.log(f"[stems] the song peaks at {peak:.2f}x full scale; turned down {20 * np.log10(peak / 0.98):.1f} dB "
                     "before separation so the vocals cancel out of the instrumental")
        out = self._dec_dir / (song.stem[:60] + ".wav")
        sf.write(str(out), y.T, sr, format="WAV", subtype="PCM_16")  # 16-bit in -> 16-bit stems out
        self._decoded_path, self._decoded_src = out, song
        return out

    def separate(self, song: str | Path, backing_split: bool = True, dereverb: bool = False,
                 instruments: bool = False, six_stems: bool = False) -> dict[str, Path]:
        song = Path(song)
        self.six_stems = six_stems
        stems = self._run("vocals", song, ["vocals", "instrumental"])
        if backing_split:
            stems.update(self._run("karaoke", stems["vocals"], ["lead", "backing"]))
        if dereverb:
            src = stems.get("lead", stems["vocals"])
            stems.update(self._run("dereverb", src, ["lead_dry"]))
        if instruments:
            stems.update(self._run("instruments", stems["instrumental"], ["drums", "bass", "other"]))
        return stems

    def close(self):
        if getattr(self, "_tmp", None):
            shutil.rmtree(self._tmp, ignore_errors=True)
        if getattr(self, "_dec_dir", None):
            shutil.rmtree(self._dec_dir, ignore_errors=True)
