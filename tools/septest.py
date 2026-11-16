"""Compare vocal separators: how much singing is left in each model's instrumental (judged by BS-RoFormer)."""
import logging, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, librosa
from audio_separator.separator import Separator
OUT = Path.home()/"Library/Application Support/UtauCover/bench/septest"
MODELS_DIR = Path.home()/"Library/Application Support/UtauCover/models"
song, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
models = sys.argv[4:] or ["UVR-MDX-NET-Voc_FT.onnx", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"]
y, sr = librosa.load(song, sr=44100, mono=False, offset=t0, duration=t1 - t0)
clip = OUT / "clip.wav"; sf.write(str(clip), y.T, sr)
sep = Separator(output_dir=str(OUT), model_file_dir=str(MODELS_DIR), output_format="WAV", log_level=logging.ERROR,
                mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": 1, "overlap": 2, "pitch_shift": 0})
def run(model, path):
    sep.load_model(model_filename=model)
    outs = [OUT / o for o in sep.separate(str(path))]
    get = lambda tag: next(o for o in outs if f"({tag})" in o.name)
    return get("Instrumental"), get("Vocals")
db = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
judge = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
res = {}
for m in models:
    t = time.time(); inst, voc = run(m, clip); dt = time.time() - t
    res[m] = (inst, voc, dt)
for m, (inst, voc, dt) in res.items():
    _, left = run(judge, inst)
    v, _ = sf.read(str(voc), dtype="float32"); l, _ = sf.read(str(left), dtype="float32")
    print(f"{m:45} {dt:5.0f}s for {t1-t0:.0f}s of audio | vocal stem {db(v):6.1f} dB | singing left in its instrumental {db(l):6.1f} dB ({db(l) - db(v):+.1f} dB vs the vocal)")
