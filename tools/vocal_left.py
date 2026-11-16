"""How much singing is left in an instrumental file (judged by BS-RoFormer), over a time window."""
import logging, sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa
from audio_separator.separator import Separator
OUT = Path.home()/"Library/Application Support/UtauCover/bench/septest"
inst_path, vocal_path, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
db = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
y, sr = librosa.load(inst_path, sr=44100, mono=False, offset=t0, duration=t1 - t0)
clip = OUT / "inst_clip.wav"; sf.write(str(clip), y.T, sr)
v, _ = librosa.load(vocal_path, sr=44100, mono=True, offset=t0, duration=t1 - t0)
sep = Separator(output_dir=str(OUT), model_file_dir=str(Path.home()/"Library/Application Support/UtauCover/models"),
                output_format="WAV", log_level=logging.ERROR)
sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
outs = [OUT / o for o in sep.separate(str(clip))]
left, _ = librosa.load(str(next(o for o in outs if "(Vocals)" in o.name)), sr=44100, mono=True)
print(f"{Path(inst_path).parent.parent.name}/{Path(inst_path).name}: instrumental {db(y):.1f} dB, vocal stem {db(v):.1f} dB, "
      f"singing left in the instrumental {db(left):.1f} dB ({db(left) - db(v):+.1f} dB vs the vocal)")
