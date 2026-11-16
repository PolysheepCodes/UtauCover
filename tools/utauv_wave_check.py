"""Open WAV files with UtauV's own audio reader and report how much signal it gets (0 = it hears silence)."""
import sys, os, glob, tempfile, subprocess
HELPER = r'''
import sys, os
from clr_loader import get_coreclr
from pythonnet import set_runtime
app = "/Applications/UtauV.app/Contents/MacOS"
set_runtime(get_coreclr(entry_dll=os.path.join(app, "UtauV.dll"), dotnet_root=app))
import clr, System
sys.path.append(app); clr.AddReference("OpenUtau.Core")
asm = System.Reflection.Assembly.LoadFrom(os.path.join(app, "OpenUtau.Core.dll"))
wave = asm.GetType("OpenUtau.Core.Format.Wave")
F = System.Reflection.BindingFlags
if len(sys.argv) > 1 and sys.argv[1] == "--list":
    for m in wave.GetMethods(F.Public | F.Static): print(m.ReturnType, m.Name, [str(p.ParameterType) for p in m.GetParameters()])
    sys.exit()
open_file = wave.GetMethod("OpenFile")
stereo = wave.GetMethod("GetStereoSamples")
import numpy as np
for p in sys.argv[1:]:
    try:
        reader = open_file.Invoke(None, [p])
        fmt = str(reader.WaveFormat)
        samples = stereo.Invoke(None, [reader])
        n = samples.Length
        arr = np.fromiter((samples[i] for i in range(0, n, 97)), dtype=np.float32, count=len(range(0, n, 97)))
        print(f"{os.path.basename(p):24} {fmt:40} -> UtauV reads {n} samples, peak {np.abs(arr).max() if n else 0:.3f}")
    except Exception as e:
        print(f"{os.path.basename(p):24} FAILED: {e}")
'''
with tempfile.TemporaryDirectory() as home:
    script = os.path.join(home, "h.py"); open(script, "w").write(HELPER)
    subprocess.run([sys.executable, script, *sys.argv[1:]], env=dict(os.environ, HOME=home))
