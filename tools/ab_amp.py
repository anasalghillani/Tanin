"""A/B the amp's inter-stage filters (audit item 12) by ear.

Renders the same DI through every amp model twice -- A = old voicing
(inter-stage filters off), B = new voicing (on) -- and writes WAV pairs you
can flip between in any player.

    .venv\\Scripts\\python.exe tools\\ab_amp.py                 # built-in riff
    .venv\\Scripts\\python.exe tools\\ab_amp.py my_di.wav        # your own DI

Tanin's recorder writes a dry DI file next to every wet recording; that is
the best input, because it is your guitar and your pickups.
Output goes to recordings\\ab\\<Model>_A_old.wav and <Model>_B_new.wav.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import amprack  # noqa: E402,F401  (sets up the package the same way the app does)
from amprack.dsp.blocks import AMP_MODELS, Amp  # noqa: E402
from amprack.dsp.chain import Chain  # noqa: E402

SR = 48000
BLOCK = 256


def builtin_riff() -> np.ndarray:
    """Palm-muted low E chugs, a power chord, then an open chord ringing out."""
    def note(f, dur, amp, decay):
        t = np.arange(int(dur * SR)) / SR
        y = sum(w * np.sin(2 * math.pi * f * h * t + h * 0.7)
                for h, w in enumerate((1.0, 0.6, 0.42, 0.25, 0.18, 0.12, 0.08), 1))
        return amp * y * np.exp(-t * decay) * (1 - np.exp(-t * 400.0)) / 2.6

    parts = [note(82.41, 0.18, 0.5, 18.0) for _ in range(8)]
    parts.append(note(82.41, 1.2, 0.4, 2.0) + note(123.47, 1.2, 0.35, 2.0))
    open_chord = sum(note(f, 2.5, 0.25, 1.2)
                     for f in (82.41, 123.47, 164.81, 207.65, 246.94, 329.63))
    parts.append(open_chord)
    return np.concatenate(parts)


def load_di(path: str) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x


def render(x: np.ndarray, model: str, interstage: bool) -> np.ndarray:
    Amp.interstage = interstage
    c = Chain()
    c.prepare(SR, BLOCK)
    c["amp"].model = model
    c["amp"].gain = 0.6
    n = (len(x) // BLOCK) * BLOCK
    out = np.zeros((2, n))
    for i in range(0, n, BLOCK):
        out[:, i:i + BLOCK] = c.process(x[i:i + BLOCK].copy())
    return out


def main() -> int:
    import soundfile as sf
    x = load_di(sys.argv[1]) if len(sys.argv) > 1 else builtin_riff()
    out_dir = os.path.join(ROOT, "recordings", "ab")
    os.makedirs(out_dir, exist_ok=True)
    for model in AMP_MODELS:
        for tag, on in (("A_old", False), ("B_new", True)):
            y = render(x, model, on)
            path = os.path.join(out_dir, f"{model}_{tag}.wav")
            sf.write(path, y.T.astype(np.float32), SR, subtype="FLOAT")
            print(f"wrote {path}")
    Amp.interstage = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
