"""CPU benchmark: the real audio callback with three instruments playing.

Phase 1 of the suite plan asks for this number before any feature work: how
much of each block's deadline the callback uses with three instruments on at
once, at 128 and 256-sample buffers.

The three instruments are guitar (every effect on), vocals (Hard Tune
preset: auto-tune, de-esser, comp, delay, reverb) and a stand-in for the piano
(a second fully loaded guitar chain, until the real synth exists).

Run:  .venv\\Scripts\\python.exe tools\\bench_instruments.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import amprack  # noqa: E402,F401  (sets the ASIO switch before sounddevice)
from amprack.engine import Engine  # noqa: E402
from amprack.instruments import GuitarInstrument  # noqa: E402
from amprack.instruments.vocals import apply_vocal_preset  # noqa: E402
from amprack.mixer import Mixer  # noqa: E402

SR = 48000


class _PianoStandIn(GuitarInstrument):
    key = "piano"
    label = "Piano (stand-in)"

    def __init__(self):
        super().__init__(input_channel=0)


def build_engine(block: int, instruments: int = 3) -> Engine:
    e = Engine(tempfile.mkdtemp(prefix="tanin-bench-"))
    if instruments >= 3:
        e.instruments.append(_PianoStandIn())
        e.mixer = Mixer([i.key for i in e.instruments])
    for inst in e.instruments:
        if inst.key in ("guitar", "piano"):
            for k in ("gate", "comp", "drive", "mod", "delay", "reverb"):
                inst.chain[k].enabled = True
    e.vocals.enabled = instruments >= 2
    apply_vocal_preset(e.vocals, "Hard Tune")
    e.samplerate = SR
    e.active_blocksize = block
    e._prepare_buffers(2)
    return e


def measure(block: int, seconds: float = 4.0, instruments: int = 3) -> dict:
    """Mean and 99th-percentile callback time as a fraction of the deadline."""
    e = build_engine(block, instruments)
    t = np.arange(int(seconds * SR)) / SR
    guitar = 0.3 * np.sin(2 * math.pi * 110 * t) * np.exp(-(t % 1.0) * 3)
    voice = 0.2 * sum(w * np.sin(2 * math.pi * 207.0 * h * t)
                      for h, w in enumerate((1, .5, .35, .2, .12), 1))
    indata = np.zeros((block, 2), dtype=np.float32)
    outdata = np.zeros((block, 2), dtype=np.float32)
    n_blocks = len(t) // block
    times = []
    for i in range(n_blocks):
        s = slice(i * block, (i + 1) * block)
        indata[:, 0] = guitar[s]
        indata[:, 1] = voice[s]
        t0 = time.perf_counter()
        e._callback(indata, outdata, block, None, None)
        times.append(time.perf_counter() - t0)
    times = np.asarray(times[20:])            # skip warm-up
    deadline = block / SR
    return {
        "block": block,
        "deadline_ms": deadline * 1000,
        "mean": float(times.mean() / deadline),
        "p99": float(np.percentile(times, 99) / deadline),
        "errors": e.callback_errors,
        "last_exc": e.last_callback_exc,
    }


def main() -> int:
    print("Tanin CPU benchmark: guitar + vocals (Hard Tune) + piano stand-in")
    for block in (128, 256):
        r = measure(block)
        print(f"  {block:>4} samples ({r['deadline_ms']:.1f} ms): "
              f"mean {r['mean']*100:5.1f}%   p99 {r['p99']*100:5.1f}%   "
              f"errors {r['errors']} {r['last_exc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
