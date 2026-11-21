"""Acoustic simulator: makes a magnetic pickup sound like an acoustic guitar.

Two parts, as in the pedals that do this:

1.  A corrective EQ. A magnetic pickup has a resonant peak around 2.5-4 kHz
    and little real top above it, and too much low-mid. That is flattened.
2.  A body. An acoustic's sound is mostly its body resonating: the air
    resonance of the sound hole around 90-120 Hz, the top plate's first
    modes around 180-250 Hz and 400 Hz, and a dense mass of short-lived
    wood modes above. The body is built here as an impulse response from
    those modes (each a decaying sine) and convolved with the corrected
    signal, so a pick attack excites a body rather than an EQ curve.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..core import Biquad, SmoothedParam, clampf
from .base import Effect, P
from .filters import _level_gain

# model: (air Hz, top mode Hz, second top Hz, body size, brightness, seed)
BODIES = {
    "Dreadnought": (98.0, 195.0, 390.0, 1.0, 0.5, 11),
    "Jumbo": (90.0, 180.0, 360.0, 1.15, 0.45, 23),
    "Parlor": (122.0, 240.0, 470.0, 0.75, 0.65, 37),
    "Nylon": (100.0, 205.0, 410.0, 0.9, 0.3, 41),
}

IR_LEN = 2048   # 43 ms at 48 kHz: the body, not the room


@lru_cache(maxsize=32)
def body_ir(model: str, sr: float) -> np.ndarray:
    air, top1, top2, size, bright, seed = BODIES[model]
    rng = np.random.default_rng(seed)
    t = np.arange(IR_LEN) / sr
    ir = np.zeros(IR_LEN)
    ir[0] = 1.0                                  # the direct string sound
    # The three big, named resonances: long and strong.
    for f, amp, tau in ((air, 0.9, 0.045 * size), (top1, 0.7, 0.03 * size),
                        (top2, 0.45, 0.02 * size)):
        ir += amp * np.exp(-t / tau) * np.sin(2 * math.pi * f * t) * (2 * math.pi * f / sr)
    # Dense wood modes: log-spread 500 Hz .. 6 kHz, shorter as they rise.
    for f in np.geomspace(500.0, 6000.0, 60):
        f *= 1.0 + rng.uniform(-0.06, 0.06)
        tau = 0.012 * (500.0 / f) ** 0.5 * size
        amp = 0.25 * (f / 500.0) ** (bright - 0.75) * rng.uniform(0.5, 1.0)
        ph = rng.uniform(0, 2 * math.pi)
        ir += amp * np.exp(-t / tau) * np.sin(2 * math.pi * f * t + ph) * (2 * math.pi * f / sr)
    ir *= np.hanning(IR_LEN * 2)[IR_LEN:] ** 0.5
    return ir / np.sqrt(np.sum(ir * ir))


@lru_cache(maxsize=32)
def _kernel(model: str, sr: float, size: int) -> np.ndarray:
    k = np.fft.rfft(body_ir(model, sr), size)
    k.flags.writeable = False
    return k


class AcousticSim(Effect):
    TYPE = "acoustic"
    LABEL = "Acoustic Sim"
    CATEGORY = "Utility"
    MODELS = tuple(BODIES)
    PARAMS = (
        P("body", "Body", 0.0, 1.0, 0.6, aliases=("body", "resonance", "bass",
                                                  "low", "depth", "size")),
        P("top", "Top", 0.0, 1.0, 0.55, aliases=("top", "treble", "high",
                                                 "sparkle", "brightness", "tone")),
        P("mix", "Mix", 0.0, 1.0, 1.0, aliases=("blend", "drywet", "effect")),
        P("level", "Level", 0.0, 1.0, 0.5, aliases=("volume", "output", "out", "vol")),
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        c = self.channels
        size = 1
        while size < max_block + IR_LEN:
            size <<= 1
        self._fft = size
        self._kernels = {m: _kernel(m, sr, size) for m in BODIES}
        self._hist = np.zeros((c, size))
        self._pk = Biquad("peak", 3200.0, 1.6, -7.0)      # pickup resonance out
        self._lm = Biquad("peak", 300.0, 0.8, -4.0)       # low-mid mud out
        self._air = Biquad("highshelf", 6000.0, 0.707, 6.0)
        self._lo = Biquad("lowshelf", 140.0, 0.707, 0.0)
        self._hp = Biquad("highpass", 60.0, 0.707)
        for f in (self._pk, self._lm, self._air, self._lo, self._hp):
            f.prepare(sr, c)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        self._hist[...] = 0.0
        for f in (self._pk, self._lm, self._air, self._lo, self._hp):
            f.reset()

    def _process(self, x, n):
        y = self._hp.process(self._lm.process(self._pk.process(x)))
        top = clampf(self.top, 0.0, 1.0)
        self._air.set(gain_db=2.0 + 10.0 * top)
        y = self._air.process(y)
        h = self._hist
        h[:, :-n] = h[:, n:]
        h[:, -n:] = y
        k = self._kernels.get(self.model) if self.model in self._kernels \
            else self._kernels["Dreadnought"]
        body = np.fft.irfft(np.fft.rfft(h, axis=-1) * k, self._fft, axis=-1)[:, -n:]
        b = clampf(self.body, 0.0, 1.0)
        self._lo.set(gain_db=(b - 0.5) * 10.0)
        wet = self._lo.process(y * (1.0 - b) * 0.6 + body * (0.4 + b))
        self._out.set(_level_gain(self.level) * 0.9)
        wet = wet * self._out.ramp(n)
        m = clampf(self.mix, 0.0, 1.0)
        if m >= 0.999:
            return wet
        return x * (1.0 - m) + wet * m
