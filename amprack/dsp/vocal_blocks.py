"""Vocal-only effect blocks: de-esser, vocal EQ, saturation, doubler.

Same conventions as ``blocks.py``: mono blocks take and return (n,), stereo
blocks (2, n); ``enabled = False`` is bit-exact; nothing allocates beyond the
block; every block is block-size invariant (state carries across blocks
sample by sample, never "per block").
"""

from __future__ import annotations

import math

import numpy as np

from .blocks import Block
from .core import Biquad, DelayLine, LFO, OnePole, SmoothedParam, clampf, db2lin


class DeEsser(Block):
    """Turns down harsh "s" and "sh" sounds only while they happen.

    A high-pass copy of the voice (the sibilance band) is measured; when it
    goes over the threshold, only that band is turned down, by at most
    ``range_db``. ``x - hp`` is used as the low band, so with no reduction
    the two bands add straight back to the input.
    """

    name = "deesser"

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.freq = 6000.0
        self.threshold_db = -30.0
        self.range_db = 8.0
        self.reduction_db = 0.0        # telemetry
        self._hp = Biquad("highpass", 6000.0, 0.707)
        self._env = OnePole(1.5)
        self._rel = OnePole(60.0)

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._hp.prepare(sr, 1)
        self._env.prepare(sr, 1)
        self._rel.prepare(sr, 1)
        self.reset()

    def reset(self):
        self._hp.reset()
        self._env.reset()
        self._rel.reset(1.0)

    def _process(self, x, n):
        self._hp.set(f0=clampf(self.freq, 3000.0, 12000.0))
        hp = self._hp.process(x)
        env = self._env.process(np.abs(hp))
        thr = db2lin(clampf(self.threshold_db, -60.0, 0.0))
        floor = db2lin(-clampf(self.range_db, 0.0, 24.0))
        need = np.clip(thr / np.maximum(env, 1e-12), floor, 1.0)
        # Instant attack, 60 ms release (same min-of-two trick as the limiter).
        g = np.minimum(need, self._rel.process(need))
        self.reduction_db = -20.0 * math.log10(max(float(g.min()), 1e-9)) if n else 0.0
        return x - hp + hp * g


class VocalEQ(Block):
    """Low cut (rumble, plosives), presence (clarity) and air (sparkle)."""

    name = "eq"

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.low_cut_hz = 90.0
        self.presence_db = 2.0       # peak at 3.5 kHz
        self.air_db = 1.5            # shelf above 10 kHz
        self._hp1 = Biquad("highpass", 90.0, 0.541)
        self._hp2 = Biquad("highpass", 90.0, 1.307)   # 4th-order Butterworth
        self._pres = Biquad("peak", 3500.0, 0.9, 2.0)
        self._air = Biquad("highshelf", 10000.0, 0.707, 1.5)

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        for f in (self._hp1, self._hp2, self._pres, self._air):
            f.prepare(sr, 1)

    def reset(self):
        for f in (self._hp1, self._hp2, self._pres, self._air):
            f.reset()

    def _process(self, x, n):
        fc = clampf(self.low_cut_hz, 20.0, 400.0)
        self._hp1.set(f0=fc)
        self._hp2.set(f0=fc)
        self._pres.set(gain_db=clampf(self.presence_db, -12.0, 12.0))
        self._air.set(gain_db=clampf(self.air_db, -12.0, 12.0))
        y = self._hp2.process(self._hp1.process(x))
        return self._air.process(self._pres.process(y))


class Saturation(Block):
    """Gentle tape/tube warmth. Odd+even harmonics from an offset tanh."""

    name = "saturation"

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.drive_db = 6.0
        self.mix = 0.5
        self._dc = OnePole(20.0)     # removes the offset's DC, slow enough to be inaudible

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._dc.prepare(sr, 1)
        self.reset()

    def reset(self):
        self._dc.reset()

    def _process(self, x, n):
        g = db2lin(clampf(self.drive_db, 0.0, 24.0))
        bias = 0.1
        y = (np.tanh(x * g + bias) - math.tanh(bias)) / g
        y = y - self._dc.process(y)
        m = clampf(self.mix, 0.0, 1.0)
        return x + (y - x) * m


class Doubler(Block):
    """Stereo double-tracking: two slightly late, slightly detuned copies,
    one panned left and one right, like singing the part twice."""

    name = "doubler"
    stereo = True

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.mix = 0.5
        self.width = 1.0
        self.detune = 0.5            # 0..1: how much the copies wander
        self._line = None
        self._lfo_l = LFO()
        self._lfo_r = LFO()

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._line = DelayLine(int(sr * 0.06) + max_block + 16, channels=1)
        self._line.prepare(max_block)
        self._lfo_l.prepare(sr, max_block)
        self._lfo_r.prepare(sr, max_block)
        self._lfo_r.phase = 0.37

    def reset(self):
        if self._line is not None:
            self._line.reset()

    def _process(self, x, n):
        mono = (x[0] + x[1]) * 0.5
        sr = self.sr
        dep = clampf(self.detune, 0.0, 1.0) * 0.0015 * sr
        dl = 0.014 * sr + self._lfo_l.block(0.45, n) * dep
        dr = 0.022 * sr + self._lfo_r.block(0.31, n) * dep
        np.maximum(dl, n + 1, out=dl)          # read-before-write contract
        np.maximum(dr, n + 1, out=dr)
        cl = self._line.read_modulated(dl, n)[0]
        cr = self._line.read_modulated(dr, n)[0]
        self._line.write(mono[None, :])
        m = clampf(self.mix, 0.0, 1.0) * 0.7
        w = clampf(self.width, 0.0, 1.0)
        # Each copy mostly on its own side; width 0 puts both in the middle.
        l_gain, r_gain = 0.5 + 0.5 * w, 0.5 - 0.5 * w
        out = np.empty_like(x)
        out[0] = x[0] + m * (cl * l_gain + cr * r_gain)
        out[1] = x[1] + m * (cr * l_gain + cl * r_gain)
        return out
