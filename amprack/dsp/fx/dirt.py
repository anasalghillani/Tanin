"""Overdrive, distortion, fuzz and boost pedals.

Each model is a small circuit described as data: what the input network
does before the clipper (the high-pass that sets how much bass gets
clipped, the op-amp's gain-bandwidth roll-off), how it clips, and what the
tone circuit after it does. Clipping runs 8x oversampled.
"""

from __future__ import annotations

import math

import numpy as np

from ..core import Biquad, DCBlocker, OnePole, Oversampler, SmoothedParam, clampf, db2lin
from .base import Effect, P, PC
from .filters import _level_gain

# --------------------------------------------------------------------------
# Clipping curves (vectorised, memoryless)
# --------------------------------------------------------------------------


def clip_soft_diode(v):
    """Diodes in an op-amp's feedback loop: a soft knee that never flattens."""
    return v / (1.0 + np.abs(v) ** 2.5) ** 0.4


def clip_asym(v):
    """One diode on one side, two on the other: even harmonics."""
    return np.where(v >= 0.0, np.tanh(v), 1.45 * np.tanh(v / 1.45))


def clip_hard_diode(v):
    """Silicon diodes to ground after an op-amp: a sharp knee."""
    return v / (1.0 + np.abs(v) ** 5.0) ** 0.2


def clip_led(v):
    """Higher forward voltage, harder corner: louder and more open."""
    return v / (1.0 + np.abs(v / 1.6) ** 6.0) ** (1.0 / 6.0) / 1.6


CLIPPERS = {
    "soft": clip_soft_diode,
    "asym": clip_asym,
    "hard": clip_hard_diode,
    "led": clip_led,
}

# model: dict of circuit facts.
#   hp:      input high-pass (Hz) -- how much bass reaches the clipper
#   gain:    (min, max) linear clipper drive
#   gbw:     op-amp gain-bandwidth (Hz) or 0: pre-clip low-pass = gbw / gain
#   clip:    curve name, or "transistor" / "muff" / "zone"
#   dry:     clean signal the op-amp passes alongside the clipped one
#   tone:    "lp" (sweeps a low-pass), "ts" (mid-hump tone), "scoop"
#            (low-pass/high-pass blend: mid scoop), "rat" (filter knob),
#            "zone" (active 3-band with sweepable mid)
#   tone_lo, tone_hi: the tone circuit's range (Hz)
#   out:     output trim so Level 0.5 is about unity loudness
DIRT_MODELS = {
    "Green":      dict(hp=720.0, gain=(2.0, 110.0), gbw=0.0, clip="soft", dry=0.5,
                       tone="ts", tone_lo=700.0, tone_hi=5500.0, out=0.767),
    "Asym":       dict(hp=500.0, gain=(2.0, 140.0), gbw=0.0, clip="asym", dry=0.5,
                       tone="ts", tone_lo=900.0, tone_hi=6000.0, out=0.537),
    "Blues":      dict(hp=200.0, gain=(1.5, 70.0), gbw=0.0, clip="soft", dry=0.15,
                       tone="lp", tone_lo=1500.0, tone_hi=9000.0, out=0.525),
    "Distortion": dict(hp=180.0, gain=(6.0, 700.0), gbw=0.0, clip="hard", dry=0.0,
                       tone="scoop", tone_lo=450.0, tone_hi=1700.0, out=0.363),
    "Rodent":     dict(hp=60.0, gain=(4.0, 2000.0), gbw=900000.0, clip="hard", dry=0.0,
                       tone="rat", tone_lo=480.0, tone_hi=16000.0, out=0.196),
    "Zone":       dict(hp=90.0, gain=(30.0, 4000.0), gbw=0.0, clip="zone", dry=0.0,
                       tone="zone", tone_lo=200.0, tone_hi=5000.0, out=0.186),
    "Muff":       dict(hp=100.0, gain=(8.0, 900.0), gbw=0.0, clip="muff", dry=0.0,
                       tone="scoop", tone_lo=400.0, tone_hi=1300.0, out=0.306),
    "Ge Fuzz":    dict(hp=40.0, gain=(4.0, 120.0), gbw=0.0, clip="transistor", dry=0.0,
                       tone="lp", tone_lo=2000.0, tone_hi=7000.0, out=0.349, sag=0.6),
    "Si Fuzz":    dict(hp=60.0, gain=(8.0, 400.0), gbw=0.0, clip="transistor", dry=0.0,
                       tone="lp", tone_lo=2500.0, tone_hi=9000.0, out=0.282, sag=0.35),
    "Gated Fuzz": dict(hp=250.0, gain=(20.0, 800.0), gbw=0.0, clip="transistor", dry=0.0,
                       tone="lp", tone_lo=1800.0, tone_hi=6000.0, out=0.309, sag=1.0,
                       gate=1.0),
}


class Dirt(Effect):
    TYPE = "dirt"
    LABEL = "Drive"
    CATEGORY = "Dirt"
    MODELS = tuple(DIRT_MODELS)
    PARAMS = (
        P("gain", "Gain", 0.0, 1.0, 0.5, aliases=("drive", "distortion", "dist",
                                                  "fuzz", "sustain", "overdrive",
                                                  "attack", "boost", "saturation")),
        P("tone", "Tone", 0.0, 1.0, 0.5, aliases=("filter", "treble", "high", "hi",
                                                  "color", "colour", "eq")),
        P("level", "Level", 0.0, 1.0, 0.5, aliases=("volume", "output", "out", "vol")),
        P("bass", "Bass", 0.0, 1.0, 0.5, aliases=("low", "lows", "body", "lo")),
        P("mid", "Mid", 0.0, 1.0, 0.5, aliases=("middle", "mids", "midrange",
                                                "midlevel")),
        P("mid_freq", "Mid Freq", 0.0, 1.0, 0.5, aliases=("midfreq", "frequency",
                                                          "freq", "midfrequency")),
        P("bias", "Bias", 0.0, 1.0, 0.5, aliases=("bias", "starve", "voltage",
                                                  "gate", "texture")),
        P("mix", "Mix", 0.0, 1.0, 1.0, aliases=("blend", "drywet")),
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        c = self.channels
        self._oss = [Oversampler() for _ in range(c)]
        for o in self._oss:
            o.prepare(sr, max_block)
        self._hp = Biquad("highpass", 700.0, 0.707)
        self._gbw = Biquad("lowpass", 8000.0, 0.707)
        self._pre_peak = Biquad("peak", 1000.0, 0.8, 0.0)
        self._inter = Biquad("lowpass", 4000.0, 0.707)
        self._inter_hp = Biquad("highpass", 100.0, 0.707)
        self._lp = Biquad("lowpass", 4000.0, 0.707)
        self._lp2 = Biquad("lowpass", 9000.0, 0.707)
        self._shp = Biquad("highpass", 1000.0, 0.707)
        self._lo = Biquad("lowshelf", 120.0, 0.707)
        self._mid = Biquad("peak", 800.0, 1.0)
        self._hi = Biquad("highshelf", 3200.0, 0.707)
        for f in (self._hp, self._gbw, self._pre_peak, self._lp, self._lp2,
                  self._shp, self._lo, self._mid, self._hi):
            f.prepare(sr, c)
        # Inter-stage filters run at the oversampled rate.
        for f in (self._inter, self._inter_hp):
            f.prepare(sr * 8, 1)
        self._inters = [(Biquad("lowpass", 4000.0, 0.707), Biquad("highpass", 100.0, 0.707))
                        for _ in range(c)]
        for lp, hp in self._inters:
            lp.prepare(sr * 8, 1)
            hp.prepare(sr * 8, 1)
        self._dc = DCBlocker()
        self._dc.prepare(sr, c)
        self._env = OnePole(25.0)
        self._env.prepare(sr, 1)
        self._gate_env = OnePole(5.0)
        self._gate_env.prepare(sr, 1)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)
        self._mixs = SmoothedParam(1.0)
        self._mixs.prepare(max_block)

    def reset(self):
        for o in self._oss:
            o.reset()
        for f in (self._hp, self._gbw, self._pre_peak, self._lp, self._lp2,
                  self._shp, self._lo, self._mid, self._hi):
            f.reset()
        for lp, hp in self._inters:
            lp.reset()
            hp.reset()
        self._dc.reset()
        self._env.reset()
        self._gate_env.reset()

    def _clip(self, up, ch, spec, gain, bias_s, env_up):
        kind = spec["clip"]
        if kind in CLIPPERS:
            return CLIPPERS[kind](up * gain)
        if kind == "muff":
            # Two cascaded transistor clipping stages with the coupling
            # filters between them: the long, smooth sustain.
            g = math.sqrt(gain)
            lp, hp = self._inters[ch]
            lp.set(f0=3500.0)
            hp.set(f0=90.0)
            s = clip_soft_diode(up * g)
            s = lp.process(hp.process(s))
            return clip_soft_diode(s * g * 1.6)
        if kind == "zone":
            # Three stages, the middle one hard: dense, compressed, tight.
            g = gain ** (1.0 / 3.0)
            lp, hp = self._inters[ch]
            lp.set(f0=5000.0)
            hp.set(f0=160.0)
            s = clip_soft_diode(up * g)
            s = lp.process(hp.process(s))
            s = clip_hard_diode(s * g)
            return clip_hard_diode(s * g * 0.7)
        # Transistor fuzz: the bias sags with the playing (blocking
        # distortion) so a hard-hit note chokes then blooms.
        sag = spec.get("sag", 0.5)
        b = np.clip(bias_s - sag * env_up * 0.6, 0.0, 1.0)
        v = up * gain
        if spec.get("gate", 0.0) > 0.0:
            dz = 0.04 + 0.25 * (1.0 - bias_s)
            v = np.sign(v) * np.maximum(np.abs(v) - dz, 0.0) * (1.0 + dz)
        off = (0.5 - b) * 0.9
        u = v + off
        y = np.where(u >= 0.0, np.tanh(u * 1.1), np.tanh(u * 2.2) * 0.45)
        y0 = np.where(off >= 0.0, np.tanh(off * 1.1), np.tanh(off * 2.2) * 0.45)
        return y - y0

    def _process(self, x, n):
        spec = DIRT_MODELS.get(self.model, DIRT_MODELS["Green"])
        gmin, gmax = spec["gain"]
        gain = gmin * (gmax / gmin) ** clampf(self.gain, 0.0, 1.0)
        self._hp.set(f0=spec["hp"])
        pre = self._hp.process(x)
        if spec["gbw"]:
            # The op-amp cannot give full gain at high frequencies: its
            # bandwidth shrinks as the gain knob rises. Most of a Rat's
            # character is this low-pass moving with Distortion.
            self._gbw.set(f0=clampf(spec["gbw"] / gain, 1200.0, 20000.0))
            pre = self._gbw.process(pre)
        mono = pre[0] if pre.shape[0] == 1 else np.mean(pre, axis=0)
        env = self._env.process(np.abs(mono)) * min(gain, 50.0)
        bias = clampf(self.bias, 0.0, 1.0)
        wet = np.empty_like(x)
        for ch in range(x.shape[0]):
            up = self._oss[ch].up(pre[ch])
            env_up = np.repeat(env, 8) if spec["clip"] == "transistor" else 0.0
            s = self._clip(up, ch, spec, gain, bias, env_up)
            wet[ch] = self._oss[ch].down(s)
        wet = self._dc.process(wet)
        if spec["dry"]:
            wet = wet + pre * spec["dry"]
        wet = self._tone(wet, spec)
        self._out.set(_level_gain(self.level) * spec["out"] * 0.55)
        wet = wet * self._out.ramp(n)
        self._mixs.set(clampf(self.mix, 0.0, 1.0))
        m = self._mixs.ramp(n)
        if np.isscalar(m) and m >= 0.999:
            return wet
        return x * (1.0 - m) + wet * m

    def _tone(self, y, spec):
        t = clampf(self.tone, 0.0, 1.0)
        lo, hi = spec["tone_lo"], spec["tone_hi"]
        kind = spec["tone"]
        if kind == "ts":
            self._lp.set(f0=lo * (hi / lo) ** t)
            self._lp2.set(f0=7500.0)
            return self._lp2.process(self._lp.process(y))
        if kind == "lp":
            self._lp.set(f0=lo * (hi / lo) ** t)
            self._lp2.set(f0=min(hi * 1.3, 18000.0))
            return self._lp2.process(self._lp.process(y))
        if kind == "rat":
            # The filter knob closes clockwise; here Tone open = bright.
            self._lp.set(f0=lo * (hi / lo) ** t)
            self._lp2.set(f0=12000.0)
            return self._lp2.process(self._lp.process(y))
        if kind == "scoop":
            # Passive low-pass/high-pass blend. At noon both are down at the
            # crossover, which is the famous mid scoop; Mid fills it back in.
            self._lp.set(f0=lo)
            self._shp.set(f0=hi)
            a = self._lp.process(y)
            b = self._shp.process(y)
            out = a * (1.0 - t) * 1.6 + b * t * 1.6
            mid = clampf(self.mid, 0.0, 1.0)
            self._mid.set(f0=math.sqrt(lo * hi), q=0.7, gain_db=(mid - 0.5) * 18.0)
            self._lp2.set(f0=7000.0)
            return self._lp2.process(self._mid.process(out))
        if kind == "zone":
            self._lo.set(f0=120.0, gain_db=(clampf(self.bass, 0.0, 1.0) - 0.5) * 30.0)
            mf = lo * (hi / lo) ** clampf(self.mid_freq, 0.0, 1.0)
            self._mid.set(f0=mf, q=1.2, gain_db=(clampf(self.mid, 0.0, 1.0) - 0.5) * 30.0)
            self._hi.set(f0=3200.0, gain_db=(t - 0.5) * 30.0)
            self._lp2.set(f0=6500.0)
            return self._lp2.process(self._hi.process(self._mid.process(self._lo.process(y))))
        return y


# ==========================================================================
# Boost
# ==========================================================================

# model: (pre_hp, peak_hz, peak_q, peak_db, clip_scale, asym)
BOOST_MODELS = {
    "Clean": (25.0, 0.0, 0.7, 0.0, 0.25, False),
    "Treble": (450.0, 3500.0, 0.6, 6.0, 1.0, True),    # germanium treble booster
    "Mid": (120.0, 900.0, 0.8, 9.0, 0.5, False),
    "Bass": (25.0, 110.0, 0.7, 8.0, 0.35, False),
}


class Boost(Effect):
    TYPE = "boost"
    LABEL = "Boost"
    CATEGORY = "Dirt"
    MODELS = tuple(BOOST_MODELS)
    PARAMS = (
        P("gain", "Boost", 0.0, 1.0, 0.5, aliases=("boost", "gain", "drive", "level2",
                                                   "volume", "amount")),
        P("tone", "Tone", 0.0, 1.0, 0.5, aliases=("treble", "filter", "range",
                                                  "color", "colour", "eq")),
        P("level", "Level", 0.0, 1.0, 0.5, aliases=("output", "out", "vol")),
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        c = self.channels
        self._oss = [Oversampler() for _ in range(c)]
        for o in self._oss:
            o.prepare(sr, max_block)
        self._hp = Biquad("highpass", 25.0, 0.707)
        self._peak = Biquad("peak", 1000.0, 0.7, 0.0)
        self._tilt = Biquad("highshelf", 1500.0, 0.707, 0.0)
        for f in (self._hp, self._peak, self._tilt):
            f.prepare(sr, c)
        self._dc = DCBlocker()
        self._dc.prepare(sr, c)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        for o in self._oss:
            o.reset()
        for f in (self._hp, self._peak, self._tilt):
            f.reset()
        self._dc.reset()

    def _process(self, x, n):
        hp, pk, pq, pdb, cs, asym = BOOST_MODELS.get(self.model, BOOST_MODELS["Clean"])
        self._hp.set(f0=hp, q=0.5)
        y = self._hp.process(x)
        if pk:
            self._peak.set(f0=pk, q=pq, gain_db=pdb)
            y = self._peak.process(y)
        self._tilt.set(gain_db=(clampf(self.tone, 0.0, 1.0) - 0.5) * 12.0)
        y = self._tilt.process(y)
        g = db2lin(26.0 * clampf(self.gain, 0.0, 1.0))
        out = np.empty_like(y)
        for ch in range(y.shape[0]):
            up = self._oss[ch].up(y[ch]) * g * cs
            if asym:
                s = np.where(up >= 0.0, np.tanh(up), 1.3 * np.tanh(up / 1.3))
            else:
                s = np.tanh(up)
            out[ch] = self._oss[ch].down(s) / cs
        out = self._dc.process(out)
        self._out.set(_level_gain(self.level) * (0.8 if asym else 1.0))
        return out * self._out.ramp(n)
