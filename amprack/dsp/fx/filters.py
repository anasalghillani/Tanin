"""Filter effects: wah, auto-filter (envelope / LFO / step), lo-fi, bit crusher."""

from __future__ import annotations

import math

import numpy as np

from ..core import Biquad, DelayLine, LFO, OnePole, SmoothedParam, clampf, db2lin
from .base import (
    LEVEL,
    MIX,
    SHAPES,
    Effect,
    EnvFollower,
    P,
    PC,
    PL,
    ShapeLFO,
    TVFilter,
    grid,
    svf_coeffs,
)

CHUNK = 32   # 0.67 ms at 48 kHz: coefficient update grid for swept filters


def _level_gain(level: float) -> float:
    """Level knob: 0.5 is unity, 0 is -inf-ish (-40 dB), 1 is +12 dB."""
    level = clampf(level, 0.0, 1.0)
    if level <= 0.5:
        return db2lin(-40.0 + 80.0 * level) if level < 0.5 else 1.0
    return db2lin((level - 0.5) * 24.0)


# ==========================================================================
# Wah
# ==========================================================================

# name: (lo_hz, hi_hz, q_heel, q_toe, bandpass weight, lowpass weight)
WAH_VOICES = {
    # Inductor wah, US voicing: a narrow peak that sweeps low and sings.
    "Classic": (350.0, 2200.0, 5.5, 3.8, 2.6, 0.40),
    # UK voicing: a wider, vocal peak that keeps more of the low end.
    "Vintage": (420.0, 1800.0, 3.8, 3.0, 1.9, 0.65),
    # Modern: the widest sweep; Q follows the Resonance knob.
    "Modern": (300.0, 2800.0, 4.5, 4.5, 3.0, 0.36),
}


class Wah(Effect):
    """Resonant wah filter driven by the playing (auto-wah), an LFO, or held
    at one spot (a cocked wah).

    The filter is one 2nd-order section: a band-pass peak plus a little of
    the matching low-pass, which share a denominator and so cost one filter.
    That mix is what separates an inductor wah (a vowel with body under it)
    from a bare band-pass sweep (thin and synthetic).
    """

    TYPE = "wah"
    LABEL = "Wah"
    CATEGORY = "Filter"
    MODELS = tuple(WAH_VOICES)
    PARAMS = (
        PC("source", "Control", ("Envelope", "LFO", "Fixed"),
           aliases=("mode", "control")),
        P("position", "Position", 0.0, 1.0, 0.25,
          aliases=("pedal", "pedalposition", "pos", "manual", "sweep", "frequency",
                   "freq", "start")),
        P("sensitivity", "Sensitivity", 0.0, 1.0, 0.6,
          aliases=("sens", "threshold", "envelope", "env", "trigger")),
        P("range", "Range", 0.0, 1.0, 0.75, aliases=("depth", "amount", "width",
                                                     "intensity")),
        PL("attack_ms", "Attack", 1.0, 80.0, 6.0, "ms", aliases=("attack", "speed")),
        PL("release_ms", "Release", 20.0, 1000.0, 160.0, "ms",
           aliases=("release", "decay")),
        PL("rate", "Rate", 0.05, 12.0, 1.5, "Hz", aliases=("speed", "lforate")),
        PC("shape", "Shape", SHAPES, aliases=("wave", "waveform")),
        P("q", "Resonance", 0.0, 1.0, 0.5, aliases=("resonance", "res", "peak",
                                                    "qfactor", "emphasis")),
        MIX,
        LEVEL,
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._f = TVFilter(2, self.channels)
        self._env = EnvFollower()
        self._env.prepare(sr)
        self._lfo = ShapeLFO()
        self._lfo.prepare(sr, max_block)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        self._f.reset()
        self._env.reset()
        self._lfo.reset()

    def control(self, x, n):
        """The 0..1 sweep position for every sample of this block."""
        pos = clampf(self.position, 0.0, 1.0)
        rng = clampf(self.range, 0.0, 1.0)
        src = self.source
        if src == "LFO":
            w = ShapeLFO.shape(self._lfo.phases(clampf(self.rate, 0.02, 20.0), n),
                               self.shape, 0.6)
            return np.clip(pos + rng * (0.5 + 0.5 * w) * (1.0 - pos), 0.0, 1.0)
        if src == "Fixed":
            return np.full(n, pos)
        self._env.set(clampf(self.attack_ms, 0.5, 200.0),
                      clampf(self.release_ms, 5.0, 3000.0))
        e = self._env.process(np.mean(x, axis=0) if x.shape[0] > 1 else x[0])
        # Sensitivity sets how hard you must pick to open it all the way:
        # 0 needs about 0 dBFS, 1 opens on a -40 dBFS note.
        k = db2lin(40.0 * clampf(self.sensitivity, 0.0, 1.0))
        c = 1.0 - np.exp(-e * k * 3.0)
        return np.clip(pos + rng * c * (1.0 - pos), 0.0, 1.0)

    def _process(self, x, n):
        lo, hi, q_heel, q_toe, kb, kl = WAH_VOICES.get(self.model, WAH_VOICES["Classic"])
        ctrl = self.control(x, n)
        res = clampf(self.q, 0.0, 1.0)
        qscale = 0.55 + 0.9 * res
        y = np.empty_like(x)
        for a, b in grid(self._t, n, CHUNK):
            c = float(ctrl[(a + b) // 2])
            fc = lo * (hi / lo) ** c
            q = (q_heel + (q_toe - q_heel) * c) * qscale
            bb, aa = svf_coeffs("bp", fc, q, self.sr)
            bl, _ = svf_coeffs("lp", fc, q, self.sr)
            # Peak gain of the band-pass is 0 dB; a real wah's peak sits well
            # above its pass band, which the weights restore.
            coef_b = bb * kb * (0.8 + 0.25 * q) + bl * kl
            y[:, a:b] = self._f.run_all(coef_b, aa, x[:, a:b])
        self._out.set(_level_gain(self.level))
        y *= self._out.ramp(n)
        m = clampf(self.mix, 0.0, 1.0)
        if m >= 0.999:
            return y
        return x * (1.0 - m) + y * m


# ==========================================================================
# Auto-filter / envelope filter / synth filter
# ==========================================================================

class AutoFilter(Effect):
    """Envelope filter, LFO filter and step (sample-and-hold) filter in one.

    Low-pass, band-pass and high-pass are resonant 2-pole sections; "Ladder"
    is a 4-pole low-pass (two sections, the second carrying the resonance),
    which is the fat, squelchy synth filter sound. The sweep can go up with
    the pick attack (the classic funk envelope filter) or down.
    """

    TYPE = "autofilter"
    LABEL = "Auto Filter"
    CATEGORY = "Filter"
    MODELS = ("Lowpass", "Bandpass", "Highpass", "Ladder")
    PARAMS = (
        PC("source", "Control", ("Envelope", "LFO", "Envelope+LFO"),
           aliases=("mode", "control")),
        PC("direction", "Direction", ("Up", "Down"), aliases=("dir", "sweepdirection")),
        P("position", "Frequency", 0.0, 1.0, 0.15,
          aliases=("freq", "frequency", "cutoff", "manual", "start", "range_lo")),
        P("sensitivity", "Sensitivity", 0.0, 1.0, 0.6,
          aliases=("sens", "threshold", "envelope", "env", "gain", "input")),
        P("range", "Range", 0.0, 1.0, 0.8, aliases=("depth", "amount", "sweep",
                                                    "intensity", "envamount")),
        P("q", "Resonance", 0.0, 1.0, 0.55, aliases=("resonance", "res", "peak",
                                                     "qfactor", "emphasis")),
        PL("attack_ms", "Attack", 1.0, 80.0, 4.0, "ms", aliases=("attack",)),
        PL("release_ms", "Release", 20.0, 1500.0, 180.0, "ms",
           aliases=("release", "decay")),
        PL("rate", "Rate", 0.05, 16.0, 2.0, "Hz", aliases=("speed", "lforate")),
        PC("shape", "Shape", SHAPES, aliases=("wave", "waveform")),
        MIX,
        LEVEL,
    )

    LO, HI = 80.0, 6000.0

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._f = TVFilter(4, self.channels)
        self._env = EnvFollower()
        self._env.prepare(sr)
        self._lfo = ShapeLFO()
        self._lfo.prepare(sr, max_block)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        self._f.reset()
        self._env.reset()
        self._lfo.reset()

    def _control(self, x, n):
        pos = clampf(self.position, 0.0, 1.0)
        rng = clampf(self.range, 0.0, 1.0)
        mod = np.zeros(n)
        if self.source in ("Envelope", "Envelope+LFO"):
            self._env.set(clampf(self.attack_ms, 0.5, 200.0),
                          clampf(self.release_ms, 5.0, 3000.0))
            e = self._env.process(np.mean(x, axis=0) if x.shape[0] > 1 else x[0])
            k = db2lin(40.0 * clampf(self.sensitivity, 0.0, 1.0))
            mod += 1.0 - np.exp(-e * k * 3.0)
        if self.source in ("LFO", "Envelope+LFO"):
            w = ShapeLFO.shape(self._lfo.phases(clampf(self.rate, 0.02, 30.0), n),
                               self.shape, 0.5)
            mod += (0.5 + 0.5 * w) * (0.5 if self.source == "Envelope+LFO" else 1.0)
        np.clip(mod, 0.0, 1.0, out=mod)
        if self.direction == "Down":
            return np.clip(pos + rng * (1.0 - mod) * (1.0 - pos), 0.0, 1.0)
        return np.clip(pos + rng * mod * (1.0 - pos), 0.0, 1.0)

    def _coeffs(self, fc, res):
        kind = self.model
        if kind == "Ladder":
            q2 = 0.6 + res * 9.0
            b1, a1 = svf_coeffs("lp", fc, 0.54, self.sr)
            b2, a2 = svf_coeffs("lp", fc, q2, self.sr)
            # Ladder filters lose pass-band level as resonance rises.
            comp = 1.0 / (1.0 + res * 0.6)
            return np.convolve(b1, b2) * comp, np.convolve(a1, a2)
        q = 0.6 + res * 11.0
        k = {"Lowpass": "lp", "Bandpass": "bp", "Highpass": "hp"}.get(kind, "lp")
        b, a = svf_coeffs(k, fc, q, self.sr)
        if k == "bp":
            b = b * (1.0 + 0.15 * q) * 4.0
        elif k == "hp":
            b = b * 2.5
        # Pad to order 4 so one TVFilter serves every model.
        return np.concatenate((b, [0.0, 0.0])), np.concatenate((a, [0.0, 0.0]))

    def _process(self, x, n):
        ctrl = self._control(x, n)
        res = clampf(self.q, 0.0, 1.0)
        y = np.empty_like(x)
        for a, b in grid(self._t, n, CHUNK):
            c = float(ctrl[(a + b) // 2])
            fc = self.LO * (self.HI / self.LO) ** c
            bb, aa = self._coeffs(fc, res)
            y[:, a:b] = self._f.run_all(bb, aa, x[:, a:b])
        np.tanh(y * 0.8, out=y)       # a resonant peak can jump 20 dB: tame it
        y *= 1.25
        self._out.set(_level_gain(self.level))
        y *= self._out.ramp(n)
        m = clampf(self.mix, 0.0, 1.0)
        if m >= 0.999:
            return y
        return x * (1.0 - m) + y * m


# ==========================================================================
# Lo-fi: radio, telephone, cassette, megaphone
# ==========================================================================

# name: (hp_hz, lp_hz, drive_scale, wobble_scale, noise_scale)
LOFI_VOICES = {
    "Radio": (280.0, 3800.0, 1.0, 0.0, 0.6),
    "Telephone": (450.0, 3000.0, 1.6, 0.0, 0.4),
    "Megaphone": (650.0, 2600.0, 3.0, 0.0, 0.2),
    "Cassette": (60.0, 7000.0, 0.6, 1.0, 1.0),
}


class LoFi(Effect):
    TYPE = "lofi"
    LABEL = "Lo-Fi"
    CATEGORY = "Filter"
    MODELS = tuple(LOFI_VOICES)
    PARAMS = (
        P("tone", "Tone", 0.0, 1.0, 0.5, aliases=("filter", "color", "colour",
                                                  "bandwidth", "freq")),
        P("drive", "Drive", 0.0, 1.0, 0.3, aliases=("gain", "distortion", "dirt",
                                                    "saturation")),
        P("noise", "Noise", 0.0, 1.0, 0.15, aliases=("hiss", "crackle")),
        P("wobble", "Wobble", 0.0, 1.0, 0.35, aliases=("wow", "flutter", "warble",
                                                       "age", "mod")),
        MIX,
        LEVEL,
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        c = self.channels
        self._hp = Biquad("highpass", 300.0, 0.707)
        self._hp2 = Biquad("highpass", 300.0, 0.707)
        self._lp = Biquad("lowpass", 3500.0, 0.707)
        self._lp2 = Biquad("lowpass", 3500.0, 0.707)
        self._mid = Biquad("peak", 1500.0, 1.2, 4.0)
        for f in (self._hp, self._hp2, self._lp, self._lp2, self._mid):
            f.prepare(sr, c)
        self._line = DelayLine(int(sr * 0.02) + max_block + 8, c)
        self._line.prepare(max_block)
        self._lfo1 = LFO()
        self._lfo2 = LFO()
        self._lfo1.prepare(sr, max_block)
        self._lfo2.prepare(sr, max_block)
        self._rng = np.random.default_rng(1234)
        self._noise_lp = OnePole(0.05)
        self._noise_lp.prepare(sr, c)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        for f in (self._hp, self._hp2, self._lp, self._lp2, self._mid):
            f.reset()
        self._line.reset()
        self._rng = np.random.default_rng(1234)

    def _process(self, x, n):
        hp, lp, dsc, wsc, nsc = LOFI_VOICES.get(self.model, LOFI_VOICES["Radio"])
        tone = clampf(self.tone, 0.0, 1.0)
        sig = x
        wob = clampf(self.wobble, 0.0, 1.0) * wsc
        if wob > 0.0:
            # Wow (slow) and flutter (fast) as a modulated read.
            w = (self._lfo1.block(0.55, n) * 1.0 + self._lfo2.block(7.3, n) * 0.18)
            d = (6.0 + w * 4.5 * wob) * 1e-3 * self.sr + n
            self._line.write(sig)
            sig = self._line.read_modulated(d, n)
        else:
            self._lfo1.advance_only(0.55, n)
            self._lfo2.advance_only(7.3, n)
            self._line.write(sig)
            sig = self._line.read_fixed(int(6e-3 * self.sr) + n, n).copy()
        f_hp = hp * (1.6 ** (0.5 - tone))
        f_lp = lp * (2.0 ** (tone - 0.5))
        self._hp.set(f0=f_hp)
        self._hp2.set(f0=f_hp)
        self._lp.set(f0=f_lp)
        self._lp2.set(f0=f_lp)
        self._mid.set(f0=math.sqrt(f_hp * f_lp), gain_db=3.0 + 3.0 * (dsc > 1.2))
        sig = self._lp2.process(self._lp.process(self._hp2.process(self._hp.process(sig))))
        sig = self._mid.process(sig)
        g = 1.0 + clampf(self.drive, 0.0, 1.0) * 14.0 * dsc
        sig = np.tanh(sig * g) / math.tanh(min(g, 3.0)) * 0.32
        nz = clampf(self.noise, 0.0, 1.0) * nsc
        if nz > 0.0:
            hiss = self._rng.standard_normal((x.shape[0], n)) * (0.004 * nz)
            sig = sig + hiss
        else:
            self._rng.standard_normal((x.shape[0], n))   # keep the stream aligned
        self._out.set(_level_gain(self.level))
        sig = sig * self._out.ramp(n)
        m = clampf(self.mix, 0.0, 1.0)
        if m >= 0.999:
            return sig
        return x * (1.0 - m) + sig * m


# ==========================================================================
# Bit crusher
# ==========================================================================

class BitCrusher(Effect):
    """Bit depth and sample-rate reduction. The aliasing is the point here,
    so unlike the drives this one is deliberately not oversampled."""

    TYPE = "crusher"
    LABEL = "Bit Crusher"
    CATEGORY = "Filter"
    MODELS = ("Standard",)
    PARAMS = (
        P("bits", "Bits", 1.0, 16.0, 8.0, "bit", aliases=("bit", "bitdepth",
                                                          "resolution", "crush")),
        PL("rate_hz", "Sample Rate", 300.0, 48000.0, 11025.0, "Hz",
           aliases=("rate", "samplerate", "downsample", "decimate", "freq",
                    "frequency")),
        P("tone", "Tone", 0.0, 1.0, 1.0, aliases=("filter", "smooth")),
        MIX,
        LEVEL,
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._held = np.zeros(self.channels)
        self._lp = Biquad("lowpass", 18000.0, 0.707)
        self._lp.prepare(sr, self.channels)
        self._idx = np.arange(max_block, dtype=np.int64)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        self._held[...] = 0.0
        self._lp.reset()

    def _process(self, x, n):
        step = max(1.0, self.sr / clampf(self.rate_hz, 100.0, self.sr))
        t = self._t + self._idx[:n]
        # Index of the sample being held at each absolute time t: a function
        # of t alone, so it is identical at any buffer size.
        h = np.ceil(np.floor(t / step) * step - 1e-9).astype(np.int64)
        rel = h - self._t
        y = np.empty_like(x)
        inside = rel >= 0
        for ch in range(x.shape[0]):
            y[ch] = np.where(inside, x[ch, np.clip(rel, 0, n - 1)], self._held[ch])
        last = rel[-1]
        if last >= 0:
            self._held[:] = x[:, last]
        levels = 2.0 ** (clampf(self.bits, 1.0, 24.0) - 1.0)
        y = np.round(y * levels) / levels
        tone = clampf(self.tone, 0.0, 1.0)
        if tone < 0.999:
            self._lp.set(f0=1500.0 * (12.0 ** tone))
            y = self._lp.process(y)
        else:
            self._lp.process(y)
        self._out.set(_level_gain(self.level))
        y = y * self._out.ramp(n)
        m = clampf(self.mix, 0.0, 1.0)
        if m >= 0.999:
            return y
        return x * (1.0 - m) + y * m
