"""Pitch effects: analog-style sub octave, rectifier octave-up fuzz, and a
polyphonic pitch shifter / harmoniser / detuner."""

from __future__ import annotations

import math

import numpy as np

from ..core import Biquad, DCBlocker, DelayLine, OnePole, Oversampler, SmoothedParam, clampf, db2lin
from .base import LEVEL, MIX, Effect, P, PC, PL
from .filters import _level_gain


# ==========================================================================
# Sub octave (flip-flop divider)
# ==========================================================================

class OctaveDown(Effect):
    """One and two octaves down, the analog way.

    The input is low-passed hard so that mostly the fundamental is left, and
    squared up by a Schmitt trigger (a comparator with hysteresis, so noise
    near zero cannot flip it). A flip-flop toggles on every rising edge: a
    square wave at half the frequency, and a second flip-flop gives a quarter.
    The sub voice is the filtered input *multiplied* by that square, which
    keeps the note's own envelope and some of its colour instead of a bare
    synth square. Like the hardware it is monophonic: play single notes.
    Fully vectorised, so it is identical at any buffer size.
    """

    TYPE = "octdown"
    LABEL = "Sub Octave"
    CATEGORY = "Pitch"
    MODELS = ("Analog", "Deep")
    PARAMS = (
        P("direct", "Direct", 0.0, 1.0, 0.7, aliases=("dry", "directlevel", "dir",
                                                       "clean")),
        P("oct1", "Octave -1", 0.0, 1.0, 0.7, aliases=("octave1", "oct", "sub",
                                                         "octave", "lower", "low1",
                                                         "subocatve", "suboctave")),
        P("oct2", "Octave -2", 0.0, 1.0, 0.0, aliases=("octave2", "sub2", "low2")),
        P("tone", "Tone", 0.0, 1.0, 0.45, aliases=("filter", "color", "colour")),
        P("tracking", "Tracking", 0.0, 1.0, 0.5, aliases=("range", "track",
                                                          "sensitivity", "sens")),
        LEVEL,
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        c = self.channels
        self._pre = [Biquad("lowpass", 500.0, 0.6), Biquad("lowpass", 500.0, 0.9)]
        self._det = [Biquad("lowpass", 400.0, 0.6), Biquad("lowpass", 400.0, 0.8)]
        for f in self._pre:
            f.prepare(sr, c)
        for f in self._det:
            f.prepare(sr, 1)
        self._hp = Biquad("highpass", 40.0, 0.707)
        self._hp.prepare(sr, c)
        self._post = [Biquad("lowpass", 900.0, 0.707), Biquad("lowpass", 900.0, 0.707)]
        for f in self._post:
            f.prepare(sr, c)
        self._env = OnePole(12.0)
        self._env.prepare(sr, 1)
        self._gate = OnePole(8.0)
        self._gate.prepare(sr, 1)
        self._idx = np.arange(max_block)
        self._dc = DCBlocker()
        self._dc.prepare(sr, c)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)
        self.reset()

    def reset(self):
        for f in self._pre + self._det + self._post + [self._hp]:
            f.reset()
        self._env.reset()
        self._gate.reset()
        self._dc.reset()
        self._state = -1.0     # Schmitt trigger output
        self._count = 0        # rising edges seen, mod 4

    def _process(self, x, n):
        track = clampf(self.tracking, 0.0, 1.0)
        fdet = 120.0 * (6.0 ** track)            # 120 .. 720 Hz
        for f in self._det:
            f.set(f0=fdet)
        for f in self._pre:
            f.set(f0=fdet * 1.4)
        mono = x[0] if x.shape[0] == 1 else np.mean(x, axis=0)
        d = mono
        for f in self._det:
            d = f.process(d)
        env = self._env.process(np.abs(d))

        # Schmitt trigger: thresholds track the envelope so the hysteresis
        # scales with the note rather than being fixed in volts.
        h = env * 0.25 + 2e-4
        hi = d > h
        lo = d < -h
        decided = hi | lo
        last = np.where(decided, self._idx[:n], -1)
        np.maximum.accumulate(last, out=last)
        val = np.where(hi, 1.0, -1.0)
        s = np.where(last >= 0, val[np.maximum(last, 0)], self._state)
        prev = np.empty(n)
        prev[0] = self._state
        prev[1:] = s[:-1]
        rising = (s > 0.0) & (prev < 0.0)
        cnt = np.cumsum(rising) + self._count
        self._state = float(s[-1])
        self._count = int(cnt[-1]) % 4
        ff1 = np.where(cnt % 2 == 0, 1.0, -1.0)
        ff2 = np.where((cnt // 2) % 2 == 0, 1.0, -1.0)

        # Below about -50 dB the tracking is guesswork; fade the subs out.
        g = self._gate.process(np.clip((env - 0.002) * 400.0, 0.0, 1.0))

        filt = self._hp.process(x)
        for f in self._pre:
            filt = f.process(filt)
        deep = self.model == "Deep"
        sub = filt * (ff1 * g) * (clampf(self.oct1, 0.0, 1.0) * 2.2)
        sub = sub + filt * (ff2 * g) * (clampf(self.oct2, 0.0, 1.0) * (2.6 if deep else 2.0))
        ft = 250.0 * (8.0 ** clampf(self.tone, 0.0, 1.0)) * (0.7 if deep else 1.0)
        for f in self._post:
            f.set(f0=ft)
            sub = f.process(sub)
        sub = self._dc.process(sub)
        y = x * clampf(self.direct, 0.0, 1.0) + sub
        self._out.set(_level_gain(self.level))
        return y * self._out.ramp(n)


# ==========================================================================
# Octave-up fuzz (rectifier)
# ==========================================================================

class OctaveUp(Effect):
    """Octave-up fuzz: a full-wave rectifier after a fuzz stage. Rectifying
    folds the negative half-cycle up, which doubles the frequency -- the
    ringing octave that is loudest on the neck pickup above the 12th fret,
    exactly like the hardware. Oversampled, because a rectifier is a corner."""

    TYPE = "octup"
    LABEL = "Octave Fuzz"
    CATEGORY = "Pitch"
    MODELS = ("Standard",)
    PARAMS = (
        P("fuzz", "Fuzz", 0.0, 1.0, 0.6, aliases=("gain", "drive", "boost",
                                                  "distortion", "dist")),
        P("octave", "Octave", 0.0, 1.0, 0.8, aliases=("oct", "blend", "upper",
                                                      "octavelevel", "octaveup")),
        P("tone", "Tone", 0.0, 1.0, 0.5, aliases=("filter",)),
        MIX,
        LEVEL,
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._oss = [Oversampler(16) for _ in range(self.channels)]
        for o in self._oss:
            o.prepare(sr, max_block)
        self._hp = Biquad("highpass", 180.0, 0.707)
        self._hp.prepare(sr, self.channels)
        self._lp = Biquad("lowpass", 4000.0, 0.707)
        self._lp.prepare(sr, self.channels)
        self._dc = DCBlocker()
        self._dc.prepare(sr, self.channels)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        for o in self._oss:
            o.reset()
        self._hp.reset()
        self._lp.reset()
        self._dc.reset()

    def _process(self, x, n):
        g = 3.0 * (60.0 ** clampf(self.fuzz, 0.0, 1.0))
        o = clampf(self.octave, 0.0, 1.0)
        pre = self._hp.process(x)
        wet = np.empty_like(x)
        for ch in range(x.shape[0]):
            up = self._oss[ch].up(pre[ch])
            s = np.tanh(up * g)
            # Rectified: the octave. sqrt(s^2 + e^2) is |s| with the corner
            # rounded off, as a real diode's knee is, so it does not alias.
            r = np.sqrt(s * s + 0.0225) * 2.0 - 0.95
            v = s * (1.0 - o) + r * o
            v = np.tanh(v * 1.8)
            wet[ch] = self._oss[ch].down(v)
        wet = self._dc.process(wet)
        self._lp.set(f0=1200.0 * (7.0 ** clampf(self.tone, 0.0, 1.0)))
        wet = self._lp.process(wet) * 0.11
        self._out.set(_level_gain(self.level))
        wet = wet * self._out.ramp(n)
        m = clampf(self.mix, 0.0, 1.0)
        if m >= 0.999:
            return wet
        return x * (1.0 - m) + wet * m


# ==========================================================================
# Pitch shifter / harmoniser / detune
# ==========================================================================

class PitchShift(Effect):
    """Polyphonic pitch shifting by delay-line splicing.

    Two read heads move through a delay line at the new speed, half a window
    apart, each faded in and out with a sin^2 window so that while one is
    jumping back the other is at full level (the two windows sum to exactly
    one). Polyphonic and with only a few ms of latency, which is what a
    whammy or a rack harmoniser needs; the cost is a slight chorusing on long
    notes, which a longer window reduces and a shorter one trades for speed.

    Two voices: voice 2 is a second interval (harmony) or the opposite
    detune (a stereo doubler when the rack is stereo).
    """

    TYPE = "pitch"
    LABEL = "Pitch Shifter"
    CATEGORY = "Pitch"
    MODELS = ("Shift", "Harmony", "Detune")
    PARAMS = (
        P("semitones", "Interval", -24.0, 24.0, 12.0, "st",
          aliases=("pitch", "interval", "shift", "semitone", "voice1", "pitch1",
                   "harmony", "transpose", "step", "steps")),
        P("cents", "Fine", -50.0, 50.0, 0.0, "ct",
          aliases=("detune", "fine", "cent", "finetune")),
        P("semitones2", "Interval 2", -24.0, 24.0, 0.0, "st",
          aliases=("voice2", "pitch2", "interval2", "shift2")),
        P("voice2", "Voice 2", 0.0, 1.0, 0.0, aliases=("level2", "mix2")),
        PL("window_ms", "Window", 10.0, 90.0, 35.0, "ms",
           aliases=("window", "splice", "glide", "smooth", "tracking")),
        MIX,
        LEVEL,
    )

    MAX_WIN_MS = 100.0

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._line = DelayLine(int(sr * self.MAX_WIN_MS * 1e-3) + max_block + 16,
                               self.channels)
        self._line.prepare(max_block)
        self._idx = np.arange(max_block, dtype=np.float64)
        self._ph = [0.0, 0.0]
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)
        self._mix = SmoothedParam(1.0)
        self._mix.prepare(max_block)

    def reset(self):
        self._line.reset()
        self._ph = [0.0, 0.0]

    def _voice(self, k, ratio, win, n):
        """Read one shifted voice. ``k`` selects the phase accumulator."""
        inc = (1.0 - ratio) / win
        ph = (self._ph[k] + self._idx[:n] * inc) % 1.0
        self._ph[k] = (self._ph[k] + n * inc) % 1.0
        ph2 = (ph + 0.5) % 1.0
        base = 2.0 + n
        da = base + ph * win
        db = base + ph2 * win
        ga = np.sin(math.pi * ph) ** 2
        gb = 1.0 - ga
        ra = self._line.read_modulated(da, n)
        rb = self._line.read_modulated(db, n)
        return ra * ga + rb * gb

    def _process(self, x, n):
        win = clampf(self.window_ms, 5.0, self.MAX_WIN_MS - 5.0) * 1e-3 * self.sr
        semi = clampf(self.semitones, -36.0, 36.0)
        cents = clampf(self.cents, -100.0, 100.0)
        self._line.write(x)
        model = self.model
        if model == "Detune":
            r1 = 2.0 ** (cents / 1200.0 if cents else 12.0 / 1200.0)
            r2 = 1.0 / r1
            v1 = self._voice(0, r1, min(win, 0.025 * self.sr), n)
            v2 = self._voice(1, r2, min(win, 0.025 * self.sr), n)
            if x.shape[0] == 2:
                wet = np.stack((v1[0], v2[1]))
            else:
                wet = (v1 + v2) * 0.75
        else:
            r1 = 2.0 ** (semi / 12.0 + cents / 1200.0)
            wet = self._voice(0, r1, win, n)
            v2l = clampf(self.voice2, 0.0, 1.0)
            if v2l > 0.0:
                r2 = 2.0 ** (clampf(self.semitones2, -36.0, 36.0) / 12.0)
                wet = wet + self._voice(1, r2, win, n) * v2l
            else:
                self._ph[1] = 0.0
        self._out.set(_level_gain(self.level))
        wet = wet * self._out.ramp(n)
        m = clampf(self.mix, 0.0, 1.0)
        if model == "Harmony":
            # Harmony keeps the played note at full level under the voices.
            return x + wet * m
        self._mix.set(m)
        mm = self._mix.ramp(n)
        return x * (1.0 - mm) + wet * mm
