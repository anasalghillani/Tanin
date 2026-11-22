"""Modulation: chorus, flanger, phaser, photo-cell vibe, tremolo, rotary speaker."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter, sosfilt

from ..core import Biquad, DelayLine, LFO, OnePole, SmoothedParam, TWO_PI, clampf
from .base import (
    cell_start,
    LEVEL,
    MIX,
    SHAPES,
    Effect,
    P,
    PC,
    PL,
    ShapeLFO,
    TVFilter,
    allpass1,
    grid,
)
from .filters import _level_gain

RATE_ALIASES = ("rate", "speed", "lforate", "frequency", "freq")
DEPTH_ALIASES = ("depth", "intensity", "amount", "width", "sweep")
SPREAD_ALIASES = ("stereo", "spread", "width2", "stereowidth", "phase")


# ==========================================================================
# Chorus
# ==========================================================================

# name: (voices, nominal delay ms, depth_ms_max, wet_lowpass_hz)
CHORUS_VOICES = {
    "Classic": (2, 9.0, 4.0, 7000.0),      # one BBD line, quadrature L/R
    "Ensemble": (2, 6.5, 2.6, 9000.0),     # shallower, richer, wider
    "Tri": (3, 11.0, 4.5, 11000.0),        # three voices 120 degrees apart
    "Dimension": (2, 7.5, 1.2, 12000.0),   # slow, subtle, very wide
}


class Chorus(Effect):
    TYPE = "chorus"
    LABEL = "Chorus"
    CATEGORY = "Modulation"
    MODELS = tuple(CHORUS_VOICES)
    STEREO = True
    PARAMS = (
        PL("rate", "Rate", 0.05, 8.0, 0.8, "Hz", aliases=RATE_ALIASES),
        P("depth", "Depth", 0.0, 1.0, 0.5, aliases=DEPTH_ALIASES),
        PL("delay_ms", "Delay", 2.0, 25.0, 8.0, "ms", aliases=("delay", "predelay",
                                                                "time", "manual")),
        P("tone", "Tone", 0.0, 1.0, 0.6, aliases=("filter", "eq", "color", "colour",
                                                  "hicut", "treble")),
        P("spread", "Spread", 0.0, 1.0, 0.8, aliases=SPREAD_ALIASES),
        P("vibrato", "Vibrato", 0.0, 1.0, 0.0, aliases=("vib", "pitch")),
        MIX,
    )

    def __init__(self):
        super().__init__()
        self.mix = 0.5

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._line = DelayLine(int(sr * 0.05) + max_block + 16, 2)
        self._line.prepare(max_block)
        self._lfo = LFO()
        self._lfo.prepare(sr, max_block)
        self._lp = Biquad("lowpass", 8000.0, 0.707)
        self._lp.prepare(sr, 2)

    def reset(self):
        self._line.reset()
        self._lp.reset()
        self._lfo.phase = 0.0

    def _process(self, x, n):
        voices, base_ms, dmax, lp = CHORUS_VOICES.get(self.model, CHORUS_VOICES["Classic"])
        base_ms = clampf(self.delay_ms, 1.0, 30.0)
        dms = dmax * clampf(self.depth, 0.0, 1.0)
        base_ms = max(base_ms, dms + 0.3)
        rate = clampf(self.rate, 0.01, 12.0)
        if self.model == "Dimension":
            rate *= 0.5
        ph = self._lfo.phase_block(rate, n)
        spread = clampf(self.spread, 0.0, 1.0)
        self._line.write(x)
        k = 1e-3 * self.sr
        wet = np.zeros((2, n))
        if voices == 3:
            mono_l = 0
            for v, (pl, pr) in enumerate(((1.0, 0.2), (0.6, 0.6), (0.2, 1.0))):
                off = v / 3.0
                d = (base_ms + dms * np.sin(TWO_PI * (ph + off))) * k + n
                r = self._line.read_modulated(d, n)
                src = r[mono_l] * 0.5 + r[1] * 0.5
                wl = 1.0 - spread * (1.0 - pl)
                wr = 1.0 - spread * (1.0 - pr)
                wet[0] += src * wl * 0.55
                wet[1] += src * wr * 0.55
        else:
            off = 0.25 + 0.25 * spread
            if self.model == "Dimension":
                off = 0.5
            dl = (base_ms + dms * np.sin(TWO_PI * ph)) * k + n
            dr = (base_ms + dms * np.sin(TWO_PI * (ph + off))) * k + n
            wet[0] = self._line.read_modulated(dl, n)[0]
            wet[1] = self._line.read_modulated(dr, n)[1]
            if spread < 1.0:
                mid = (wet[0] + wet[1]) * 0.5
                wet[0] = mid + (wet[0] - mid) * spread
                wet[1] = mid + (wet[1] - mid) * spread
        self._lp.set(f0=lp * (0.35 + 0.65 * clampf(self.tone, 0.0, 1.0)))
        wet = self._lp.process(wet)
        m = clampf(self.mix, 0.0, 1.0)
        vib = clampf(self.vibrato, 0.0, 1.0)
        if vib > 0.0:
            # Vibrato: fade the dry out so only the pitch-wobbled voice is left.
            return x * (1.0 - m) * (1.0 - vib) + wet * (m + (1.0 - m) * vib)
        return x * (1.0 - m * 0.5) + wet * m


# ==========================================================================
# Flanger
# ==========================================================================

# name: (min_ms, sweep_ms, feedback scale, default shape, through_zero)
FLANGER_VOICES = {
    "Classic": (0.6, 4.0, 0.85, "Triangle", False),
    "Jet": (0.5, 6.0, 0.97, "Triangle", False),
    "Vintage": (0.8, 7.0, 0.75, "Sine", False),
    "Studio": (0.6, 5.0, 0.9, "Sine", False),
    "Through-Zero": (0.0, 3.0, 0.0, "Sine", True),
}


class Flanger(Effect):
    TYPE = "flanger"
    LABEL = "Flanger"
    CATEGORY = "Modulation"
    MODELS = tuple(FLANGER_VOICES)
    STEREO = True
    PARAMS = (
        PL("rate", "Rate", 0.03, 8.0, 0.3, "Hz", aliases=RATE_ALIASES),
        P("depth", "Depth", 0.0, 1.0, 0.7, aliases=DEPTH_ALIASES),
        P("manual", "Manual", 0.0, 1.0, 0.3, aliases=("delay", "predelay", "time",
                                                      "range", "offset", "center")),
        P("feedback", "Regen", -1.0, 1.0, 0.5, aliases=("regen", "resonance", "res",
                                                        "fb", "regeneration",
                                                        "emphasis")),
        P("spread", "Spread", 0.0, 1.0, 0.5, aliases=SPREAD_ALIASES),
        MIX,
    )

    CHUNK = 24

    def __init__(self):
        super().__init__()
        self.mix = 0.5

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._line = DelayLine(int(sr * 0.03) + max_block + 16, 2)
        self._line.prepare(max(max_block, self.CHUNK))
        self._lfo = ShapeLFO()
        self._lfo.prepare(sr, max_block)

    def reset(self):
        self._line.reset()
        self._lfo.reset()

    def _process(self, x, n):
        min_ms, sweep_ms, fbs, shape, tz = FLANGER_VOICES.get(self.model,
                                                              FLANGER_VOICES["Classic"])
        k = 1e-3 * self.sr
        rate = clampf(self.rate, 0.01, 12.0)
        ph = self._lfo.phases(rate, n)
        spread = 0.5 * clampf(self.spread, 0.0, 1.0)
        lfo_l = 0.5 + 0.5 * ShapeLFO.shape(ph, shape)
        lfo_r = 0.5 + 0.5 * ShapeLFO.shape(ph + spread, shape)
        depth = clampf(self.depth, 0.0, 1.0)
        manual = clampf(self.manual, 0.0, 1.0)
        m = clampf(self.mix, 0.0, 1.0)
        if tz:
            # Through-zero: the dry signal is itself delayed by the middle of
            # the sweep, so the wet tap passes in front of it and back --
            # the tape-flanging "zero point" no single-tap flanger reaches.
            centre = (0.4 + 2.0 * manual) * k
            self._line.write(x)
            dl = centre + (lfo_l - 0.5) * 2.0 * centre * depth + n
            dr = centre + (lfo_r - 0.5) * 2.0 * centre * depth + n
            dry = self._line.read_fixed(int(round(centre)) + n, n)
            wet = np.stack((self._line.read_modulated(np.maximum(dl, n), n)[0],
                            self._line.read_modulated(np.maximum(dr, n), n)[1]))
            return dry * (1.0 - m * 0.5) + wet * m
        c = self.CHUNK
        lo = max(min_ms * k, c + 1.0) + manual * 3.0 * k
        span = sweep_ms * k * depth
        fb = clampf(self.feedback, -1.0, 1.0) * fbs
        out = np.empty((2, n))
        for a, b in grid(self._t, n, c):
            kk = b - a
            dl = lo + span * lfo_l[a:b]
            dr = lo + span * lfo_r[a:b]
            wl = self._line.read_modulated(dl, kk)[0]
            wr = self._line.read_modulated(dr, kk)[1]
            wet = np.stack((wl, wr))
            self._line.write(x[:, a:b] + wet * fb)
            out[:, a:b] = wet
        return x * (1.0 - m * 0.5) + out * m


# ==========================================================================
# Phaser
# ==========================================================================

# name: (stages, lo_hz, hi_hz, feedback scale, stage spread)
PHASER_VOICES = {
    "Four-Stage": (4, 180.0, 2400.0, 0.55, 0.12),
    "Stone": (4, 140.0, 3200.0, 0.85, 0.10),
    "Six-Stage": (6, 160.0, 3000.0, 0.6, 0.15),
    "Twelve-Stage": (12, 120.0, 4000.0, 0.4, 0.20),
}


class Phaser(Effect):
    """All-pass phaser with feedback.

    The four all-pass stages keep their own state (each stage's one state
    variable, which, like a capacitor voltage, stays meaningful when the
    stage's frequency moves), so a fast deep sweep glides instead of
    bursting. Within each 64-sample chunk the stages and the feedback around
    them are solved exactly as one filter, H = P / (Q - g P), so the feedback
    is sample-accurate rather than a chunk late (a chunk of delay in the loop
    would be a comb filter, not a phaser): the combined filter's starting
    state comes from the stage states, and the stages are then run over the
    chunk to carry their states forward. Stages are spread slightly in
    frequency, as mismatched parts are in the hardware. Above four stages the
    extra ones run as a separate cascade outside the loop.
    """

    TYPE = "phaser"
    LABEL = "Phaser"
    CATEGORY = "Modulation"
    MODELS = tuple(PHASER_VOICES)
    STEREO = True
    PARAMS = (
        PL("rate", "Rate", 0.03, 10.0, 0.5, "Hz", aliases=RATE_ALIASES),
        P("depth", "Depth", 0.0, 1.0, 0.8, aliases=DEPTH_ALIASES),
        P("manual", "Manual", 0.0, 1.0, 0.3, aliases=("center", "offset", "freq",
                                                      "frequency", "range")),
        P("feedback", "Color", 0.0, 1.0, 0.4, aliases=("color", "colour", "regen",
                                                       "resonance", "res", "fb",
                                                       "emphasis")),
        P("spread", "Spread", 0.0, 1.0, 0.4, aliases=SPREAD_ALIASES),
        PC("shape", "Shape", ("Sine", "Triangle"), aliases=("wave", "waveform")),
        MIX,
    )

    CHUNK = 64   # 1.3 ms: far finer than any sweep a phaser makes

    def __init__(self):
        super().__init__()
        self.mix = 0.5

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._st = np.zeros((2, 4))           # per-stage TDF-II state
        self._tail_zi = np.zeros((2, 4, 2))   # (channel, section, state)
        self._lfo = ShapeLFO()
        self._lfo.prepare(sr, max_block)

    def reset(self):
        self._st[...] = 0.0
        self._tail_zi[...] = 0.0
        self._lfo.reset()

    @staticmethod
    def _free_response(al, st, g, k):
        """First k output samples of the loop with zero input, from the
        stage states. Each stage: y = a u + s, s' = u - a y. The loop input
        is g * wet, and wet depends on it instantly (gain prod(a)), so each
        sample solves that one-unknown equation."""
        s = list(st)
        d = 1.0
        for a in al:
            d *= a
        out = np.zeros(k)
        for n in range(k):
            # wet = c + d*u with u = g*wet  ->  wet = c / (1 - g d)
            c = 0.0
            for i, a in enumerate(al):
                c = a * c + s[i]
            wet = c / (1.0 - g * d)
            u = g * wet
            v = u
            for i, a in enumerate(al):
                y = a * v + s[i]
                s[i] = v - a * y
                v = y
            out[n] = wet
        return out

    def _process(self, x, n):
        stages, lo, hi, fbs, sp = PHASER_VOICES.get(self.model, PHASER_VOICES["Four-Stage"])
        ph = self._lfo.phases(clampf(self.rate, 0.01, 15.0), n)
        spread = 0.5 * clampf(self.spread, 0.0, 1.0)
        depth = clampf(self.depth, 0.0, 1.0)
        manual = clampf(self.manual, 0.0, 1.0)
        g = clampf(self.feedback, 0.0, 1.0) * fbs * 0.7
        m = clampf(self.mix, 0.0, 1.0)
        span = math.log(hi / lo)
        mult = [math.exp(sp * (i - 1.5) * 0.5) for i in range(4)]
        out = np.empty((2, n))
        # Longer chains cost more per chunk; their sweep is no faster, so
        # they update on a coarser grid.
        chunk = self.CHUNK if stages <= 4 else self.CHUNK * 2
        for a, b in grid(self._t, n, chunk):
            if self._cell is None or cell_start(self._t, a, chunk):
                cell = []
                for ch in range(2):
                    w = 0.5 + 0.5 * ShapeLFO.shape(ph[a:a + 1] + (spread if ch else 0.0),
                                                   self.shape)[0]
                    pos = manual * (1.0 - depth) + w * depth
                    fc = lo * math.exp(span * clampf(pos, 0.0, 1.0))
                    al = [allpass1(fc * mult[i], self.sr) for i in range(4)]
                    num = np.array([1.0])
                    den = np.array([1.0])
                    for c in al:
                        num = np.convolve(num, [c, 1.0])
                        den = np.convolve(den, [1.0, c])
                    qd = den - g * num
                    cell.append((fc, al, num / qd[0], qd / qd[0]))
                self._cell = cell
            for ch in range(2):
                fc, al, bq, aq = self._cell[ch]
                # Combined filter's DF2T state from the stages' states.
                fr = self._free_response(al, self._st[ch], g, 4)
                zi = np.array([fr[j] + sum(aq[i] * fr[j - i] for i in range(1, j + 1))
                               for j in range(4)])
                xs = x[ch, a:b]
                wet, _ = lfilter(bq, aq, xs, zi=zi)
                # Carry each stage's state across the chunk.
                v = xs + g * wet
                for i, c in enumerate(al):
                    v, zf = lfilter([c, 1.0], [1.0, c], v, zi=self._st[ch, i:i + 1])
                    self._st[ch, i] = zf[0]
                y = x[ch, a:b] * (1.0 - m) + wet * m
                if stages > 4:
                    # The remaining stages, outside the loop, on the wet part
                    # only. They move slowly within a chunk, so plain
                    # second-order sections are fine here.
                    dry = (1.0 - m) * x[ch, a:b]
                    pairs = (stages - 4) // 2
                    sos = np.empty((pairs, 6))
                    for j in range(pairs):
                        a1 = allpass1(fc * mult[j % 4] * 1.07, self.sr)
                        a2 = allpass1(fc * mult[(j + 2) % 4] * 0.93, self.sr)
                        sos[j, :3] = (a1 * a2, a1 + a2, 1.0)
                        sos[j, 3:] = (1.0, a1 + a2, a1 * a2)
                    wetp, self._tail_zi[ch, :pairs] = sosfilt(
                        sos, y - dry, zi=self._tail_zi[ch, :pairs])
                    y = dry + wetp
                out[ch, a:b] = y
        return out * 1.25


# ==========================================================================
# Photo-cell vibe
# ==========================================================================

class Vibe(Effect):
    """Four unequal phase-shift stages swept by a lamp and photocells.

    The stage capacitors differ by up to ~70:1, so the four notches sit far
    apart and move at different speeds -- the throb that a regular phaser
    does not have. The lamp's light follows the LFO, and the photocell's
    resistance follows the light with a fast attack and slow recovery, so the
    sweep is lopsided. "Chorus" mixes the shifted signal with the dry one;
    "Vibrato" is the shifted signal alone (pitch wobble).
    """

    TYPE = "vibe"
    LABEL = "Photo Vibe"
    CATEGORY = "Modulation"
    MODELS = ("Chorus", "Vibrato")
    PARAMS = (
        PL("rate", "Speed", 0.3, 12.0, 2.5, "Hz", aliases=RATE_ALIASES),
        P("depth", "Intensity", 0.0, 1.0, 0.75, aliases=DEPTH_ALIASES),
        LEVEL,
    )

    # Stage capacitors (uF) -- the ratios are what matter.
    CAPS = (0.015, 0.22, 0.00047, 0.0047)
    CHUNK = 32

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._zi = np.zeros((4, self.channels, 1))   # per-stage TDF-II state
        self._lfo = LFO()
        self._lfo.prepare(sr, max_block)
        self._att = OnePole(4.0)
        self._rel = OnePole(45.0)
        self._att.prepare(sr, 1)
        self._rel.prepare(sr, 1)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        self._zi[...] = 0.0
        self._att.reset()
        self._rel.reset()
        self._lfo.phase = 0.0

    def _process(self, x, n):
        lamp = 0.5 + 0.5 * self._lfo.block(clampf(self.rate, 0.1, 20.0), n)
        lamp = lamp * lamp
        light = np.maximum(self._att.process(lamp), self._rel.process(lamp))
        depth = clampf(self.depth, 0.0, 1.0)
        # Cell resistance: bright = low R. 6k..220k, narrowed by Intensity.
        r_lo, r_hi = 6e3, 6e3 + 2.14e5 * (0.25 + 0.75 * depth)
        y = np.empty_like(x)
        chorus = self.model == "Chorus"
        for a, b in grid(self._t, n, self.CHUNK):
            if self._cell is None or cell_start(self._t, a, self.CHUNK):
                L = float(light[a])
                self._cell = r_hi * (r_lo / r_hi) ** L
            R = self._cell
            # Stage by stage, each keeping its own state: the 0.22 uF stage
            # sits at a few Hz, where a combined high-order filter would be
            # numerically fragile under modulation.
            v = x[:, a:b]
            for i, c in enumerate(self.CAPS):
                al = allpass1(1.0 / (TWO_PI * R * c * 1e-6), self.sr)
                v, self._zi[i] = lfilter([al, 1.0], [1.0, al], v, axis=-1, zi=self._zi[i])
            y[:, a:b] = (x[:, a:b] + v) * 0.5 if chorus else v
        self._out.set(_level_gain(self.level) * (1.2 if chorus else 1.0))
        return y * self._out.ramp(n)


# ==========================================================================
# Tremolo
# ==========================================================================

class Tremolo(Effect):
    """Bias (smooth), opto (lopsided, as a lamp and photocell), harmonic
    (lows and highs pulse in opposition), square (choppy) and auto-pan."""

    TYPE = "tremolo"
    LABEL = "Tremolo"
    CATEGORY = "Modulation"
    MODELS = ("Bias", "Opto", "Harmonic", "Square", "Pan")
    STEREO = True
    PARAMS = (
        PL("rate", "Rate", 0.3, 16.0, 4.5, "Hz", aliases=RATE_ALIASES),
        P("depth", "Depth", 0.0, 1.0, 0.6, aliases=DEPTH_ALIASES),
        PC("shape", "Shape", SHAPES, aliases=("wave", "waveform")),
        P("smooth", "Smooth", 0.0, 1.0, 0.5, aliases=("softness", "edge")),
        P("spread", "Spread", 0.0, 1.0, 0.0, aliases=SPREAD_ALIASES),
        LEVEL,
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._lfo = ShapeLFO()
        self._lfo.prepare(sr, max_block)
        self._att = OnePole(3.0)
        self._rel = OnePole(30.0)
        self._att.prepare(sr, 1)
        self._rel.prepare(sr, 1)
        self._lp = Biquad("lowpass", 700.0, 0.707)
        self._hp = Biquad("highpass", 700.0, 0.707)
        self._lp.prepare(sr, 2)
        self._hp.prepare(sr, 2)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        self._lfo.reset()
        for f in (self._att, self._rel, self._lp, self._hp):
            f.reset()

    def _process(self, x, n):
        model = self.model
        ph = self._lfo.phases(clampf(self.rate, 0.05, 25.0), n)
        shape = "Square" if model == "Square" else self.shape
        if model in ("Bias", "Opto", "Harmonic", "Pan") and self.shape == "Sine":
            shape = "Sine"
        w = 0.5 + 0.5 * ShapeLFO.shape(ph, shape, self.smooth)
        d = clampf(self.depth, 0.0, 1.0)
        if model == "Opto":
            w = np.maximum(self._att.process(w * w), self._rel.process(w * w))
            w = np.clip(w, 0.0, 1.0)
        if model == "Pan":
            gl = 1.0 - d * w
            gr = 1.0 - d * (1.0 - w)
            y = np.stack((x[0] * gl, x[1] * gr))
        elif model == "Harmonic":
            lo = self._lp.process(x)
            hi = self._hp.process(x)
            y = lo * (1.0 - d * w) + hi * (1.0 - d * (1.0 - w))
            y *= 1.0 + 0.5 * d
        else:
            sp = clampf(self.spread, 0.0, 1.0)
            if sp > 0.0:
                w2 = 0.5 + 0.5 * ShapeLFO.shape(ph + 0.5 * sp, shape, self.smooth)
                y = np.stack((x[0] * (1.0 - d * w), x[1] * (1.0 - d * w2)))
            else:
                g = 1.0 - d * w
                if model == "Bias":
                    # A biased power tube pulses a touch harder on the up-swing.
                    g = g * (1.0 + 0.08 * d * (w - 0.5))
                y = x * g
        self._out.set(_level_gain(self.level) * (1.0 + 0.4 * d))
        return y * self._out.ramp(n)


# ==========================================================================
# Rotary speaker
# ==========================================================================

class Rotary(Effect):
    """Rotating horn and drum.

    A crossover splits the signal at 800 Hz. Each rotor has its own speed,
    and when you switch between slow and fast each one gets there at its
    own pace (the light horn in about half a second, the heavy drum in two
    or three) -- the speed-up swirl is the sound. The horn throws both
    Doppler pitch wobble and loudness change at two mics on opposite sides,
    the drum mostly loudness.
    """

    TYPE = "rotary"
    LABEL = "Rotary"
    CATEGORY = "Modulation"
    MODELS = ("Standard", "Vintage")
    STEREO = True
    PARAMS = (
        PC("speed", "Speed", ("Slow", "Fast", "Brake"), default="Slow",
           aliases=("rotor", "speedswitch", "fastslow", "mode")),
        P("rate", "Rate Trim", 0.0, 1.0, 0.5, aliases=("rate", "speedtrim",
                                                       "frequency")),
        P("accel", "Ramp", 0.0, 1.0, 0.5, aliases=("acceleration", "ramp", "rise",
                                                   "inertia")),
        P("balance", "Horn/Drum", 0.0, 1.0, 0.5, aliases=("balance", "horn", "mix2")),
        P("drive", "Drive", 0.0, 1.0, 0.15, aliases=("gain", "overdrive", "tube")),
        P("spread", "Mic Spread", 0.0, 1.0, 0.8, aliases=SPREAD_ALIASES),
        MIX,
    )

    SPEEDS = {"Slow": (0.83, 0.67), "Fast": (6.8, 5.9), "Brake": (0.0, 0.0)}

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._xl = [Biquad("lowpass", 800.0, 0.707), Biquad("lowpass", 800.0, 0.707)]
        self._xh = [Biquad("highpass", 800.0, 0.707), Biquad("highpass", 800.0, 0.707)]
        for f in self._xl + self._xh:
            f.prepare(sr, 1)
        self._horn = DelayLine(int(sr * 0.01) + max_block + 16, 1)
        self._drum = DelayLine(int(sr * 0.01) + max_block + 16, 1)
        self._horn.prepare(max_block)
        self._drum.prepare(max_block)
        self._idx = np.arange(1, max_block + 1, dtype=np.float64)
        self._f = [0.83, 0.67]      # current rotor speeds (Hz)
        self._ph = [0.0, 0.25]      # rotor angles (cycles)

    def reset(self):
        for f in self._xl + self._xh:
            f.reset()
        self._horn.reset()
        self._drum.reset()

    def _rotor(self, k, target, tau_s, n):
        """Per-sample speed (exponential approach) and angle for rotor k."""
        f0 = self._f[k]
        if tau_s <= 0.0:
            f = np.full(n, target)
        else:
            f = target + (f0 - target) * np.exp(-self._idx[:n] / (tau_s * self.sr))
        ph = self._ph[k] + np.cumsum(f) / self.sr
        self._f[k] = float(f[-1])
        self._ph[k] = float(ph[-1]) % 1.0
        return ph

    def _process(self, x, n):
        mono = np.mean(x, axis=0)
        drive = clampf(self.drive, 0.0, 1.0)
        if drive > 0.0:
            gd = 1.0 + drive * 8.0
            mono = np.tanh(mono * gd) / math.tanh(gd) * (1.0 + drive)
        lo = self._xl[1].process(self._xl[0].process(mono))
        hi = self._xh[1].process(self._xh[0].process(mono))
        hs, ds = self.SPEEDS.get(self.speed, self.SPEEDS["Slow"])
        trim = 0.7 + 0.6 * clampf(self.rate, 0.0, 1.0)
        acc = 0.25 + 1.5 * clampf(self.accel, 0.0, 1.0)
        vintage = self.model == "Vintage"
        ph_h = self._rotor(0, hs * trim, 0.55 * acc * (1.3 if vintage else 1.0), n)
        ph_d = self._rotor(1, ds * trim, 2.6 * acc * (1.3 if vintage else 1.0), n)
        sp = 0.5 * clampf(self.spread, 0.0, 1.0)
        k = 1e-3 * self.sr
        self._horn.write(hi[None, :])
        self._drum.write(lo[None, :])
        out = np.empty((2, n))
        for ch, mic in enumerate((-sp / 2.0, sp / 2.0)):
            ah = TWO_PI * (ph_h + mic)
            ad = TWO_PI * (ph_d + mic)
            # Horn: ~0.45 ms Doppler swing, strong beaming.
            dh = (1.2 + 0.45 * np.sin(ah)) * k + n
            h = self._horn.read_modulated(dh, n)[0] * (1.0 + 0.55 * np.cos(ah))
            dd = (1.2 + 0.12 * np.sin(ad)) * k + n
            d = self._drum.read_modulated(dd, n)[0] * (1.0 + 0.3 * np.cos(ad))
            bal = clampf(self.balance, 0.0, 1.0)
            out[ch] = h * (0.5 + bal) * 0.4 + d * (1.5 - bal) * 0.4
        m = clampf(self.mix, 0.0, 1.0)
        if m >= 0.999:
            return out
        return x * (1.0 - m) + out * m
