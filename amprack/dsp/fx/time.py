"""Time effects: a many-voiced echo (digital, bucket-brigade, tape, multi-head,
oil can, lo-fi, multi-tap, ping-pong) and a reverb with real spring tanks
next to a feedback-delay-network plate, hall, room and chamber."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import sosfilt

from ..core import Biquad, DelayLine, LFO, OnePole, SmoothedParam, clampf, db2lin
from .base import MIX, Effect, P, PC, PL, grid

# ==========================================================================
# Echo
# ==========================================================================

# model: (lp_lo, lp_span, hp, saturate, wow, flutter, chorus)
ECHO_VOICES = {
    "Digital":    (3000.0, 6.0, 40.0, 0.0, 0.0, 0.0, 0.0),
    "Analog":     (1100.0, 4.5, 120.0, 1.0, 0.0, 0.0, 0.25),
    "Memory":     (1300.0, 4.0, 110.0, 1.0, 0.0, 0.0, 1.0),
    "Tape":       (900.0, 5.0, 150.0, 1.2, 1.0, 1.0, 0.0),
    "Multi-Head": (850.0, 5.0, 160.0, 1.2, 0.8, 1.0, 0.0),
    "Oil Can":    (700.0, 3.0, 200.0, 0.8, 2.5, 3.0, 0.0),
    "Lo-Fi":      (900.0, 2.5, 300.0, 1.5, 0.3, 0.5, 0.0),
    "Multi-Tap":  (2500.0, 6.0, 60.0, 0.0, 0.0, 0.0, 0.0),
    "Ping-Pong":  (2500.0, 6.0, 60.0, 0.0, 0.0, 0.0, 0.0),
}

HEADS = ("1", "2", "3", "1+2", "2+3", "1+3", "1+2+3")


class Echo(Effect):
    TYPE = "echo"
    LABEL = "Echo"
    CATEGORY = "Time"
    MODELS = tuple(ECHO_VOICES)
    STEREO = True
    PARAMS = (
        PL("time_ms", "Time", 20.0, 2000.0, 380.0, "ms",
           aliases=("time", "delay", "delaytime", "length", "repeat", "repeatrate")),
        P("feedback", "Repeats", 0.0, 1.05, 0.35,
          aliases=("repeats", "regen", "regeneration", "fb", "sustain",
                   "intensity", "decay")),
        P("tone", "Tone", 0.0, 1.0, 0.5, aliases=("filter", "color", "colour",
                                                  "treble", "eq", "hicut", "damp",
                                                  "damping")),
        P("mod", "Mod", 0.0, 1.0, 0.25, aliases=("modulation", "depth", "wow",
                                                 "flutter", "chorus", "vibrato",
                                                 "warble", "age")),
        PL("mod_rate", "Mod Rate", 0.1, 8.0, 0.8, "Hz",
           aliases=("rate", "speed", "modrate")),
        PC("heads", "Heads", HEADS, default="3", aliases=("head", "mode", "tap",
                                                          "pattern", "select")),
        P("spread", "Spread", 0.0, 1.0, 0.5, aliases=("stereo", "width", "spread")),
        MIX,
    )

    CHUNK = 256
    MAX_MS = 2100.0

    def __init__(self):
        super().__init__()
        self.mix = 0.3

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._line = DelayLine(int(sr * self.MAX_MS * 1e-3) + max_block + 64, 2)
        self._line.prepare(max(max_block, self.CHUNK))
        self._lp = Biquad("lowpass", 4000.0, 0.707)
        self._hp = Biquad("highpass", 100.0, 0.707)
        self._lp.prepare(sr, 2)
        self._hp.prepare(sr, 2)
        self._wow = LFO()
        self._flut = LFO()
        self._chor = LFO()
        for l in (self._wow, self._flut, self._chor):
            l.prepare(sr, max(max_block, self.CHUNK))
        self._time = OnePole(120.0)   # glide on time changes, like a tape motor
        self._time.prepare(sr, 1)
        self._time_init = False
        self._held = np.zeros(2)

    def reset(self):
        self._line.reset()
        self._lp.reset()
        self._hp.reset()
        self._time_init = False
        self._held[:] = 0.0

    def _taps(self):
        """(fraction of time, gain) for each read head / tap."""
        m = self.model
        if m == "Multi-Head":
            sel = self.heads if self.heads in HEADS else "3"
            on = [h for h in ("1", "2", "3") if h in sel]
            return [(int(h) / 3.0, 1.0 / math.sqrt(len(on))) for h in on]
        if m == "Multi-Tap":
            return [(0.25, 0.45), (0.5, 0.6), (0.75, 0.5), (1.0, 0.8)]
        return [(1.0, 1.0)]

    def _process(self, x, n):
        voice = ECHO_VOICES.get(self.model, ECHO_VOICES["Digital"])
        lp_lo, lp_span, hp, sat, wow, flut, chor = voice
        sr = self.sr
        target = clampf(self.time_ms, 20.0, self.MAX_MS - 50.0) * 1e-3 * sr
        if not self._time_init:
            self._time.reset(target)
            self._time.zi[...] = target * (1.0 - self._time.b[0])
            self._time_init = True
        tline = self._time.process(np.full(n, target))
        modd = clampf(self.mod, 0.0, 1.0)
        rate = clampf(self.mod_rate, 0.05, 12.0)
        wobble = np.zeros(n)
        if wow > 0.0 and modd > 0.0:
            wobble += self._wow.block(rate * 0.6, n) * (0.0035 * wow * modd * sr)
            wobble += self._flut.block(rate * 7.5, n) * (0.00035 * flut * modd * sr)
        else:
            self._wow.advance_only(rate * 0.6, n)
            self._flut.advance_only(rate * 7.5, n)
        chorus = None
        if chor > 0.0 and modd > 0.0:
            chorus = self._chor.block(rate * 1.4, n) * (0.0025 * chor * modd * sr)
        else:
            self._chor.advance_only(rate * 1.4, n)
        tone = clampf(self.tone, 0.0, 1.0)
        self._lp.set(f0=lp_lo * (lp_span ** tone))
        self._hp.set(f0=hp)
        fb = clampf(self.feedback, 0.0, 1.05)
        if sat <= 0.0:
            fb = min(fb, 0.95)
        taps = self._taps()
        pingpong = self.model == "Ping-Pong"
        lofi = self.model == "Lo-Fi"
        spread = clampf(self.spread, 0.0, 1.0)
        out = np.empty((2, n))
        for a, b in grid(self._t, n, self.CHUNK):
            k = b - a
            base = tline[a:b] + wobble[a:b]
            wet = np.zeros((2, k))
            last = None
            for frac, g in taps:
                d = np.maximum(base * frac, k + 2.0)
                if chorus is not None:
                    dl = d + chorus[a:b]
                    dr = d - chorus[a:b]
                    r = np.stack((self._line.read_modulated(dl, k)[0],
                                  self._line.read_modulated(dr, k)[1]))
                else:
                    r = self._line.read_modulated(d, k)
                wet += r * g
                last = r
            if self.model == "Multi-Tap":
                fbsrc = last
            else:
                fbsrc = wet
            fed = self._hp.process(self._lp.process(fbsrc))
            if sat > 0.0:
                fed = np.tanh(fed * (1.0 + 0.6 * sat)) / (1.0 + 0.6 * sat) * 1.35
            if lofi:
                fed = np.round(fed * 64.0) / 64.0
            if pingpong:
                new = np.empty((2, k))
                mono_in = (x[0, a:b] + x[1, a:b]) * 0.5
                new[0] = mono_in + fed[1] * fb
                new[1] = fed[0] * fb
                self._line.write(new)
            else:
                self._line.write(x[:, a:b] + fed * fb)
            out[:, a:b] = wet
        if not pingpong and spread > 0.0 and self.model != "Multi-Head":
            # Widen by offsetting the right repeat's level a little: no
            # second line, no comb when summed to mono.
            mid = (out[0] + out[1]) * 0.5
            side = (out[0] - out[1]) * 0.5
            out = np.stack((mid + side * (1.0 + spread), mid - side * (1.0 + spread)))
        m = clampf(self.mix, 0.0, 1.0)
        return x + out * m


# ==========================================================================
# Reverb
# ==========================================================================

# FDN line lengths (ms) for the big rooms; scaled by Size.
_FDN_MS = (29.7, 37.1, 41.1, 43.7, 53.0, 59.9, 67.3, 73.1)
_HAD = None


def _hadamard8():
    global _HAD
    if _HAD is None:
        h = np.array([[1.0]])
        for _ in range(3):
            h = np.block([[h, h], [h, -h]])
        _HAD = h / math.sqrt(8.0)
    return _HAD


# model: (size_scale, rt60_lo, rt60_hi, diffusion, hf_damp_base, predelay_extra,
#         wet trim)
VERB_VOICES = {
    "Plate": (0.45, 0.8, 5.0, 0.75, 0.15, 0.0, 0.4),
    "Hall": (1.0, 1.2, 7.0, 0.6, 0.35, 12.0, 0.75),
    "Room": (0.4, 0.25, 1.6, 0.55, 0.45, 2.0, 1.4),
    "Chamber": (0.6, 0.6, 3.0, 0.65, 0.3, 6.0, 1.0),
    "Digital": (0.85, 0.8, 9.0, 0.7, 0.1, 5.0, 0.5),
}

SPRING_VOICES = ("Spring", "Tank")


class Verb(Effect):
    """Spring tanks plus a feedback-delay-network reverb.

    Spring: each spring is a loop whose delay is the time for a wave to run
    up the spring and back, with a long chain of all-pass filters inside the
    loop. All-pass filters delay some frequencies more than others
    (dispersion), so every round trip smears a click into a rising chirp --
    the "boing" of a real tank, which no plain delay or comb gives. Three
    springs of different lengths are mixed across the stereo field. "Tank"
    is the stand-alone tube unit: driven harder (the Dwell knob), wetter,
    with more drip.

    The rooms are an 8-line feedback delay network with a Hadamard mixing
    matrix (every line feeds every other at equal energy), per-line damping,
    and input diffusion. Line gains are set from the reverb time, so Decay is
    an honest RT60 rather than a feedback amount.
    """

    TYPE = "verb"
    LABEL = "Reverb"
    CATEGORY = "Time"
    MODELS = SPRING_VOICES + tuple(VERB_VOICES)
    STEREO = True
    PARAMS = (
        P("decay", "Decay", 0.0, 1.0, 0.5, aliases=("time", "length", "reverbtime",
                                                    "size2", "rt60", "sustain")),
        P("tone", "Tone", 0.0, 1.0, 0.5, aliases=("damping", "damp", "color",
                                                  "colour", "filter", "treble",
                                                  "hicut", "brightness")),
        P("size", "Size", 0.0, 1.0, 0.6, aliases=("room", "roomsize", "space")),
        PL("predelay_ms", "Pre-Delay", 1.0, 200.0, 10.0, "ms",
           aliases=("predelay", "pre", "delay")),
        P("dwell", "Dwell", 0.0, 1.0, 0.5, aliases=("drive", "input", "gain",
                                                    "intensity")),
        P("spread", "Width", 0.0, 1.0, 0.9, aliases=("width", "stereo")),
        MIX,
    )

    CHUNK = 128
    SPRING_STAGES = 48

    def __init__(self):
        super().__init__()
        self.mix = 0.3

    # -- set-up --------------------------------------------------------------

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        mb = max(max_block, self.CHUNK)
        scale = sr / 1000.0
        self._pre = DelayLine(int(sr * 0.25) + mb + 16, 2)
        self._pre.prepare(mb)
        # FDN: lines long enough for Size 1 of the biggest voice.
        self._fdn = [DelayLine(int(ms * 1.6 * scale) + mb + 16, 1) for ms in _FDN_MS]
        for d in self._fdn:
            d.prepare(mb)
        self._damp = OnePole(1.0)
        self._damp.prepare(sr, 8)
        # Input diffusers (Schroeder all-pass), short and mutually prime.
        self._diff_len = [int(t * sr / 48000.0) for t in (142, 107, 379, 277)]
        self._diff = [DelayLine(l + mb + 16, 1) for l in self._diff_len]
        for d in self._diff:
            d.prepare(mb)
        # Springs: three loops.
        self._spr_len = [int(sr * t) for t in (0.0337, 0.0411, 0.0479)]
        self._spr = [DelayLine(l + mb + 16, 1) for l in self._spr_len]
        for d in self._spr:
            d.prepare(mb)
        self._spr_sos = None
        self._spr_key = None
        self._spr_zi = None
        self._spr_in_hp = Biquad("highpass", 120.0, 0.707)
        self._spr_in_hp.prepare(sr, 1)
        self._spr_out_lp = Biquad("lowpass", 4200.0, 0.707)
        self._spr_out_lp.prepare(sr, 2)
        self._out_hp = Biquad("highpass", 90.0, 0.707)
        self._out_hp.prepare(sr, 2)
        self._mixs = SmoothedParam(0.3)
        self._mixs.prepare(max_block)

    def reset(self):
        self._pre.reset()
        for d in self._fdn + self._diff + self._spr:
            d.reset()
        self._damp.reset()
        if self._spr_zi is not None:
            self._spr_zi[...] = 0.0
        self._spr_in_hp.reset()
        self._spr_out_lp.reset()
        self._out_hp.reset()

    # -- spring --------------------------------------------------------------

    def _spring_filter(self, tone):
        """Dispersive all-pass cascade plus the loop's damping, as SOS."""
        key = round(tone, 3)
        if key == self._spr_key and self._spr_sos is not None:
            return self._spr_sos
        a = 0.62   # all-pass coefficient: sets where the chirp sits
        sec = []
        for _ in range(self.SPRING_STAGES // 2):
            # (a + z^-1)^2 / (1 + a z^-1)^2
            sec.append([a * a, 2 * a, 1.0, 1.0, 2 * a, a * a])
        # Loop damping: second-order low-pass.
        fc = 2500.0 * (2.2 ** tone)
        w0 = 2 * math.pi * fc / self.sr
        al = math.sin(w0) / (2 * 0.6)
        cw = math.cos(w0)
        b0 = (1 - cw) / 2 / (1 + al)
        sec.append([b0, 2 * b0, b0, 1.0, -2 * cw / (1 + al), (1 - al) / (1 + al)])
        sos = np.array(sec)
        if self._spr_zi is None or self._spr_zi.shape[0] != sos.shape[0]:
            self._spr_zi = np.zeros((sos.shape[0], 3, 2))
        self._spr_sos = sos
        self._spr_key = key
        return sos

    def _spring(self, src, n):
        tank = self.model == "Tank"
        dwell = clampf(self.dwell, 0.0, 1.0)
        g_in = 1.0 + dwell * (14.0 if tank else 5.0)
        mono = self._spr_in_hp.process((src[0] + src[1]) * 0.5)
        drv = np.tanh(mono * g_in) / math.tanh(g_in) * (0.6 + 0.6 * dwell)
        tone = clampf(self.tone, 0.0, 1.0)
        sos = self._spring_filter(tone)
        rt = 1.2 + 3.5 * clampf(self.decay, 0.0, 1.0)
        out = np.zeros((2, n))
        pans = ((1.0, 0.25), (0.6, 0.6), (0.25, 1.0))
        w = clampf(self.spread, 0.0, 1.0)
        for a, b in grid(self._t, n, self.CHUNK):
            k = b - a
            seg = drv[a:b]
            for i, line in enumerate(self._spr):
                L = self._spr_len[i]
                g = 10.0 ** (-3.0 * L / (rt * self.sr))
                r = line.read_fixed(L, k)[0]
                f, self._spr_zi[:, i, :] = sosfilt(sos, r, zi=self._spr_zi[:, i, :])
                line.write((seg + f * g)[None, :])
                pl, pr = pans[i]
                pl = 1.0 - w * (1.0 - pl)
                pr = 1.0 - w * (1.0 - pr)
                out[0, a:b] += f * pl
                out[1, a:b] += f * pr
        out = self._spr_out_lp.process(out)
        return out * (0.3 if tank else 0.33)

    # -- FDN -----------------------------------------------------------------

    def _room(self, src, n):
        size_s, rt_lo, rt_hi, diff, hfb, _, trim = VERB_VOICES[self.model]
        size = clampf(self.size, 0.0, 1.0)
        scale = size_s * (0.55 + 0.9 * size)
        lens = [max(int(ms * scale * self.sr / 1000.0), self.CHUNK + 2) for ms in _FDN_MS]
        rt = rt_lo * (rt_hi / rt_lo) ** clampf(self.decay, 0.0, 1.0)
        gains = np.array([10.0 ** (-3.0 * L / (rt * self.sr)) for L in lens])[:, None]
        damp = clampf(hfb + (1.0 - clampf(self.tone, 0.0, 1.0)) * 0.5, 0.0, 0.92)
        self._damp.b[0] = 1.0 - damp
        self._damp.a[1] = -damp
        H = _hadamard8()
        mono = (src[0] + src[1]) * 0.5
        out = np.zeros((2, n))
        sl = np.array([1, -1, 1, -1, 1, 1, -1, -1], dtype=np.float64) * 0.35
        sr_ = np.array([1, 1, -1, -1, 1, -1, 1, -1], dtype=np.float64) * 0.35
        w = clampf(self.spread, 0.0, 1.0)
        for a, b in grid(self._t, n, self.CHUNK):
            k = b - a
            seg = mono[a:b]
            # Input diffusion.
            for i, dl in enumerate(self._diff):
                L = max(self._diff_len[i], k + 1)
                buf = dl.read_fixed(L, k)[0]
                dl.write((seg + buf * diff)[None, :])
                seg = buf - seg * diff
            taps = np.empty((8, k))
            for i, dl in enumerate(self._fdn):
                taps[i] = dl.read_fixed(lens[i], k)[0]
            taps = self._damp.process(taps) * gains
            fbk = H @ taps
            for i, dl in enumerate(self._fdn):
                dl.write((fbk[i] + seg * 0.5)[None, :])
            left = sl @ taps
            right = sr_ @ taps
            mid = (left + right) * 0.5
            side = (left - right) * 0.5 * w
            out[0, a:b] = mid + side
            out[1, a:b] = mid - side
        return out * trim

    # -- main ----------------------------------------------------------------

    def _process(self, x, n):
        spring = self.model in SPRING_VOICES
        extra = 0.0 if spring else VERB_VOICES.get(self.model, VERB_VOICES["Hall"])[5]
        pd = int((clampf(self.predelay_ms, 0.0, 240.0) + extra) * 1e-3 * self.sr)
        pd = min(max(pd, 0), int(self.sr * 0.25))
        self._pre.write(x)
        src = self._pre.read_fixed(pd + n, n)
        if spring:
            wet = self._spring(src, n)
        else:
            if self.model not in VERB_VOICES:
                self.model = "Hall"
            wet = self._room(src, n)
        wet = self._out_hp.process(wet)
        self._mixs.set(clampf(self.mix, 0.0, 1.0))
        m = self._mixs.ramp(n)
        return x * (1.0 - 0.3 * m) + wet * m * 1.6
