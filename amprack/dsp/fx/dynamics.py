"""Dynamics and EQ: compressor family, gate, graphic and parametric EQ."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import sosfilt

from ..core import OnePole, SmoothedParam, _rbj, clampf, db2lin
from .base import Effect, P, PC, PL
from .filters import _level_gain

# ==========================================================================
# Compressor
# ==========================================================================

# model: (detector_tau_ms, knee_db, ratio_override, attack scale, opto)
COMP_MODELS = {
    "Studio": (2.0, 6.0, 0.0, 1.0, False),
    "Opto": (8.0, 10.0, 0.0, 2.0, True),
    "Squash": (1.0, 3.0, 8.0, 0.5, False),
    "Limiter": (0.5, 2.0, 30.0, 0.2, False),
}


class Comp(Effect):
    """Feed-forward compressor family.

    Studio: clean VCA-style with a soft knee. Opto: slow, program-dependent
    release (a quick initial recovery then a long tail, like a photocell
    that has been lit hard). Squash: the pedal sound -- a fixed high ratio
    where Sustain pushes the signal into it. Limiter: brick-ish ceiling.
    Stereo input is detected linked, so the image does not wander.
    """

    TYPE = "comp"
    LABEL = "Compressor"
    CATEGORY = "Dynamics"
    MODELS = tuple(COMP_MODELS)
    PARAMS = (
        P("sustain", "Sustain", 0.0, 1.0, 0.5,
          aliases=("threshold", "thresh", "sensitivity", "sens", "compression",
                   "comp", "squeeze", "amount", "peakreduction", "sustain")),
        PL("ratio", "Ratio", 1.0, 20.0, 4.0, ":1", aliases=("ratio",)),
        PL("attack_ms", "Attack", 0.1, 100.0, 8.0, "ms", aliases=("attack",)),
        PL("release_ms", "Release", 20.0, 2000.0, 180.0, "ms", aliases=("release",)),
        P("level", "Level", 0.0, 1.0, 0.5, aliases=("output", "volume", "gain",
                                                    "makeup", "out", "vol")),
        P("mix", "Mix", 0.0, 1.0, 1.0, aliases=("blend", "parallel", "drywet")),
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._det = OnePole(2.0)
        self._att = OnePole(8.0)
        self._rel = OnePole(180.0)
        self._rel2 = OnePole(1500.0)
        for f in (self._det, self._att, self._rel, self._rel2):
            f.prepare(sr, 1)
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)
        self.gain_reduction_db = 0.0

    def reset(self):
        for f in (self._det, self._att, self._rel, self._rel2):
            f.reset()

    def _process(self, x, n):
        tau, knee, rfix, ascale, opto = COMP_MODELS.get(self.model, COMP_MODELS["Studio"])
        if abs(self._det.tau_ms - tau) > 1e-9:
            self._det.set_tau(tau)
        rect = np.max(np.abs(x), axis=0)
        env = self._det.process(rect)
        env_db = 20.0 * np.log10(env + 1e-9)
        s = clampf(self.sustain, 0.0, 1.0)
        thr = -6.0 - 44.0 * s
        ratio = rfix if rfix else clampf(self.ratio, 1.0, 50.0)
        over = env_db - thr
        slope = 1.0 / ratio - 1.0
        gr = np.where(over <= -knee / 2, 0.0,
                      np.where(over >= knee / 2, over * slope,
                               slope * (over + knee / 2) ** 2 / (2 * knee)))
        self._att.set_tau(clampf(self.attack_ms, 0.05, 500.0) * ascale)
        self._rel.set_tau(clampf(self.release_ms, 5.0, 5000.0))
        fast = self._att.process(gr)
        slow = self._rel.process(gr)
        if opto:
            # Two-stage release: half of it recovers at the set speed, the
            # rest over 1.5 s, so a long hard passage stays held down.
            slow = 0.5 * slow + 0.5 * self._rel2.process(gr)
        sm = np.minimum(fast, slow)
        self.gain_reduction_db = float(sm[-1])
        # Auto make-up: about half the static reduction at the threshold
        # setting, so turning Sustain up does not just turn the pedal down.
        makeup = -thr * (1.0 - 1.0 / ratio) * 0.3
        g = 10.0 ** ((sm + makeup) / 20.0)
        self._out.set(_level_gain(self.level))
        wet = x * g * self._out.ramp(n)
        m = clampf(self.mix, 0.0, 1.0)
        if m >= 0.999:
            return wet
        return x * (1.0 - m) + wet * m


# ==========================================================================
# Gate
# ==========================================================================

class Gate(Effect):
    """Hysteresis gate, same law as the main gate (it opens at the threshold
    and closes 8 dB below it, so a note at the edge does not chatter)."""

    TYPE = "gate"
    LABEL = "Gate"
    CATEGORY = "Dynamics"
    MODELS = ("Standard",)
    PARAMS = (
        P("threshold_db", "Threshold", -90.0, -20.0, -55.0, "dB",
          aliases=("threshold", "thresh", "sensitivity", "sens", "level")),
        PL("release_ms", "Release", 5.0, 1000.0, 120.0, "ms",
           aliases=("release", "decay", "hold")),
        PL("attack_ms", "Attack", 0.1, 50.0, 2.0, "ms", aliases=("attack",)),
    )

    HYST_DB = 8.0

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._fast = OnePole(0.5)
        self._slow = OnePole(15.0)
        self._sm = OnePole(2.0)
        self._rel = OnePole(120.0)
        for f in (self._fast, self._slow, self._sm, self._rel):
            f.prepare(sr, 1)
        self._idx = np.arange(max_block)
        self._open = 0.0

    def reset(self):
        for f in (self._fast, self._slow, self._sm, self._rel):
            f.reset()
        self._open = 0.0

    def _process(self, x, n):
        rect = np.max(np.abs(x), axis=0)
        env = np.maximum(self._fast.process(rect), self._slow.process(rect))
        on = db2lin(clampf(self.threshold_db, -120.0, 0.0))
        off = on * db2lin(-self.HYST_DB)
        decided = (env > on) | (env < off)
        last = np.where(decided, self._idx[:n], -1)
        np.maximum.accumulate(last, out=last)
        state = np.where(env > on, 1.0, 0.0)
        target = np.where(last >= 0, state[np.maximum(last, 0)], self._open)
        self._open = float(target[-1])
        self._sm.set_tau(clampf(self.attack_ms, 0.05, 200.0))
        self._rel.set_tau(clampf(self.release_ms, 1.0, 5000.0))
        g = np.maximum(self._sm.process(target), self._rel.process(target))
        return x * g


# ==========================================================================
# Graphic EQ
# ==========================================================================

GEQ_BANDS = {
    "5-Band": (100.0, 400.0, 800.0, 1600.0, 3200.0),
    "6-Band": (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0),
    "7-Band": (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0),
    "8-Band": (62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
    "10-Band": (31.0, 62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0,
                16000.0),
}
MAX_BANDS = 10


def _band_params():
    out = []
    for i in range(MAX_BANDS):
        out.append(P(f"b{i + 1}", f"Band {i + 1}", -15.0, 15.0, 0.0, "dB",
                     aliases=(f"band{i + 1}", f"b{i + 1}", f"eq{i + 1}")))
    return tuple(out)


class GraphicEQ(Effect):
    TYPE = "geq"
    LABEL = "Graphic EQ"
    CATEGORY = "EQ"
    MODELS = tuple(GEQ_BANDS)
    PARAMS = _band_params() + (
        P("level_db", "Level", -15.0, 15.0, 0.0, "dB", aliases=("level", "volume",
                                                                "gain", "output",
                                                                "master")),
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._key = None
        self._sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        self._zi = np.zeros((MAX_BANDS, self.channels, 2))
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        self._zi[...] = 0.0

    @classmethod
    def bands(cls, model) -> tuple:
        return GEQ_BANDS.get(model, GEQ_BANDS["7-Band"])

    def _design(self):
        bands = self.bands(self.model)
        gains = tuple(round(clampf(getattr(self, f"b{i + 1}"), -24.0, 24.0), 3)
                      for i in range(len(bands)))
        key = (self.model, gains)
        if key == self._key:
            return
        # Octave-ish bands: Q follows the spacing so neighbours overlap the
        # way a real graphic's sliders do.
        ratio = bands[1] / bands[0]
        q = math.sqrt(ratio) / (ratio - 1.0) * 1.1
        sec = []
        for f, g in zip(bands, gains):
            b, a = _rbj("peak", self.sr, f, q, g)
            sec.append(np.concatenate((b, a)))
        self._sos = np.array(sec)
        self._key = key

    def _process(self, x, n):
        self._design()
        k = self._sos.shape[0]
        y, self._zi[:k] = sosfilt(self._sos, x, axis=-1, zi=self._zi[:k])
        self._out.set(db2lin(clampf(self.level_db, -24.0, 24.0)))
        return y * self._out.ramp(n)


# ==========================================================================
# Parametric EQ
# ==========================================================================

class ParametricEQ(Effect):
    TYPE = "peq"
    LABEL = "Parametric EQ"
    CATEGORY = "EQ"
    MODELS = ("Standard",)
    PARAMS = (
        PL("low_cut", "Low Cut", 10.0, 500.0, 10.0, "Hz", aliases=("lowcut", "hpf",
                                                                   "highpass")),
        PL("low_freq", "Low Freq", 40.0, 500.0, 100.0, "Hz", aliases=("lowfreq",
                                                                     "bassfreq")),
        P("low_db", "Low", -15.0, 15.0, 0.0, "dB", aliases=("low", "bass", "lowgain")),
        PL("mid1_freq", "Mid 1 Freq", 100.0, 3000.0, 500.0, "Hz",
           aliases=("midfreq", "mid1freq", "lowmidfreq", "freq1", "frequency")),
        P("mid1_db", "Mid 1", -15.0, 15.0, 0.0, "dB", aliases=("mid", "mid1",
                                                              "lowmid", "gain1",
                                                              "middle")),
        PL("mid1_q", "Mid 1 Q", 0.3, 8.0, 1.0, "", aliases=("q", "q1", "midq",
                                                           "bandwidth", "width")),
        PL("mid2_freq", "Mid 2 Freq", 400.0, 8000.0, 2000.0, "Hz",
           aliases=("mid2freq", "highmidfreq", "freq2")),
        P("mid2_db", "Mid 2", -15.0, 15.0, 0.0, "dB", aliases=("mid2", "highmid",
                                                              "gain2")),
        PL("mid2_q", "Mid 2 Q", 0.3, 8.0, 1.0, "", aliases=("q2",)),
        PL("high_freq", "High Freq", 1500.0, 12000.0, 4000.0, "Hz",
           aliases=("highfreq", "treblefreq")),
        P("high_db", "High", -15.0, 15.0, 0.0, "dB", aliases=("high", "treble",
                                                             "highgain")),
        PL("high_cut", "High Cut", 2000.0, 20000.0, 20000.0, "Hz",
           aliases=("highcut", "lpf", "lowpass")),
        P("level_db", "Level", -15.0, 15.0, 0.0, "dB", aliases=("level", "volume",
                                                               "output", "gain")),
    )

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        self._key = None
        self._zi = np.zeros((6, self.channels, 2))
        self._sos = None
        self._out = SmoothedParam(1.0)
        self._out.prepare(max_block)

    def reset(self):
        self._zi[...] = 0.0

    def _design(self):
        vals = tuple(round(float(self.g(p.name)), 4) for p in self.PARAMS)
        if vals == self._key:
            return
        sr = self.sr
        secs = [
            _rbj("highpass", sr, self.g("low_cut"), 0.707, 0.0),
            _rbj("lowshelf", sr, self.g("low_freq"), 0.707, self.g("low_db")),
            _rbj("peak", sr, self.g("mid1_freq"), self.g("mid1_q"), self.g("mid1_db")),
            _rbj("peak", sr, self.g("mid2_freq"), self.g("mid2_q"), self.g("mid2_db")),
            _rbj("highshelf", sr, self.g("high_freq"), 0.707, self.g("high_db")),
            _rbj("lowpass", sr, min(self.g("high_cut"), sr * 0.45), 0.707, 0.0),
        ]
        if self.g("low_cut") <= 10.5:
            secs[0] = (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        if self.g("high_cut") >= 19999.0:
            secs[5] = (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        self._sos = np.array([np.concatenate(s) for s in secs])
        self._key = vals

    def _process(self, x, n):
        self._design()
        y, self._zi = sosfilt(self._sos, x, axis=-1, zi=self._zi)
        self._out.set(db2lin(self.g("level_db")))
        return y * self._out.ramp(n)
