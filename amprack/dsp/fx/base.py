"""Shared machinery for the slot effects.

Every effect in ``amprack.dsp.fx`` is a :class:`Effect`: a ``Block`` that
also *describes itself* -- its models and its parameters, with ranges and
units -- so that the UI can build a card for it and the song-tone mapper can
translate someone else's knob values onto it without a hand-written table per
effect.

Effects always see 2-D audio, ``(channels, n)``. The slot rack decides the
channel count (1 in front of the amp, 2 after the cab) and an effect that is
stereo by nature (``STEREO = True``) is always run with two channels; in a mono
slot the rack feeds it the same signal on both sides and sums the result.

Two rules that every effect here follows, because the harness checks them:

*   **Block-size invariance.** Anything that has to run in sub-blocks (a
    feedback loop shorter than the driver buffer, a filter whose coefficients
    move) runs on a grid of *absolute* sample positions (:func:`grid`), not on
    offsets inside the current buffer. The output is then the same sample for
    sample at a 64 or a 2048 buffer, so a sound does not change with the
    latency setting.
*   **Nothing allocates in proportion to time.** State lives in ``prepare``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from ..blocks import Block
from ..core import OnePole, TWO_PI, clampf


# --------------------------------------------------------------------------
# Parameter description
# --------------------------------------------------------------------------

class Param:
    """One user-facing control.

    ``kind`` is ``"lin"`` (linear between lo and hi), ``"log"`` (geometric,
    for Hz and ms), ``"choice"`` (one of ``choices``) or ``"bool"``.
    ``aliases`` are other names the same control goes by on other gear
    (lower-case, no spaces), used by the tone mapper.
    """

    __slots__ = ("name", "label", "lo", "hi", "default", "kind", "choices",
                 "unit", "aliases")

    def __init__(self, name, label, lo=0.0, hi=1.0, default=0.5, kind="lin",
                 choices=(), unit="", aliases=()):
        self.name = name
        self.label = label
        self.lo = lo
        self.hi = hi
        self.default = default
        self.kind = kind
        self.choices = tuple(choices)
        self.unit = unit
        self.aliases = tuple(aliases)

    def clamp(self, v):
        if self.kind == "choice":
            return v if v in self.choices else self.default
        if self.kind == "bool":
            return bool(v)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return self.default
        if not math.isfinite(v):
            return self.default
        return clampf(v, self.lo, self.hi)

    def from_unit(self, u: float):
        """Map 0..1 onto this parameter's range (log params geometrically)."""
        u = clampf(float(u), 0.0, 1.0)
        if self.kind == "log":
            return self.lo * (self.hi / self.lo) ** u
        if self.kind == "choice":
            i = min(int(u * len(self.choices)), len(self.choices) - 1)
            return self.choices[i]
        if self.kind == "bool":
            return u >= 0.5
        return self.lo + (self.hi - self.lo) * u

    def to_unit(self, v) -> float:
        if self.kind == "log":
            return math.log(max(float(v), 1e-9) / self.lo) / math.log(self.hi / self.lo)
        if self.kind == "choice":
            return self.choices.index(v) / max(len(self.choices) - 1, 1) \
                if v in self.choices else 0.0
        if self.kind == "bool":
            return 1.0 if v else 0.0
        return (float(v) - self.lo) / (self.hi - self.lo)


def P(name, label, lo=0.0, hi=1.0, default=0.5, unit="", aliases=()):
    return Param(name, label, lo, hi, default, "lin", (), unit, aliases)


def PL(name, label, lo, hi, default, unit="", aliases=()):
    return Param(name, label, lo, hi, default, "log", (), unit, aliases)


def PC(name, label, choices, default=None, aliases=()):
    return Param(name, label, 0, 1, default if default is not None else choices[0],
                 "choice", choices, "", aliases)


MIX = P("mix", "Mix", 0.0, 1.0, 1.0, aliases=("blend", "drywet", "wet", "effect",
                                               "effectlevel", "e.level", "elevel"))
LEVEL = P("level", "Level", 0.0, 1.0, 0.5, aliases=("volume", "output", "out",
                                                      "outputlevel", "vol", "gain2"))


# --------------------------------------------------------------------------
# Effect base
# --------------------------------------------------------------------------

class Effect(Block):
    TYPE = ""          # registry key, stored in presets
    LABEL = ""         # Tanin's display name
    CATEGORY = ""      # Filter, Pitch, Modulation, Time, Dirt, Dynamics, EQ, Utility
    MODELS: tuple = ("Standard",)
    PARAMS: tuple = ()
    STEREO = False     # True: makes a stereo image; always run on 2 channels

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.model = self.MODELS[0]
        self.channels = 1
        for p in self.PARAMS:
            setattr(self, p.name, p.default)
        self._t = 0    # absolute sample position, for the chunk grid
        self._cell = None  # coefficients held for the current grid cell

    # -- description --------------------------------------------------------

    @classmethod
    def param(cls, name: str) -> Param | None:
        for p in cls.PARAMS:
            if p.name == name:
                return p
        return None

    def get_params(self) -> dict:
        d = {"model": self.model}
        for p in self.PARAMS:
            d[p.name] = getattr(self, p.name)
        return d

    def set_params(self, d: dict) -> None:
        """Set from a dict; anything missing returns to its default and
        anything malformed falls back to the default (this can run on the
        audio thread, so it must not raise)."""
        if not isinstance(d, dict):
            d = {}
        m = d.get("model", self.MODELS[0])
        self.model = m if m in self.MODELS else self.MODELS[0]
        for p in self.PARAMS:
            setattr(self, p.name, p.clamp(d.get(p.name, p.default)))

    # -- processing ---------------------------------------------------------

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        if self.STEREO:
            self.channels = 2
        self._t = 0
        self._cell = None

    def process(self, x):
        if not self.enabled:
            return x
        n = x.shape[-1]
        if n > self.max_block or n == 0:
            return x
        y = self._process(x, n)
        self._t += n
        return y

    def g(self, name: str) -> float:
        """A numeric parameter, clamped to its declared range."""
        p = self.param(name)
        v = getattr(self, name)
        return p.clamp(v) if p is not None else v


# --------------------------------------------------------------------------
# Absolute-position chunk grid
# --------------------------------------------------------------------------

def cell_start(t0: int, a: int, c: int) -> bool:
    """True when piece ``a`` of a block starting at ``t0`` opens a new grid
    cell. Swept filters compute coefficients only there and hold them for
    the whole cell, so a cell split across two buffers sounds identical."""
    return (t0 + a) % c == 0


def grid(t0: int, n: int, c: int):
    """Split a block that starts at absolute sample ``t0`` into pieces whose
    edges fall on multiples of ``c``. Identical pieces whatever the buffer."""
    a = 0
    while a < n:
        b = min(n, a + c - ((t0 + a) % c))
        yield a, b
        a = b


# --------------------------------------------------------------------------
# Time-varying IIR filter
# --------------------------------------------------------------------------

class TVFilter:
    """An IIR filter whose coefficients may change between chunks.

    State is kept as the last ``order`` inputs and outputs (direct form I),
    which mean the same thing whatever the coefficients are; before each chunk
    they are converted to lfilter's transposed-direct-form state for the new
    coefficients. That is what makes a swept filter (wah, phaser, auto-filter)
    glide instead of ticking: direct form II state carried over a coefficient
    change is a different signal, and a fast sweep turns it into crackle.
    """

    __slots__ = ("order", "channels", "xh", "yh", "_idx", "_mask")

    def __init__(self, order: int, channels: int = 1):
        self.order = order
        self.channels = channels
        self.xh = np.zeros((channels, order))   # xh[:, j] = x[n-1-j]
        self.yh = np.zeros((channels, order))
        m = np.arange(order)[:, None]
        j = np.arange(order)[None, :]
        idx = m + j + 1
        self._mask = idx <= order
        self._idx = np.where(self._mask, idx, 0)

    def reset(self):
        self.xh[...] = 0.0
        self.yh[...] = 0.0

    def _zi(self, b, a):
        """DF2T state for (b, a) from the DF1 history, all channels: (C, N).
        zi[m] = sum_j b[m+j+1] x[n-1-j] - a[m+j+1] y[n-1-j]."""
        bm = np.where(self._mask, b[self._idx], 0.0)
        am = np.where(self._mask, a[self._idx], 0.0)
        return self.xh @ bm.T - self.yh @ am.T

    def _push(self, x, y, sl):
        N = self.order
        k = x.shape[-1]
        if k >= N:
            self.xh[sl] = x[..., :-N - 1:-1]
            self.yh[sl] = y[..., :-N - 1:-1]
        else:
            self.xh[sl] = np.concatenate((x[..., ::-1], self.xh[sl]), axis=-1)[..., :N]
            self.yh[sl] = np.concatenate((y[..., ::-1], self.yh[sl]), axis=-1)[..., :N]

    def run(self, b, a, x, ch: int = 0):
        """Filter one channel's chunk (k,) with coefficients (b, a), a[0]=1."""
        zi = self._zi(b, a)[ch]
        y, _ = lfilter(b, a, x, zi=zi)
        self._push(x, y, ch)
        return y

    def run_all(self, b, a, x):
        """Filter every channel's chunk (C, k) with the same coefficients."""
        y, _ = lfilter(b, a, x, axis=-1, zi=self._zi(b, a))
        self._push(x, y, slice(None))
        return y


# --------------------------------------------------------------------------
# Coefficient helpers (all return normalised b, a with a[0] = 1)
# --------------------------------------------------------------------------

def svf_coeffs(kind: str, fc: float, q: float, sr: float):
    """Bilinear (pre-warped) 2nd-order sections, RBJ shapes."""
    fc = clampf(fc, 10.0, sr * 0.45)
    w0 = TWO_PI * fc / sr
    cw, sw = math.cos(w0), math.sin(w0)
    al = sw / (2.0 * max(q, 0.05))
    a0 = 1.0 + al
    if kind == "lp":
        b = ((1 - cw) * 0.5, 1 - cw, (1 - cw) * 0.5)
    elif kind == "hp":
        b = ((1 + cw) * 0.5, -(1 + cw), (1 + cw) * 0.5)
    elif kind == "bp":       # constant 0 dB peak
        b = (al, 0.0, -al)
    elif kind == "notch":
        b = (1.0, -2 * cw, 1.0)
    else:
        b = (1.0, 0.0, 0.0)
        return np.array(b), np.array((1.0, 0.0, 0.0))
    a = (1.0, -2 * cw / a0, (1 - al) / a0)
    return np.array(b) / a0, np.array(a)


def allpass1(fc: float, sr: float) -> float:
    """Coefficient of a first-order allpass (a + z^-1)/(1 + a z^-1) whose
    phase passes -90 degrees at fc."""
    t = math.tan(math.pi * clampf(fc, 10.0, sr * 0.45) / sr)
    return (t - 1.0) / (t + 1.0)


# --------------------------------------------------------------------------
# Envelope follower and LFO shapes
# --------------------------------------------------------------------------

class EnvFollower:
    """Peak follower with separate attack and release.

    Two one-poles run side by side on the rectified signal and the larger
    wins: the fast one carries the rise, the slow one holds the fall. A third,
    short smoother takes the audio-rate ripple off. Fully vectorised and
    independent of the block size because all three are plain filters.
    """

    def __init__(self, attack_ms=3.0, release_ms=120.0):
        self.att = OnePole(attack_ms)
        self.rel = OnePole(release_ms)
        self.smooth = OnePole(4.0)

    def prepare(self, sr):
        for f in (self.att, self.rel, self.smooth):
            f.prepare(sr, 1)

    def reset(self):
        for f in (self.att, self.rel, self.smooth):
            f.reset()

    def set(self, attack_ms, release_ms):
        if abs(self.att.tau_ms - attack_ms) > 1e-6:
            self.att.set_tau(attack_ms)
        if abs(self.rel.tau_ms - release_ms) > 1e-6:
            self.rel.set_tau(release_ms)

    def process(self, mono: np.ndarray) -> np.ndarray:
        r = np.abs(mono)
        e = np.maximum(self.att.process(r), self.rel.process(r) * 1.6)
        return self.smooth.process(e)


SHAPES = ("Sine", "Triangle", "Square", "Saw Up", "Saw Down", "Random")


class ShapeLFO:
    """Phase accumulator with selectable shapes, kept as (whole cycles,
    fractional phase) so a random step is a function of the cycle number and
    therefore the same at any buffer size."""

    def __init__(self):
        self.phase = 0.0
        self.cycle = 0
        self.sr = 48000.0
        self._idx = np.zeros(0)

    def prepare(self, sr, max_block):
        self.sr = float(sr)
        self._idx = np.arange(max_block + 1, dtype=np.float64)

    def reset(self):
        self.phase = 0.0
        self.cycle = 0

    def phases(self, rate_hz: float, n: int):
        """(n,) absolute phase in cycles (cycle number + fraction)."""
        inc = rate_hz / self.sr
        ph = self.phase + self._idx[:n] * inc
        tot = self.phase + n * inc
        whole = math.floor(tot)
        out = ph + self.cycle
        self.cycle += whole
        self.phase = tot - whole
        return out

    @staticmethod
    def shape(ph: np.ndarray, kind: str, smooth: float = 0.0) -> np.ndarray:
        """Bipolar [-1, 1] wave of the absolute phase ``ph``."""
        f = ph - np.floor(ph)
        if kind == "Triangle":
            return 1.0 - 4.0 * np.abs(f - 0.5)
        if kind == "Square":
            # Softened square: a hard square on a gain or a filter is a click.
            k = 4.0 + 60.0 * (1.0 - clampf(smooth, 0.0, 1.0))
            return np.tanh(k * np.sin(TWO_PI * f)) / math.tanh(k)
        if kind == "Saw Up":
            return 2.0 * f - 1.0
        if kind == "Saw Down":
            return 1.0 - 2.0 * f
        if kind == "Random":
            c = np.floor(ph)
            v = np.sin(c * 12.9898 + 78.233) * 43758.5453
            return 2.0 * (v - np.floor(v)) - 1.0
        return np.sin(TWO_PI * f)
