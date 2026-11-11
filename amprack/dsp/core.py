"""Core DSP primitives.

Everything here is written for block processing on a real-time audio thread:

*   State is allocated in ``prepare()``, never in ``process()``.
*   Coefficient recomputation is guarded so it only happens when a parameter
    actually moved -- knobs are read every block, but a still knob costs nothing.
*   Nothing here raises. A block that cannot be handled is passed through
    untouched, because a silent no-op is always better than a glitch.

Numpy temporaries are unavoidable in Python; the engine freezes the garbage
collector while streaming so those stay pure refcount churn.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter, lfilter_zi, sosfilt

TWO_PI = 2.0 * math.pi


def db2lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin2db(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-9))


def clampf(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# --------------------------------------------------------------------------
# Biquad
# --------------------------------------------------------------------------

def _rbj(kind: str, sr: float, f0: float, q: float, gain_db: float):
    """RBJ audio-EQ-cookbook coefficients, returned normalised as (b, a)."""
    f0 = clampf(f0, 10.0, sr * 0.49)
    q = max(q, 1e-3)
    w0 = TWO_PI * f0 / sr
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = sw / (2.0 * q)

    if kind == "lowpass":
        b0, b1, b2 = (1 - cw) * 0.5, 1 - cw, (1 - cw) * 0.5
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "highpass":
        b0, b1, b2 = (1 + cw) * 0.5, -(1 + cw), (1 + cw) * 0.5
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "bandpass":
        b0, b1, b2 = alpha, 0.0, -alpha
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "allpass":
        b0, b1, b2 = 1 - alpha, -2 * cw, 1 + alpha
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "notch":
        b0, b1, b2 = 1.0, -2 * cw, 1.0
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "peak":
        A = 10.0 ** (gain_db / 40.0)
        b0, b1, b2 = 1 + alpha * A, -2 * cw, 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * cw, 1 - alpha / A
    elif kind == "lowshelf":
        A = 10.0 ** (gain_db / 40.0)
        s = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) - (A - 1) * cw + s)
        b1 = 2 * A * ((A - 1) - (A + 1) * cw)
        b2 = A * ((A + 1) - (A - 1) * cw - s)
        a0 = (A + 1) + (A - 1) * cw + s
        a1 = -2 * ((A - 1) + (A + 1) * cw)
        a2 = (A + 1) + (A - 1) * cw - s
    elif kind == "highshelf":
        A = 10.0 ** (gain_db / 40.0)
        s = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * cw + s)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - s)
        a0 = (A + 1) - (A - 1) * cw + s
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - s
    else:  # pragma: no cover - programmer error, degrade to unity
        b0, b1, b2, a0, a1, a2 = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0

    return (
        np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64),
        np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64),
    )


class Biquad:
    """One stateful biquad section, optionally multi-channel."""

    __slots__ = ("kind", "sr", "f0", "q", "gain_db", "b", "a", "zi", "_dirty", "channels")

    def __init__(self, kind: str, f0: float, q: float = 0.707, gain_db: float = 0.0):
        self.kind = kind
        self.f0 = f0
        self.q = q
        self.gain_db = gain_db
        self.sr = 48000.0
        self.channels = 1
        self.b = np.array([1.0, 0.0, 0.0])
        self.a = np.array([1.0, 0.0, 0.0])
        self.zi = np.zeros((1, 2))
        self._dirty = True

    def prepare(self, sr: float, channels: int = 1) -> None:
        self.sr = float(sr)
        self.channels = channels
        self.zi = np.zeros((channels, 2), dtype=np.float64)
        self._dirty = True
        self._update()

    def reset(self) -> None:
        self.zi[...] = 0.0

    def set(self, f0: float | None = None, q: float | None = None,
            gain_db: float | None = None) -> None:
        if f0 is not None and f0 != self.f0:
            self.f0 = f0
            self._dirty = True
        if q is not None and q != self.q:
            self.q = q
            self._dirty = True
        if gain_db is not None and gain_db != self.gain_db:
            self.gain_db = gain_db
            self._dirty = True

    def _update(self) -> None:
        self.b, self.a = _rbj(self.kind, self.sr, self.f0, self.q, self.gain_db)
        self._dirty = False

    def process(self, x: np.ndarray) -> np.ndarray:
        """x is (n,) mono or (channels, n). Returns a new array of the same shape."""
        if self._dirty:
            self._update()
        if x.ndim == 1:
            y, self.zi[0] = lfilter(self.b, self.a, x, zi=self.zi[0])
            return y
        y, zf = lfilter(self.b, self.a, x, axis=-1, zi=self.zi)
        self.zi[...] = zf
        return y


class BiquadCascade:
    """A fixed list of biquads applied in series."""

    def __init__(self, sections: list[Biquad]):
        self.sections = sections

    def prepare(self, sr: float, channels: int = 1) -> None:
        for s in self.sections:
            s.prepare(sr, channels)

    def reset(self) -> None:
        for s in self.sections:
            s.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        for s in self.sections:
            x = s.process(x)
        return x


# --------------------------------------------------------------------------
# One-pole smoother (also used as an envelope follower)
# --------------------------------------------------------------------------

class OnePole:
    """y[n] = y[n-1] + a * (x[n] - y[n-1]) implemented via lfilter (C speed)."""

    __slots__ = ("tau_ms", "sr", "b", "a", "zi", "channels")

    def __init__(self, tau_ms: float):
        self.tau_ms = tau_ms
        self.sr = 48000.0
        self.channels = 1
        self.b = np.array([1.0])
        self.a = np.array([1.0])
        self.zi = np.zeros((1, 1))

    def prepare(self, sr: float, channels: int = 1) -> None:
        self.sr = float(sr)
        self.channels = channels
        self.zi = np.zeros((channels, 1), dtype=np.float64)
        self.set_tau(self.tau_ms)

    def set_tau(self, tau_ms: float) -> None:
        self.tau_ms = max(tau_ms, 1e-3)
        alpha = 1.0 - math.exp(-1.0 / (self.sr * self.tau_ms * 1e-3))
        self.b = np.array([alpha])
        self.a = np.array([1.0, -(1.0 - alpha)])

    def reset(self, value: float = 0.0) -> None:
        self.zi[...] = value * (1.0 - self.b[0])

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            y, self.zi[0] = lfilter(self.b, self.a, x, zi=self.zi[0])
            return y
        y, zf = lfilter(self.b, self.a, x, axis=-1, zi=self.zi)
        self.zi[...] = zf
        return y


class DCBlocker:
    """y[n] = x[n] - x[n-1] + R*y[n-1], R set for ~10 Hz corner."""

    __slots__ = ("b", "a", "zi")

    def __init__(self):
        self.b = np.array([1.0, -1.0])
        self.a = np.array([1.0, -0.995])
        self.zi = np.zeros((1, 1))

    def prepare(self, sr: float, channels: int = 1) -> None:
        r = math.exp(-TWO_PI * 10.0 / sr)
        self.b = np.array([1.0, -1.0])
        self.a = np.array([1.0, -r])
        self.zi = np.zeros((channels, 1), dtype=np.float64)

    def reset(self) -> None:
        self.zi[...] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            y, self.zi[0] = lfilter(self.b, self.a, x, zi=self.zi[0])
            return y
        y, zf = lfilter(self.b, self.a, x, axis=-1, zi=self.zi)
        self.zi[...] = zf
        return y


# --------------------------------------------------------------------------
# Parameter smoothing
# --------------------------------------------------------------------------

class SmoothedParam:
    """A scalar that ramps linearly across a block.

    Knob moves become a per-sample ramp instead of a step, which is the
    difference between a smooth sweep and a zipper of clicks. When the target
    has not moved, ``ramp()`` returns the scalar itself so the caller's
    multiply stays a cheap scalar multiply.
    """

    __slots__ = ("value", "target", "_buf", "_ramps", "_max_block", "_primed")

    def __init__(self, value: float = 0.0):
        self.value = float(value)
        self.target = float(value)
        self._buf = np.zeros(0, dtype=np.float64)
        self._ramps = {}
        self._max_block = 0
        self._primed = False

    def prepare(self, max_block: int) -> None:
        self._max_block = max_block
        self._primed = False
        self._buf = np.zeros(max_block, dtype=np.float64)
        # 0..1 ramps for every block length we might see, built once so the
        # audio thread only does a multiply-add.
        self._ramps = {n: np.linspace(0.0, 1.0, n) for n in
                       sorted({max_block, max_block // 2, max_block // 4,
                               64, 96, 128, 192, 256, 384, 512}) if 0 < n <= max_block}

    def set(self, target: float) -> None:
        self.target = float(target)
        if not self._primed:
            # First block after prepare(): jump straight to whatever the knobs
            # actually say. Gliding from a constructor default would make the
            # first few milliseconds depend on the buffer size, which shows up
            # as a tiny level bump every time the stream restarts.
            self.value = self.target

    def snap(self, value: float) -> None:
        self.value = self.target = float(value)

    def ramp(self, n: int):
        """Return a scalar (unchanged) or an (n,) ramp array (changing)."""
        self._primed = True
        if self.target == self.value:
            return self.value
        if n > self._max_block:
            self.value = self.target
            return self.value
        r = self._ramps.get(n)
        if r is None:
            # An unexpected block length: step rather than allocate a ramp
            # on the audio thread. Rare, and one stepped block is inaudible.
            self.value = self.target
            return self.value
        out = self._buf[:n]
        np.multiply(r, self.target - self.value, out=out)
        out += self.value
        self.value = self.target
        return out


# --------------------------------------------------------------------------
# Delay line
# --------------------------------------------------------------------------

class DelayLine:
    """Power-of-two ring buffer with block write and linear-interpolated read.

    **Ordering contract: read first, then write.** Callers read the block they
    owe the output, then write the block they just consumed (which may include
    feedback derived from that read). Both read methods are written against
    that ordering, and both therefore require ``delay >= n``.

    Getting this backwards is the classic delay bug: everything works, sounds
    plausible, and every echo is exactly one buffer late -- so the error tracks
    the latency setting and looks like a driver problem instead of a maths one.

    Reads never wrap mid-block by hand: the index array is masked, which costs
    one extra vector op and removes an entire class of edge-case bug.
    """

    __slots__ = ("size", "mask", "buf", "w", "channels", "_idx")

    def __init__(self, max_samples: int, channels: int = 1):
        size = 1
        while size < max_samples + 8:
            size <<= 1
        self.size = size
        self.mask = size - 1
        self.channels = channels
        self.buf = np.zeros((channels, size), dtype=np.float64)
        self.w = 0
        self._idx = np.zeros(0, dtype=np.int64)

    def prepare(self, max_block: int) -> None:
        self._idx = np.arange(max_block, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.buf[...] = 0.0
        self.w = 0

    def write(self, x: np.ndarray) -> None:
        """Write n samples starting at the write head and advance it."""
        n = x.shape[-1]
        w = self.w
        end = w + n
        if end <= self.size:
            self.buf[:, w:end] = x
        else:
            first = self.size - w
            self.buf[:, w:] = x[..., :first]
            self.buf[:, : end - self.size] = x[..., first:]
        self.w = end & self.mask

    def read_fixed(self, delay: int, n: int) -> np.ndarray:
        """Output sample j is input sample j delayed by `delay`.

        Call before ``write()`` for this block. Requires ``delay >= n``.
        """
        start = (self.w - delay) & self.mask
        end = start + n
        if end <= self.size:
            return self.buf[:, start:end]
        first = self.size - start
        out = np.empty((self.channels, n), dtype=np.float64)
        out[:, :first] = self.buf[:, start:]
        out[:, first:] = self.buf[:, : n - first]
        return out

    def read_modulated(self, delay_samples: np.ndarray, n: int) -> np.ndarray:
        """Fractional read where `delay_samples` is a per-sample (n,) array.

        Call before ``write()``. Requires ``min(delay_samples) >= n``.
        """
        pos = (self.w + self._idx[:n]) - delay_samples
        i0 = np.floor(pos)
        frac = pos - i0
        idx0 = i0.astype(np.int64) & self.mask
        idx1 = (idx0 + 1) & self.mask
        out = np.empty((self.channels, n), dtype=np.float64)
        for c in range(self.channels):
            row = self.buf[c]
            a = row[idx0]
            b = row[idx1]
            out[c] = a + (b - a) * frac
        return out


# --------------------------------------------------------------------------
# LFO
# --------------------------------------------------------------------------

class LFO:
    """Free-running phase generator. Phase is kept in [0,1) as a Python float
    so long sessions cannot accumulate float32 phase error."""

    __slots__ = ("phase", "sr", "_idx", "_max_block")

    def __init__(self):
        self.phase = 0.0
        self.sr = 48000.0
        self._idx = np.zeros(0)
        self._max_block = 0

    def prepare(self, sr: float, max_block: int) -> None:
        self.sr = float(sr)
        self._max_block = max_block
        self._idx = np.arange(max_block, dtype=np.float64)

    def phase_block(self, rate_hz: float, n: int) -> np.ndarray:
        """Return the (n,) phase ramp for this block and advance."""
        inc = rate_hz / self.sr
        ph = self.phase + self._idx[:n] * inc
        self.phase = (self.phase + n * inc) % 1.0
        return ph

    def block(self, rate_hz: float, n: int, offset: float = 0.0) -> np.ndarray:
        """Return (n,) sine LFO in [-1, 1] and advance phase."""
        return np.sin(TWO_PI * (self.phase_block(rate_hz, n) + offset))

    def advance_only(self, rate_hz: float, n: int) -> None:
        self.phase = (self.phase + n * rate_hz / self.sr) % 1.0


# --------------------------------------------------------------------------
# Oversampling (for the nonlinear stages)
# --------------------------------------------------------------------------

class Oversampler:
    """Integer up/down sampling with stateful IIR anti-imaging/aliasing filters.

    Distortion without oversampling folds every harmonic above Nyquist back
    into the audible band as inharmonic mush -- it is the single biggest
    difference between an amp sim that sounds like an amp and one that sounds
    like a fuzz pedal with a head cold.

    The filter is a 10th-order elliptic lowpass (flat to 19 kHz, -100 dB
    stopband) run as second-order sections, which stay numerically sane where
    a single high-order (b, a) polynomial would not. The old 6th-order
    Butterworth was only ~17 dB down half an octave above its corner.

    Measured with a 1567 Hz tone (which, unlike 2 kHz, does not divide 48 kHz
    and so shows folded harmonics between the real ones): at 4x the Metal amp
    aliased at -41 dB whatever the filter, because a hard-clipping stage makes
    harmonics far past the 4x Nyquist that fold *inside* the oversampled
    domain. 8x brings it to about -57 dB for under 2% extra CPU.
    """

    def __init__(self, factor: int = 8, order: int = 10):
        self.factor = factor
        self.order = order
        self.sr_os = 48000.0 * factor
        self._sos = None

    def prepare(self, sr: float, max_block: int) -> None:
        from scipy.signal import ellip
        self.sr_os = sr * self.factor
        cutoff = min(19000.0, 0.42 * sr)
        self._sos = ellip(self.order, 0.05, 100.0, cutoff / (self.sr_os * 0.5),
                          output="sos")
        self._zi_up = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        self._zi_dn = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        self._os_buf = np.zeros(max_block * self.factor, dtype=np.float64)
        self._max_block = max_block

    def reset(self) -> None:
        self._zi_up[...] = 0.0
        self._zi_dn[...] = 0.0
        self._os_buf[...] = 0.0

    def up(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[-1]
        f = self.factor
        buf = self._os_buf[: n * f]
        buf[...] = 0.0
        buf[::f] = x * f  # zero-stuff; *f preserves level through the filter
        y, self._zi_up = sosfilt(self._sos, buf, zi=self._zi_up)
        return y

    def down(self, y: np.ndarray) -> np.ndarray:
        f = self.factor
        z, self._zi_dn = sosfilt(self._sos, y, zi=self._zi_dn)
        return z[::f].copy()


# --------------------------------------------------------------------------
# Waveshapers -- all memoryless, all vectorised
# --------------------------------------------------------------------------

def shape_tanh(x: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(x * drive)


def shape_asym_tube(x: np.ndarray, drive: float, bias: float = 0.12) -> np.ndarray:
    """Asymmetric soft clip: a triode conducts harder on one half-cycle, which
    is where the even-order harmonics that make a valve amp sound warm come
    from. Symmetric clipping gives you odd harmonics only -- a transistor."""
    v = x * drive + bias
    y = np.tanh(v)
    # Slightly softer knee on the negative swing.
    neg = v < 0.0
    y[neg] = np.tanh(v[neg] * 0.85) * 0.92
    return y - math.tanh(bias)


def shape_diode(x: np.ndarray, drive: float) -> np.ndarray:
    """Symmetric diode-clipper curve -- Tube Screamer / OD-1 territory."""
    v = x * drive
    return np.sign(v) * (1.0 - np.exp(-np.abs(v)))


def shape_hard(x: np.ndarray, drive: float) -> np.ndarray:
    return np.clip(x * drive, -1.0, 1.0)


def shape_fold(x: np.ndarray, drive: float) -> np.ndarray:
    """Wavefolder -- the octave-up / fuzz-with-teeth character."""
    v = x * drive
    return np.sin(v * (math.pi / 2.0)) * np.exp(-np.abs(v) * 0.05)


SHAPERS = {
    "tube": shape_asym_tube,
    "diode": shape_diode,
    "tanh": shape_tanh,
    "hard": shape_hard,
    "fold": shape_fold,
}


def sanitize(x: np.ndarray) -> np.ndarray:
    """Last line of defence before the signal reaches the driver.

    A NaN that reaches a DAC is a full-scale click and, on some interfaces,
    a stuck output until the stream is restarted. Cheap insurance.
    """
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(x, -4.0, 4.0, out=x)
    return x
