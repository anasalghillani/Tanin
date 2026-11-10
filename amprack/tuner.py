"""Chromatic tuner.

Detection runs on the UI thread against a snapshot of the input ring, so a
slow frame never touches the audio callback.

The pitch estimator is McLeod's normalised square-difference function. Plain
autocorrelation picks the octave below the fundamental often enough on a wound
low E to be genuinely annoying; NSDF's normalisation is what fixes that.
"""

from __future__ import annotations

import math

import numpy as np

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

STANDARD_TUNING = (
    ("E2", 82.41), ("A2", 110.00), ("D3", 146.83),
    ("G3", 196.00), ("B3", 246.94), ("E4", 329.63),
)


class InputRing:
    """Lock-free-enough single-producer ring for the tuner and scope.

    The reader may tear across a write; for a pitch estimate that is at worst
    one bad frame out of ten per second, and the alternative is a lock on the
    audio thread, which is never worth it.
    """

    def __init__(self, size: int = 16384):
        self.size = size
        self.buf = np.zeros(size, dtype=np.float32)
        self.w = 0

    def push(self, x: np.ndarray, n: int) -> None:
        w = self.w
        end = w + n
        if end <= self.size:
            self.buf[w:end] = x
        else:
            first = self.size - w
            self.buf[w:] = x[:first]
            self.buf[:end - self.size] = x[first:]
        self.w = end % self.size

    def snapshot(self, n: int) -> np.ndarray:
        n = min(n, self.size)
        w = self.w
        start = (w - n) % self.size
        if start + n <= self.size:
            return self.buf[start:start + n].astype(np.float64)
        first = self.size - start
        out = np.empty(n, dtype=np.float64)
        out[:first] = self.buf[start:]
        out[first:] = self.buf[:n - first]
        return out


def detect_pitch(x: np.ndarray, sr: float,
                 fmin: float = 60.0, fmax: float = 1400.0) -> tuple[float, float]:
    """Return (frequency_hz, clarity 0..1). frequency is 0.0 when unsure."""
    n = len(x)
    if n < 1024:
        return 0.0, 0.0
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 1e-4:
        return 0.0, 0.0

    size = 1
    while size < 2 * n:
        size <<= 1
    spec = np.fft.rfft(x, size)
    acf = np.fft.irfft(spec * np.conj(spec), size)[:n]

    # NSDF denominator: cumulative power of both windows at each lag.
    power = np.cumsum(x[::-1] ** 2)[::-1]
    total = float(np.sum(x * x))
    m = total + power - np.cumsum(x ** 2) + 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        nsdf = 2.0 * acf / m
    nsdf = np.nan_to_num(nsdf, nan=0.0)

    lag_min = max(2, int(sr / fmax))
    lag_max = min(n - 2, int(sr / fmin))
    if lag_max <= lag_min + 2:
        return 0.0, 0.0

    seg = nsdf[lag_min:lag_max]
    # Peak picking: collect local maxima, take the first that clears 90% of
    # the global best. Taking the global best alone is what causes octave
    # errors -- the true fundamental is usually the *earliest* strong peak.
    peaks = []
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.0:
            peaks.append(i)
    if not peaks:
        return 0.0, 0.0
    best = max(seg[i] for i in peaks)
    if best < 0.35:
        return 0.0, 0.0
    thresh = best * 0.90
    chosen = next(i for i in peaks if seg[i] >= thresh)

    # Parabolic interpolation around the chosen peak.
    y0, y1, y2 = seg[chosen - 1], seg[chosen], seg[chosen + 1]
    denom = y0 - 2 * y1 + y2
    shift = 0.0 if abs(denom) < 1e-12 else 0.5 * (y0 - y2) / denom
    lag = lag_min + chosen + shift
    if lag <= 0:
        return 0.0, 0.0
    return float(sr / lag), float(min(max(y1, 0.0), 1.0))


def freq_to_note(freq: float, a4: float = 440.0) -> tuple[str, int, float]:
    """Return (note name, octave, cents deviation)."""
    if freq <= 0:
        return "--", 0, 0.0
    midi = 69.0 + 12.0 * math.log2(freq / a4)
    nearest = int(round(midi))
    cents = (midi - nearest) * 100.0
    name = NOTE_NAMES[nearest % 12]
    octave = nearest // 12 - 1
    return name, octave, cents
