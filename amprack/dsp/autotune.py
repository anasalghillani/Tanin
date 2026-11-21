"""Auto-tune: pitch detection, note snapping and a real-time PSOLA shifter.

How it works, in three steps:

1.  **Detect.** Every ``hop`` samples (about 5 ms) the last ~40 ms of input
    is run through McLeod's normalised square-difference function (the same
    method as the tuner, vectorised). That gives the sung pitch and a clarity
    score; low clarity or a quiet input counts as unvoiced (breaths, "s", "t").
2.  **Decide.** The pitch is snapped to the nearest note allowed by the key,
    scale and octave range. The *correction* (target minus sung, in
    semitones) glides toward its new value with the retune time: 0 ms is the
    hard robot effect, a few hundred ms lets vibrato through. Humanize slows
    the retune on held notes, so long notes keep their natural wobble.
3.  **Shift.** TD-PSOLA (time-domain pitch-synchronous overlap-add): the
    input is cut into two-period Hann-windowed grains, one per period, and the
    grains are laid back down closer together (higher) or further apart
    (lower). Because each grain's contents are untouched, the vowel colour
    (the formants) stays put, which is why a shifted voice does not turn into
    a chipmunk. Switching Formant off resamples the grains as well, for the
    chipmunk/monster effect on purpose.

Latency is a fixed ``LATENCY_S`` (12.5 ms). Grains need one period of audio
either side of their centre, so notes below about 160 Hz borrow the most
recent complete grain instead, which is inaudible in practice.

The per-sample work runs in a numba-compiled kernel; numba is optional and the
plain-Python fallback produces identical output, only far slower (fine for the
offline harness on a machine without numba, not for live use).

Real-time rules: all buffers are allocated in ``prepare()``; the kernel is
compiled there too (on the control thread), never in the callback. Detection
uses fixed-size FFTs, so its cost does not grow with anything.
"""

from __future__ import annotations

import math

import numpy as np

from .blocks import Block
from .scales import (
    RANGE_HIGH,
    RANGE_LOW,
    allowed_notes,
    hz_to_midi,
    pitch_class_mask,
    snap,
)

try:  # numba turns the per-sample loop into machine code
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def njit(*_a, **_k):
        def wrap(fn):
            return fn
        return wrap


LATENCY_S = 0.0125
FMIN = 60.0          # lowest sung pitch tracked (about B1)
FMAX = 1000.0        # highest (about B5)
CLARITY_MIN = 0.6    # below this the frame counts as unvoiced
LEVEL_MIN = 10 ** (-50 / 20)
VOICING_TAU_S = 0.012


@njit(cache=True, fastmath=False)
def _psola(x, out, n, inb, acc, wsum, mask, st, T0, r, formant, D, mix):
    """Process ``n`` samples. ``st`` = [t, synth mark, analysis mark].

    t     absolute input sample index (the next one to be written)
    s     next synthesis grain centre (output timeline, same index space)
    am    analysis mark nearest to s (input timeline, spaced by T0)
    """
    t = int(st[0])
    s = st[1]
    am = st[2]
    T1 = T0 / r                       # spacing of output grains
    rf = 1.0 if formant else r        # read stride inside a grain
    hw_out = T0 / rf                  # output half-width of a grain
    H = int(hw_out)
    for i in range(n):
        inb[t & mask] = x[i]
        to = t - D
        # Lay down every grain that reaches the sample about to be output.
        while s - hw_out <= to:
            # Analysis mark nearest the synthesis mark, on the T0 grid.
            while am + 0.5 * T0 <= s:
                am += T0
            while am - 0.5 * T0 > s:
                am -= T0
            a = am
            limit = t - T0 - 2.0      # newest centre with a full grain behind it
            if a > limit:
                a -= T0 * math.ceil((a - limit) / T0)
            si = int(math.floor(s + 0.5))
            for k in range(-H, H + 1):
                w = 0.5 + 0.5 * math.cos(math.pi * k / hw_out)
                if w <= 0.0:
                    continue
                p = a + k * rf
                ip = int(math.floor(p))
                fr = p - ip
                v0 = inb[ip & mask]
                v1 = inb[(ip + 1) & mask]
                j = (si + k) & mask
                acc[j] += w * (v0 + (v1 - v0) * fr)
                wsum[j] += w
            s += T1
        j = to & mask
        ws = wsum[j]
        wet = acc[j] / ws if ws > 1.0 else acc[j]
        acc[j] = 0.0
        wsum[j] = 0.0
        dry = inb[j]
        out[i] = dry + (wet - dry) * mix
        t += 1
    st[0] = t
    st[1] = s
    st[2] = am


class AutoTune(Block):
    """Mono block. All parameters are plain attributes, as for every block."""

    name = "autotune"

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.key = 0                 # 0 = C ... 11 = B
        self.scale = "Chromatic"
        self.custom = [True] * 12    # used when scale == "Custom"
        self.low_note = 36           # C2: never snap below
        self.high_note = 84          # C6: never snap above
        self.octave = 0              # -2 .. +2
        self.retune_ms = 20.0        # 0 = hard tune, 400 = natural
        self.humanize = 0.0          # 0 .. 1
        self.formant = True          # keep the vowel colour when shifting
        self.mix = 1.0
        self.a4 = 440.0
        # Piano mode (phase 6): a (12,) bool mask of held pitch classes, or
        # None. Set through engine.post() so it lands between blocks.
        self.external_mask = None

        # Telemetry for the UI's note display (written once per hop).
        self.detected_hz = 0.0
        self.clarity = 0.0
        self.target_note = -1.0      # MIDI, including octave shift; -1 unvoiced
        self.correction_cents = 0.0

        self._cache_key = None
        self._notes = np.zeros(0)
        self.latency = 0
        self.hop = 256

    # -- lifecycle ---------------------------------------------------------

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        sr = self.sr
        self.latency = int(round(LATENCY_S * sr))
        self.hop = _pow2_at_least(sr / 200.0)
        self.win = _pow2_at_least(sr * 0.04)
        size = _pow2_at_least(4 * self.win + 4 * self.latency)
        self._mask = size - 1
        self._inb = np.zeros(size, dtype=np.float64)
        self._acc = np.zeros(size, dtype=np.float64)
        self._wsum = np.zeros(size, dtype=np.float64)
        self._st = np.zeros(3, dtype=np.float64)
        self._out = np.zeros(max(int(max_block), 1), dtype=np.float64)
        self._frame = np.zeros(self.win, dtype=np.float64)
        self._fft_n = 2 * self.win
        # NSDF denominator helper: lags searched.
        self._lag_min = max(2, int(sr / FMAX))
        self._lag_max = min(self.win - 2, int(sr / FMIN))
        self._a_ret = 1.0
        self.reset()
        # Compile now (control thread), so the first audio block doesn't.
        _psola(np.zeros(1), np.zeros(1), 1, self._inb, self._acc, self._wsum,
               self._mask, self._st.copy(), 240.0, 1.0, True, self.latency, 1.0)
        self.reset()

    def reset(self):
        if not hasattr(self, "_inb"):
            return
        self._inb[...] = 0.0
        self._acc[...] = 0.0
        self._wsum[...] = 0.0
        self._st[0] = 0.0
        self._st[1] = float(-self.latency)
        self._st[2] = float(-self.latency)
        self._T0 = self.sr / 200.0
        self._corr = 0.0          # smoothed correction, semitones
        self._oct = 0.0           # smoothed octave shift, semitones
        self._held = 0.0
        self._last_target = -1.0
        self._was_off = False
        self.detected_hz = 0.0
        self.clarity = 0.0
        self.target_note = -1.0
        self.correction_cents = 0.0

    # -- processing --------------------------------------------------------

    def process(self, x):
        if not self.enabled:
            self._was_off = True
            return x
        if self._was_off:
            # Coming back from bypass: start from silence rather than replay
            # whatever was in the rings when it was switched off.
            self.reset()
        return super().process(x)

    def _process(self, x, n):
        out = self._out
        H = self.hop
        mix = float(min(max(self.mix, 0.0), 1.0))
        formant = bool(self.formant)
        i = 0
        while i < n:
            t = int(self._st[0])
            k = min(n - i, H - (t % H))
            ratio = 2.0 ** ((self._corr + self._oct) / 12.0)
            _psola(x[i:i + k], out[i:i + k], k, self._inb, self._acc,
                   self._wsum, self._mask, self._st, self._T0, ratio,
                   formant, self.latency, mix)
            i += k
            if (t + k) % H == 0:
                self._analyse()
        return out[:n]

    # -- analysis (once per hop) ------------------------------------------

    def _notes_for(self) -> np.ndarray:
        ext = self.external_mask
        if ext is not None:
            key = ("ext", tuple(bool(v) for v in ext), self.low_note, self.high_note)
        else:
            key = (self.key, self.scale, tuple(bool(v) for v in self.custom),
                   self.low_note, self.high_note)
        if key != self._cache_key:
            if ext is not None:
                mask = np.asarray(ext, dtype=bool)
                if mask.shape != (12,) or not mask.any():
                    mask = np.ones(12, dtype=bool)
            else:
                mask = pitch_class_mask(int(self.key) % 12, self.scale, self.custom)
            lo = int(min(max(self.low_note, RANGE_LOW), RANGE_HIGH))
            hi = int(min(max(self.high_note, lo), RANGE_HIGH))
            self._notes = allowed_notes(mask, lo, hi)
            self._cache_key = key
        return self._notes

    def _analyse(self) -> None:
        f, clarity, level = self._detect()
        sr = self.sr
        hop_s = self.hop / sr
        voiced = f > 0.0 and clarity >= CLARITY_MIN and level >= LEVEL_MIN
        self.clarity = clarity
        a_v = 1.0 - math.exp(-hop_s / VOICING_TAU_S)
        if voiced:
            self.detected_hz = f
            self._T0 = sr / f
            m = hz_to_midi(f, self.a4)
            target = snap(m, self._notes_for())
            if target == self._last_target:
                self._held += hop_s
            else:
                self._held = 0.0
                self._last_target = target
            tau = max(float(self.retune_ms), 0.0) / 1000.0
            tau += float(min(max(self.humanize, 0.0), 1.0)) * 0.3 * min(1.0, self._held / 0.3)
            a_c = 1.0 if tau <= 0.0 else 1.0 - math.exp(-hop_s / tau)
            self._corr += (target - m - self._corr) * a_c
            octv = 12.0 * int(min(max(self.octave, -2), 2))
            self._oct += (octv - self._oct) * a_v
            self.target_note = target + octv
            self.correction_cents = 100.0 * self._corr
        else:
            self.detected_hz = 0.0
            self._held = 0.0
            self._last_target = -1.0
            self._corr += (0.0 - self._corr) * a_v
            self._oct += (0.0 - self._oct) * a_v
            self.target_note = -1.0
            self.correction_cents = 0.0

    def _detect(self) -> tuple[float, float, float]:
        """McLeod NSDF on the newest ``win`` samples. Returns (hz, clarity, rms)."""
        W = self.win
        t = int(self._st[0])
        mask = self._mask
        start = (t - W) & mask
        end = start + W
        fr = self._frame
        if end <= mask + 1:
            fr[:] = self._inb[start:end]
        else:
            first = mask + 1 - start
            fr[:first] = self._inb[start:]
            fr[first:] = self._inb[:W - first]
        fr -= fr.mean()
        sq = fr * fr
        level = math.sqrt(float(sq.mean()))
        if level < LEVEL_MIN:
            return 0.0, 0.0, level
        spec = np.fft.rfft(fr, self._fft_n)
        acf = np.fft.irfft(spec * np.conj(spec), self._fft_n)[:W]
        # m(tau) = sum over the overlap of x_j^2 + x_{j+tau}^2
        c = np.cumsum(sq)
        total = c[-1]
        lo, hi = self._lag_min, self._lag_max
        lags = np.arange(lo - 1, hi + 1)
        m = (total - c[lags - 1]) + c[W - 1 - lags] + 1e-12
        seg = 2.0 * acf[lo - 1:hi + 1] / m
        mid = seg[1:-1]
        peak = (mid > seg[:-2]) & (mid >= seg[2:]) & (mid > 0.0)
        idx = np.flatnonzero(peak)
        if idx.size == 0:
            return 0.0, 0.0, level
        vals = mid[idx]
        best = float(vals.max())
        if best < 0.35:
            return 0.0, best, level
        chosen = int(idx[np.argmax(vals >= 0.9 * best)]) + 1   # index into seg
        y0, y1, y2 = seg[chosen - 1], seg[chosen], seg[chosen + 1]
        den = y0 - 2.0 * y1 + y2
        shift = 0.0 if abs(den) < 1e-12 else 0.5 * (y0 - y2) / den
        lag = (lo - 1) + chosen + shift
        if lag <= 0:
            return 0.0, 0.0, level
        return float(self.sr / lag), float(min(max(y1, 0.0), 1.0)), level


def _pow2_at_least(x: float) -> int:
    n = 1
    while n < x:
        n <<= 1
    return n
