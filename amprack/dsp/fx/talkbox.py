"""Talkbox: your mouth, picked up by the mic, shapes the guitar.

A real talkbox pipes the amp into your mouth through a tube; your mouth
filters it and a vocal mic picks the result up. Here the mic is analysed
instead and its mouth shape is put on the guitar, so there is no tube, no
driver and nothing extra to plug in. The mic arrives through the engine's
mic sidechain (:mod:`.sidechain`), from the Vocals input.

Two models (picked by ear in the listen test, 2026-11-23):

*   **Vowel** -- the mic's formants (the three resonances that make a vowel:
    ah, ee, oo) are tracked by linear prediction (LPC) every 5 ms and three
    resonant filters on the guitar follow them. A talking wah.
*   **Vocoder** -- a 16-band channel vocoder: the mic's level in each band
    sets the guitar's level in that band. More robotic, more consonant.

Speaker safety (Anas plays through speakers, so the mic hears the guitar):

*   Only the *shape* of the mic is used, never its level: the vowel tracker is
    level independent and the vocoder bands are normalised by the mic's own
    broadband level. The output level follows the guitar, with at most +18 dB
    of make-up, so a speaker-to-mic loop cannot build up volume.
*   A gate with an automatic noise floor closes the mouth when you are not
    talking. Closed sounds like a talkbox with the mouth shut (muffled), or
    can be set to mute or to pass the guitar dry.
*   While a talkbox is shaping sound the Vocals strip's dry voice is faded to
    the Voice knob (0 by default), so the room hears the guitar talking, not
    you twice.

Everything runs per sample on absolute sample positions (analysis every
``HOP`` samples of the stream, not of the buffer), so the sound is identical
at every buffer size.
"""

from __future__ import annotations

import math

import numpy as np

from ..core import clampf
from .base import LEVEL, MIX, Effect, P, PC, PL
from . import sidechain

try:
    from numba import njit
except Exception:  # pragma: no cover - only without numba (slow, offline use)
    def njit(*_a, **_k):
        def wrap(fn):
            return fn
        return wrap


HOP = 240            # analysis every 5 ms at 48 kHz (scaled with the rate)
DEC = 4              # vowel analysis runs at sr / 4 (12 kHz at 48 kHz)
WIN = 256            # analysis window at the decimated rate (~21 ms)
ORDER = 14           # LPC order at 12 kHz: room for 3-4 formants
NF = 128             # envelope points searched for formant peaks
N_BANDS = 16         # vocoder bands
F_LO, F_HI = 120.0, 7000.0

# st[] slots (scalar state shared by both models)
S_PRE, S_HOPE, S_ENVIN, S_ENVWET, S_OPEN, S_FLOOR, S_TARGET, S_MICENV = range(8)
N_ST = 8


@njit(cache=True)
def _levinson(r, a, tmp, p):
    """Autocorrelation r[0..p] -> predictor a[0..p] (a[0] = 1), in place."""
    for i in range(p + 1):
        a[i] = 0.0
    a[0] = 1.0
    e = r[0]
    for i in range(1, p + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / e
        if k > 0.995:
            k = 0.995
        elif k < -0.995:
            k = -0.995
        for j in range(i + 1):
            tmp[j] = a[j]
        for j in range(1, i):
            a[j] = tmp[j] + k * tmp[i - j]
        a[i] = k
        e *= 1.0 - k * k
        if e <= 1e-12:
            break


@njit(cache=True)
def _biquad(x, c, z):
    """One DF2T biquad sample. c = (b0, b1, b2, a1, a2), z = 2 states."""
    y = c[0] * x + z[0]
    z[0] = c[1] * x - c[3] * y + z[1]
    z[1] = c[2] * x - c[4] * y
    return y


@njit(cache=True)
def _analyse(dring, dpos, win, lagw, ctab, stab, r, a, tmp, H, fhz, ftar):
    """LPC formant pick on the last WIN decimated mic samples -> ftar (3)."""
    p = ORDER
    for lag in range(p + 1):
        s = 0.0
        for i in range(lag, WIN):
            x0 = dring[(dpos + i) % WIN] * win[i]
            x1 = dring[(dpos + i - lag) % WIN] * win[i - lag]
            s += x0 * x1
        r[lag] = s * lagw[lag]
    if r[0] < 1e-12:
        return
    r[0] *= 1.001                     # white-noise floor: finite peaks
    _levinson(r, a, tmp, p)
    for f in range(NF):
        re = 0.0
        im = 0.0
        for j in range(p + 1):
            re += a[j] * ctab[f, j]
            im -= a[j] * stab[f, j]
        H[f] = 1.0 / (re * re + im * im + 1e-12)
    found = 0
    new0 = ftar[0]
    new1 = ftar[1]
    new2 = ftar[2]
    for f in range(1, NF - 1):
        if H[f] > H[f - 1] and H[f] >= H[f + 1]:
            if found == 0:
                new0 = fhz[f]
            elif found == 1:
                new1 = fhz[f]
            elif found == 2:
                new2 = fhz[f]
            found += 1
            if found == 3:
                break
    if found >= 2:
        ftar[0] = 0.5 * ftar[0] + 0.5 * new0
        ftar[1] = 0.5 * ftar[1] + 0.5 * new1
        if found >= 3:
            ftar[2] = 0.5 * ftar[2] + 0.5 * new2


@njit(cache=True)
def _kernel(x, mic, n, t0, mode, sr, hop,
            st, lpc_c, lpz, dring, dposa, win, lagw, ctab, stab, r, a, tmp, H, fhz,
            ftar, fcur, ic1, ic2, closed_z,
            vc, vm, vmz, vcz, em, ec,
            gate_db, shift, q, glide_c, closed_mode, mix, level, out):
    C = x.shape[0]
    # follower coefficients
    a_in = math.exp(-1.0 / (0.005 * sr))
    r_in = math.exp(-1.0 / (0.120 * sr))
    a_open = math.exp(-1.0 / (0.010 * sr))
    r_open = math.exp(-1.0 / (0.080 * sr))
    a_em = math.exp(-1.0 / (0.002 * sr))
    r_em = glide_c
    a_ec = math.exp(-1.0 / (0.005 * sr))
    r_ec = math.exp(-1.0 / (0.060 * sr))
    a_mb = math.exp(-1.0 / (0.005 * sr))
    r_mb = math.exp(-1.0 / (0.300 * sr))
    lp_c = 1.0 - math.exp(-2.0 * math.pi * 700.0 / sr)
    kq = 1.0 / q
    fg0, fg1, fg2 = 1.0, 0.8, 0.5
    nb = vc.shape[0]
    for i in range(n):
        t = t0 + i
        m = mic[i]
        # ---- gate: hop energy, auto noise floor --------------------------
        st[S_HOPE] += m * m
        if t % hop == 0:
            e = st[S_HOPE] / hop
            st[S_HOPE] = 0.0
            db = 10.0 * math.log10(e + 1e-14)
            fl = st[S_FLOOR]
            if db < fl:
                fl += (db - fl) * 0.5
            else:
                fl += 0.01          # creeps up ~2 dB/s: follows room noise, not words
            if fl < -100.0:
                fl = -100.0
            elif fl > -25.0:
                fl = -25.0
            st[S_FLOOR] = fl
            thr = gate_db if gate_db > fl + 8.0 else fl + 8.0
            st[S_TARGET] = 1.0 if db > thr else 0.0
            if mode == 0 and st[S_TARGET] > 0.5:
                _analyse(dring, dposa[0], win, lagw, ctab, stab, r, a, tmp, H, fhz, ftar)
        # ---- mouth open / closed -----------------------------------------
        tgt = st[S_TARGET]
        o = st[S_OPEN]
        c = a_open if tgt > o else r_open
        o = c * o + (1.0 - c) * tgt
        st[S_OPEN] = o
        # ---- guitar level (what the output follows) ----------------------
        gin = 0.0
        for ch in range(C):
            gin += abs(x[ch, i])
        gin /= C
        ei = st[S_ENVIN]
        c = a_in if gin > ei else r_in
        ei = c * ei + (1.0 - c) * gin
        st[S_ENVIN] = ei
        wet_abs = 0.0
        if mode == 0:
            # ---- Vowel: decimated LPC input, three gliding resonators ----
            mp = m - 0.95 * st[S_PRE]
            st[S_PRE] = m
            v = _biquad(mp, lpc_c[0], lpz[0])
            v = _biquad(v, lpc_c[1], lpz[1])
            if t % DEC == 0:
                dring[dposa[0]] = v
                dposa[0] = (dposa[0] + 1) % WIN
            for j in range(3):
                fcur[j] += (ftar[j] * shift - fcur[j]) * (1.0 - glide_c)
            g0 = math.tan(math.pi * min(fcur[0], 0.45 * sr) / sr)
            g1 = math.tan(math.pi * min(fcur[1], 0.45 * sr) / sr)
            g2 = math.tan(math.pi * min(fcur[2], 0.45 * sr) / sr)
            for ch in range(C):
                xi = x[ch, i]
                acc = 0.0
                for j in range(3):
                    g = g0 if j == 0 else (g1 if j == 1 else g2)
                    a1 = 1.0 / (1.0 + g * (g + kq))
                    a2 = g * a1
                    a3 = g * a2
                    v3 = xi - ic2[ch, j]
                    v1 = a1 * ic1[ch, j] + a2 * v3
                    v2 = ic2[ch, j] + a2 * ic1[ch, j] + a3 * v3
                    ic1[ch, j] = 2.0 * v1 - ic1[ch, j]
                    ic2[ch, j] = 2.0 * v2 - ic2[ch, j]
                    w = fg0 if j == 0 else (fg1 if j == 1 else fg2)
                    acc += w * v1 * kq
                out[ch, i] = acc
                wet_abs += abs(acc)
        else:
            # ---- Vocoder: mic band shape (level-free) on guitar bands ----
            mp = m - 0.9 * st[S_PRE]
            st[S_PRE] = m
            mb = st[S_MICENV]
            am = abs(mp)
            c = a_mb if am > mb else r_mb
            mb = c * mb + (1.0 - c) * am
            st[S_MICENV] = mb
            inv_mb = 1.0 / (mb + 1e-6)
            for b in range(nb):
                y = _biquad(mp, vm[b], vmz[b, 0])
                y = _biquad(y, vm[b], vmz[b, 1])
                ay = abs(y)
                e = em[b]
                c = a_em if ay > e else r_em
                em[b] = c * e + (1.0 - c) * ay
            for ch in range(C):
                xi = x[ch, i]
                acc = 0.0
                for b in range(nb):
                    y = _biquad(xi, vc[b], vcz[ch, b, 0])
                    y = _biquad(y, vc[b], vcz[ch, b, 1])
                    ay = abs(y)
                    e = ec[ch, b]
                    c = a_ec if ay > e else r_ec
                    e = c * e + (1.0 - c) * ay
                    ec[ch, b] = e
                    shape = em[b] * inv_mb
                    if shape > 4.0:
                        shape = 4.0
                    acc += y / (e + 1e-5) * shape
                out[ch, i] = acc
                wet_abs += abs(acc)
        # ---- level follows the guitar (speaker safe), closed mouth -------
        wet_abs /= C
        ew = st[S_ENVWET]
        c = a_in if wet_abs > ew else r_in
        ew = c * ew + (1.0 - c) * wet_abs
        st[S_ENVWET] = ew
        mk = ei / (ew + 1e-9)
        if mk > 8.0:
            mk = 8.0
        for ch in range(C):
            xi = x[ch, i]
            closed_z[ch] += (xi - closed_z[ch]) * lp_c
            if closed_mode == 0:
                cl = closed_z[ch] * 0.3
            elif closed_mode == 1:
                cl = 0.0
            else:
                cl = xi
            y = o * out[ch, i] * mk + (1.0 - o) * cl
            out[ch, i] = (mix * y + (1.0 - mix) * xi) * level


def _bp(fc, q, sr):
    """RBJ constant-0 dB-peak band-pass as (b0, b1, b2, a1, a2)."""
    fc = min(max(fc, 20.0), 0.45 * sr)
    w0 = 2 * math.pi * fc / sr
    al = math.sin(w0) / (2 * q)
    a0 = 1 + al
    return np.array([al / a0, 0.0, -al / a0, -2 * math.cos(w0) / a0, (1 - al) / a0])


def _butter_lp4(fc, sr):
    """Two biquads of a 4th-order Butterworth low-pass (anti-alias for /DEC)."""
    secs = []
    w0 = 2 * math.pi * fc / sr
    for qq in (0.5412, 1.3066):
        al = math.sin(w0) / (2 * qq)
        cw = math.cos(w0)
        a0 = 1 + al
        secs.append([(1 - cw) / 2 / a0, (1 - cw) / a0, (1 - cw) / 2 / a0,
                     -2 * cw / a0, (1 - al) / a0])
    return np.array(secs)


class Talkbox(Effect):
    TYPE = "talkbox"
    LABEL = "Talkbox"
    CATEGORY = "Filter"
    MODELS = ("Vowel", "Vocoder")
    PARAMS = (
        P("gate_db", "Gate", -75.0, -20.0, -50.0, "dB", aliases=("threshold", "sens",
                                                                  "sensitivity")),
        P("mouth", "Mouth", -12.0, 12.0, 0.0, "st", aliases=("formant", "shift", "size")),
        P("resonance", "Resonance", 0.0, 1.0, 0.5, aliases=("q", "res", "peak")),
        PL("glide_ms", "Glide", 2.0, 80.0, 15.0, "ms", aliases=("speed", "release",
                                                                "smooth")),
        PC("closed", "Closed", ("Muffled", "Mute", "Dry"), aliases=("mouthclosed",)),
        P("voice", "Voice", 0.0, 1.0, 0.0, aliases=("vocal", "dryvoice")),
        MIX,
        LEVEL,
    )

    def __init__(self):
        super().__init__()
        self.sidechain = sidechain.MIC

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        sr = float(sr)
        C = self.channels
        self.hop = max(16, int(round(HOP * sr / 48000.0)))
        dsr = sr / DEC
        self.st = np.zeros(N_ST)
        self.lpc_c = _butter_lp4(min(4500.0, 0.4 * dsr), sr)
        self.lpz = np.zeros((2, 2))
        self.dring = np.zeros(WIN)
        self.dpos = np.zeros(1, dtype=np.int64)
        self.win = np.hanning(WIN)
        i = np.arange(ORDER + 1)
        self.lagw = np.exp(-0.5 * (2 * math.pi * 50.0 * i / dsr) ** 2)
        self.fhz = np.geomspace(200.0, min(4000.0, 0.45 * dsr), NF)
        w = 2 * math.pi * self.fhz[:, None] / dsr * i[None, :]
        self.ctab = np.cos(w)
        self.stab = np.sin(w)
        self.r = np.zeros(ORDER + 1)
        self.a = np.zeros(ORDER + 1)
        self.tmp = np.zeros(ORDER + 1)
        self.H = np.zeros(NF)
        self.ftar = np.array([700.0, 1200.0, 2600.0])     # neutral "ah"
        self.fcur = self.ftar.copy()
        self.ic1 = np.zeros((C, 3))
        self.ic2 = np.zeros((C, 3))
        self.closed_z = np.zeros(C)
        self.vc = np.zeros((N_BANDS, 5))
        self.vm = np.zeros((N_BANDS, 5))
        self.vmz = np.zeros((N_BANDS, 2, 2))
        self.vcz = np.zeros((C, N_BANDS, 2, 2))
        self.em = np.zeros(N_BANDS)
        self.ec = np.zeros((C, N_BANDS))
        self._vkey = None
        self._silence = np.zeros(max_block)
        self.reset()

    def reset(self):
        if not hasattr(self, "st"):
            return
        for arr in (self.st, self.lpz, self.dring, self.ic1, self.ic2, self.closed_z,
                    self.vmz, self.vcz, self.em, self.ec):
            arr[...] = 0.0
        self.st[S_FLOOR] = -90.0
        self.dpos[0] = 0
        self.ftar[:] = (700.0, 1200.0, 2600.0)
        self.fcur[:] = self.ftar

    def _vocoder_coeffs(self, q, shift):
        """Band filters; recomputed only when Resonance or Mouth move."""
        key = (round(q, 4), round(shift, 4))
        if key == self._vkey:
            return
        self._vkey = key
        edges = np.geomspace(F_LO, F_HI, N_BANDS)
        bq = 2.0 + 6.0 * q
        for b, fc in enumerate(edges):
            self.vc[b] = _bp(fc, bq, self.sr)
            self.vm[b] = _bp(fc / shift, bq, self.sr)    # mouth size: listen higher/lower

    def _process(self, x, n):
        mic = self.sidechain.view(n) if self.sidechain is not None else None
        if mic is None:
            mic = self._silence[:n]
        res = clampf(self.resonance, 0.0, 1.0)
        shift = 2.0 ** (clampf(self.mouth, -12.0, 12.0) / 12.0)
        glide = clampf(self.glide_ms, 1.0, 200.0)
        glide_c = math.exp(-1.0 / (glide * 1e-3 * self.sr))
        mode = 0 if self.model == "Vowel" else 1
        if mode == 1:
            self._vocoder_coeffs(res, shift)
        closed = {"Muffled": 0, "Mute": 1, "Dry": 2}.get(self.closed, 0)
        y = np.empty_like(x)
        _kernel(x, mic, n, self._t, mode, float(self.sr), self.hop,
                self.st, self.lpc_c, self.lpz, self.dring, self.dpos, self.win, self.lagw,
                self.ctab, self.stab, self.r, self.a, self.tmp, self.H, self.fhz,
                self.ftar, self.fcur, self.ic1, self.ic2, self.closed_z,
                self.vc, self.vm, self.vmz, self.vcz, self.em, self.ec,
                float(clampf(self.gate_db, -100.0, 0.0)), shift, 3.0 + 13.0 * res, glide_c,
                closed, clampf(self.mix, 0.0, 1.0), 2.0 * clampf(self.level, 0.0, 1.0), y)
        # Tell the engine how much dry voice to keep while the mouth is open.
        sc = self.sidechain
        if sc is not None and sc.n == n:
            sc.claimed = True
            open_amt = float(self.st[S_OPEN]) * clampf(self.mix, 0.0, 1.0)
            sc.voice_gain = min(sc.voice_gain,
                                1.0 - open_amt * (1.0 - clampf(self.voice, 0.0, 1.0)))
        return y

    @property
    def mouth_open(self) -> float:
        """0..1, for the UI."""
        return float(self.st[S_OPEN]) if hasattr(self, "st") else 0.0
