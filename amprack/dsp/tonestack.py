"""Passive amp tone stacks, modelled from their circuits.

The classic Fender/Marshall/Vox "TMB" (treble-middle-bass) stack is a
passive RC network; its knobs interact (turning the bass up moves the mid
scoop, the treble knob changes how much the bass does), which shelving EQ
cannot imitate. Following Yeh and Smith ("Discretization of the '59 Fender
Bassman Tone Stack", DAFx 2006) its exact third-order transfer function is
written in terms of the component values and the three pot positions, then
turned into a digital filter with the bilinear transform. Different amps
are the same network with different parts (``STACKS``), which is most of
why a Marshall and a Fender with the knobs at noon sound so different.

tools/fx_tests.py checks the formula against a nodal-analysis solve of the
circuit at a spread of frequencies and knob settings.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import bilinear, lfilter

from .core import clampf

# name: R1 treble pot, R2 bass pot, R3 mid pot, R4 slope resistor, C1..C3
STACKS = {
    "tweed": dict(R1=250e3, R2=1e6, R3=25e3, R4=56e3, C1=250e-12, C2=20e-9, C3=20e-9),
    "blackface": dict(R1=250e3, R2=250e3, R3=10e3, R4=100e3, C1=250e-12, C2=100e-9,
                      C3=47e-9),
    "marshall": dict(R1=220e3, R2=1e6, R3=22e3, R4=33e3, C1=470e-12, C2=22e-9,
                     C3=22e-9),
    "vox": dict(R1=1e6, R2=1e6, R3=10e3, R4=100e3, C1=50e-12, C2=22e-9, C3=22e-9),
    "mesa": dict(R1=250e3, R2=250e3, R3=10e3, R4=100e3, C1=750e-12, C2=20e-9, C3=20e-9),
    "orange": dict(R1=250e3, R2=1e6, R3=25e3, R4=47e3, C1=470e-12, C2=22e-9, C3=47e-9),
    "hiwatt": dict(R1=100e3, R2=100e3, R3=10e3, R4=47e3, C1=1e-9, C2=47e-9, C3=22e-9),
    "modern": dict(R1=250e3, R2=1e6, R3=25e3, R4=47e3, C1=500e-12, C2=22e-9, C3=22e-9),
}


def tmb_analog(t, m, l, R1, R2, R3, R4, C1, C2, C3):
    """Analog TMB transfer function. Returns (b, a), highest power first,
    for scipy. t, m, l are treble, middle, bass pot positions in 0..1."""
    # Derived by nodal analysis of the classic TMB network (in -> C1 -> treble
    # pot top; slope resistor R4 and C2/C3 meet at node X; bass = l*R2,
    # mid = m*R3 to ground). Normalised so a0 = 1, b0 = 0.
    # Real pots never reach 0 ohm; a tiny floor keeps the order at 3.
    R2l, R3m = R2 * max(l, 1e-4), R3 * max(m, 1e-4)
    b1 = (t * C1 * R1 + C1 * R2l + C1 * R3m + C2 * R2l + C2 * R3m + C3 * R3m)
    b2 = (C1 * C2 * (R1 * R2l + R1 * R3m + R1 * R4 * t + R2l * R4 + R3m * R4)
          + C1 * C3 * (R1 * R3m + R1 * R4 * t + R2l * R3m + R2l * R4 + R3m * R4)
          + C2 * C3 * R2l * R3m)
    b3 = C1 * C2 * C3 * R2l * (R1 * R3m + R1 * R4 * t + R3m * R4)
    a1 = (C1 * R1 + C1 * R2l + C1 * R3m + C2 * R2l + C2 * R3m + C2 * R4
          + C3 * R3m + C3 * R4)
    a2 = (C1 * C2 * (R1 * R2l + R1 * R3m + R1 * R4 + R2l * R4 + R3m * R4)
          + C1 * C3 * (R1 * R3m + R1 * R4 + R2l * R3m + R2l * R4 + R3m * R4)
          + C2 * C3 * (R2l * R3m + R2l * R4))
    a3 = C1 * C2 * C3 * R2l * (R1 * R3m + R1 * R4 + R3m * R4)
    return np.array([b3, b2, b1, 0.0]), np.array([a3, a2, a1, 1.0])


def pot_log(x: float) -> float:
    """Audio-taper pot: most of the resistance change is at the top end."""
    x = clampf(x, 0.0, 1.0)
    return (10.0 ** (2.0 * x) - 1.0) / 99.0


class ToneStack:
    """Stateful digital TMB stack for one channel.

    Coefficients are recomputed only when a knob or the stack type moves.
    The output is normalised so that knobs at noon give the same level in
    the 300 Hz - 3 kHz band for every stack type: the circuits themselves
    lose 10-20 dB and differ by several dB, which would otherwise turn
    picking an amp into a volume change.
    """

    def __init__(self):
        self.sr = 48000.0
        self.kind = ""
        self.key = None
        self.b = np.array([1.0, 0.0, 0.0, 0.0])
        self.a = np.array([1.0, 0.0, 0.0, 0.0])
        self.zi = np.zeros(3)
        self.norm = {}

    def prepare(self, sr: float):
        self.sr = float(sr)
        self.key = None
        self.norm = {}
        self.zi[...] = 0.0

    def reset(self):
        self.zi[...] = 0.0

    def _norm(self, kind):
        g = self.norm.get(kind)
        if g is None:
            bs, as_ = tmb_analog(0.5, 0.5, pot_log(0.5), **STACKS[kind])
            w = 2 * math.pi * np.geomspace(300.0, 3000.0, 16)
            h = np.polyval(bs, 1j * w) / np.polyval(as_, 1j * w)
            g = 1.0 / float(np.sqrt(np.mean(np.abs(h) ** 2)))
            self.norm[kind] = g
        return g

    def set(self, kind: str, treble: float, mid: float, bass: float):
        key = (kind, round(treble, 4), round(mid, 4), round(bass, 4))
        if key == self.key:
            return
        comp = STACKS[kind]
        # Treble and mid pots are linear in most amps, bass is audio taper.
        bs, as_ = tmb_analog(clampf(treble, 0.0, 1.0), clampf(mid, 0.0, 1.0),
                             pot_log(bass), **comp)
        bz, az = bilinear(bs, as_, fs=self.sr)
        g = self._norm(kind)
        self.b = bz / az[0] * g
        self.a = az / az[0]
        self.key = key

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y
