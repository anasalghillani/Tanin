"""Mixer: per-instrument strips and the send matrix to three buses.

Every instrument's stereo output goes through a strip (volume, pan, mute,
solo) and then to each bus at its own send level:

    phones  what you hear (the audio interface output)
    record  what the recorder writes
    stream  what Stream Out / Mic Out sends to Discord

So the vocals can go to Discord while the guitar only goes to your ears, or
the other way round. This is the routing matrix from the suite plan; the UI
shows it as a grid of toggles.

Defaults send everything everywhere at unity gain, which keeps today's
behaviour exactly: with one instrument at 0 dB, centre pan, the phones bus is
bit-for-bit the instrument's output.

Gains glide across each block (``SmoothedParam``), so moving a fader or
flipping a send never clicks. Nothing here allocates on the audio thread.
"""

from __future__ import annotations

import numpy as np

from .dsp.core import SmoothedParam, clampf, db2lin

BUSES = ("phones", "record", "stream")


class Strip:
    def __init__(self, key: str):
        self.key = key
        self.volume_db = 0.0
        self.pan = 0.0           # -1 left .. +1 right; balance law, centre = unity
        self.mute = False
        self.solo = False
        self.sends = {b: 1.0 for b in BUSES}
        self.peak = 0.0          # telemetry: last block's output peak
        # One left and one right gain per bus.
        self._g = [[SmoothedParam(1.0), SmoothedParam(1.0)] for _ in BUSES]

    def prepare(self, max_block: int) -> None:
        for pair in self._g:
            for p in pair:
                p.prepare(max_block)

    def targets(self, any_solo: bool):
        """(left, right) gain per bus for this block."""
        on = not self.mute and (self.solo or not any_solo)
        v = db2lin(clampf(self.volume_db, -60.0, 12.0)) if on else 0.0
        if self.volume_db <= -60.0:
            v = 0.0
        pan = clampf(self.pan, -1.0, 1.0)
        gl = v * min(1.0, 1.0 - pan)
        gr = v * min(1.0, 1.0 + pan)
        out = []
        for b in BUSES:
            s = clampf(float(self.sends.get(b, 0.0)), 0.0, 1.0)
            out.append((gl * s, gr * s))
        return out


class Mixer:
    def __init__(self, keys):
        self.strips = {k: Strip(k) for k in keys}
        self.order = list(keys)
        self.bus = np.zeros((len(BUSES), 2, 0), dtype=np.float64)
        self._tmp = np.zeros(0, dtype=np.float64)

    def prepare(self, max_block: int) -> None:
        self.bus = np.zeros((len(BUSES), 2, max_block), dtype=np.float64)
        self._tmp = np.zeros(max_block, dtype=np.float64)
        for s in self.strips.values():
            s.prepare(max_block)

    def __getitem__(self, key: str) -> Strip:
        return self.strips[key]

    def bus_view(self, name: str, n: int) -> np.ndarray:
        return self.bus[BUSES.index(name), :, :n]

    def process(self, outputs, n: int) -> None:
        """``outputs[i]`` is the (2, n) block of ``order[i]``, or None if off.

        Fills ``self.bus[:, :, :n]``. Audio thread.
        """
        bus = self.bus
        bus[:, :, :n] = 0.0
        any_solo = False
        for s in self.strips.values():
            if s.solo:
                any_solo = True
                break
        tmp = self._tmp[:n]
        for i, key in enumerate(self.order):
            st = self.strips[key]
            src = outputs[i]
            tg = st.targets(any_solo)
            if src is None:
                # Keep the smoothers moving so a later un-mute starts right.
                for b in range(len(BUSES)):
                    st._g[b][0].snap(tg[b][0])
                    st._g[b][1].snap(tg[b][1])
                st.peak = 0.0
                continue
            st.peak = float(np.max(np.abs(src[:, :n]))) if n else 0.0
            for b in range(len(BUSES)):
                for c in (0, 1):
                    p = st._g[b][c]
                    p.set(tg[b][c])
                    g = p.ramp(n)
                    if isinstance(g, float) and g == 0.0:
                        continue
                    if isinstance(g, float) and g == 1.0:
                        bus[b, c, :n] += src[c, :n]
                    else:
                        np.multiply(src[c, :n], g, out=tmp)
                        bus[b, c, :n] += tmp
