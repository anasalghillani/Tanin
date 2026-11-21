"""Guitar: today's amp and pedalboard chain behind the Instrument interface."""

from __future__ import annotations

import numpy as np

from ..dsp.chain import Chain
from ..instrument import Instrument


class GuitarInstrument(Instrument):
    key = "guitar"
    label = "Guitar"
    accent = "#ff8a3d"

    def __init__(self, input_channel: int = 0):
        super().__init__(input_channel)
        self.chain = Chain()

    def prepare(self, sr: float, max_block: int) -> None:
        super().prepare(sr, max_block)
        self.chain.prepare(sr, max_block)

    def reset(self) -> None:
        self.chain.reset()

    def process(self, mono: np.ndarray | None, n: int) -> np.ndarray:
        out = self.chain.process(mono)
        self.input_peak = self.chain.input_peak
        return out

    def snapshot(self) -> dict:
        return self.chain.snapshot()

    def apply(self, state: dict) -> None:
        self.chain.apply(state)
