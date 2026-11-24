"""The mic sidechain: the live mic signal, handed to effects that need it.

A talkbox shapes the guitar with the player's mouth, so the guitar chain has
to see the mic input in the same audio block. The engine copies the mic
channel into :data:`MIC` at the top of every callback, before any instrument
runs; effects that want it read ``MIC.view(n)``.

Real-time rules: the buffer is allocated in ``prepare`` (control thread) and
only copied into on the audio thread. A block the engine didn't fill (no mic,
an offline render, a test) reads as silence.
"""

from __future__ import annotations

import numpy as np


class Sidechain:
    def __init__(self, max_block: int = 4096):
        self.buf = np.zeros(max_block)
        self.n = 0              # samples written for the current block
        # Set by a talkbox while it is shaping sound: how much of the dry voice
        # the Vocals strip should still play (0 = muted). Read by the engine.
        self.voice_gain = 1.0
        self.claimed = False

    def prepare(self, max_block: int) -> None:
        if self.buf.shape[0] < max_block:
            self.buf = np.zeros(max_block)
        self.n = 0

    def write(self, x: np.ndarray, n: int) -> None:
        """Audio thread: copy this block's mic samples."""
        if n > self.buf.shape[0]:
            self.n = 0
            return
        self.buf[:n] = x[:n]
        self.n = n

    def clear(self) -> None:
        self.n = 0

    def view(self, n: int) -> np.ndarray | None:
        """The mic samples for a block of n, or None when there are none."""
        if self.n != n:
            return None
        return self.buf[:n]


MIC = Sidechain()
