"""The Instrument interface: one sound source with its own processing chain.

The engine used to own exactly one guitar chain. It now owns a list of
instruments (guitar, vocals, and later piano), runs each enabled one in the
same audio callback, and hands their outputs to the mixer.

Contract, same real-time rules as every DSP block:

*   ``prepare(sr, max_block)`` allocates; ``process()`` never does beyond the
    block, never takes a lock, never touches a file.
*   ``process(mono, n)`` takes the instrument's input channel as (n,) float64
    (or ``None`` for an instrument with no audio input, such as a MIDI synth)
    and returns a (2, n) float64 stereo block.
*   ``enabled = False`` means the engine skips it entirely: an instrument that
    is off costs no CPU.
*   ``snapshot()`` / ``apply()`` round-trip the instrument's sound for presets.
"""

from __future__ import annotations

import numpy as np


class Instrument:
    key = "instrument"          # stable id used in settings and the mixer
    label = "Instrument"        # what the UI shows
    accent = "#ffffff"
    uses_audio_input = True     # False for MIDI-only sources

    def __init__(self, input_channel: int | None = 0):
        self.enabled = True
        # Which hardware input feeds it. Each instrument picks its own, so an
        # interface with two jacks can run guitar on In 1 and a mic on In 2.
        self.input_channel = input_channel
        self.sr = 48000.0
        self.max_block = 512
        self.input_peak = 0.0
        self.in_clips = 0

    # -- lifecycle ---------------------------------------------------------

    def prepare(self, sr: float, max_block: int) -> None:
        self.sr = float(sr)
        self.max_block = int(max_block)

    def reset(self) -> None:
        """Kill anything ringing (delay tails, reverb). Audio thread only."""

    # -- audio thread ------------------------------------------------------

    def process(self, mono: np.ndarray | None, n: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # -- presets -----------------------------------------------------------

    def snapshot(self) -> dict:
        return {}

    def apply(self, state: dict) -> None:
        pass


def coerce(value, default):
    """``value`` as the same type as ``default``, or ``default`` if it can't be.

    Presets are user files and get applied on the audio thread, where a string
    in place of a float would raise on every block, so anything unusable falls
    back to the default instead.
    """
    if isinstance(default, bool):
        return value if isinstance(value, bool) else default
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            return default
        v = float(value)
        if not np.isfinite(v):
            return default
        return int(v) if isinstance(default, int) else v
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    if isinstance(default, list):
        if isinstance(value, (list, tuple)) and len(value) == len(default):
            return [coerce(v, d) for v, d in zip(value, default)]
        return list(default)
    return default
