"""The signal chain: what feeds what, and where mono becomes stereo."""

from __future__ import annotations

import numpy as np

from .blocks import (
    Amp,
    Cabinet,
    Compressor,
    Delay,
    Drive,
    IRLoader,
    Modulation,
    NoiseGate,
    Output,
    Reverb,
)
from .core import SmoothedParam, clampf, db2lin, sanitize

# Order matters and matches how these are wired on a real board: dynamics and
# dirt in front of the amp, time-based effects after the cab where they belong.
MONO_ORDER = ("gate", "comp", "drive", "amp", "cab", "ir")
STEREO_ORDER = ("mod", "delay", "reverb")


class Chain:
    def __init__(self):
        self.blocks = {
            "gate": NoiseGate(),
            "comp": Compressor(),
            "drive": Drive(),
            "amp": Amp(),
            "cab": Cabinet(),
            "ir": IRLoader(),
            "mod": Modulation(),
            "delay": Delay(),
            "reverb": Reverb(),
            "output": Output(),
        }
        self.input_gain_db = 0.0
        self.global_bypass = False
        self.input_peak = 0.0

        self._in_gain = SmoothedParam(1.0)
        self.sr = 48000.0
        self.max_block = 512

    # -- lifecycle ---------------------------------------------------------

    def prepare(self, sr: float, max_block: int) -> None:
        self.sr = float(sr)
        self.max_block = int(max_block)
        for b in self.blocks.values():
            b.prepare(sr, max_block)
        self._in_gain.prepare(max_block)
        self._in_gain.snap(db2lin(self.input_gain_db))

    def reset(self) -> None:
        for b in self.blocks.values():
            b.reset()

    def __getitem__(self, key: str):
        return self.blocks[key]

    # -- processing --------------------------------------------------------

    def process(self, mono_in: np.ndarray) -> np.ndarray:
        """mono_in is (n,) float64. Returns (2, n) float64."""
        n = mono_in.shape[-1]
        self.input_peak = float(np.max(np.abs(mono_in))) if n else 0.0

        self._in_gain.set(db2lin(clampf(self.input_gain_db, -24.0, 36.0)))
        sig = mono_in * self._in_gain.ramp(n)

        if self.global_bypass:
            out = np.stack((sig, sig))
            return sanitize(out)

        for key in MONO_ORDER:
            sig = self.blocks[key].process(sig)

        stereo = np.stack((sig, sig))

        for key in STEREO_ORDER:
            stereo = self.blocks[key].process(stereo)

        stereo = self.blocks["output"].process(stereo)
        return sanitize(stereo)

    # -- state -------------------------------------------------------------

    PARAM_SPEC = {
        "gate": ("enabled", "threshold_db", "attack_ms", "release_ms"),
        "comp": ("enabled", "threshold_db", "ratio", "attack_ms", "release_ms",
                 "makeup_db", "mix"),
        "drive": ("enabled", "model", "drive", "tone", "level", "mix"),
        "amp": ("enabled", "model", "gain", "bass", "mid", "treble", "presence",
                "master", "sag"),
        "cab": ("enabled", "model", "mic", "position", "distance"),
        "ir": ("enabled", "mix", "gain_db", "ir_path"),
        "mod": ("enabled", "mode", "rate", "depth", "feedback", "mix",
                "stereo_spread"),
        "delay": ("enabled", "mode", "time_ms", "feedback", "mix", "tone",
                  "modulation"),
        "reverb": ("enabled", "mode", "size", "damping", "mix", "width",
                   "predelay_ms"),
        "output": ("enabled", "volume_db", "width"),
    }

    def snapshot(self) -> dict:
        state = {"input_gain_db": self.input_gain_db}
        for key, names in self.PARAM_SPEC.items():
            b = self.blocks[key]
            state[key] = {nm: getattr(b, nm) for nm in names}
        return state

    def apply(self, state: dict) -> None:
        """Load a preset. Safe from any thread for scalars; the engine routes
        this through its command queue so it lands between blocks.

        A preset is a *complete* sound: any block or parameter it leaves out
        goes back to its factory default rather than keeping whatever the last
        preset set. Otherwise the gate from a metal preset stays on in the
        clean preset loaded after it, and a preset sounds different depending
        on what was loaded before it.

        Two things are deliberately not reset when absent: the input gain,
        which belongs to the guitar and interface rather than to the sound,
        and the IR file, which the engine loads (file I/O cannot happen here,
        see ``Engine.apply_preset``).

        Values are coerced to the default's type and anything unusable falls
        back to the default, because this runs on the audio thread and a
        string where a float belongs would raise on every block.
        """
        if not isinstance(state, dict):
            return
        if "input_gain_db" in state:
            self.input_gain_db = _coerce(state["input_gain_db"], self.input_gain_db)
        for key, names in self.PARAM_SPEC.items():
            src = state.get(key)
            if not isinstance(src, dict):
                src = {}
            b = self.blocks[key]
            defaults = _DEFAULTS[key]
            for nm in names:
                if nm == "ir_path":
                    continue
                setattr(b, nm, _coerce(src.get(nm, defaults[nm]), defaults[nm]))


def _coerce(value, default):
    """``value`` as the same type as ``default``, or ``default`` if it can't be."""
    if isinstance(default, bool):
        return value if isinstance(value, bool) else default
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            return default
        v = float(value)
        return v if np.isfinite(v) else default
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    return default


# What a freshly built chain looks like: the reference "omitted means this".
# Built once at import from the blocks' own constructors, so a new parameter
# picks up its default here without a second list to keep in sync.
_DEFAULTS = Chain().snapshot()
