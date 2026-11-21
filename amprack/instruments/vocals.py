"""Vocals: the mic chain, with auto-tune in the middle of it.

Gate -> De-esser -> Auto-Tune -> EQ -> Comp -> Saturation   (mono)
     -> Doubler -> Delay -> Reverb -> Output                 (stereo)

The de-esser sits before auto-tune so a harsh "s" cannot confuse the pitch
detector; EQ and compression come after it so they shape the corrected voice.
Delay, Reverb and the safety-limited Output are the same blocks the guitar
uses.
"""

from __future__ import annotations

import numpy as np

from ..dsp.autotune import AutoTune
from ..dsp.blocks import Compressor, Delay, NoiseGate, Output, Reverb
from ..dsp.core import SmoothedParam, clampf, db2lin, sanitize
from ..dsp.vocal_blocks import DeEsser, Doubler, Saturation, VocalEQ
from ..instrument import Instrument, coerce

MONO_ORDER = ("gate", "deesser", "autotune", "eq", "comp", "saturation")
STEREO_ORDER = ("doubler", "delay", "reverb")


class VocalChain:
    PARAM_SPEC = {
        "gate": ("enabled", "threshold_db", "attack_ms", "release_ms"),
        "deesser": ("enabled", "freq", "threshold_db", "range_db"),
        "autotune": ("enabled", "key", "scale", "custom", "low_note", "high_note",
                     "octave", "retune_ms", "humanize", "formant", "mix"),
        "eq": ("enabled", "low_cut_hz", "presence_db", "air_db"),
        "comp": ("enabled", "threshold_db", "ratio", "attack_ms", "release_ms",
                 "makeup_db", "mix"),
        "saturation": ("enabled", "drive_db", "mix"),
        "doubler": ("enabled", "mix", "width", "detune"),
        "delay": ("enabled", "mode", "time_ms", "feedback", "mix", "tone",
                  "modulation"),
        "reverb": ("enabled", "mode", "size", "damping", "mix", "width",
                   "predelay_ms"),
        "output": ("enabled", "volume_db", "width"),
    }

    def __init__(self):
        self.blocks = {
            "gate": NoiseGate(),
            "deesser": DeEsser(),
            "autotune": AutoTune(),
            "eq": VocalEQ(),
            "comp": Compressor(),
            "saturation": Saturation(),
            "doubler": Doubler(),
            "delay": Delay(),
            "reverb": Reverb(),
            "output": Output(),
        }
        # Voice-friendly starting points for the shared blocks.
        g = self.blocks["gate"]
        g.threshold_db, g.release_ms = -50.0, 150.0
        c = self.blocks["comp"]
        c.threshold_db, c.ratio, c.attack_ms, c.release_ms = -18.0, 3.0, 5.0, 120.0
        self.blocks["output"].volume_db = 0.0
        self.blocks["reverb"].mode = "Plate"
        self.blocks["delay"].mode = "Digital"
        self.input_gain_db = 0.0
        self.global_bypass = False
        self.input_peak = 0.0
        self._in_gain = SmoothedParam(1.0)

    def __getitem__(self, key: str):
        return self.blocks[key]

    def prepare(self, sr: float, max_block: int) -> None:
        for b in self.blocks.values():
            b.prepare(sr, max_block)
        self._in_gain.prepare(max_block)
        self._in_gain.snap(db2lin(self.input_gain_db))

    def reset(self) -> None:
        for b in self.blocks.values():
            b.reset()

    def process(self, mono_in: np.ndarray) -> np.ndarray:
        n = mono_in.shape[-1]
        self.input_peak = float(np.max(np.abs(mono_in))) if n else 0.0
        self._in_gain.set(db2lin(clampf(self.input_gain_db, -24.0, 36.0)))
        sig = mono_in * self._in_gain.ramp(n)
        if self.global_bypass:
            return sanitize(np.stack((sig, sig)))
        for key in MONO_ORDER:
            sig = self.blocks[key].process(sig)
        stereo = np.stack((sig, sig))
        for key in STEREO_ORDER:
            stereo = self.blocks[key].process(stereo)
        return sanitize(self.blocks["output"].process(stereo))

    def snapshot(self) -> dict:
        state = {"input_gain_db": self.input_gain_db}
        for key, names in self.PARAM_SPEC.items():
            b = self.blocks[key]
            state[key] = {nm: _copy(getattr(b, nm)) for nm in names}
        return state

    def apply(self, state: dict) -> None:
        """Load a vocal preset. Same rules as the guitar's ``Chain.apply``:
        anything the preset leaves out goes back to its default, and unusable
        values fall back to the default instead of raising."""
        if not isinstance(state, dict):
            return
        if "input_gain_db" in state:
            self.input_gain_db = coerce(state["input_gain_db"], self.input_gain_db)
        for key, names in self.PARAM_SPEC.items():
            src = state.get(key)
            if not isinstance(src, dict):
                src = {}
            b = self.blocks[key]
            defaults = _DEFAULTS[key]
            for nm in names:
                setattr(b, nm, coerce(src.get(nm, defaults[nm]), defaults[nm]))


def _copy(v):
    return list(v) if isinstance(v, list) else v


_DEFAULTS = VocalChain().snapshot()


class VocalsInstrument(Instrument):
    key = "vocals"
    label = "Vocals"
    accent = "#ff5fd2"

    def __init__(self, input_channel: int = 1):
        super().__init__(input_channel)
        # Off until the user turns it on, so an open mic never leaks into a
        # guitar session by surprise.
        self.enabled = False
        self.chain = VocalChain()
        self.preset = "Natural Polish"
        self.chain.apply(VOCAL_PRESETS[self.preset])

    @property
    def autotune(self) -> AutoTune:
        return self.chain["autotune"]

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


def _at(**kw):
    """An auto-tune section: key and scale are left to the user's current
    choice when a preset is loaded (see ``apply_vocal_preset``)."""
    base = {"enabled": True}
    base.update(kw)
    return base


# Factory vocal presets. Like guitar presets, each is a complete sound:
# anything not listed is the block's default.
VOCAL_PRESETS: dict[str, dict] = {
    "Natural Polish": {
        "gate": {"enabled": True, "threshold_db": -52.0},
        "autotune": _at(retune_ms=60.0, humanize=0.5, mix=1.0),
        "eq": {"enabled": True, "low_cut_hz": 90.0, "presence_db": 2.0, "air_db": 1.5},
        "comp": {"enabled": True, "threshold_db": -20.0, "ratio": 3.0,
                 "attack_ms": 5.0, "release_ms": 120.0, "makeup_db": 3.0},
        "reverb": {"enabled": True, "mode": "Plate", "size": 0.35, "mix": 0.12},
    },
    "Hard Tune": {
        "gate": {"enabled": True, "threshold_db": -50.0},
        "deesser": {"enabled": True},
        "autotune": _at(retune_ms=0.0, humanize=0.0, mix=1.0),
        "eq": {"enabled": True, "low_cut_hz": 110.0, "presence_db": 3.0, "air_db": 3.0},
        "comp": {"enabled": True, "threshold_db": -22.0, "ratio": 4.0,
                 "attack_ms": 3.0, "release_ms": 100.0, "makeup_db": 4.0},
        "delay": {"enabled": True, "mode": "Digital", "time_ms": 250.0,
                  "feedback": 0.2, "mix": 0.12},
        "reverb": {"enabled": True, "mode": "Plate", "size": 0.4, "mix": 0.15},
    },
    "Radio Voice": {
        "gate": {"enabled": True, "threshold_db": -48.0},
        "deesser": {"enabled": True, "threshold_db": -28.0},
        "autotune": _at(retune_ms=40.0, humanize=0.3, mix=1.0),
        "eq": {"enabled": True, "low_cut_hz": 140.0, "presence_db": 5.0, "air_db": 0.0},
        "comp": {"enabled": True, "threshold_db": -26.0, "ratio": 6.0,
                 "attack_ms": 2.0, "release_ms": 80.0, "makeup_db": 7.0},
        "saturation": {"enabled": True, "drive_db": 9.0, "mix": 0.6},
    },
    "Big Choir": {
        "gate": {"enabled": True, "threshold_db": -50.0},
        "autotune": _at(retune_ms=25.0, humanize=0.2, mix=1.0),
        "eq": {"enabled": True, "low_cut_hz": 100.0, "presence_db": 2.0, "air_db": 3.0},
        "comp": {"enabled": True, "threshold_db": -20.0, "ratio": 3.0, "makeup_db": 3.0},
        "doubler": {"enabled": True, "mix": 0.8, "width": 1.0, "detune": 0.7},
        "reverb": {"enabled": True, "mode": "Hall", "size": 0.8, "mix": 0.35},
    },
    "Chat Voice": {
        "gate": {"enabled": True, "threshold_db": -48.0, "release_ms": 200.0},
        "autotune": {"enabled": False},
        "eq": {"enabled": True, "low_cut_hz": 100.0, "presence_db": 1.5, "air_db": 0.0},
        "comp": {"enabled": True, "threshold_db": -22.0, "ratio": 3.0, "makeup_db": 4.0},
    },
}

# Settings that belong to the song, not to the sound: a preset never changes
# them, so switching from Natural Polish to Hard Tune keeps your key.
_KEEP_ON_PRESET = ("key", "scale", "custom", "low_note", "high_note", "octave")


def apply_vocal_preset(inst: VocalsInstrument, name: str) -> bool:
    """Load a factory vocal preset, keeping the user's key and range.
    Run it through ``engine.post`` while audio is running."""
    preset = VOCAL_PRESETS.get(name)
    if preset is None:
        return False
    at = inst.autotune
    keep = {k: _copy(getattr(at, k)) for k in _KEEP_ON_PRESET}
    inst.chain.apply(preset)
    for k, v in keep.items():
        setattr(at, k, v)
    inst.preset = name
    return True
