"""Slot effects: the gear a song tone can call for, and free play too.

``FX_TYPES`` maps a type key (stored in presets) to its class. Each class
describes its own models and parameters (see ``base.Effect``).
"""

from __future__ import annotations

from .acoustic import AcousticSim
from .base import Effect, Param
from .dirt import Boost, Dirt
from .dynamics import Comp, Gate, GraphicEQ, ParametricEQ
from .filters import AutoFilter, BitCrusher, LoFi, Wah
from .mod import Chorus, Flanger, Phaser, Rotary, Tremolo, Vibe
from .pitch import OctaveDown, OctaveUp, PitchShift
from .time import Echo, Verb

_ALL = (
    Wah, AutoFilter, LoFi, BitCrusher,
    OctaveDown, OctaveUp, PitchShift,
    Chorus, Flanger, Phaser, Vibe, Tremolo, Rotary,
    Echo, Verb,
    Dirt, Boost,
    Comp, Gate, GraphicEQ, ParametricEQ,
    AcousticSim,
)

FX_TYPES: dict[str, type[Effect]] = {cls.TYPE: cls for cls in _ALL}

__all__ = ["FX_TYPES", "Effect", "Param"] + [c.__name__ for c in _ALL]
