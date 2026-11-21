"""Concrete instruments. See ``amprack/instrument.py`` for the interface."""

from .guitar import GuitarInstrument
from .vocals import VocalsInstrument

__all__ = ["GuitarInstrument", "VocalsInstrument"]
