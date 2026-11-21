"""A rack of effect slots that can each hold any slot effect.

The chain has three racks, matching where gear sits on a real rig:

*   ``pre``  -- 4 mono slots in front of the amp (stomp boxes)
*   ``loop`` -- 4 stereo slots after the cab (the effects loop)
*   ``rack`` -- 4 stereo slots at the very end (studio rack)

Switching what a slot holds must not allocate on the audio thread (presets
arrive through the engine's command queue and are applied there), so every
slot keeps one *prepared* instance of every effect type and switching is a
reference swap plus a state reset. That costs memory up front (measured in
tools/fx_tests.py) and buys glitch-free tone switches mid-song.
"""

from __future__ import annotations

import numpy as np

from ..blocks import Block
from . import FX_TYPES

N_SLOTS = 4


class Slot:
    __slots__ = ("type", "enabled", "pool", "fx")

    def __init__(self):
        self.type = ""
        self.enabled = True
        self.pool: dict = {}
        self.fx = None


class FxSlots(Block):
    stereo = True

    def __init__(self, name: str, channels: int, n_slots: int = N_SLOTS):
        super().__init__()
        self.name = name
        self.channels = channels
        self.stereo = channels == 2
        self.enabled = True
        self.slots = [Slot() for _ in range(n_slots)]
        for s in self.slots:
            for key, cls in FX_TYPES.items():
                fx = cls()
                fx.channels = 2 if cls.STEREO else channels
                s.pool[key] = fx

    # -- lifecycle ----------------------------------------------------------

    def prepare(self, sr, max_block):
        super().prepare(sr, max_block)
        for s in self.slots:
            for fx in s.pool.values():
                fx.channels = 2 if fx.STEREO else self.channels
                fx.prepare(sr, max_block)
                fx.reset()

    def reset(self):
        for s in self.slots:
            if s.fx is not None:
                s.fx.reset()

    # -- control ------------------------------------------------------------

    def set_type(self, i: int, fx_type: str) -> None:
        """Put an effect type in slot i ('' empties it). Allocation-free."""
        if not 0 <= i < len(self.slots):
            return
        s = self.slots[i]
        fx_type = fx_type if fx_type in FX_TYPES else ""
        if fx_type == s.type:
            return
        new = s.pool.get(fx_type) if fx_type else None
        if new is not None:
            new.reset()
            new.set_params({})
        s.type = fx_type
        s.fx = new

    def effect(self, i: int):
        return self.slots[i].fx if 0 <= i < len(self.slots) else None

    @property
    def active(self) -> bool:
        return any(s.fx is not None and s.enabled and s.fx.enabled for s in self.slots)

    # -- state (presets) ----------------------------------------------------

    def state(self) -> dict:
        return {
            "enabled": self.enabled,
            "slots": [
                {"type": s.type, "enabled": s.enabled,
                 "params": s.fx.get_params() if s.fx is not None else {}}
                for s in self.slots
            ],
        }

    def apply_state(self, src) -> None:
        """Load from a preset dict. Missing or malformed means empty/default.
        Runs on the audio thread: no allocation beyond small dicts."""
        if not isinstance(src, dict):
            src = {}
        en = src.get("enabled", True)
        self.enabled = en if isinstance(en, bool) else True
        slots = src.get("slots", [])
        if not isinstance(slots, list):
            slots = []
        for i, s in enumerate(self.slots):
            d = slots[i] if i < len(slots) and isinstance(slots[i], dict) else {}
            t = d.get("type", "")
            self.set_type(i, t if isinstance(t, str) else "")
            e = d.get("enabled", True)
            s.enabled = e if isinstance(e, bool) else True
            if s.fx is not None:
                s.fx.set_params(d.get("params", {}))

    # -- audio --------------------------------------------------------------

    def process(self, x):
        if not self.enabled:
            return x
        n = x.shape[-1]
        if n > self.max_block or n == 0:
            return x
        mono_path = x.ndim == 1
        sig = x[None, :] if mono_path else x
        touched = False
        for s in self.slots:
            fx = s.fx
            if fx is None or not s.enabled or not fx.enabled:
                continue
            touched = True
            if fx.STEREO and sig.shape[0] == 1:
                y = fx.process(np.concatenate((sig, sig)))
                sig = ((y[0] + y[1]) * 0.5)[None, :]
            else:
                sig = fx.process(sig)
        if not touched:
            return x
        return sig[0] if mono_path else sig
