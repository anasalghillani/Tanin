"""Keys, scales and note snapping for auto-tune (and later the piano's chord mode).

Notes are MIDI numbers: 60 is middle C (C4), 69 is A4. Pitch is converted to
a *fractional* MIDI number so "how far from the note" is just a subtraction.
"""

from __future__ import annotations

import math

import numpy as np

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Semitone steps above the key's root that belong to each scale.
SCALES: dict[str, tuple[int, ...]] = {
    "Chromatic": tuple(range(12)),
    "Major": (0, 2, 4, 5, 7, 9, 11),
    "Natural Minor": (0, 2, 3, 5, 7, 8, 10),
    "Harmonic Minor": (0, 2, 3, 5, 7, 8, 11),
    "Melodic Minor": (0, 2, 3, 5, 7, 9, 11),
    "Major Pentatonic": (0, 2, 4, 7, 9),
    "Minor Pentatonic": (0, 3, 5, 7, 10),
    "Blues": (0, 3, 5, 6, 7, 10),
    "Dorian": (0, 2, 3, 5, 7, 9, 10),
    "Phrygian": (0, 1, 3, 5, 7, 8, 10),
    "Mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "Custom": (),          # uses the 12 toggles in ``custom`` instead
}
SCALE_NAMES = tuple(SCALES)

# C1 .. C6 is the auto-tune range the plan promises.
RANGE_LOW = 24     # C1
RANGE_HIGH = 84    # C6


def hz_to_midi(f: float, a4: float = 440.0) -> float:
    return 69.0 + 12.0 * math.log2(f / a4)


def midi_to_hz(m: float, a4: float = 440.0) -> float:
    return a4 * 2.0 ** ((m - 69.0) / 12.0)


def note_name(m: int) -> str:
    """60 -> 'C4'."""
    m = int(m)
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def parse_note(name: str) -> int:
    """'C4' -> 60, 'F#2' -> 42. Raises ValueError on junk."""
    s = name.strip()
    for k in (2, 1):
        head, tail = s[:k], s[k:]
        if head in NOTE_NAMES:
            try:
                return NOTE_NAMES.index(head) + 12 * (int(tail) + 1)
            except ValueError:
                break
    raise ValueError(f"not a note: {name!r}")


def pitch_class_mask(key: int, scale: str, custom=None) -> np.ndarray:
    """A (12,) bool array: which pitch classes (C=0..B=11) are allowed."""
    mask = np.zeros(12, dtype=bool)
    if scale == "Custom":
        if custom is not None:
            for pc, on in enumerate(list(custom)[:12]):
                mask[pc] = bool(on)
        if not mask.any():           # an empty custom scale would mean "never sing"
            mask[:] = True
        return mask
    steps = SCALES.get(scale, SCALES["Chromatic"])
    for s in steps:
        mask[(int(key) + s) % 12] = True
    return mask


def allowed_notes(mask: np.ndarray, low: int, high: int) -> np.ndarray:
    """MIDI notes inside [low, high] whose pitch class is in ``mask``."""
    low = int(max(0, min(low, 127)))
    high = int(max(low, min(high, 127)))
    notes = [m for m in range(low, high + 1) if mask[m % 12]]
    if not notes:                    # range too narrow for the scale: nearest scale note
        notes = [m for m in range(0, 128) if mask[m % 12]]
    return np.asarray(notes, dtype=np.float64)


def snap(midi: float, notes: np.ndarray) -> float:
    """Nearest allowed note. Ties go down (a stable, predictable choice)."""
    if notes.size == 0:
        return float(round(midi))
    i = int(np.argmin(np.abs(notes - midi + 1e-9)))
    return float(notes[i])
