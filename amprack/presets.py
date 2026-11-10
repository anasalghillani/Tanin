"""Factory presets and preset file handling.

A preset is a partial snapshot: anything it omits keeps the current value,
which is what makes ``Chain.apply`` safe to call on a running stream.
"""

from __future__ import annotations

import json
import os

FACTORY: dict[str, dict] = {
    "Clean Chime": {
        "amp": {"model": "Clean", "gain": 0.28, "bass": 0.55, "mid": 0.45,
                "treble": 0.62, "presence": 0.40, "master": 0.62, "sag": 0.20},
        "cab": {"enabled": True, "model": "2x12 Vintage", "mic": "Condenser",
                "position": 0.45, "distance": 0.35},
        "comp": {"enabled": True, "threshold_db": -24.0, "ratio": 3.0,
                 "attack_ms": 12.0, "release_ms": 180.0, "makeup_db": 4.0, "mix": 0.7},
        "drive": {"enabled": False},
        "mod": {"enabled": True, "mode": "Chorus", "rate": 0.25, "depth": 0.4,
                "mix": 0.35, "stereo_spread": 0.7},
        "delay": {"enabled": False},
        "reverb": {"enabled": True, "mode": "Hall", "size": 0.5, "damping": 0.5,
                   "mix": 0.22, "predelay_ms": 18.0, "width": 0.9},
        "output": {"volume_db": -6.0},
    },
    "Blues Breakup": {
        "amp": {"model": "Crunch", "gain": 0.42, "bass": 0.52, "mid": 0.62,
                "treble": 0.55, "presence": 0.35, "master": 0.55, "sag": 0.55},
        "cab": {"enabled": True, "model": "1x12 Combo", "mic": "Ribbon",
                "position": 0.5, "distance": 0.3},
        "drive": {"enabled": True, "model": "Screamer", "drive": 0.25,
                  "tone": 0.55, "level": 0.55, "mix": 1.0},
        "comp": {"enabled": False},
        "mod": {"enabled": False},
        "delay": {"enabled": True, "mode": "Tape", "time_ms": 340.0,
                  "feedback": 0.22, "mix": 0.18, "tone": 0.4, "modulation": 0.25},
        "reverb": {"enabled": True, "mode": "Room", "size": 0.4, "damping": 0.55,
                   "mix": 0.18},
        "output": {"volume_db": -6.0},
    },
    "Classic Crunch": {
        "amp": {"model": "Crunch", "gain": 0.65, "bass": 0.5, "mid": 0.6,
                "treble": 0.6, "presence": 0.45, "master": 0.5, "sag": 0.35},
        "cab": {"enabled": True, "model": "4x12 Classic", "mic": "Dynamic",
                "position": 0.55, "distance": 0.25},
        "drive": {"enabled": False},
        "gate": {"enabled": True, "threshold_db": -58.0},
        "mod": {"enabled": False},
        "delay": {"enabled": False},
        "reverb": {"enabled": True, "mode": "Room", "size": 0.35, "damping": 0.6,
                   "mix": 0.14},
        "output": {"volume_db": -6.0},
    },
    "Modern Lead": {
        "amp": {"model": "Lead", "gain": 0.72, "bass": 0.45, "mid": 0.65,
                "treble": 0.58, "presence": 0.5, "master": 0.5, "sag": 0.4},
        "cab": {"enabled": True, "model": "4x12 Modern", "mic": "Dynamic",
                "position": 0.6, "distance": 0.25},
        "drive": {"enabled": True, "model": "Boost", "drive": 0.35,
                  "tone": 0.6, "level": 0.5},
        "gate": {"enabled": True, "threshold_db": -52.0},
        "mod": {"enabled": False},
        "delay": {"enabled": True, "mode": "Analog", "time_ms": 420.0,
                  "feedback": 0.34, "mix": 0.26, "tone": 0.45},
        "reverb": {"enabled": True, "mode": "Hall", "size": 0.6, "damping": 0.45,
                   "mix": 0.24, "predelay_ms": 24.0},
        "output": {"volume_db": -7.0},
    },
    "Metal Chug": {
        "amp": {"model": "Metal", "gain": 0.78, "bass": 0.6, "mid": 0.38,
                "treble": 0.62, "presence": 0.55, "master": 0.5, "sag": 0.2},
        "cab": {"enabled": True, "model": "4x12 Modern", "mic": "Dynamic",
                "position": 0.65, "distance": 0.2},
        "gate": {"enabled": True, "threshold_db": -46.0, "attack_ms": 1.0,
                 "release_ms": 60.0},
        "drive": {"enabled": True, "model": "Screamer", "drive": 0.15,
                  "tone": 0.7, "level": 0.5},
        "comp": {"enabled": False},
        "mod": {"enabled": False},
        "delay": {"enabled": False},
        "reverb": {"enabled": True, "mode": "Plate", "size": 0.3, "damping": 0.7,
                   "mix": 0.10},
        "output": {"volume_db": -7.0},
    },
    "Ambient Wash": {
        "amp": {"model": "Clean", "gain": 0.22, "bass": 0.5, "mid": 0.4,
                "treble": 0.6, "presence": 0.3, "master": 0.6, "sag": 0.15},
        "cab": {"enabled": True, "model": "2x12 Vintage", "mic": "Condenser",
                "position": 0.4, "distance": 0.5},
        "comp": {"enabled": True, "threshold_db": -28.0, "ratio": 4.0,
                 "makeup_db": 6.0, "mix": 0.8},
        "drive": {"enabled": False},
        "mod": {"enabled": True, "mode": "Chorus", "rate": 0.12, "depth": 0.7,
                "mix": 0.5, "stereo_spread": 1.0},
        "delay": {"enabled": True, "mode": "Ping-Pong", "time_ms": 560.0,
                  "feedback": 0.55, "mix": 0.42, "tone": 0.35},
        "reverb": {"enabled": True, "mode": "Cathedral", "size": 1.0,
                   "damping": 0.25, "mix": 0.5, "predelay_ms": 40.0, "width": 1.0},
        "output": {"volume_db": -8.0},
    },
    "Funk Squish": {
        "amp": {"model": "Clean", "gain": 0.3, "bass": 0.42, "mid": 0.6,
                "treble": 0.68, "presence": 0.45, "master": 0.6, "sag": 0.25},
        "cab": {"enabled": True, "model": "1x12 Combo", "mic": "Condenser",
                "position": 0.6, "distance": 0.3},
        "comp": {"enabled": True, "threshold_db": -30.0, "ratio": 8.0,
                 "attack_ms": 4.0, "release_ms": 90.0, "makeup_db": 9.0, "mix": 1.0},
        "drive": {"enabled": False},
        "mod": {"enabled": True, "mode": "Phaser", "rate": 0.35, "depth": 0.6,
                "feedback": 0.4, "mix": 0.5},
        "delay": {"enabled": False},
        "reverb": {"enabled": True, "mode": "Room", "size": 0.25, "damping": 0.6,
                   "mix": 0.12},
        "output": {"volume_db": -6.0},
    },
    "Surf Twang": {
        "amp": {"model": "Clean", "gain": 0.35, "bass": 0.58, "mid": 0.4,
                "treble": 0.72, "presence": 0.55, "master": 0.6, "sag": 0.3},
        "cab": {"enabled": True, "model": "2x12 Vintage", "mic": "Dynamic",
                "position": 0.65, "distance": 0.25},
        "drive": {"enabled": False},
        "comp": {"enabled": False},
        "mod": {"enabled": True, "mode": "Tremolo", "rate": 0.4, "depth": 0.7,
                "mix": 1.0, "stereo_spread": 0.2},
        "delay": {"enabled": False},
        "reverb": {"enabled": True, "mode": "Plate", "size": 0.75, "damping": 0.3,
                   "mix": 0.42, "predelay_ms": 8.0, "width": 1.0},
        "output": {"volume_db": -6.0},
    },
    "Fuzz Blowout": {
        "amp": {"model": "Crunch", "gain": 0.5, "bass": 0.6, "mid": 0.5,
                "treble": 0.5, "presence": 0.4, "master": 0.45, "sag": 0.7},
        "cab": {"enabled": True, "model": "4x12 Classic", "mic": "Ribbon",
                "position": 0.4, "distance": 0.2},
        "drive": {"enabled": True, "model": "Fuzz", "drive": 0.7, "tone": 0.45,
                  "level": 0.45, "mix": 1.0},
        "gate": {"enabled": True, "threshold_db": -50.0},
        "mod": {"enabled": False},
        "delay": {"enabled": False},
        "reverb": {"enabled": True, "mode": "Room", "size": 0.45, "damping": 0.5,
                   "mix": 0.16},
        "output": {"volume_db": -8.0},
    },
}


def ensure_factory(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for name, state in FACTORY.items():
        path = os.path.join(directory, f"{name}.json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)


def list_presets(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    names = [os.path.splitext(f)[0] for f in os.listdir(directory)
             if f.lower().endswith(".json")]
    return sorted(names, key=str.lower)


def load_preset(directory: str, name: str) -> dict:
    with open(os.path.join(directory, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def save_preset(directory: str, name: str, state: dict) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    return path


def delete_preset(directory: str, name: str) -> None:
    path = os.path.join(directory, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
