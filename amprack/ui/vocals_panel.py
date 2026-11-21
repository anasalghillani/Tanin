"""Vocals tab (temporary home until the suite's own Vocals page, phase 4).

Two rows, read left to right in the order you set things up:

    on/off · mic input · preset · key · scale · range · octave · formants · note
    chain switches                    retune · humanize · tune mix · level · Discord

A third row of 12 note switches appears only for the Custom scale.

Everything is a plain attribute write on the vocal chain (safe from the UI
thread, see engine.py); only preset loads go through ``engine.post`` so a
whole preset lands between two audio blocks.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..dsp.scales import NOTE_NAMES, RANGE_HIGH, RANGE_LOW, SCALE_NAMES, note_name
from ..instruments.vocals import VOCAL_PRESETS, apply_vocal_preset
from ..tuner import freq_to_note
from . import theme
from .widgets import Knob, PowerLED

ACCENT = "#ff5fd2"

# (block key, chip label, tooltip)
CHIPS = (
    ("gate", "Gate", "Noise gate: silences the mic between phrases"),
    ("deesser", "De-ess", "Softens harsh S and SH sounds"),
    ("autotune", "Tune", "Auto-tune on / off"),
    ("eq", "EQ", "Low cut (rumble) plus presence and air"),
    ("comp", "Comp", "Compressor: evens out loud and quiet words"),
    ("saturation", "Warmth", "Gentle tape-style saturation"),
    ("doubler", "Double", "Double-tracked, wide stereo voice"),
    ("delay", "Delay", "Echo"),
    ("reverb", "Reverb", "Room / plate / hall"),
)


def _label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Hint")
    return lab


class VocalsPanel(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.inst = engine.vocals
        self.chain = self.inst.chain
        self.at = self.inst.autotune
        self._syncing = False
        self.on_resize = None     # the main window sets this to re-fit the dock

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 8, 14, 8)
        v.setSpacing(6)

        # -- row 1: setup ---------------------------------------------------
        r1 = QHBoxLayout()
        r1.setSpacing(8)
        self.led = PowerLED(ACCENT, 20)
        self.led.setToolTip("Vocals on / off. Off costs no CPU.")
        self.led.toggled.connect(self._set_enabled)
        r1.addWidget(self.led)
        title = QLabel("VOCALS")
        title.setObjectName("Section")
        title.setStyleSheet(f"color:{ACCENT};")
        r1.addWidget(title)
        r1.addSpacing(6)

        self.in_combo = QComboBox()
        self.in_combo.setToolTip("Which jack on your interface the mic is in")
        self.in_combo.currentIndexChanged.connect(self._set_input)
        r1.addWidget(_label("Mic"))
        r1.addWidget(self.in_combo)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(VOCAL_PRESETS))
        self.preset_combo.setToolTip("Vocal sound. Keeps your key and range.")
        self.preset_combo.activated.connect(self._load_preset)
        r1.addWidget(_label("Preset"))
        r1.addWidget(self.preset_combo)

        self.key_combo = QComboBox()
        self.key_combo.addItems(list(NOTE_NAMES))
        self.key_combo.setToolTip("Key of the song")
        self.key_combo.currentIndexChanged.connect(
            lambda i: self._set_at("key", int(i)))
        r1.addWidget(_label("Key"))
        r1.addWidget(self.key_combo)

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(list(SCALE_NAMES))
        self.scale_combo.setToolTip("Which notes of the key are allowed")
        self.scale_combo.currentTextChanged.connect(self._set_scale)
        r1.addWidget(self.scale_combo)

        self.low_combo = QComboBox()
        self.high_combo = QComboBox()
        for m in range(RANGE_LOW, RANGE_HIGH + 1):
            self.low_combo.addItem(note_name(m), m)
            self.high_combo.addItem(note_name(m), m)
        self.low_combo.setToolTip("Lowest note auto-tune may pick")
        self.high_combo.setToolTip("Highest note auto-tune may pick")
        self.low_combo.currentIndexChanged.connect(self._set_range)
        self.high_combo.currentIndexChanged.connect(self._set_range)
        r1.addWidget(_label("Range"))
        r1.addWidget(self.low_combo)
        r1.addWidget(_label("to"))
        r1.addWidget(self.high_combo)

        self.oct_combo = QComboBox()
        for o in (-2, -1, 0, 1, 2):
            self.oct_combo.addItem(f"{o:+d} oct" if o else "0 oct", o)
        self.oct_combo.setToolTip("Sing low, come out high (or the other way)")
        self.oct_combo.currentIndexChanged.connect(
            lambda _i: self._set_at("octave", int(self.oct_combo.currentData())))
        r1.addWidget(self.oct_combo)

        self.formant = QCheckBox("Keep voice colour")
        self.formant.setToolTip(
            "On: an octave shift still sounds like you. "
            "Off: chipmunk / monster effect.")
        self.formant.toggled.connect(lambda on: self._set_at("formant", bool(on)))
        r1.addStretch(1)

        self.note = QLabel("—")
        self.note.setStyleSheet(
            f"color:{ACCENT}; font-size:22px; font-weight:600;")
        self.note.setMinimumWidth(70)
        self.note.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.note_detail = _label("")
        r1.addWidget(self.note_detail)
        r1.addWidget(self.note)
        v.addLayout(r1)

        # -- row 2: chain switches and knobs --------------------------------
        r2 = QHBoxLayout()
        r2.setSpacing(6)
        self.chips: dict[str, QPushButton] = {}
        for key, text, tip in CHIPS:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setMinimumWidth(52)
            b.toggled.connect(lambda on, k=key: self._set_block(k, on))
            self.chips[key] = b
            r2.addWidget(b)
        r2.addStretch(1)

        self.k_retune = Knob("retune", 0, 400, 20, " ms", 0, ACCENT, 2.0, 42)
        self.k_retune.setToolTip("0 ms: hard robot snap. 400 ms: natural, "
                                 "keeps your vibrato.")
        self.k_retune.valueChanged.connect(lambda x: self._set_at("retune_ms", float(x)))
        self.k_human = Knob("humanize", 0, 100, 0, "%", 0, ACCENT, 1.0, 42)
        self.k_human.setToolTip("Lets long held notes keep their natural wobble")
        self.k_human.valueChanged.connect(lambda x: self._set_at("humanize", x / 100.0))
        self.k_mix = Knob("tune mix", 0, 100, 100, "%", 0, ACCENT, 1.0, 42)
        self.k_mix.setToolTip("How much of the tuned voice you hear")
        self.k_mix.valueChanged.connect(lambda x: self._set_at("mix", x / 100.0))
        self.k_level = Knob("level", -40, 6, 0, " dB", 1, theme.AMBER, 1.0, 42)
        self.k_level.setToolTip("Vocal level in the mix")
        self.k_level.valueChanged.connect(
            lambda x: setattr(self.engine.mixer["vocals"], "volume_db", float(x)))
        for k in (self.k_retune, self.k_human, self.k_mix, self.k_level):
            r2.addWidget(k)

        self.discord = QCheckBox("Send to Stream Out")
        self.discord.setToolTip(
            "Include your voice in Stream Out (Discord / OBS). Off: only you "
            "and the recording hear it.")
        self.discord.toggled.connect(
            lambda on: self.engine.mixer["vocals"].sends.__setitem__(
                "stream", 1.0 if on else 0.0))
        toggles = QVBoxLayout()
        toggles.setSpacing(4)
        toggles.addWidget(self.formant)
        toggles.addWidget(self.discord)
        r2.addSpacing(8)
        r2.addLayout(toggles)
        v.addLayout(r2)

        # -- row 3: custom scale notes ---------------------------------------
        self.custom_row = QWidget()
        r3 = QHBoxLayout(self.custom_row)
        r3.setContentsMargins(0, 0, 0, 0)
        r3.setSpacing(4)
        r3.addWidget(_label("Custom notes"))
        self.note_btns = []
        for pc, name in enumerate(NOTE_NAMES):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setFixedWidth(46)
            b.toggled.connect(self._set_custom)
            self.note_btns.append(b)
            r3.addWidget(b)
        r3.addStretch(1)
        v.addWidget(self.custom_row)

        for combo in (self.in_combo, self.preset_combo, self.key_combo,
                      self.scale_combo, self.low_combo, self.high_combo,
                      self.oct_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.sync_from_engine()

    def preferred_height(self) -> int:
        """Height the dock needs for this tab (more with the custom row)."""
        return 142 + (40 if self.custom_row.isVisibleTo(self) else 0)

    # -- control writes ------------------------------------------------------

    def _set_enabled(self, on: bool) -> None:
        if self._syncing:
            return
        inst = self.inst
        if on and not inst.enabled:
            # Clear stale reverb / delay tails before it is heard again.
            self.engine.post(inst.reset)
        inst.enabled = bool(on)

    def _set_input(self, _i: int) -> None:
        d = self.in_combo.currentData()
        if d is not None and not self._syncing:
            self.inst.input_channel = int(d)

    def _set_at(self, name: str, value) -> None:
        if not self._syncing:
            setattr(self.at, name, value)

    def _set_scale(self, name: str) -> None:
        self.custom_row.setVisible(name == "Custom")
        self._set_at("scale", name)
        if self.on_resize is not None:
            self.on_resize()

    def _set_range(self, _i: int = 0) -> None:
        if self._syncing:
            return
        lo = int(self.low_combo.currentData())
        hi = int(self.high_combo.currentData())
        if hi < lo:          # keep the range the right way round
            self._syncing = True
            self.high_combo.setCurrentIndex(self.low_combo.currentIndex())
            self._syncing = False
            hi = lo
        self.at.low_note, self.at.high_note = lo, hi

    def _set_custom(self, _on: bool = False) -> None:
        if not self._syncing:
            self.at.custom = [b.isChecked() for b in self.note_btns]

    def _set_block(self, key: str, on: bool) -> None:
        if not self._syncing:
            self.chain[key].enabled = bool(on)

    def _load_preset(self, _i: int = 0) -> None:
        name = self.preset_combo.currentText()
        self.engine.post(lambda: apply_vocal_preset(self.inst, name))
        # When audio runs, the preset lands at the next block (~5 ms).
        QTimer.singleShot(60, self.sync_from_engine)

    # -- engine -> widgets ---------------------------------------------------

    def set_input_channels(self, n: int) -> None:
        keep = self.inst.input_channel
        self._syncing = True
        self.in_combo.clear()
        # Always offer two: the AudioBox has two jacks even while audio is
        # stopped, and the engine falls back to input 1 if one is missing.
        for c in range(max(2, n)):
            self.in_combo.addItem(f"In {c + 1}", c)
        idx = self.in_combo.findData(keep)
        self.in_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._syncing = False

    def sync_from_engine(self) -> None:
        at, ch = self.at, self.chain
        self._syncing = True
        try:
            self.led.setChecked(self.inst.enabled, emit=False)
            if self.in_combo.count() == 0:
                for c in range(2):
                    self.in_combo.addItem(f"In {c + 1}", c)
            i = self.in_combo.findData(self.inst.input_channel)
            if i >= 0:
                self.in_combo.setCurrentIndex(i)
            self.preset_combo.setCurrentText(self.inst.preset)
            self.key_combo.setCurrentIndex(int(at.key) % 12)
            self.scale_combo.setCurrentText(at.scale)
            self.custom_row.setVisible(at.scale == "Custom")
            for b, on in zip(self.note_btns, at.custom):
                b.setChecked(bool(on))
            self.low_combo.setCurrentIndex(max(0, self.low_combo.findData(int(at.low_note))))
            self.high_combo.setCurrentIndex(max(0, self.high_combo.findData(int(at.high_note))))
            self.oct_combo.setCurrentIndex(max(0, self.oct_combo.findData(int(at.octave))))
            self.formant.setChecked(bool(at.formant))
            for key, b in self.chips.items():
                b.setChecked(bool(ch[key].enabled))
            self.k_retune.setValue(float(at.retune_ms), emit=False)
            self.k_human.setValue(float(at.humanize) * 100.0, emit=False)
            self.k_mix.setValue(float(at.mix) * 100.0, emit=False)
            strip = self.engine.mixer["vocals"]
            self.k_level.setValue(float(strip.volume_db), emit=False)
            self.discord.setChecked(strip.sends.get("stream", 1.0) > 0.0)
        finally:
            self._syncing = False

    def tick(self) -> None:
        """Called by the main window ~10 times a second: the note readout."""
        at = self.at
        if not (self.inst.enabled and self.engine.running and at.enabled):
            self.note.setText("—")
            self.note_detail.setText("tune off" if self.inst.enabled and not at.enabled
                                     else "")
            return
        if at.target_note < 0 or at.detected_hz <= 0:
            self.note.setText("—")
            self.note_detail.setText("listening…")
            return
        name, octv, cents = freq_to_note(at.detected_hz, at.a4)
        self.note.setText(note_name(int(at.target_note)))
        self.note_detail.setText(f"you sing {name}{octv} {cents:+.0f}¢  →")

    # -- settings ------------------------------------------------------------

    def state(self) -> dict:
        strip = self.engine.mixer["vocals"]
        return {
            "enabled": bool(self.inst.enabled),
            "input_channel": self.inst.input_channel,
            "preset": self.inst.preset,
            "chain": self.inst.snapshot(),
            "volume_db": strip.volume_db,
            "to_stream": strip.sends.get("stream", 1.0) > 0.0,
        }

    def restore(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        inst = self.inst
        if isinstance(data.get("chain"), dict):
            inst.apply(data["chain"])
        if data.get("preset") in VOCAL_PRESETS:
            inst.preset = data["preset"]
        ch = data.get("input_channel")
        if isinstance(ch, int) and 0 <= ch < 8:
            inst.input_channel = ch
        inst.enabled = bool(data.get("enabled", False))
        strip = self.engine.mixer["vocals"]
        vol = data.get("volume_db", 0.0)
        if isinstance(vol, (int, float)):
            strip.volume_db = float(vol)
        strip.sends["stream"] = 1.0 if data.get("to_stream", True) else 0.0
        self.sync_from_engine()
