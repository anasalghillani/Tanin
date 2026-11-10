"""Pedal cards.

Each card is a thin view over one DSP block. Knobs write straight to block
attributes -- a float store the audio thread picks up on its next block. No
queue, no lock, and at worst a knob move lands a few milliseconds late.

Layouts are declared as data at the bottom of this file, so adding a control
is one line rather than a new widget class.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..dsp.blocks import AMP_MODELS, CAB_MODELS, DRIVE_MODELS, MIC_TILT, Delay, Modulation, Reverb
from . import theme
from .widgets import Knob, PowerLED

KNOBS_PER_ROW = 4
KNOB_SIZE = 48


class KnobSpec:
    __slots__ = ("attr", "label", "lo", "hi", "default", "unit", "dec", "curve",
                 "mul", "tip")

    def __init__(self, attr, label, lo, hi, default, unit="", dec=1, curve=1.0,
                 mul=1.0, tip=""):
        self.attr = attr
        self.label = label
        self.lo = lo
        self.hi = hi
        self.default = default
        self.unit = unit
        self.dec = dec
        self.curve = curve
        self.mul = mul       # block attribute = knob value * mul
        self.tip = tip


class SelectSpec:
    __slots__ = ("attr", "options", "label")

    def __init__(self, attr, options, label=""):
        self.attr = attr
        self.options = list(options)
        self.label = label


class PedalCard(QFrame):
    """One module: power LED, optional selectors, a grid of knobs."""

    def __init__(self, engine, key: str, title: str,
                 selectors: list[SelectSpec], knobs: list[KnobSpec],
                 always_on: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.engine = engine
        self.key = key
        self.block = engine.chain[key]
        self.accent = theme.accent(key)
        self.always_on = always_on
        self._knobs: dict[str, Knob] = {}
        self._selects: dict[str, QComboBox] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 9, 12, 11)
        root.setSpacing(8)

        # -- header
        head = QHBoxLayout()
        head.setSpacing(8)
        if always_on:
            self.led = None
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{self.accent}; font-size:13px;")
            head.addWidget(dot)
        else:
            self.led = PowerLED(self.accent, 18)
            self.led.setChecked(bool(self.block.enabled), emit=False)
            self.led.toggled.connect(self._on_power)
            self.led.setToolTip("Enable / bypass this module")
            head.addWidget(self.led)
        name = QLabel(title)
        name.setObjectName("CardTitle")
        head.addWidget(name)
        head.addStretch(1)
        root.addLayout(head)

        # -- selectors
        for spec in selectors:
            row = QHBoxLayout()
            row.setSpacing(6)
            if spec.label:
                lab = QLabel(spec.label)
                lab.setObjectName("Section")
                row.addWidget(lab)
            combo = QComboBox()
            combo.addItems(spec.options)
            current = getattr(self.block, spec.attr, spec.options[0])
            if current in spec.options:
                combo.setCurrentText(current)
            combo.currentTextChanged.connect(
                lambda v, a=spec.attr: setattr(self.block, a, v))
            row.addWidget(combo, 1)
            self._selects[spec.attr] = combo
            root.addLayout(row)

        # -- knobs
        grid = QGridLayout()
        grid.setSpacing(2)
        grid.setContentsMargins(0, 2, 0, 0)
        for i, ks in enumerate(knobs):
            k = Knob(ks.label, ks.lo, ks.hi, ks.default, ks.unit, ks.dec,
                     self.accent, ks.curve, KNOB_SIZE)
            k.setValue(getattr(self.block, ks.attr, ks.default * ks.mul) / ks.mul,
                       emit=False)
            k.valueChanged.connect(
                lambda v, a=ks.attr, m=ks.mul: setattr(self.block, a, v * m))
            k.setToolTip(ks.tip or f"{ks.label.title()} — drag to set, "
                                   "shift for fine, double-click to reset")
            self._knobs[ks.attr] = k
            grid.addWidget(k, i // KNOBS_PER_ROW, i % KNOBS_PER_ROW,
                           Qt.AlignHCenter | Qt.AlignTop)
        root.addLayout(grid)
        root.addStretch(1)

        self._sync_active()

    # -- power -------------------------------------------------------------

    def _on_power(self, on: bool) -> None:
        self.block.enabled = on
        self._sync_active()

    def _sync_active(self) -> None:
        on = self.always_on or bool(self.block.enabled)
        self.setProperty("active", "true" if on else "false")
        for k in self._knobs.values():
            k.setActive(on)
        for c in self._selects.values():
            c.setEnabled(on)
        self.style().unpolish(self)
        self.style().polish(self)

    # -- preset load -------------------------------------------------------

    def refresh_from_block(self) -> None:
        """Pull every control back from the DSP block after a preset load."""
        if self.led is not None:
            self.led.setChecked(bool(self.block.enabled), emit=False)
        for attr, combo in self._selects.items():
            v = getattr(self.block, attr, None)
            if isinstance(v, str) and combo.findText(v) >= 0:
                combo.blockSignals(True)
                combo.setCurrentText(v)
                combo.blockSignals(False)
        for attr, knob in self._knobs.items():
            spec_mul = self._mul.get(attr, 1.0)
            knob.setValue(getattr(self.block, attr, 0.0) / spec_mul, emit=False)
        self._sync_active()

    _mul: dict[str, float] = {}


class IRCard(PedalCard):
    """The cabinet-IR loader needs a file picker, so it gets its own subclass."""

    def __init__(self, engine, parent=None):
        super().__init__(
            engine, "ir", "IR LOADER", [],
            [KnobSpec("mix", "mix", 0, 100, 100, "%", 0, 1.0, 0.01),
             KnobSpec("gain_db", "trim", -24, 24, 0, " dB", 1)],
            parent=parent)
        self.file_label = QLabel("no impulse loaded")
        self.file_label.setObjectName("Hint")
        self.file_label.setWordWrap(True)

        row = QHBoxLayout()
        row.setSpacing(6)
        load = QPushButton("Load .wav")
        load.clicked.connect(self._pick)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        row.addWidget(load, 1)
        row.addWidget(clear)

        lay = self.layout()
        lay.insertWidget(1, self.file_label)
        lay.insertLayout(2, row)

    def _pick(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load impulse response", "", "Audio (*.wav *.aiff *.flac)")
        if not path:
            return
        err = self.engine.load_ir(path)
        if err:
            self.file_label.setText(f"failed: {err}")
            self.file_label.setStyleSheet(f"color:{theme.RED};")
            return
        self.file_label.setText(os.path.basename(path))
        self.file_label.setStyleSheet(f"color:{theme.TEXT_DIM};")
        if self.led is not None:
            self.led.setChecked(True)

    def _clear(self) -> None:
        self.engine.clear_ir()
        self.file_label.setText("no impulse loaded")
        self.file_label.setStyleSheet("")
        if self.led is not None:
            self.led.setChecked(False)


# ==========================================================================
# The board
# ==========================================================================

PCT = 0.01   # knob shows 0-100, block wants 0-1
TEN = 0.1    # knob shows 0-10 like an amp panel, block wants 0-1

CARDS = {
    "gate": dict(
        title="NOISE GATE", selectors=[], knobs=[
            KnobSpec("threshold_db", "thresh", -80, -20, -55, " dB", 0,
                     tip="Below this level the signal is muted. Raise it until "
                         "hum disappears, then back off slightly."),
            KnobSpec("attack_ms", "open", 0.5, 20, 2, " ms", 1),
            KnobSpec("release_ms", "close", 20, 600, 120, " ms", 0),
        ]),
    "comp": dict(
        title="COMPRESSOR", selectors=[], knobs=[
            KnobSpec("threshold_db", "thresh", -50, 0, -20, " dB", 0),
            KnobSpec("ratio", "ratio", 1.5, 20, 4, ":1", 1, curve=2.0),
            KnobSpec("attack_ms", "attack", 1, 60, 8, " ms", 1, curve=2.0),
            KnobSpec("release_ms", "release", 20, 800, 150, " ms", 0, curve=2.0),
            KnobSpec("makeup_db", "makeup", 0, 24, 0, " dB", 1),
            KnobSpec("mix", "mix", 0, 100, 100, "%", 0, mul=PCT,
                     tip="Blend in the uncompressed signal to keep the pick "
                         "attack while still levelling the sustain."),
        ]),
    "drive": dict(
        title="DRIVE", selectors=[SelectSpec("model", list(DRIVE_MODELS))],
        knobs=[
            KnobSpec("drive", "drive", 0, 10, 3.5, "", 1, mul=TEN),
            KnobSpec("tone", "tone", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("level", "level", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("mix", "mix", 0, 100, 100, "%", 0, mul=PCT),
        ]),
    "amp": dict(
        title="AMPLIFIER", selectors=[SelectSpec("model", list(AMP_MODELS))],
        always_on=True, knobs=[
            KnobSpec("gain", "gain", 0, 10, 4.5, "", 1, mul=TEN),
            KnobSpec("bass", "bass", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("mid", "mid", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("treble", "treble", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("presence", "presence", 0, 10, 3.5, "", 1, mul=TEN),
            KnobSpec("sag", "sag", 0, 10, 3, "", 1, mul=TEN,
                     tip="Power-supply sag. Softens the attack of loud notes "
                         "the way a cranked valve amp does."),
            KnobSpec("master", "master", 0, 10, 5, "", 1, mul=TEN),
        ]),
    "cab": dict(
        title="CABINET",
        selectors=[SelectSpec("model", list(CAB_MODELS)),
                   SelectSpec("mic", list(MIC_TILT))],
        knobs=[
            KnobSpec("position", "position", 0, 10, 5, "", 1, mul=TEN,
                     tip="0 = cone edge (dark and round), 10 = dust cap "
                         "(bright and aggressive)."),
            KnobSpec("distance", "distance", 0, 10, 3, "", 1, mul=TEN,
                     tip="Closer adds low end via proximity effect."),
        ]),
    "mod": dict(
        title="MODULATION", selectors=[SelectSpec("mode", Modulation.MODES)],
        knobs=[
            KnobSpec("rate", "rate", 0, 10, 6, "", 1, mul=TEN),
            KnobSpec("depth", "depth", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("feedback", "regen", 0, 10, 3, "", 1, mul=TEN),
            KnobSpec("stereo_spread", "width", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("mix", "mix", 0, 100, 50, "%", 0, mul=PCT),
        ]),
    "delay": dict(
        title="DELAY", selectors=[SelectSpec("mode", Delay.MODES)],
        knobs=[
            KnobSpec("time_ms", "time", 20, 2000, 380, " ms", 0, curve=2.0),
            KnobSpec("feedback", "repeats", 0, 95, 35, "%", 0, mul=PCT),
            KnobSpec("tone", "tone", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("modulation", "wow", 0, 10, 1.5, "", 1, mul=TEN,
                     tip="Tape-style pitch drift. Only active in Tape mode."),
            KnobSpec("mix", "mix", 0, 100, 30, "%", 0, mul=PCT),
        ]),
    "reverb": dict(
        title="REVERB", selectors=[SelectSpec("mode", Reverb.MODES)],
        knobs=[
            KnobSpec("size", "size", 0, 10, 6, "", 1, mul=TEN),
            KnobSpec("damping", "damp", 0, 10, 4.5, "", 1, mul=TEN),
            KnobSpec("predelay_ms", "predelay", 0, 150, 15, " ms", 0),
            KnobSpec("width", "width", 0, 10, 8, "", 1, mul=TEN),
            KnobSpec("mix", "mix", 0, 100, 25, "%", 0, mul=PCT),
        ]),
}


def build_card(engine, key: str, parent=None) -> PedalCard:
    if key == "ir":
        card = IRCard(engine, parent)
        card._mul = {"mix": PCT, "gain_db": 1.0}
        return card
    spec = CARDS[key]
    card = PedalCard(engine, key, spec["title"], spec["selectors"], spec["knobs"],
                     spec.get("always_on", False), parent)
    card._mul = {k.attr: k.mul for k in spec["knobs"]}
    return card
