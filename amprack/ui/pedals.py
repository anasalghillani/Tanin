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
                 always_on: bool = False, parent=None, description: str = "",
                 knob_size: int = KNOB_SIZE, per_row: int = KNOBS_PER_ROW,
                 spacious: bool = False):
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
        root.setSpacing(10)

        # -- header
        head = QHBoxLayout()
        head.setSpacing(8)
        if always_on:
            self.led = None
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background:{self.accent}; border-radius:5px;")
            head.addWidget(dot)
        else:
            self.led = PowerLED(self.accent, 22)
            self.led.setChecked(bool(self.block.enabled), emit=False)
            self.led.toggled.connect(self._on_power)
            self.led.setToolTip("Enable / bypass this module")
            head.addWidget(self.led)
        name = QLabel(title.title() if title.isupper() and len(title) > 3 else title)
        name.setObjectName("CardTitle")
        head.addWidget(name)
        head.addStretch(1)
        root.addLayout(head)
        if description:
            desc = QLabel(description)
            desc.setObjectName("Hint")
            desc.setWordWrap(True)
            root.addWidget(desc)

        # -- selectors
        sel_row = QHBoxLayout()
        sel_row.setSpacing(10)
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
            combo.setMinimumWidth(170)
            row.addWidget(combo, 1)
            self._selects[spec.attr] = combo
            sel_row.addLayout(row)
        if selectors:
            sel_row.addStretch(1)
            root.addLayout(sel_row)

        # -- knobs
        grid = QGridLayout()
        grid.setHorizontalSpacing(max(2, knob_size // 4))
        grid.setVerticalSpacing(2)
        grid.setContentsMargins(0, 2, 0, 0)
        for i, ks in enumerate(knobs):
            k = Knob(ks.label, ks.lo, ks.hi, ks.default, ks.unit, ks.dec,
                     self.accent, ks.curve, knob_size)
            k.setValue(getattr(self.block, ks.attr, ks.default * ks.mul) / ks.mul,
                       emit=False)
            k.valueChanged.connect(
                lambda v, a=ks.attr, m=ks.mul: setattr(self.block, a, v * m))
            k.setToolTip(ks.tip or f"{ks.label.title()} — drag to set, "
                                   "shift for fine, double-click to reset")
            self._knobs[ks.attr] = k
            grid.addWidget(k, i // per_row, i % per_row,
                           Qt.AlignHCenter | Qt.AlignTop)
        knob_row = QHBoxLayout()
        if spacious:
            # Focus view: knobs sit in the middle of the panel, not stuck
            # under the title with a void below them.
            root.addStretch(1)
            knob_row.addStretch(1)
        knob_row.addLayout(grid)
        knob_row.addStretch(1)
        root.addLayout(knob_row)
        root.addStretch(2 if spacious else 1)

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

    def __init__(self, engine, parent=None, **kw):
        super().__init__(
            engine, "ir", "IR LOADER", [],
            [KnobSpec("mix", "mix", 0, 100, 100, "%", 0, 1.0, 0.01),
             KnobSpec("gain_db", "trim", -24, 24, 0, " dB", 1)],
            parent=parent, **kw)
        self.file_label = QLabel("no impulse loaded")
        self.file_label.setObjectName("Hint")
        self.file_label.setWordWrap(True)

        row = QHBoxLayout()
        row.setSpacing(6)
        load = QPushButton("Load .wav")
        load.clicked.connect(self._pick)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        row.addWidget(load)
        row.addWidget(clear)
        row.addWidget(self.file_label, 1)

        lay = self.layout()
        at = 2 if kw.get("description") else 1
        lay.insertLayout(at, row)

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
        title="NOISE GATE", short="GATE",
        desc="Mutes the hum and hiss between notes.", selectors=[], knobs=[
            KnobSpec("threshold_db", "thresh", -80, -20, -55, " dB", 0,
                     tip="Below this level the signal is muted. Raise it until "
                         "hum disappears, then back off slightly."),
            KnobSpec("attack_ms", "open", 0.5, 20, 2, " ms", 1),
            KnobSpec("release_ms", "close", 20, 600, 120, " ms", 0),
        ]),
    "comp": dict(
        title="COMPRESSOR", short="COMP",
        desc="Evens out loud and soft picking and adds sustain.", selectors=[], knobs=[
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
        title="DRIVE", short="DRIVE",
        desc="Overdrive pedal in front of the amp: push it for more grit.", selectors=[SelectSpec("model", list(DRIVE_MODELS))],
        knobs=[
            KnobSpec("drive", "drive", 0, 10, 3.5, "", 1, mul=TEN),
            KnobSpec("tone", "tone", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("level", "level", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("mix", "mix", 0, 100, 100, "%", 0, mul=PCT),
        ]),
    "amp": dict(
        title="AMPLIFIER", short="AMP",
        desc="The heart of the tone. Gain sets how dirty it is, the EQ shapes it.", selectors=[SelectSpec("model", list(AMP_MODELS))],
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
        title="CABINET", short="CAB",
        desc="The speaker box and the microphone in front of it.",
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
        title="MODULATION", short="MOD",
        desc="Chorus, flanger, phaser, tremolo or vibrato: movement and shimmer.", selectors=[SelectSpec("mode", Modulation.MODES)],
        knobs=[
            KnobSpec("rate", "rate", 0, 10, 6, "", 1, mul=TEN),
            KnobSpec("depth", "depth", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("feedback", "regen", 0, 10, 3, "", 1, mul=TEN),
            KnobSpec("stereo_spread", "width", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("mix", "mix", 0, 100, 50, "%", 0, mul=PCT),
        ]),
    "delay": dict(
        title="DELAY", short="DELAY",
        desc="Echoes. Time is the gap between repeats, repeats is how many.", selectors=[SelectSpec("mode", Delay.MODES)],
        knobs=[
            KnobSpec("time_ms", "time", 20, 2000, 380, " ms", 0, curve=2.0),
            KnobSpec("feedback", "repeats", 0, 95, 35, "%", 0, mul=PCT),
            KnobSpec("tone", "tone", 0, 10, 5, "", 1, mul=TEN),
            KnobSpec("modulation", "wow", 0, 10, 1.5, "", 1, mul=TEN,
                     tip="Tape-style pitch drift. Only active in Tape mode."),
            KnobSpec("mix", "mix", 0, 100, 30, "%", 0, mul=PCT),
        ]),
    "reverb": dict(
        title="REVERB", short="REVERB",
        desc="The sound of a room around you, from a small room to a cathedral.", selectors=[SelectSpec("mode", Reverb.MODES)],
        knobs=[
            KnobSpec("size", "size", 0, 10, 6, "", 1, mul=TEN),
            KnobSpec("damping", "damp", 0, 10, 4.5, "", 1, mul=TEN),
            KnobSpec("predelay_ms", "predelay", 0, 150, 15, " ms", 0),
            KnobSpec("width", "width", 0, 10, 8, "", 1, mul=TEN),
            KnobSpec("mix", "mix", 0, 100, 25, "%", 0, mul=PCT),
        ]),
}


IR_DESC = ("Loads a real cabinet recording (an impulse response .wav) in "
           "place of, or blended with, the built-in cabinet.")

ORDER = ("gate", "comp", "drive", "amp", "cab", "ir", "mod", "delay", "reverb")


def card_title(key: str) -> tuple[str, str]:
    """(short tile name, full title)"""
    if key == "ir":
        return "IR", "IR LOADER"
    return CARDS[key]["short"], CARDS[key]["title"]


def build_card(engine, key: str, parent=None, **kw) -> PedalCard:
    if key == "ir":
        card = IRCard(engine, parent, description=IR_DESC, **kw)
        card._mul = {"mix": PCT, "gain_db": 1.0}
        return card
    spec = CARDS[key]
    card = PedalCard(engine, key, spec["title"], spec["selectors"], spec["knobs"],
                     spec.get("always_on", False), parent,
                     description=spec.get("desc", ""), **kw)
    card._mul = {k.attr: k.mul for k in spec["knobs"]}
    return card
