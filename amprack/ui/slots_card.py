"""Effect slot racks: four slots, any effect in any slot.

The controls are generated from each effect's parameter list
(``amprack/dsp/fx``), so a new effect or parameter shows up here with no UI
code. Knobs write straight to the effect's attributes, like every other card;
changing what sits in a slot is a structural change and goes through
``Engine.post`` so it lands between audio blocks.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..dsp.fx import FX_TYPES
from . import theme
from .widgets import Knob, PowerLED

EMPTY = "Empty"

RACK_INFO = {
    "pre": ("PEDALS", "PEDALS",
            "Four pedal slots in front of the amp: wahs, auto-wah, fuzz, "
            "octave, compressor, EQ. Pick an effect for each slot."),
    "loop": ("FX LOOP", "LOOP",
             "Four slots after the amp and cabinet: modulation, delay, "
             "reverb, EQ, in stereo."),
    "rack": ("STUDIO RACK", "RACK",
             "Four studio-rack slots at the very end of the chain."),
}

# Category order for the effect menu.
_CATS = ("Filter", "Dirt", "Pitch", "Dynamics", "EQ", "Modulation", "Time",
         "Utility")


def _menu() -> list[tuple[str, str]]:
    """[(label shown, type key)], grouped by category."""
    def rank(item):
        cls = item[1]
        cat = getattr(cls, "CATEGORY", "")
        return (_CATS.index(cat) if cat in _CATS else len(_CATS), cls.LABEL)
    return [(cls.LABEL, key) for key, cls in sorted(FX_TYPES.items(), key=rank)]


def _knob_spec(p):
    """(label, lo, hi, default, unit, decimals, curve, mul) for one Param.
    0-1 controls show 0-10 like the rest of Tanin's panels."""
    label = p.label.lower()
    if p.lo == 0.0 and p.hi == 1.0 and not p.unit:
        return label, 0.0, 10.0, p.default * 10.0, "", 1, 1.0, 0.1
    unit = f" {p.unit}" if p.unit else ""
    span = p.hi - p.lo
    dec = 0 if span >= 50 else (1 if span >= 2 else 2)
    curve = 1.0
    if p.kind == "log":
        ratio = p.hi / max(p.lo, 1e-9)
        curve = 2.5 if ratio > 100 else (2.0 if ratio > 10 else 1.5)
    return label, p.lo, p.hi, p.default, unit, dec, curve, 1.0


class SlotRackCard(QFrame):
    """Card for one FxSlots rack (``pre``, ``loop`` or ``rack``)."""

    def __init__(self, engine, key: str, parent=None, knob_size: int = 60,
                 per_row: int = 8, **_kw):
        super().__init__(parent)
        self.setObjectName("Card")
        self.engine = engine
        self.key = key
        self.block = engine.chain[key]
        self.accent = theme.accent(key)
        self.always_on = False
        self.knob_size = knob_size
        self.per_row = per_row
        self._sel = 0
        self._knobs: dict[str, tuple[Knob, float]] = {}
        self._choices: dict[str, QComboBox] = {}
        self._menu = _menu()
        title, _short, desc = RACK_INFO[key]

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 9, 12, 11)
        root.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.led = PowerLED(self.accent, 22)
        self.led.setChecked(bool(self.block.enabled), emit=False)
        self.led.toggled.connect(self._on_power)
        self.led.setToolTip("Enable / bypass all four slots")
        head.addWidget(self.led)
        name = QLabel(title.title())
        name.setObjectName("CardTitle")
        head.addWidget(name)
        head.addStretch(1)
        root.addLayout(head)
        d = QLabel(desc)
        d.setObjectName("Hint")
        d.setWordWrap(True)
        root.addWidget(d)

        # Slot buttons: 1-4, each showing what is in it.
        row = QHBoxLayout()
        row.setSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._slot_btns: list[QPushButton] = []
        for i in range(len(self.block.slots)):
            b = QPushButton()
            b.setCheckable(True)
            b.setMinimumHeight(34)
            b.setToolTip("Select this slot to choose and edit its effect")
            self._group.addButton(b, i)
            self._slot_btns.append(b)
            row.addWidget(b, 1)
        self._group.idClicked.connect(self._select)
        root.addLayout(row)

        # Slot header: on/off, effect type, model.
        bar = QHBoxLayout()
        bar.setSpacing(10)
        self.slot_led = PowerLED(self.accent, 20)
        self.slot_led.toggled.connect(self._on_slot_power)
        self.slot_led.setToolTip("Enable / bypass this slot")
        bar.addWidget(self.slot_led)
        self.type_combo = QComboBox()
        self.type_combo.addItem(EMPTY, "")
        for label, t in self._menu:
            self.type_combo.addItem(label, t)
        self.type_combo.setMinimumWidth(170)
        self.type_combo.currentIndexChanged.connect(self._on_type)
        bar.addWidget(self.type_combo)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.model_combo.currentTextChanged.connect(self._on_model)
        bar.addWidget(self.model_combo)
        bar.addStretch(1)
        root.addLayout(bar)

        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 2, 0, 0)
        root.addWidget(self.body)
        root.addStretch(1)

        self._group.button(0).setChecked(True)
        self.refresh_from_block()

    # -- helpers -----------------------------------------------------------

    def _fx(self):
        return self.block.effect(self._sel)

    def _slot_text(self, i: int) -> str:
        s = self.block.slots[i]
        if s.fx is None:
            return f"{i + 1}  ·  empty"
        return f"{i + 1}  ·  {FX_TYPES[s.type].LABEL}"

    def _refresh_slot_buttons(self) -> None:
        for i, b in enumerate(self._slot_btns):
            b.setText(self._slot_text(i))

    def summary(self) -> str:
        """Short text for the chain tile."""
        names = [FX_TYPES[s.type].LABEL.split()[0] for s in self.block.slots
                 if s.fx is not None and s.enabled]
        if not names:
            return "empty"
        return names[0] if len(names) == 1 else f"{names[0]} +{len(names) - 1}"

    # -- events ------------------------------------------------------------

    def _on_power(self, on: bool) -> None:
        self.block.enabled = on
        self._sync_active()

    def _on_slot_power(self, on: bool) -> None:
        self.block.slots[self._sel].enabled = on
        self._sync_active()

    def _select(self, i: int) -> None:
        self._sel = i
        self._build_editor()

    def _on_type(self, _idx: int) -> None:
        t = self.type_combo.currentData() or ""
        s = self.block.slots[self._sel]
        if t == s.type:
            return
        i = self._sel
        self.block.preload(i, t)          # build here, never on the audio thread
        self.engine.post(lambda: self.block.set_type(i, t))
        # The swap lands on the next audio block; redraw after it.
        QTimer.singleShot(40, self._build_editor)

    def _on_model(self, m: str) -> None:
        fx = self._fx()
        if fx is not None and m and m in fx.MODELS:
            fx.model = m

    # -- editor ------------------------------------------------------------

    def _clear_body(self) -> None:
        while self.body_lay.count():
            item = self.body_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                lay = item.layout()
                while lay.count():
                    sub = lay.takeAt(0)
                    if sub.widget() is not None:
                        sub.widget().deleteLater()
        self._knobs.clear()
        self._choices.clear()

    def _build_editor(self) -> None:
        self._refresh_slot_buttons()
        self._clear_body()
        s = self.block.slots[self._sel]
        fx = s.fx
        self.type_combo.blockSignals(True)
        self.type_combo.setCurrentIndex(max(0, self.type_combo.findData(s.type)))
        self.type_combo.blockSignals(False)
        self.slot_led.setChecked(bool(s.enabled), emit=False)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if fx is None:
            self.model_combo.setVisible(False)
            self.slot_led.setVisible(False)
            hint = QLabel("Empty slot. Pick an effect above.")
            hint.setObjectName("Hint")
            self.body_lay.addWidget(hint)
        else:
            self.slot_led.setVisible(True)
            self.model_combo.addItems(list(fx.MODELS))
            self.model_combo.setCurrentText(fx.model)
            self.model_combo.setVisible(len(fx.MODELS) > 1)
            self._build_controls(fx)
        self.model_combo.blockSignals(False)
        self._sync_active()

    def _build_controls(self, fx) -> None:
        sel_row = QHBoxLayout()
        sel_row.setSpacing(10)
        grid = QGridLayout()
        grid.setHorizontalSpacing(max(2, self.knob_size // 4))
        grid.setVerticalSpacing(2)
        n = 0
        for p in fx.PARAMS:
            if p.kind == "choice":
                lab = QLabel(p.label)
                lab.setObjectName("Section")
                combo = QComboBox()
                combo.addItems(list(p.choices))
                combo.setCurrentText(str(getattr(fx, p.name, p.default)))
                combo.currentTextChanged.connect(
                    lambda v, f=fx, a=p.name: setattr(f, a, v))
                sel_row.addWidget(lab)
                sel_row.addWidget(combo)
                self._choices[p.name] = combo
            elif p.kind == "bool":
                cb = QCheckBox(p.label)
                cb.setChecked(bool(getattr(fx, p.name, p.default)))
                cb.toggled.connect(lambda v, f=fx, a=p.name: setattr(f, a, bool(v)))
                sel_row.addWidget(cb)
            else:
                label, lo, hi, dflt, unit, dec, curve, mul = _knob_spec(p)
                k = Knob(label, lo, hi, dflt, unit, dec, self.accent, curve,
                         self.knob_size)
                k.setValue(float(getattr(fx, p.name, p.default)) / mul, emit=False)
                k.valueChanged.connect(
                    lambda v, f=fx, a=p.name, m=mul: setattr(f, a, v * m))
                k.setToolTip(f"{p.label}: drag to set, shift for fine, "
                             "double-click to reset")
                self._knobs[p.name] = (k, mul)
                grid.addWidget(k, n // self.per_row, n % self.per_row,
                               Qt.AlignHCenter | Qt.AlignTop)
                n += 1
        if sel_row.count():
            sel_row.addStretch(1)
            self.body_lay.addLayout(sel_row)
        wrap = QHBoxLayout()
        wrap.addLayout(grid)
        wrap.addStretch(1)
        self.body_lay.addLayout(wrap)

    def _sync_active(self) -> None:
        on = bool(self.block.enabled)
        s = self.block.slots[self._sel]
        slot_on = on and s.enabled
        for k, _m in self._knobs.values():
            k.setActive(slot_on)
        for c in self._choices.values():
            c.setEnabled(slot_on)
        self.setProperty("active", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    # -- preset load -------------------------------------------------------

    def refresh_from_block(self) -> None:
        self.led.setChecked(bool(self.block.enabled), emit=False)
        self._build_editor()
