"""Custom controls: knob, LED toggle, meters, tuner face, loop progress.

All hand-painted. A stock slider would work and would feel wrong -- the whole
point of a pedalboard UI is that your hand already knows where things are.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme

START_ANGLE = 225.0     # degrees, measured counter-clockwise from 3 o'clock
SWEEP = -270.0


def _c(s: str, a: int = 255) -> QColor:
    col = QColor(s)
    col.setAlpha(a)
    return col


# ==========================================================================
# Knob
# ==========================================================================

class Knob(QWidget):
    """Continuous control. Drag vertically, shift for fine, double-click to
    reset, wheel to nudge. Value text appears while you touch it and fades to
    the label the rest of the time, so a board full of knobs stays readable."""

    valueChanged = Signal(float)

    def __init__(self, label: str, minimum: float, maximum: float, default: float,
                 unit: str = "", decimals: int = 1, accent: str = theme.AMBER,
                 curve: float = 1.0, size: int = 54, parent=None):
        super().__init__(parent)
        self.label = label
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.default = float(default)
        self.unit = unit
        self.decimals = decimals
        self.accent = accent
        self.curve = curve          # >1 gives finer control at the low end
        self.knob_size = size
        self.bipolar = minimum < 0 < maximum and abs(minimum + maximum) < 1e-6

        self._value = float(default)
        self._drag_y = None
        self._drag_norm = 0.0
        self._hover = False
        self._active = True

        self.setFixedSize(size + 14, size + 30)
        self.setCursor(Qt.SizeVerCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # -- value mapping -----------------------------------------------------

    def _to_norm(self, v: float) -> float:
        span = self.maximum - self.minimum
        if span <= 0:
            return 0.0
        t = (v - self.minimum) / span
        t = min(max(t, 0.0), 1.0)
        return t ** (1.0 / self.curve) if self.curve != 1.0 else t

    def _from_norm(self, t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        if self.curve != 1.0:
            t = t ** self.curve
        return self.minimum + t * (self.maximum - self.minimum)

    def value(self) -> float:
        return self._value

    def setValue(self, v: float, emit: bool = True) -> None:
        v = min(max(float(v), self.minimum), self.maximum)
        if abs(v - self._value) < 1e-12:
            return
        self._value = v
        self.update()
        if emit:
            self.valueChanged.emit(v)

    def setActive(self, on: bool) -> None:
        if on != self._active:
            self._active = on
            self.update()

    def text(self) -> str:
        v = self._value
        if self.decimals == 0:
            s = f"{v:.0f}"
        else:
            s = f"{v:.{self.decimals}f}"
        return f"{s}{self.unit}"

    # -- interaction -------------------------------------------------------

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_y = e.position().y()
            self._drag_norm = self._to_norm(self._value)
            self.setCursor(Qt.BlankCursor)
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_y is None:
            return
        dy = self._drag_y - e.position().y()
        span = 260.0
        if e.modifiers() & Qt.ShiftModifier:
            span = 1400.0       # fine
        elif e.modifiers() & Qt.ControlModifier:
            span = 110.0        # coarse
        t = self._drag_norm + dy / span
        self.setValue(self._from_norm(t))

    def mouseReleaseEvent(self, e):
        self._drag_y = None
        self.setCursor(Qt.SizeVerCursor)
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.setValue(self.default)

    def wheelEvent(self, e):
        step = 0.02 if not (e.modifiers() & Qt.ShiftModifier) else 0.004
        d = e.angleDelta().y() / 120.0
        self.setValue(self._from_norm(self._to_norm(self._value) + d * step))
        e.accept()

    def keyPressEvent(self, e):
        step = 0.01 if not (e.modifiers() & Qt.ShiftModifier) else 0.002
        if e.key() in (Qt.Key_Up, Qt.Key_Right):
            self.setValue(self._from_norm(self._to_norm(self._value) + step))
        elif e.key() in (Qt.Key_Down, Qt.Key_Left):
            self.setValue(self._from_norm(self._to_norm(self._value) - step))
        else:
            super().keyPressEvent(e)

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    # -- painting ----------------------------------------------------------

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = self.width()
        s = self.knob_size
        cx, cy = w / 2.0, s / 2.0 + 6
        r = s / 2.0
        track_r = r + 4

        acc = _c(self.accent) if self._active else _c(theme.TEXT_FAINT)
        t = self._to_norm(self._value)

        arc_rect = QRectF(cx - track_r, cy - track_r, track_r * 2, track_r * 2)

        # Track
        pen = QPen(_c(theme.STROKE), 3.0, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(arc_rect, int(START_ANGLE * 16), int(SWEEP * 16))

        # Value arc -- from centre outwards when the control is bipolar
        if self._active:
            pen = QPen(acc, 3.0, Qt.SolidLine, Qt.RoundCap)
            p.setPen(pen)
            if self.bipolar:
                mid = START_ANGLE + SWEEP * 0.5
                p.drawArc(arc_rect, int(mid * 16), int(SWEEP * (t - 0.5) * 16))
            else:
                p.drawArc(arc_rect, int(START_ANGLE * 16), int(SWEEP * t * 16))

        # Body
        body = QRectF(cx - r, cy - r, r * 2, r * 2)
        g = QRadialGradient(QPointF(cx - r * 0.35, cy - r * 0.45), r * 1.7)
        g.setColorAt(0.0, _c("#3b4250"))
        g.setColorAt(0.55, _c("#262b35"))
        g.setColorAt(1.0, _c("#171a20"))
        p.setBrush(QBrush(g))
        p.setPen(QPen(_c("#0d0f13"), 1.2))
        p.drawEllipse(body)

        inner = body.adjusted(r * 0.24, r * 0.24, -r * 0.24, -r * 0.24)
        g2 = QLinearGradient(inner.topLeft(), inner.bottomRight())
        g2.setColorAt(0.0, _c("#2e3440"))
        g2.setColorAt(1.0, _c("#1d2129"))
        p.setBrush(QBrush(g2))
        p.setPen(QPen(_c("#12151a"), 1.0))
        p.drawEllipse(inner)

        # Pointer
        ang = math.radians(START_ANGLE + SWEEP * t)
        x0 = cx + math.cos(ang) * (r * 0.30)
        y0 = cy - math.sin(ang) * (r * 0.30)
        x1 = cx + math.cos(ang) * (r * 0.80)
        y1 = cy - math.sin(ang) * (r * 0.80)
        p.setPen(QPen(acc if self._active else _c(theme.TEXT_FAINT), 2.6,
                      Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        if self._active and (self._hover or self._drag_y is not None):
            glow = QRadialGradient(QPointF(cx, cy), track_r * 1.5)
            glow.setColorAt(0.0, _c(self.accent, 0))
            glow.setColorAt(0.75, _c(self.accent, 26))
            glow.setColorAt(1.0, _c(self.accent, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QRectF(cx - track_r * 1.5, cy - track_r * 1.5,
                                 track_r * 3, track_r * 3))

        # Caption: the value while you are touching it, the name otherwise.
        showing_value = self._hover or self._drag_y is not None
        f = QFont(self.font())
        f.setPointSizeF(7.6)
        f.setBold(showing_value)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 0.3 if showing_value else 0.8)
        p.setFont(f)
        p.setPen(QPen(acc if showing_value and self._active
                      else _c(theme.TEXT_DIM if self._active else theme.TEXT_FAINT)))
        label = self.text() if showing_value else self.label.upper()
        p.drawText(QRectF(0, s + 8, w, 16), Qt.AlignHCenter | Qt.AlignTop, label)
        p.end()


# ==========================================================================
# Power LED toggle
# ==========================================================================

class PowerLED(QWidget):
    toggled = Signal(bool)

    def __init__(self, accent: str = theme.AMBER, size: int = 20, parent=None):
        super().__init__(parent)
        self.accent = accent
        self._on = False
        self._hover = False
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)

    def isChecked(self) -> bool:
        return self._on

    def setChecked(self, on: bool, emit: bool = True) -> None:
        on = bool(on)
        if on == self._on:
            return
        self._on = on
        self.update()
        if emit:
            self.toggled.emit(on)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.setChecked(not self._on)

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(1.5, 1.5, self.width() - 3, self.height() - 3)
        if self._on:
            g = QRadialGradient(r.center(), r.width())
            g.setColorAt(0.0, _c(self.accent))
            g.setColorAt(0.6, _c(self.accent, 210))
            g.setColorAt(1.0, _c(self.accent, 70))
            p.setBrush(QBrush(g))
            p.setPen(QPen(_c(self.accent), 1.4))
        else:
            p.setBrush(QBrush(_c("#20242c")))
            p.setPen(QPen(_c(theme.STROKE_HI if self._hover else theme.STROKE), 1.4))
        p.drawEllipse(r)
        if self._on:
            p.setBrush(QBrush(_c("#ffffff", 150)))
            p.setPen(Qt.NoPen)
            hl = QRectF(r.left() + r.width() * 0.26, r.top() + r.height() * 0.20,
                        r.width() * 0.26, r.height() * 0.22)
            p.drawEllipse(hl)
        p.end()


# ==========================================================================
# Meters
# ==========================================================================

class LevelMeter(QWidget):
    """Peak meter with a slow-falling hold marker.

    dBFS, not linear -- a linear meter spends 90% of its length on the top
    6 dB and tells you nothing about where you actually are.
    """

    def __init__(self, channels: int = 2, horizontal: bool = True, parent=None):
        super().__init__(parent)
        self.horizontal = horizontal
        self.labels: list[str] = []
        self.highlight = -1     # index drawn in the accent colour
        self.channels = 0
        self.setChannelCount(channels)

    def setChannelCount(self, channels: int) -> None:
        channels = max(1, int(channels))
        if channels == self.channels:
            return
        self.channels = channels
        self._levels = [0.0] * channels
        self._peaks = [0.0] * channels
        self._peak_age = [0] * channels
        if self.horizontal:
            self.setMinimumSize(90, 8 * channels + 2)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.setFixedHeight(8 * channels + 2)
        else:
            self.setFixedWidth(12 * channels + 2)
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.updateGeometry()

    def setLevels(self, *vals: float) -> None:
        if len(vals) == 1 and hasattr(vals[0], "__len__"):
            vals = tuple(vals[0])
        for i, v in enumerate(vals[: self.channels]):
            v = max(0.0, min(float(v), 1.5))
            # Fall at roughly 20 dB/s so the eye can follow it.
            self._levels[i] = v if v > self._levels[i] else self._levels[i] * 0.80
            if v >= self._peaks[i]:
                self._peaks[i] = v
                self._peak_age[i] = 0
            else:
                self._peak_age[i] += 1
                if self._peak_age[i] > 22:
                    self._peaks[i] *= 0.94
        self.update()

    @staticmethod
    def _db_pos(v: float) -> float:
        """Map amplitude to 0..1 across a -60..0 dBFS scale."""
        if v <= 1e-5:
            return 0.0
        db = 20.0 * math.log10(v)
        return min(max((db + 60.0) / 60.0, 0.0), 1.0)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        n = self.channels
        pad = 14 if (self.labels and self.horizontal) else 0
        vpad = 14 if (self.labels and not self.horizontal) else 0
        for i in range(n):
            if self.horizontal:
                bar = QRectF(pad, i * 8 + 1, w - pad, 6)
            else:
                bar = QRectF(i * 12 + 2, 0, 8, h - vpad)
            p.fillRect(bar, _c("#14171d"))

            if vpad and i < len(self.labels):
                f = QFont(self.font())
                f.setPointSizeF(7.0)
                f.setBold(i == self.highlight)
                p.setFont(f)
                p.setPen(QPen(_c(theme.AMBER if i == self.highlight
                                 else theme.TEXT_FAINT)))
                p.drawText(QRectF(bar.left() - 3, h - vpad + 1, bar.width() + 6, 12),
                           Qt.AlignHCenter | Qt.AlignVCenter, self.labels[i])

            if pad and i < len(self.labels):
                f = QFont(self.font())
                f.setPointSizeF(7.0)
                f.setBold(i == self.highlight)
                p.setFont(f)
                p.setPen(QPen(_c(theme.AMBER if i == self.highlight
                                 else theme.TEXT_FAINT)))
                p.drawText(QRectF(0, bar.top() - 2, pad - 3, 11),
                           Qt.AlignRight | Qt.AlignVCenter, self.labels[i])

            frac = self._db_pos(self._levels[i])
            if frac > 0:
                if self.horizontal:
                    fill = QRectF(bar.left(), bar.top(), bar.width() * frac, bar.height())
                    grad = QLinearGradient(bar.left(), 0, bar.right(), 0)
                else:
                    fill = QRectF(bar.left(), bar.bottom() - bar.height() * frac,
                                  bar.width(), bar.height() * frac)
                    grad = QLinearGradient(0, bar.bottom(), 0, bar.top())
                grad.setColorAt(0.0, _c(theme.GREEN))
                grad.setColorAt(0.72, _c(theme.GREEN))
                grad.setColorAt(0.87, _c(theme.YELLOW))
                grad.setColorAt(1.0, _c(theme.RED))
                p.fillRect(fill, QBrush(grad))

            pk = self._db_pos(self._peaks[i])
            if pk > 0.01:
                col = _c(theme.RED) if self._peaks[i] >= 0.995 else _c(theme.TEXT)
                if self.horizontal:
                    x = bar.left() + bar.width() * pk
                    p.fillRect(QRectF(x - 1, bar.top(), 2, bar.height()), col)
                else:
                    y = bar.bottom() - bar.height() * pk
                    p.fillRect(QRectF(bar.left(), y - 1, bar.width(), 2), col)
        p.end()


class LoopProgress(QWidget):
    """Loop position ring/bar with a record-state colour."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fraction = 0.0
        self.state = "empty"
        self.length = 0.0
        self.setMinimumWidth(150)
        self.setFixedHeight(26)

    def update_state(self, state: str, fraction: float, length: float) -> None:
        self.state = state
        self.fraction = fraction
        self.length = length
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        path = QPainterPath()
        path.addRoundedRect(r, 6, 6)
        p.fillPath(path, _c("#14171d"))

        col = {"recording": theme.RED, "overdubbing": theme.AMBER,
               "playing": theme.GREEN, "stopped": theme.TEXT_DIM}.get(
            self.state, theme.TEXT_FAINT)

        if self.state == "recording":
            p.setClipPath(path)
            p.fillRect(r, _c(theme.RED, 40))
            p.setClipping(False)
        elif self.length > 0:
            p.setClipPath(path)
            fill = QRectF(r.left(), r.top(), r.width() * self.fraction, r.height())
            p.fillRect(fill, _c(col, 55))
            x = r.left() + r.width() * self.fraction
            p.fillRect(QRectF(x - 1, r.top(), 2, r.height()), _c(col))
            p.setClipping(False)

        p.setPen(QPen(_c(theme.STROKE), 1))
        p.drawPath(path)

        f = QFont(self.font())
        f.setPointSizeF(8.0)
        f.setBold(True)
        f.setLetterSpacing(QFont.PercentageSpacing, 108)
        p.setFont(f)
        p.setPen(QPen(_c(col)))
        if self.state == "empty":
            txt = "EMPTY"
        elif self.state == "recording":
            txt = "REC"
        else:
            txt = f"{self.state.upper()}  {self.length:.1f}s"
        p.drawText(r, Qt.AlignCenter, txt)
        p.end()


# ==========================================================================
# Tuner face
# ==========================================================================

class TunerDisplay(QWidget):
    """Note name, cents needle, and a lit segment strip.

    Green within +/-3 cents, which is roughly the point where a guitar's own
    intonation stops being the limiting factor.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.note = "--"
        self.octave = 0
        self.cents = 0.0
        self.freq = 0.0
        self.active = False
        self._needle = 0.0        # smoothed, so the needle does not twitch
        self.setMinimumSize(230, 74)

    def update_pitch(self, note: str, octave: int, cents: float,
                     freq: float, active: bool) -> None:
        self.note, self.octave, self.freq, self.active = note, octave, freq, active
        if active:
            self.cents = cents
            self._needle += (cents - self._needle) * 0.35
        else:
            self._needle *= 0.85
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()

        in_tune = self.active and abs(self.cents) <= 3.0
        col = _c(theme.GREEN if in_tune else
                 (theme.AMBER if self.active else theme.TEXT_FAINT))

        # Segment strip: 21 segments spanning +/-50 cents.
        segs = 21
        strip_h = 8 if h < 140 else 14
        strip_y = h - strip_h - 14
        seg_w = (w - 12) / segs
        centre = segs // 2
        lit = centre
        if self.active:
            lit = int(round(centre + max(-1.0, min(1.0, self._needle / 50.0)) * centre))
            lit = max(0, min(segs - 1, lit))
        for i in range(segs):
            x = 6 + i * seg_w
            rect = QRectF(x + 1, strip_y, seg_w - 2, strip_h)
            is_centre = i == centre
            if self.active and i == lit:
                c = col
            elif is_centre:
                c = _c(theme.GREEN, 90)
            else:
                c = _c("#20242c")
            if self.active and ((centre < i <= lit) or (lit <= i < centre)):
                c = _c(theme.AMBER, 110)
            p.fillRect(rect, c)

        # Note name -- grows with the widget so the big tuner page reads
        # from across the room.
        note_h = max(40.0, (h - 30) * 0.68)
        f = QFont(self.font())
        f.setPixelSize(int(max(30, note_h * 0.62)))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QPen(col))
        name = self.note if self.active else "--"
        p.drawText(QRectF(0, 0, w, note_h), Qt.AlignCenter, name)

        f.setPixelSize(int(max(11, note_h * 0.12)))
        f.setBold(False)
        p.setFont(f)
        p.setPen(QPen(_c(theme.TEXT_DIM)))
        if self.active:
            sub = f"{self.octave}   {self.freq:6.1f} Hz   {self.cents:+.0f}¢"
        else:
            sub = "play a string"
        p.drawText(QRectF(0, note_h, w, max(16, strip_y - note_h - 4)),
                   Qt.AlignCenter, sub)
        p.end()


class LoadIndicator(QWidget):
    """DSP load as a fraction of the block deadline. Above ~70% you are one
    background task away from a dropout, so it goes amber then red."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.load = 0.0
        self.peak = 0.0
        self.setFixedSize(76, 18)

    def set_load(self, load: float, peak: float) -> None:
        self.load, self.peak = load, peak
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        path = QPainterPath()
        path.addRoundedRect(r, 4, 4)
        p.fillPath(path, _c("#14171d"))
        frac = min(self.load, 1.0)
        col = theme.GREEN if frac < 0.5 else (theme.YELLOW if frac < 0.75 else theme.RED)
        p.setClipPath(path)
        p.fillRect(QRectF(r.left(), r.top(), r.width() * frac, r.height()), _c(col, 90))
        pk = min(self.peak, 1.0)
        p.fillRect(QRectF(r.left() + r.width() * pk - 1, r.top(), 2, r.height()),
                   _c(col, 200))
        p.setClipping(False)
        p.setPen(QPen(_c(theme.STROKE), 1))
        p.drawPath(path)
        f = QFont(self.font())
        f.setPointSizeF(7.5)
        p.setFont(f)
        p.setPen(QPen(_c(theme.TEXT_DIM)))
        p.drawText(r, Qt.AlignCenter, f"DSP {self.load*100:.0f}%")
        p.end()


# ==========================================================================
# Signal-chain tile
# ==========================================================================

class ChainTile(QWidget):
    """One stop in the signal chain. Click the body to edit the module, click
    the dot to switch it on or off. Off modules go grey so the path your
    signal actually takes reads at a glance."""

    selected = Signal(str)
    powerToggled = Signal(str, bool)

    def __init__(self, key: str, title: str, accent: str,
                 always_on: bool = False, parent=None):
        super().__init__(parent)
        self.key = key
        self.title = title
        self.accent = accent
        self.always_on = always_on
        self.subtitle = ""
        self._on = True
        self._selected = False
        self._hover = False
        self.setFixedHeight(66)
        self.setMinimumWidth(92)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)

    def _led_rect(self) -> QRectF:
        return QRectF(self.width() - 24, 9, 14, 14)

    def set_state(self, on: bool, subtitle: str) -> None:
        on = True if self.always_on else bool(on)
        if on != self._on or subtitle != self.subtitle:
            self._on, self.subtitle = on, subtitle
            self.update()

    def set_selected(self, sel: bool) -> None:
        if sel != self._selected:
            self._selected = sel
            self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        if not self.always_on and self._led_rect().adjusted(-6, -6, 6, 6).contains(
                e.position()):
            self._on = not self._on
            self.update()
            self.powerToggled.emit(self.key, self._on)
        else:
            self.selected.emit(self.key)

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        path = QPainterPath()
        path.addRoundedRect(r, 8, 8)
        bg = "#262b36" if self._selected else ("#20242d" if self._hover else "#1a1d24")
        p.fillPath(path, _c(bg))
        acc = _c(self.accent) if self._on else _c(theme.TEXT_FAINT)

        # accent bar along the top
        p.save()
        p.setClipPath(path)
        p.fillRect(QRectF(r.left(), r.top(), r.width(), 4),
                   _c(self.accent, 255 if self._on else 60))
        p.restore()

        border = _c(self.accent) if self._selected else _c(
            theme.STROKE_HI if self._hover else theme.STROKE)
        p.setPen(QPen(border, 1.6 if self._selected else 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

        # power dot
        if not self.always_on:
            lr = self._led_rect()
            if self._on:
                p.setBrush(QBrush(acc))
                p.setPen(QPen(_c(self.accent), 1.2))
            else:
                p.setBrush(QBrush(_c("#14171d")))
                p.setPen(QPen(_c(theme.STROKE_HI), 1.2))
            p.drawEllipse(lr)

        f = QFont(self.font())
        f.setPixelSize(11)
        f.setBold(True)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
        p.setFont(f)
        p.setPen(QPen(_c(theme.TEXT) if self._on else _c(theme.TEXT_FAINT)))
        p.drawText(QRectF(11, 12, r.width() - 40, 16), Qt.AlignLeft | Qt.AlignVCenter,
                   self.title)

        f.setBold(False)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 0.0)
        f.setPixelSize(11)
        p.setFont(f)
        p.setPen(QPen(acc if self._on else _c(theme.TEXT_FAINT)))
        sub = self.subtitle if self._on else "off"
        p.drawText(QRectF(11, 34, r.width() - 18, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   p.fontMetrics().elidedText(sub, Qt.ElideRight, int(r.width() - 18)))
        p.end()
