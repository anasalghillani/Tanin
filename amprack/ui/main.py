"""Main window."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import presets as presetlib
from ..engine import (
    BLOCK_SIZES,
    SAMPLE_RATES,
    Engine,
    broadcast_candidates,
    default_devices,
    find_device,
    resolve_device,
    hostapi_name,
    is_asio,
    list_devices,
    suggest_broadcast_device,
)
from ..tuner import detect_pitch, freq_to_note
from . import theme
from .pedals import build_card
from .widgets import Knob, LevelMeter, LoadIndicator, LoopProgress, PowerLED, TunerDisplay

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRESET_DIR = os.path.join(APP_DIR, "presets")
RECORD_DIR = os.path.join(APP_DIR, "recordings")
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")

BOARD_ROWS = (
    ("gate", "comp", "drive", "amp"),
    ("cab", "ir", "mod", "delay", "reverb"),
)


def _section(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Section")
    return lab


def _card(obj_name: str = "Bar") -> QFrame:
    f = QFrame()
    f.setObjectName(obj_name)
    return f


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tanin")
        self.resize(1360, 900)
        self.setMinimumSize(1080, 720)

        self.engine = Engine(RECORD_DIR)
        presetlib.ensure_factory(PRESET_DIR)
        self._load_settings()

        self.cards = {}
        self._tuner_on = False
        self._tick = 0

        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        lay.addWidget(self._build_header())
        self.audio_panel = self._build_audio_panel()
        self.audio_panel.setVisible(False)
        lay.addWidget(self.audio_panel)
        lay.addWidget(self._build_io_strip())
        lay.addWidget(self._build_board(), 1)
        lay.addWidget(self._build_transport())

        self._install_shortcuts()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(33)

        QTimer.singleShot(120, self._autostart)

    # ==================================================================
    # Header: identity, presets, status
    # ==================================================================

    def _build_header(self) -> QFrame:
        bar = _card()
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 8, 14, 8)
        h.setSpacing(10)

        title = QLabel("TANIN")
        f = QFont(title.font())
        f.setPointSizeF(13)
        f.setBold(True)
        f.setLetterSpacing(QFont.PercentageSpacing, 150)
        title.setFont(f)
        title.setStyleSheet(f"color:{theme.AMBER};")
        h.addWidget(title)

        h.addSpacing(14)
        h.addWidget(_section("PRESET"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(190)
        self.preset_combo.activated.connect(self._on_preset_selected)
        h.addWidget(self.preset_combo)

        save = QPushButton("Save")
        save.setToolTip("Overwrite the selected preset")
        save.clicked.connect(self._save_preset)
        save_as = QPushButton("Save As…")
        save_as.clicked.connect(self._save_preset_as)
        delete = QPushButton("Delete")
        delete.clicked.connect(self._delete_preset)
        for b in (save, save_as, delete):
            h.addWidget(b)
        self._refresh_presets()

        h.addStretch(1)

        self.status_device = QLabel("—")
        self.status_device.setObjectName("Hint")
        h.addWidget(self.status_device)

        self.status_latency = QLabel("—")
        self.status_latency.setObjectName("Hint")
        h.addWidget(self.status_latency)

        self.load_ind = LoadIndicator()
        self.load_ind.setToolTip("Share of each audio buffer spent in the DSP. "
                                 "Under 50% is comfortable.")
        h.addWidget(self.load_ind)

        self.status_xruns = QLabel("")
        self.status_xruns.setObjectName("Hint")
        h.addWidget(self.status_xruns)

        self.btn_audio = QPushButton("Audio Setup")
        self.btn_audio.setCheckable(True)
        self.btn_audio.toggled.connect(self.audio_panel_toggle)
        h.addWidget(self.btn_audio)

        return bar

    def audio_panel_toggle(self, on: bool) -> None:
        self.audio_panel.setVisible(on)

    # ==================================================================
    # Audio setup panel
    # ==================================================================

    def _build_audio_panel(self) -> QFrame:
        bar = _card()
        g = QGridLayout(bar)
        g.setContentsMargins(14, 10, 14, 12)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)

        self.in_combo = QComboBox()
        self.out_combo = QComboBox()
        self.in_combo.setMinimumWidth(330)
        self.out_combo.setMinimumWidth(330)
        self._inputs = list_devices("input")
        self._outputs = list_devices("output")
        for d in self._inputs:
            self.in_combo.addItem(f"[{d['api']}]  {d['name']}", d["index"])
        for d in self._outputs:
            self.out_combo.addItem(f"[{d['api']}]  {d['name']}", d["index"])

        self.ch_combo = QComboBox()
        self.sr_combo = QComboBox()
        for sr in SAMPLE_RATES:
            self.sr_combo.addItem(f"{sr} Hz", sr)
        self.bs_combo = QComboBox()
        for bs in BLOCK_SIZES:
            self.bs_combo.addItem(f"{bs}", bs)
        self.excl_check = QCheckBox("WASAPI exclusive mode")
        self.excl_check.setToolTip(
            "Takes sole control of the interface. This is where the low latency "
            "comes from on Windows without a vendor ASIO driver — but nothing "
            "else can use the device while Tanin has it.")
        self.excl_check.setChecked(self.engine.exclusive)

        self.in_combo.currentIndexChanged.connect(self._update_channel_combo)
        self.in_combo.currentIndexChanged.connect(self._sync_api_mode)
        self.in_combo.currentIndexChanged.connect(self._populate_broadcast_combo)
        self.ch_combo.currentIndexChanged.connect(self._apply_input_channel)

        self.bcast_combo = QComboBox()
        self.bcast_combo.setMinimumWidth(330)
        self._populate_broadcast_combo()
        self.bcast_combo.currentIndexChanged.connect(self._change_broadcast_device)

        self.btn_asio_panel = QPushButton("ASIO Control Panel…")
        self.btn_asio_panel.setToolTip(
            "The ASIO buffer size belongs to the driver, not to Tanin. "
            "Set it there and restart audio here.")
        self.btn_asio_panel.clicked.connect(self._open_asio_panel)

        g.addWidget(_section("INPUT"), 0, 0)
        g.addWidget(self.in_combo, 0, 1)
        g.addWidget(_section("RATE"), 0, 2)
        g.addWidget(self.sr_combo, 0, 3)

        g.addWidget(_section("OUTPUT"), 1, 0)
        g.addWidget(self.out_combo, 1, 1)
        g.addWidget(_section("BUFFER"), 1, 2)
        g.addWidget(self.bs_combo, 1, 3)
        g.addWidget(self.excl_check, 1, 4)
        g.addWidget(self.btn_asio_panel, 1, 5)

        self.btn_apply = QPushButton("Start Audio")
        self.btn_apply.setObjectName("Go")
        self.btn_apply.clicked.connect(self._restart_audio)
        g.addWidget(self.btn_apply, 0, 6)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop_audio)
        g.addWidget(self.btn_stop, 1, 6)

        self.audio_error = QLabel("")
        self.audio_error.setObjectName("Hint")
        self.audio_error.setWordWrap(True)

        self.bcast_status = QLabel("")
        self.bcast_status.setObjectName("Hint")
        self.bcast_status.setWordWrap(True)

        self.api_hint = QLabel("")
        self.api_hint.setObjectName("Hint")
        self.api_hint.setWordWrap(True)
        g.addWidget(self.api_hint, 2, 0, 1, 7)
        g.addWidget(self.audio_error, 3, 0, 1, 7)

        g.setColumnStretch(1, 1)
        self._apply_device_settings_to_ui()
        self._sync_api_mode()
        return bar

    def _sync_api_mode(self) -> None:
        """ASIO owns both directions and its own buffer size, so stop
        pretending the user can set those here."""
        dev = self.in_combo.currentData()
        asio = is_asio(dev)
        self.btn_asio_panel.setVisible(asio)
        self.excl_check.setVisible(not asio)
        self.out_combo.setEnabled(not asio)
        self.bs_combo.setEnabled(not asio)
        if asio:
            for i in range(self.out_combo.count()):
                if self.out_combo.itemData(i) == dev:
                    self.out_combo.setCurrentIndex(i)
                    break
            self.api_hint.setText(
                "ASIO: one driver handles input and output, and the buffer "
                "size comes from the driver's own control panel. This is the "
                "lowest-latency path on Windows — set the buffer to 128 or 256 "
                "samples there, then press Start Audio.")
        else:
            self.api_hint.setText(
                "WASAPI/WDM-KS: Tanin sets the buffer size. Exclusive mode "
                "gives lower latency but takes sole control of the device. "
                "If your interface has an ASIO driver, that will beat this.")

    def _populate_broadcast_combo(self) -> None:
        keep = self.bcast_combo.currentData()
        self.bcast_combo.blockSignals(True)
        self.bcast_combo.clear()
        self._bcast_cands = broadcast_candidates(self.engine.input_device)
        for d in self._bcast_cands:
            self.bcast_combo.addItem(f"[{d['api'].replace('Windows ','')}]  "
                                     f"{d['name']}", d["index"])
        if keep is not None:
            i = self.bcast_combo.findData(keep)
            if i >= 0:
                self.bcast_combo.setCurrentIndex(i)
        self.bcast_combo.blockSignals(False)

    def _test_broadcast(self) -> None:
        e = self.engine
        if not e.broadcast.running:
            self.bcast_status.setText("turn Stream Out on first")
            self.bcast_status.setStyleSheet(f"color:{theme.YELLOW};")
            return
        e.broadcast.test_tone(2.0)
        self.bcast_status.setText(
            "tone playing — you should not hear it, your friends should")
        self.bcast_status.setStyleSheet(f"color:{theme.accent('mod')};")

    def _broadcast_hint(self) -> str:
        idx = self.bcast_combo.currentData()
        for d in getattr(self, "_bcast_cands", []):
            if d["index"] == idx:
                return ("Streaming. In Discord, share the Tanin window "
                        "with sound — your mic keeps working as normal.")
        return ""

    def _toggle_broadcast(self, on: bool) -> None:
        e = self.engine
        e.broadcast.enabled = on
        e.broadcast_device = self.bcast_combo.currentData()
        if not on:
            e.broadcast.stop()
            self.bcast_status.setText("off")
            self.bcast_status.setStyleSheet(f"color:{theme.TEXT_FAINT};")
            return
        if not e.running:
            self.bcast_status.setText("starts with the audio engine")
            self.bcast_status.setStyleSheet(f"color:{theme.YELLOW};")
            return
        err = e.broadcast.start(e.broadcast_device, e.samplerate)
        if err:
            self.bcast_status.setText(f"failed: {err}")
            self.bcast_status.setStyleSheet(f"color:{theme.RED};")
        else:
            self.bcast_status.setText(self._broadcast_hint() or "streaming")
            self.bcast_status.setStyleSheet(f"color:{theme.GREEN};")

    def _change_broadcast_device(self) -> None:
        self.engine.broadcast_device = self.bcast_combo.currentData()
        if self.engine.broadcast.enabled and self.engine.running:
            self._toggle_broadcast(True)

    def _apply_input_channel(self) -> None:
        """Live, with no stream restart: the callback re-reads this int every
        block, so switching jacks while playing is free."""
        d = self.ch_combo.currentData()
        if d is not None:
            self.engine.input_channel = int(d)
            if hasattr(self, "in_meter"):
                self.in_meter.highlight = int(d)

    def _update_channel_combo(self) -> None:
        idx = self.in_combo.currentIndex()
        if idx < 0 or idx >= len(self._inputs):
            return
        n = self._inputs[idx]["channels"]
        keep = self.ch_combo.currentIndex()
        self.ch_combo.clear()
        for c in range(n):
            self.ch_combo.addItem(f"In {c + 1}", c)
        if 0 <= keep < n:
            self.ch_combo.setCurrentIndex(keep)

    def _apply_device_settings_to_ui(self) -> None:
        def select(combo, value):
            for i in range(combo.count()):
                if combo.itemData(i) == value:
                    combo.setCurrentIndex(i)
                    return True
            return False

        if not select(self.in_combo, self.engine.input_device):
            self.in_combo.setCurrentIndex(0)
        if not select(self.out_combo, self.engine.output_device):
            self.out_combo.setCurrentIndex(0)
        self._update_channel_combo()
        select(self.ch_combo, self.engine.input_channel)
        select(self.sr_combo, self.engine.samplerate)
        select(self.bs_combo, self.engine.blocksize)
        if self.engine.broadcast_device is not None:
            select(self.bcast_combo, self.engine.broadcast_device)
        if hasattr(self, "bcast_led"):
            self.bcast_led.setChecked(self.engine.broadcast.enabled, emit=False)

    # ==================================================================
    # I/O strip: input trim, tuner, output
    # ==================================================================

    def _build_io_strip(self) -> QFrame:
        bar = _card()
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 8, 14, 8)
        h.setSpacing(16)

        # Input
        in_box = QVBoxLayout()
        in_box.setSpacing(4)
        head_in = QHBoxLayout()
        head_in.setSpacing(6)
        head_in.addWidget(_section("INPUT"))
        self.ch_combo.setMinimumWidth(78)
        head_in.addWidget(self.ch_combo)
        head_in.addStretch(1)
        in_box.addLayout(head_in)

        self.in_meter = LevelMeter(2)
        self.in_meter.setFixedWidth(170)
        self.in_meter.setToolTip(
            "Every hardware input on the interface. Play — whichever bar moves "
            "is your guitar; pick that one in the dropdown. Aim for peaks "
            "around -12 to -6 dBFS.")
        in_box.addWidget(self.in_meter)
        in_row = QHBoxLayout()
        self.k_in = Knob("trim", -24, 36, 0, " dB", 1, theme.GREEN, 1.0, 40)
        self.k_in.valueChanged.connect(
            lambda v: setattr(self.engine.chain, "input_gain_db", v))
        in_row.addWidget(self.k_in)
        self.dry_check = QCheckBox("Monitor DI")
        self.dry_check.setToolTip("Blend the raw, un-amped guitar into the output.")
        self.dry_check.toggled.connect(
            lambda v: setattr(self.engine, "monitor_dry", v))
        in_row.addWidget(self.dry_check)
        in_row.addStretch(1)
        in_box.addLayout(in_row)
        h.addLayout(in_box)

        h.addWidget(self._vline())

        # Tuner
        tun_box = QVBoxLayout()
        tun_box.setSpacing(4)
        trow = QHBoxLayout()
        trow.addWidget(_section("TUNER"))
        self.tuner_led = PowerLED(theme.GREEN, 16)
        self.tuner_led.toggled.connect(self._set_tuner)
        trow.addWidget(self.tuner_led)
        trow.addWidget(QLabel(" (T)"))
        trow.itemAt(2).widget().setObjectName("Hint")
        trow.addStretch(1)
        tun_box.addLayout(trow)
        self.tuner = TunerDisplay()
        tun_box.addWidget(self.tuner)
        h.addLayout(tun_box)

        h.addWidget(self._vline())
        h.addStretch(1)

        # Output
        out_box = QVBoxLayout()
        out_box.setSpacing(4)
        out_box.addWidget(_section("OUTPUT"))
        self.out_meter = LevelMeter(2)
        self.out_meter.setFixedWidth(190)
        out_box.addWidget(self.out_meter)
        h.addLayout(out_box)

        outb = self.engine.chain["output"]
        self.k_width = Knob("width", 0, 200, 100, "%", 0, theme.accent("master"),
                            1.0, 40)
        self.k_width.valueChanged.connect(lambda v: setattr(outb, "width", v / 100.0))
        self.k_out = Knob("volume", -40, 6, -6, " dB", 1, theme.AMBER, 1.0, 46)
        self.k_out.valueChanged.connect(lambda v: setattr(outb, "volume_db", v))
        h.addWidget(self.k_width)
        h.addWidget(self.k_out)

        btns = QVBoxLayout()
        btns.setSpacing(5)
        self.btn_bypass = QPushButton("BYPASS")
        self.btn_bypass.setCheckable(True)
        self.btn_bypass.setObjectName("Danger")
        self.btn_bypass.setToolTip("Straight through, no processing (B)")
        self.btn_bypass.toggled.connect(
            lambda v: setattr(self.engine.chain, "global_bypass", v))
        btns.addWidget(self.btn_bypass)
        panic = QPushButton("PANIC")
        panic.setToolTip("Silence every delay and reverb tail immediately")
        panic.clicked.connect(self.engine.panic)
        btns.addWidget(panic)
        h.addLayout(btns)

        return bar

    @staticmethod
    def _vline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet(f"color:{theme.STROKE}; background:{theme.STROKE};")
        line.setFixedWidth(1)
        return line

    # ==================================================================
    # The board
    # ==================================================================

    def _build_board(self) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        holder = QWidget()
        v = QVBoxLayout(holder)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        for row_keys in BOARD_ROWS:
            row = QHBoxLayout()
            row.setSpacing(10)
            for key in row_keys:
                card = build_card(self.engine, key)
                self.cards[key] = card
                row.addWidget(card, 0, Qt.AlignTop)
            row.addStretch(1)
            v.addLayout(row)
        v.addStretch(1)
        area.setWidget(holder)
        return area

    # ==================================================================
    # Transport: looper, recorder, metronome
    # ==================================================================

    def _build_transport(self) -> QFrame:
        bar = _card()
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 9, 14, 10)
        h.setSpacing(14)

        lp = self.engine.looper

        # -- looper
        loop_box = QVBoxLayout()
        loop_box.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(_section("LOOPER"))
        hint = QLabel("space = rec / play / overdub")
        hint.setObjectName("Hint")
        head.addWidget(hint)
        head.addStretch(1)
        loop_box.addLayout(head)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_loop = QPushButton("● REC")
        self.btn_loop.setObjectName("Danger")
        self.btn_loop.setMinimumWidth(96)
        self.btn_loop.clicked.connect(self._loop_toggle)
        row.addWidget(self.btn_loop)

        for text, slot, tip in (
            ("■ Stop", self._loop_stop, "Stop playback (S)"),
            ("↺ Undo", self._loop_undo, "Remove the last overdub (U)"),
            ("✕ Clear", self._loop_clear, "Erase the loop (C)"),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            b.setToolTip(tip)
            row.addWidget(b)

        self.loop_half = QPushButton("½×")
        self.loop_half.setCheckable(True)
        self.loop_half.setToolTip("Half speed — plays the loop an octave down")
        self.loop_half.toggled.connect(
            lambda v: setattr(lp, "speed", 0.5 if v else 1.0))
        row.addWidget(self.loop_half)

        self.loop_rev = QPushButton("◀ Rev")
        self.loop_rev.setCheckable(True)
        self.loop_rev.setToolTip("Play the loop backwards")
        self.loop_rev.toggled.connect(lambda v: setattr(lp, "reverse", v))
        row.addWidget(self.loop_rev)

        self.k_loop = Knob("loop", 0, 150, 100, "%", 0, theme.accent("looper"),
                           1.0, 40)
        self.k_loop.valueChanged.connect(lambda v: setattr(lp, "level", v / 100.0))
        row.addWidget(self.k_loop)

        self.k_decay = Knob("decay", 50, 100, 100, "%", 0, theme.accent("looper"),
                            1.0, 40)
        self.k_decay.setToolTip("Below 100% older layers fade as you overdub")
        self.k_decay.valueChanged.connect(
            lambda v: setattr(lp, "feedback", v / 100.0))
        row.addWidget(self.k_decay)
        row.addStretch(1)
        loop_box.addLayout(row)

        self.loop_bar = LoopProgress()
        loop_box.addWidget(self.loop_bar)
        h.addLayout(loop_box, 1)

        h.addWidget(self._vline())

        # -- recorder
        rec_box = QVBoxLayout()
        rec_box.setSpacing(6)
        rec_box.addWidget(_section("RECORDER"))
        rrow = QHBoxLayout()
        rrow.setSpacing(6)
        self.btn_rec = QPushButton("● RECORD")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setObjectName("Danger")
        self.btn_rec.setMinimumWidth(104)
        self.btn_rec.toggled.connect(self._toggle_record)
        rrow.addWidget(self.btn_rec)
        folder = QPushButton("Folder")
        folder.clicked.connect(self._open_recordings)
        rrow.addWidget(folder)
        rec_box.addLayout(rrow)

        self.di_check = QCheckBox("also save DI")
        self.di_check.setChecked(True)
        self.di_check.setToolTip(
            "Writes a second file with the raw guitar, so you can re-amp the "
            "same take through a different rig later.")
        self.di_check.toggled.connect(
            lambda v: setattr(self.engine.recorder, "record_dry", v))
        rec_box.addWidget(self.di_check)

        self.rec_status = QLabel("idle")
        self.rec_status.setObjectName("Hint")
        rec_box.addWidget(self.rec_status)
        h.addLayout(rec_box)

        h.addWidget(self._vline())

        # -- metronome
        met = self.engine.metronome
        met_box = QVBoxLayout()
        met_box.setSpacing(6)
        mhead = QHBoxLayout()
        mhead.addWidget(_section("CLICK"))
        self.met_led = PowerLED(theme.YELLOW, 16)
        self.met_led.toggled.connect(lambda v: setattr(met, "enabled", v))
        mhead.addWidget(self.met_led)
        mhead.addStretch(1)
        met_box.addLayout(mhead)

        mrow = QHBoxLayout()
        mrow.setSpacing(6)
        self.bpm = QSpinBox()
        self.bpm.setRange(30, 300)
        self.bpm.setValue(100)
        self.bpm.setSuffix(" bpm")
        self.bpm.valueChanged.connect(lambda v: setattr(met, "bpm", float(v)))
        mrow.addWidget(self.bpm)
        self.beats = QSpinBox()
        self.beats.setRange(1, 12)
        self.beats.setValue(4)
        self.beats.setPrefix("/ ")
        self.beats.valueChanged.connect(
            lambda v: setattr(met, "beats_per_bar", int(v)))
        mrow.addWidget(self.beats)
        met_box.addLayout(mrow)

        self.k_click = Knob("level", 0, 100, 35, "%", 0, theme.YELLOW, 1.0, 38)
        self.k_click.valueChanged.connect(lambda v: setattr(met, "level", v / 100.0))
        met_box.addWidget(self.k_click, 0, Qt.AlignLeft)
        h.addLayout(met_box)

        h.addWidget(self._vline())

        # -- stream out (Discord / OBS)
        bc_box = QVBoxLayout()
        bc_box.setSpacing(6)
        bhead = QHBoxLayout()
        bhead.addWidget(_section("STREAM OUT"))
        self.bcast_led = PowerLED(theme.accent("mod"), 16)
        self.bcast_led.setToolTip(
            "ASIO never reaches Windows, so Discord and OBS cannot hear it. "
            "This sends a second copy of your tone somewhere they can.")
        self.bcast_led.toggled.connect(self._toggle_broadcast)
        bhead.addWidget(self.bcast_led)
        bhead.addStretch(1)
        bc_box.addLayout(bhead)

        brow = QHBoxLayout()
        brow.setSpacing(6)
        self.bcast_combo.setMinimumWidth(230)
        brow.addWidget(self.bcast_combo, 1)
        self.btn_bcast_test = QPushButton("Test")
        self.btn_bcast_test.setToolTip(
            "Plays a 2-second tone into the Stream Out device only.\n"
            "You should NOT hear it. Whoever is listening on Discord should.")
        self.btn_bcast_test.clicked.connect(self._test_broadcast)
        brow.addWidget(self.btn_bcast_test)
        bc_box.addLayout(brow)
        bc_box.addWidget(self.bcast_status)
        h.addLayout(bc_box, 1)

        return bar

    # ==================================================================
    # Actions
    # ==================================================================

    def _install_shortcuts(self) -> None:
        for keys, slot in (
            ("Space", self._loop_toggle),
            ("S", self._loop_stop),
            ("U", self._loop_undo),
            ("C", self._loop_clear),
            ("B", lambda: self.btn_bypass.toggle()),
            ("T", lambda: self.tuner_led.setChecked(not self.tuner_led.isChecked())),
            ("R", lambda: self.btn_rec.toggle()),
            ("M", lambda: self.met_led.setChecked(not self.met_led.isChecked())),
        ):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(slot)

    def _loop_toggle(self) -> None:
        self.engine.looper.toggle()

    def _loop_stop(self) -> None:
        self.engine.looper.stop()

    def _loop_undo(self) -> None:
        self.engine.looper.undo()

    def _loop_clear(self) -> None:
        self.engine.looper.clear()

    def _set_tuner(self, on: bool) -> None:
        self._tuner_on = on
        if not on:
            self.tuner.update_pitch("--", 0, 0.0, 0.0, False)

    def _toggle_record(self, on: bool) -> None:
        rec = self.engine.recorder
        if on:
            if not self.engine.running:
                self.btn_rec.setChecked(False)
                self.rec_status.setText("start audio first")
                return
            rec.record_dry = self.di_check.isChecked()
            path = rec.start()
            self.rec_status.setText(os.path.basename(path))
        else:
            path = rec.stop() or rec.wet_path
            if rec.last_error:
                self.rec_status.setText(f"error: {rec.last_error}")
            elif path:
                self.rec_status.setText(f"saved {os.path.basename(path)}")

    ASIO_PANELS = (
        r"C:\Program Files\PreSonus\Universal Control\Universal Control.exe",
        r"C:\Program Files\Focusrite\Focusrite Control\Focusrite Control.exe",
        r"C:\Program Files\Steinberg\Yamaha Steinberg USB Driver\ysusbaudio.exe",
    )

    def _open_asio_panel(self) -> None:
        for path in self.ASIO_PANELS:
            if os.path.exists(path):
                try:
                    subprocess.Popen([path])
                    return
                except Exception:
                    pass
        QMessageBox.information(
            self, "Tanin",
            "Could not find your interface's control panel automatically.\n\n"
            "Open your audio interface's own utility (for a PreSonus AudioBox "
            "that is PreSonus Universal Control), set the buffer size to 128 "
            "or 256 samples, then press Start Audio here again.")

    def _open_recordings(self) -> None:
        os.makedirs(RECORD_DIR, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(RECORD_DIR)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", RECORD_DIR])

    # -- presets -----------------------------------------------------------

    def _refresh_presets(self, select: str | None = None) -> None:
        current = select or self.preset_combo.currentText()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(presetlib.list_presets(PRESET_DIR))
        if current:
            i = self.preset_combo.findText(current)
            if i >= 0:
                self.preset_combo.setCurrentIndex(i)
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self) -> None:
        name = self.preset_combo.currentText()
        if not name:
            return
        try:
            state = presetlib.load_preset(PRESET_DIR, name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Tanin", f"Could not load '{name}':\n{exc}")
            return
        # Route through the engine queue so it lands between blocks rather
        # than halfway through one.
        self.engine.post(lambda: self.engine.chain.apply(state))
        QTimer.singleShot(60, self._refresh_all_cards)

    def _refresh_all_cards(self) -> None:
        for card in self.cards.values():
            card.refresh_from_block()
        ch = self.engine.chain
        self.k_in.setValue(ch.input_gain_db, emit=False)
        self.k_out.setValue(ch["output"].volume_db, emit=False)
        self.k_width.setValue(ch["output"].width * 100.0, emit=False)

    def _save_preset(self) -> None:
        name = self.preset_combo.currentText()
        if not name:
            self._save_preset_as()
            return
        presetlib.save_preset(PRESET_DIR, name, self.engine.chain.snapshot())
        self.statusBar().showMessage(f"Saved '{name}'", 2500)

    def _save_preset_as(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        name = name.strip()
        if not ok or not name:
            return
        presetlib.save_preset(PRESET_DIR, name, self.engine.chain.snapshot())
        self._refresh_presets(select=name)
        self.statusBar().showMessage(f"Saved '{name}'", 2500)

    def _delete_preset(self) -> None:
        name = self.preset_combo.currentText()
        if not name:
            return
        if QMessageBox.question(self, "Tanin", f"Delete preset '{name}'?") \
                != QMessageBox.Yes:
            return
        presetlib.delete_preset(PRESET_DIR, name)
        self._refresh_presets()

    # -- audio -------------------------------------------------------------

    def _collect_device_settings(self) -> None:
        e = self.engine
        e.input_device = self.in_combo.currentData()
        e.output_device = self.out_combo.currentData()
        d = self.ch_combo.currentData()
        e.input_channel = int(d) if d is not None else 0
        e.samplerate = self.sr_combo.currentData() or 48000
        e.blocksize = self.bs_combo.currentData() or 256
        e.exclusive = self.excl_check.isChecked()

    def _restart_audio(self) -> None:
        self._collect_device_settings()
        err = self.engine.start()
        if err:
            self.audio_error.setText(
                f"Could not start: {err}\n"
                "Try a larger buffer, 48000 Hz, or turning off exclusive mode.")
            self.audio_error.setStyleSheet(f"color:{theme.RED};")
            self.btn_audio.setChecked(True)
        else:
            ms = self.engine.stream_latency_ms
            msg = (f"Running — {self.engine.active_blocksize} samples, "
                   f"{ms:.1f} ms round trip.")
            if self.engine.exclusive_fallback:
                msg = ("Exclusive mode was refused by this driver; running in "
                       "shared mode. ") + msg
                self.excl_check.setChecked(False)
            if self.engine.rate_fallback:
                asked, got = self.engine.rate_fallback
                msg = (f"{asked} Hz was refused by the device; running at "
                       f"{got} Hz instead. ") + msg
                self._apply_device_settings_to_ui()
            if ms > 26:
                msg += ("  That is high enough to feel like a delay while "
                        "playing. Lower the buffer size"
                        + (" in the ASIO control panel." if self.engine.using_asio
                           else " here."))
                self.audio_error.setStyleSheet(f"color:{theme.YELLOW};")
            else:
                self.audio_error.setStyleSheet(f"color:{theme.GREEN};")
            self.audio_error.setText(msg)
            self._save_settings()
            self._refresh_all_cards()

    def _stop_audio(self) -> None:
        self.engine.stop()
        self.audio_error.setText("Stopped.")
        self.audio_error.setStyleSheet(f"color:{theme.TEXT_DIM};")

    def _autostart(self) -> None:
        self._restart_audio()
        if not self.engine.running:
            self.btn_audio.setChecked(True)
        elif self.preset_combo.count() and not self._settings.get("preset"):
            i = self.preset_combo.findText("Classic Crunch")
            if i >= 0:
                self.preset_combo.setCurrentIndex(i)
                self._on_preset_selected()

    # -- settings ----------------------------------------------------------

    def _load_settings(self) -> None:
        self._settings = {}
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as fh:
                self._settings = json.load(fh)
        except Exception:
            pass
        e = self.engine
        din, dout = default_devices()
        # Prefer name+API. If the saved interface is unplugged, fall back to
        # the default rather than to whatever now sits at the old index.
        e.input_device = resolve_device(
            self._settings.get("input_name", ""),
            self._settings.get("input_api", ""), "input",
            self._settings.get("input_device"), din)
        e.output_device = resolve_device(
            self._settings.get("output_name", ""),
            self._settings.get("output_api", ""), "output",
            self._settings.get("output_device"), dout)
        e.input_channel = self._settings.get("input_channel", 0)
        e.broadcast.enabled = self._settings.get("broadcast", False)
        e.broadcast_device = find_device(self._settings.get("broadcast_name", ""),
                                         self._settings.get("broadcast_api", ""),
                                         "output")
        if e.broadcast_device is None:
            e.broadcast_device = suggest_broadcast_device(e.input_device)
        e.samplerate = self._settings.get("samplerate", 48000)
        e.blocksize = self._settings.get("blocksize", 256)
        e.exclusive = self._settings.get("exclusive", True)

    def _save_settings(self) -> None:
        e = self.engine
        def dev_name(idx):
            try:
                import sounddevice as sd
                return sd.query_devices(idx)["name"]
            except Exception:
                return ""

        data = {
            "input_device": e.input_device,
            "output_device": e.output_device,
            "input_name": dev_name(e.input_device),
            "input_api": hostapi_name(e.input_device),
            "output_name": dev_name(e.output_device),
            "output_api": hostapi_name(e.output_device),
            "input_channel": e.input_channel,
            "samplerate": e.samplerate,
            "blocksize": e.blocksize,
            "exclusive": e.exclusive,
            "preset": self.preset_combo.currentText(),
            "broadcast": self.engine.broadcast.enabled,
            "broadcast_name": dev_name(self.engine.broadcast_device),
            "broadcast_api": hostapi_name(self.engine.broadcast_device),
        }
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            pass

    # ==================================================================
    # UI tick
    # ==================================================================

    def _on_tick(self) -> None:
        e = self.engine
        self._tick += 1

        nch = max(1, e.n_input_channels)
        if nch != self.in_meter.channels:
            self.in_meter.setChannelCount(nch)
            self.in_meter.labels = [str(i + 1) for i in range(nch)]
            self.in_meter.highlight = e.input_channel
        self.in_meter.setLevels(e.channel_peaks[:nch])
        out = e.chain["output"]
        self.out_meter.setLevels(out.peak_l, out.peak_r)
        self.load_ind.set_load(e.load, e.peak_load)

        if e.running:
            ms = e.stream_latency_ms
            col = theme.GREEN if ms < 14 else (theme.YELLOW if ms < 26 else theme.RED)
            api = "ASIO" if e.using_asio else ("WASAPI excl" if e.exclusive
                                               else "shared")
            self.status_latency.setText(
                f"{api} · {e.samplerate/1000:.1f} kHz · {e.active_blocksize} · "
                f"{ms:.1f} ms")
            self.status_latency.setStyleSheet(f"color:{col};")
        else:
            self.status_latency.setText("audio stopped")
            self.status_latency.setStyleSheet("")

        problems = []
        if e.xruns:
            problems.append(f"{e.xruns} dropouts")
        if e.callback_errors:
            problems.append(f"{e.callback_errors} errors")
        if e.broadcast.running and (e.broadcast.underruns or e.broadcast.overruns):
            problems.append(f"stream out {e.broadcast.underruns + e.broadcast.overruns} gaps")
        self.status_xruns.setText("  ⚠ " + ", ".join(problems) if problems else "")
        self.status_xruns.setStyleSheet(f"color:{theme.RED};" if problems else "")

        lp = e.looper
        self.loop_bar.update_state(lp.state, lp.position_fraction, lp.length_seconds)
        self.btn_loop.setText({
            "empty": "● REC", "recording": "■ END REC",
            "playing": "● OVERDUB", "overdubbing": "▶ PLAY",
            "stopped": "▶ PLAY"}.get(lp.state, "● REC"))

        if e.recorder.recording:
            self.rec_status.setText(f"recording  {e.recorder.seconds_written:5.1f}s")
        elif self.btn_rec.isChecked():
            # Stopping audio ends the take; keep the button honest.
            self.btn_rec.setChecked(False)

        # Tuner at ~8 Hz: the estimate needs a long window and the eye cannot
        # follow anything faster anyway.
        if self._tuner_on and self._tick % 4 == 0 and e.running:
            x = e.input_ring.snapshot(8192)
            freq, clarity = detect_pitch(x, float(e.samplerate))
            if freq > 0 and clarity > 0.4:
                name, octv, cents = freq_to_note(freq)
                self.tuner.update_pitch(name, octv, cents, freq, True)
            else:
                self.tuner.update_pitch("--", 0, 0.0, 0.0, False)

        if self._tick % 30 == 0 and e.running:
            dev = "—"
            try:
                import sounddevice as sd
                dev = sd.query_devices(e.input_device)["name"]
            except Exception:
                pass
            mode = "exclusive" if e.exclusive else "shared"
            self.status_device.setText(f"{dev}  ·  in {e.input_channel + 1}  ·  {mode}")

    def closeEvent(self, event):
        try:
            self._save_settings()
            self.engine.recorder.stop()
            self.engine.stop()
        finally:
            super().closeEvent(event)


def run() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Tanin")
    app.setStyleSheet(theme.QSS)
    win = MainWindow()
    win.show()
    return app.exec()
