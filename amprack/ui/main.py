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
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
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
    hostapi_name,
    is_asio,
    list_devices,
    suggest_broadcast_device,
)
from ..tuner import detect_pitch, freq_to_note
from . import theme
from .pedals import ORDER, build_card, card_title
from .widgets import (ChainTile, Knob, LevelMeter, LoadIndicator, LoopProgress,
                      PowerLED, TunerDisplay)

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRESET_DIR = os.path.join(APP_DIR, "presets")
RECORD_DIR = os.path.join(APP_DIR, "recordings")
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")



def _section(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Section")
    return lab


def _card(obj_name: str = "Bar") -> QFrame:
    f = QFrame()
    f.setObjectName(obj_name)
    return f


def _hint(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Hint")
    lab.setWordWrap(True)
    return lab


class MainWindow(QMainWindow):
    """Layout, top to bottom:

    top bar      preset name, status, tuner / bypass / settings
    sidebar      the preset list -- one click loads, arrow keys browse
    chain        IN -> nine module tiles -> OUT, in the order the signal runs
    focus panel  the one module you are editing, with big knobs
    dock         looper / recorder / click, one tab at a time

    Device setup and Stream Out live in a Settings window: set once,
    rarely touched, so they do not get space on the main screen.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tanin")
        self.resize(1360, 880)
        self.setMinimumSize(1100, 720)

        self.engine = Engine(RECORD_DIR)
        presetlib.ensure_factory(PRESET_DIR)
        self._load_settings()

        self.cards = {}
        self.tiles: dict[str, ChainTile] = {}
        self._selected = "amp"
        self._tuner_on = False
        self._tick = 0

        # The settings window owns the device widgets, and the rest of the
        # window reads them, so it is built first.
        self.settings_dlg = self._build_settings_dialog()

        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)

        lay.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_sidebar())
        right = QVBoxLayout()
        right.setSpacing(12)
        right.addWidget(self._build_chain())
        right.addWidget(self._build_focus(), 1)
        right.addWidget(self._build_dock())
        body.addLayout(right, 1)
        lay.addLayout(body, 1)

        self._select_module(self._selected)
        self._install_shortcuts()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(33)

        QTimer.singleShot(120, self._autostart)

    # ==================================================================
    # Top bar
    # ==================================================================

    def _build_header(self) -> QFrame:
        bar = _card()
        h = QHBoxLayout(bar)
        h.setContentsMargins(18, 10, 12, 10)
        h.setSpacing(8)

        title = QLabel("TANIN")
        title.setObjectName("Brand")
        title.setFixedWidth(206)
        h.addWidget(title)

        prev_b = QPushButton("‹")
        prev_b.setObjectName("Icon")
        prev_b.setToolTip("Previous preset")
        prev_b.clicked.connect(lambda: self._step_preset(-1))
        next_b = QPushButton("›")
        next_b.setObjectName("Icon")
        next_b.setToolTip("Next preset")
        next_b.clicked.connect(lambda: self._step_preset(1))
        self.preset_name = QLabel("—")
        self.preset_name.setObjectName("PresetName")
        self.preset_name.setMinimumWidth(220)
        h.addWidget(prev_b)
        h.addWidget(next_b)
        h.addSpacing(6)
        h.addWidget(self.preset_name)

        h.addStretch(1)

        self.status_pill = QPushButton("audio stopped")
        self.status_pill.setObjectName("Pill")
        self.status_pill.setToolTip("Audio status. Click to open audio settings.")
        self.status_pill.clicked.connect(self._show_settings)
        h.addWidget(self.status_pill)

        self.bcast_chip = QPushButton("● Stream Out")
        self.bcast_chip.setObjectName("Pill")
        self.bcast_chip.setToolTip("Your tone is also going to Discord / OBS. "
                                   "Click for Stream Out settings.")
        self.bcast_chip.clicked.connect(self._show_settings)
        self.bcast_chip.setStyleSheet(f"color:{theme.accent('mod')};")
        self.bcast_chip.setVisible(False)
        h.addWidget(self.bcast_chip)

        self.load_ind = LoadIndicator()
        self.load_ind.setToolTip("Share of each audio buffer spent in the DSP. "
                                 "Under 50% is comfortable.")
        h.addWidget(self.load_ind)
        h.addSpacing(6)

        self.btn_tuner = QPushButton("Tuner")
        self.btn_tuner.setCheckable(True)
        self.btn_tuner.setToolTip("Show the tuner (T)")
        self.btn_tuner.toggled.connect(self._set_tuner)
        h.addWidget(self.btn_tuner)

        self.btn_bypass = QPushButton("Bypass")
        self.btn_bypass.setCheckable(True)
        self.btn_bypass.setObjectName("Danger")
        self.btn_bypass.setToolTip("Straight through, no processing (B)")
        self.btn_bypass.toggled.connect(
            lambda v: setattr(self.engine.chain, "global_bypass", v))
        h.addWidget(self.btn_bypass)

        panic = QPushButton("Panic")
        panic.setObjectName("Ghost")
        panic.setToolTip("Silence every delay and reverb tail immediately")
        panic.clicked.connect(self.engine.panic)
        h.addWidget(panic)

        gear = QPushButton("⚙  Settings")
        gear.setObjectName("Ghost")
        gear.setToolTip("Audio device, buffer size and Stream Out")
        gear.clicked.connect(self._show_settings)
        h.addWidget(gear)
        return bar

    # ==================================================================
    # Preset sidebar
    # ==================================================================

    def _build_sidebar(self) -> QFrame:
        bar = _card()
        bar.setFixedWidth(220)
        v = QVBoxLayout(bar)
        v.setContentsMargins(12, 16, 12, 12)
        v.setSpacing(8)
        v.addWidget(_section("PRESETS"))

        self.preset_list = QListWidget()
        self.preset_list.setObjectName("Presets")
        self.preset_list.currentRowChanged.connect(self._on_preset_selected)
        v.addWidget(self.preset_list, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
        save = QPushButton("Save")
        save.setToolTip("Overwrite the selected preset with the current sound")
        save.clicked.connect(self._save_preset)
        save.setObjectName("Primary")
        save_as = QPushButton("Save as…")
        save_as.clicked.connect(self._save_preset_as)
        row.addWidget(save, 1)
        row.addWidget(save_as, 1)
        v.addLayout(row)
        delete = QPushButton("Delete")
        delete.setObjectName("Ghost")
        delete.clicked.connect(self._delete_preset)
        v.addWidget(delete)

        self.preset_msg = _hint()
        v.addWidget(self.preset_msg)
        self._refresh_presets()
        return bar

    # ==================================================================
    # Signal chain: IN -> tiles -> OUT
    # ==================================================================

    def _build_chain(self) -> QFrame:
        bar = _card()
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 10, 10, 10)
        h.setSpacing(2)

        for n, key in enumerate(ORDER):
            if n:
                arrow = QLabel("›")
                arrow.setObjectName("Arrow")
                h.addWidget(arrow)
            short, _full = card_title(key)
            tile = ChainTile(key, short, theme.accent(key),
                             always_on=(key == "amp"))
            tile.selected.connect(self._select_module)
            tile.powerToggled.connect(self._tile_power)
            self.tiles[key] = tile
            h.addWidget(tile, 1)

        return bar

    def _tile_power(self, key: str, on: bool) -> None:
        self.engine.chain[key].enabled = on
        self.cards[key].refresh_from_block()

    def _select_module(self, key: str) -> None:
        self._selected = key
        for k, t in self.tiles.items():
            t.set_selected(k == key)
        if self.btn_tuner.isChecked():
            self.btn_tuner.setChecked(False)
        self.focus_stack.setCurrentWidget(self.cards[key])

    def _tile_subtitle(self, key: str) -> str:
        b = self.engine.chain[key]
        if key == "ir":
            name = getattr(self.cards.get("ir"), "file_label", None)
            t = name.text() if name is not None else ""
            return t if t and not t.startswith("no ") else "no file"
        if key in ("gate", "comp"):
            return "On"
        return str(getattr(b, "model", None) or getattr(b, "mode", "") or "On")

    def _refresh_tiles(self) -> None:
        for key, tile in self.tiles.items():
            tile.set_state(bool(getattr(self.engine.chain[key], "enabled", True)),
                           self._tile_subtitle(key))

    # ==================================================================
    # Focus panel: one module, big
    # ==================================================================

    @staticmethod
    def _vline() -> QFrame:
        line = QFrame()
        line.setObjectName("VLine")
        line.setFixedWidth(1)
        return line

    def _build_io_column(self, title: str, meter: LevelMeter, knob: Knob,
                         extra: QWidget | None = None) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(6)
        col.addWidget(_section(title), 0, Qt.AlignHCenter)
        if extra is not None:
            col.addWidget(extra, 0, Qt.AlignHCenter)
        col.addWidget(meter, 1, Qt.AlignHCenter)
        col.addWidget(knob, 0, Qt.AlignHCenter)
        return col

    def _build_focus(self) -> QFrame:
        frame = _card("Focus")
        h = QHBoxLayout(frame)
        h.setContentsMargins(18, 18, 18, 14)
        h.setSpacing(18)

        # Input column
        self.ch_combo.setFixedWidth(76)
        self.ch_combo.setToolTip("Which jack on your interface the guitar is in")
        self.in_meter = LevelMeter(2, horizontal=False)
        self.in_meter.setToolTip(
            "Every hardware input on the interface. Play: whichever bar moves "
            "is your guitar; pick that one above. Aim for peaks around "
            "-12 to -6 dBFS.")
        self.k_in = Knob("input", -24, 36, 0, " dB", 1, theme.GREEN, 1.0, 40)
        self.k_in.setToolTip("Input level. Turn up if the meter barely moves.")
        self.k_in.valueChanged.connect(
            lambda v: setattr(self.engine.chain, "input_gain_db", v))
        h.addLayout(self._build_io_column("IN", self.in_meter, self.k_in,
                                          self.ch_combo))
        h.addWidget(self._vline())

        self.focus_stack = QStackedWidget()
        for key in ORDER:
            card = build_card(self.engine, key, knob_size=72, per_row=8,
                              spacious=True)
            self.cards[key] = card
            self.focus_stack.addWidget(card)

        tuner_page = QWidget()
        tv = QVBoxLayout(tuner_page)
        tv.setContentsMargins(4, 4, 4, 4)
        tv.addWidget(_section("TUNER"))
        self.tuner = TunerDisplay()
        self.tuner.setMinimumHeight(180)
        tv.addWidget(self.tuner, 1)
        tv.addWidget(_hint("Press T or the Tuner button again to go back."))
        self.focus_stack.addWidget(tuner_page)
        self._tuner_page = tuner_page

        h.addWidget(self.focus_stack, 1)
        h.addWidget(self._vline())

        # Output column
        outb = self.engine.chain["output"]
        self.out_meter = LevelMeter(2, horizontal=False)
        self.out_meter.labels = ["L", "R"]
        self.k_out = Knob("volume", -40, 6, -6, " dB", 1, theme.AMBER, 1.0, 40)
        self.k_out.valueChanged.connect(lambda v: setattr(outb, "volume_db", v))
        h.addLayout(self._build_io_column("OUT", self.out_meter, self.k_out))
        return frame

    # ==================================================================
    # Dock: looper / recorder / click
    # ==================================================================

    def _build_dock(self) -> QTabWidget:
        self.dock = QTabWidget()
        self.dock.setFixedHeight(164)
        self.dock.addTab(self._build_looper(), "Looper")
        self.dock.addTab(self._build_recorder(), "Recorder")
        self.dock.addTab(self._build_click(), "Click")
        return self.dock

    def _build_looper(self) -> QWidget:
        lp = self.engine.looper
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_loop = QPushButton("● REC")
        self.btn_loop.setObjectName("Primary")
        self.btn_loop.setMinimumWidth(110)
        self.btn_loop.setToolTip("Record, then play, then overdub (Space)")
        self.btn_loop.clicked.connect(self._loop_toggle)
        row.addWidget(self.btn_loop)

        for text, slot, tip in (
            ("■  Stop", self._loop_stop, "Stop playback (S)"),
            ("↺  Undo", self._loop_undo, "Remove the last overdub (U)"),
            ("✕  Clear", self._loop_clear, "Erase the loop (C)"),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            b.setToolTip(tip)
            row.addWidget(b)

        self.loop_half = QPushButton("½ speed")
        self.loop_half.setCheckable(True)
        self.loop_half.setToolTip("Half speed: plays the loop an octave down")
        self.loop_half.toggled.connect(
            lambda v: setattr(lp, "speed", 0.5 if v else 1.0))
        row.addWidget(self.loop_half)

        self.loop_rev = QPushButton("Reverse")
        self.loop_rev.setCheckable(True)
        self.loop_rev.setToolTip("Play the loop backwards")
        self.loop_rev.toggled.connect(lambda v: setattr(lp, "reverse", v))
        row.addWidget(self.loop_rev)
        row.addStretch(1)

        self.k_loop = Knob("level", 0, 150, 100, "%", 0, theme.accent("looper"),
                           1.0, 34)
        self.k_loop.valueChanged.connect(lambda v: setattr(lp, "level", v / 100.0))
        row.addWidget(self.k_loop)
        self.k_decay = Knob("decay", 50, 100, 100, "%", 0, theme.accent("looper"),
                            1.0, 34)
        self.k_decay.setToolTip("Below 100% older layers fade as you overdub")
        self.k_decay.valueChanged.connect(
            lambda v: setattr(lp, "feedback", v / 100.0))
        row.addWidget(self.k_decay)
        v.addLayout(row)

        self.loop_bar = LoopProgress()
        v.addWidget(self.loop_bar)
        return w

    def _build_recorder(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(14, 10, 14, 10)
        h.setSpacing(10)
        self.btn_rec = QPushButton("● RECORD")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setObjectName("Danger")
        self.btn_rec.setMinimumWidth(120)
        self.btn_rec.setToolTip("Record what you hear to a WAV file (R)")
        self.btn_rec.toggled.connect(self._toggle_record)
        h.addWidget(self.btn_rec)

        self.di_check = QCheckBox("Also save the clean guitar (DI)")
        self.di_check.setChecked(True)
        self.di_check.setToolTip(
            "Writes a second file with the raw guitar, so you can re-amp the "
            "same take through a different sound later.")
        self.di_check.toggled.connect(
            lambda v: setattr(self.engine.recorder, "record_dry", v))
        h.addWidget(self.di_check)

        folder = QPushButton("Open recordings folder")
        folder.clicked.connect(self._open_recordings)
        h.addWidget(folder)

        self.rec_status = QLabel("idle")
        self.rec_status.setObjectName("Hint")
        h.addWidget(self.rec_status, 1)
        return w

    def _build_click(self) -> QWidget:
        met = self.engine.metronome
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(14, 10, 14, 10)
        h.setSpacing(10)
        self.met_led = PowerLED(theme.YELLOW, 20)
        self.met_led.setToolTip("Metronome on / off (M)")
        self.met_led.toggled.connect(lambda v: setattr(met, "enabled", v))
        h.addWidget(self.met_led)
        h.addWidget(_section("METRONOME"))

        self.bpm = QSpinBox()
        self.bpm.setRange(30, 300)
        self.bpm.setValue(100)
        self.bpm.setSuffix(" bpm")
        self.bpm.valueChanged.connect(lambda v: setattr(met, "bpm", float(v)))
        h.addWidget(self.bpm)
        self.beats = QSpinBox()
        self.beats.setRange(1, 12)
        self.beats.setValue(4)
        self.beats.setSuffix(" beats / bar")
        self.beats.valueChanged.connect(
            lambda v: setattr(met, "beats_per_bar", int(v)))
        h.addWidget(self.beats)

        self.k_click = Knob("level", 0, 100, 35, "%", 0, theme.YELLOW, 1.0, 34)
        self.k_click.valueChanged.connect(lambda v: setattr(met, "level", v / 100.0))
        h.addWidget(self.k_click)
        h.addStretch(1)
        return w

    # ==================================================================
    # Settings window: devices, buffer, Stream Out
    # ==================================================================

    def _show_settings(self) -> None:
        self.settings_dlg.show()
        self.settings_dlg.raise_()
        self.settings_dlg.activateWindow()

    # kept for anything that still calls the old toggle
    def audio_panel_toggle(self, on: bool) -> None:
        if on:
            self._show_settings()
        else:
            self.settings_dlg.hide()

    def _build_settings_dialog(self) -> QDialog:
        dlg = QDialog(self)
        dlg.setWindowTitle("Tanin — Settings")
        dlg.setMinimumWidth(720)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(14, 6, 14, 14)
        v.setSpacing(4)
        v.addWidget(self._build_audio_panel())
        v.addWidget(self._build_extras_panel())
        v.addWidget(self._build_stream_panel())
        close = QPushButton("Done")
        close.clicked.connect(dlg.hide)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        v.addSpacing(8)
        v.addLayout(row)
        return dlg

    def _build_audio_panel(self) -> QGroupBox:
        box = QGroupBox("AUDIO DEVICE")
        g = QGridLayout(box)
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
            "comes from on Windows without a vendor ASIO driver, but nothing "
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

        opts = QHBoxLayout()
        opts.addWidget(self.excl_check)
        opts.addWidget(self.btn_asio_panel)
        opts.addStretch(1)
        self.btn_apply = QPushButton("Start Audio")
        self.btn_apply.setObjectName("Go")
        self.btn_apply.clicked.connect(self._restart_audio)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop_audio)
        opts.addWidget(self.btn_apply)
        opts.addWidget(self.btn_stop)
        g.addLayout(opts, 2, 0, 1, 4)

        self.api_hint = _hint()
        self.audio_error = _hint()
        g.addWidget(self.api_hint, 3, 0, 1, 4)
        g.addWidget(self.audio_error, 4, 0, 1, 4)

        g.setColumnStretch(1, 1)
        self._apply_device_settings_to_ui()
        self._sync_api_mode()
        return box

    def _build_extras_panel(self) -> QGroupBox:
        box = QGroupBox("MONITORING")
        h = QHBoxLayout(box)
        h.setSpacing(14)
        self.dry_check = QCheckBox("Blend the clean guitar (DI) into what you hear")
        self.dry_check.toggled.connect(
            lambda v: setattr(self.engine, "monitor_dry", v))
        h.addWidget(self.dry_check)
        h.addStretch(1)
        outb = self.engine.chain["output"]
        self.k_width = Knob("width", 0, 200, 100, "%", 0, theme.accent("master"),
                            1.0, 34)
        self.k_width.setToolTip("Stereo width of the output. 0% is mono.")
        self.k_width.valueChanged.connect(lambda v: setattr(outb, "width", v / 100.0))
        h.addWidget(self.k_width)
        return box

    def _build_stream_panel(self) -> QGroupBox:
        box = QGroupBox("STREAM OUT  (Discord / OBS)")
        v = QVBoxLayout(box)
        v.setSpacing(8)
        v.addWidget(_hint(
            "ASIO audio never reaches Windows, so Discord and OBS can't hear "
            "it. Stream Out sends a second copy of your tone to a device they "
            "can capture. Then share the Tanin window with sound in Discord; "
            "your mic keeps working as normal."))
        row = QHBoxLayout()
        row.setSpacing(8)
        self.bcast_led = QCheckBox("Send my tone to")
        self.bcast_led.toggled.connect(self._toggle_broadcast)
        row.addWidget(self.bcast_led)
        row.addWidget(self.bcast_combo, 1)
        self.btn_bcast_test = QPushButton("Test")
        self.btn_bcast_test.setToolTip(
            "Plays a 2-second tone into the Stream Out device only.\n"
            "You should NOT hear it. Whoever is listening on Discord should.")
        self.btn_bcast_test.clicked.connect(self._test_broadcast)
        row.addWidget(self.btn_bcast_test)
        v.addLayout(row)
        self.bcast_status = _hint()
        v.addWidget(self.bcast_status)
        self.bcast_led.setChecked(self.engine.broadcast.enabled)
        return box

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
            self.bcast_led.blockSignals(True)
            self.bcast_led.setChecked(self.engine.broadcast.enabled)
            self.bcast_led.blockSignals(False)

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
            ("T", lambda: self.btn_tuner.toggle()),
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
        if on:
            self.focus_stack.setCurrentWidget(self._tuner_page)
        else:
            self.tuner.update_pitch("--", 0, 0.0, 0.0, False)
            self.focus_stack.setCurrentWidget(self.cards[self._selected])

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
            path = rec.stop()
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

    def _current_preset(self) -> str:
        item = self.preset_list.currentItem()
        return item.text() if item is not None else ""

    def _flash(self, msg: str) -> None:
        self.preset_msg.setText(msg)
        QTimer.singleShot(2500, lambda: self.preset_msg.setText(""))

    def _refresh_presets(self, select: str | None = None) -> None:
        current = select or self._current_preset()
        self.preset_list.blockSignals(True)
        self.preset_list.clear()
        self.preset_list.addItems(presetlib.list_presets(PRESET_DIR))
        if current:
            found = self.preset_list.findItems(current, Qt.MatchExactly)
            if found:
                self.preset_list.setCurrentItem(found[0])
        self.preset_list.blockSignals(False)

    def _select_preset(self, name: str) -> bool:
        found = self.preset_list.findItems(name, Qt.MatchExactly)
        if not found:
            return False
        if self.preset_list.currentItem() is found[0]:
            self._on_preset_selected()
        else:
            self.preset_list.setCurrentItem(found[0])   # loads via the signal
        return True

    def _step_preset(self, d: int) -> None:
        n = self.preset_list.count()
        if n:
            self.preset_list.setCurrentRow((self.preset_list.currentRow() + d) % n)

    def _on_preset_selected(self, *_args) -> None:
        name = self._current_preset()
        if not name:
            return
        try:
            state = presetlib.load_preset(PRESET_DIR, name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Tanin", f"Could not load '{name}':\n{exc}")
            return
        self.preset_name.setText(name)
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
        self._refresh_tiles()

    def _save_preset(self) -> None:
        name = self._current_preset()
        if not name:
            self._save_preset_as()
            return
        presetlib.save_preset(PRESET_DIR, name, self.engine.chain.snapshot())
        self._flash(f"Saved '{name}'")

    def _save_preset_as(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        name = name.strip()
        if not ok or not name:
            return
        presetlib.save_preset(PRESET_DIR, name, self.engine.chain.snapshot())
        self._refresh_presets(select=name)
        self.preset_name.setText(name)
        self._flash(f"Saved '{name}'")

    def _delete_preset(self) -> None:
        name = self._current_preset()
        if not name:
            return
        if QMessageBox.question(self, "Tanin", f"Delete preset '{name}'?") \
                != QMessageBox.Yes:
            return
        presetlib.delete_preset(PRESET_DIR, name)
        self._refresh_presets()
        self.preset_name.setText(self._current_preset() or "—")

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
            self._show_settings()
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
            self._show_settings()
            return
        want = self._settings.get("preset") or "Classic Crunch"
        if not self._select_preset(want):
            self._select_preset("Classic Crunch")

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
        # Prefer name+API; the stored index is only a last-resort hint because
        # PortAudio renumbers devices whenever the hardware or host API set
        # changes.
        e.input_device = (
            find_device(self._settings.get("input_name", ""),
                        self._settings.get("input_api", ""), "input")
            or self._settings.get("input_device", din) or din)
        e.output_device = (
            find_device(self._settings.get("output_name", ""),
                        self._settings.get("output_api", ""), "output")
            or self._settings.get("output_device", dout) or dout)
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
            "preset": self._current_preset(),
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

        if self._tick % 3 == 0:
            self._update_status()

        lp = e.looper
        self.loop_bar.update_state(lp.state, lp.position_fraction, lp.length_seconds)
        self.btn_loop.setText({
            "empty": "● REC", "recording": "■ END REC",
            "playing": "● OVERDUB", "overdubbing": "▶ PLAY",
            "stopped": "▶ PLAY"}.get(lp.state, "● REC"))

        if e.recorder.recording:
            self.rec_status.setText(f"recording  {e.recorder.seconds_written:5.1f}s")

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

        if self._tick % 6 == 0:
            self._refresh_tiles()

    def _update_status(self) -> None:
        e = self.engine
        problems = []
        if e.xruns:
            problems.append(f"{e.xruns} dropouts")
        if e.callback_errors:
            problems.append(f"{e.callback_errors} errors")
        if e.broadcast.running and (e.broadcast.underruns or e.broadcast.overruns):
            problems.append(
                f"stream out {e.broadcast.underruns + e.broadcast.overruns} gaps")
        if e.running:
            ms = e.stream_latency_ms
            col = theme.GREEN if ms < 14 else (theme.YELLOW if ms < 26 else theme.RED)
            api = "ASIO" if e.using_asio else ("WASAPI excl" if e.exclusive
                                               else "shared")
            text = f"●  {ms:.1f} ms  ·  {api}  ·  {e.active_blocksize}"
        else:
            col, text = theme.TEXT_FAINT, "●  audio stopped"
        if problems:
            col, text = theme.RED, text + "  ·  ⚠ " + ", ".join(problems)
        self.status_pill.setText(text)
        self.status_pill.setStyleSheet(f"color:{col};")
        self.bcast_chip.setVisible(e.broadcast.running)
        rec_tab = "Recorder  ●" if e.recorder.recording else "Recorder"
        if self.dock.tabText(1) != rec_tab:
            self.dock.setTabText(1, rec_tab)

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
