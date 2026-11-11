"""Palette and stylesheet.

Dark, because you look at this in a dim room with a guitar in your hands, and
because a bright panel behind a glowing meter destroys the one thing the meter
is for.

Modern-flat: surfaces are separated by tone, not by outlines; one warm brand
colour for "this is selected / this is the main action"; each module keeps a
softened hue of its own so the chain reads at a glance.
"""

# Surfaces, darkest to lightest
BG = "#0d0e11"         # window
PANEL = "#15171b"      # cards
PANEL_HI = "#1c1f24"   # raised: buttons, inputs, selected rows
HOVER = "#23262c"
WELL = "#0f1013"       # recessed: meter troughs, knob tracks
STROKE = "#24272d"     # hairlines, used sparingly
STROKE_HI = "#353942"

TEXT = "#eceef1"
TEXT_DIM = "#8d939c"
TEXT_FAINT = "#5b616b"

RED = "#ff5f6d"
GREEN = "#3ddc97"
YELLOW = "#ffc857"
AMBER = "#ff8a3d"      # brand

# One hue per module. Cool for time/space, warm for gain.
ACCENTS = {
    "gate":   "#9aa8b8",
    "comp":   "#4fd1c5",
    "drive":  "#ffb547",
    "amp":    "#ff8a3d",
    "cab":    "#c49bff",
    "ir":     "#a99bff",
    "mod":    "#56c8ff",
    "delay":  "#5ea4ff",
    "reverb": "#8e94ff",
    "looper": "#3ddc97",
    "rec":    "#ff5f6d",
    "master": "#ff8a3d",
}


def accent(key: str) -> str:
    return ACCENTS.get(key, AMBER)


def rgba(hex_col: str, alpha: float) -> str:
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


FONT = '"Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif'

import os as _os

CHEVRON = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets",
                        "chevron.svg").replace("\\", "/")

QSS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: {FONT};
    font-size: 13px;
}}
QLabel {{ background: transparent; }}

/* ---- surfaces -------------------------------------------------------- */
QFrame#Bar, QFrame#Card, QFrame#Focus {{
    background: {PANEL};
    border: none;
    border-radius: 14px;
}}
QFrame#Card[active="false"] {{ background: {PANEL}; }}
QFrame#Focus QFrame#Card {{ background: transparent; }}
QFrame#Focus QStackedWidget, QFrame#Focus QStackedWidget > QWidget {{
    background: transparent;
}}
QFrame#VLine {{ background: {STROKE}; border: none; }}

/* ---- type ------------------------------------------------------------ */
QLabel#CardTitle {{
    font-size: 20px; font-weight: 600; color: {TEXT};
}}
QLabel#Hint {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel#Section {{
    color: {TEXT_FAINT}; font-size: 11px; font-weight: 600;
    letter-spacing: 1px;
}}
QLabel#PresetName {{ font-size: 18px; font-weight: 600; color: {TEXT}; }}
QLabel#Brand {{
    font-size: 15px; font-weight: 700; color: {AMBER}; letter-spacing: 3px;
}}
QLabel#Arrow {{ color: {TEXT_FAINT}; font-size: 14px; }}

/* ---- inputs ---------------------------------------------------------- */
QComboBox {{
    background: {PANEL_HI};
    border: none;
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 20px;
    color: {TEXT};
}}
QComboBox:hover {{ background: {HOVER}; }}
QComboBox:disabled {{ color: {TEXT_FAINT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: url("{CHEVRON}");
    width: 10px; height: 6px;
    margin-right: 10px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL_HI};
    border: 1px solid {STROKE_HI};
    border-radius: 8px;
    selection-background-color: {rgba(AMBER, 0.18)};
    selection-color: {TEXT};
    outline: none;
    padding: 4px;
}}

QSpinBox, QDoubleSpinBox {{
    background: {PANEL_HI}; border: none; border-radius: 8px;
    padding: 5px 8px; min-height: 20px;
}}
QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}

QCheckBox {{ spacing: 9px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border-radius: 5px;
    border: 1px solid {STROKE_HI}; background: {PANEL_HI};
}}
QCheckBox::indicator:hover {{ border-color: {TEXT_FAINT}; }}
QCheckBox::indicator:checked {{ background: {AMBER}; border-color: {AMBER}; }}

/* ---- buttons --------------------------------------------------------- */
QPushButton {{
    background: {PANEL_HI};
    border: none;
    border-radius: 8px;
    padding: 7px 14px;
    color: {TEXT};
    font-weight: 500;
}}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {PANEL}; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; }}
QPushButton:checked {{
    background: {rgba(AMBER, 0.16)}; color: {AMBER};
}}
QPushButton#Danger {{ color: {TEXT}; }}
QPushButton#Danger:checked {{ background: {rgba(RED, 0.18)}; color: {RED}; }}
QPushButton#Primary, QPushButton#Go {{
    background: {AMBER}; color: #16110c; font-weight: 600;
}}
QPushButton#Primary:hover, QPushButton#Go:hover {{ background: #ff9b57; }}
QPushButton#Ghost {{ background: transparent; color: {TEXT_DIM}; }}
QPushButton#Ghost:hover {{ background: {PANEL_HI}; color: {TEXT}; }}
QPushButton#Icon {{
    background: transparent; color: {TEXT_DIM};
    padding: 0px 9px 3px 9px; font-size: 22px; font-weight: 300;
}}
QPushButton#Icon:hover {{ background: {PANEL_HI}; color: {TEXT}; }}
QPushButton#Pill {{
    border-radius: 14px; padding: 5px 14px; font-weight: 500;
    background: {PANEL_HI}; font-size: 12px;
}}

/* ---- preset list ----------------------------------------------------- */
QListWidget#Presets {{
    background: transparent; border: none; outline: none; font-size: 13px;
}}
QListWidget#Presets::item {{
    padding: 9px 12px; border-radius: 8px; margin: 1px 0; color: {TEXT_DIM};
}}
QListWidget#Presets::item:hover {{ background: {PANEL_HI}; color: {TEXT}; }}
QListWidget#Presets::item:selected {{
    background: {rgba(AMBER, 0.14)}; color: {AMBER}; font-weight: 600;
}}

/* ---- dock tabs: segmented control ------------------------------------ */
QTabWidget::pane {{
    background: {PANEL}; border: none; border-radius: 14px;
    top: 6px;
}}
QTabWidget::tab-bar {{ left: 4px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {TEXT_FAINT}; padding: 6px 16px;
    margin-right: 4px; font-weight: 600; font-size: 12px;
    border: none; border-radius: 8px;
}}
QTabBar::tab:selected {{ background: {PANEL_HI}; color: {TEXT}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}

/* ---- settings window ------------------------------------------------- */
QDialog {{ background: {BG}; }}
QGroupBox {{
    background: {PANEL}; border: none; border-radius: 14px;
    margin-top: 26px; padding: 16px 14px 12px 14px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 4px; top: 4px; color: {TEXT_DIM};
    font-weight: 600; font-size: 12px;
}}

/* ---- misc ------------------------------------------------------------ */
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {STROKE_HI}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {STROKE_HI}; border-radius: 4px; min-width: 30px; }}

QToolTip {{
    background: {PANEL_HI}; color: {TEXT};
    border: 1px solid {STROKE_HI}; padding: 6px 8px; border-radius: 6px;
}}
"""
