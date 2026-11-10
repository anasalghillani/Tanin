"""Palette and stylesheet.

Dark, because you look at this in a dim room with a guitar in your hands, and
because a bright panel behind a glowing meter destroys the one thing the meter
is for. Each module gets its own hue so the chain reads at a glance without
having to parse any text.
"""

BG = "#121419"
PANEL = "#1c1f26"
PANEL_HI = "#232732"
STROKE = "#333947"
STROKE_HI = "#4a5264"
TEXT = "#dae0ea"
TEXT_DIM = "#7f8896"
TEXT_FAINT = "#5a616e"

RED = "#ff4d5e"
GREEN = "#41d98b"
YELLOW = "#ffcc4d"
AMBER = "#ff8c32"

# One hue per module. Cool for time/space, warm for gain.
ACCENTS = {
    "gate":   "#8ea3b8",
    "comp":   "#4ecdc4",
    "drive":  "#ffb020",
    "amp":    "#ff6b35",
    "cab":    "#c084fc",
    "ir":     "#a78bfa",
    "mod":    "#3fc7ff",
    "delay":  "#4d9dff",
    "reverb": "#8b8cf8",
    "looper": "#41d98b",
    "rec":    "#ff4d5e",
    "master": "#ff8c32",
}


def accent(key: str) -> str:
    return ACCENTS.get(key, AMBER)


QSS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
}}

QFrame#Card {{
    background: {PANEL};
    border: 1px solid {STROKE};
    border-radius: 10px;
}}
QFrame#Card[active="false"] {{
    background: #191c22;
    border: 1px solid #2a2f3a;
}}

QFrame#Bar {{
    background: {PANEL};
    border: 1px solid {STROKE};
    border-radius: 10px;
}}

QLabel#CardTitle {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.6px;
    color: {TEXT_DIM};
    background: transparent;
}}
QLabel#Hint {{ color: {TEXT_FAINT}; font-size: 11px; background: transparent; }}
QLabel#Section {{
    color: {TEXT_FAINT}; font-size: 10px; font-weight: 700;
    letter-spacing: 1.4px; background: transparent;
}}
QLabel {{ background: transparent; }}

QComboBox {{
    background: {PANEL_HI};
    border: 1px solid {STROKE};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 20px;
    color: {TEXT};
}}
QComboBox:hover {{ border-color: {STROKE_HI}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL_HI};
    border: 1px solid {STROKE_HI};
    selection-background-color: {AMBER};
    selection-color: #14161a;
    outline: none;
    padding: 3px;
}}

QPushButton {{
    background: {PANEL_HI};
    border: 1px solid {STROKE};
    border-radius: 6px;
    padding: 6px 12px;
    color: {TEXT};
    font-weight: 600;
}}
QPushButton:hover {{ background: #2b3140; border-color: {STROKE_HI}; }}
QPushButton:pressed {{ background: #171a20; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: #1a1d23; }}
QPushButton:checked {{
    background: {AMBER}; border-color: {AMBER}; color: #14161a;
}}

QPushButton#Danger:checked {{ background: {RED}; border-color: {RED}; color: #14161a; }}
QPushButton#Go:checked      {{ background: {GREEN}; border-color: {GREEN}; color: #14161a; }}

QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {STROKE}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {STROKE_HI}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {STROKE}; border-radius: 5px; min-width: 30px; }}

QToolTip {{
    background: {PANEL_HI}; color: {TEXT};
    border: 1px solid {STROKE_HI}; padding: 5px 7px; border-radius: 5px;
}}

QSpinBox, QDoubleSpinBox {{
    background: {PANEL_HI}; border: 1px solid {STROKE};
    border-radius: 6px; padding: 3px 6px;
}}
QCheckBox {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid {STROKE_HI}; background: {PANEL_HI};
}}
QCheckBox::indicator:checked {{ background: {AMBER}; border-color: {AMBER}; }}
"""
