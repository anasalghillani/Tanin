"""App identity: window/taskbar icon (logo D, T + ط) and Windows taskbar grouping.

Icon files live in assets/brand/app-icon/ (regenerate with
assets/brand/make_bilingual.py, then copy the d-t-taa sizes across).
"""
import sys
from pathlib import Path

ICON_DIR = Path(__file__).resolve().parent.parent / "assets" / "brand" / "app-icon"
APP_ID = "Tanin.GuitarAmp"


def _set_windows_app_id() -> None:
    # Without this, pythonw.exe groups under Python's taskbar entry and shows
    # Python's icon instead of ours.
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass


def icon():
    from PySide6.QtGui import QIcon
    ic = QIcon()
    for png in sorted(ICON_DIR.glob("tanin-*.png")):
        ic.addFile(str(png))
    return ic


def apply(app) -> None:
    """Call right after creating the QApplication, before any window is shown."""
    _set_windows_app_id()
    ic = icon()
    if not ic.isNull():
        app.setWindowIcon(ic)
