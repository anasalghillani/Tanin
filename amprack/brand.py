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


WORDMARK = (Path(__file__).resolve().parent.parent / "assets" / "brand" / "bilingual"
            / "tanin-wordmark-d-horizontal-dark.svg")


def wordmark_pixmap(height: int, dpr: float = 1.0):
    """The 'Tanin | طنين' wordmark rendered crisp at the given logical height.
    Returns None if the file or Qt SVG support is missing, so callers can fall
    back to plain text."""
    try:
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QPainter, QPixmap
        from PySide6.QtSvg import QSvgRenderer
    except Exception:
        return None
    r = QSvgRenderer(str(WORDMARK))
    if not r.isValid():
        return None
    size = r.defaultSize()
    w = height * size.width() / max(1, size.height())
    pm = QPixmap(int(w * dpr), int(height * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    r.render(p, QRectF(0, 0, pm.width(), pm.height()))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def apply(app) -> None:
    """Call right after creating the QApplication, before any window is shown."""
    _set_windows_app_id()
    ic = icon()
    if not ic.isNull():
        app.setWindowIcon(ic)
