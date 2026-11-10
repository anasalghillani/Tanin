"""Bilingual Tanin / طنين marks: icons that merge the Latin T with the Arabic ط,
and wordmark lockups with both names. Text is converted to outlines, so the
SVGs do not need the font installed.

    .venv\\Scripts\\python.exe assets\\brand\\make_bilingual.py

Font: Dubai (ships with Windows, designed with matching Latin and Arabic).
"""
import math
import sys
from pathlib import Path

from PySide6.QtCore import QPointF
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QGuiApplication, QImage,
                           QPainter, QPainterPath)

from make_logo import (AMBER, AMBER_HI, BG, BG_HI, TEXT, TEXT_DIM, ICO_SIZES,
                       SIZES, render, svg, write_ico)

HERE = Path(__file__).resolve().parent
INK_LIGHT = "#15171b"  # text colour on light backgrounds (theme PANEL)

_st = ('fill="none" stroke="url(#amber)" stroke-linecap="round" '
       'stroke-linejoin="round"')
_TAA = (f'<path d="M44 178 H160 A37 37 0 0 0 160 104 H128 C111 104 100 116 100 132" '
        f'{_st} stroke-width="20"/>'
        f'<path d="M100 46 V178" {_st} stroke-width="20"/>')


def _wave(cx: float, half: float, y0: float, amp: float, period: float) -> str:
    pts = []
    n = int(half * 2)
    for i in range(n + 1):
        x = cx - half + i
        t = abs(x - cx)
        y = y0 - amp * math.exp(-t / 40) * math.sin(2 * math.pi * t / period)
        pts.append(f"{x:.1f} {y:.1f}")
    return "M" + " L".join(pts)


# D: ط whose upright wears a T crossbar - read as T (Latin) and ط (Arabic).
ICON_D = svg('<g transform="translate(8 16)">' + _TAA +
             f'<path d="M58 46 H142" {_st} stroke-width="20"/></g>')

# E: same, but the crossbar is a ringing string (the "resonance" in طنين).
ICON_E = svg('<g transform="translate(8 16)">' + _TAA +
             f'<path d="{_wave(100, 46, 46, 11, 46)}" {_st} stroke-width="16"/></g>')

ICONS = {"d-t-taa": ICON_D, "e-t-taa-wave": ICON_E}


# ---- text to SVG outlines -------------------------------------------------
def load_font() -> str:
    fams = []
    for f in ("DUBAI-MEDIUM.TTF", "DUBAI-REGULAR.TTF"):
        fid = QFontDatabase.addApplicationFont(f"C:/Windows/Fonts/{f}")
        if fid >= 0:
            fams += QFontDatabase.applicationFontFamilies(fid)
    return fams[0] if fams else "Segoe UI"


def text_path(text: str, family: str, px: float, weight=QFont.Medium) -> QPainterPath:
    f = QFont(family)
    f.setPixelSize(int(px))
    f.setWeight(weight)
    p = QPainterPath()
    p.addText(QPointF(0, 0), f, text)  # baseline at y=0, shaped (Arabic joins)
    return p


def path_d(p: QPainterPath, dx: float, dy: float) -> str:
    out, i, n = [], 0, p.elementCount()
    while i < n:
        e = p.elementAt(i)
        if e.type == QPainterPath.MoveToElement:
            out.append(f"M{e.x+dx:.2f} {e.y+dy:.2f}")
        elif e.type == QPainterPath.LineToElement:
            out.append(f"L{e.x+dx:.2f} {e.y+dy:.2f}")
        elif e.type == QPainterPath.CurveToElement:
            c2, end = p.elementAt(i + 1), p.elementAt(i + 2)
            out.append(f"C{e.x+dx:.2f} {e.y+dy:.2f} {c2.x+dx:.2f} {c2.y+dy:.2f} "
                       f"{end.x+dx:.2f} {end.y+dy:.2f}")
            i += 2
        i += 1
    return "".join(out) + "Z"


def icon_body(icon_svg: str) -> str:
    """Inner markup of an icon SVG (defs + shapes), for nesting in a lockup."""
    return icon_svg.split(">", 1)[1].rsplit("</svg>", 1)[0]


def lockup(icon_svg: str, family: str, ink: str, stacked: bool) -> str:
    icon = 160
    en = text_path("Tanin", family, 96 if stacked else 104)
    ar = text_path("طنين", family, 84 if stacked else 104)
    be, ba = en.boundingRect(), ar.boundingRect()
    gap = 44
    # QtSvg ignores nested <svg>, so scale a group instead.
    parts = [f'<g transform="scale({icon / 256})">{icon_body(icon_svg)}</g>']
    if stacked:
        x = icon + gap
        en_base = 78
        ar_base = en_base + 78
        parts.append(f'<path d="{path_d(en, x - be.left(), en_base)}" fill="{ink}"/>')
        parts.append(f'<path d="{path_d(ar, x - ba.left(), ar_base)}" fill="{AMBER}"/>')
        w = x + max(be.width(), ba.width()) + 8
        h = int(ar_base + ba.bottom() + 6)
        # keep the icon vertically centred on the two-line text block
        parts[0] = (f'<g transform="translate(0 {(h - icon) / 2:.1f}) scale({icon / 256})">'
                    f'{icon_body(icon_svg)}</g>')
    else:
        base = icon / 2 + be.height() * 0.36
        x = icon + gap
        parts.append(f'<path d="{path_d(en, x - be.left(), base)}" fill="{ink}"/>')
        x += be.width() + 36
        parts.append(f'<rect x="{x:.1f}" y="{icon*0.24:.1f}" width="4" '
                     f'height="{icon*0.52:.1f}" rx="2" fill="{ink}" opacity="0.3"/>')
        x += 4 + 36
        parts.append(f'<path d="{path_d(ar, x - ba.left(), base)}" fill="{AMBER}"/>')
        w = x + ba.width() + 8
        h = icon
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h}" '
            f'width="{w:.0f}" height="{h}">\n' + "\n".join(parts) + "\n</svg>\n")


def render_wide(svg_text: str, height: int) -> QImage:
    from PySide6.QtCore import QByteArray, QRectF, Qt
    from PySide6.QtSvg import QSvgRenderer
    r = QSvgRenderer(QByteArray(svg_text.encode()))
    vb = r.viewBoxF()
    w = int(round(height * vb.width() / vb.height()))
    img = QImage(w, height, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p, QRectF(0, 0, w, height))
    p.end()
    return img


def preview(path: Path, lockups: dict) -> None:
    W, H = 1600, 920
    img = QImage(W, H, QImage.Format_ARGB32_Premultiplied)
    img.fill(QColor(BG))
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    lab = QFont("Segoe UI", 15, QFont.DemiBold)

    def label(x, y, t):
        p.setPen(QColor(TEXT_DIM))
        p.setFont(lab)
        p.drawText(x, y, t)

    # icons with small sizes
    for i, (name, (title, s)) in enumerate({
            "d": ("D  T + ط", ICON_D), "e": ("E  T + ط, ringing string", ICON_E)}.items()):
        x0 = 40 + i * 800
        label(x0, 50, title)
        p.drawImage(x0, 70, render(s, 240))
        xx = x0 + 280
        for sz in (64, 48, 32, 24, 16):
            p.drawImage(xx, 70 + 120 - sz // 2, render(s, sz))
            xx += sz + 24
    # lockups
    y = 360
    for key in ("d-horizontal", "d-stacked"):
        label(40, y, "Wordmark  " + ("horizontal" if "hor" in key else "stacked") + "  (with D)")
        dark = render_wide(lockups[key + "-dark"], 120)
        p.drawImage(40, y + 20, dark)
        p.fillRect(800, y + 10, 760, 140, QColor("#f3f4f6"))
        light = render_wide(lockups[key + "-light"], 100)
        p.drawImage(820, y + 30, light)
        y += 200
    # in the app's title bar
    label(40, y, "In the app's top bar (32px icon + 18px wordmark)")
    p.fillRect(40, y + 20, 1520, 56, QColor(BG_HI))
    p.drawImage(56, y + 32, render_wide(lockups["d-horizontal-dark"], 32))
    y += 120
    label(40, y, "Fonts: Dubai (Windows) for both scripts, converted to outlines. "
          "Arabic in amber, English in the UI's text colour.")
    p.end()
    img.save(str(path))


def main() -> None:
    app = QGuiApplication(sys.argv)  # noqa: F841
    family = load_font()
    out = HERE / "bilingual"
    out.mkdir(exist_ok=True)
    for key, s in ICONS.items():
        (out / f"tanin-{key}.svg").write_text(s, encoding="utf-8")
        for sz in SIZES:
            render(s, sz).save(str(out / f"tanin-{key}-{sz}.png"))
        write_ico(out / f"tanin-{key}.ico", s)
    lockups = {}
    for icon_key, icon in (("d", ICON_D), ("e", ICON_E)):
        for stacked in (False, True):
            for theme, ink in (("dark", TEXT), ("light", INK_LIGHT)):
                k = f"{icon_key}-{'stacked' if stacked else 'horizontal'}-{theme}"
                s = lockup(icon, family, ink, stacked)
                lockups[k] = s
                (out / f"tanin-wordmark-{k}.svg").write_text(s, encoding="utf-8")
                render_wide(s, 320).save(str(out / f"tanin-wordmark-{k}.png"))
    preview(HERE / "preview-bilingual.png", lockups)
    print("font:", family)


if __name__ == "__main__":
    main()
