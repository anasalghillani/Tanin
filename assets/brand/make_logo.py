"""Generate the Tanin logo options: SVG sources, PNG sizes, Windows .ico, and a
preview sheet. Run with the project venv:

    .venv\\Scripts\\python.exe assets\\brand\\make_logo.py

Colours come from amprack/ui/theme.py (BG / PANEL_HI / AMBER) so the icon sits
in the same family as the app.
"""
import math
import struct
import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

HERE = Path(__file__).resolve().parent
BG, BG_HI, AMBER, AMBER_HI = "#0d0e11", "#1c1f24", "#ff8a3d", "#ffb547"
TEXT, TEXT_DIM = "#eceef1", "#8d939c"
SIZES = [16, 24, 32, 48, 64, 128, 256, 512, 1024]
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]

DEFS = f"""<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="{BG_HI}"/><stop offset="1" stop-color="{BG}"/>
  </linearGradient>
  <linearGradient id="amber" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{AMBER_HI}"/><stop offset="1" stop-color="{AMBER}"/>
  </linearGradient>
</defs>
<rect width="256" height="256" rx="56" fill="url(#bg)"/>"""


def svg(body: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" '
            f'width="256" height="256">\n{DEFS}\n{body}\n</svg>\n')


def arc_pair(r: float, half_deg: float, width: float, opacity: float) -> str:
    c, d = 128, math.radians(half_deg)
    dx, dy = r * math.cos(d), r * math.sin(d)
    st = (f'fill="none" stroke="url(#amber)" stroke-width="{width}" '
          f'stroke-linecap="round" opacity="{opacity}"')
    return (f'<path d="M{c+dx:.1f} {c-dy:.1f} A{r} {r} 0 0 1 {c+dx:.1f} {c+dy:.1f}" {st}/>'
            f'<path d="M{c-dx:.1f} {c-dy:.1f} A{r} {r} 0 0 0 {c-dx:.1f} {c+dy:.1f}" {st}/>')


# A: "Ring" - a struck point with resonance spreading out either side.
OPT_A = svg(
    '<circle cx="128" cy="128" r="20" fill="url(#amber)"/>'
    + arc_pair(50, 42, 15, 1.0)
    + arc_pair(84, 40, 13, 0.55)
)

# B: "Taa" - the Arabic letter ط (first letter of طنين), drawn as a single
# geometric stroke.
_taa = ('fill="none" stroke="url(#amber)" stroke-width="20" '
        'stroke-linecap="round" stroke-linejoin="round"')
OPT_B = svg(
    '<g transform="translate(8 16)">'
    f'<path d="M44 178 H160 A37 37 0 0 0 160 104 H128 C111 104 100 116 100 132" {_taa}/>'
    f'<path d="M100 46 V178" {_taa}/>'
    '</g>'
)


def _wave() -> str:
    pts = []
    for i in range(0, 161):
        x = 48 + i
        t = abs(x - 128)
        y = 86 - 17 * math.exp(-t / 45) * math.sin(2 * math.pi * t / 56)
        pts.append(f"{x:.1f} {y:.1f}")
    return "M" + " L".join(pts)


# C: "Wave T" - a T whose crossbar is a ringing (decaying) waveform.
OPT_C = svg(
    f'<path d="{_wave()}" fill="none" stroke="url(#amber)" stroke-width="15" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M128 84 V200" fill="none" stroke="url(#amber)" stroke-width="18" '
    'stroke-linecap="round"/>'
)

OPTIONS = {"a-ring": ("A  Ring", OPT_A), "b-taa": ("B  Taa (ط)", OPT_B),
           "c-wave": ("C  Wave T", OPT_C)}


def render(svg_text: str, size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(QByteArray(svg_text.encode())).render(p, QRectF(0, 0, size, size))
    p.end()
    return img


def png_bytes(img: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def write_ico(path: Path, svg_text: str) -> None:
    # PNG-compressed ICO entries (supported since Windows Vista).
    blobs = [png_bytes(render(svg_text, s)) for s in ICO_SIZES]
    out = struct.pack("<HHH", 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    for s, b in zip(ICO_SIZES, blobs):
        dim = 0 if s >= 256 else s
        out += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(b), offset)
        offset += len(b)
    path.write_bytes(out + b"".join(blobs))


def preview(path: Path) -> None:
    W, H = 1500, 1000
    img = QImage(W, H, QImage.Format_ARGB32_Premultiplied)
    img.fill(QColor(BG))
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    col_w = W // 3
    for i, (key, (label, s)) in enumerate(OPTIONS.items()):
        x0 = i * col_w
        p.setPen(QColor(TEXT_DIM))
        p.setFont(QFont("Segoe UI", 16, QFont.DemiBold))
        p.drawText(x0 + 40, 60, label)
        p.drawImage(x0 + (col_w - 300) // 2, 90, render(s, 300))
        # small sizes, at true pixel size, on dark then on light
        y = 440
        for bg in (BG, "#f3f4f6"):
            p.fillRect(x0 + 40, y, col_w - 80, 90, QColor(bg))
            xx = x0 + 70
            for sz in (64, 48, 32, 24, 16):
                p.drawImage(xx, y + (90 - sz) // 2, render(s, sz))
                xx += sz + 28
            y += 110
        # wordmark lockup
        p.drawImage(x0 + 50, 700, render(s, 96))
        p.setPen(QColor(TEXT))
        p.setFont(QFont("Segoe UI", 34, QFont.DemiBold))
        p.drawText(x0 + 162, 760, "Tanin")
        p.setPen(QColor(AMBER))
        p.setFont(QFont("Segoe UI", 24))
        p.drawText(x0 + 168, 800, "طنين")
    p.setPen(QColor(TEXT_DIM))
    p.setFont(QFont("Segoe UI", 13))
    p.drawText(40, 960, "Row 2-3: real pixel sizes (64 / 48 / 32 / 24 / 16) on the app's dark "
               "background and on a light Windows taskbar.")
    p.end()
    img.save(str(path))


def main() -> None:
    app = QGuiApplication(sys.argv)  # noqa: F841 - needed for fonts
    for key, (_, s) in OPTIONS.items():
        d = HERE / key
        d.mkdir(exist_ok=True)
        (d / f"tanin-{key}.svg").write_text(s, encoding="utf-8")
        for sz in SIZES:
            render(s, sz).save(str(d / f"tanin-{key}-{sz}.png"))
        write_ico(d / f"tanin-{key}.ico", s)
    preview(HERE / "preview.png")
    print("done")


if __name__ == "__main__":
    main()
