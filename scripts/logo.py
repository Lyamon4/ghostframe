"""Usage: python scripts/logo.py assets [preview.png]

ghostframe logo: SVG files + raster icons from one geometry (48-unit grid).

The mark: a ghost (the frame you see) with its in-between frame trailing
behind it as scanlines, offset along the motion.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageChops

out = Path(sys.argv[1])
INK_DARK = "#f2f2f0"   # on dark backgrounds
INK_LIGHT = "#16171a"  # on light backgrounds
ACCENT = "#7fe3bd"     # spectral mint: the generated frame
ACCENT_LIGHT = "#1f9d72"

# ghost body: x0..x0+24, dome radius 12, sides down to y=37, three hem bumps r=4
TRAIL_DX = -13
STRIPES = [(5.5 + i * 6.2, 3.8) for i in range(6)]  # (y, height)
EYES = [(23.2, 14.0, 3.6, 5.6), (31.2, 14.0, 3.6, 5.6)]
SHIFT = 2.0  # centre the mark (spans x 4..41) in the 48 box


def ghost_path(dx=0.0):
    x0 = 17 + dx + SHIFT
    return (f"M{x0} 17A12 12 0 0 1 {x0+24} 17L{x0+24} 37"
            f"A4 4 0 0 1 {x0+16} 37A4 4 0 0 1 {x0+8} 37A4 4 0 0 1 {x0} 37Z")


def mark_svg(ink, accent, pid, x=0, y=0):
    eyes = "".join(f'<rect x="{ex+SHIFT}" y="{ey}" width="{w}" height="{h}" rx="0.8"/>' for ex, ey, w, h in EYES)
    stripes = "".join(f'<rect x="0" y="{sy}" width="48" height="{sh}"/>' for sy, sh in STRIPES)
    return f'''<g transform="translate({x} {y})">
    <defs>
      <clipPath id="{pid}s">{stripes}</clipPath>
      <mask id="{pid}t" maskUnits="userSpaceOnUse" x="0" y="0" width="48" height="48"><rect width="48" height="48" fill="#fff"/><path d="{ghost_path()}" fill="#000" stroke="#000" stroke-width="3.4"/></mask>
      <mask id="{pid}e" maskUnits="userSpaceOnUse" x="0" y="0" width="48" height="48"><rect width="48" height="48" fill="#fff"/><g fill="#000">{eyes}</g></mask>
    </defs>
    <g mask="url(#{pid}t)"><path d="{ghost_path(TRAIL_DX)}" fill="{accent}" clip-path="url(#{pid}s)"/></g>
    <path d="{ghost_path()}" fill="{ink}" mask="url(#{pid}e)"/>
  </g>'''


FONT = "Inter, 'Segoe UI Variable Display', 'Segoe UI', 'SF Pro Display', Helvetica, Arial, sans-serif"


def logo_svg(ink, accent, pid):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 252 48" width="504" height="96" role="img" aria-label="ghostframe">
  <title>ghostframe</title>
  {mark_svg(ink, accent, pid)}
  <text x="56" y="34.5" fill="{ink}" font-family="{FONT}" font-size="31" font-weight="600" letter-spacing="-0.9">ghost<tspan fill="{ink}" fill-opacity="0.55">frame</tspan></text>
</svg>
'''


def standalone_mark(ink, accent, pid):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="96" height="96" role="img" aria-label="ghostframe">
  <title>ghostframe</title>
  {mark_svg(ink, accent, pid)}
</svg>
'''


# ---- raster ---------------------------------------------------------------

def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ghost_mask(size, s, dx=0.0, grow=0.0):
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    x0 = (17 + dx + SHIFT)

    def c(cx, cy, r):
        d.ellipse([(cx - r - grow) * s, (cy - r - grow) * s, (cx + r + grow) * s, (cy + r + grow) * s], fill=255)
    c(x0 + 12, 17, 12)
    d.rectangle([(x0 - grow) * s, 17 * s, (x0 + 24 + grow) * s, 37 * s], fill=255)
    for k in (4, 12, 20):
        c(x0 + k, 37, 4)
    return m


def icon(px, ink=INK_DARK, accent=ACCENT, outline=None, pad=0.0):
    S = 16
    size = px * S
    s = size / (48 + 2 * pad)
    off = pad
    # draw in a padded coordinate system by shifting via dx/dy: simplest is a
    # bigger canvas then crop
    big = int(round((48 + 2 * pad) * s))
    def shifted(mask):
        if not pad:
            return mask
        canvas = Image.new("L", (big, big), 0)
        canvas.paste(mask.crop((0, 0, big, big)), (int(round(off * s)), int(round(off * s))))
        return canvas
    front = shifted(ghost_mask(big, s))
    halo = shifted(ghost_mask(big, s, grow=1.7))
    trail = shifted(ghost_mask(big, s, dx=TRAIL_DX))
    stripes = Image.new("L", (big, big), 0)
    ds = ImageDraw.Draw(stripes)
    for sy, sh in STRIPES:
        ds.rectangle([0, (sy + off) * s, big, (sy + sh + off) * s], fill=255)
    trail = ImageChops.multiply(trail, stripes)
    trail = ImageChops.subtract(trail, halo)
    eyes = Image.new("L", (big, big), 0)
    de = ImageDraw.Draw(eyes)
    for ex, ey, w, h in EYES:
        de.rounded_rectangle([(ex + SHIFT + off) * s, (ey + off) * s, (ex + SHIFT + w + off) * s, (ey + h + off) * s], radius=0.8 * s, fill=255)
    front = ImageChops.subtract(front, eyes)

    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    if outline:
        ol = ImageChops.subtract(shifted(ghost_mask(big, s, grow=48 / px * 0.9)), front)
        ol = ImageChops.subtract(ol, trail)
        img.paste(Image.new("RGBA", (big, big), hexrgb(outline) + (255,)), (0, 0), ol.point(lambda v: v * 0.75))
    img.paste(Image.new("RGBA", (big, big), hexrgb(accent) + (255,)), (0, 0), trail)
    img.paste(Image.new("RGBA", (big, big), hexrgb(ink) + (255,)), (0, 0), front)
    return img.resize((px, px), Image.LANCZOS)


def main():
    (out / "logo-dark.svg").write_text(logo_svg(INK_DARK, ACCENT, "gd"), encoding="utf-8")
    (out / "logo-light.svg").write_text(logo_svg(INK_LIGHT, ACCENT_LIGHT, "gl"), encoding="utf-8")
    (out / "mark.svg").write_text(standalone_mark(INK_DARK, ACCENT, "gm"), encoding="utf-8")
    icon(32, outline="#111214").save(out / "tray-icon.png")
    big = icon(256, pad=2)
    big.save(out / "icon-256.png")
    sizes = [16, 20, 24, 32, 48, 64, 128, 256]
    big.save(out / "ghostframe.ico", sizes=[(n, n) for n in sizes])
    # preview sheet
    if len(sys.argv) > 2:
        sheet = Image.new("RGBA", (900, 420), (0, 0, 0, 255))
        for row, bg in enumerate([(14, 15, 18), (243, 243, 243)]):
            ImageDraw.Draw(sheet).rectangle([0, row * 210, 900, row * 210 + 210], fill=bg + (255,))
            x = 20
            for n in [16, 24, 32, 64, 128]:
                im = icon(n, outline="#111214" if n <= 32 else None) if row == 0 or n <= 32 else icon(n, INK_LIGHT, ACCENT_LIGHT)
                sheet.alpha_composite(im, (x, row * 210 + 105 - n // 2))
                x += n + 30
            for n in [16, 32]:  # scaled-up views of the tiny ones
                im = icon(n, outline="#111214").resize((n * 6, n * 6), Image.NEAREST)
                sheet.alpha_composite(im, (x, row * 210 + 105 - n * 3))
                x += n * 6 + 20
        sheet.save(sys.argv[2])


main()
print("ok")
