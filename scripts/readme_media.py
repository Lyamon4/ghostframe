"""README media from real ghostframe output.

Usage:
  python scripts/readme_media.py --photo PHOTO --texture TEXTURE --work DIR [--exe ghostframe.exe] [--out assets]

1. A camera move (pan + slow zoom) over PHOTO is rendered at 60 fps; that is
   the ground truth.
2. The even frames are the "game" at 30 fps; `ghostframe interp` doubles them
   back to 60 fps.
3. Every generated frame is compared with the true one, and the images are
   built from those frames: hero.png (on TEXTURE), compare.png, demo.webp.
"""
import argparse
import importlib.util
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 640, 360       # the "game" resolution
FRAMES = 61           # 1 s of ground truth at 60 fps
FONTS = Path("C:/Windows/Fonts")


def font(name, size):
    for n in name:
        try:
            return ImageFont.truetype(str(FONTS / n), size)
        except OSError:
            continue
    return ImageFont.load_default(size)


SEMI = ("seguisb.ttf", "DejaVuSans-Bold.ttf")
REG = ("segoeui.ttf", "DejaVuSans.ttf")


def camera(photo, t):
    """Frame t (0..1) of a pan from left to right with a slow zoom in."""
    sw, sh = photo.size
    e = 0.5 - 0.5 * math.cos(math.pi * t)  # ease in/out, like a game camera
    ww = 520 - 60 * e
    wh = ww * H / W
    cx = ww / 2 + 4 + (sw - ww - 8) * e
    cy = sh * 0.52 - 22 * e
    x0, y0 = cx - ww / 2, cy - wh / 2
    s = ww / W
    return photo.transform((W, H), Image.AFFINE, (s, 0, x0, 0, s, y0), resample=Image.BICUBIC)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def run(args):
    work = Path(args.work)
    truth, game, out = work / "truth", work / "game30", work / "ghostframe60"
    for d in (truth, game, out):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    photo = Image.open(args.photo).convert("RGB")
    for n in range(FRAMES):
        f = camera(photo, n / (FRAMES - 1))
        f.save(truth / f"{n:06}.png")
        if n % 2 == 0:
            f.save(game / f"{n // 2:06}.png")
    subprocess.run([args.exe, "interp", str(game), str(out), "--factor", "2"], check=True)

    T = [load(truth / f"{n:06}.png") for n in range(FRAMES)]
    G = [load(out / f"{n:06}.png") for n in range(FRAMES)]
    rows = {"repeat": [], "cross-fade": [], "ghostframe": []}
    for n in range(1, FRAMES - 1, 2):
        rows["repeat"].append(psnr(T[n - 1], T[n]))
        blend = ((T[n - 1].astype(np.uint16) + T[n + 1]) // 2).astype(np.uint8)
        rows["cross-fade"].append(psnr(blend, T[n]))
        rows["ghostframe"].append(psnr(G[n], T[n]))
    stats = {k: float(np.mean(v)) for k, v in rows.items()}
    for k, v in stats.items():
        print(f"{k:12s} {v:5.1f} dB")
    return T, G, stats


# ---- images -------------------------------------------------------------

INK = (22, 23, 26)


def rounded(img, r):
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], r, fill=255)
    out = img.convert("RGBA")
    out.putalpha(m)
    return out


def shadowed(canvas, card, xy, r=10, blur=14, alpha=70):
    sh = Image.new("RGBA", (card.size[0] + blur * 4, card.size[1] + blur * 4), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([blur * 2, blur * 2, blur * 2 + card.size[0], blur * 2 + card.size[1]], r, fill=(20, 30, 10, alpha))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(sh, (xy[0] - blur * 2, xy[1] - blur * 2 + 6))
    canvas.alpha_composite(rounded(card, r), xy)


def hero(texture, T, G, out, logo):
    WB, HB = 1600, 640
    tex = Image.open(texture).convert("RGB")
    s = max(WB / tex.width, HB / tex.height)
    tex = tex.resize((math.ceil(tex.width * s), math.ceil(tex.height * s)), Image.LANCZOS)
    top = int((tex.height - HB) * 0.42)
    canvas = tex.crop((0, top, WB, top + HB)).convert("RGBA")
    # soften the left half so the type reads
    veil = Image.new("L", (WB, HB), 0)
    vd = ImageDraw.Draw(veil)
    for x in range(WB):
        vd.line([(x, 0), (x, HB)], fill=int(150 * max(0.0, 1 - x / (WB * 0.62)) ** 1.5))
    canvas = Image.composite(Image.new("RGBA", (WB, HB), (247, 245, 236, 255)), canvas, veil)

    d = ImageDraw.Draw(canvas)
    mark = logo.icon(112, ink="#16171a", accent="#2f7d4f")
    canvas.alpha_composite(mark, (96, 150))
    d.text((228, 150), "ghostframe", font=font(SEMI, 92), fill=INK)
    d.text((100, 300), "Open-source frame generation for games.", font=font(REG, 38), fill=(40, 44, 40))
    d.text((100, 352), "Rust + WGSL compute shaders. No paid DLLs, no vendor lock-in.", font=font(REG, 26), fill=(70, 78, 68))
    d.text((100, 430), "30 fps in   \u2192   60 fps out", font=font(SEMI, 30), fill=(47, 125, 79))

    # film strip: real, generated, real
    n = 31
    cards = [(T[n - 1], "real frame"), (G[n], "generated by ghostframe"), (T[n + 1], "real frame")]
    cw, ch = 336, 189
    x0, y0, gap = 1170, 22, 14
    for i, (f, label) in enumerate(cards):
        img = Image.fromarray(f).resize((cw, ch), Image.LANCZOS)
        y = y0 + i * (ch + gap)
        x = x0 + (40 if i == 1 else 0)
        shadowed(canvas, img, (x, y))
        lf = font(SEMI if i == 1 else REG, 17)
        tw = d.textlength(label, font=lf)
        pad = 10
        chip = (x + 10, y + ch - 36, x + 10 + tw + 2 * pad, y + ch - 10)
        d.rounded_rectangle(chip, 8, fill=(47, 125, 79, 235) if i == 1 else (22, 23, 26, 190))
        d.text((chip[0] + pad, chip[1] + 3), label, font=lf, fill=(255, 255, 255))
    canvas.convert("RGB").save(out / "hero.png", optimize=True)


def compare(T, G, stats, out):
    n = 31
    blend = ((T[n - 1].astype(np.uint16) + T[n + 1]) // 2).astype(np.uint8)
    panels = [
        (T[n - 1], "repeat (no frame generation)", stats["repeat"]),
        (blend, "cross-fade", stats["cross-fade"]),
        (G[n], "ghostframe", stats["ghostframe"]),
        (T[n], "the real frame", None),
    ]
    # crop around the climbers (they move with the camera pan)
    cx, cy, cw, ch = 300, 70, 300, 210
    z = 2
    pw, ph = cw * z // 1.5, ch * z // 1.5
    pw, ph = int(pw), int(ph)
    pad, head = 18, 54
    canvas = Image.new("RGB", (pad + 4 * (pw + pad), head + ph + pad + 4), (14, 15, 18))
    d = ImageDraw.Draw(canvas)
    for i, (f, label, db) in enumerate(panels):
        crop = Image.fromarray(f).crop((cx, cy, cx + cw, cy + ch)).resize((pw, ph), Image.LANCZOS)
        x = pad + i * (pw + pad)
        canvas.paste(crop, (x, head))
        color = (127, 227, 189) if label == "ghostframe" else (230, 230, 226)
        d.text((x, 12), label, font=font(SEMI, 20), fill=color)
        if db is not None:
            d.text((x + pw - d.textlength(f"{db:.1f} dB", font=font(REG, 18)), 15), f"{db:.1f} dB", font=font(REG, 18), fill=(150, 150, 146))
    canvas.save(out / "compare.png", optimize=True)


def animation(T, G, out):
    fw, fh = 480, 270
    pad, head, foot = 14, 44, 30
    width = 2 * fw + 3 * pad
    frames = []
    seq = list(range(FRAMES)) + list(range(FRAMES - 2, 0, -1))  # ping-pong
    lf, sf = font(SEMI, 19), font(REG, 15)
    for k, n in enumerate(seq):
        c = Image.new("RGB", (width, head + fh + foot), (14, 15, 18))
        d = ImageDraw.Draw(c)
        left = T[n - n % 2]  # a 30 fps game shows every frame twice
        c.paste(Image.fromarray(left).resize((fw, fh), Image.LANCZOS), (pad, head))
        c.paste(Image.fromarray(G[n]).resize((fw, fh), Image.LANCZOS), (2 * pad + fw, head))
        d.text((pad, 12), "the game: 30 fps", font=lf, fill=(230, 230, 226))
        d.text((2 * pad + fw, 12), "with ghostframe: 60 fps", font=lf, fill=(127, 227, 189))
        if n % 2:
            tag = "generated"
            tw = d.textlength(tag, font=sf)
            d.rounded_rectangle((2 * pad + 2 * fw - tw - 26, head + 10, 2 * pad + 2 * fw - 10, head + 34), 6, fill=(47, 125, 79))
            d.text((2 * pad + 2 * fw - tw - 18, head + 12), tag, font=sf, fill=(255, 255, 255))
        d.text((pad, head + fh + 7), "slowed down 2\u00d7 so every frame is visible", font=sf, fill=(120, 120, 116))
        frames.append(c)
    # 60 fps content shown at 30 fps = half speed. Animated WebP: full
    # colour and a fraction of a GIF's size; GitHub shows it inline.
    frames[0].save(out / "demo.webp", save_all=True, append_images=frames[1:], duration=33, loop=0, quality=82, method=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--texture", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--exe", default="target/release/ghostframe")
    ap.add_argument("--out", default="assets")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("logo", Path(__file__).with_name("logo.py"))
    logo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logo)
    out = Path(a.out)
    T, G, stats = run(a)
    hero(a.texture, T, G, out, logo)
    compare(T, G, stats, out)
    animation(T, G, out)
    for f in ("hero.png", "compare.png", "demo.webp"):
        print(f, (out / f).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
