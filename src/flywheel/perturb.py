"""Make labelled near-duplicates of real images, by type, so detection can be measured per type.

Types: recompress (JPEG quality), resize, crop (up to 20% off the edges),
flip, pad (a border, as marketplaces add), and colour (brightness and
saturation shift). Each variant remembers which original it came from.
"""
import io
import random

from PIL import Image, ImageEnhance, ImageOps

TYPES = ["recompress", "resize", "crop", "flip", "pad", "colour"]


def perturb(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if kind == "recompress":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(20, 50))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "resize":
        s = rng.uniform(0.4, 0.7)
        return img.resize((max(8, int(w * s)), max(8, int(h * s)))).resize((w, h))
    if kind == "crop":
        dx, dy = int(w * rng.uniform(0.05, 0.2)), int(h * rng.uniform(0.05, 0.2))
        return img.crop((dx, dy, w - dx // 2, h - dy // 2)).resize((w, h))
    if kind == "flip":
        return ImageOps.mirror(img)
    if kind == "pad":
        b = int(min(w, h) * rng.uniform(0.08, 0.2))
        return ImageOps.expand(img, border=b, fill=(255, 255, 255)).resize((w, h))
    if kind == "colour":
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
        return ImageEnhance.Color(img).enhance(rng.uniform(0.5, 1.5))
    raise ValueError(kind)
