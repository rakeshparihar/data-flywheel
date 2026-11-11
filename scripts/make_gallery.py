"""Render what the checks actually flagged, on the same images, seed and cached embeddings as evals/run_eval.py.

docs/img/duplicates.png: one planted near-duplicate per type, and which method caught it at the tuned rule
                         (pHash Hamming <= 2 or cosine >= 0.95).
docs/img/label_qa.png:   images the cluster check flagged as mislabelled: the label it was given, and whether
                         that label had really been flipped on purpose.

Usage: DF_BASE_IMAGES=1000 python scripts/make_gallery.py   (after run_eval.py has cached the embeddings)
"""
import os
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

from flywheel.dedupe import embedding_pairs, hash_pairs, phash
from flywheel.perturb import TYPES, perturb
from flywheel.quality import cluster_suspects

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
GREEN, RED, GREY = "#1a7f37", "#c62828", "#333333"


def font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def tile(img: Image.Image, scale: int = 2) -> Image.Image:
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def main() -> None:
    seed = int(os.getenv("DF_SEED", "5"))
    n_base = int(os.getenv("DF_BASE_IMAGES", "1500"))
    cache = Path(os.getenv("DF_CACHE_DIR", "artifacts/embeddings"))
    rng = random.Random(seed)
    ds = load_dataset(os.getenv("DF_DATASET", "ashraq/fashion-product-images-small"), split="train")
    meta = ds.remove_columns(["image"]).to_list()
    idx = rng.sample(range(len(meta)), n_base)
    base = [ds[i]["image"].convert("RGB") for i in idx]
    variants, origin, vtype = [], [], []
    for k, img in enumerate(base[: n_base // 2]):  # same construction and RNG order as run_eval.py
        t = TYPES[k % len(TYPES)]
        variants.append(perturb(img, t, rng))
        origin.append(k)
        vtype.append(t)
    images = base + variants
    emb = np.load(cache / f"all_{seed}_{n_base}.npy")
    hp, ep = hash_pairs(phash(images), 2), embedding_pairs(emb, 0.95)

    f, b, small = font(FONT, 14), font(BOLD, 14), font(FONT, 12)
    out = Path("docs/img")
    out.mkdir(parents=True, exist_ok=True)

    # One example per duplicate type: prefer a caught one, else show the miss.
    cols = []
    for t in TYPES:
        js = [j for j, tt in enumerate(vtype) if tt == t]
        pick = next((j for j in js if (origin[j], n_base + j) in hp | ep), js[0])
        pair = (origin[pick], n_base + pick)
        how = [m for m, s in (("pHash", hp), ("embeddings", ep)) if pair in s]
        cols.append((t, base[origin[pick]], variants[pick], " and ".join(how) if how else "missed"))
    cw, ch = 270, 250
    sheet = Image.new("RGB", (cw * len(cols) + 40, ch), "white")
    d = ImageDraw.Draw(sheet)
    for c, (t, a, v, how) in enumerate(cols):
        x = c * cw
        d.text((x + 10, 6), t, font=b, fill=GREY)
        sheet.paste(tile(a), (x + 10, 30))
        sheet.paste(tile(v), (x + 140, 30))
        d.text((x + 10, 194), "original", font=small, fill=GREY)
        d.text((x + 140, 194), "variant", font=small, fill=GREY)
        d.text((x + 10, 222), "caught by " + how if how != "missed" else "missed", font=b, fill=GREEN if how != "missed" else RED)
    sheet.save(out / "duplicates.png")

    # Label QA, reproducing run_eval.py's flips exactly.
    labels = [meta[i]["articleType"] for i in idx]
    common = {l for l in labels if labels.count(l) >= 15}
    keep = [k for k, l in enumerate(labels) if l in common]
    noisy = [labels[k] for k in keep]
    flip_rng = np.random.default_rng(seed)
    flipped = set(flip_rng.choice(len(keep), int(0.05 * len(keep)), replace=False).tolist())
    names = sorted(common)
    for i in flipped:
        noisy[i] = flip_rng.choice([n for n in names if n != noisy[i]])
    suspects = sorted(cluster_suspects(emb[keep], noisy, k=40, seed=seed))
    hits = [s for s in suspects if s in flipped][:5]
    misses = [s for s in suspects if s not in flipped][:3]
    cards = hits + misses
    cw, ch = 170, 250
    sheet = Image.new("RGB", (cw * len(cards), ch), "white")
    d = ImageDraw.Draw(sheet)
    for c, s in enumerate(cards):
        x = c * cw
        sheet.paste(tile(base[keep[s]]), (x + 25, 8))
        d.text((x + 8, 176), f"given: {noisy[s]}", font=small, fill=GREY)
        if s in flipped:
            d.text((x + 8, 196), f"true: {labels[keep[s]]}", font=small, fill=GREY)
            d.text((x + 8, 218), "real label error", font=b, fill=GREEN)
        else:
            d.text((x + 8, 196), "label was correct", font=small, fill=GREY)
            d.text((x + 8, 218), "false alarm", font=b, fill=RED)
    sheet.save(out / "label_qa.png")
    print(f"duplicates.png: {len(cols)} types; label_qa.png: {len(hits)} real errors, {len(misses)} false alarms "
          f"(of {len(suspects)} flagged, {len(set(suspects) & flipped)} real)")


if __name__ == "__main__":
    main()
