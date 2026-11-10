"""Sweep the two duplicate thresholds on the same images as run_eval.py (reuses its embedding cache).

The default thresholds (Hamming <= 8, cosine >= 0.92) gave too many false pairs on small catalog
photos of plain products on white backgrounds; this finds where precision and recall trade off.

Usage: DF_BASE_IMAGES=1000 python evals/threshold_sweep.py --write
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

from flywheel.dedupe import embedding_pairs, hash_pairs, phash
from flywheel.perturb import TYPES, perturb

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    a = p.parse_args()
    seed, n_base = int(os.getenv("DF_SEED", "5")), int(os.getenv("DF_BASE_IMAGES", "1500"))
    cache = Path(os.getenv("DF_CACHE_DIR", "artifacts/embeddings")) / f"all_{seed}_{n_base}.npy"
    if not cache.exists():
        print("run evals/run_eval.py first to build the embedding cache", file=sys.stderr)
        return 1
    rng = random.Random(seed)  # same sequence of draws as run_eval.py
    ds = load_dataset(os.getenv("DF_DATASET", "ashraq/fashion-product-images-small"), split="train")
    meta_len = len(ds)
    idx = rng.sample(range(meta_len), n_base)
    base = [ds[i]["image"].convert("RGB") for i in idx]
    variants, origin = [], []
    for k, img in enumerate(base[: n_base // 2]):
        variants.append(perturb(img, TYPES[k % len(TYPES)], rng))
        origin.append(k)
    emb = np.load(cache)
    assert len(emb) == n_base + len(variants), "cache does not match these images"
    hashes = phash(base + variants)
    truth = {(o, n_base + j) for j, o in enumerate(origin)}

    def score(found):
        tp = len(found & truth)
        return tp / max(len(found), 1), tp / len(truth), len(found - truth)

    lines = ["# Duplicate threshold sweep", "", f"Same {n_base} photos and {len(variants)} planted near-duplicates as RESULTS.md.", "",
             "| Rule | Precision | Recall | False pairs |", "| --- | --- | --- | --- |"]
    hp = {h: hash_pairs(hashes, h) for h in (2, 4, 6, 8)}
    ep = {c: embedding_pairs(emb, c) for c in (0.92, 0.95, 0.97)}
    for h, f in hp.items():
        lines.append(f"| pHash <= {h} | " + " | ".join(f"{x:.1%}" if i < 2 else str(x) for i, x in enumerate(score(f))) + " |")
    for c, f in ep.items():
        lines.append(f"| cosine >= {c} | " + " | ".join(f"{x:.1%}" if i < 2 else str(x) for i, x in enumerate(score(f))) + " |")
    for h in (2, 4):
        for c in (0.95, 0.97):
            lines.append(f"| pHash <= {h} or cosine >= {c} | " + " | ".join(f"{x:.1%}" if i < 2 else str(x) for i, x in enumerate(score(hp[h] | ep[c]))) + " |")
    report = "\n".join(lines) + "\n"
    print(report)
    if a.write:
        (ROOT / "evals/SWEEP.md").write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
