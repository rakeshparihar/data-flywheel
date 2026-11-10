"""The four parts of the loop, each measured.

1. Duplicates: originals plus labelled variants of six types; precision and recall per type for
   pHash alone, embeddings alone, and the cascade.
2. Label QA: flip 5% of labels on purpose and see how many the cluster check finds.
3. Drift: classifier two-sample test between batches that differ (Summer vs Winter) and batches that don't (random halves).
4. Active labeling: accuracy after each labelling round for random, uncertainty, and uncertainty + diversity.
Plus an ANN throughput benchmark on 1M synthetic vectors.

Usage: python evals/run_eval.py --write
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

from flywheel.dedupe import ann_benchmark, embedding_pairs, hash_pairs, phash
from flywheel.encoder import Encoder
from flywheel.perturb import TYPES, perturb
from flywheel.quality import cluster_suspects, drift_auc, select

ROOT = Path(__file__).resolve().parent.parent


def cached(path: Path, fn):
    if path.exists():
        return np.load(path)
    arr = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    p.add_argument("--ann-vectors", type=int, default=1_000_000)
    args = p.parse_args()
    seed = int(os.getenv("DF_SEED", "5"))
    n_base = int(os.getenv("DF_BASE_IMAGES", "1500"))
    max_h, min_cos = int(os.getenv("DF_HASH_MAX_DISTANCE", "8")), float(os.getenv("DF_COSINE_MIN", "0.92"))
    cache = Path(os.getenv("DF_CACHE_DIR", "artifacts/embeddings"))
    rng = random.Random(seed)

    ds = load_dataset(os.getenv("DF_DATASET", "ashraq/fashion-product-images-small"), split="train")
    meta = ds.remove_columns(["image"]).to_list()
    idx = rng.sample(range(len(meta)), n_base)
    base = [ds[i]["image"].convert("RGB") for i in idx]
    variants, origin, vtype = [], [], []
    for k, img in enumerate(base[: n_base // 2]):  # half the originals get one variant each
        t = TYPES[k % len(TYPES)]
        variants.append(perturb(img, t, rng))
        origin.append(k)
        vtype.append(t)
    images = base + variants
    enc = Encoder(os.getenv("DF_ENCODER", "google/siglip-base-patch16-224"))
    t0 = time.perf_counter()
    emb = cached(cache / f"all_{seed}_{n_base}.npy", lambda: enc.images(images))
    embed_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    hashes = phash(images)
    hash_s = time.perf_counter() - t0

    truth = {(o, n_base + j) for j, o in enumerate(origin)}
    hp, ep = hash_pairs(hashes, max_h), embedding_pairs(emb, min_cos)
    rows = []
    for name, found in [("pHash only", hp), ("embeddings only", ep), ("cascade (pHash, then embeddings)", hp | ep)]:
        per_type = {t: np.mean([(o, n_base + j) in found for j, (o, tt) in enumerate(zip(origin, vtype)) if tt == t]) for t in TYPES}
        precision = len(found & truth) / max(len(found), 1)
        rows.append((name, precision, len(found & truth) / len(truth), per_type, len(found - truth)))

    labels = [meta[i]["articleType"] for i in idx]
    common = {l for l in labels if labels.count(l) >= 15}
    keep = [k for k, l in enumerate(labels) if l in common]
    noisy = [labels[k] for k in keep]
    flip_rng = np.random.default_rng(seed)
    flipped = set(flip_rng.choice(len(keep), int(0.05 * len(keep)), replace=False).tolist())
    names = sorted(common)
    for f in flipped:
        noisy[f] = flip_rng.choice([n for n in names if n != noisy[f]])
    suspects = set(cluster_suspects(emb[keep], noisy, k=40, seed=seed))
    qa_precision = len(suspects & flipped) / max(len(suspects), 1)
    qa_recall = len(suspects & flipped) / len(flipped)

    seasons = np.array([meta[i]["season"] for i in idx])
    summer, winter = np.nonzero(seasons[:n_base] == "Summer")[0], np.nonzero(seasons[:n_base] == "Winter")[0]
    perm = np.random.default_rng(seed).permutation(n_base)
    auc_shift = drift_auc(emb[summer], emb[winter])
    auc_none = drift_auc(emb[perm[: n_base // 2]], emb[perm[n_base // 2: n_base]])

    x = emb[keep]
    y = np.array([labels[k] for k in keep])
    order = np.random.default_rng(seed).permutation(len(keep))
    test, pool = order[: len(keep) // 3], order[len(keep) // 3:]
    curves = {}
    for strategy in ["random", "uncertainty", "uncertainty + diversity"]:
        r = np.random.default_rng(seed)
        labelled = r.choice(pool, 40, replace=False)
        accs = []
        for _ in range(6):
            model = LogisticRegression(max_iter=2000).fit(x[labelled], y[labelled])
            accs.append((len(labelled), float((model.predict(x[test]) == y[test]).mean())))
            available = np.setdiff1d(pool, labelled)
            key = {"random": "random", "uncertainty": "uncertainty", "uncertainty + diversity": "diverse"}[strategy]
            labelled = np.concatenate([labelled, select(key, model, x, available, 40, r)])
        curves[strategy] = accs

    ann = ann_benchmark(args.ann_vectors, dim=256)

    lines = ["# Eval results", "",
             f"{n_base} real catalog photos plus {len(variants)} labelled near-duplicates ({len(variants) // len(TYPES)} per type). "
             f"Embedding: {embed_s / 60:.1f} min ({len(images) / max(embed_s, 1e-9):.1f} img/s); pHash: {hash_s:.1f} s ({len(images) / hash_s:.0f} img/s). CPU, 2 cores.", "",
             f"## Duplicates (pHash Hamming <= {max_h}, cosine >= {min_cos})", "",
             "| Method | Precision | Recall | " + " | ".join(TYPES) + " | False pairs |", "| --- " * (4 + len(TYPES)) + "|"]
    lines += [f"| {n} | {pr:.1%} | {rc:.1%} | " + " | ".join(f"{pt[t]:.0%}" for t in TYPES) + f" | {fp} |" for n, pr, rc, pt, fp in rows]
    lines += ["", "False pairs include genuinely different products that look alike (plain items on white backgrounds); they were not labelled as duplicates.", "",
              "## Label QA without ground truth", "",
              f"5% of labels flipped on purpose ({len(flipped)} of {len(keep)}). The cluster check flagged {len(suspects)} images: "
              f"precision {qa_precision:.0%}, recall {qa_recall:.0%}.", "",
              "## Drift", "", "| Batches compared | Classifier AUC (0.5 = no drift) |", "| --- | --- |",
              f"| Summer vs Winter products | {auc_shift:.3f} |", f"| Two random halves | {auc_none:.3f} |", "",
              "## Active labeling (article type, accuracy on held-out images)", "",
              "| Labels | " + " | ".join(curves) + " |", "| --- " * (len(curves) + 1) + "|"]
    for r_i in range(len(curves["random"])):
        lines.append(f"| {curves['random'][r_i][0]} | " + " | ".join(f"{curves[s][r_i][1]:.1%}" for s in curves) + " |")
    lines += ["", f"## Index throughput at scale (synthetic vectors)", "",
              f"IVF index over {ann['vectors']:,} random {ann['dim']}-dimension vectors (SigLIP's 768 dims would need PCA to 256 or 3 GB of RAM): "
              f"build {ann['build_s']} s, {ann['queries_per_s']:.0f} queries/s, recall@5 of planted near-duplicates {ann['recall_at_5']:.1%}. "
              "This measures index speed, not image quality."]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.write:
        (ROOT / "evals/RESULTS.md").write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
