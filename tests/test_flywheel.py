import random

import numpy as np
from PIL import Image, ImageDraw

from flywheel.dedupe import ann_benchmark, embedding_pairs, hash_pairs, hamming, phash
from flywheel.perturb import TYPES, perturb
from flywheel.quality import cluster_suspects, drift_auc, select


def picture(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for _ in range(6):
        x, y = rng.integers(0, 80, 2)
        d.rectangle([x, y, x + rng.integers(8, 30), y + rng.integers(8, 30)], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    return img


def test_every_perturbation_keeps_size():
    rng = random.Random(0)
    for t in TYPES:
        assert perturb(picture(1), t, rng).size == (96, 96)


def test_phash_matches_recompressed_copy_not_other_image():
    rng = random.Random(0)
    a, b = picture(1), picture(2)
    h = phash([a, perturb(a, "recompress", rng), b])
    assert hamming(h[1:2], int(h[0]))[0] <= 8
    assert (0, 1) in hash_pairs(h, 8) and (0, 2) not in hash_pairs(h, 8)


def test_embedding_pairs_use_cosine_threshold():
    e = np.array([[1, 0], [0.99, 0.141], [0, 1]], dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    assert embedding_pairs(e, 0.95, k=2) == {(0, 1)}


def test_cluster_check_finds_flipped_labels():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(4, 16)) * 5
    emb = np.vstack([centres[i] + rng.normal(size=(50, 16)) for i in range(4)])
    labels = [str(i // 50) for i in range(200)]
    labels[3], labels[60] = "3", "0"  # two wrong labels
    assert {3, 60} <= set(cluster_suspects(emb, labels, k=4))


def test_drift_auc_separates_shifted_batches():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, (200, 8)), rng.normal(0.8, 1, (200, 8))
    assert drift_auc(a, b) > 0.8
    assert drift_auc(a[:100], a[100:]) < 0.65


def test_selection_strategies_return_unlabelled_items():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4))
    y = (x[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(x[:50], y[:50])
    available = np.arange(50, 300)
    for s in ["random", "uncertainty", "diverse"]:
        picks = select(s, model, x, available, 10, rng)
        assert len(set(picks.tolist())) == 10 and set(picks.tolist()) <= set(available.tolist())


def test_ann_benchmark_finds_planted_neighbours():
    out = ann_benchmark(5000, dim=32, queries=100)
    assert out["recall_at_5"] > 0.9
