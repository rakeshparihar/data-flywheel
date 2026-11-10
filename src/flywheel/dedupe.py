"""Near-duplicate detection: perceptual hash first, embeddings second.

Stage 1, pHash: a 64-bit hash of each image's low-frequency structure;
images within a small Hamming distance are near-copies (recompressed or
resized). Cheap: one decode and a DCT per image.
Stage 2, embeddings: cosine similarity of SigLIP embeddings via a Faiss
inner-product index; catches crops, flips, padding and colour changes that
pHash misses.
"""
import faiss
import imagehash
import numpy as np


def phash(images) -> np.ndarray:
    return np.array([int(str(imagehash.phash(im)), 16) for im in images], dtype=np.uint64)


def hamming(a: np.ndarray, b: int) -> np.ndarray:
    x = np.bitwise_xor(a, np.uint64(b))
    return np.array([bin(int(v)).count("1") for v in x])


def hash_pairs(hashes: np.ndarray, max_distance: int) -> set[tuple[int, int]]:
    """All pairs within max_distance. Quadratic, fine for tens of thousands; use multi-index hashing beyond."""
    pairs = set()
    for i in range(len(hashes)):
        d = hamming(hashes[i + 1:], int(hashes[i]))
        for j in np.nonzero(d <= max_distance)[0]:
            pairs.add((i, i + 1 + int(j)))
    return pairs


def embedding_pairs(emb: np.ndarray, min_cosine: float, k: int = 10) -> set[tuple[int, int]]:
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))
    sims, ids = index.search(emb.astype(np.float32), k + 1)
    pairs = set()
    for i in range(len(emb)):
        for s, j in zip(sims[i], ids[i]):
            if j != i and j >= 0 and s >= min_cosine:
                pairs.add((min(i, int(j)), max(i, int(j))))
    return pairs


def ann_benchmark(n: int, dim: int = 768, queries: int = 1000, seed: int = 0) -> dict:
    """Build and query an IVF index over n random unit vectors with planted near-duplicates; measures
    build time, query throughput and recall of the planted pairs. Synthetic: speed, not image quality."""
    import time

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q_idx = rng.choice(n, queries, replace=False)
    q = x[q_idx] + 0.05 * rng.standard_normal((queries, dim)).astype(np.float32)  # near-duplicates of known items
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    nlist = int(4 * np.sqrt(n))
    quant = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    t = time.perf_counter()
    index.train(x[rng.choice(n, min(n, 50 * nlist), replace=False)])
    index.add(x)
    build = time.perf_counter() - t
    index.nprobe = 16
    t = time.perf_counter()
    _, ids = index.search(q, 5)
    search = time.perf_counter() - t
    recall = float(np.mean([q_idx[i] in ids[i] for i in range(queries)]))
    return {"vectors": n, "dim": dim, "build_s": round(build, 1), "queries_per_s": round(queries / search, 0), "recall_at_5": round(recall, 3)}
