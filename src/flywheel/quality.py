"""Label-free QA, drift detection and active-learning selection on embeddings."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_samples
from sklearn.model_selection import cross_val_predict


def cluster_suspects(emb: np.ndarray, labels: list[str], k: int = 30, seed: int = 0) -> list[int]:
    """Images whose label disagrees with the dominant label of their embedding cluster: likely label errors.
    Needs no ground truth beyond the labels being checked."""
    clusters = KMeans(k, n_init=4, random_state=seed).fit_predict(emb)
    suspects = []
    for c in range(k):
        idx = np.nonzero(clusters == c)[0]
        if len(idx) < 5:
            continue
        values, counts = np.unique([labels[i] for i in idx], return_counts=True)
        top = values[counts.argmax()]
        if counts.max() / len(idx) >= 0.7:  # only trust clusters with a clear majority
            suspects += [int(i) for i in idx if labels[i] != top]
    return suspects


def silhouette_by_label(emb: np.ndarray, labels: list[str]) -> dict[str, float]:
    s = silhouette_samples(emb, labels, metric="cosine")
    return {lab: float(s[[l == lab for l in labels]].mean()) for lab in sorted(set(labels))}


def drift_auc(reference: np.ndarray, current: np.ndarray, seed: int = 0) -> float:
    """Classifier two-sample test: train a model to tell the batches apart. AUC near 0.5 means no detectable drift."""
    x = np.vstack([reference, current])
    y = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    p = cross_val_predict(LogisticRegression(max_iter=1000), x, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def select(strategy: str, model, pool: np.ndarray, available: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick n items from `available` (indices into pool) to send for labelling."""
    if strategy == "random":
        return rng.choice(available, n, replace=False)
    probs = model.predict_proba(pool[available])
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]  # small margin = the model cannot decide
    if strategy == "uncertainty":
        return available[np.argsort(margin)[:n]]
    # uncertainty + diversity: take the 5n most uncertain, then spread picks with k-means
    cand = available[np.argsort(margin)[: 5 * n]]
    km = KMeans(n, n_init=2, random_state=0).fit(pool[cand])
    picks = []
    for c in range(n):
        members = np.nonzero(km.labels_ == c)[0]
        d = np.linalg.norm(pool[cand[members]] - km.cluster_centers_[c], axis=1)
        picks.append(cand[members[d.argmin()]])
    return np.array(picks)
