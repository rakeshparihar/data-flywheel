# Eval results

1000 real catalog photos plus 500 labelled near-duplicates (83 per type). Embedding: 4.2 min (6.0 img/s); pHash: 0.2 s (6543 img/s). CPU, 2 cores.

## Duplicates (pHash Hamming <= 8, cosine >= 0.92)

| Method | Precision | Recall | recompress | resize | crop | flip | pad | colour | False pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pHash only | 16.4% | 52.0% | 100% | 100% | 4% | 0% | 10% | 98% | 1326 |
| embeddings only | 55.0% | 66.6% | 6% | 26% | 89% | 98% | 93% | 89% | 272 |
| cascade (pHash, then embeddings) | 23.3% | 96.8% | 100% | 100% | 89% | 98% | 94% | 100% | 1590 |

False pairs include genuinely different products that look alike (plain items on white backgrounds); they were not labelled as duplicates.

## Label QA without ground truth

5% of labels flipped on purpose (34 of 692). The cluster check flagged 58 images: precision 41%, recall 71%.

## Drift

| Batches compared | Classifier AUC (0.5 = no drift) |
| --- | --- |
| Summer vs Winter products | 0.901 |
| Two random halves | 0.493 |

## Active labeling (article type, accuracy on held-out images)

| Labels | random | uncertainty | uncertainty + diversity |
| --- | --- | --- | --- |
| 40 | 24.8% | 24.8% | 24.8% |
| 80 | 44.8% | 39.1% | 55.2% |
| 120 | 55.7% | 45.7% | 64.8% |
| 160 | 61.3% | 58.7% | 77.0% |
| 200 | 61.3% | 69.1% | 79.6% |
| 240 | 67.8% | 80.9% | 79.1% |

## Index throughput at scale (synthetic vectors)

IVF index over 1,000,000 random 256-dimension vectors (SigLIP's 768 dims would need PCA to 256 or 3 GB of RAM): build 137.3 s, 3342 queries/s, recall@5 of planted near-duplicates 65.2%. This measures index speed, not image quality.
