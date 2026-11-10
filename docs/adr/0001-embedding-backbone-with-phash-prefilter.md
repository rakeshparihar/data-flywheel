# ADR 1: One embedding backbone for every check, with a pHash prefilter for duplicates

## Status

Accepted.

## Context

The flywheel has four jobs: find near-duplicate images, find wrong labels, detect drift between batches, and pick which images to label next. Two designs were argued by two advocates:

- **A, specialised tools per job:** perceptual hashing for duplicates, a confident-learning library for label errors, per-feature statistical tests for drift. Each is the best-known method for its job.
- **B, one embedding backbone:** embed every image once with a frozen SigLIP encoder and run all four checks on the vectors. One model to version, one cache, and the checks agree on what "similar" means.

A conceded that pHash misses crops, flips and colour changes, which product catalogs are full of. B conceded that embedding every upload is the expensive step and that pHash is roughly two orders of magnitude cheaper for exact recompressions. Both converged, so no judge round was needed.

## Decision

- SigLIP embeddings are the shared representation for duplicates, label QA, drift and active labeling.
- Duplicate detection is a cascade: pHash first (cheap, catches re-encodes and resizes), embeddings second (crops, flips, padding, colour).
- Every check reports a measured number against planted ground truth before it is trusted: labelled variants for duplicates, deliberately flipped labels for QA, a known seasonal split for drift.

## Consequences

- One encoder upgrade changes all four checks at once; the eval re-runs all four, so a regression in one shows up.
- Thresholds are global. The first choice (Hamming 8, cosine 0.92) gave 1,590 false pairs; a sweep moved them to Hamming 2 and cosine 0.95 (precision 90.3%, recall 78.0%). Per-category thresholds are the next step.
- 768-dimension vectors at catalog scale need PCA or product quantisation in the index; the ANN benchmark measures that path on synthetic vectors.
