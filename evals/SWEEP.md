# Duplicate threshold sweep

Same 1000 photos and 500 planted near-duplicates as RESULTS.md.

| Rule | Precision | Recall | False pairs |
| --- | --- | --- | --- |
| pHash <= 2 | 88.6% | 45.2% | 29 |
| pHash <= 4 | 62.1% | 48.8% | 149 |
| pHash <= 6 | 33.0% | 50.4% | 511 |
| pHash <= 8 | 16.4% | 52.0% | 1326 |
| cosine >= 0.92 | 55.0% | 66.6% | 272 |
| cosine >= 0.95 | 94.0% | 40.6% | 13 |
| cosine >= 0.97 | 100.0% | 19.0% | 0 |
| pHash <= 2 or cosine >= 0.95 | 90.3% | 78.0% | 42 |
| pHash <= 2 or cosine >= 0.97 | 91.3% | 60.8% | 29 |
| pHash <= 4 or cosine >= 0.95 | 71.3% | 80.6% | 162 |
| pHash <= 4 or cosine >= 0.97 | 68.2% | 63.8% | 149 |
