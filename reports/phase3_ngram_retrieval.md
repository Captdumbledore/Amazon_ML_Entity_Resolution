# Phase 3: Character N-Gram Retrieval Final Metrics

## Overview
Redesigned highly optimized sparse_dot_topn retrieval.

## Results (OVERALL)

| Config | K | Recall | S1 Recovery | Precision | Total Cands | Avg Cands | Zero S1 | Runtime |
|---|---|---|---|---|---|---|---|---|
| 3_5 | 5 | 44.83% | 17.47% | 31.03% | 2,206,812 | 5.0 | 2 | 4h3m17s |
| 3_5 | 10 | 52.85% | 25.10% | 18.29% | 4,413,621 | 10.0 | 2 | 4h3m17s |
| 3_5 | 20 | 58.82% | 31.34% | 10.30% | 8,727,261 | 19.8 | 2 | 4h3m17s |
| 3_5 | 50 | 64.98% | 38.21% | 4.58% | 21,668,181 | 49.1 | 2 | 4h3m17s |
| 3_6 | 5 | 45.77% | 17.65% | 31.69% | 2,206,820 | 5.0 | 1 | 4h6m40s |
| 3_6 | 10 | 53.89% | 25.24% | 18.65% | 4,413,640 | 10.0 | 1 | 4h6m40s |
| 3_6 | 20 | 59.93% | 31.47% | 10.37% | 8,827,280 | 20.0 | 1 | 4h6m40s |
| 3_6 | 50 | 66.13% | 38.18% | 4.58% | 22,068,200 | 50.0 | 1 | 4h6m40s |

## Comparison to Phase 2 Baselines
*Baseline metrics file not found to display inline.*