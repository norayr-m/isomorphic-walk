# Engine vs biology — 3D-rebuild experiment

Date: 2026-04-19  Agent: Hari Fold Antigyan (4e4d0ac9)

## Graph
- Nodes: 10739
- Seed: node 5542 (entity 25 / 23S, seq 2451 — PTC catalytic A2451)
- BFS reached: 10464/10739 in 21 frames

## Biological displacement (4V9D → 4V9C, working copy, 23S-anchored)
- Nodes with displacement data: 4768/10739
- mean 1.02 Å, median 0.49 Å, p90 2.08, p99 10.76, max 21.34

## Correlation
- n compared: 4686
- Spearman ρ = **+0.3244** (p = 2.91e-115)
- Pearson  r = **+0.0834** (p = 1.07e-08)

## Verdict
SIGNAL: Spearman ρ = +0.324 indicates BFS frame-of-arrival is correlated with biological displacement under translocation. Sign is positive: drivers are reached late in the wavefront.

## Top entities (sorted by mean displacement)
| ent | n | median_frame | median_disp | mean_disp |
|---:|---:|---:|---:|---:|
| 22 | 76 | 8.0 | 13.07 | 12.51 |
| 24 | 182 | 7.0 | 3.74 | 3.81 |
| 32 | 149 | 15.0 | 1.70 | 1.89 |
| 49 | 63 | 13.0 | 0.84 | 0.92 |
| 26 | 118 | 10.0 | 0.68 | 0.72 |
| 46 | 94 | 9.0 | 0.68 | 0.69 |
| 37 | 136 | 5.0 | 0.55 | 0.59 |
| 25 | 2897 | 8.0 | 0.43 | 0.58 |
| 55 | 38 | 6.0 | 0.58 | 0.56 |
| 29 | 201 | 10.0 | 0.50 | 0.55 |
| 42 | 103 | 9.0 | 0.50 | 0.54 |
| 35 | 122 | 7.0 | 0.48 | 0.53 |
| 34 | 142 | 8.0 | 0.47 | 0.52 |
| 28 | 209 | 9.0 | 0.52 | 0.52 |
| 43 | 110 | 9.0 | 0.48 | 0.50 |
