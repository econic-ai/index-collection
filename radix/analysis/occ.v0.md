# Radix Tree Occupancy Analysis

capacity_bits=20, total_slots=1048576

## Load Factor 1% (10485 / 1048576)

### Chunk Occupancy (16 slots each)

| Slots filled | Count | % of chunks |
|-------------|-------|-------------|
| 0 | 55889 | 85.28% |
| 1 | 8856 | 13.51% |
| 2 | 745 | 1.14% |
| 3 | 45 | 0.07% |
| 4 | 1 | 0.00% |

Full chunks (16/16): 0 / 65536 (0.00%)

### Group Occupancy (64 slots each)

| Range | Count | % of groups |
|-------|-------|-------------|
| 0 (empty) | 8681 | 52.98% |
| 1-16 | 7703 | 47.02% |

Full groups (64/64): 0 / 16384 (0.00%)
Full buckets (256/256): 0 / 4096 (0.00%)

### Probe Trace — Hits (10000 lookups)

| Resolution level | Count | % |
|-----------------|-------|----|
| Scalar preferred (match) | 9948 | 99.5% |
| Scalar preferred (empty) | 0 | 0.0% |
| Chunk 0 — NEON remainder | 52 | 0.5% |
| Chunks 1-3 (same group) | 0 | 0.0% |
| Bucket overflows | 0 | 0.0% |
| Avg buckets probed | 1.000 |

### Probe Trace — Misses (10000 lookups)

| Resolution level | Count | % |
|-----------------|-------|----|
| Scalar preferred (match) | 0 | 0.0% |
| Scalar preferred (empty) | 9897 | 99.0% |
| Chunk 0 — NEON remainder | 103 | 1.0% |
| Chunks 1-3 (same group) | 0 | 0.0% |
| Bucket overflows | 0 | 0.0% |
| Avg buckets probed | 1.000 |

## Load Factor 25% (262144 / 1048576)

### Chunk Occupancy (16 slots each)

| Slots filled | Count | % of chunks |
|-------------|-------|-------------|
| 0 | 1226 | 1.87% |
| 1 | 4775 | 7.29% |
| 2 | 9541 | 14.56% |
| 3 | 13021 | 19.87% |
| 4 | 12631 | 19.27% |
| 5 | 10223 | 15.60% |
| 6 | 6825 | 10.41% |
| 7 | 3953 | 6.03% |
| 8 | 1934 | 2.95% |
| 9 | 860 | 1.31% |
| 10 | 351 | 0.54% |
| 11 | 131 | 0.20% |
| 12 | 47 | 0.07% |
| 13 | 15 | 0.02% |
| 14 | 3 | 0.00% |

Full chunks (16/16): 0 / 65536 (0.00%)

### Group Occupancy (64 slots each)

| Range | Count | % of groups |
|-------|-------|-------------|
| 1-16 | 9324 | 56.91% |
| 17-32 | 7060 | 43.09% |

Full groups (64/64): 0 / 16384 (0.00%)
Full buckets (256/256): 0 / 4096 (0.00%)

### Probe Trace — Hits (10000 lookups)

| Resolution level | Count | % |
|-----------------|-------|----|
| Scalar preferred (match) | 9948 | 99.5% |
| Scalar preferred (empty) | 0 | 0.0% |
| Chunk 0 — NEON remainder | 52 | 0.5% |
| Chunks 1-3 (same group) | 0 | 0.0% |
| Bucket overflows | 0 | 0.0% |
| Avg buckets probed | 1.000 |

### Probe Trace — Misses (10000 lookups)

| Resolution level | Count | % |
|-----------------|-------|----|
| Scalar preferred (match) | 0 | 0.0% |
| Scalar preferred (empty) | 7478 | 74.8% |
| Chunk 0 — NEON remainder | 2522 | 25.2% |
| Chunks 1-3 (same group) | 0 | 0.0% |
| Bucket overflows | 0 | 0.0% |
| Avg buckets probed | 1.000 |

## Load Factor 50% (524288 / 1048576)

### Chunk Occupancy (16 slots each)

| Slots filled | Count | % of chunks |
|-------------|-------|-------------|
| 0 | 22 | 0.03% |
| 1 | 188 | 0.29% |
| 2 | 700 | 1.07% |
| 3 | 1839 | 2.81% |
| 4 | 3745 | 5.71% |
| 5 | 6118 | 9.34% |
| 6 | 7855 | 11.99% |
| 7 | 8950 | 13.66% |
| 8 | 9317 | 14.22% |
| 9 | 8209 | 12.53% |
| 10 | 6581 | 10.04% |
| 11 | 4729 | 7.22% |
| 12 | 3066 | 4.68% |
| 13 | 1959 | 2.99% |
| 14 | 1082 | 1.65% |
| 15 | 636 | 0.97% |
| 16 | 540 | 0.82% |

Full chunks (16/16): 540 / 65536 (0.82%)

### Group Occupancy (64 slots each)

| Range | Count | % of groups |
|-------|-------|-------------|
| 1-16 | 21 | 0.13% |
| 17-32 | 8935 | 54.53% |
| 33-48 | 7377 | 45.03% |
| 49-63 | 51 | 0.31% |

Full groups (64/64): 0 / 16384 (0.00%)
Full buckets (256/256): 0 / 4096 (0.00%)

### Probe Trace — Hits (10000 lookups)

| Resolution level | Count | % |
|-----------------|-------|----|
| Scalar preferred (match) | 9948 | 99.5% |
| Scalar preferred (empty) | 0 | 0.0% |
| Chunk 0 — NEON remainder | 52 | 0.5% |
| Chunks 1-3 (same group) | 0 | 0.0% |
| Bucket overflows | 0 | 0.0% |
| Avg buckets probed | 1.000 |

### Probe Trace — Misses (10000 lookups)

| Resolution level | Count | % |
|-----------------|-------|----|
| Scalar preferred (match) | 0 | 0.0% |
| Scalar preferred (empty) | 5048 | 50.5% |
| Chunk 0 — NEON remainder | 4906 | 49.1% |
| Chunks 1-3 (same group) | 46 | 0.5% |
| Bucket overflows | 0 | 0.0% |
| Avg buckets probed | 1.000 |

## Load Factor 75% (786432 / 1048576)

### Chunk Occupancy (16 slots each)

| Slots filled | Count | % of chunks |
|-------------|-------|-------------|
| 1 | 6 | 0.01% |
| 2 | 32 | 0.05% |
| 3 | 108 | 0.16% |
| 4 | 309 | 0.47% |
| 5 | 729 | 1.11% |
| 6 | 1521 | 2.32% |
| 7 | 2651 | 4.05% |
| 8 | 3927 | 5.99% |
| 9 | 5290 | 8.07% |
| 10 | 6278 | 9.58% |
| 11 | 7219 | 11.02% |
| 12 | 7366 | 11.24% |
| 13 | 6938 | 10.59% |
| 14 | 5969 | 9.11% |
| 15 | 4981 | 7.60% |
| 16 | 12212 | 18.63% |

Full chunks (16/16): 12212 / 65536 (18.63%)

### Group Occupancy (64 slots each)

| Range | Count | % of groups |
|-------|-------|-------------|
| 17-32 | 162 | 0.99% |
| 33-48 | 8620 | 52.61% |
| 49-63 | 7355 | 44.89% |
| 64 (full) | 247 | 1.51% |

Full groups (64/64): 247 / 16384 (1.51%)
Full buckets (256/256): 0 / 4096 (0.00%)

### Probe Trace — Hits (10000 lookups)

| Resolution level | Count | % |
|-----------------|-------|----|
| Scalar preferred (match) | 9948 | 99.5% |
| Scalar preferred (empty) | 0 | 0.0% |
| Chunk 0 — NEON remainder | 52 | 0.5% |
| Chunks 1-3 (same group) | 0 | 0.0% |
| Bucket overflows | 0 | 0.0% |
| Avg buckets probed | 1.000 |

### Probe Trace — Misses (10000 lookups)

| Resolution level | Count | % |
|-----------------|-------|----|
| Scalar preferred (match) | 0 | 0.0% |
| Scalar preferred (empty) | 3018 | 30.2% |
| Chunk 0 — NEON remainder | 5740 | 57.4% |
| Chunks 1-3 (same group) | 1242 | 12.4% |
| Bucket overflows | 161 | 1.6% |
| Avg buckets probed | 1.017 |

