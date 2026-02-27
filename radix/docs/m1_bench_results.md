# M1 Benchmark Results

## Scope

- Comparison: `radix_tree` (CPU) vs `radix_tree_gpu` (GPU)
- Operations: `iter`, `contains_greedy` (`greedy_search` in notebook labels)
- Metric: `mean`
- Unit: `ns`
- Source files:
  - `analysis/data/m1_radix_tree.csv`
  - `analysis/data/m1_radix_tree_gpu.csv`

`GPU vs CPU` is computed as `CPU mean / GPU mean` (`> 1` means GPU is faster, `< 1` means GPU is slower).

## M1 Iteration Results (Mean Latency)

| Index size | Occupancy | Radix CPU | Radix GPU | GPU vs CPU |
|---|---:|---:|---:|---:|
| 256K | 0.10 | 31.1 us | 280.1 us | **0.11x (worse)** |
| 256K | 0.90 | 202.9 us | 369.0 us | **0.55x (worse)** |
| 1M | 0.10 | 362.2 us | 387.5 us | **0.93x (worse)** |
| 1M | 0.90 | 842.4 us | 686.6 us | **1.23x (better)** |
| 8M | 0.10 | 4.19 ms | 1.36 ms | **3.09x (better)** |
| 8M | 0.90 | 6.38 ms | 3.87 ms | **1.65x (better)** |
| 128M | 0.10 | 84.53 ms | 11.29 ms | **7.49x (better)** |
| 128M | 0.90 | 110.09 ms | 44.76 ms | **2.46x (better)** |

### Iteration Detailed Analysis

- **Crossover behavior:** GPU is clearly slower at `256K`, near parity at `1M`, and clearly faster at `8M` and `128M`.
- **Best gain:** `128M @ 0.10` with **7.49x** GPU speedup.
- **Occupancy effect:** For larger sizes, GPU speedup is stronger at lower occupancy (`0.10`) than at high occupancy (`0.90`), but remains >1 in both at `8M+`.
- **Aggregate signal:** geometric-mean speedup is:
  - **1.24x** at `0.10`
  - **1.29x** at `0.90`
  - **1.27x overall**

## M1 Greedy Search Results (Mean Latency)

| Index size | Occupancy | Radix CPU | Radix GPU | GPU vs CPU |
|---|---:|---:|---:|---:|
| 256K | 0.10 | 9.7 us | 212.7 us | **0.05x (worse)** |
| 256K | 0.90 | 1.8 us | 244.1 us | **0.01x (worse)** |
| 1M | 0.10 | 6.5 us | 287.1 us | **0.02x (worse)** |
| 1M | 0.90 | 45.1 us | 256.4 us | **0.18x (worse)** |
| 8M | 0.10 | 88.5 us | 573.5 us | **0.15x (worse)** |
| 8M | 0.90 | 521.6 us | 585.8 us | **0.89x (worse)** |
| 128M | 0.10 | 4.80 ms | 6.10 ms | **0.79x (worse)** |
| 128M | 0.90 | 23.62 ms | 5.99 ms | **3.94x (better)** |

### Greedy Search Detailed Analysis

- **Broadly CPU-favored regime:** GPU is slower in 7/8 measured greedy-search points.
- **Single strong GPU win:** `128M @ 0.90` shows **3.94x** GPU speedup.
- **Near-parity point:** `8M @ 0.90` is close (`0.89x`), suggesting transition pressure at high occupancy and larger sizes.
- **Aggregate signal:** geometric-mean speedup is:
  - **0.11x** at `0.10` (GPU materially worse)
  - **0.26x** at `0.90` (still GPU-worse overall, with one outlier win)
  - **0.17x overall**

## M1 Set Predicate Results (Mean Latency)

### Descriptor

`set_predicate` benchmarks multi-set predicate algebra. The M1 dataset includes:
- **Family-2 (2-array predicates):** `intersect_2`, `diff_ab`, `sym_diff`
- **Family-3 (3-array predicates):** `consensus_3`, `unique_a`

All values below are from the latest `mean ns` rows per exact key:
`(impl, index_size, occupancy, overlap, predicate)`.

### Family-2 (2-array) Summary

**What it is:** Binary set algebra (`A∩B`, `A−B`, symmetric difference) across size/occupancy/overlap grid.

| Predicate | CPU geomean (ns) | GPU geomean (ns) | GPU vs CPU geomean | Worst point (GPU vs CPU) | Best point (GPU vs CPU) |
|---|---:|---:|---:|---|---|
| `intersect_2` | 6,611,876.090 | 1,043,831.786 | **6.3342x (better)** | 1.0255x @ `256K / lf=0.10 / ov=0.10` | 30.8566x @ `128M / lf=0.90 / ov=0.50` |
| `diff_ab` | 6,707,314.159 | 1,045,280.993 | **6.4168x (better)** | 1.0472x @ `256K / lf=0.10 / ov=0.90` | 31.7595x @ `128M / lf=0.90 / ov=0.50` |
| `sym_diff` | 7,296,802.324 | 1,069,589.375 | **6.8221x (better)** | 1.1052x @ `256K / lf=0.10 / ov=0.90` | 30.9768x @ `128M / lf=0.90 / ov=0.50` |
| **Family-2 aggregate** | **6,865,444.255** | **1,052,834.765** | **6.5209x (better)** | — | — |

**Qualitative summary:** Family-2 is consistently GPU-favorable across the full matrix, with near-parity only at the smallest-size edge and very large gains in larger/high-load settings.

### Family-3 (3-array) Summary

**What it is:** Ternary set predicates (consensus/uniqueness style filters) across the same size/occupancy/overlap grid.

| Predicate | CPU geomean (ns) | GPU geomean (ns) | GPU vs CPU geomean | Worst point (GPU vs CPU) | Best point (GPU vs CPU) |
|---|---:|---:|---:|---|---|
| `consensus_3` | 11,065,618.826 | 1,122,597.087 | **9.8572x (better)** | 2.0367x @ `256K / lf=0.10 / ov=0.10` | 34.8142x @ `128M / lf=0.90 / ov=0.50` |
| `unique_a` | 10,666,949.184 | 1,100,965.090 | **9.6887x (better)** | 2.2124x @ `256K / lf=0.10 / ov=0.90` | 30.6936x @ `8M / lf=0.90 / ov=0.50` |
| **Family-3 aggregate** | **10,864,455.518** | **1,111,728.475** | **9.7726x (better)** | — | — |

**Qualitative summary:** Family-3 shows the strongest GPU scaling in M1. Unlike Family-2, even worst-case points remain clearly GPU-faster.

### Set Predicate Overall

| Scope | CPU geomean (ns) | GPU geomean (ns) | GPU vs CPU geomean |
|---|---:|---:|---:|
| All `set_predicate` points (Family-2 + Family-3) | 8,249,058.311 | 1,076,008.302 | **7.6664x (better)** |

**Qualitative summary:** `set_predicate` is a strong win for GPU in M1. Performance gains are broad, with strongest acceleration at larger index sizes and higher occupancy/overlap pressure.

## Research-Paper Summary

M1 results show that GPU benefit is operation-dependent and scale-sensitive. For **iteration**, the data exhibits a clean size-driven crossover: GPU underperforms at small index sizes, reaches near parity around 1M, and becomes decisively faster at 8M and 128M. For **greedy search**, the opposite pattern dominates: GPU is generally slower across the matrix, with one notable high-load large-index win (`128M @ 0.90`). This suggests that the GPU path amortizes setup and transfer overhead effectively for throughput-dominant sequential scans (`iter`) but is not yet consistently advantageous for the current greedy lookup path, except in specific high-workload regimes.

## Notes / Validity Considerations

- All comparisons use the same unit (`ns`) and same aggregation (`mean`) from benchmark CSV outputs.
- These are point measurements from the latest dataset snapshot; confidence intervals are not yet reported.
- If this section is promoted to final paper text, add repeated-run statistics and hardware/runtime configuration metadata.
