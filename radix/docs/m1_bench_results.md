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

## M1 Set Predicate Arity Study (2, 3, and 5 Arrays)

### Why this matters

This section documents the expanded `set_predicate` benchmark where predicates are grouped by algebraic arity:
- **Family-2 (2-array):** `intersect_2`, `diff_ab`, `sym_diff`
- **Family-3 (3-array):** `consensus_3`, `unique_a`
- **Family-5 (5-array):** `consensus_5`, `quorum_3of5`, `unique_a_5`

These are representative of increasingly complex relational/inference-style filters used in database query plans, rule engines, and feature-inference passes. As arity increases, each operation performs more combinational work per key, so this section focuses on whether GPU acceleration scales with that added algebraic complexity.

All values are computed from the latest `mean ns` rows per exact key:
`(impl, index_size, occupancy, overlap, predicate)`, with `GPU vs CPU = CPU/GPU`.

### Family-Level Results (Exact)

| Family | Predicates | CPU geomean (ns) | GPU geomean (ns) | GPU vs CPU geomean | Worst point (GPU vs CPU) | Best point (GPU vs CPU) |
|---|---|---:|---:|---:|---|---|
| Family-2 | `intersect_2`, `diff_ab`, `sym_diff` | 6,471,685.488 | 1,108,146.507 | **5.8401x (better)** | 0.8072x @ `diff_ab / 256K / lf=0.10 / ov=0.90` | 29.6955x @ `diff_ab / 128M / lf=0.90 / ov=0.50` |
| Family-3 | `consensus_3`, `unique_a` | 10,169,468.988 | 1,174,722.126 | **8.6569x (better)** | 1.7729x @ `consensus_3 / 256K / lf=0.10 / ov=0.50` | 31.8218x @ `consensus_3 / 128M / lf=0.90 / ov=0.50` |
| Family-5 | `consensus_5`, `quorum_3of5`, `unique_a_5` | 17,597,755.876 | 1,350,895.109 | **13.0267x (better)** | 2.7859x @ `unique_a_5 / 256K / lf=0.10 / ov=0.90` | 45.5320x @ `consensus_5 / 128M / lf=0.90 / ov=0.50` |

### Per-Predicate Results (Exact)

| Family | Predicate | CPU geomean (ns) | GPU geomean (ns) | GPU vs CPU geomean |
|---|---|---:|---:|---:|
| Family-2 | `intersect_2` | 6,208,306.919 | 1,091,103.639 | **5.6899x** |
| Family-2 | `diff_ab` | 6,330,137.364 | 1,093,075.029 | **5.7911x** |
| Family-2 | `sym_diff` | 6,897,090.050 | 1,140,973.527 | **6.0449x** |
| Family-3 | `consensus_3` | 10,329,492.694 | 1,184,853.369 | **8.7180x** |
| Family-3 | `unique_a` | 10,011,924.357 | 1,164,677.511 | **8.5963x** |
| Family-5 | `consensus_5` | 17,623,026.629 | 1,344,976.549 | **13.1029x** |
| Family-5 | `quorum_3of5` | 17,543,319.071 | 1,370,045.594 | **12.8049x** |
| Family-5 | `unique_a_5` | 17,627,048.778 | 1,337,873.820 | **13.1754x** |

### Cross-Family Comparative Lift (Normalized)

| Comparison | Ratio |
|---|---:|
| Family-3 vs Family-2 (`F3/F2`) | **1.4823x** |
| Family-5 vs Family-3 (`F5/F3`) | **1.5048x** |
| Family-5 vs Family-2 (`F5/F2`) | **2.2306x** |

Interpretation: GPU benefit grows materially with predicate arity. Moving from 2-array to 5-array predicates more than doubles normalized acceleration in this M1 snapshot.

### Scaling by Index Size (Family Geomean GPU vs CPU)

| Index size | Family-2 | Family-3 | Family-5 |
|---|---:|---:|---:|
| 256K | 1.4086x | 2.3632x | 3.8538x |
| 1M | 4.4957x | 6.9307x | 10.7024x |
| 8M | 11.9855x | 17.2527x | 25.0453x |
| 128M | 15.3267x | 19.8757x | 27.8771x |

This confirms a two-dimensional trend:
1. **Bigger index size** increases GPU advantage.
2. **Higher predicate arity** increases GPU advantage at every size tier.

### Fully Disaggregated Scenario Table (No pooling across size/lf/overlap)

The table below is disaggregated by `index_size × lf × overlap`.  
Each family value is the geomean `CPU/GPU` across predicates within that family **for that exact scenario only**.

| Index size | Occupancy (lf) | Overlap | Family-2 | Family-3 | Family-5 | F3/F2 | F5/F3 | F5/F2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256K | 0.10 | 0.10 | 1.3196x | 2.0989x | 3.6533x | 1.5906x | 1.7406x | 2.7685x |
| 256K | 0.10 | 0.50 | 1.1141x | 1.8560x | 3.6317x | 1.6659x | 1.9567x | 3.2597x |
| 256K | 0.10 | 0.90 | 0.8927x | 2.0421x | 3.3393x | 2.2874x | 1.6352x | 3.7405x |
| 256K | 0.90 | 0.10 | 1.4010x | 2.4340x | 3.7805x | 1.7374x | 1.5532x | 2.6985x |
| 256K | 0.90 | 0.50 | 3.2249x | 3.9296x | 5.2476x | 1.2185x | 1.3354x | 1.6272x |
| 256K | 0.90 | 0.90 | 1.3171x | 2.2891x | 3.7272x | 1.7379x | 1.6282x | 2.8298x |
| 1M | 0.10 | 0.10 | 3.8875x | 5.8034x | 8.6821x | 1.4928x | 1.4961x | 2.2333x |
| 1M | 0.10 | 0.50 | 3.7651x | 5.8989x | 9.6752x | 1.5667x | 1.6402x | 2.5697x |
| 1M | 0.10 | 0.90 | 3.0769x | 5.7365x | 9.6291x | 1.8644x | 1.6786x | 3.1295x |
| 1M | 0.90 | 0.10 | 4.6193x | 6.8377x | 10.2930x | 1.4802x | 1.5053x | 2.2283x |
| 1M | 0.90 | 0.50 | 9.1202x | 12.0219x | 15.7528x | 1.3182x | 1.3103x | 1.7272x |
| 1M | 0.90 | 0.90 | 4.3517x | 6.8659x | 11.4580x | 1.5777x | 1.6688x | 2.6330x |
| 8M | 0.10 | 0.10 | 10.4210x | 15.0011x | 21.8148x | 1.4395x | 1.4542x | 2.0934x |
| 8M | 0.10 | 0.50 | 9.9906x | 14.8178x | 22.7087x | 1.4832x | 1.5325x | 2.2730x |
| 8M | 0.10 | 0.90 | 8.5773x | 15.0550x | 23.4383x | 1.7552x | 1.5568x | 2.7326x |
| 8M | 0.90 | 0.10 | 11.8816x | 17.1828x | 24.1146x | 1.4462x | 1.4034x | 2.0296x |
| 8M | 0.90 | 0.50 | 23.2927x | 27.8482x | 36.4146x | 1.1956x | 1.3076x | 1.5633x |
| 8M | 0.90 | 0.90 | 11.9947x | 16.4686x | 24.2063x | 1.3730x | 1.4699x | 2.0181x |
| 128M | 0.10 | 0.10 | 14.3436x | 19.7080x | 27.0665x | 1.3740x | 1.3734x | 1.8870x |
| 128M | 0.10 | 0.50 | 13.4089x | 18.7030x | 25.0297x | 1.3948x | 1.3383x | 1.8666x |
| 128M | 0.10 | 0.90 | 11.1159x | 18.2312x | 24.6575x | 1.6401x | 1.3525x | 2.2182x |
| 128M | 0.90 | 0.10 | 13.3529x | 19.8677x | 26.4964x | 1.4879x | 1.3336x | 1.9843x |
| 128M | 0.90 | 0.50 | 28.7948x | 25.7872x | 37.9353x | 0.8956x | 1.4711x | 1.3174x |
| 128M | 0.90 | 0.90 | 15.7689x | 17.9064x | 27.9522x | 1.1356x | 1.5610x | 1.7726x |

### Implications and Value

- **Database execution value:** Multi-way predicate filters (joins/intersections/differences over multiple posting lists) are where the GPU path compounds gains, especially for large cardinalities.
- **Inference/rule-engine value:** As predicate logic becomes higher-order (3-way and 5-way consensus/quorum/uniqueness), normalized GPU benefit grows rather than flattening.
- **Practical deployment signal:** Family-2 can approach parity in smallest-edge cases, but Family-3 and Family-5 remain robustly GPU-faster even in their worst measured points.
- **Roadmap guidance:** Prioritize GPU kernels for higher-arity predicate plans first; they deliver the largest and most stable acceleration envelope in this benchmark suite.

## Research-Paper Summary

M1 results show that GPU benefit is operation-dependent and scale-sensitive. For **iteration**, the data exhibits a clean size-driven crossover: GPU underperforms at small index sizes, reaches near parity around 1M, and becomes decisively faster at 8M and 128M. For **greedy search**, the opposite pattern dominates: GPU is generally slower across the matrix, with one notable high-load large-index win (`128M @ 0.90`). This suggests that the GPU path amortizes setup and transfer overhead effectively for throughput-dominant sequential scans (`iter`) but is not yet consistently advantageous for the current greedy lookup path, except in specific high-workload regimes.

## Notes / Validity Considerations

- All comparisons use the same unit (`ns`) and same aggregation (`mean`) from benchmark CSV outputs.
- These are point measurements from the latest dataset snapshot; confidence intervals are not yet reported.
- If this section is promoted to final paper text, add repeated-run statistics and hardware/runtime configuration metadata.
