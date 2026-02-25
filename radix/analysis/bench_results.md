# Benchmark Results

Capacity: 262,144 raw slots (both implementations).
Hashbrown constructed with `with_capacity(229376)` to match raw allocation.

## Run 1 — consolidated single-function contains_v3 (no cold split)

Date: 2026-02-23
Notes: `contains_v3` inlined as single function (no `#[cold]`/`#[inline(never)]` split).
Mode: `--quick`

### Lookup Hit

| Load | hashbrown | radix_tree | Ratio |
|------|-----------|------------|-------|
| tiny | 751 ps | 2.11 ns | 2.8x slower |
| 1% | 2.51 ns | 3.04 ns | 1.2x slower |
| 25% | 2.67 ns | 4.30 ns | 1.6x slower |
| 50% | 2.84 ns | 7.20 ns | 2.5x slower |
| 75% | 3.26 ns | 9.59 ns | 2.9x slower |

### Lookup Miss

| Load | hashbrown | radix_tree | Ratio |
|------|-----------|------------|-------|
| 1% | 1.68 ns | 3.04 ns | 1.8x slower |
| 25% | 1.87 ns | 6.39 ns | 3.4x slower |
| 50% | 2.58 ns | 10.18 ns | 3.9x slower |
| 75% | 9.57 ns | 14.22 ns | 1.5x slower |

### Insert Marginal

| Load | hashbrown | radix_tree | Ratio |
|------|-----------|------------|-------|
| 1% | 37.1 µs | 6.4 µs | 5.8x faster |
| 25% | 27.4 µs | 7.4 µs | 3.7x faster |
| 50% | 3.1 µs | 23.2 µs | 7.5x slower |
| 75% | 28.3 µs | 8.3 µs | 3.4x faster |

## Run 2 — scalar byte-by-byte group scan (no SIMD)

Date: 2026-02-23
Notes: Group scan replaced with per-byte iteration and early check/fail. SIMD variant commented out.
Mode: `--quick`

### Lookup Hit

| Load | hashbrown | radix_tree | Ratio | vs Run 1 |
|------|-----------|------------|-------|----------|
| tiny | 694 ps | 1.64 ns | 2.4x slower | +22% faster |
| 1% | 2.50 ns | 2.54 ns | 1.0x | +16% faster |
| 25% | 2.71 ns | 3.86 ns | 1.4x slower | +10% faster |
| 50% | 2.83 ns | 7.08 ns | 2.5x slower | ~same |
| 75% | 3.33 ns | 11.64 ns | 3.5x slower | -21% slower |

### Lookup Miss

| Load | hashbrown | radix_tree | Ratio | vs Run 1 |
|------|-----------|------------|-------|----------|
| 1% | 1.67 ns | 2.34 ns | 1.4x slower | +23% faster |
| 25% | 1.87 ns | 6.71 ns | 3.6x slower | -5% slower |
| 50% | 2.57 ns | 12.21 ns | 4.8x slower | -20% slower |
| 75% | 9.11 ns | 18.82 ns | 2.1x slower | -32% slower |

### Insert Marginal

| Load | hashbrown | radix_tree | Ratio |
|------|-----------|------------|-------|
| 1% | 39.0 µs | 6.6 µs | 5.9x faster |
| 25% | 3.1 µs | 7.1 µs | 2.3x slower |
| 50% | 26.9 µs | 7.7 µs | 3.5x faster |
| 75% | 28.8 µs | 8.2 µs | 3.5x faster |
