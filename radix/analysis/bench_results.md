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

## Run 3 — 4×64 group-per-bucket structure (Bucket + FpGroup)

Date: 2026-02-23
Notes: Bucket = 4 × FpGroup (each 64 bytes, `#[repr(C, align(64))]`).
Hash → bucket + group(2 bits) + slot(6 bits) + fp(8 bits). Group+slot extracted as single byte.
Lookup/insert only touch the hash-directed group; overflow to same group in next bucket.
bucket_bits = capacity_bits − 8; bucket_count = 1024 (was 4096).
Mode: `--quick`

### Lookup Hit

| Load | hashbrown | radix_tree | Ratio | vs Run 2 |
|------|-----------|------------|-------|----------|
| tiny | 720 ps | 1.67 ns | 2.3x slower | ~same |
| 1% | 2.73 ns | 2.95 ns | 1.1x slower | −16% slower |
| 25% | 2.81 ns | 4.29 ns | 1.5x slower | −11% slower |
| 50% | 2.79 ns | 7.81 ns | 2.8x slower | −10% slower |
| 75% | 3.25 ns | 12.52 ns | 3.9x slower | −8% slower |

### Lookup Miss

| Load | hashbrown | radix_tree | Ratio | vs Run 2 |
|------|-----------|------------|-------|----------|
| 1% | 1.67 ns | 2.62 ns | 1.6x slower | −12% slower |
| 25% | 2.06 ns | 7.18 ns | 3.5x slower | −7% slower |
| 50% | 2.80 ns | 13.30 ns | 4.8x slower | −9% slower |
| 75% | 8.62 ns | 20.46 ns | 2.4x slower | −9% slower |

### Insert Marginal

| Load | hashbrown | radix_tree | Ratio | vs Run 2 |
|------|-----------|------------|-------|----------|
| 1% | 68.7 µs | 11.3 µs | 6.1x faster | −71% slower |
| 25% | 66.7 µs | 7.54 µs | 8.8x faster | −6% slower |
| 50% | 3.62 µs | 7.85 µs | 2.2x slower | ~same |
| 75% | 69.1 µs | 7.61 µs | 9.1x faster | +7% faster |

## Run 4 — progressive 16-byte NEON chunk walk with rotated preferred_offset

Date: 2026-02-23
Notes: Both `insert` and `contains` use identical progressive walk: rotate match/empty
masks by `preferred_offset`, iterate set bits via `trailing_zeros`, checking preferred
offset position first in *every* chunk. Chunks scanned cyclically from start_chunk.
Insert is now progressive (early-exit on first empty) instead of full-group scan.
Mode: `--quick`

### Lookup Hit

| Load | hashbrown | radix_tree | Ratio | vs Run 3 |
|------|-----------|------------|-------|----------|
| tiny | 705 ps | 4.59 ns | 6.5x slower | −175% slower |
| 1% | 2.85 ns | 8.48 ns | 3.0x slower | −187% slower |
| 25% | 3.00 ns | 8.47 ns | 2.8x slower | −97% slower |
| 50% | 2.82 ns | 7.58 ns | 2.7x slower | +3% faster |
| 75% | 3.19 ns | 8.15 ns | 2.6x slower | +35% faster |

### Lookup Miss

| Load | hashbrown | radix_tree | Ratio | vs Run 3 |
|------|-----------|------------|-------|----------|
| 1% | 1.71 ns | 6.07 ns | 3.6x slower | −132% slower |
| 25% | 2.06 ns | 6.44 ns | 3.1x slower | +10% faster |
| 50% | 2.97 ns | 7.30 ns | 2.5x slower | +45% faster |
| 75% | 9.72 ns | 11.14 ns | 1.1x slower | +46% faster |

### Insert Marginal

| Load | hashbrown | radix_tree | Ratio | vs Run 3 |
|------|-----------|------------|-------|----------|
| 1% | 66.5 µs | 8.22 µs | 8.1x faster | +27% faster |
| 25% | 3.03 µs | 5.47 µs | 1.8x slower | +27% faster |
| 50% | 3.31 µs | 5.03 µs | 1.5x slower | +36% faster |
| 75% | 4.07 µs | 6.22 µs | 1.5x slower | +18% faster |

## Run 5 — single scalar preferred-offset probe + bulk hash-line prefetch

Date: 2026-02-25
Notes: Scalar byte check at `preferred_offset` first; two explicit `prfm` prefetches
for the hash cache lines covering the 16-byte chunk; `preferred_offset` masked out of
the NEON scan via `pref_bit` to avoid duplicate work. Insert uses same pattern.
Mode: `--quick`

### Lookup Hit

| Load | hashbrown | radix_tree | Ratio | vs Run 4 |
|------|-----------|------------|-------|----------|
| tiny | 764 ps | 909 ps | 1.2x slower | +80% faster |
| 1% | 2.66 ns | 2.85 ns | 1.1x slower | +66% faster |
| 25% | 2.85 ns | 3.58 ns | 1.3x slower | +58% faster |
| 50% | 3.12 ns | 5.70 ns | 1.8x slower | +25% faster |
| 75% | 3.48 ns | 8.32 ns | 2.4x slower | −2% slower |

### Lookup Miss

| Load | hashbrown | radix_tree | Ratio | vs Run 4 |
|------|-----------|------------|-------|----------|
| 1% | 1.71 ns | 2.75 ns | 1.6x slower | +55% faster |
| 25% | 1.90 ns | 6.22 ns | 3.3x slower | +3% faster |
| 50% | 2.65 ns | 10.11 ns | 3.8x slower | −38% slower |
| 75% | 9.79 ns | 12.94 ns | 1.3x slower | −16% slower |

### Insert Marginal

| Load | hashbrown | radix_tree | Ratio | vs Run 4 |
|------|-----------|------------|-------|----------|
| 1% | 30.6 µs | 26.6 µs | 1.2x faster | noisy |
| 25% | 3.17 µs | 3.85 µs | 1.2x slower | +30% faster |
| 50% | 3.45 µs | 4.70 µs | 1.4x slower | +7% faster |
| 75% | 4.00 µs | 5.86 µs | 1.5x slower | +6% faster |
