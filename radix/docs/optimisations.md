# Radix Tree — Optimisation Log

Tracked performance issues identified from benchmark comparison between
`m0_hashbrown` (SwissTable baseline) and `RadixTree` (M2 single-group
SIMD implementation). Benchmark data from 2026-02-22 runs.

---

## P0 — Early Termination on Lookup Miss

**Status**: pending

**Problem**: `lookup` iterates all 64 probe rounds for every miss. Each
round performs a SIMD scan of a different bucket (64 bytes). A single
lookup miss touches up to 64 × 64 = 4,096 bytes of fingerprint data.
Result: lookup_miss is 600–850× slower than hashbrown.

**Root cause**: the code does not exploit the no-deletion invariant.
Since elements are never removed, if a bucket at round R currently has
any empty slot, it was *never* full during the table's lifetime.
Insertion would never have overflowed past round R. Therefore, if the
key is not found at round R in a non-full bucket, it cannot exist at
any round > R.

**Fix**: after scanning candidates at round R, if `empty_mask != 0`,
return false immediately.

**Expected impact**: ~64× improvement in lookup_miss. At LF 0.50 most
buckets are half-empty, so nearly all misses terminate at round 0.

---

## P1 — NEON movemask via Narrowing Intrinsics

**Status**: pending

**Problem**: `mask_from_cmp16` stores a NEON vector to memory, then
loops over 16 bytes to rebuild a bitmask. This generates a store +
16 loads + 16 shifts + 16 ORs.

**Fix**: use `vshrn_n_u16` / `vget_lane_u64` narrowing pattern to
extract the bitmask in 2–3 NEON instructions instead of a scalar loop.

**Expected impact**: ~2× improvement in SIMD scan throughput.

---

## P2 — Insert Short-Circuit on Empty Preferred Slot

**Status**: pending

**Problem**: insert always performs a full SIMD scan before placing,
even when the preferred slot is empty (common at low load). The SIMD
scan is needed for deduplication but is wasted when the slot is clearly
available.

**Fix**: before the SIMD scan, check
`self.fingerprints[preferred_idx] == RADIX_FP_EMPTY`. If empty, place
directly (no duplicate can exist at this round because the bucket was
never full enough to push an earlier copy elsewhere). Only fall through
to the full scan when the preferred slot is occupied.

**Expected impact**: ~2–3× improvement in insert at low load factors.

---

## P3 — Keep SIMD Scan Off the Hot Inline Path

**Status**: pending

**Problem**: adding NEON intrinsics regressed lookup_hit from 7.6 ns
to 12.0 ns at LF 0.50 (58% regression). The preferred-slot fast path
(1 byte compare + 1 u64 compare) does not use SIMD, but the larger
function body likely hurt inlining or icache.

**Fix**: mark the SIMD scan fallback as `#[cold]` or move it to a
separate `#[inline(never)]` function so the hot preferred-slot path
stays compact in the caller's instruction stream.

**Expected impact**: restore lookup_hit from ~12 ns back toward ~7 ns.

---

## P4 — Fresh-Bit Overflow Instead of Rotation

**Status**: pending (deferred to M4 milestone)

**Problem**: `hash_parts_for_round` derives each round by rotating the
mixed hash by 1 bit. With `bucket_bits=12`, consecutive rounds produce
highly correlated bucket indices, causing overflow clustering and
redundant probes.

**Fix**: consume independent hash segments per round (as specified in
the M4 architecture), or re-mix with the round number to decorrelate
rounds.

**Expected impact**: fewer rounds needed at high load, better tail
latency. Full quantification deferred to M4 benchmarks.

---

## Benchmark Reference (2026-02-22, capacity_bits=18)

### lookup_hit (mean ns)

| LF   | hashbrown | RadixTree (scalar) | RadixTree (SIMD) |
|------|-----------|--------------------|--------------------|
| 0.50 | 3.06      | 7.60               | 12.04              |
| 0.75 | 3.08      | 11.81              | 16.25              |
| 0.90 | 3.19      | 14.63              | 20.36              |
| 0.95 | 3.21      | 16.93              | 22.64              |
| 0.99 | 3.19      | 21.89              | 24.32              |

### lookup_miss (mean ns)

| LF   | hashbrown | RadixTree (scalar) | RadixTree (SIMD) |
|------|-----------|--------------------|--------------------|
| 0.50 | 2.04      | 574                | 1,740              |
| 0.75 | 2.25      | 874                | 1,630              |
| 0.90 | 2.65      | 1,508              | 1,716              |
| 0.95 | 2.67      | 1,951              | 1,787              |
| 0.99 | 2.78      | 2,398              | 1,645              |

### insert_marginal (mean ns / 128-element batch)

| LF   | hashbrown | RadixTree (scalar) | RadixTree (SIMD) |
|------|-----------|--------------------|--------------------|
| 0.50 | 2,267     | 5,093              | 14,445             |
| 0.99 | 2,502     | 97,763             | 79,777             |
