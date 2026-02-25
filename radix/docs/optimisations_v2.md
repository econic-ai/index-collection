# Radix Tree v4 — Drop-in Instruction Set

## Context

The v3 radix tree benchmarks revealed that our 64-slot bucket with 8-byte group scanning achieves parity with hashbrown at low load but degrades at 50-75% occupancy. The per-group operation cost is identical to hashbrown — both use NEON 8-byte loads — so the divergence is in how many groups are scanned before the answer is found.

Analysis of the (preferred_slot, fp8) pair revealed that it functions as a 14-bit probabilistic key stored in 8 bits. The position encodes 6 bits of hash entropy for free — implicit in the array index, zero storage cost. This is structurally identical to a quotient filter, and it outperforms Bloom and cuckoo filters at equivalent storage. The full coordinate tuple (bucket + slot + fp8) yields 26 bits of discrimination from a single stored byte.

This information density property only holds if the positional encoding is preserved through the insert path. The v3 insert used linear displacement from the preferred slot, which destroyed positional information in every group beyond the first. The changes below restructure the bucket layout, the SIMD scan, and the insert path to preserve the (position, fp8) pair as a meaningful composite key across all groups.

We also introduce a group-of-groups structure — 4 × 64-byte groups per bucket — which provides 2 additional bits of hash-directed placement above the intra-group level, expanding the hierarchical discrimination while keeping each group aligned to a cache line. On overflow, the scan advances to the same hash-directed group in the next bucket, not a different group within the same bucket. The group coordinate is fixed by the hash and preserved through overflow — only the bucket is uncertain. This keeps the (group, slot, fp8) portion of the composite key valid for displaced elements.

---

## Change 1: Cache-line-aligned bucket arrays

### Why

The flat `Vec<u8>` with manual arithmetic works but doesn't guarantee that bucket boundaries coincide with cache line boundaries. If bucket 0 starts at an address 48 bytes into a cache line, every bucket straddles two cache lines. NEON loads within a bucket are safe (unaligned loads have no penalty on AArch64), but a bucket that straddles two cache lines means two cache fetches instead of one on first access.

`Vec<Vec<u8>>` was considered and rejected — each inner Vec is a separate heap allocation with 24 bytes of overhead, scattered across the heap. Sequential overflow becomes a pointer chase instead of a stride.

`Vec<[u8; 64]>` is contiguous but `[u8; 64]` has alignment of 1 — no cache line guarantee.

### Implementation

A newtype wrapper with forced alignment for the group, and a bucket struct containing four groups:

```rust
#[repr(C, align(64))]
pub struct FpGroup([u8; 64]);

#[repr(C)]
pub struct Bucket {
    pub groups: [FpGroup; 4],
}
```

Each `FpGroup` is exactly one cache line. The `repr(align(64))` forces the allocator to provide a 64-byte-aligned base address. Every group sits on a cache line boundary by construction. The `Bucket` inherits alignment from its first field — no separate annotation needed, and `repr(C)` guarantees the four groups are contiguous (256 bytes total).

The fingerprint arena becomes:

```rust
pub fingerprints: Vec<Bucket>,
```

Access pattern: `self.fingerprints[bucket].groups[group].0[slot]` — the compiler resolves this to the same arithmetic as the flat array, but alignment is guaranteed and the type system enforces the bucket/group/slot hierarchy.

### Considerations

- The hash arena (`Vec<u64>`) does not need this treatment — u64 has natural 8-byte alignment, and hash confirmations are point reads, not SIMD scans.
- The `FpGroup` type should derive `Clone` and implement `Default` (zeroed) for clean initialisation.

---

## Change 2: 4 × 64-byte groups per bucket

### Why

The current design has 64 slots per bucket with no sub-bucket structure for hash-directed group selection. The paper's analysis identified that group direction above the SIMD floor has measurable value — 2 bits of hash entropy selecting one of four groups avoids three unnecessary cache line loads.

With 4 groups of 64 slots per bucket, each group is one cache line. The hash directs to the preferred group. If the element isn't in that group, the scan advances to the same group in the next bucket — not to a different group in the same bucket. At low load, most lookups resolve in the first group touched — one cache line. At high load, overflow traverses the same group across successive buckets, each a single cache line load.

### Implementation

The bucket becomes a logical unit of 4 groups:

```rust
pub const GROUPS_PER_BUCKET: usize = 4;
pub const GROUP_SLOTS: usize = 64;
pub const BUCKET_SLOTS: usize = GROUPS_PER_BUCKET * GROUP_SLOTS; // 256

pub const GROUP_BITS: u8 = 2;   // 2 bits select 1 of 4 groups
pub const SLOT_BITS: u8 = 6;    // 6 bits select 1 of 64 slots within group
```

Hash bit partitioning (left to right from the mixed hash):

```
[bucket_bits | group_bits (2) | slot_bits (6) | fp_bits (8) | overflow... ]
```

The `HashParts` struct expands:

```rust
pub struct HashParts {
    pub bucket: usize,
    pub group: usize,        // 0..3
    pub preferred_slot: usize, // 0..63 within group
    pub fingerprint: u8,
}
```

Group index extraction:

```rust
let group = ((h >> group_shift) & 0x3) as usize;
```

The fingerprint arena is now `Vec<Bucket>` (see Change 1). Accessing a specific group: `self.fingerprints[bucket].groups[group].0[slot]`.

### Overflow direction

On overflow, the scan advances to the **same group in the next bucket**, not the next group in the same bucket. The group bits are hash-derived — they carry 2 bits of entropy. Moving to a different group within the same bucket discards those bits. Moving to the same group in the next bucket preserves them.

The scan order on lookup:
1. Group *g* in bucket *b* (preferred)
2. Group *g* in bucket *b+1*
3. Group *g* in bucket *b+2*
4. ...until an empty slot is found in group *g* of some bucket

The group coordinate is fixed by the hash. Only the bucket advances. This means the (group, slot, fp8) portion of the coordinate tuple remains valid for displaced elements — only the bucket is uncertain. For the distributed probabilistic case, a remote node computing preferred coordinates gets 3 of 4 layers right even for overflowed elements.

Insert follows the same path: when group *g* in bucket *b* is full, try group *g* in bucket *b+1*, applying the preferred-offset-first rule (Change 7) in each.

### Considerations

- Total capacity per bucket is now 256 slots. With `capacity_bits = 18`, that gives 1,024 buckets of 256 slots each. Adjust the bucket_bits calculation: `bucket_bits = capacity_bits - GROUP_BITS - SLOT_BITS`.
- The hash arena similarly restructures: 256 u64s per bucket, grouped in 64-element regions aligned with each FpGroup.
- Only the hash-directed group is ever accessed within a bucket. The other three groups in the same bucket belong to different hash partitions. This means effective per-group load factor is the global load factor — elements are distributed uniformly across all groups across all buckets.

---

## Change 3: 16-byte NEON scan as primary lookup

### Why

NEON 128-bit registers can load and compare 16 bytes in a single operation via `vld1q_u8` / `vceqq_u8`. The v3 code used 8-byte groups (`vld1_u8`), requiring 8 operations to cover a 64-slot group. With 16-byte loads, the same group is covered in 4 operations — half the loop iterations. The mask extraction is wider (128-bit comparison result packed to 16-bit mask) but the reduction in iterations should more than compensate.

### Implementation

```rust
const NEON_WIDTH: usize = 16;
const NEON_GROUPS_PER_FP_GROUP: usize = GROUP_SLOTS / NEON_WIDTH; // 4
```

The scan loop for one FpGroup (64 bytes = 4 × 16-byte NEON loads):

```rust
#[cfg(target_arch = "aarch64")]
unsafe {
    let zero = vdupq_n_u8(0);
    let target = vdupq_n_u8(fp);

    for ni in 0..NEON_GROUPS_PER_FP_GROUP {
        // Start from the NEON chunk containing the preferred slot
        let chunk = (start_chunk + ni) & (NEON_GROUPS_PER_FP_GROUP - 1);
        let p = fp_ptr.add(chunk * NEON_WIDTH);

        let vals = vld1q_u8(p);
        let empty_cmp = vceqq_u8(vals, zero);
        let match_cmp = vceqq_u8(vals, target);

        let empty_mask = pack_16_to_mask(empty_cmp);  // 16-bit mask
        let match_mask = pack_16_to_mask(match_cmp);   // 16-bit mask

        // Process matches and empties...
    }
}
```

The `start_chunk` is derived from the preferred slot: `preferred_slot / NEON_WIDTH`.

### Considerations

- Also benchmark the 8-byte variant for comparison. Record results for both.
- The 16-bit mask packing from a 128-bit NEON comparison result requires extracting the high bit of each byte. The existing `pack_msbs` operates on u64; a 16-byte variant needs two calls (high and low 64-bit lanes) combined, or a dedicated 16-byte pack function.
- Within each 16-byte chunk, preferred offset processing uses the 4-bit offset (`preferred_slot & 15`).

---

## Change 4: Unified walk from preferred offset

### Why

The SIMD scan produces two masks: `match_mask` (fp8 matches) and `empty_mask` (vacant slots). The question is how to iterate them.

The naive approach — `trailing_zeros` on the match mask to jump to set bits — has two problems. First, it doesn't respect displacement ordering. It might confirm a false positive at position 2 before checking position 9, even though the element was displaced forward from position 9. Second, empty detection happens after all matches are exhausted — meaning you confirm false positives at positions *beyond* an empty slot, where the element can't possibly be.

A linear walk from the preferred offset solves both. It visits positions in displacement order, interleaving match and empty checks, so the first empty slot terminates the scan before wasting hash confirmations on unreachable false positives.

Combining both masks into a single check further tightens the loop. At high load, most slots are occupied by non-matching, non-empty elements — irrelevant positions. Without the combined mask, each irrelevant slot costs two branches (check match, check empty). With it, one branch skips past all irrelevant positions.

### Implementation

After the NEON compare produces `match_mask` and `empty_mask` for a chunk:

```rust
let combined = match_mask | empty_mask;
let mut pos = preferred_offset;
loop {
    let bit = 1 << pos;
    if combined & bit != 0 {
        if match_mask & bit != 0 {
            if *hash_ptr.add(chunk_base + pos) == id {
                return true;
            }
        } else {
            break; // empty — element can't be beyond this point
        }
    }
    pos = (pos + 1) & (NEON_WIDTH - 1);
    if pos == preferred_offset { break; }
}
```

Match is checked before empty, biasing toward the common case where the element is at its preferred offset. The `combined` OR is computed once before the loop — one instruction — and saves one branch per non-interesting slot.

### Considerations

- At low load: preferred offset is likely empty or holds the element. One iteration, one or two branches. Minimal cost.
- At moderate load: element displaced a few slots forward. Walk terminates at the element or at the first empty, whichever comes first. False positives beyond the empty are never touched.
- At high load with fp8: ~74% of occupied slots in a 16-position chunk are non-matching non-empty. The combined mask lets these skip with one branch instead of two — roughly 12 of 16 positions benefit.
- The walk respects the insert invariant: the element lies between the preferred offset and the first empty slot (wrapping). `trailing_zeros` ignores this invariant. The walk enforces it.
- This replaces both the preferred-bit pre-check and the trailing_zeros loop with a single unified iteration. No separate fast path, no fallback — one code path.

---

## Change 5: Uniform loop body across all groups and chunks

### Why

The v3 code had branching logic between the first group (with preferred-slot special casing) and subsequent groups. Branch misprediction on Apple Silicon costs 12-15 cycles — more than the entire NEON scan of a chunk. A uniform loop body eliminates this risk.

The unified walk from preferred offset (Change 4) runs identically in every NEON chunk. In the primary chunk, it starts from the hash-directed preferred offset. In non-primary chunks, it starts from the same offset — which is statistically grounded by the insert change (Change 7) that tries this offset first in every group. The walk handles match and empty detection in the same loop, so there is no structural difference between chunks.

The cost of uniformity is a small number of redundant hash confirmations in non-primary positions (~0.2% false positive rate per slot). The benefit is zero branching, one code path through the instruction cache, and predictable pipeline behaviour.

### Implementation

The scan loop advances through buckets, always accessing the same hash-directed group:

```rust
for bi in 0..max_overflow_buckets {
    let bucket = (preferred_bucket + bi) & bucket_mask;
    let group_base = &self.fingerprints[bucket].groups[preferred_group];
    let hash_base = /* corresponding hash arena region */;

    for ni in 0..NEON_GROUPS_PER_FP_GROUP {
        let chunk = (start_chunk + ni) & (NEON_GROUPS_PER_FP_GROUP - 1);
        // ... NEON load, compare, then unified walk from preferred offset (Change 4) ...
        // ... walk handles match confirmation and empty termination in one loop ...
    }

    // If any NEON chunk in this group had an empty slot, element can't
    // have been displaced beyond this group. Terminate.
}
```

No `if bi == 0` branches. No special first-iteration logic. The group index is constant — only the bucket advances.

### Considerations

- `start_chunk` is computed once from `preferred_slot / NEON_WIDTH` and used in every bucket iteration. The preferred group and preferred slot within it are constant — derived from the hash, not from overflow state.
- Early termination on empty: if any NEON chunk within a group contains an empty slot, the element cannot have been displaced beyond that point. This terminates the inner loop. Across buckets, a group with any empty slot means the element was never displaced to the next bucket's group — terminate the outer loop.
- The group index never changes during overflow. Only the bucket index advances. This is the key structural difference from v3, which overflowed linearly through all slots in a flat bucket.

---

## Change 6: Benchmark both 8-byte and 16-byte NEON widths

### Why

The 16-byte NEON path halves the number of loop iterations per group (4 instead of 8) but each iteration has heavier mask extraction (packing 16 bytes to a 16-bit mask vs 8 bytes to an 8-bit mask). Which wins depends on the relative cost of loop overhead vs mask packing on the target hardware (Apple Silicon).

### Implementation

Feature-flag the NEON width so both can be benchmarked without code duplication:

```rust
#[cfg(feature = "neon16")]
const NEON_WIDTH: usize = 16;

#[cfg(not(feature = "neon16"))]
const NEON_WIDTH: usize = 8;
```

Run benchmarks at each load factor with both feature flags. Record:
- Lookup hit latency at tiny, 1%, 25%, 50%, 75% load
- Lookup miss latency at the same load factors
- Insert marginal cost

Compare against each other and against hashbrown baseline.

### Considerations

- The 8-byte path uses `vld1_u8` (64-bit D-register), the 16-byte path uses `vld1q_u8` (128-bit Q-register). Both are single-cycle loads from L1. The difference is entirely in comparison and mask extraction.
- The mask packing for 16 bytes requires handling both 64-bit lanes of the Q-register result. The existing `pack_msbs` function operates on a single u64 lane — two calls are needed per NEON compare, or a dedicated 16-byte packing function.

---

## Change 7: Insert uses preferred offset in every group

### Why

The (position, fp8) pair carries 14 bits of discrimination, but only if the element is actually placed at its preferred offset. In v3, insert places at the nearest empty slot after the preferred slot via linear scan. If the element overflows to a non-primary group, it lands at whatever slot happens to be empty — no preference. This destroys the positional information in every group beyond the first.

If insert tries the preferred offset first in every group, then the unified walk starting from that offset (Change 4) has statistical basis across the entire overflow chain, not just the primary group. At load factor α, the probability that the preferred offset in a non-primary group was empty at insertion time is approximately (1 - α):

| Load | P(element at preferred offset in overflow group) |
|------|--------------------------------------------------|
| 25%  | ~75%                                             |
| 50%  | ~50%                                             |
| 75%  | ~25%                                             |

When the element is at the preferred offset, the first hash confirmation in each group is a true positive — the 14-bit composite key resolves immediately. When it isn't, the cost is at most one wasted hash confirmation (if another element has matching fp8 at that offset, ~1/255 probability).

### Implementation

In the insert path, when scanning a group for an empty slot. The insert overflows to the same group in the next bucket, matching the lookup path:

```rust
for bi in 0..max_overflow_buckets {
    let bucket = (preferred_bucket + bi) & bucket_mask;
    let group = &mut self.fingerprints[bucket].groups[preferred_group];

    // Dedup: scan group for existing fp matches, confirm against full hash
    // ... (SIMD scan for match_mask, iterate and confirm) ...

    // Try preferred offset first
    if group.0[preferred_offset] == RADIX_FP_EMPTY {
        group.0[preferred_offset] = fp;
        self.hashes[/* corresponding index */] = id;
        self.len += 1;
        return true;
    }

    // Fall back to linear scan for next empty slot within the group
    // (or SIMD scan for empty mask, pick first set bit)
    // ...

    // Group full — advance to same group in next bucket
}
```

This applies in every bucket the insert touches — the preferred offset is tried first in each, preserving positional information density across the full overflow chain.

### Considerations

- This is a small change in the insert path but it has structural consequences: the element's position now carries hash-derived information across every bucket the overflow chain touches, making the fingerprint array a denser information structure.
- Dedup must still scan all fp-matching slots in the group before placing. The preferred offset is checked for emptiness *after* dedup confirms the element isn't already present.
- The linear fallback after the preferred offset check scans forward from `preferred_offset + 1`, wrapping within the 64-slot group. Or use SIMD to find the nearest empty slot. Either is fine — the preferred offset attempt is the important part.
- Because overflow advances to the same group in the next bucket, the preferred offset is the same in every iteration. The insert and lookup paths are symmetric: both fix the group and offset, advancing only the bucket.

---

## Benchmark Protocol

Each change is applied incrementally and benchmarked independently against the hashbrown baseline. Results are recorded in the benchmark results document with the following columns:

| Change | Load | hashbrown | radix_tree | Ratio | vs Previous |
|--------|------|-----------|------------|-------|-------------|

Changes are applied in this order:
1. Aligned bucket and group arrays (structural, establishes cache-line-aligned `Bucket` / `FpGroup` types)
2. 4 × 64-byte group structure with same-group-next-bucket overflow (structural, changes hash partitioning and overflow direction)
3. 16-byte NEON scan (swap in, benchmark, compare with 8-byte)
4. Unified walk from preferred offset with combined mask (replaces trailing_zeros iteration)
5. Uniform loop body (remove branching)
6. Preferred offset in insert (preserves positional information across overflow)

Each benchmark run uses `--quick` mode at load factors: tiny, 1%, 25%, 50%, 75%. Record lookup hit, lookup miss, and insert marginal.

If any change degrades performance, record the regression and analyse before proceeding. Not every change will improve every metric — the goal is to understand the contribution of each.