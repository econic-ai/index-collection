# Radix Hash Index — Benchmark Specifications

---

## m1: GPU-Accelerated Array Operations

GPU benchmarks targeting full-array contiguous operations where the Apple Silicon GPU's memory bandwidth dominates CPU NEON at scale. All m1 operations share the same access profile as `iter`: sequential, coalesced, uniform work per element, no early exit. Unified memory — both CPU and GPU paths read the same physical bytes.

Existing m1 benchmarks (`iter`, `contains_greedy`) establish the crossover profile: GPU slower at ≤256K slots, near parity at 1M, 3-7× faster at 8M-128M.

---

### m1.1 — Fingerprint Diff

**Question:** "What changed between my copy of this node's fingerprint array and the current version?"

**Context:** In a distributed system, each node caches fp8 arrays from remote nodes for probabilistic routing. When a remote node mutates its index, the local cache becomes stale. The diff identifies which positions changed, producing a sparse update set. This is the differential sync primitive from the paper's section 8.

**Operation:**

Given two fp8 arrays of identical geometry (same capacity), produce a bitmask of positions where they differ.

```
fn fingerprint_diff(fp_a: &[u8], fp_b: &[u8]) -> BitVec
```

Per-element logic:
```
diff[pos] = fp_a[pos] != fp_b[pos]
```

**CPU implementation:**
NEON-accelerated pass. Load 16 bytes from each array, `veorq_u8` (XOR), compare result against zero via `vceqq_u8`, invert, pack to bitmask. Sequential, cache-friendly, full pass.

**GPU implementation:**
Metal compute kernel. One thread per position. XOR + compare, write bit to output buffer. Standard parallel map — no compaction, no atomics, no divergence.

```metal
kernel void fingerprint_diff(
    const device uchar* fp_a   [[buffer(0)]],
    const device uchar* fp_b   [[buffer(1)]],
    device uint*        output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint word_idx = gid / 32;
    uint bit_idx  = gid % 32;
    bool differs  = (fp_a[gid] != fp_b[gid]);
    if (differs) {
        atomic_fetch_or_explicit(
            (device atomic_uint*)&output[word_idx],
            (1u << bit_idx),
            memory_order_relaxed
        );
    }
}
```

**Benchmark parameters:**
- Sizes: 256K, 1M, 8M, 128M
- Mutation rates: 1%, 10%, 50% of positions differ (controls output density, not scan cost — full pass regardless)
- Metric: time per diff operation (ns)
- Compare: CPU NEON vs GPU Metal

**Expected profile:** Same crossover as `iter`. Full sequential coalesced pass, uniform work per element. Mutation rate should not significantly affect latency (the scan touches every position regardless), but verify this — output write patterns differ with density.

**Compares against (for paper narrative):**
- CPU NEON XOR pass (direct m1 comparison)
- Traditional approach: exchange full keysets or maintain separate change logs. The diff operates on the structure that already exists for primary indexing — zero auxiliary storage.

---

### m1.2 — Set Intersection (Approximate)

**Question:** "Which keys do these two nodes probably share?"

**Context:** Two nodes each hold a fragment of a distributed graph. Without exchanging full keysets, we want to estimate the shared keyset. Each node's fp8 array is a positional summary of its contents. Comparing co-indexed positions identifies probable matches — same position, same non-empty fingerprint implies the same key with high probability (FPR ≈ 1/255 per co-occupied slot, much lower when full coordinate tuple considered).

This is approximate private set intersection without cryptographic overhead. No keys are disclosed.

**Operation:**

Given two fp8 arrays of identical geometry, find all positions where both arrays hold the same non-empty fp8 value. Return the matching positions (stream compaction) and the match count (cardinality).

```
fn set_intersection(fp_a: &[u8], fp_b: &[u8]) -> (Vec<usize>, usize)
```

Per-element logic:
```
match[pos] = fp_a[pos] != 0 && fp_b[pos] != 0 && fp_a[pos] == fp_b[pos]
```

**Cardinality as secondary output:**
The compacted match set yields an exact count. This count, divided by the total occupied positions, gives a **similarity ratio** — the estimated fraction of keys shared between the two nodes. This feeds replication priority decisions: nodes with low overlap need sync; nodes with high overlap are already consistent.

```
similarity = |intersection| / min(|occupied_a|, |occupied_b|)
```

Cardinality is not a separate benchmark — it's a free reduction on the intersection output. Report it alongside the intersection result as a secondary metric.

**CPU implementation:**
NEON-accelerated pass. Load 16 bytes from each array. Compare both against zero (occupied mask), AND the occupied masks, compare fp values via `vceqq_u8`, AND with co-occupied mask. Pack to bitmask, collect matching positions. Running popcount for cardinality.

**GPU implementation:**
Metal compute kernel. Parallel predicated compare + stream compaction via atomic counter.

```metal
kernel void set_intersection(
    const device uchar* fp_a     [[buffer(0)]],
    const device uchar* fp_b     [[buffer(1)]],
    device uint*        matches  [[buffer(2)]],
    device atomic_uint* count    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uchar a = fp_a[gid];
    uchar b = fp_b[gid];
    if (a != 0 && b != 0 && a == b) {
        uint idx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        matches[idx] = gid;
    }
}
```

Note: atomic counter for compaction produces unordered output. If ordered output is needed, use a two-pass approach (predicate pass + prefix sum + scatter). For cardinality-only mode, skip the matches buffer entirely and just accumulate the count.

**Benchmark parameters:**
- Sizes: 256K, 1M, 8M, 128M
- Occupancy of each array: 0.50, 0.90
- Overlap: 10%, 50%, 90% of keys shared between the two arrays
- Metric: time per intersection operation (ns), cardinality (count), similarity ratio
- Compare: CPU NEON vs GPU Metal

**Expected profile:** Same crossover as `iter` and `fingerprint_diff` — full sequential coalesced pass. Stream compaction adds minor overhead from atomic writes, proportional to match count. High overlap + high occupancy = more atomic contention, but the scan dominates.

**Compares against (for paper narrative):**
- Sort-merge intersection on full keysets: O(n log n), requires exchanging and sorting actual keys
- Bloom-filter-based intersection: build Bloom on set A, probe all of set B against it. k reads per probe, requires maintaining a separate Bloom filter per node
- Positional intersection: one parallel pass, one comparison per position, zero key disclosure, zero auxiliary storage

---

### m1.3 — Multi-Set Predicate Evaluation (Bayesian Model)

**Question:** "Given N fingerprint arrays from different nodes, what can we infer about the distribution of a key across the network — and can we do it in a single parallel pass?"

**Context:** m1.2 compares two arrays with a fixed predicate (`a != 0 && b != 0 && a == b`). In a real distributed system, a coordinator holds cached fp8 summaries from multiple nodes and needs to answer compound questions: "Which keys exist on A but not on B or C?" (replication gap), "Which keys exist on all of A, B, and C?" (consensus set), "Which keys exist on exactly one node?" (unique ownership). These are set-algebraic predicates over N arrays evaluated at every position.

The algebraic predicates define the *operation*. The Bayesian model defines the *interpretation*: each positional match is not a certainty but a probabilistic signal with a characterisable false positive rate. Composing multiple predicates compounds the error model — the probability that position `p` satisfies the predicate by coincidence depends on the number of arrays, the occupancy of each, and the fp8 collision rate.

**Algebraic predicates (2-array):**

| Predicate | Semantics | Per-position expression |
|---|---|---|
| A ∩ B | Shared keys | `a != 0 && b != 0 && a == b` |
| A \ B | In A, not B | `a != 0 && (b == 0 \|\| a != b)` |
| B \ A | In B, not A | `b != 0 && (a == 0 \|\| a != b)` |
| A △ B | Symmetric diff | `(a != 0 \|\| b != 0) && a != b` |

**Algebraic predicates (3-array):**

| Predicate | Semantics | Per-position expression |
|---|---|---|
| A ∩ B ∩ C | Consensus | `a != 0 && a == b && a == c` |
| A \ (B ∪ C) | Unique to A | `a != 0 && a != b && a != c` |
| (A ∩ B) \ C | Shared by A,B but not C | `a != 0 && a == b && a != c` |
| Exactly one | Unique ownership | `(a != b && a != c && b != c) && (a != 0 \|\| b != 0 \|\| c != 0)` |

Every predicate is a per-element boolean expression. The scan shape is identical to m1.2: linear, coalesced, uniform work per thread. The only variable is the predicate body.

**GPU execution model — predicate as flat mask:**

Rather than compiling a separate kernel per predicate, the predicate can be encoded as a lookup table. For N=2 arrays, each position produces a 2-bit key `(a_occupied, b_occupied, a_eq_b)` — 8 possible states. A single `uint8` mask encodes which states satisfy the predicate. The kernel evaluates the state, tests the mask bit, and emits the result. One kernel, any 2-array predicate.

For N=3, the state space is larger but still bounded: occupancy of each array (3 bits) plus pairwise equality (3 bits for ab, ac, bc) = 64 states. A `uint64` mask encodes the predicate. Same kernel structure, same dispatch, any 3-array predicate.

```metal
// Generalised 2-array predicate kernel
kernel void set_predicate_2(
    const device uchar* fp_a     [[buffer(0)]],
    const device uchar* fp_b     [[buffer(1)]],
    device uint*        matches  [[buffer(2)]],
    device atomic_uint* count    [[buffer(3)]],
    constant uchar&     mask     [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uchar a = fp_a[gid];
    uchar b = fp_b[gid];
    uint state = (uint(a != 0) << 2) | (uint(b != 0) << 1) | uint(a == b);
    if ((mask >> state) & 1) {
        uint idx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        matches[idx] = gid;
    }
}
```

This eliminates kernel proliferation. Adding a new predicate is a one-byte mask change, not a new shader compilation.

**Bayesian interpretation — precision and recall:**

Each positional match is a probabilistic signal, not a certainty. The error model has two independent dimensions: **precision** (when the predicate fires, how likely is it correct?) and **recall** (of all true matches, how many does the predicate detect?). These map to FPR and FNR respectively, and they have different sources and different scaling behaviour.

*Precision (false positive rate):*

A false positive occurs when the predicate fires at a position where the underlying keys are actually different but the fp8 values happen to match. For non-zero fp8 values drawn from 255 possible values:

- 2-array intersection: FPR = 1/255 ≈ 0.39% per co-occupied position
- 3-array consensus: FPR = (1/255)² ≈ 0.0015% — each additional confirming array multiplicatively reduces uncertainty
- N-array consensus: FPR = (1/255)^(N-1)

This is the strong result. When the predicate fires across multiple arrays, the probability of coincidence drops exponentially. A 3-array consensus match is 255× more reliable than a 2-array match. This is genuinely different from ANDing independent Bloom filters, where each filter has its own fixed FPR — here the signals share structural geometry, so positional agreement across independent arrays is exponentially unlikely to be coincidental.

*Recall (false negative rate) — the honest cost:*

A false negative occurs when a key truly satisfies the predicate but is not detected because it was displaced from its canonical position in one or more arrays. At probe depth k=1, a key is at its preferred position with probability (1 - α) where α is the load factor. For the key to be detected by a positional scan, it must be at its canonical position in *every* array involved in the predicate.

Concrete FNR at k=1:

| Predicate | FNR formula | α=0.25 | α=0.50 | α=0.75 | α=0.90 |
|---|---|---|---|---|---|
| A ∩ B (2-array) | 1 - (1-α)² | 43.8% | 75.0% | 93.8% | 99.0% |
| A ∩ B ∩ C (3-array) | 1 - (1-α)³ | 57.8% | 87.5% | 98.4% | 99.9% |
| A \ B (difference) | α | 25.0% | 50.0% | 75.0% | 90.0% |

At k=4 (probing four chunk positions per array):

| Predicate | FNR formula | α=0.25 | α=0.50 | α=0.75 | α=0.90 |
|---|---|---|---|---|---|
| A ∩ B (2-array) | 1 - (1-α⁴)² | 0.77% | 11.7% | 53.2% | 87.0% |
| A ∩ B ∩ C (3-array) | 1 - (1-α⁴)³ | 1.15% | 16.9% | 67.4% | 93.6% |
| A \ B (difference) | α⁴ | 0.39% | 6.25% | 31.6% | 65.6% |

The FNR gets worse with more arrays because displacement in *any single array* breaks the match. This is the fundamental trade-off: **high precision, tuneable recall**. The precision story is excellent — when the predicate fires, it is almost certainly correct. The recall story depends entirely on load factor and probe depth.

*Framing for the paper:*

The multi-set predicate is not an exhaustive set operation. It is a **high-confidence positive signal with tuneable coverage**. At low load (α ≤ 0.25) with k=4, recall exceeds 99% for 2-array predicates — effectively complete. At high load (α ≥ 0.75), recall degrades significantly and the predicate becomes a spot-check rather than a census.

This makes the predicate well-suited for:
- **Conflict detection** — "do any of these nodes disagree?" requires high precision, tolerates low recall (one confirmed conflict is enough to trigger full reconciliation)
- **Consensus verification** — "do all replicas agree on this key?" requires high precision on the positive signal
- **Replication priority** — cardinality of the intersection estimates overlap; even with low recall, the *ratio* of detected matches to scanned positions is an unbiased estimator of true overlap

It is less suited for:
- **Exhaustive set intersection** — missing 75%+ of true matches at high load makes the result incomplete
- **Authoritative difference computation** — use the hash arrays for exact answers when completeness is required

The difference predicate (`A \ B`) has an inverted error profile worth noting separately: it has **zero false positives** on the fp8 comparison (if the bytes differ, they differ) but has false negatives at rate 1/255 (two different keys producing the same fp8, making a true difference invisible). This is the mirror image of the intersection error profile.

**Considerations for implementation:**

1. **Array registration.** The GPU path needs Metal buffers for each input array. For N=2 this is one extra buffer (the second array, wrapped per-call as in m1.2). For N=3+, either pre-register arrays or accept per-call buffer creation overhead. At large array sizes the scan dominates buffer setup.

2. **Predicate mask validation.** Not all 8-bit (2-array) or 64-bit (3-array) masks produce meaningful predicates. The mask `0x00` matches nothing; `0xFF` matches everything. Document which masks correspond to standard set operations — provide a named constant table mapping predicate names to mask values.

3. **Output mode.** Some predicates (intersection, consensus) produce sparse output — stream compaction via atomic counter. Others (symmetric diff at high divergence) produce dense output — bitmask may be more efficient than position list. The kernel should support both output modes, selectable per invocation.

4. **Cardinality-only mode.** For predicates where only the count matters (e.g. similarity ratio, replication gap size), skip the matches buffer entirely and just accumulate the atomic counter. This halves the output bandwidth.

5. **Multi-predicate fusion.** A coordinator evaluating multiple predicates over the same arrays (e.g. "compute intersection AND difference AND unique ownership in one pass") could fuse them into a single kernel dispatch that evaluates multiple masks and writes to multiple output buffers. This amortises the scan cost across predicates. On CPU, this requires either three NEON passes or a more complex fused loop. On GPU, it is three mask evaluations per thread in the same dispatch with the same memory traffic — the marginal cost of an additional predicate is nearly zero. This is where the GPU advantage compounds beyond raw bandwidth.

6. **Mask indirection cost.** Benchmark the mask-based generalised kernel against hardcoded kernels for the standard predicates (intersection, difference). The mask adds a few ALU operations per thread (state computation, bit shift, test). If the overhead is unmeasurable against the memory bandwidth cost, that is a clean result validating the generalised approach. If it costs 5-10%, report it — the flexibility is likely still worth it, but the paper should state the cost honestly.

7. **Error model calibration.** The theoretical FPR/FNR per predicate must be validated empirically, similar to m2.1. The benchmark should report not just latency but also the observed false positive and false negative rates against ground truth (the hash arrays provide exact membership). This is essential for the Bayesian framing — the theoretical tables above are predictions; the benchmark confirms or corrects them.

8. **3-array memory pressure.** At 128M slots, three arrays consume ~384MB of fp8 data alone (plus hash arenas if needed for ground truth validation). The GPU path reads 3 bytes per position per thread — 50% more bandwidth than the 2-array case. This should push the GPU crossover point slightly lower (GPU wins earlier because there is more memory work to amortise per position). Verify this empirically.

**Benchmark parameters (when implemented):**
- Sizes: 256K, 1M, 8M, 128M
- Array count: 2, 3
- Predicates: intersection, difference, consensus, unique ownership
- Occupancy per array: 0.50, 0.90
- Overlap: 10%, 50%, 90%
- Metric: time per predicate evaluation (ns), cardinality, empirical FPR/FNR vs ground truth
- Compare: CPU NEON vs GPU Metal, hardcoded kernel vs mask-based generalised kernel

---

## m2: Probabilistic Signal Comparison

CPU benchmarks comparing the radix tree's probabilistic signal quality and cost against traditional probabilistic structures. These benchmarks do not involve GPU — the value proposition here is architectural (zero additional storage, signal as structural byproduct) rather than computational.

---

### m2.1 — Probabilistic Membership: Positional Probe vs Bloom Filter

**Question:** "Given a membership query against a probabilistic summary, how does the positional probe compare to a Bloom filter in cost and accuracy?"

**Context:** The paper's central claim is that the radix tree's structural geometry emits a probabilistic signal at zero additional storage cost. The positional probe — compute preferred address, read one byte, compare fp8 — provides a membership answer with characterisable FPR. A Bloom filter provides the same answer with tuneable FPR but requires a dedicated structure. This benchmark empirically validates the theoretical cost model.

**Operation:**

For a set of N query keys against a fixed index of S keys:

*Positional probe (k=1):*
```
preferred_pos = hash_to_position(key)
result = fp_array[preferred_pos] == expected_fp(key)
```
One hash computation, one byte read, one comparison.

*Positional probe (k=4):*
```
for each chunk i in 0..4:
    pos_i = hash_to_chunk_position(key, chunk=i)
    if fp_array[pos_i] == expected_fp(key): return probable_hit
return probable_miss
```
One hash computation, four byte reads (at four independently computed addresses), four comparisons. Lower FNR than k=1, slightly higher FPR.

*Bloom filter (k=3, sized for ~1% FPR):*
```
for each hash function h_i in 0..k:
    if bit_array[h_i(key)] == 0: return definite_miss
return probable_hit
```
k hash computations, k random bit reads, k comparisons. Zero false negatives.

**What to measure:**

| Metric | How | Why |
|---|---|---|
| Latency per query | Time N queries, divide | Raw cost comparison |
| False positive rate | Query M absent keys, count false positives | Empirical FPR vs theoretical |
| False negative rate | Query M present keys, count misses | Positional probe has FN; Bloom does not |
| Storage overhead | Bytes consumed by auxiliary structure | Bloom: k×n bits. Positional: 0. |

**Benchmark parameters:**
- Index sizes: 256K, 1M, 8M
- Load factors: 0.25, 0.50, 0.75, 0.90
- Query batch: 100K keys (50% present, 50% absent)
- Bloom configuration: k=3, sized at 10 bits/key (~1% FPR)
- Positional probe: k=1 and k=4
- All CPU, single-threaded

**Expected results:**

Latency: Bloom and positional probe should be in the same ballpark — both are a small number of memory reads. Positional probe at k=1 is likely faster (one read vs three). At k=4, similar cost to Bloom k=3.

FPR: Bloom at k=3 with 10 bits/key → ~1% FPR. Positional k=1 at 75% load → 0.29% FPR (from the independence analysis). Positional probe should have *lower* FPR than Bloom at standard configurations, despite zero additional storage.

FNR: Bloom → 0% (guaranteed no false negatives). Positional k=1 → α (load factor) — elements displaced from preferred position are missed. Positional k=4 → α⁴. At 75% load: k=1 misses 75%, k=4 misses ~31.6%. This is the trade-off: the positional signal has false negatives that Bloom does not.

Storage: Bloom at 10 bits/key, 1M keys → 1.25 MB auxiliary. Positional probe → 0 bytes auxiliary.

**The narrative:** The positional probe is not a Bloom filter replacement — it has false negatives. It is a *different kind of signal*: lower FPR than Bloom at zero storage cost, but with a tuneable FNR controlled by probe depth k. The system designer chooses k based on the cost of a false negative vs the cost of additional probes. In distributed routing where a false negative means "ask the node anyway" (wasted round trip) and a false positive means the same, the asymmetry may favour the positional probe's economics despite the FN rate.

**Compares against (for paper narrative):**
- Bloom filter (k=3, 10 bits/key): the standard baseline
- Cuckoo filter (if implementation available): similar FPR to Bloom, better cache behaviour, supports deletion
- No filter (brute force query all nodes): establishes the baseline cost that any filter is trying to reduce
- Theoretical predictions from the signal equation in section 5: validate that empirical FPR/FNR match the model

---

## Summary

| Bench | ID | Domain | Access pattern | GPU involvement |
|---|---|---|---|---|
| Fingerprint diff | m1.1 | Differential sync | Coalesced full-array map | GPU vs CPU |
| Set intersection + cardinality | m1.2 | Approximate PSI, replication priority | Coalesced full-array predicate + compaction | GPU vs CPU |
| Multi-set predicate evaluation | m1.3 | Bayesian inference over N arrays, replication gaps, consensus | Coalesced full-array mask-driven predicate | GPU vs CPU |
| Probabilistic membership | m2.1 | Distributed routing signal quality | Scattered single-byte probes | CPU only |