# Radix Hash Tree — Implementation Plan (Rust)

## Introduction

This project implements a radix hash tree — a fixed-size, open-addressed hash index designed for high-throughput concurrent workloads on uniformly distributed 64-bit keys. The implementation serves two purposes: as a practical hash index for the mgraph distributed synchronisation engine, and as an experimental platform for a research paper on entropy-directed placement in hash tables.

The core research question is whether partitioning hash bits into non-overlapping functional segments — bucket selection, group direction, preferred-slot placement, fingerprint filtering, and overflow routing — yields measurable performance improvements, and at which granularities those improvements materialise.

Recent theoretical work by Farach-Colton, Krapivin, and Kuszmaul (2025) disproved a forty-year-old conjecture that uniform probing is optimal among greedy open-addressed hash tables. Their constructions (funnel hashing, elastic hashing) achieve better probe complexity by controlling the fraction of elements reaching each level of a geometric sub-array hierarchy. The present work investigates a complementary factor: reducing the probe cost *within* each level by directing elements to specific positions using hash entropy. These two factors — inter-level spillover fraction and intra-level probe cost — are independent terms in the amortised cost equation, and should compose.

The implementation is structured as a sequence of milestones, each isolating one mechanism and benchmarking its contribution. The progression tells a specific story:

1. **Within a single SIMD-width group (64 slots)**, preferred-slot placement is theoretically sound but practically invisible behind a one-instruction fingerprint scan. This is an honest result — SIMD eliminates the search cost that direction would reduce.

2. **Across multiple SIMD groups within a bucket (4×64 = 256 slots)**, entropy-directed group selection avoids scanning groups that don't contain the target. The performance gain is real and measurable — proportional to the number of groups skipped.

3. **In a distributed context**, the same preferred-slot addressing enables a capability SIMD cannot provide: positional probabilistic membership queries against a remote node's fingerprint array, without scanning. The hash index and the probabilistic membership structure are the same bytes, accessed differently.

Each milestone produces benchmark data that feeds directly into the research paper. The milestones are designed so that negative results (M2) are as valuable as positive ones — they bound where each mechanism contributes and where it does not.

The target platform is Apple Silicon (AArch64/NEON) with x86_64 (SSE2/AVX2) as secondary. The implementation language is Rust.

---

## Architecture Decisions

### Trait-Based Milestone Isolation

All milestones implement a common trait defining the core operations: insert, lookup, lookup-miss, length, and load factor. Benchmarks are written once against this trait. Each milestone is a separate module with its own struct implementing the trait.

Feature flags gate compilation of each milestone. This allows benchmarks to target any combination of implementations in a single run without losing access to prior milestones.

### Configuration

All structural parameters are set at instantiation time. The developer provides a configuration specifying:

- **Prefix bits**: determines bucket count (2^prefix_bits). Defines the top-level fan-out of the radix structure.
- **Arena capacity**: total number of slots reserved across all buckets. This is the fixed memory commitment.

Sensible defaults must be provided for developers who do not care about tuning. The defaults should target a moderate workload: enough buckets to keep per-bucket occupancy reasonable, enough total capacity for a few million entries.

Validation at construction time: prefix bits must be consistent with arena capacity (arena must hold at least 2^prefix_bits × slots_per_bucket slots). Construction fails with a clear error if the configuration is incoherent.

These parameters are immutable after construction. The table does not resize.

### Arena Layout

Two contiguous allocations, indexed in parallel by slot position:

- **Fingerprint arena**: 1 byte per slot (fp8). This is the SIMD scan target. Must be cache-line aligned.
- **Hash arena**: 8 bytes per slot (u64). This is the confirmation target.

Within the fingerprint arena, slots are ordered so that a single bucket's fingerprints are contiguous, and within a bucket, each group's 64 fingerprints are contiguous. A directed probe to a specific bucket and group resolves to a single 64-byte region — one cache line on most architectures.

No payload arena is needed for benchmarking. The operations are insert-by-ID and lookup-by-ID, where the ID is the 64-bit hash itself.

Empty slots are indicated by a reserved sentinel value in the fingerprint arena (0x00). This means a hash that produces fp8 = 0x00 must be remapped to a non-zero value (e.g. 0x01). This costs one value from the fingerprint space (255 usable values instead of 256) and is acceptable.

### Hash Function and Bit Extraction

A single hash function produces a 64-bit value from the input ID. The choice of hash function is not prescribed, but it must produce well-distributed output across all bit positions.

Bits are consumed left to right (most significant first):

- First segment: prefix bits for bucket selection.
- Subsequent segments: group selection, preferred slot, fingerprint, overflow bits.

The exact bit widths for each segment depend on the milestone (e.g. no group bits in M2, 2 group bits in M3). The extraction logic must be parameterised by the configuration, not hard-coded.

### Benchmarking

All milestones report the same metrics across the same conditions.

**Operations**: insert, lookup-hit, lookup-miss.

**Load factors**: 50%, 75%, 90%, 95%, 99%.

**Metrics per operation**:
- Throughput (operations/second)
- Mean latency (nanoseconds)
- p50, p95, p99, p999 latency

**Comparison baseline**: hashbrown (Rust's Swiss Table implementation) at the same load factors and dataset sizes.

Each benchmark run pre-populates the table to the target load factor, then measures the operation. Lookup-hit draws from IDs known to be inserted. Lookup-miss draws from IDs known to be absent. Insert benchmarks measure insertion into a table already at the target load factor (i.e. the cost of inserting the marginal element, not the average cost of filling from empty).

Use Criterion for benchmarking. Report results in a format suitable for inclusion in the paper (tables or CSV export).

---

## Milestones

### M0 — Baseline (Linear Probing)

**Purpose**: establish a naive control implementation. Every subsequent milestone measures improvement against this.

**Structure**: flat array of slots. No fingerprints, no SIMD, no preferred slots. Open addressing with linear probing.

**Insert**: hash the ID, compute slot index, linear probe forward until an empty slot is found.

**Lookup**: hash the ID, compute slot index, linear probe forward comparing full hashes until found or empty slot encountered.

**Configuration**: prefix bits and arena capacity define the array size. Slots per bucket is 1 (i.e. the array is a flat sequence of slots, no bucket abstraction beyond the initial hash modulus).

**Benchmark targets**: establish baseline throughput and latency at each load factor. Compare against hashbrown to validate that the baseline is reasonable (it should be slower — that's expected).

**Deliverable**: working implementation, benchmark results, confirmation that hashbrown outperforms the baseline as expected.

---

### M1 — Hash Function and Bit Partitioning

**Purpose**: establish the hash and bit extraction infrastructure used by all subsequent milestones. Validate that bit segments are well-distributed and independent.

**No performance benchmark at this milestone.** This is a correctness foundation.

**Requirements**:
- Select a hash function. Document the choice and rationale.
- Implement bit extraction parameterised by configuration (prefix width, group bits, slot bits, fingerprint bits).
- Validate distribution: for a large sample of uniformly random 64-bit IDs, confirm that each extracted segment (bucket, group, slot, fingerprint) is approximately uniformly distributed.
- Validate independence: confirm that knowing the bucket bits gives no information about the group, slot, or fingerprint bits. A chi-squared test or equivalent over pairwise segment combinations is sufficient.

**Deliverable**: hash and extraction module with distribution tests passing.

---

### M2 — Single SIMD Group (B=64)

**Purpose**: introduce SIMD fingerprint scanning. Establish whether preferred-slot placement adds measurable value within a single SIMD-width group. The expected result is near-parity between directed and undirected variants — this is an honest data point, not a failure.

**Structure**: each bucket contains exactly one group of 64 slots. Fingerprint arena is laid out as described above. Hash arena in parallel.

**Insert**: extract bucket and fingerprint. Check preferred slot first — if empty, place there. Otherwise, SIMD scan the group for any empty slot. Place at the first available. If the group is full, the insert fails (no overflow at this milestone).

**Lookup**: extract bucket and fingerprint. SIMD scan the 64-byte fingerprint region for matches. For each candidate, confirm against the full hash in the hash arena. Return on first confirmed match.

**Lookup-miss**: same as lookup, but the target ID is known to be absent. The SIMD scan returns no candidates (or candidates that fail hash confirmation). This path never touches the hash arena if the fingerprint has no collisions in the group.

**Two variants to benchmark**:
- **(a) SIMD-only**: skip preferred slot, always SIMD scan for both insert and lookup.
- **(b) SIMD + preferred slot**: check preferred slot first, fall back to SIMD.

**SIMD implementation notes**: target both AArch64 (NEON) and x86_64 (SSE2/AVX2) if feasible. If only one platform is available for benchmarking, document which and note the other as future work.

**Benchmark targets**: compare (a) vs (b) at each load factor. Compare both against M0 and hashbrown. Report whether preferred-slot fast path produces a measurable latency difference. Quantify the improvement from SIMD fingerprint scanning itself (M2a vs M0).

**Deliverable**: working SIMD scan, two variants benchmarked, honest assessment of preferred-slot value at this granularity.

---

### M3 — Multi-Group Bucket (B=256, 4×64)

**Purpose**: demonstrate that entropy-directed group selection delivers measurable performance improvement at granularities above SIMD width. This is the primary practical result for the paper.

**Structure**: each bucket contains 4 groups of 64 slots (256 slots per bucket). Fingerprint arena layout: groups within a bucket are contiguous (group 0 occupies bytes 0–63 of the bucket's region, group 1 occupies 64–127, etc.). This ensures that scanning a second group within the same bucket is a sequential cache line read.

**Insert**: extract bucket, group, preferred slot, and fingerprint. Attempt placement in the directed group: preferred slot first, then SIMD scan for empty. If the directed group is full, try remaining groups in order. If all groups are full, insert fails (no overflow at this milestone).

**Lookup**: extract bucket, group, and fingerprint. SIMD scan the directed group first. If found, return. If not, scan remaining groups. (The fallback policy — scan all groups or stop after directed — is a design choice to benchmark.)

**Three variants to benchmark**:
- **(a) Sequential scan**: ignore group bits, scan all 4 groups in order for every operation.
- **(b) Random group selection**: select a group at random (not from hash bits — use a PRNG seeded separately) to isolate structural benefit of grouping from the value of deterministic direction.
- **(c) Entropy-directed group selection**: use 2 hash bits to select the group. Preferred slot within the group.

**Benchmark targets**: compare (a) vs (b) vs (c) at each load factor. The key metric is the gap between (a) and (c) — this quantifies the value of direction. The gap between (b) and (c) isolates whether deterministic direction outperforms random selection (it should, because elements cluster at their directed group, so the directed group is more likely to contain the target).

Also compare (c) against M2 and hashbrown. Report cache miss counts if perf counters are available — the 4× cache line reduction from directed group selection is the mechanistic explanation for any throughput improvement.

**Deliverable**: three variants benchmarked, ablation isolating group-direction value, cache behaviour quantified.

---

### M4 — Overflow

**Purpose**: demonstrate that fresh-bit bucket selection on overflow prevents clustering and improves tail latency compared to linear overflow.

**Structure**: same as M3 (4×64 groups per bucket). When all groups in a bucket are full, overflow to another bucket.

**Two variants to benchmark**:
- **(a) Linear overflow**: on bucket full, proceed to bucket index + 1.
- **(b) Fresh-bit overflow**: consume the next segment of hash bits to select an independent overflow bucket. The directed group mechanism applies within the overflow bucket as normal.

**Overflow round tracking**: each element must record how many overflow rounds were needed for its placement (for analysis, not stored in the arena — tracked during insertion and reported as a distribution).

**Bit budget accounting**: document how many overflow rounds are supported before hash bits are exhausted. For a 64-bit hash with, say, 16-bit prefix + 2-bit group + 6-bit slot + 8-bit fingerprint = 32 bits consumed in round 1, each overflow round consuming another 16 bits of bucket selection, approximately 2 independent rounds are available. If this is insufficient at high load, document the limitation and note that 128-bit hashes extend the budget.

**Benchmark targets**: compare (a) vs (b) at 90%, 95%, 99% load. Focus on tail latency (p99, p999). Mean throughput may be similar — the hypothesis is that tail diverges sharply. Report the distribution of overflow rounds at each load factor.

**Deliverable**: two overflow strategies benchmarked, tail latency comparison, overflow round distribution data.

---

### M5 — Optimisations

**Purpose**: engineering improvements that make the implementation credible as a practical system. These are not research contributions but support the claim that the benchmarked design could ship.

**Bucket-full flag**: a single bit per bucket (or per group) indicating whether the group is completely occupied. Checked before SIMD scan on insert — if the group is full, skip directly to the next group or overflow. Avoids a wasted SIMD scan. Benchmark the difference: this matters most at high load where full groups are common.

**Prefetching**: when overflow is needed, issue a prefetch for the overflow bucket's fingerprint cache line before completing the current group's scan. Benchmark with and without prefetching at high load factors.

**Concurrency**: per-bucket or per-group locking for concurrent insert. Read path (lookup) is lock-free — fingerprint scan and hash confirmation are read-only. Benchmark throughput scaling with 1, 2, 4, 8 threads for both insert-heavy and read-heavy workloads. Report contention rate.

**Deliverable**: each optimisation benchmarked independently (ablation), then combined. Final comparison of the fully optimised M5 against hashbrown.

---

### M6 — Probabilistic Membership

**Purpose**: demonstrate that the fingerprint arena with preferred-slot addressing serves as a probabilistic membership structure without scanning. This is the distributed systems argument — the same structure that serves local exact-match lookup also supports remote probabilistic queries.

**This milestone does not require networking.** It simulates the distributed case by separating the fingerprint arena from the hash arena and measuring probe accuracy against the fingerprint arena alone.

**Operation — single-probe membership test**: given an ID, compute bucket, group, and preferred slot. Read the single fp8 byte at that position. Report: empty (definite negative), fp8 match (probable positive), fp8 mismatch (probable negative, subject to displacement false negatives).

**Operation — k-probe membership test**: same as above, but on mismatch, scan k additional slots within the group before returning probable negative. Parameterise k from 0 (preferred slot only) to 63 (full group scan).

**Measurements**:
- **False positive rate**: fraction of absent IDs that produce an fp8 match at the preferred slot. Theoretical: 1/255 per probe (given 255 usable fingerprint values). Confirm empirically.
- **False negative rate**: fraction of present IDs that are not found at their preferred slot (due to displacement at insertion). This varies with load factor — measure across the full range.
- **Accuracy at k probes**: for k = 0, 1, 2, 4, 8, 16, 63, report the combined false positive and false negative rates.

**Fingerprint width sweep**: repeat the above measurements with fp16 (2 bytes per slot, 128 bytes per group) and fp32 (4 bytes per slot, 256 bytes per group). This requires variant arena layouts — the fingerprint arena widens but the hash arena and addressing logic remain the same.

**Bandwidth cost analysis**: for each fingerprint width, compute the bytes-per-query for the single-probe case and the bytes-per-bucket for a full summary exchange. Compare against a Bloom filter and cuckoo filter at equivalent false positive rates in terms of bits per element.

**Benchmark targets**: measure single-probe throughput (this should be very fast — one address computation + one byte read). Compare against Bloom filter and cuckoo filter lookup throughput at equivalent FPR. The hypothesis is that single-probe preferred-slot lookup is faster because it avoids hashing to multiple positions (Bloom) or scanning a bucket (cuckoo).

**Deliverable**: false positive/negative rate tables across load factors and fingerprint widths, accuracy-vs-probes curve, bandwidth comparison against standard probabilistic structures, throughput comparison.

---

## Benchmark Summary Matrix

Each milestone reports results in a consistent format.

| Milestone | Variants | Compared Against | Key Question |
|---|---|---|---|
| M0 | Linear probing | hashbrown | Is the baseline credible? |
| M2 | SIMD-only vs SIMD+preferred | M0, hashbrown | Does preferred slot help within SIMD width? |
| M3 | Sequential vs random-group vs directed-group | M2, hashbrown | Does directed group selection reduce latency? |
| M4 | Linear overflow vs fresh-bit overflow | M3 | Does fresh-bit overflow improve tail latency? |
| M5 | Each optimisation independently, then combined | M4, hashbrown | Does the optimised system compete with production hash tables? |
| M6 | Single-probe vs k-probe, fp8/fp16/fp32 | Bloom filter, cuckoo filter | Does preferred-slot addressing enable competitive probabilistic membership? |

Load factors for all milestones: 50%, 75%, 90%, 95%, 99%.

Operations for M0–M5: insert, lookup-hit, lookup-miss.

Operations for M6: probabilistic-positive, probabilistic-negative, k-probe sweep.

---

## Paper Narrative Mapping

Each milestone supports a specific section of the research paper.

- **M0** → baseline for all comparisons. No paper section of its own; referenced throughout.
- **M2** → "preferred slot within SIMD width is theoretically motivated but practically marginal." Honest negative result that establishes credibility.
- **M3** → "entropy-directed group selection delivers measurable improvement above SIMD width." Primary practical contribution. Ablation data supports the cost decomposition argument.
- **M4** → "fresh-bit overflow prevents clustering." Supports the f(k) analysis and connection to Farach-Colton et al.
- **M5** → "the design is practical, not just theoretical." Credibility for the system as a whole.
- **M6** → "the same structure serves probabilistic membership for distributed systems." The reframe — preferred-slot addressing enables capabilities that SIMD scanning cannot provide remotely.