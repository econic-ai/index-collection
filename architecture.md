# Bucketed Radix-Hash Index: Architecture & Implementation Plan

This document specifies the design and implementation plan for a concurrent, cache‑optimised, bucketed radix‑hash index.

Its goals are to:

- Define an exact, high‑throughput index structure for `get`, `insert`, `delete`, and iteration, with a fixed global bucket count chosen at initialisation.
- Optimise both point lookups and full iteration, with iterators implemented directly over masks and slots for maximum locality.
- Use epoch‑guarded, immutable bucket configurations that can be atomically replaced to support per‑bucket growth without global table resizing.
- Allow callers to supply their own epoch guards so that many operations (especially iterations and batched lookups) can run under a single guard, excluding epoch overhead from hot‑path benchmarks.
- Exploit cache locality via careful arena allocation so that bucket metadata (configs, masks, fingerprints) and chunks are laid out to minimise cache lines touched per operation.
- Provide a clear foundation for future extensions such as multi‑chunk buckets, richer fingerprints (fp16), and advanced probing strategies, without changing the external API or correctness guarantees.


## Comparison with Previous RadixIndexV2 POC

Quick visual comparison of the existing `RadixIndexV2` POC vs the new architecture.

| Feature | POC V2 | Target Design |
|--------|--------|---------------|
| Fixed global bucket count | ✅ (derived from capacity; no global resize) | ✅ (user- or keyspace-derived; still fixed) |
| Per-bucket growth | ❌ (64 slots hard limit; panic on full) | ✅ (per-bucket `BucketConfig` growth with extra chunks; no panic) |
| Bucket selection | ✅ (radix on leading hash bits) | ✅ (same radix selection, but driven by explicit `BucketConfig`) |
| Preferred slot from hash | ✅ (disjoint bits; linear probe with wrap) | ✅ (retained; can add per-bucket seed and bounded neighbourhood) |
| Fingerprint prefilter | ✅ (single fp8 tag array per bucket) | ✅ (fp8 baseline, with clean upgrade path to dual fp8 = fp16) |
| 64-bit occupancy mask | ✅ (one `AtomicU64` per bucket) | ✅ (per-chunk mask; may pair with a write-reservation mask) |
| Tombstone support | ✅ (`kind` byte in record; mask drives liveness) | ✅ (mask-driven liveness retained; tombstones optional/internal) |
| Per-bucket record locality | ✅ (contiguous `Rec<K,V>` per bucket from arena) | ✅ (contiguous chunks per bucket, possibly multi-chunk) |
| Concurrency on mask | ✅ (`Acquire/Release` loads; CAS for move-update) | ✅ (mask remains the publish point; cleaner insert/update patterns) |
| Epoch-guarded configs | ❌ (no epoch / config indirection) | ✅ (immutable `BucketConfig` replaced atomically; safe per-bucket growth) |
| User-supplied guard | ❌ (no guard concept; per-op atomics only) | ✅ (`*_with_guard` APIs so callers can hoist a guard across many ops) |
| Iteration strategy | ✅ (mask hoisting into local `Vec<u64>`; then no atomics) | ✅ (iterator owns a guard and walks buckets/chunks directly; mask/tag-driven) |
| Clear / reset behaviour | ✅ (`clear_all` resets masks; tags left as-is) | ✅ (explicit clear; plus well-defined rebuild/growth paths) |
| Stats & diagnostics | ✅ (`collect_stats` for buckets + arena) | ✅ (extended stats tied to `BucketConfig`, multi-chunk layouts, probe depth) |
| Memory layout / arenas | ✅ (single arena for records; metadata in parallel arrays) | ✅ (clear split: dense bucket metadata + chunk arena for data plane) |


## Features & Design Goals

This version of the radix-hash index is intended to be a stable, high-throughput core that can support both point lookups and heavy iteration workloads without sacrificing correctness. The design emphasises exact semantics, simple mental models for concurrency, and cache-aware memory layout so that the structure remains predictable under load and easy to extend in future versions.

### 1. Index-Level and Bucket-Level Configuration

The index is built around a small set of configuration objects that describe how keys are mapped to buckets and how each bucket is laid out internally. Index-level configuration captures global parameters (such as bucket count and hash bit allocation), while per-bucket configurations capture local layout decisions (such as the number of chunks in a bucket and any per-bucket seeds).

- Fixed global bucket count chosen at initialisation (user-specified or derived from capacity / keyspace).
- Index-level configuration struct to hold global mapping parameters (bucket_count, bucket_bits, slot_bits, hash layout), designed so it can later be made replaceable if global reconfiguration is required.
- Per-bucket immutable `BucketConfig` objects describing chunk count, per-bucket capacity, and layout metadata.
- Atomic pointer from each bucket to its current `BucketConfig`, allowing buckets to grow or change layout independently.

### 2. Bucket and Chunk Layout

Buckets are designed to be independently scalable units that can evolve from a simple single-chunk layout to more complex arrangements as they become hot or heavily loaded. Chunks provide the physical storage for keys, values, and metadata in a fixed-size, cache-friendly form.

- Per-bucket layout described by `BucketConfig`, with one or more contiguous chunks per bucket.
- Each chunk stores an occupancy mask, fingerprint array, and a fixed-size array of records.
- Record layout (`Rec<K, V>`) is contiguous within a chunk to maximise spatial locality for iteration and clustered accesses.
- Hash bits are partitioned into roles (bucket selection, chunk selection, fingerprint, preferred slot) to keep addressing logic simple and fast.

### 3. Concurrency, Publication, and Update Semantics

The index uses a clear publish-before-visible model to ensure that readers never observe partially written records. Concurrency is managed at the mask and configuration level, avoiding per-slot locks and allowing readers to remain lock-free.

- Epoch-guarded `BucketConfig` objects: configs are immutable once published and replaced atomically; old configs and their chunks are reclaimed safely via epochs.
- Mask-driven publication: writers store tag and record fields first, then publish via a `fetch_or` on the occupancy mask; readers only trust slots whose mask bit is set.
- Optional write-reservation mask per chunk to allow writers to claim slots without racing on record memory, while keeping readers unaware of in-progress writes.
- Update semantics based on "always new slot + tombstone old": updates write a new version into a new slot and clear the old bit, avoiding in-place mutation of live slots.

### 4. Lookup and Insertion Strategies

Lookup and insertion are structured to minimise both probe length and the number of full key comparisons. A preferred slot is derived from the hash, and a small fingerprint is used to filter candidates cheaply before comparing keys.

- Preferred slot per key derived from its hash, with a simple forward (and optionally wrapping) probe sequence.
- Per-bucket seed for preferred slot or probe sequence to reduce clustering when key distributions are not perfectly uniform.
- Per-slot fp8 fingerprint (derived from disjoint hash bits) used as a fast prefilter before full key comparison.
- Design kept compatible with a second fp8 plane (fp16) for workloads where full key comparison cost dominates.
- Exact probe region per key: the probing algorithm defines a bounded neighbourhood that is fully scanned, guaranteeing no false negatives.

### 5. Iteration as a First-Class Operation

Iteration over all keys or values is a core use case and is treated as a first-class concern in the design. Iterators work directly over masks and slots under a single guard, avoiding per-step synchronisation and retaining strong correctness guarantees.

- Iterators pin a single epoch guard for the duration of the traversal and reuse it across all buckets and chunks.
- Iteration walks buckets in index order and, within each bucket, walks chunks and masks directly, using bit tests to locate live slots.
- Key/value iteration variants are provided (values only, or key/value pairs) with no extra allocations and minimal branching.
- Iteration logic is independent of `get`/`insert` and is optimised separately for sequential access patterns.

### 6. Memory Layout and Allocation Strategy

Memory layout is designed to favour cache-friendly access patterns both for point lookups and for sequential iteration. Control-plane metadata is stored densely, while data-plane chunks are allocated from a dedicated arena.

- Contiguous array of bucket metadata, indexable by bucket id, for predictable iteration and background maintenance.
- Chunks for each bucket allocated from a dedicated arena, with each bucket's chunks stored contiguously for efficient scanning.
- Clear separation between control plane (bucket metadata, configs) and data plane (chunks, masks, fingerprints, records). 
- Layout chosen so that the hot path in `get` and `insert` touches as few cache lines as possible, while iteration enjoys linear scans over chunk memory.

### 7. API Ergonomics and Guard Hoisting

The public API is designed to support both simple single-operation calls and advanced batch or streaming workloads. Guard-aware variants allow callers to take explicit control over epoch pinning when they need tighter control over performance or semantics.

- Standard operations (`get`, `insert`, `delete`) that internally manage their own guard for ease of use.
- Guard-aware variants (`*_with_guard`) that accept a user-supplied guard, allowing many operations to run under a single pinned epoch.
- Pre-hashed variants that accept a 64-bit hash alongside the key to avoid repeated hashing in hot paths.
- Clear separation between correctness semantics (exactness, concurrency guarantees) and convenience features, so advanced users can reason about what the index does on their behalf.

## Experiments and Benchmark Plan

The architecture includes several deliberate degrees of freedom that are best evaluated empirically rather than decided purely on theory. This section records the initial set of experiment themes to be explored in microbenchmarks and workload benchmarks as the implementation matures.

### A. Fingerprints and Tag Handling

These experiments focus on how fingerprints are represented and scanned, and how that affects the balance between tag work and full key comparisons.

- Compare fp8 vs fp16 (two fp8 planes) in terms of:
  - Negative lookup cost and full key comparison rate.
  - Latency and throughput under mixed read/write workloads.
- Within fp16, compare scanning strategies:
  - Naive per-slot comparison of both tags.
  - Tag1-first, with Tag2 only checked on Tag1 match.
  - SIMD or wide-lane tag scanning versus simple scalar loops where appropriate.
- Evaluate alternative fp16 layouts:
  - Bit-sliced fp16 representation (16 bit-planes over the chunk) to enable chunk-wide bit-level matching.
  - fp16 and fp8 variants using 32-slot chunks where all tag arrays fit into a single cache line, tying into chunk-size experiments.

### B. Chunk Size and Mask Scheme


These experiments examine the trade-offs between chunk size, cache behaviour, and the complexity and cost of atomic mask operations.

- Compare 64-slot versus 32-slot chunk layouts while holding per-bucket capacity constant:
  - Number of cache lines touched per lookup and per iteration.
  - Probe length distributions and tail latency for lookups.
- Compare mask schemes for coordinating writers and readers:
  - Dedicated occupancy mask plus separate write-reservation mask per chunk.
  - Single packed 64-bit mask combining occupancy and reservations, using CAS-based updates.
- Measure the impact of each scheme on insert throughput, CAS contention, and read-path latency under low and high contention.

### E. Epoch Guards, API Variants, and Iteration

These experiments explore how the choice of guard management and iteration strategy affects end-to-end performance and usability.

- Compare per-operation guard management against user-hoisted guards via `*_with_guard` APIs:
  - Overhead of guard pin/unpin at various operation rates.
  - Benefits of hoisting guards in batched or streaming workloads.
- Compare iteration strategies:
  - POC-style mask hoisting into a temporary vector, followed by a read-only pass.
  - V3-style iterator that owns a guard and walks buckets and chunks directly over masks and slots.
- Assess how these choices interact with typical usage patterns (many small point operations vs long-running scans) to guide API defaults and documentation recommendations.
