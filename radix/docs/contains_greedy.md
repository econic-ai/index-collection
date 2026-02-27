# Greedy Scan Membership Test — `contains_greedy`

## Context

The radix tree's `contains` operation uses **entropy-directed probing**: bits extracted from the mixed hash select a bucket, group, and preferred slot, narrowing the search to a single cache-line-aligned 64-byte group. Within that group, a scalar check at the preferred offset resolves most lookups in one byte read plus one hash confirmation. Overflow follows a deterministic path through the same group in successive buckets. The probe depth is logarithmic in the displacement chain length, and at low load the expected cost is effectively O(1) with a small constant.

This design exploits the structure of the hash to avoid scanning irrelevant regions of the arena. It is the foundation of the radix tree's competitive single-key lookup performance and the basis of the probabilistic membership signal (`contains_probable`) described in the paper.

`contains_greedy` takes the opposite approach: it **ignores the hash-directed probe path entirely** and performs a linear scan of the full fingerprint arena, checking every slot against the target fingerprint and confirming hash matches. It answers the same question — "is this key in the index?" — but through brute-force enumeration rather than directed search.

---

## Why greedy scan exists

The greedy scan is not a replacement for entropy-directed probing. It exists to test a specific hypothesis about hardware-accelerated linear traversal on Apple Silicon's unified memory architecture.

The GPU iteration benchmarks (m1) demonstrated that at large table sizes (8M+ slots), the Apple GPU's linear scan throughput exceeds the CPU's NEON-accelerated scan by 2-7×. The `iter` kernel scans all fingerprints for non-zero bytes — a uniform, coalesced, branch-free workload that maps perfectly to GPU execution.

`contains_greedy` extends this observation to membership testing. The kernel shape is nearly identical to `iter`: scan all fingerprints, but instead of collecting all occupied keys, filter for a specific fingerprint value and confirm against a specific hash. The per-element predicate is slightly more expensive (two comparisons instead of one), but the access pattern — sequential, coalesced, uniform work per thread — is identical.

The question is whether GPU-accelerated linear scan can outperform CPU entropy-directed probing for single-key membership, and at what table size the crossover occurs.

---

## Relationship to entropy-directed probing

This capability deliberately contradicts the core thesis of the radix tree design. The paper argues that bit entropy extracted from the hash function provides both:

1. **Optimised probing** — hash bits direct the search to a narrow region, avoiding full-arena traversal.
2. **Probabilistic signals** — the fingerprint + positional encoding yields a high-discrimination composite key that enables remote membership estimation without full-key confirmation.

Greedy scan discards property (1) entirely. It does not use bucket, group, or slot bits from the hash. It treats the fingerprint arena as an unstructured byte array and scans it linearly. The only hash-derived information it uses is the 8-bit fingerprint for initial filtering and the full hash for confirmation.

This raises a tension: if hardware-accelerated linear scan can match or exceed entropy-directed probing at sufficient scale, the value of the positional encoding and hierarchical hash partitioning becomes a function of table size and available compute. At small scales, entropy-directed probing dominates. At large scales, the GPU's raw memory bandwidth may make the directed structure unnecessary for membership testing — though it remains essential for insert, iteration ordering, and probabilistic signalling.

The implications of this tension for the paper's thesis are deferred to later analysis. For now, `contains_greedy` provides the empirical data needed to characterise the crossover.

---

## Operation semantics

```
contains_greedy(id: u64) -> bool
```

Scans the entire fingerprint arena for slots where `fp[slot] == target_fp`. For each match, confirms `hashes[slot] == id`. Returns `true` on the first confirmed match, `false` if no match is found after scanning all slots.

The operation is exact — no false positives, no false negatives. It is semantically equivalent to `contains` but with a different performance profile.

---

## CPU implementation

NEON-accelerated linear scan, structurally similar to the iterator's `advance_chunk`:

1. Broadcast the target fingerprint to a 16-byte NEON register.
2. For each 16-byte chunk of the fingerprint arena:
   - Load 16 bytes via `vld1q_u8`.
   - Compare against the target fp via `vceqq_u8`.
   - Pack the comparison result to a 16-bit mask.
   - For each set bit in the mask, confirm `hashes[slot] == id`.
   - Return `true` on first confirmed match.
3. Return `false` after scanning all chunks.

Expected throughput: comparable to `iter` at the fingerprint scan level, with additional hash confirmation cost proportional to the false positive rate (~1/255 per occupied slot).

---

## GPU implementation

Metal compute kernel dispatched over all slots:

```metal
kernel void contains_greedy(
    const device uchar*  fps    [[buffer(0)]],
    const device ulong*  hashes [[buffer(1)]],
    device atomic_uint*  found  [[buffer(2)]],
    constant uchar&      fp     [[buffer(3)]],
    constant ulong&      key    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (fps[gid] == fp && hashes[gid] == key) {
        atomic_store_explicit(found, 1, memory_order_relaxed);
    }
}
```

Each thread checks one slot. The atomic store fires at most once (when the key is found). No compaction, no output buffer, no inter-thread coordination beyond the single atomic.

The kernel reuses the zero-copy fingerprint and hash buffers created at table construction (same as the `iter` kernel). The only additional buffers are a 4-byte atomic flag and two constants (fp, key).

---

## Benchmark design

Added to the m1 bench binary as `op=contains_greedy`. Same table sizes and load factors as `iter`:

- Sizes: 256K, 1M, 8M, 128M
- Load factors: 0.10, 0.90
- Query: a key known to be present in the table (hit case)
- Metric: time per single `contains_greedy` call (ns)

The benchmark measures single-key greedy scan latency, not batched throughput. This isolates the per-call cost including GPU dispatch overhead, providing a direct comparison against `contains` (entropy-directed probe) at the same table sizes.

---

## Expected results

| Size | `contains` (probe) | `contains_greedy` CPU | `contains_greedy` GPU |
|------|--------------------|-----------------------|-----------------------|
| 256K | ~3-10 ns | ~15-30 µs | ~280 µs (dispatch-bound) |
| 128M | ~3-10 ns (cache miss at high load) | ~80-110 ms | ~11-15 ms |

At all sizes, entropy-directed `contains` should be faster for single-key queries. The greedy scan's value is as a baseline for understanding GPU linear scan cost, and as a building block for future batch operations where many keys are tested against the same arena in a single dispatch.
