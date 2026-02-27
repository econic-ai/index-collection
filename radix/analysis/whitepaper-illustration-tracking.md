# Whitepaper Illustration Tracking

| Title | Description | Information to Create? | Needs Codebase Verification? | Required Format |
|---|---|---|---|---|
| Figure 1.1 — The dual-structure problem | Node architecture diagram: hash index + Bloom filter with dual update/memory/sync overhead, contrasted with single-structure alternative. | Partial | No | svg diagram |
| Figure 2.1 — The hash as a segmented bar | 64-bit hash segmented into bucket/group/chunk/offset/fingerprint/overflow with position-vs-stored annotations. | Yes | No | svg diagram |
| Figure 2.2 — Positional entropy compounding through the hierarchy | Stacked/pyramid view of discrimination bits accumulating from fp8 + positional layers to total bits. | Yes | No | svg diagram |
| Figure 3.1 — Physical layout: bucket → group → chunk → slot | Memory layout diagram (256B bucket, 4x64B groups, 4x16B chunks, 16 fp8 slots) with cache-line and NEON annotations. | Yes | Yes | svg diagram |
| Figure 3.2 — Hash bit partitioning for a concrete configuration | Worked partition diagram for `capacity_bits=20` with exact bit widths/ranges and physical mapping. | Yes | Yes | svg diagram |
| Figure 3.3 — The probe sequence: power-of-k-choices across chunks | Sequence diagram of scalar preferred-offset checks across chunks with NEON fallback, contrasted with linear probing. | Yes | Yes | svg diagram |
| Figure 3.4 — Overflow direction: same group, next bucket | Two-bucket diagram showing overflow preserving group/chunk/offset tuple while advancing bucket only. | Yes | Yes | svg diagram |
| Figure 4.1 — Chunk occupancy distribution at each load factor | Histogram/heatmap of filled-slot distribution per chunk at 25/50/75% to visualise clustering. | Partial | Yes | png chart |
| Figure 4.2 — Latency progression across runs | Run-by-run trend chart (Runs 1-7) for hit/miss/insert at 75% load with design-change labels. | Partial | Yes | png chart |
| Figure 5.1 — Same bytes, two access modes | Side-by-side local SIMD confirmation vs remote positional-byte signal read from same array bytes. | Yes | Yes | svg diagram |
| Figure 5.2 — Signal accuracy as a function of probe depth k | Line chart of `P(correct)` vs `k` for multiple load factors with FP/FN tradeoff crossover. | Partial | Yes | png chart |
| Figure 6.1 — Query routing: 1000 nodes, narrowed by the signal | Distributed routing flow: full fan-out vs signal-narrowed candidate set with byte-read and round-trip annotations. | Partial | Partial | svg diagram |
| Figure 7.1 — The verification cost gradient | Layered verification-cost curve (slot→cluster) on log scale with probability overlay and decision crossing. | Partial | Partial | png chart |
| Figure 8.1 — What is disclosed: position + fp8 vs the full key | Dataflow diagram from full key to exposed coordinate+fp8 representation, emphasizing non-reconstructability. | Yes | Yes | svg diagram |
| Figure 8.2 — Differential sync: exchanging position deltas | Two-node sync diagram showing sparse position/fp8 deltas instead of full-key exchange. | Partial | Yes | svg diagram |

## Legend

- **Yes**: Enough structural/detail information is already in the draft to produce a defensible figure now.
- **Partial**: Concept and framing are clear, but numeric assumptions, exact datasets, or final scope need confirmation.
- **No**: Not enough information currently available to construct credibly.

### Codebase Verification Legend

- **Yes**: Needs implementation-level checking against real code paths/bit layouts/behavior before final art.
- **Partial**: Requires some code or systems assumptions to be confirmed, but can be drafted mostly from the narrative.
- **No**: Can be created from conceptual framing without code verification.
