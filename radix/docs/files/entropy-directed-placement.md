# Building an Index That Works Locally and Distributedly — and What We Found Along the Way

**Jordan — Datakey Pty Ltd / UNSW**

---

This started as an exercise in designing an optimal index.

The requirements: SIMD-friendly memory layout for efficient insertion and lookup, with uniformly distributed random key sets. The kind of index you'd want at the heart of a distributed synchronisation engine — fast locally, useful distributedly.

The question turned out to be about bit entropy and preferred placement. A well-understood approach — every hash table computes a preferred position from the key. What's less explored is what preferred placement is worth when you apply it at every level of a hierarchy, and how that value changes as the scope grows.

Starting small — within a single SIMD-width group of slots — preferred placement effectively disappears. The hardware scans 64 positions in one instruction. There's nothing to save by knowing which slot to check first. This is the floor.

Getting larger — across groups within a bucket, across buckets on overflow — preferred placement becomes an optimisation. Entropy bits direct the operation to the right group, the right overflow target. When the preferred position is taken, fresh bits redistribute to a new location. The cost of being wrong is a cache line, maybe two.

Getting larger still, and something shifts. Not in the mechanism — that's the same at every level. What shifts is the cost of checking. At the slot level, verifying whether the preferred position holds what you're looking for costs a byte read. At the bucket level, it costs a scan. At the node level, it costs a network round trip. The question "is this key at this position?" is probabilistic from the very first insertion — whether your preferred slot is occupied depends on load factor, nothing more. But at small scale, resolving the uncertainty is so cheap that it feels deterministic. At large scale, the cost of resolution is high enough that you'd rather act on the probability than pay to verify.

This is not only about collision, redistribution, and preferred next position for insertion. It is also about maybes. The domain of Bloom filters and cuckoo filters — probabilistic indexes that trade certainty for efficiency. What we found is that the boundary between a deterministic index and a probabilistic one isn't a line between local and distributed. It's a gradient, present from the first slot, governed by what it costs to check.

The task we set ourselves: can an index be designed to leverage absolute guarantees at small scale while allowing probabilistic guarantees at larger scale to be used for further optimisations? This might operate within a single index — using the same preferred-placement coordinates at every layer, resolving fully at the bottom, accepting probability at the top. Or it might operate across multiple indexes — parallel structures over the same key set, tuned for different points on the certainty-cost curve.

Some background context. The primary use case is a distributed key-value index where keys are stored across separate nodes. An index that provides optimal memory layout for SIMD operations at the core, efficient redistribution on collision at the bucket level, and useful probabilistic answers at the node level — without requiring a separate filter structure — is what motivated this work.

We discuss the architecture in terms of uniformly distributed keys, but note that the same structural pattern generalises to non-uniform distributions. Intentional clustering — where the hash function preserves locality rather than destroying it — would serve similarity search and near-neighbour queries through the same hierarchy. That direction is identified as future work; the architecture accommodates it, and this paper validates the foundation it would build on.

---

# An Optimised SIMD Radix Hash Index: Preferred Slot Placement from SIMD Floor to Distributed Cluster

---

## Preferred Placement at Every Layer

Every hash table computes a preferred position from the key. That's all hashing is — turn a key into an address, go there, deal with collisions. The mechanism is universal and ancient. What changes between designs is the collision strategy and the scale at which the initial placement operates.

This paper examines the value of preferred placement at every layer of a hierarchical index — from a single slot within a SIMD-width group, through groups within a bucket, buckets within a dataset, datasets within a node, up to nodes within a distributed cluster. The mechanism is the same at each layer: extract a segment of the hash, compute a target position. What varies is what that direction is worth.

At the SIMD floor — 64 fingerprints scanned in one instruction — preferred placement adds nothing measurable. Hardware already searches the full group as fast as it could check a single slot. We benchmark this and confirm it. This is the baseline.

As the structure scales beyond a single SIMD width, direction begins to pay. Four groups in a bucket means four possible cache line loads; two bits of hash entropy select the right one, avoiding three. At overflow, fresh hash bits select the next bucket independently rather than probing forward, preventing the clustering that degrades tail latency at high load. Each layer outward from the SIMD floor is a layer where direction replaces search, and the cost of undirected search grows with the space.

At the distributed boundary, the transition completes. A remote node's fingerprint array cannot be scanned locally. But the preferred coordinates can still be computed — bucket, group, slot — and a single byte read from a cached copy of the array yields a probabilistic membership answer. Empty slot: definite absence. Fingerprint match: probable presence. No scan, no round trip, no auxiliary filter structure. The primary index *is* the probabilistic filter, addressed positionally rather than scanned.

We build this index in Rust, layer by layer, benchmarking each addition. The results trace the value curve of a single mechanism applied at increasing scale — from redundant, to optimising, to enabling.

---

## Hash Bits as a Budget

<!-- 
- A single hash produces a fixed number of entropy bits. Every structural decision consumes some of them.
- Partition scheme: non-overlapping segments, each serving one purpose. No reuse, no correlation between decisions.
- Bit extraction left to right: bucket selection, group selection, preferred slot, fingerprint, overflow bits.
- Key size determines the total budget. 64-bit keys constrain the depth of independent addressing. 128-bit and 256-bit keys extend it.
- The extraction is parameterised, not hard-coded — different configurations consume different widths at each layer.
- Diagram: the hash laid out as a segmented bar, each segment labelled with its structural role.
-->

---

## The SIMD Floor

<!--
- The innermost layer: a group of 64 slots with one-byte fingerprints. 64 bytes = one cache line = one SIMD register.
- Arena layout: fingerprint arena (fp8 per slot, contiguous, aligned) and hash arena (u64 per slot, parallel). Separated for access pattern — SIMD touches only fingerprints, confirmation touches only hashes.
- Empty sentinel: 0x00 in the fingerprint arena. fp8 values of 0x00 remapped to 0x01.
- Preferred slot within the group: extract 6 bits, compute target, check first. If occupied, SIMD scan the full group.
- Benchmark: preferred slot vs SIMD-only. Show near-parity. The honest result — direction is redundant at SIMD width.
- Explain why: SIMD scans 64 bytes in ~1 cycle. A conditional branch (check preferred, then maybe SIMD) may cost more than always scanning. The floor is hardware-determined.

Benchmark data from M2 goes here.
-->

---

## Directing Above the Floor

<!--
4.1 — Multi-group buckets.

- Scale to 256 slots: 4 groups of 64. Fingerprint layout: groups contiguous within a bucket. Each group = one cache line.
- 2 entropy bits select the target group. Scan one group instead of four.
- Benchmark: directed vs sequential vs random group selection. Ablation isolating the direction value.
- Cache miss counts if available — the mechanistic explanation for throughput improvement.
- Discuss: fingerprint width reshapes this layer. fp16 = 32 slots per group = 8 groups per 256-slot bucket. More groups = more direction value. fp32 = 16 slots = 16 groups. Show the parameterisation.

Benchmark data from M3 goes here.

4.2 — Overflow.

- When all groups in a bucket are full: fresh-bit overflow vs linear next-bucket.
- Fresh bits compute an independent bucket target. Directed group selection applies within the overflow bucket.
- Redistribution property: elements colliding in bucket A scatter uniformly at the next round. No cascading clustering.
- Bit budget accounting: how many independent overflow rounds the hash supports. 64-bit hash limitations, 128-bit extensions.
- Benchmark: tail latency comparison at 90%, 95%, 99% load. Overflow round distribution.

Benchmark data from M4 goes here.

4.3 — System optimisations.

- Bucket-full flag: skip full groups without SIMD scan. Benchmark at high load.
- Prefetching: issue prefetch for overflow target while completing current scan.
- Concurrency: per-group or per-bucket locking on insert. Lock-free read path (fingerprint scan + hash confirmation are read-only). Thread scaling benchmarks.
- These are engineering contributions, not research — benchmarked inline for credibility.

Benchmark data from M5 goes here.
-->

---

## The Same Bytes, Differently

<!--
5.1 — The transition.

- The fingerprint array that SIMD scans locally can be addressed positionally from a remote node.
- No scanning — compute preferred coordinates (bucket, group, slot), read one byte.
- Three outcomes: empty = definite negative, fp match = probable positive, fp mismatch = probable negative (subject to displacement).
- This is not an optimisation of existing lookup. It's a capability scan-based designs cannot provide at distance.

5.2 — Single-probe and k-probe membership.

- Single-probe: preferred slot only. Fast, cheap, bounded confidence.
- k-probe: scan k additional slots in the group. Tunable confidence threshold from load factor.
- False positive rate: theoretical 1/255 for fp8, confirmed empirically.
- False negative rate: function of load factor and displacement probability. Measure across the full range.
- Accuracy-at-k-probes curve.

Benchmark data from M6 (accuracy) goes here.

5.3 — Fingerprint width as a network parameter.

- fp8: 1/255 FPR, 64 slots per group, 256 bytes per bucket summary.
- fp16: 1/65535 FPR, 32 slots per group, 512 bytes per bucket summary.
- fp32: ~1/4B FPR, 16 slots per group, 1024 bytes per bucket summary.
- Same stored hash, different projection for network exchange. Local arena stays fp8 for density; remote summaries widen for precision.
- Bandwidth cost analysis: bytes per query, bytes per summary exchange. Comparison against Bloom filter and cuckoo filter at equivalent FPR and bits per element.

Benchmark data from M6 (fingerprint sweep) goes here.

5.4 — Differential sync.

- Nodes cache snapshots of each other's fingerprint arrays. On mutation, exchange deltas — positions where fp has changed.
- Preferred-slot determinism: same key lands at same position on both nodes. Identical insertions reconcile trivially.
- Conflict resolution for displaced entries.
- CRDT-like merge property emerging from deterministic placement.

5.5 — Parallel indexes.

- Multiple indexes on the same key set with different fingerprint widths.
- A local fp8 index optimised for throughput alongside a fp16 projection optimised for distributed queries.
- Not two separate data structures — two views of the same hash values, laid out for different access patterns.
- Key size as a factor: 64-bit keys vs 256-bit keys. Larger keys provide more entropy budget for deeper hierarchies and wider fingerprints.
-->

---

## The Parameterised Design Space

<!--
- Generalise: the index is a family, not a single design. Three inputs determine the configuration:
  - Key size: total entropy budget. Constrains how many layers get independent addressing.
  - SIMD width: hardware constant. Determines the scan floor.
  - Fingerprint width: trades local slot density against distributed false positive rate.
- The dimensional hierarchy: cluster → node → dataset → bucket → group → slot. Same mechanism at every layer, parameters shift.
- Cost model: given these inputs, derive optimal fingerprint width per layer, groups per bucket, expected performance at each access pattern.
- Worked examples:
  - 64-bit keys on Apple Silicon (NEON, 128-bit registers). Practical configuration for a single-node index.
  - 256-bit keys across a wide-area cluster with high network latency. Where wider fingerprints and deeper hierarchies earn their cost.
-->

---

## Relationship to Recent Theory

<!--
- Focused context, not a survey. No standalone literature review.
- The cost decomposition: C = Σ f(k) × p(k). Inter-level spillover fraction × intra-level search cost.
- Farach-Colton, Krapivin, and Kuszmaul (2025): disproved Yao's 1985 conjecture. Funnel hashing and elastic hashing achieve better bounds by optimising f(k) through geometric sub-array sizing.
- This design: flat-array overflow with fresh-bit selection. Same total spillover, different structural accounting. Does not claim equivalence — trades asymptotic guarantee for simplicity and constant per-round cost.
- The probing theory lineage: Peterson (1957), Knuth (1963), Ullman (1972), Yao (1985). Where this work sits in the tradition — not contesting the theory, using its framework.
- Composability as an open question: could geometric sizing (optimising f(k)) compose with directed group selection (optimising p(k) above SIMD width)?
- Prior work referenced inline throughout the paper where each mechanism is introduced. Swiss Table context in "The SIMD Floor". Power-of-two-choices context in overflow discussion. Extendible hashing context in bit partitioning.
-->

---

## Open Questions

<!--
- Formal characterisation of displacement probability as a function of load factor, group size, and fingerprint width. This determines the false negative rate envelope for the probabilistic mode.
- Optimal fingerprint width for a given network cost model. The trade-off between summary bandwidth and false positive round trips has a minimum — what determines it?
- Composability with geometric sub-array sizing. Implement funnel hashing with directed group selection within each level. Does the improvement multiply?
- Behaviour under adversarial key distributions. The design assumes well-distributed hashes. What happens with skewed inputs and how does the hash function selection interact?
- The relationship between theoretical probe complexity and wall-clock cost when SIMD collapses intra-group search to constant time.
-->

---

## Concluding Remarks

<!-- 
- Summary: preferred placement is the universal mechanism. Its value depends on the distance from the SIMD floor.
- The practical result: an index that serves both local exact-match and distributed probabilistic membership from the same bytes.
- The milestones trace the value curve — from redundant to enabling — with benchmarks at each step.
- Open conjectures and formal questions for future work.
-->

---

## References

- Azar, Y., Broder, A. Z., Karlin, A. R., & Upfal, E. (1994). Balanced Allocations. *Proceedings of the 26th Annual ACM Symposium on Theory of Computing (STOC)*, 593–602.
- Bender, M. A., Conway, A., Farach-Colton, M., Kuszmaul, W., & Tagliavini, G. (2021). Iceberg Hashing: Optimizing Many Hash-Table Criteria at Once. *Journal of the ACM*, 70(6), 1–51. arXiv:2109.04548.
- Broder, A. Z., & Karlin, A. R. (1990). Multilevel Adaptive Hashing. *Proceedings of the 1st Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, 43–53.
- Fagin, R., Nievergelt, J., Pippenger, N., & Strong, H. R. (1979). Extendible Hashing — A Fast Access Method for Dynamic Files. *ACM Transactions on Database Systems*, 4(3), 315–344.
- Farach-Colton, M., Krapivin, A., & Kuszmaul, W. (2025). Optimal Bounds for Open Addressing Without Reordering. arXiv:2501.02305v2.
- Fotakis, D., Pagh, R., Sanders, P., & Spirakis, P. (2005). Space Efficient Hash Tables with Worst Case Constant Access Time. *Theory of Computing Systems*, 38(2), 229–248.
- Knuth, D. E. (1963). Notes on "Open" Addressing. Unpublished memorandum.
- Kulukundis, M. (2017). Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step. CppCon 2017. Implemented in Google Abseil as `flat_hash_map`.
- Pagh, R., & Rodler, F. F. (2004). Cuckoo Hashing. *Journal of Algorithms*, 51(2), 122–144.
- Pandey, P., Bender, M. A., Conway, A., Farach-Colton, M., Kuszmaul, W., Tagliavini, G., & Johnson, R. (2022). IcebergHT: High Performance PMEM Hash Tables Through Stability and Low Associativity. arXiv:2210.04068.
- Peterson, W. W. (1957). Addressing for Random-Access Storage. *IBM Journal of Research and Development*, 1(2), 130–146.
- Ullman, J. D. (1972). A Note on the Efficiency of Hashing Functions. *Journal of the ACM*, 19(3), 569–575.
- Vöcking, B. (2003). How Asymmetry Helps Load Balancing. *Journal of the ACM*, 50(4), 568–589.
- Yao, A. C. (1985). Uniform Hashing is Optimal. *Journal of the ACM*, 32(3), 687–693.

---

*Draft status: Preface and introduction written. Section content outlined in comments. Benchmark data to be inserted from milestone results. Prior work referenced inline where each mechanism is introduced.*
