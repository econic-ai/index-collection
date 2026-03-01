# Redesigning the humble radix hash index for GPU-accelerated distributed intelligence systems.

**Jordan — Datakey Pty Ltd / UNSW**

---

## Blog Introduction — Every book has a preferred place: Local exactness, Distributed Probability.

<!-- TONE: opinionated, quirky, passionate. This is the hook, not the research. Express the value proposition of the novel idea. -->

- If you've found yourself bit-wrangling on the SIMD floor, optimising operations in the nanosecond and picosecond range — this research may interest you.
- If you have a particular interest in the emergence of distributed intelligent systems, or recognise the game-changing nature of Apple Silicon's CPU-GPU architecture as a direct competitor to NVIDIA's market dominance, or found recent breakthroughs in computer science on space and time geometry fascinating — you may also find value in this line of exploration.

**The operational gap — local exactness vs distributed probabilistic certainty:**

- Using bit entropy and preferred placement.
- Moving away from guaranteed membership rules expected from local index data structures.
- Moving toward the realm of Bloom and cuckoo filters (partial one-sided guarantees) and vector spaces (geometric certainties on relationships rather than membership).
- The index's internal geometric structure provides key benefits over both approaches.
- Current iteration applies to uniformly distributed keys (16-byte hashes) — the realm usually served by Bloom and cuckoo filters.
- The same architectural principle may extend to clustered indexes such as vectors (LSH) and distinct sets such as database indexes and graphs — a direction explored briefly in §10 but not proven in this paper.

**What this research explores:**

- Redesign of a radix hash tree, a data structure commonly used in IP routing tables, file systems (Linux kernel radix trees), in-memory databases (ART in HyPer, DuckDB), and key-value stores (LMDB).
- Designed for **speed** — specifically traversal: find, iterate, compare.
- Designed for **algebraic predicates**, particularly as applied to Bayesian inference — A ∪ B, A ∩ B, A in B but not C.
- Designed for **fast delta operations** — identifying differences as distributions.
- Designed to **leverage GPU hardware** that performs these operations more effectively.

**Figure 0.1 — One index, three execution models**
*Conceptual diagram: a single fp8 byte array at centre. Three arrows diverge from it — CPU directed probe (single-key, nanosecond, scalar/NEON), GPU batch scan (full-array, coalesced, Metal compute), and distributed probabilistic query (remote cached summary, one-byte positional check). All three paths read the same physical bytes. The diagram primes the reader for the paper's core thesis: one structure, one memory layout, three modes of reasoning.*

<!-- VISUAL: not a data figure — a conceptual/architectural diagram. Should be clean, minimal, readable at blog-post width. -->

**Application — distributed databases:**

- Distributed indexes only have application in distributed databases and their many names — Content Distribution Networks (CDN) being one.
- Compute time on data structures directly determines responsiveness.
- Inefficient data transmissions lead to clogged operations.
- Clogging is not just about content distributed from node A to the network — it's about determining *where content should be*. It's about the coordination or intelligence layer itself.
- Particularly important in the frame of a CDN moving from pull-and-cache to proactive push on probability of utility.

**Apple Silicon framing:**

- Although not explored exhaustively in the paper, we design around Apple Silicon.
- Unified GPU-CPU memory architecture means: no dual indexes, no copying between CPU and GPU for operations better performed on one than the other.
- The comparison is CPU vs CPU+GPU on unified memory.
- General assertion (not proven): NVIDIA or similar discrete GPU would likely yield better raw GPU performance, but the additional costs of memory transfer or dual index storage can only degrade total system performance. Stated as a position, not a proof.

**Figure 0.2 — Unified memory vs discrete GPU memory architecture**
*Side-by-side comparison. Left: Apple Silicon unified memory — CPU and GPU share a single physical memory space, both read the same bytes at the same addresses, zero transfer overhead. The index exists once. Right: discrete GPU (NVIDIA/PCIe) — CPU and GPU have separate memory spaces connected by a bus. The index must either be duplicated (double storage, synchronisation cost) or transferred per operation (latency overhead). The diagram illustrates why unified memory makes the index a shared compute surface rather than a structure that must be copied between execution models.*

<!-- VISUAL: architectural diagram, not data. Should clearly show the single-memory vs dual-memory distinction and the cost implications for index operations. -->

**Closing segue:**

- The arena of software and data architectures is changing.
- Moving to systems built on probabilistic outputs — nowhere more evident than in the explosion of intelligent systems across the digital landscape.
- What was once absolutely guaranteed gives way to "within acceptable bounds based on this costing framework."
- How much do you want absolute certainty, meets: is 1% inaccuracy worth half the costs?

---

## 1. The Problem Space

<!-- STRATEGY: Technical framing of the problem. The blog intro gave the philosophical hook; this section gives the concrete engineering problem. A reader finishing §1 should understand: what two things are currently separate that shouldn't be, what that separation costs, and what unification would mean. -->

- Local index data structures (hash tables, B-trees, radix trees) provide exact answers: a key is here, or it is not. They are optimised for this — every byte of storage serves the lookup path.
- Distributed systems need a different answer: is a key *probably* on this node? That question is currently served by separate probabilistic structures — Bloom filters, cuckoo filters — maintained alongside the primary index.
- The dual-structure problem: two data structures, two storage budgets, two maintenance paths, two sync mechanisms. Every insert touches both. Every delete either can't update the filter (standard Bloom) or requires 4× the filter storage (counting Bloom). Every sync cycle transmits both structures or accepts staleness.
- The cost is not just storage — it is operational complexity. The filter and the index can diverge. The filter's guarantees degrade under mutation. The system must manage two lifecycles for one logical operation.
- The question this paper asks: can the primary index itself emit a probabilistic signal of sufficient quality to replace the auxiliary filter? Can one structure, one maintenance path, one sync mechanism serve both exact local lookup and probabilistic distributed reasoning?

**Figure 1.1 — The dual-structure problem**
*Two structures side by side: a primary index (serving exact local lookup) and an auxiliary probabilistic filter (serving distributed membership queries). Arrows indicate the dual costs: every insert writes to both, every delete is either impossible or expensive on the filter side, every sync cycle must transmit or rebuild the filter independently. The question mark between them represents the unification this paper explores.*

![Figure 1.1 — The dual-structure problem](images/whitepaper-figure-1-1-dual-structure-problem-v2.png)

---

## 2. The Geometry of Entropy: Placing Things Where They Can Be Found

<!-- STRATEGY: This section introduces the core theoretical mechanism — position as information. The reader should finish understanding: how a hash is partitioned into a bit budget, how those bits are spent on hierarchical positioning, how the remaining bits become a fingerprint, why independence between position and fingerprint is essential, and what discrimination results. This is the foundation for §7's signal equation and §9's architecture comparison. -->

- A hash is a bit budget. A 128-bit hash contains more entropy than any single data structure can consume. The design question is how to spend those bits — which segments determine where a key is placed, and which segments identify the key once it's there.
- The structure of bit entropy applied to a single flat array, partitioned into a hierarchy:
  - **Buckets** — coarse location, determined by the highest-entropy bit segment.
  - **Groups** — cache-line-aligned subdivisions within a bucket.
  - **Chunks** — NEON-width (16-byte) subdivisions within a group.
  - **Slots** — individual byte positions within a chunk, each holding one fp8 fingerprint.
- Preferred slot placement: a key's hash determines its preferred coordinate (bucket, group, chunk, slot offset), and a separate 8-bit segment of the hash becomes the fp8 fingerprint stored at that position.
- The independence requirement: position bits and fingerprint bits must be drawn from non-overlapping hash segments. If the fingerprint determines placement (same bits used for both), the total discrimination is only 8 bits. If position and fingerprint use independent bit segments, their discrimination multiplies — position narrows the search space, fingerprint confirms identity within that space.
- On collision (preferred slot occupied), the key can be directed to an alternative position using additional hash bits, maintaining random distribution properties without reusing the fingerprint bits.

**Table 2.1 — Discrimination: dependent vs independent bit allocation**
*Shows how total discrimination scales with the number of independent position bits. The fp8 fingerprint is always 8 bits stored; the position bits are implicit in the key's address within the hierarchy. When position and fingerprint are independent, their discrimination compounds.*

| Scheme | Stored bits | Position bits | Total discrimination | FPR |
|---|---|---|---|---|
| fp8 as placement (same bits) | 8 | 8 (same) | 8 | 1/255 |
| 4-bit offset + fp8 | 8 | 4 (independent) | 12 | 1/4,080 |
| 6-bit slot + fp8 | 8 | 6 (independent) | 14 | 1/16,320 |
| Full coordinate tuple + fp8 | 8 | 18+ (independent) | 26+ | 1/67M |

**Figure 2.1 — The hash as a segmented bar**
*A 128-bit hash visualised as a horizontal bar, divided into labelled segments: bucket bits, group bits, chunk bits, slot offset bits, fp8 bits, and remaining unused bits. Each segment is colour-coded to show which bits determine position (address) and which determine identity (fingerprint). The non-overlapping boundary between position and fingerprint segments is highlighted as the independence requirement.*

![Figure 2.1 — The hash as a segmented bar](images/whitepaper-figure-2-1-hash-segmented-bar-v2.png)

**Figure 2.2 — Positional entropy compounding through the hierarchy**
*Diagram showing how discrimination accumulates as a key's address is resolved through each level of the hierarchy: bucket narrows to one of many, group narrows further, chunk further still, slot offset to a single position. At each level, the bits consumed are independent of the fp8. The cumulative discrimination at the leaf (26+ bits) is annotated alongside the FPR (1/67M).*

![Figure 2.2 — Positional entropy compounding through the hierarchy](images/whitepaper-figure-2-2-positional-entropy-compounding-v2.png)

### Geometry of the layout

- The bucket → group → chunk → slot hierarchy is designed to align with hardware boundaries: groups are cache-line-sized (64 bytes), chunks are NEON-register-sized (16 bytes). This means that probing a group is one cache line fetch, and scanning a chunk is one SIMD load.
- The geometry maximises the use of bit entropy for preferred placement while dedicating the minimum necessary bits to collision detection via the fp8 fingerprint.
- This is the mathematically efficient use of the available bit budget for the chosen hardware constraints — a different cache line size or SIMD width would produce a different geometry from the same principle.

---

## 3. The Index Architecture

<!-- STRATEGY: This section describes the concrete implementation decisions. §2 established the principle (position is information, independence is essential); §3 shows how that principle is instantiated in a specific physical layout. Four design decisions, each with a rationale: bit sequence ordering, cache-line alignment, probe sequence, overflow direction. Each figure appears immediately after the point it illustrates. -->

- **Bit sequence ordering.** The hash bits are consumed in the order: bucket, fp8, group, chunk, preferred slot offset. The fp8 is positioned before the directional bits (group, chunk, offset) but after the bucket. This ordering facilitates recalculation on collision — when a preferred slot is occupied, the directional bits can be shifted to draw from unused hash segments for an alternative position, without disturbing the fp8 or bucket. Buckets do not need this recalculation because they are probed sequentially on overflow.

- **Cache-line alignment and the physical layout.** The fp8 array is laid out so that each group (64 bytes = 64 fp8 slots) aligns to a cache line boundary. A single group probe is guaranteed to be one cache fetch. Each group subdivides into 4 chunks of 16 bytes — one NEON register width. This means scanning a chunk is a single SIMD load, and scanning a full group is four SIMD loads within one cache line.

**Figure 3.1 — Physical layout: bucket → group → chunk → slot**
*A single bucket (256 bytes) = 4 groups (64 bytes each, cache-line-aligned), each group = 4 NEON-width chunks (16 bytes), each chunk = 16 fp8 slots. Annotations: cache line boundaries and NEON load widths.*

![Figure 3.1 — Physical layout: bucket → group → chunk → slot](images/whitepaper-figure-3-1-physical-layout-v2.png)

**Figure 3.2 — Hash bit partitioning for a concrete configuration**
*capacity_bits=20: Bucket (12 bits), group (2 bits), slot/chunk (6 bits → chunk 2 bits + offset 4 bits), fp8 (8 bits), overflow (34 bits spare). A concrete instantiation of the abstract segmented bar from Figure 2.1.*

![Figure 3.2 — Hash bit partitioning for a concrete configuration](images/whitepaper-figure-3-2-hash-bit-partitioning-capacity20-v2.png)

- **The probe sequence: power-of-k-choices across chunks.** A group contains 4 chunks. Rather than linearly probing through slots within a single chunk, the hash derives a different preferred offset in each of the four chunks — four independent shots at an empty preferred slot. On insert, the key takes the first available preferred slot. On lookup, all four are checked as scalar probes. If all four preferred positions are occupied, the probe falls back to a sequential NEON scan of the full group. This is balanced allocations (Azar et al., 1994) applied at the sub-bucket level — §4.3 validates the occupancy distribution empirically.

**Figure 3.3 — The probe sequence: power-of-k-choices across chunks**
*A single group (4 chunks). Hash derives different preferred offset per chunk. Scalar probe sequence: p0→p1→p2→p3. If all four occupied, NEON fallback.*

![Figure 3.3 — The probe sequence: power-of-k-choices across chunks](images/whitepaper-figure-3-3-probe-sequence-both-views-v2.png)

- **Overflow direction: same group, next bucket.** When a group is full (all preferred positions occupied and NEON scan finds no empty slot), the key overflows to the same group index in the next sequential bucket — not to the next group within the same bucket. The group coordinate is fixed by the hash; only the bucket advances. This preserves the (group, chunk, offset, fp8) coordinate tuple through overflow — the key's sub-bucket address and fingerprint are unchanged, only the coarse bucket location shifts.

**Figure 3.4 — Overflow direction: same group, next bucket**
*Group g in bucket b overflows to group g in bucket b+1, not group g+1 in bucket b. Group coordinate fixed by hash. Only the bucket advances. The (group, chunk, offset, fp8) tuple is preserved through overflow.*

![Figure 3.4 — Overflow direction: same group, next bucket](images/whitepaper-figure-3-4-overflow-same-group-next-bucket-v2.png)

- **Connection to the probabilistic signal.** The overflow strategy has a direct consequence for the signal quality characterised in §7. When a key is displaced from its preferred bucket, the fp8 and sub-bucket coordinates (group, chunk, offset) are preserved. A remote positional probe checking the preferred bucket will miss the key (contributing to the false negative rate), but the key retains enough of its coordinate tuple that a probe checking overflow buckets can still match on (group, chunk, offset, fp8) — the same 26+ bits of discrimination. The FNR model in §7 treats displacement as binary (at preferred position or not) rather than a gradient because the overflow design ensures that displacement degrades only the bucket coordinate, not the full positional signal.

---

## 4. Baseline Build

<!-- STRATEGY: Build layer by layer, measure each addition. Inline benchmarks per design decision. The reader sees empirical evidence alongside each architectural choice, not a separate results section. Honest accounting: where the radix index wins, where it loses, and why. -->

### 4.1 Against the Baseline

<!-- Comparison against hashbrown (Swiss Table) as production baseline at 256K index size. -->

- The baseline comparison is against hashbrown — Rust's default production hash map, based on Google's Swiss Table design (Kulukundis, 2017). It is SIMD-accelerated, heavily optimised, and battle-tested in production systems. If the radix index cannot justify itself against this baseline, the additional architectural complexity is not warranted.
- All comparisons in this subsection are at 256K index capacity, single-threaded, on Apple Silicon (M1).

*Personal Note: The radix index at certain operations will never beat a generic Swiss Table due simply to the fact that there are more SPU operations that must be performed. Having said that, the same is true in reverse — Swiss Table cannot compete on operations where the more efficient layout of information in memory matters.*

**Table 4.1 — Final benchmark matrix: radix tree vs hashbrown**
*Operations (insert, lookup hit, lookup miss, iteration) × load factors (1%, 25%, 50%, 75%). Each cell shows mean latency for both structures and the ratio. The radix tree is competitive on lookup, slower on insert (more bit manipulation per operation), and dramatically faster on iteration.*

![Table 4.1 — Final benchmark matrix](images/whitepaper-figure-4-8-final-benchmark-matrix-v2.png)

**Figure 4.1 — Latency progression across runs**
*Matrix line chart: columns = operations (insert, lookup hit, lookup miss, iter), rows = occupancy levels. Run-by-run radix-tree mean latency with optimisation-tag labels and hashbrown reference baseline. Shows the progressive improvement as each design decision (§4.2, §4.3) is applied.*

![Figure 4.1 — Latency progression across runs](images/whitepaper-figure-4-2-latency-progression-matrix-v3.png)

**Table 4.2 — Iteration performance: radix tree vs hashbrown**
*The headline result for the baseline comparison. The radix tree's contiguous fp8 layout produces 3–6× faster iteration across all load factors. The advantage is largest at moderate load (25%) where hashbrown's sparse slot array wastes the most cache fetches on empty control bytes.*

| Load | hashbrown | radix_tree | Ratio |
|---|---|---|---|
| 1% | 15.3 µs | 13.0 µs | 1.2× faster |
| 25% | 272 µs | 43.9 µs | 6.2× faster |
| 50% | 523 µs | 101 µs | 5.2× faster |
| 75% | 666 µs | 207 µs | 3.2× faster |

- The iteration advantage is not algorithmic cleverness — it is memory layout. The fp8 array is contiguous and dense; hashbrown's slot array is sparse with control bytes interleaved. A sequential scan of the fp8 array saturates the memory bus with useful data. This same property — contiguous, cache-friendly, sequential access — is what makes the GPU operations in §5 possible. The layout that wins at iteration on CPU is the layout that wins at batch operations on GPU.

### 4.2 The SIMD Floor vs Scalar Scans

<!-- STRATEGY: The crossover between scalar and SIMD probing is load-dependent. At low load, scalar wins because most preferred positions are empty — one check resolves. At high load, NEON wins because the probe must search further. The hybrid strategy (scalar first, NEON fallback) captures both. -->

- At low load, most slots are empty. A scalar check of the preferred position resolves almost immediately — the slot is empty, return "not found." SIMD scanning the full chunk is wasted work when the first byte answers the question.
- At high load, the preferred position is likely occupied by a different key. The probe must search further. NEON scanning 16 positions in parallel becomes cheaper than sequential scalar checks through occupied slots.
- The crossover between these two strategies is a function of load factor. The hybrid approach — scalar pre-check of the preferred position, NEON fallback only when needed — captures the best of both.

**Table 4.3 — Run 1 vs Run 2: SIMD scan vs scalar byte-by-byte**
*Where SIMD wins (high load) and where scalar wins (low load). The crossover point.*

| Load (α) | SIMD scan (ns) | Scalar byte-by-byte (ns) | Faster |
|---|---|---|---|
| 1% | (DATA PENDING) | (DATA PENDING) | — |
| 25% | (DATA PENDING) | (DATA PENDING) | — |
| 50% | (DATA PENDING) | (DATA PENDING) | — |
| 75% | (DATA PENDING) | (DATA PENDING) | — |

**Table 4.4 — Probe trace: where lookups resolve**
*At each load factor, what percentage of lookups resolve at each level of the probe sequence. The 99.5% scalar preferred hit rate at 50% load is the headline — almost all lookups are answered by a single byte read at the preferred position. This data feeds directly into §4.3's before/after comparison and §7's signal equation.*

| Load (α) | Resolved at preferred (scalar) | Resolved in chunk 0 (NEON) | Resolved in chunks 1-3 | Overflow to next bucket |
|---|---|---|---|---|
| 1% | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) |
| 25% | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) |
| 50% | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) |
| 75% | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) |

**Table 4.5 — Run 4 vs Run 5: pure NEON vs scalar-first with NEON fallback**
*The hybrid strategy recovers low-load performance (scalar resolves immediately) while retaining high-load performance (NEON available when the scalar pre-check fails).*

| Load (α) | Pure NEON (ns) | Scalar-first + NEON fallback (ns) | Improvement |
|---|---|---|---|
| 1% | (DATA PENDING) | (DATA PENDING) | — |
| 25% | (DATA PENDING) | (DATA PENDING) | — |
| 50% | (DATA PENDING) | (DATA PENDING) | — |
| 75% | (DATA PENDING) | (DATA PENDING) | — |

### 4.3 Linear Probing, Clustering, and the Power of k Choices

<!-- STRATEGY: TWO-PART NARRATIVE.
     Part 1: The problem — linear probing and Knuth degradation.
     Part 2: The solution — power-of-k-choices as before/after, demonstrated through occupancy data.
     Closes with GPU implications — why placement matters for CPU but not for GPU. -->

#### The problem: linear probing within a single chunk

- At 75% load, Knuth's analysis of linear probing predicts an expected 8.5 slots scanned on a miss. This is the clustering problem: occupied runs grow, adjacent insertions extend those runs, and probe lengths increase superlinearly with load factor.
- With linear probing within chunk 0 only, the chunk absorbs all insertions sequentially. At high load, probe chains extend deep into the chunk, pushing overflow into adjacent buckets. The occupancy distribution is skewed — chunk 0 is heavily loaded while chunks 1-3 are underutilised.

**Table 4.6 — Knuth's predicted probe lengths vs observed (linear probing baseline)**
*Knuth's theoretical predictions for linear probing alongside observed occupancy distribution with single-chunk linear probing. The predicted values are from the standard analysis (Knuth, 1963); the observed values validate the degradation pattern.*

| Load (α) | Knuth predicted (miss) | Observed: chunk 0 only | Observed: chunks 1-3 | Observed: overflow |
|---|---|---|---|---|
| 25% | 0.89 slots | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) |
| 50% | 2.5 slots | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) |
| 75% | 8.5 slots | (DATA PENDING) | (DATA PENDING) | (DATA PENDING) |

#### The solution: independent preferred offsets across chunks

- Power-of-k-choices (Azar et al., 1994): instead of probing linearly within one chunk, derive a different preferred offset per chunk from independent hash bit segments. Four independent shots at an empty preferred slot. On insert, take the first available. On lookup, check all four as scalar probes before falling back to NEON.
- The key insight is that this transforms the occupancy distribution. Rather than one long probe chain in chunk 0, load distributes across four independent positions in four chunks.
- The evidence is in the occupancy data, not the latency numbers directly:
  - **BEFORE** (linear within chunk 0): at 75% load, chunk 0 absorbs all insertions sequentially. Overflow follows the linear probing pattern. Probe chains correlate with cluster size.
  - **AFTER** (k=4 across chunks): at 75% load, each chunk independently has a 75% chance of its preferred slot being occupied, but the probability that ALL FOUR preferred slots are occupied is 0.75⁴ = 31.6%. The occupancy data should show ~68.4% of keys placed at a preferred slot in one of the four chunks (resolved by scalar probe), ~31.6% requiring NEON fallback, and dramatically reduced overflow into adjacent buckets.
- The probe trace data (Table 4.4) already shows the AFTER picture. Re-presenting it here alongside the linear probing baseline (Table 4.6) makes the before/after visible.
- The four chunk offsets must come from independent hash bits. If they were correlated, the four probes would cluster together and the advantage disappears. This is the same independence requirement that §2 establishes for position vs fingerprint bits — applied here at the chunk level.

**Figure 4.2 — Chunk occupancy distribution: linear probing vs power-of-k-choices**
*Side-by-side or overlay at each load factor. Left/before: linear probing within chunk 0 — clustering in chunk 0 with long tails into overflow. Right/after: power-of-k-choices across four chunks — even distribution with sharply reduced overflow. The visual makes the occupancy shift immediately apparent.*

![Figure 4.2 — Chunk occupancy distribution at each load factor](images/whitepaper-figure-4-1-chunk-occupancy-distribution-v2.png)

**Table 4.7 — P(all k preferred positions occupied) by load factor**
*Theoretical prediction from the balanced allocations model. At k=4 and 75% load, only 31.6% of insertions exhaust all preferred positions and require NEON fallback. The observed occupancy distribution (Figure 4.2) validates these predictions. The gap between k=1 (75%) and k=4 (31.6%) is the entire value of the optimisation.*

| Load (α) | k=1 | k=2 | k=4 |
|---|---|---|---|
| 25% | 25% | 6.3% | 0.4% |
| 50% | 50% | 25% | 6.3% |
| 75% | 75% | 56.3% | 31.6% |

#### Implications for GPU operations

- On GPU, the full-array scan is effectively a linear probe over the entire array — occupancy distribution does not affect scan cost because every slot is visited regardless of whether it is occupied.
- For CPU directed probes, the power-of-k-choices is essential: it determines how quickly a lookup resolves and how often NEON fallback is needed. For GPU batch operations, placement is irrelevant — the GPU reads everything in a single coalesced pass.
- This is another angle on the CPU/GPU complementarity established in §5: CPU benefits from entropy-directed placement because it probes selectively. GPU does not care about placement because it reads everything. Both benefit from the same index structure for different reasons — CPU from the hierarchical geometry, GPU from the contiguous memory layout.

---

## 5. Enter GPU into GPU-Friendly Operations

<!-- STRATEGY: The measurement section. Every subsection presents GPU benchmark results for a specific operation, using two complementary visualisations where possible: a bar chart showing absolute CPU vs GPU latency (the "what happened" view), and a normalised GPU/CPU ratio line chart (the "crossover shape" view). Tables are used only where specific numbers matter more than the visual trend (greedy search counterexample, final summary). The section closes with an overlaid ratio chart across all operations — the headline figure showing operation-shape dependence. -->

*Personal Note: Baselining GPU operations without utilising Apple's unified architecture. Results did show improvements compared to CPU operations at sufficient loads. On the whole they could only be worse than the same operations on GPUs optimised for unified memory, and so were not included in the results.*

### 5.1 Dispatch Overhead and the Crossover Profile

<!-- Merged from original 5.1 (Index Size Matters) and 5.2 (GPU/CPU Factor Performance). -->

- GPU utilisation carries a fixed overhead: kernel launch, buffer binding, thread group setup. At small index sizes (e.g. 256K), this overhead dominates — the per-element work is too small to amortise the dispatch cost. The GPU is slower, sometimes dramatically so.
- As index sizes expand, the fixed cost is amortised across more elements. The GPU architecture — designed for large parallelisable operations with coalesced memory access — begins to show its advantage.
- The crossover from GPU-slower to GPU-faster is the fundamental characteristic of every operation in this section. Iteration is the simplest full-array operation and establishes the baseline profile against which all other operations are compared.

**Figure 5.1 — Iteration: absolute GPU vs CPU latency by index size**
*Bar chart. X-axis: index size (256K, 1M, 8M, 128M). Paired bars: CPU NEON (one colour) and GPU Metal (another). At 256K the GPU bar towers over CPU. At 1M they approach parity. At 8M+ the GPU bar is visibly shorter. The dispatch overhead is tangible as the gap at small sizes.* (DATA PENDING)

<!-- VISUAL: grouped bar chart, log scale on Y-axis may be needed given the range from ns to ms. -->

**Figure 5.2 — Iteration: GPU/CPU ratio by index size**
*Line chart. X-axis: index size (log scale). Y-axis: GPU/CPU ratio. Horizontal parity line at 1.0. The curve starts well above 1.0 (GPU slower), crosses parity around 1M, and descends to ~0.13 at 128M (GPU 7.5× faster). This is the reference curve for all subsequent operations.* (DATA PENDING)

<!-- VISUAL: clean single-line chart. The parity line and crossover point should be annotated. -->

### 5.2 Framing the Greedy Search

- The greedy search (`contains_greedy`) is the counterargument: GPU linear scan looking for a specific key, compared to CPU directed probe with early exit.
- On CPU, the directed probe checks the preferred position first. If the key is there (or the slot is empty), the lookup returns immediately — one byte read. The expected probe depth is a function of load factor and k (§4.3). At low load, most lookups resolve in 1–2 probes.
- On GPU, the linear scan visits every slot regardless. There is no early exit. The full array must be read before the answer is known.
- The CPU's early-exit advantage means the GPU almost never wins on single-key lookup. The crossover only occurs at very large sizes and high occupancy, where the CPU's probe chains lengthen enough to approach the GPU's fixed full-scan cost.

**Table 5.1 — Greedy search: GPU vs CPU by index size and occupancy**
*The counterexample. Specific numbers matter here — the magnitudes make the early-exit advantage visceral. At 256K/0.90, CPU is ~100× faster. The GPU only wins at 128M/0.90 with 3.94× advantage. Kept as a table because the raw numbers are the argument.*

| Index size | Occupancy | CPU directed (ns) | GPU greedy (ns) | GPU/CPU ratio |
|---|---|---|---|---|
| 256K | 0.10 | (DATA PENDING) | (DATA PENDING) | — |
| 256K | 0.90 | (DATA PENDING) | (DATA PENDING) | — |
| 1M | 0.10 | (DATA PENDING) | (DATA PENDING) | — |
| 1M | 0.90 | (DATA PENDING) | (DATA PENDING) | — |
| 8M | 0.10 | (DATA PENDING) | (DATA PENDING) | — |
| 8M | 0.90 | (DATA PENDING) | (DATA PENDING) | — |
| 128M | 0.10 | (DATA PENDING) | (DATA PENDING) | — |
| 128M | 0.90 | (DATA PENDING) | (DATA PENDING) | — |

- The tension: if hardware-accelerated linear scan matches entropy-directed probing at sufficient scale, the value of positional encoding becomes a function of table size and available compute. At 128M slots with high occupancy, the GPU's brute-force approach wins despite scanning everything. This does not invalidate the positional design — it shows that the CPU and GPU benefit from the same index for different reasons (§4.3 GPU implications).

### 5.3 Probabilistic Differential and Cardinality

- Fingerprint diff (m1.1): given two fp8 arrays (a current index and a cached copy), produce a bitmask of changed positions. This is the differential sync primitive — §8.1 grounds it in the distributed coordination context.
- The crossover profile is expected to match iteration: same access pattern (sequential coalesced pass over the full array), uniform work per element (XOR + compare). The bar chart confirms or departs from this expectation.
- Cardinality as a natural outgrowth: the popcount of the diff bitmask gives the number of changed positions. This is a free reduction — no additional pass, no additional storage.
- The cardinality feeds into decisions about what to do with the diff — §8.1 grounds this in the coordination context.
- At scale (10M+ slots), the GPU produces both the diff bitmask and the cardinality in a single pass, firmly in GPU-advantage territory.

**Figure 5.3 — Fingerprint diff: absolute GPU vs CPU latency by index size**
*Bar chart, same format as Figure 5.1. Confirms whether the crossover profile matches iteration or deviates (e.g. due to the XOR + compare per element vs simple read).* (DATA PENDING)

**Figure 5.4 — Fingerprint diff: GPU/CPU ratio by index size**
*Line chart, same format as Figure 5.2. Overlaying or comparing against the iteration reference curve shows whether the diff operation tracks iteration or diverges.* (DATA PENDING)

### 5.4 Algebraic Predicates

- Set intersection (m1.2): positions where both arrays hold the same non-empty fp8. The cardinality of the intersection — the similarity ratio — is a single-number reduction. §8.2 shows how this feeds coordination decisions. Stream compaction produces the position list as a secondary output.
- Multi-set predicate evaluation (m1.3): arbitrary boolean predicates across N co-indexed arrays. Intersection, difference, consensus, unique ownership — any predicate expressible as a mask over the per-position state. A single pass over N arrays evaluates the predicate at every position.
- The 2-array and 3-array operations were benchmarked separately. The 3-array operations show better GPU advantage than 2-array, which in turn show better advantage than iteration. This is the expected arc: more arrays means more memory reads per position, more work to amortise the dispatch overhead, earlier crossover and steeper advantage at scale. The GPU's bandwidth advantage compounds with the number of arrays being evaluated.

**Figure 5.5 — Algebraic predicates (2-array): absolute GPU vs CPU latency by index size**
*Bar chart. Covers set intersection and 2-array difference/symmetric difference. Paired bars at each index size.* (DATA PENDING)

**Figure 5.6 — Algebraic predicates (3-array): absolute GPU vs CPU latency by index size**
*Bar chart. Covers 3-array consensus, unique ownership, and other 3-array predicates. The GPU bars should be visibly more advantageous relative to CPU than in the 2-array chart.* (DATA PENDING)

**Figure 5.7 — Algebraic predicates: GPU/CPU ratio by index size, 2-array vs 3-array**
*Line chart. Two line groups: 2-array operations and 3-array operations. The 3-array lines should sit below the 2-array lines (better GPU advantage) at every index size, and cross parity earlier. The vertical separation between the two groups is the empirical demonstration that GPU advantage scales with work-per-position.* (DATA PENDING)

### 5.5 GPU Conclusions

- Operation shape determines GPU value: full-array coalesced scans cross over early; scattered probes (greedy search) cross over late or never. The GPU accelerates *batch reasoning over arrays*, not individual lookups.
- GPU advantage scales with work-per-position: the 2-array → 3-array improvement arc demonstrates this empirically. More arrays, more reads per position, more bandwidth to amortise, better GPU advantage.
- The m1 operations (diff, intersection, multi-set predicates) are not standalone benchmarks — they are the computational primitives that §8 will show drive coordination in distributed systems.
- On unified memory, CPU and GPU paths read the same physical bytes. The index is a shared compute surface: CPU for directed single-key probes, GPU for batch array operations, no copy between them.

**Figure 5.8 — GPU/CPU ratio by index size: all operations overlaid**
*The headline figure for §5. Line chart with all operations on one set of axes: iteration (baseline), fingerprint diff, 2-array predicates, 3-array predicates, greedy search. Parity line at 1.0. The reader sees three things at once: (1) coalesced full-array operations cluster together and cross over early, (2) 3-array lines sit above 2-array lines showing the work-per-position scaling, (3) greedy search sits below everything, crossing over only at extreme scale if at all. The operation-shape dependence of GPU value is the visual takeaway.* (DATA PENDING)

<!-- VISUAL: this is the single most important figure in §5. Clean, well-labelled, with the parity line prominent. Legend should group operations by type (full-array coalesced vs directed probe). -->

**Table 5.2 — GPU crossover summary**
*Consolidation table. One row per operation. The reader who wants one reference point per operation goes here.*

| Operation | Array count | Crossover size (GPU ≈ CPU) | GPU advantage at 128M |
|---|---|---|---|
| Iteration | 1 | (DATA PENDING) | (DATA PENDING) |
| Fingerprint diff | 2 | (DATA PENDING) | (DATA PENDING) |
| Set intersection | 2 | (DATA PENDING) | (DATA PENDING) |
| Multi-set predicate | 2 | (DATA PENDING) | (DATA PENDING) |
| Multi-set predicate | 3 | (DATA PENDING) | (DATA PENDING) |
| Greedy search | 1 | (DATA PENDING) | (DATA PENDING) |

---

## 6. Probabilistic Architectures in the Wild

<!-- STRATEGY: Landscape section. What exists, how each works, what each costs. No comparison to the positional probe — that is §7's job. Sets up the vocabulary, the terminology (FPR/FNR), and the problem space so the reader understands what the field currently offers before we introduce ours. Opens with a segue from the exact/deterministic world of §4–5 into the probabilistic world of §6–9. -->

- Up to this point, the paper has dealt in exact answers: a key is at this position, this lookup took this many nanoseconds, this operation crosses over at this index size. Deterministic, measurable, local. But distribution introduces a fundamental constraint — no single node has complete knowledge.
- In a distributed system, the question changes from "is this key here?" to "is this key *probably* on that node?" This is not a degradation; it is the natural language of coordination under incomplete knowledge. The field has operated on this principle for decades: Bloom filters (1970), consistent hashing, gossip protocols, vector similarity search.
- The following subsections survey the tools that currently serve this probabilistic layer — what they provide, what they cost, and what gaps remain. §7 then shows how the index architecture from §2–3 provides a different answer to the same questions.

### 6.1 Bloom and Cuckoo Filters

- **Bloom filters.** A bit array with k hash functions. On insert, k bit positions are set. On query, all k positions are checked — if any is unset, the key is definitely absent. If all are set, the key is *probably* present. The structure is compact (typically 10 bits per key for ~1% error rate) and query cost is fixed (k hash computations + k bit reads).
- **Cuckoo filters.** A compact alternative that stores fingerprints in a cuckoo hash table. Supports deletion (unlike standard Bloom), achieves similar or better FPR at lower space for many configurations, and provides constant-time lookup. But like Bloom, it is a structure *separate* from the primary index — a second data structure with its own storage, its own maintenance path, and its own sync lifecycle.
- Costs common to both:
  - Separate from the index — different data structure, different storage, different maintenance path. Every mutation to the primary index requires a corresponding mutation to the filter.
  - **Cannot communicate in deltas.** A change to the underlying set requires retransmission of the full filter or a full rebuild, not a sparse positional update.
  - Standard Bloom cannot delete. Counting Bloom supports deletion but at 4× the storage (40 bits per key). Cuckoo filters support deletion but with capacity constraints and potential failure under high churn.

**Terminology — two kinds of wrong answer:**

- **False positive (FP):** the structure says "yes, this key is present" when it is not. Measured as the false positive rate (FPR). This is the error Bloom and cuckoo filters are designed around — tuneable, bounded, well-characterised.
- **False negative (FN):** the structure says "no, this key is absent" when it is present. Measured as the false negative rate (FNR). Bloom and cuckoo filters guarantee zero false negatives by construction. This is their defining property.
- But the zero-FNR guarantee only holds for the local, fully-maintained filter. In a distributed system with mutation:
  - After deletes (standard Bloom): deleted keys remain as phantom positives. The FPR degrades monotonically. The filter accumulates wrong "yes" answers it cannot retract.
  - After stale sync: a remote node inserts new keys, but the locally cached filter has not been updated. The filter says "no" for keys that now exist — effectively false negatives from the system's perspective. The one-sided guarantee erodes to two-sided in practice.
- The positional probe (§7) is honest about having both FP and FN from the start. It does not claim a one-sided guarantee that degrades under real conditions. Instead, it provides a characterised two-sided error model where both rates are quantifiable functions of known parameters (load factor, probe depth). The system designer sees the full error budget up front rather than discovering hidden FN costs at runtime.

**Table 6.1 — Probabilistic structures: capabilities and costs**
*Landscape summary of existing approaches before the positional probe is introduced. All values are from established literature. This table is the "before" picture — §9 will extend it with the positional probe for the full architecture comparison.*

| Structure | FPR | FNR | Supports delete | Storage per key | Delta sync | Separate from index |
|---|---|---|---|---|---|---|
| Standard Bloom | ~1% at 10 bits/key | 0% (local) | No | 10 bits | No (full retransmit) | Yes |
| Counting Bloom | ~1% at 40 bits/key | 0% (local) | Yes | 40 bits | No (full retransmit) | Yes |
| Cuckoo filter | <1% at ~12 bits/key | 0% (local) | Yes (capacity-limited) | ~12 bits | No (full retransmit) | Yes |

### 6.2 Vector Similarity and Geometric Probability

- Describe the approach: embedding items in a vector space, similarity as geometric proximity. Nearest-neighbour search replaces exact membership — the question becomes "what is closest to this?" rather than "is this here?"
- Address fundamental differences — solving a different problem (similarity, not membership), not readily associated with distributed key-value databases.
- Describe fundamental sameness — addressing probabilistic relationships via geometry. The principle that *position encodes information* appears in both domains. A hash index uses positional encoding for membership; a vector index uses positional encoding for similarity. The mechanism differs, the principle is shared.
- Illustrate examples in the wild: Milvus distributed vector search, algorithms like LSH that provide similarity certainties of diminishing confidence as the bit comparison reduces.
- This subsection exists in a hash index paper because the shared principle — lossy positional encoding as a probabilistic reasoning surface — is the generalisation that §7.4 will make explicit. Vector similarity is out of scope for direct application here, but on the table as a shared architectural principle and a future direction (§10).

### 6.3 The Distributed Database Problem

- The coordination layer problem: it is not just about storing data, it is about determining *where data should be* and *what has changed*.
- This problem is concrete and well-known in existing systems:
  - Cassandra uses Bloom filters for SSTable lookup — avoiding unnecessary disk reads by probabilistically ruling out SSTables that do not contain a key.
  - Dynamo uses Merkle trees for anti-entropy repair — detecting divergence between replicas by comparing hash tree roots.
  - CDN routing uses consistent hashing with Bloom-based membership caches — routing content requests to the node most likely to hold the content.
- Each of these systems maintains a separate probabilistic structure alongside its primary data. Each pays the dual-structure cost described in §1.
- Pull-and-cache vs proactive push on probability of utility. Traditional CDN architecture waits for a request, then fetches. A coordination layer that knows *what has changed* and *how much overlap exists* between nodes can proactively push content where it is likely to be needed — shifting from reactive to anticipatory.
- The tools in §6.1 and §6.2 each address fragments of this problem. No single existing structure provides membership testing, delta synchronisation, set-algebraic inference, and cardinality estimation from the same bytes.

---

## 7. Probabilistic Inference from Structural Proximity

<!-- STRATEGY: The theoretical core. The index emits a probabilistic signal as a structural byproduct of independent positional encoding. This section characterises that signal: what it is, what each observation tells you, how observations accumulate, and where the same pattern applies beyond coordination. The value is match quality — independence makes matches trustworthy. Each subsection opens with a plain statement of what it establishes. -->

### 7.1 The Signal Equation

The index stores one fp8 byte per slot. That byte, combined with the slot's position in the hierarchy, serves two roles: local exact lookup and remote probabilistic signal. This subsection establishes why the same byte can do both, and identifies the independence requirement as the mechanism that makes the probabilistic signal trustworthy.

**Figure 7.1 — Same bytes, two access modes**
*The same fp8 byte array viewed from two perspectives. Left: local exact mode — the probe sequence continues through the hierarchy until empty or verified match against the full key. Right: remote probabilistic mode — each position is an independent observation yielding one of three outcomes (empty, match, mismatch) with quantifiable evidential weight. The bytes are identical; the interpretation differs.*

![Figure 7.1 — Same bytes, two access modes](images/whitepaper-figure-5-1-same-bytes-two-access-modes-v2.png)

- Locally, the fp8 is a filter on the hot path: check the fingerprint before fetching the full key from the hash arena. If the fp8 doesn't match, skip the expensive verification. If it does, verify.
- Remotely, the fp8 is the *only* information available. A node holding a cached copy of another node's fp8 array cannot verify against the full key — it can only observe the fp8 value at a given position and draw probabilistic conclusions.
- The signal equation: P(correct answer) as a function of load factor (α), probe depth (k), fingerprint size (f), and discrimination depth (d). <!-- FLAG: needs explicit formula at copy stage. -->
- The quality of this signal depends entirely on the independence requirement established in §2: position bits and fingerprint bits are drawn from non-overlapping hash segments. A key's position in the hierarchy and its fp8 value are statistically independent events. This independence is what separates a useful signal from noise — as §7.2 shows.

### 7.2 The Three Observations

At any probe position, three outcomes are possible. This subsection characterises the evidential weight of each and shows how they accumulate over multiple probes. The central result: empty is a definitive miss, match is strong evidence whose quality depends on the independence requirement, and mismatch carries exactly zero information.

- **Empty (0x00 sentinel):** If key K exists in the index, every slot before and including K's position in the probe sequence must be occupied — either by K itself or by the key that displaced K during insertion. An empty slot is impossible if K exists. Therefore: P(empty | K exists) = 0. Empty is a **definitive miss**, regardless of load factor, probe depth, or bit allocation.
- **fp8 match:** Either K is at this position, or a different key occupies K's position and happens to share K's fp8. With independent bit allocation, the occupying key's fp8 is uniformly distributed among 255 non-zero values — it matches K's fp8 with probability 1/255. With the full coordinate tuple (bucket + group + chunk + offset + fp8), discrimination is 26+ bits, and the false positive probability drops to 1/67M. A match is **strong evidence** that K exists here.
- **fp8 mismatch:** The slot is occupied by a different key with a different fp8. This outcome is **exactly equally likely** whether K exists or not: P(mismatch | K exists) = α × 254/255 = P(mismatch | K absent). The likelihood ratio is 1:1. A mismatch carries zero information about K's existence. Continue probing.

**Table 7.1 — Per-probe observation probabilities**
*The three possible observations at any single independent probe position, with conditional probabilities under each hypothesis and the resulting likelihood ratio. Empty is definitive. Match is strong. Mismatch is worthless.*

| Observation | P(obs \| K exists) α=0.50 | P(obs \| K absent) α=0.50 | LR α=0.50 | P(obs \| K exists) α=0.75 | P(obs \| K absent) α=0.75 | LR α=0.75 |
|---|---|---|---|---|---|---|
| Empty | 0 | 0.500 | ∞ → miss | 0 | 0.250 | ∞ → miss |
| fp8 match | 0.502 | 0.00196 | 256:1 | 0.253 | 0.00294 | 86:1 |
| fp8 mismatch | 0.498 | 0.498 | 1:1 | 0.747 | 0.747 | 1:1 |

- Because mismatch is non-informative, seeing one does not change the probability of K's existence. The posterior after a mismatch equals the prior. Each independent preferred position is a fresh observation with identical odds — mismatches do not accumulate evidence in either direction.

The independence requirement is what makes match trustworthy. Without it — if fp8 determines placement — every key at a given position shares the same fp8 by construction. A match has a likelihood ratio near 1:1 and tells you almost nothing.

**Table 7.2 — Match discrimination: dependent vs independent bit allocation**
*The same data as Table 2.1, restated as the consequence for match quality. Independence transforms a match from meaningless to near-certain.*

| Scheme | Match likelihood ratio | FPR on match | Meaning |
|---|---|---|---|
| fp8 as placement (dependent) | ~1:1 | ~100% | Match is meaningless — always requires full verification |
| 4-bit offset + fp8 (independent) | ~16:1 | 1/4,080 | Moderate evidence |
| 6-bit slot + fp8 (independent) | ~64:1 | 1/16,320 | Strong evidence |
| Full coordinate tuple + fp8 (independent) | ~67M:1 | 1/67,000,000 | Near-certain — actionable without verification in most contexts |

- In the distributed context, a remote node checks K's k preferred positions in a cached fp8 array. Whether K is found at one of those positions is a function of placement probability — already characterised by §4.3's occupancy distribution (Table 4.7). What §7.2 adds is the *quality* of that finding: when a match occurs, independence gives it 26+ bits of discrimination, making it actionable without full key verification. The same 26 bits that make the signal actionable also define the operating vocabulary of the coordination protocols in §8 — sufficient for coordination decisions, without requiring the full key.

### 7.3 The Declining Guarantee

The signal quality described in §7.2 applies at independent preferred positions — slots derived directly from the hash via §3's power-of-k-choices. When the probe extends beyond those positions into the linear overflow path, discrimination decreases. This subsection characterises the gradient.

- Each of the k independent preferred positions carries the full coordinate tuple: bucket + group + chunk + offset + fp8 = 26+ bits of discrimination. At these positions, the per-probe analysis from §7.2 applies without modification.
- When a key overflows into sequential positions (NEON scan within a chunk, overflow to next bucket), the slot is no longer hash-derived. The fp8 is preserved (§3, overflow direction), but the positional bits no longer contribute independent discrimination. Match quality at overflow positions is lower — the fp8 alone provides 8 bits (1/255), not 26+.
- Verification cost as the governing variable: at 1/67M FPR (preferred position), most applications can act on the match without verification. At 1/255 FPR (overflow position), verification may be warranted depending on the cost of a false positive.
- This is not a failure mode — it is a design parameter. The system chooses its operating point on the gradient: how many preferred positions to check (k), whether to extend into overflow, and when to pay for full verification.

**Figure 7.3 — The verification cost gradient**
*Confidence decreases as the probe moves from preferred positions (full coordinate discrimination, 26+ bits) outward through overflow positions (fp8 only, 8 bits). At each level, the match FPR is quantifiable. The system decides whether that FPR justifies acting on the signal or paying for full key verification.*

![Figure 7.3 — The verification cost gradient](images/whitepaper-figure-7-1-verification-cost-gradient-v2.png)

### 7.4 Beyond Coordination: The General Primitive

The per-probe analysis and verification gradient above are described in terms of index keys and distributed nodes. But the underlying operation is not specific to any of these. The same pattern — N large arrays of lossy positional encodings, a boolean predicate, a GPU pass, a result with quantifiable confidence — appears wherever you need to reason about massive datasets without complete information.

*Personal Note: The coordination layer in mgraph is where I first recognised this pattern, but the primitive is broader than any single system. What we have is parallel hypothesis testing over incomplete knowledge at memory-bandwidth speed.*

- The core primitive generalised: given N arrays where each element is a lossy positional encoding, evaluate arbitrary boolean predicates across them in a single GPU pass, with quantifiable confidence on every result.
- **Anomaly detection over distributed logs:** N nodes publish fp8 summaries of observed events. Difference predicates identify event signatures localised to one node. One confirmed anomaly triggers investigation.
- **Federated learning model divergence:** participants publish positional fingerprint summaries of updated parameter buckets. Consensus and difference predicates detect convergence, divergence, and potential poisoning without exchanging full parameters.
- **Graph neighbourhood comparison:** set intersection of fp8 arrays around a vertex estimates shared neighbourhood structure across graph fragments without materialising full adjacency lists.
- **Approximate join estimation in query planning:** intersection cardinality between table partition summaries estimates join output size, feeding query optimiser cost models.
- **Record linkage across organisations:** approximate set intersection at memory-bandwidth speed. The computational pattern is identical to any other intersection predicate — §8.4 discusses the coordination implications for actors operating on incomplete knowledge.
- **Sensor fusion with disagreeing sources:** multi-set predicates over sensor summaries identify agreement (high confidence detections), disagreement (noise or occlusion), and partial agreement (probable detection, needs confirmation). The quantifiable FPR per predicate maps directly to confidence levels.

**Table 7.3 — The general primitive across domains**
*The same computational pattern across six domains. What changes is semantic interpretation — what a false positive means, what an unresolved query means, and how much coverage the application requires. The operation, the confidence model, and the GPU execution are identical.*

| Domain | Primary predicate | What FP means | What unresolved means | Tolerance |
|---|---|---|---|---|
| Anomaly detection | A \ (B ∪ C ∪ D) | False anomaly alert | Missed anomaly | High |
| Federated learning | A ∩ B ∩ C | Falsely reported convergence | Undetected convergence | Moderate |
| Graph comparison | A ∩ B | Falsely shared neighbour | Undetected shared neighbour | High |
| Join estimation | A ∩ B | Overestimated join size | Underestimated join size | High |
| Record linkage | A ∩ B | False record match | Undetected shared record | Low to moderate |
| Sensor fusion | A ∩ B ∩ C | False consensus detection | Missed consensus | Moderate |

The index doesn't care what the fp8 values represent. The predicate kernel doesn't care what the predicate means. The error model doesn't care what domain the false positive belongs to. What the architecture provides is a **general-purpose probabilistic evaluation surface**: ask any boolean question across N large incomplete-knowledge datasets and get a quantified probabilistic answer at memory-bandwidth speed.

---

## 8. The Cost of Coordination

<!-- STRATEGY: The bridge between theory (§7) and system evaluation (§9). §7 characterised what a single node can infer from its index. §8 shows what happens when nodes coordinate. Every operation is a measured GPU primitive from §5 applied to a coordination problem. The section closes with the broadening claim: these primitives serve both the database's internal coordination and application-layer inference — and the pattern generalises to any system where actors coordinate through partial knowledge. -->

§7 characterised what a single node can infer from its index — the signal quality, the three observations, the confidence gradient. In a distributed system, nodes must coordinate. Each operation in this section is a measured GPU primitive from §5 applied to a coordination problem. These primitives serve two consumers: the database coordinating its own state (sync, replication, consistency), and the applications and agents using the database to reason about the data it holds.

### 8.1 Differential Sync

- A remote node mutates its index. The local node holds a cached copy of the remote's fp8 array.
- Fingerprint diff (m1.1) produces the set of changed positions. The local node requests only those positions' updated data.
- The diff cost is O(S) to compute (one GPU-acceleratable pass over the array — see Figures 5.3–5.4 for the measured crossover profile), but the *transmission* cost is proportional to the number of changed positions, not the total array size.
- After M mutations, approximately M positions changed. The delta is M bytes plus position indices, not S bytes.
- Cardinality as a scheduling signal: the popcount of the diff bitmask tells you how many positions changed — a single number, before examining which positions. That count alone is often sufficient to make a scheduling decision: is this cache stale enough to warrant a sync now, or can it wait? The threshold is application-defined; the cost of obtaining it is a single reduction over the diff bitmask.
- Compare: Bloom filter sync requires retransmitting the entire filter (S × 10 bits) or maintaining a separate change log. The positional approach syncs deltas from the structure that already exists.

**Figure 8.1 — Differential sync: exchanging position deltas**
*Two nodes, each holding an fp8 array. Node A mutates its index. Node B holds a cached copy. The diff operation (XOR + compare, one GPU pass) produces a bitmask of changed positions. Node B requests only the changed positions' updated data. The transmission is proportional to M (mutations), not S (total array size).*

![Figure 8.1 — Differential sync](images/whitepaper-figure-8-2-differential-sync-position-deltas-v2.png)

### 8.2 Overlap Estimation and Replication Priority

- Set intersection (m1.2) answers "how much overlap do nodes A and B have?" — the cardinality as a similarity ratio.
- The mechanism: positions where both arrays hold the same non-empty fp8 are candidate matches. The count of matching positions, divided by total occupied positions, gives the similarity ratio. The raw count is slightly inflated by the 1/255 FP rate per position — at scale, this inflation is small and predictable (for S occupied positions, expected false matches ≈ S × α/255).
- A coordinator managing C nodes needs to prioritise which pairs to sync. High overlap = low priority. Low overlap = urgent. The GPU scans two co-indexed arrays and produces a count. At 10M+ slots, this is firmly in GPU-advantage territory (§5.4 crossover data).
- The cardinality is the system-level primitive — a single number feeding scheduling decisions. The position list is a secondary output useful for targeted verification.

### 8.3 Multi-Set Predicates as Distributed Inference

Multi-set predicate evaluation (m1.3) enables compound questions across N cached summaries. Where §8.1 asks "what changed?" and §8.2 asks "how similar?", multi-set predicates ask "what is the relationship across three or more datasets?" — and each confirming array makes the answer more trustworthy.

- Concrete predicates:
  - **Replication gap** (`A \ (B ∪ C)`): keys unique to node A — durability risk. Cardinality triggers replication scheduling.
  - **Consistency verification** (`A ∩ B ∩ C`): positions where all replicas agree. The complement identifies potential consistency violations — trigger anti-entropy repair if disagreement exceeds threshold.
  - **Migration planning** (`A \ B`): estimate transfer volume before committing to a migration operation.
- Multi-predicate fusion: compute replication gaps AND consistency AND migration estimates in a single GPU pass over the same arrays. The marginal cost of each additional predicate is nearly zero on GPU.
- Predicate outputs are **statistical estimators**: the detected count, adjusted by the expected placement rate (Table 4.7), gives an estimated total.

The precision of these estimators improves with each confirming array:

**Table 8.1 — Multi-array precision compounding**
*Each confirming array exponentially reduces the false positive probability on a positive match. A single-array match is strong evidence. A three-array consensus match is near-certain.*

| Configuration | Match FPR | Confidence on positive | Notes |
|---|---|---|---|
| 1-array, single match | 1/255 | 99.6% | Baseline — one confirming array |
| 2-array intersection | (1/255)² = 1/65,025 | 99.998% | Two independent confirmations |
| 3-array consensus | (1/255)³ = 1/16.6M | ~100% | Three confirmations — actionable without verification |

- The precision on what is detected is the table above. The coverage of what could be detected is a deployment parameter the system operator controls — a known function of α, k, and N, all characterised by §4.3's placement probabilities.

### 8.4 Coordination Through Incomplete Knowledge

All coordination operations in §8.1–8.3 operate on fp8 summaries — at most 26 bits of a 128-bit key. The full key is never needed for coordination. This is not a limitation to work around — it is the operating medium. Partial knowledge, with quantified confidence from §7, is sufficient for coordination decisions.

**Figure 8.2 — What is available for coordination**
*Left: the full 128-bit key — the data layer. Right: position + fp8 (26 bits) — the coordination layer. All operations in §8.1–8.3 operate on the right side. The 26 bits are sufficient to determine what changed (diff), how similar two datasets are (intersection), and what the relationship is across N datasets (multi-set predicates). The confidence on every answer is quantifiable from the signal characterisation in §7.*

![Figure 8.2 — What is available for coordination](images/whitepaper-figure-8-1-disclosure-current-capability-v2.png)

- The database's internal coordination (sync, replication, consistency) and the application-layer coordination it serves (entity resolution, pattern detection, overlap estimation, consensus building) use the same primitives. The same bytes, the same predicates, the same confidence model. The distinction between database housekeeping and application intelligence dissolves — both are actors coordinating through exchange of lossy summaries, making decisions that are quantifiably good enough.
- §7.4 names the computational pattern: N arrays, boolean predicate, GPU pass, quantified confidence. §8.4 names the system pattern: actors with partial knowledge of a shared domain, coordinating through exchange of lossy summaries.

*Personal Note: This pattern — coordination through incomplete knowledge with quantified confidence — extends well beyond database infrastructure. Anywhere actors hold partial views of a shared domain and need to coordinate:*

- *Multi-agent AI coordination — autonomous agents negotiating shared resources or resolving conflicting goals exchange positional summaries of their state. Enough to detect overlap and conflict, enough to schedule coordination, enough to build a shared operating picture — with quantifiable confidence on every exchange.*
- *Supply chain verification — participants confirm shared inventory or shipment records through intersection cardinality. "Do our records agree, and by how much?" is answerable from lossy summaries alone.*
- *Cross-organisational fraud detection — banks or insurers compare transaction fingerprints to identify correlated anomalies appearing across institutions. The multi-set predicate tells you where patterns converge across three or more datasets — the precision compounding from Table 8.1 makes even rare pattern matches trustworthy.*
- *Collaborative intelligence in defence and civil security — allied but not fully trusted parties share sensor or intelligence summaries to establish common operating pictures. The declining guarantee (§7.3) maps naturally: confidence in shared assessments is quantifiable, and each party controls how much they contribute.*
- *Decentralised AI training marketplaces — model providers publish fingerprint summaries of training data coverage. Buyers estimate overlap with their own datasets to assess marginal value — the intersection cardinality is sufficient for the purchasing decision.*

- This is the shift the blog introduction promised: digital systems moving from "I need the full picture before I act" to "I have enough confidence to act now, and I know exactly how much confidence that is." The index is a concrete implementation of this shift. The signal characterisation from §7 is the costing framework. The coordination primitives in §8.1–8.3 are the operations. §8.4 is where the technical paper meets the directional claim.

---

## 9. Comparing Architectures

<!-- The closing argument. Two clean inputs: §7 gives per-query signal quality, §8 gives per-coordination-cycle cost. This section assembles the total cost equation, plugs in measured components from benchmarks, and shows the crossover across workload profiles. -->

### 9.1 The Two Architectures

**Architecture A** (traditional): primary index + Bloom filter. Two structures, two update paths, one signal (probabilistic membership).

**Architecture B** (proposed): primary index only. One structure, one update path, multiple signals (membership, diff, intersection, multi-set predicates, cardinality estimation) from the same bytes.

### 9.2 Per-Query Signal Quality

<!-- STRATEGY: Present as mathematical characterisation, not experimental result. The FPR/FNR values follow from the independence analysis (§7) and standard Bloom filter theory. No empirical benchmark needed — both models are well-understood. -->

- Bloom k=3 with 10 bits/key: FPR ≈ (1 - e^(-kn/m))^k ≈ 1%. Zero false negatives by construction. This is the textbook result (Bloom, 1970; Broder & Mitzenmacher, 2004).
- Positional k=1 at 75% load: FPR = 0.29% from the independence analysis — full coordinate tuple provides 26+ bits of discrimination from 8 stored (§7.1). FNR = α = 75% — keys displaced from preferred position are missed.
- Positional k=4 at 75% load: FNR drops to α⁴ = 31.6%. FPR increases slightly due to additional probe positions but remains below Bloom's 1%.
- The positional probe is not a Bloom filter replacement. It is a different kind of signal occupying a different region of the error space: lower FPR than Bloom at zero auxiliary storage, but with a tuneable FNR that Bloom does not have.

**Figure 9.1 — Error space: positional probe vs Bloom filter**
*Left: FPR and FNR at each load factor for positional k=1, k=4, and Bloom k=3. Right: FPR vs FNR plot — Bloom as a fixed point on the y-axis (0% FNR, ~1% FPR), positional probe curves sweeping through the space as α varies. Shows that the two structures occupy fundamentally different regions of the error space.*

<!-- This figure is derived from the mathematical models, not from empirical measurement. The curves are the signal equation from §7. -->

### 9.3 Per-Mutation Maintenance Cost

<!-- STRATEGY: This is operation counting — a structural property of the algorithms, not an empirical question. Bloom's per-insert cost is k_b additional hash+write operations. Positional's additional cost is zero by construction. -->

- Bloom: every mutation to the underlying set requires k_b additional hash computations and k_b writes to the filter, beyond the primary index operation. This is inherent to the structure — the filter is a separate data structure that must be kept in sync with the primary index.
- Positional: the fp8 byte is written as part of the primary insert operation. Delete clears the byte as part of the primary delete. There is no second structure to update. The maintenance cost for the probabilistic signal is zero by construction.
- Over M mutations per unit time, Architecture A pays M × k_b additional hash operations that Architecture B does not. At k_b = 3, this is a 3× multiplier on the mutation-related workload directed at maintaining the probabilistic signal.
- For write-heavy workloads where M >> Q, this maintenance term dominates the total cost difference between architectures.

### 9.4 The Delete Problem

<!-- STRATEGY: Bloom's inability to delete is a well-established structural property (Bloom 1970, Fan et al. 2000). The FPR degradation under phantom keys is derivable. The positional probe's delete behaviour is a structural property of the design. The visual is derived from the degradation formula, not from empirical measurement. -->

- Standard Bloom cannot delete. When a key is removed from the primary index, its k_b bit positions in the filter remain set. These phantom positives accumulate: after D deletes, up to D keys contribute false positive signal that cannot be retracted. The effective FPR grows monotonically with the ratio of deleted keys to total capacity (Bloom, 1970; Fan et al., 2000).
- The only remedy for standard Bloom is periodic full rebuild at cost O(S × k_b) — reconstructing the entire filter from the current keyset. The rebuild frequency R becomes a system parameter trading freshness against cost.
- Counting Bloom (Fan et al., 2000) supports deletion by replacing each bit with a counter, but at 4× storage: 40 bits per key instead of 10.
- Positional probe: delete clears one fp8 byte. The signal at that position immediately returns to "empty" — the theoretical FPR is restored without any rebuild, any additional storage, or any deferred maintenance.

**Figure 9.2 — FPR stability under mutation: Bloom vs positional probe**
*Derived from the Bloom phantom-key degradation model. X-axis: cumulative deletes as fraction of original keyset. Y-axis: effective FPR. Bloom (standard): monotonically increasing curve. Bloom (with periodic rebuild): sawtooth — degrades, rebuilds, degrades again. Counting Bloom: flat but at 4× storage baseline. Positional probe: flat at theoretical FPR, no rebuild required.*

<!-- This visual is the strongest single illustration of the delete advantage. Derived, not measured. -->

**Table 9.1 — Delete handling comparison**

| Structure | Supports delete | Storage per key | FPR after deletes | Recovery mechanism |
|---|---|---|---|---|
| Standard Bloom | No | 10 bits | Degrades monotonically | Full rebuild |
| Counting Bloom | Yes | 40 bits | Stable | Counter decrement |
| Positional probe | Yes | 0 auxiliary (10.67 bits/slot at α=0.75) | Stable | None required |

- For workloads with significant delete/replace volume — common in caches, session stores, and any system with TTL-based eviction — this is the strongest structural differentiator.

### 9.5 Remote Sync Cost

<!-- STRATEGY: The diff operation (m1.1) is measured in the GPU benchmarks. The comparison to Bloom's sync cost is structural — Bloom has no delta mechanism, so it must retransmit the full filter. The transmission cost comparison is arithmetic. -->

- Bloom: no delta mechanism exists. Syncing a remote Bloom filter cache requires retransmitting the entire filter — S × 10 bits per node, regardless of how many keys changed. Alternatively, maintain a separate change log, which is itself an additional structure requiring its own synchronisation.
- Positional: diff the cached fp8 arrays (§5.4, §8.1), transmit only the changed positions. The diff is a single GPU-acceleratable pass producing a sparse delta proportional to mutations, not dataset size.
- At 1M keys with 1% mutation rate: Bloom retransmits ~1.25 MB per sync. Positional transmits ~10K changed positions (~40 KB including position indices). A 30× reduction in sync bandwidth from a structural property of the architecture.
- The bandwidth advantage scales with the ratio of dataset size to mutation rate. For large, slowly-mutating datasets (common in distributed indexes), the ratio is extreme.

### 9.6 Storage Scaling

<!-- STRATEGY: Pure arithmetic. No benchmark, no derivation needed — these are multiplication tables. The insight is in the utility comparison, not the numbers themselves. -->

- The fp8 array is not zero storage — it is the primary index, cached remotely. At load factor α, effective storage per key = 8/α bits.
- At α=0.75: fp8 = 10.67 bits/key. Standard Bloom = 10 bits/key. Counting Bloom = 40 bits/key. The fp8 cache is roughly comparable to standard Bloom in size, and 3.75× more efficient than counting Bloom which is required for delete support.

**Table 9.3 — Storage comparison at scale**

| Nodes | Keys/node | Bloom (10 bits/key) | Counting Bloom (40 bits/key) | fp8 cache (α=0.75) |
|---|---|---|---|---|
| 1,000 | 1M | 1.25 GB | 5.0 GB | 1.33 GB |
| 1,000 | 10M | 12.5 GB | 50 GB | 13.3 GB |
| 10,000 | 10M | 125 GB | 500 GB | 133 GB |

- The argument is not zero vs something. It is **comparable storage, incomparable utility**. The fp8 cache at 1.33 GB provides: probabilistic membership, fingerprint diff for sync, set intersection, multi-set predicate evaluation, iteration over remote index contents, and cardinality estimation. The Bloom cache at 1.25 GB provides: probabilistic membership. One signal vs six operations from the same bytes.

### 9.7 The Total Cost Equation

<!-- STRATEGY: The equation assembles structural properties (§9.3, §9.4), mathematical characterisations (§9.2), measured GPU performance (§9.5 via m1.1), and arithmetic (§9.6). Each input is either a property of the algorithm, a result from the literature, or a measured value from the GPU benchmarks in §5. The equation itself is analytical. -->

Over a time window with Q queries, M mutations (fraction r deletes), across C nodes:

**Architecture A (index + Bloom):**
```
C_A = Q × k_b                           # query: k hash ops + k reads
    + M × k_b                           # maintenance: k hash ops + k writes per mutation
    + M × r × (S × k_b) / R            # amortised rebuild for deletes
    + C × S × 10 bits                   # sync: full filter retransmit
    + C × S × 10 bits                   # storage: auxiliary filter cache
```

**Architecture B (index only):**
```
C_B = Q × 1                             # query: 1 hash op + k_p reads (k_p probes from one hash)
    + M × 0                             # maintenance: zero additional
    + 0                                 # rebuild: never
    + C × Δ(M)                          # sync: delta proportional to mutations
    + 0                                 # storage: no auxiliary (fp8 already cached)
```

- **Read-heavy** (Q >> M): per-query cost dominates. Architectures roughly comparable. Positional saves k_b - 1 hash computations per query — small absolute difference.
- **Write-heavy** (M >> Q): maintenance term dominates. Bloom pays M × k_b additional operations. Positional pays zero. Clean k_b× multiplier on mutation cost.
- **Delete-heavy** (high r): rebuild or counting-Bloom penalty dominates. Standard Bloom becomes unusable without periodic rebuilds. Counting Bloom survives at 4× storage. Positional unaffected.
- **High node-count** (large C): sync and storage terms dominate. Positional syncs deltas (proportional to mutations). Bloom syncs full structures (proportional to dataset size).
- **Multi-signal** (coordination beyond membership): Architecture A requires additional structures for each operation (change logs for sync, separate intersection mechanisms, etc.). Architecture B provides all coordination primitives from the same cached arrays at zero additional cost.

### 9.8 Routing Economics

<!-- STRATEGY: This is an analytical cost model. The inputs (FPR, FNR) are derived from §7. The routing cost is arithmetic on those inputs plus assumed network costs. Present as a worked example, not a simulation result. -->

- Consider a cluster of C = 1000 nodes, each holding a partition of a distributed index. A query arrives and must be routed to the node(s) holding the relevant key. The coordinator checks cached probabilistic summaries from all nodes.
- Bloom k=3 at 1% FPR: expected false positive nodes per query = C × FPR = 1000 × 0.01 = 10. Each false positive triggers an unnecessary round trip. Zero false negatives — the correct node is always identified.
- Positional k=1 at 75% load, 0.29% FPR: expected false positive nodes = 1000 × 0.0029 ≈ 3. But the correct node is missed with probability α = 0.75, requiring a fallback mechanism (broadcast or secondary lookup).
- Positional k=4 at 75% load: FNR drops to α⁴ = 31.6%. FPR remains below 1.2%. Expected false positives ≈ 12, false negative probability = 31.6%.
- Total routing cost per query = (expected FP nodes × round_trip_cost) + (FNR × fallback_cost). The optimal k minimises this sum for a given network cost model. At the point where round_trip_cost ≈ fallback_cost, the crossover favours higher k. Where fallback is cheap (e.g. broadcast to a small subset), lower k with fewer false positives is preferred.
- The system designer parameterises k against their network economics — the same tuneable trade-off that §7.2 characterises in abstract terms, here grounded in concrete routing costs.

---

## 10. Open Questions

- **Formal bounds on the signal equation under adversarial key distributions.** The signal equation assumes uniform hashing — 16-byte hashes with good distribution properties. The FPR depends on the independence of position bits and fingerprint bits, which holds under uniform distribution but could degrade under adversarial or pathological key patterns. Characterising the signal's behaviour when this assumption is weakened is the primary theoretical vulnerability.

- **Empirical validation of the signal equation.** The FPR and FNR characterisations in §7 and §9 are derived from the independence analysis under uniform hashing. Empirical validation — measuring observed FPR/FNR against the theoretical model across hash function families and real-world key distributions — would confirm that the assumptions hold in practice. This is the most direct avenue for strengthening the analytical results presented in this paper.

- **The LSH instantiation.** Does the architecture hold when the hash family produces locality-sensitive rather than uniformly distributed fingerprints? The same index geometry — positional encoding, fp8 fingerprints, hierarchical probe sequence — could in principle serve similarity search rather than exact membership. If so, the probabilistic evaluation surface extends from key-value coordination to vector neighbourhood scanning. Same geometry, similarity semantics.

- **Behaviour at extreme scale.** At millions of nodes and billions of keys, several questions arise: does the coordinator model (§8) remain feasible when caching fp8 arrays from millions of nodes? Do the statistical estimators in §8.3 remain calibrated when the number of arrays N in a multi-set predicate grows large? The (1/255)^(N-1) precision improvement is theoretically unbounded, but at large N, even tiny per-position errors may compound differently than the model predicts.

- **GPU crossover on non-Apple architectures.** The GPU benchmarks in §5 are measured on Apple Silicon unified memory, where CPU and GPU share physical bytes without transfer overhead. On discrete GPU architectures (NVIDIA, AMD) with PCIe transfer, the crossover profile will differ — raw GPU compute may be faster, but the transfer cost adds a fixed overhead per operation that shifts the crossover point. Whether the net effect improves or degrades total system performance for the operations in §5.4 and §5.5 is an open empirical question.

- **Privacy guarantees under repeated observation.** §8.4 notes that a single fp8 array snapshot reveals at most 26 bits of a 128-bit key. However, an adversary observing multiple sync cycles sees delta patterns — which positions change, how frequently, in what temporal clusters. Over time, repeated observation may leak more information than a single snapshot suggests. The "not a security guarantee" caveat is stated, but formally characterising the information leakage rate under repeated differential sync is necessary before the architecture can be recommended for adversarial-trust environments.

---

## 11. Concluding Remarks

<!-- STRATEGIC INTENT — do not draft copy yet.

Four beats:

1. WHAT WE BUILT. One sentence. A radix hash index whose internal geometry emits a probabilistic signal as a structural byproduct, designed for GPU-accelerated batch operations on Apple Silicon unified memory. Name it cleanly, don't rehash the architecture.

2. WHAT WE FOUND. Three novel results, stated without hedging:
   - Position is information. Independent bit allocation between placement and fingerprint creates 26+ bits of discrimination from 8 stored bits. This is the theoretical contribution — the mechanism that makes everything else work.
   - The same bytes serve both exact local lookup and probabilistic distributed inference. No auxiliary structure, no separate maintenance path, no divergence between the index and the signal. Six coordination primitives (membership, diff, intersection, predicates, iteration, cardinality) from a single cached array. This is the architectural contribution.
   - GPU acceleration applies to the distributed operations — not individual lookups, but the batch array reasoning that coordination requires. Crossover is operation-shape dependent: coalesced full-array scans at 3–7× on Apple Silicon at scale. This is the systems contribution.

3. WHAT IT COSTS. Honest accounting alongside the summary table:
   - Lookup is 2.4× slower than hashbrown at 75% load — the price of the richer geometry.
   - The signal has false negatives that Bloom does not — tuneable via probe depth k, but present.
   - Recall degrades with load factor and with the number of arrays in multi-set predicates.
   - These are not flaws to apologise for — they are design parameters the system operator controls.

4. WHERE IT LEADS. Directional, not a repetition of §10:
   - The paper demonstrates the primitive on uniformly distributed keys for coordination.
   - The broader claim (§7.4): parallel hypothesis testing over incomplete knowledge at memory-bandwidth speed applies wherever large lossy-encoded datasets need probabilistic reasoning.
   - LSH instantiation is the most immediate extension.
   - Low-trust coordination applications (§8.4) are the most consequential.
   - The shift from declarative to probabilistic architectures promised in the blog introduction is not a future possibility — it is what the index already does.
   - Close on the design principle: one structure, one memory layout, three execution models, quantifiable confidence at every level. The index works. The signal is free. The equation tells you how much to trust it.
-->

**Table 11.1 — Summary: what the index provides**

| Property | Value |
|---|---|
| Local lookup (hit, 1% load) | ~2.8 ns (1.0× hashbrown) |
| Local lookup (hit, 75% load) | ~8.2 ns (2.4× hashbrown) |
| Iteration (50% load) | ~101 µs (5.2× faster than hashbrown) |
| GPU crossover (full-array ops) | 3-7× faster at 8M+ slots |
| Probabilistic miss (k=1, 75% load) | 99.71% correct in 1 byte read |
| Discrimination (full coordinate + fp8) | 26+ bits from 8 stored |
| Auxiliary filter storage required | 0 (fp8 array is the primary index) |
| Coordination primitives from same bytes | 6 (membership, diff, intersection, predicates, iteration, cardinality) |

---

## References

- Azar, Y., Broder, A. Z., Karlin, A. R., & Upfal, E. (1994). Balanced Allocations. *STOC*, 593–602.
- Bender, M. A., Conway, A., Farach-Colton, M., Kuszmaul, W., & Tagliavini, G. (2021). Iceberg Hashing. *JACM*, 70(6), 1–51.
- Farach-Colton, M., Krapivin, A., & Kuszmaul, W. (2025). Optimal Bounds for Open Addressing Without Reordering. arXiv:2501.02305v2.
- Knuth, D. E. (1963). Notes on "Open" Addressing. Unpublished memorandum.
- Kulukundis, M. (2017). Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step. CppCon 2017.
- Pagh, R., & Rodler, F. F. (2004). Cuckoo Hashing. *Journal of Algorithms*, 51(2), 122–144.
- Pandey, P., Bender, M. A., Johnson, R., & Patro, R. (2017). A General-Purpose Counting Filter. *SIGMOD*, 775–787.
- Peterson, W. W. (1957). Addressing for Random-Access Storage. *IBM J. Res. Dev.*, 1(2), 130–146.
- Vöcking, B. (2003). How Asymmetry Helps Load Balancing. *JACM*, 50(4), 568–589.
- Yao, A. C. (1985). Uniform Hashing is Optimal. *JACM*, 32(3), 687–693.
