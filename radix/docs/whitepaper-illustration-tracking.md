# Whitepaper Illustration Tracking

Legend:
- `real?`: `📊` real data, `🧠` conceptual/illustrative
- `Clarification required?`: `❓` yes, `✅` no
- `Completed?`: `✅` when finalised, otherwise blank

| name | type | real? | Clarification required? | Completed? | Description |
|---|---|---:|---:|---:|---|
| 0_1_one_index_two_execution_models | image | 🧠 | ✅ |  | Single index in unified memory with CPU directed probes and GPU batch scans over same bytes. |
| 0_2_unified_vs_discrete_memory_architecture | image | 🧠 | ✅ |  | Side-by-side architecture diagram: Apple unified memory vs discrete GPU memory + transfer/duplication cost. |
| 1_1_same_memory_incompatible_access_patterns | image | 🧠 | ✅ |  | CPU-friendly scattered structure vs GPU-friendly contiguous structure; central design question. |
| 2_1_hash_segmented_bar | image | 🧠 | ✅ |  | 128-bit hash segmented into bucket/group/chunk/offset/fp8/unused with independence boundary. |
| 2_2_bit_consumption_hierarchy | image | 🧠 | ✅ |  | Visual of positional bits consumed through bucket->group->chunk->slot and effect on probe depth. |
| 2_3_physical_memory_layout | image | 🧠 | ✅ |  | Bucket/group/chunk/slot memory geometry with 64B cache-line and 16B NEON boundaries. |
| 3_1_physical_layout_bucket_group_chunk_slot | image | 🧠 | ✅ | ✅ | Same physical layout as 2.3 but annotated with implementation perspective. |
| 3_2_hash_bit_partitioning_capacity20 | image | 🧠 | ✅ | ✅ | Concrete partition example for capacity_bits=20: bucket/group/chunk/offset/fp8/spare. |
| 3_3_probe_sequence_power_of_k_choices | image | 🧠 | ✅ | ✅ | Preferred offsets across 4 chunks (p0->p1->p2->p3) plus NEON fallback path. |
| 3_4_overflow_same_group_next_bucket | image | 🧠 | ✅ | ✅ | Overflow policy preserving group/chunk/offset/fp8 tuple while advancing bucket only. |
| 4_1_final_benchmark_matrix_radix_vs_hashbrown | table | 📊 | ✅ |  | CPU baseline matrix at 256K across ops and load factors comparing radix_tree and hashbrown. |
| 4_1_latency_progression_across_runs | image | 📊 | ✅ |  | Run-by-run latency progression matrix with optimisation tags and hashbrown baseline. |
| 4_2_iteration_performance_radix_vs_hashbrown | table | 📊 | ✅ |  | Iteration latencies and ratios across load factors, radix vs hashbrown. |
| 4_3_scalar_vs_full_neon_scan | table | 📊 | ❓ |  | Scalar preferred probe vs full NEON scan by load; currently DATA PENDING placeholders. |
| 4_4_pure_neon_vs_scalar_first_fallback | table | 📊 | ❓ |  | Pure NEON vs scalar-first+fallback by load; currently DATA PENDING placeholders. |
| 4_5_occupancy_distribution_linear_before | table | 📊 | ❓ |  | Occupancy resolution breakdown for linear probing baseline; currently DATA PENDING. |
| 4_6_occupancy_distribution_k_choices_after | table | 📊 | ❓ |  | Occupancy resolution with k=4 preferred offsets; currently DATA PENDING. |
| 4_2_chunk_occupancy_heatmap_before_after | image | 📊 | ✅ |  | Heatmap comparison of chunk occupancy and overflow before/after k-choices strategy. |
| 4_7_p_all_k_preferred_occupied | table | 📊 | ✅ |  | Theoretical probabilities for k=1/2/4 at load factors 25/50/75%. |
| 5_1_iteration_gpu_vs_cpu_by_size_occupancy | table | 📊 | ✅ |  | Iteration crossover table (CPU vs GPU) across index sizes and occupancies. |
| 5_1a_iteration_raw_latency_columns | image | 📊 | ✅ | ✅ | Iteration raw mean latency columns (CPU gray, GPU red), split by occupancy. |
| 5_1_iteration_ratio_curve | image | 📊 | ✅ | ✅ | Iteration CPU/GPU ratio curve with parity line (higher is better). |
| 5_2_greedy_search_gpu_vs_cpu_by_size_occupancy | table | 📊 | ✅ |  | Greedy search CPU/GPU comparison across index sizes and occupancies. |
| 5_2a_index_diff_raw_latency_columns | image | 📊 | ✅ | ✅ | Index diff raw mean latency columns (CPU gray, GPU red), split by occupancy. |
| 5_2_index_diff_ratio_curve | image | 📊 | ✅ | ✅ | Index diff CPU/GPU ratio curve with iteration reference overlays (higher is better). |
| 5_3_predicate_family_summary | table | 📊 | ✅ |  | Family-2/3/5 summary with geomean speedups, lift, worst/best points. |
| 5_4_two_dimensional_scaling_size_by_family | table | 📊 | ✅ |  | GPU advantage by index size x predicate family (2/3/5 arrays). |
| 5_3a_algebraic_predicates_raw_latency_matrix | image | 📊 | ✅ | ✅ | Consolidated raw latency matrix for Family-2/3/5 across index size and occupancy. |
| 5_3_algebraic_predicates_ratio_by_family | image | 📊 | ✅ | ✅ | Key arity fan chart showing Family-2/3/5 divergence by size (CPU/GPU, higher better). |
| 5_4_all_operations_overlay_ratio_curve | image | 📊 | ✅ | ✅ | Overlay chart: iteration, greedy, Family-2/3/5 with parity line (CPU/GPU, higher better). |
| 5_5_gpu_crossover_summary | table | 📊 | ❓ |  | Operation-wise crossover summary; includes DATA PENDING entries for index diff. |
| 6_1_primitives_to_operations_mapping | table | 📊 | ✅ |  | Mapping benchmarked primitives to database/inference operations they compose. |
| 6_2_query_complexity_scaling_sql_to_arity | table | 📊 | ✅ |  | SQL complexity mapping to 2/3/5-array arity and measured GPU advantage. |
| 6_3_gpu_advantage_by_size_and_query_complexity | table | 📊 | ✅ |  | Two-axis scaling table: index size x query complexity (2/3/5 arrays). |
| 6_4_index_design_cpu_shaped_vs_gpu_native | table | 🧠 | ✅ |  | Property comparison (B-tree, Swiss Table, radix) vs GPU requirements. |
| 7_1_summary_what_the_index_provides | table | 📊 | ✅ |  | Final summary table of lookup, iteration, arity scaling, crossover, and peak gains. |

