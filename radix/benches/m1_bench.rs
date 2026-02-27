use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use radix::IndexTable;
use radix::radix_tree::{
    RadixTree, RadixTreeConfig,
    fingerprint_diff, set_intersection,
    set_predicate_2, set_predicate_3,
    PRED2_INTERSECT, PRED2_DIFF_AB, PRED2_SYM_DIFF,
    PRED3_CONSENSUS, PRED3_UNIQUE_A,
};
use std::collections::HashSet;
use std::hint::black_box;

const LOAD_FACTORS: &[f64] = &[0.10, 0.90];

const CAPACITIES: &[(usize, &str)] = &[
    (1 << 18, "256K"),
    (1 << 20, "1M"),
    (1 << 23, "8M"),
    (1 << 27, "128M"),
];

#[inline]
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn make_ids(n: usize, seed: u64) -> Vec<u64> {
    (0..n).map(|i| mix64(seed.wrapping_add(i as u64))).collect()
}

fn capacity_bits_from_slots(total_slots: usize) -> u8 {
    debug_assert!(total_slots.is_power_of_two());
    total_slots.trailing_zeros() as u8
}

fn build_filled_radix(capacity: usize, load_factor: f64) -> (RadixTree, usize) {
    let bits = capacity_bits_from_slots(capacity);
    let mut table = RadixTree::new(RadixTreeConfig { capacity_bits: bits })
        .expect("valid radix benchmark config");
    let target_len = (capacity as f64 * load_factor) as usize;
    let ids = make_ids(target_len + 10_000, 0x1234_5678);
    for (i, &id) in ids.iter().take(target_len).enumerate() {
        assert!(
            table.insert(id),
            "m1: insert failed at {i}/{target_len} (cap={capacity}, lf={load_factor:.2})"
        );
    }
    (table, target_len)
}

fn should_run_op(op_name: &str) -> bool {
    match std::env::var("OPS").ok() {
        Some(raw) => {
            let set: HashSet<String> = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .collect();
            set.contains(op_name)
        }
        None => true,
    }
}

fn bench_iter_across_sizes(c: &mut Criterion) {
    if !should_run_op("iter") {
        return;
    }

    #[cfg(feature = "gpu32")]
    let impl_name = "radix_tree_gpu";
    #[cfg(not(feature = "gpu32"))]
    let impl_name = "radix_tree";

    for &(capacity, size_label) in CAPACITIES {
        for &lf in LOAD_FACTORS {
            let param = format!("size={size_label}/lf={lf:.2}");
            let (table, target_len) = build_filled_radix(capacity, lf);
            assert_eq!(table.len(), target_len);

            let mut group = c.benchmark_group(format!("impl={impl_name}/op=iter"));
            group.throughput(Throughput::Elements(target_len as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(&param),
                &param,
                |b, _| {
                    b.iter(|| {
                        let mut count: u64 = 0;
                        for key in table.iter() {
                            black_box(key);
                            count += 1;
                        }
                        debug_assert_eq!(count, target_len as u64);
                        count
                    });
                },
            );
            group.finish();
        }
    }
}

fn bench_contains_greedy_across_sizes(c: &mut Criterion) {
    if !should_run_op("contains_greedy") {
        return;
    }

    #[cfg(feature = "gpu32")]
    let impl_name = "radix_tree_gpu";
    #[cfg(not(feature = "gpu32"))]
    let impl_name = "radix_tree";

    for &(capacity, size_label) in CAPACITIES {
        for &lf in LOAD_FACTORS {
            let param = format!("size={size_label}/lf={lf:.2}");
            let (table, target_len) = build_filled_radix(capacity, lf);
            assert_eq!(table.len(), target_len);

            // Pick a key known to be present (the last inserted key).
            let probe_key = make_ids(target_len, 0x1234_5678)[target_len - 1];
            assert!(table.contains(probe_key), "probe key must be present");

            let mut group = c.benchmark_group(format!("impl={impl_name}/op=contains_greedy"));
            group.bench_with_input(
                BenchmarkId::from_parameter(&param),
                &param,
                |b, _| {
                    b.iter(|| {
                        black_box(table.contains_greedy(black_box(probe_key)))
                    });
                },
            );
            group.finish();
        }
    }
}

/// Build two RadixTree instances with controlled key overlap.
/// `overlap` is the fraction of table A's keys that are also in table B.
/// Both tables are filled to `lf` load factor.
fn build_pair(capacity: usize, lf: f64, overlap: f64) -> (RadixTree, RadixTree) {
    let bits = capacity_bits_from_slots(capacity);
    let target_len = (capacity as f64 * lf) as usize;
    let shared_n = (target_len as f64 * overlap) as usize;
    let unique_n = target_len - shared_n;

    let shared_keys = make_ids(shared_n + 10_000, 0xAAAA_BBBB);
    let unique_a_keys = make_ids(unique_n + 10_000, 0xCCCC_1111);
    let unique_b_keys = make_ids(unique_n + 10_000, 0xDDDD_2222);

    let mut a = RadixTree::new(RadixTreeConfig { capacity_bits: bits })
        .expect("build_pair: table A");
    let mut b = RadixTree::new(RadixTreeConfig { capacity_bits: bits })
        .expect("build_pair: table B");

    for &k in shared_keys.iter().take(shared_n) {
        a.insert(k);
        b.insert(k);
    }
    for &k in unique_a_keys.iter().take(unique_n) {
        a.insert(k);
    }
    for &k in unique_b_keys.iter().take(unique_n) {
        b.insert(k);
    }

    (a, b)
}

/// Build three RadixTree instances with controlled overlap.
/// `overlap_ab` is the fraction of A's keys shared with B.
/// `overlap_ac` is the fraction of A's keys shared with C.
fn build_triple(
    capacity: usize,
    lf: f64,
    overlap_ab: f64,
    overlap_ac: f64,
) -> (RadixTree, RadixTree, RadixTree) {
    let bits = capacity_bits_from_slots(capacity);
    let target_len = (capacity as f64 * lf) as usize;
    let shared_ab = (target_len as f64 * overlap_ab) as usize;
    let shared_ac = (target_len as f64 * overlap_ac) as usize;
    let unique_a = target_len.saturating_sub(shared_ab.max(shared_ac));

    let ab_keys = make_ids(shared_ab + 10_000, 0xAA11_BB22);
    let ac_keys = make_ids(shared_ac + 10_000, 0xAA11_CC33);
    let unique_a_keys = make_ids(unique_a + 10_000, 0xAAAA_0001);
    let unique_b_keys = make_ids(target_len + 10_000, 0xBBBB_0002);
    let unique_c_keys = make_ids(target_len + 10_000, 0xCCCC_0003);

    let mut a = RadixTree::new(RadixTreeConfig { capacity_bits: bits }).expect("build_triple: A");
    let mut b = RadixTree::new(RadixTreeConfig { capacity_bits: bits }).expect("build_triple: B");
    let mut c = RadixTree::new(RadixTreeConfig { capacity_bits: bits }).expect("build_triple: C");

    for &k in ab_keys.iter().take(shared_ab) { a.insert(k); b.insert(k); }
    for &k in ac_keys.iter().take(shared_ac) { a.insert(k); c.insert(k); }
    for &k in unique_a_keys.iter().take(unique_a) { a.insert(k); }

    // Fill B and C to target load with unique keys
    let b_remaining = target_len.saturating_sub(b.len());
    for &k in unique_b_keys.iter().take(b_remaining) { b.insert(k); }
    let c_remaining = target_len.saturating_sub(c.len());
    for &k in unique_c_keys.iter().take(c_remaining) { c.insert(k); }

    (a, b, c)
}

const OVERLAPS: &[f64] = &[0.10, 0.50, 0.90];

fn bench_fingerprint_diff(c: &mut Criterion) {
    if !should_run_op("fingerprint_diff") {
        return;
    }

    #[cfg(feature = "gpu32")]
    let impl_name = "radix_tree_gpu";
    #[cfg(not(feature = "gpu32"))]
    let impl_name = "radix_tree";

    for &(capacity, size_label) in CAPACITIES {
        for &lf in LOAD_FACTORS {
            for &overlap in &[0.50, 0.90, 0.99] {
                let mutation = 1.0 - overlap;
                let param = format!("size={size_label}/lf={lf:.2}/mut={mutation:.2}");
                let (a, b) = build_pair(capacity, lf, overlap);

                let mut group = c.benchmark_group(format!("impl={impl_name}/op=fingerprint_diff"));
                group.throughput(Throughput::Elements(capacity as u64));
                group.bench_with_input(
                    BenchmarkId::from_parameter(&param),
                    &param,
                    |bench, _| {
                        bench.iter(|| {
                            black_box(fingerprint_diff(&a, &b))
                        });
                    },
                );
                group.finish();
            }
        }
    }
}

fn bench_set_intersection(c: &mut Criterion) {
    if !should_run_op("set_intersection") {
        return;
    }

    #[cfg(feature = "gpu32")]
    let impl_name = "radix_tree_gpu";
    #[cfg(not(feature = "gpu32"))]
    let impl_name = "radix_tree";

    for &(capacity, size_label) in CAPACITIES {
        for &lf in LOAD_FACTORS {
            for &overlap in OVERLAPS {
                let param = format!("size={size_label}/lf={lf:.2}/overlap={overlap:.2}");
                let (a, b) = build_pair(capacity, lf, overlap);

                let mut group = c.benchmark_group(format!("impl={impl_name}/op=set_intersection"));
                group.throughput(Throughput::Elements(capacity as u64));
                group.bench_with_input(
                    BenchmarkId::from_parameter(&param),
                    &param,
                    |bench, _| {
                        bench.iter(|| {
                            black_box(set_intersection(&a, &b))
                        });
                    },
                );
                group.finish();
            }
        }
    }
}

fn bench_set_predicate(c: &mut Criterion) {
    if !should_run_op("set_predicate") {
        return;
    }

    #[cfg(feature = "gpu32")]
    let impl_name = "radix_tree_gpu";
    #[cfg(not(feature = "gpu32"))]
    let impl_name = "radix_tree";

    struct Pred2 { name: &'static str, mask: u8 }
    struct Pred3 { name: &'static str, mask: u64 }

    let preds_2: &[Pred2] = &[
        Pred2 { name: "intersect_2", mask: PRED2_INTERSECT },
        Pred2 { name: "diff_ab",     mask: PRED2_DIFF_AB },
        Pred2 { name: "sym_diff",    mask: PRED2_SYM_DIFF },
    ];
    let preds_3: &[Pred3] = &[
        Pred3 { name: "consensus_3", mask: PRED3_CONSENSUS },
        Pred3 { name: "unique_a",    mask: PRED3_UNIQUE_A },
    ];

    for &(capacity, size_label) in CAPACITIES {
        for &lf in LOAD_FACTORS {
            for &overlap in OVERLAPS {
                // 2-array predicates
                let (a, b) = build_pair(capacity, lf, overlap);
                for pred in preds_2 {
                    let param = format!(
                        "size={size_label}/lf={lf:.2}/overlap={overlap:.2}/pred={}",
                        pred.name
                    );
                    let mut group = c.benchmark_group(
                        format!("impl={impl_name}/op=set_predicate")
                    );
                    group.throughput(Throughput::Elements(capacity as u64));
                    group.bench_with_input(
                        BenchmarkId::from_parameter(&param),
                        &param,
                        |bench, _| {
                            bench.iter(|| {
                                black_box(set_predicate_2(&a, &b, pred.mask))
                            });
                        },
                    );
                    group.finish();
                }

                // 3-array predicates
                let (a3, b3, c3) = build_triple(capacity, lf, overlap, overlap);
                for pred in preds_3 {
                    let param = format!(
                        "size={size_label}/lf={lf:.2}/overlap={overlap:.2}/pred={}",
                        pred.name
                    );
                    let mut group = c.benchmark_group(
                        format!("impl={impl_name}/op=set_predicate")
                    );
                    group.throughput(Throughput::Elements(capacity as u64));
                    group.bench_with_input(
                        BenchmarkId::from_parameter(&param),
                        &param,
                        |bench, _| {
                            bench.iter(|| {
                                black_box(set_predicate_3(&a3, &b3, &c3, pred.mask))
                            });
                        },
                    );
                    group.finish();
                }
            }
        }
    }
}

criterion_group!(
    m1_gpu_comparison,
    bench_iter_across_sizes,
    bench_contains_greedy_across_sizes,
    bench_fingerprint_diff,
    bench_set_intersection,
    bench_set_predicate
);
criterion_main!(m1_gpu_comparison);
