use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use radix::IndexTable;
use radix::radix_tree::{RadixTree, RadixTreeConfig};
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

fn bench_iter_across_sizes(c: &mut Criterion) {
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

criterion_group!(m1_gpu_comparison, bench_iter_across_sizes);
criterion_main!(m1_gpu_comparison);
