use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use radix::IndexTable;
#[cfg(feature = "bench-m0-hashbrown")]
use radix::{M0Table, RadixConfig};
#[cfg(feature = "bench-radix-tree")]
use radix::radix_tree::{RadixTree, RadixTreeConfig};
use std::collections::HashSet;
use std::env;
use std::hint::black_box;

const LOAD_FACTORS: &[f64] = &[0.01, 0.25, 0.50, 0.75];
// const LOAD_FACTORS: &[f64] = &[0.50, 0.75, 0.90, 0.95, 0.99];
const CAPACITY: usize = 1 << 18;
const LOOKUPS_PER_ITERATION: u64 = 1;
const INSERTS_PER_ITERATION: u64 = 128;
const TINY_TARGET_LEN: usize = 1;

#[cfg(not(any(feature = "bench-m0-hashbrown", feature = "bench-radix-tree")))]
compile_error!("No benchmark implementation feature selected.");

fn make_ids(n: usize, seed: u64) -> Vec<u64> {
    (0..n).map(|i| mix64(seed.wrapping_add(i as u64))).collect()
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[cfg(feature = "bench-radix-tree")]
fn capacity_bits_from_slots(total_slots: usize) -> u8 {
    debug_assert!(total_slots.is_power_of_two());
    total_slots.trailing_zeros() as u8
}

#[cfg(feature = "bench-m0-hashbrown")]
fn new_m0_table() -> M0Table {
    M0Table::new(RadixConfig {
        prefix_bits: 10,
        arena_capacity: CAPACITY,
    })
    .expect("valid m0 benchmark config")
}

#[cfg(feature = "bench-radix-tree")]
fn new_radix_tree_table() -> RadixTree {
    let capacity_bits = capacity_bits_from_slots(CAPACITY);
    RadixTree::new(RadixTreeConfig { capacity_bits }).expect("valid radix benchmark config")
}

fn build_filled_table<T, F>(load_factor: f64, impl_name: &str, mut table_factory: F) -> Result<(T, Vec<u64>), String>
where
    T: IndexTable + Clone,
    F: FnMut() -> T,
{
    let target_len = (CAPACITY as f64 * load_factor) as usize;
    let mut table = table_factory();

    assert_eq!(
        table.capacity(),
        CAPACITY,
        "impl={impl_name}: expected capacity {CAPACITY}, got {} — \
         the benchmark assumes both implementations allocate exactly {CAPACITY} slots",
        table.capacity(),
    );

    let present = make_ids(target_len + 10_000, 0x1234_5678);
    for (inserted_count, id) in present.iter().take(target_len).enumerate() {
        let inserted = table.insert(*id);
        if !inserted {
            return Err(format!(
                "impl={impl_name} failed to preload to lf={load_factor:.2}: \
                 insert failed at {} / {} (effective_lf={:.6})",
                inserted_count + 1,
                target_len,
                (inserted_count + 1) as f64 / CAPACITY as f64
            ));
        }
    }

    Ok((table, present[..target_len].to_vec()))
}

fn build_tiny_hit_table<T, F>(impl_name: &str, mut table_factory: F) -> Result<(T, u64), String>
where
    T: IndexTable + Clone,
    F: FnMut() -> T,
{
    let mut table = table_factory();

    assert_eq!(
        table.capacity(),
        CAPACITY,
        "impl={impl_name}: expected capacity {CAPACITY}, got {} — \
         the benchmark assumes both implementations allocate exactly {CAPACITY} slots",
        table.capacity(),
    );

    let key = make_ids(TINY_TARGET_LEN, 0x2222_3333)
        .first()
        .copied()
        .expect("tiny key generation");
    if !table.insert(key) {
        return Err(format!(
            "impl={impl_name} failed tiny preload for contains_hit_tiny"
        ));
    }
    Ok((table, key))
}

fn parse_csv_set(var_name: &str) -> Option<HashSet<String>> {
    env::var(var_name).ok().map(|raw| {
        raw.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect()
    })
}

fn normalize_load_factor_token(token: &str) -> Option<String> {
    let parsed = token.trim().parse::<f64>().ok()?;
    let normalized = if parsed > 1.0 { parsed / 100.0 } else { parsed };
    if !(0.0..=1.0).contains(&normalized) {
        return None;
    }
    Some(format!("{normalized:.2}"))
}

fn selected_load_factors() -> Vec<f64> {
    let Some(raw) = env::var("LFS").ok() else {
        return LOAD_FACTORS.to_vec();
    };

    let selected: HashSet<String> = raw
        .split(',')
        .filter_map(normalize_load_factor_token)
        .collect();

    LOAD_FACTORS
        .iter()
        .copied()
        .filter(|lf| selected.contains(&format!("{lf:.2}")))
        .collect()
}

fn should_run_op(op_name: &str) -> bool {
    match parse_csv_set("OPS") {
        Some(set) => set.contains(op_name),
        None => true,
    }
}

fn should_run_impl(impl_name: &str) -> bool {
    match parse_csv_set("IMPL") {
        Some(set) => set.contains("all") || set.contains(impl_name),
        None => true,
    }
}

fn bench_lookup_hit<T, F>(c: &mut Criterion, impl_name: &str, load_factors: &[f64], table_factory: F)
where
    T: IndexTable + Clone,
    F: FnMut() -> T + Copy,
{
    let mut group = c.benchmark_group(format!("impl={impl_name}/op=lookup_hit"));
    for &lf in load_factors {
        let (table, keys) =
            build_filled_table::<T, _>(lf, impl_name, table_factory).unwrap_or_else(|msg| panic!("{msg}"));
        group.throughput(Throughput::Elements(LOOKUPS_PER_ITERATION));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{lf:.2}")), &lf, |b, _| {
            let mut idx = 0usize;
            b.iter(|| {
                let key = keys[idx % keys.len()];
                idx += 1;
                black_box(table.contains(key))
            });
        });
    }

    group.finish();
}

fn bench_lookup_miss<T, F>(c: &mut Criterion, impl_name: &str, load_factors: &[f64], table_factory: F)
where
    T: IndexTable + Clone,
    F: FnMut() -> T + Copy,
{
    let mut group = c.benchmark_group(format!("impl={impl_name}/op=lookup_miss"));
    for &lf in load_factors {
        let (table, keys) =
            build_filled_table::<T, _>(lf, impl_name, table_factory).unwrap_or_else(|msg| panic!("{msg}"));
        let misses = make_ids(keys.len(), 0x9999_0000);
        group.throughput(Throughput::Elements(LOOKUPS_PER_ITERATION));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{lf:.2}")), &lf, |b, _| {
            let mut idx = 0usize;
            b.iter(|| {
                let key = misses[idx % misses.len()];
                idx += 1;
                black_box(table.contains(key))
            });
        });
    }

    group.finish();
}

fn bench_lookup_hit_tiny<T, F>(c: &mut Criterion, impl_name: &str, table_factory: F)
where
    T: IndexTable + Clone,
    F: FnMut() -> T + Copy,
{
    let mut group = c.benchmark_group(format!("impl={impl_name}/op=lookup_hit_tiny"));
    let tiny_lf = TINY_TARGET_LEN as f64 / CAPACITY as f64;
    let (table, key) =
        build_tiny_hit_table::<T, _>(impl_name, table_factory).unwrap_or_else(|msg| panic!("{msg}"));
    group.throughput(Throughput::Elements(LOOKUPS_PER_ITERATION));
    group.bench_with_input(
        BenchmarkId::from_parameter(format!("{tiny_lf:.6}")),
        &tiny_lf,
        |b, _| b.iter(|| black_box(table.contains(key))),
    );
    group.finish();
}

fn bench_insert<T, F>(c: &mut Criterion, impl_name: &str, load_factors: &[f64], table_factory: F)
where
    T: IndexTable + Clone,
    F: FnMut() -> T + Copy,
{
    let mut group = c.benchmark_group(format!("impl={impl_name}/op=insert_marginal"));
    for &lf in load_factors {
        let (table, _) =
            build_filled_table::<T, _>(lf, impl_name, table_factory).unwrap_or_else(|msg| panic!("{msg}"));
        let seeds = make_ids(10_000, 0xabcd_ef00);

        group.throughput(Throughput::Elements(INSERTS_PER_ITERATION));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{lf:.2}")), &lf, |b, _| {
            let mut idx = 0usize;
            b.iter_batched(
                || table.clone(),
                |mut t| {
                    for _ in 0..INSERTS_PER_ITERATION {
                        let key = seeds[idx % seeds.len()].wrapping_add(idx as u64);
                        idx += 1;
                        let _ = black_box(t.insert(key));
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn run_impl_benches<T, F>(c: &mut Criterion, impl_name: &str, table_factory: F)
where
    T: IndexTable + Clone,
    F: FnMut() -> T + Copy,
{
    if !should_run_impl(impl_name) {
        return;
    }

    let load_factors = selected_load_factors();
    if load_factors.is_empty() {
        return;
    }

    if should_run_op("lookup_hit") {
        bench_lookup_hit::<T, _>(c, impl_name, &load_factors, table_factory);
    }
    if should_run_op("lookup_miss") {
        bench_lookup_miss::<T, _>(c, impl_name, &load_factors, table_factory);
    }
    if should_run_op("lookup_hit_tiny") {
        bench_lookup_hit_tiny::<T, _>(c, impl_name, table_factory);
    }
    if should_run_op("insert_marginal") {
        bench_insert::<T, _>(c, impl_name, &load_factors, table_factory);
    }
}

fn benches(c: &mut Criterion) {
    #[cfg(feature = "bench-m0-hashbrown")]
    run_impl_benches::<M0Table, _>(c, "m0_hashbrown", new_m0_table);
    #[cfg(feature = "bench-radix-tree")]
    run_impl_benches::<RadixTree, _>(c, "radix_tree", new_radix_tree_table);
}

criterion_group!(radix_m0, benches);
criterion_main!(radix_m0);
