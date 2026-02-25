# Radix (Milestone M0) Developer How-To

This directory is an isolated, single-package Rust project for the radix-hash milestones.

## Layout

- `src/lib.rs`: shared API trait plus M0 baseline implementation.
- `src/main.rs`: tiny runnable entrypoint.
- `benches/m0_bench.rs`: Criterion benchmark scaffold for M0.
- `docs/architecture.md`: architecture and milestone design notes.

## Commands

From `radix/`:

- `make help`
- `make build`
- `make run`
- `make test`
- `make bench`
- `make bench-quick`
- `make extract IMPL=m0_hashbrown [TAG=run1]`

From repository root:

- Build: `cargo build --manifest-path radix/Cargo.toml`
- Run: `cargo run --manifest-path radix/Cargo.toml`
- Test: `cargo test --manifest-path radix/Cargo.toml`
- Bench (Criterion): `cargo bench --manifest-path radix/Cargo.toml`
- Bench (compile-time impl select): `cargo bench --bench m0_bench --no-default-features --features bench-radix-tree --manifest-path radix/Cargo.toml`
- Bench compile only (M0 bench target): `cargo bench --bench m0_bench --no-run --manifest-path radix/Cargo.toml`
- Export to CSV: `python3 analysis/export_criterion.py --impl m0_hashbrown --tag run1 --criterion-dir target/criterion --out-dir analysis/data`

If you are already inside `radix/`, you can run the same commands without `--manifest-path`.

## API Sketch

Current API trait:

- `IndexTable::insert(id) -> bool`
- `IndexTable::contains(id) -> bool`
- `IndexTable::get(id) -> Option<u64>`
- `IndexTable::len() -> usize`
- `IndexTable::capacity() -> usize`
- `IndexTable::is_empty() -> bool`
- `IndexTable::load_factor() -> f64`

Current implementation:

- `M0Table` in `src/lib.rs`
  - backed by `hashbrown::HashSet<u64>` (SwissTable)
  - serves as the project baseline implementation for M0
  - keeps the same trait-facing API used by future custom milestones

## Benchmark Selection and Filters

Implementation selection is compile-time:

- `IMPL=m0_hashbrown` -> feature `bench-m0-hashbrown`
- `IMPL=radix_tree` -> feature `bench-radix-tree`

`make bench` maps `IMPL` to the feature and rebuilds the benchmark target for that implementation.

The benchmark binary reads optional runtime filters:

- `OPS`: operation list (supported: `lookup_hit`, `lookup_hit_tiny`, `lookup_miss`, `insert_marginal`)
- `LFS`: load factors list as decimals or percents (examples: `0.5,0.9` or `50,90`)

Examples:

- `make bench IMPL=m0_hashbrown`
- `make bench IMPL=radix_tree OPS=lookup_hit,lookup_miss LFS=50,90,99`

## Analysis Output

Export appends summary rows to per-implementation CSV files in `analysis/data`:

- `analysis/data/m0_hashbrown.csv`

Rows include: timestamp, impl, op, load_factor, metric, value, unit, optional tag.

## Minimal Example

```rust
use radix::{IndexTable, M0Table, RadixConfig};

fn main() {
    let mut table = M0Table::new(RadixConfig {
        prefix_bits: 10,
        arena_capacity: 1 << 20,
    })
    .expect("valid config");

    table.insert(42);
    assert!(table.contains(42));
}
```
