use core::hash::{BuildHasher, Hasher};

pub mod m0_table;
pub mod radix_tree;

pub use m0_table::M0Table;

/// Stafford variant of splitmix64. Shared hash function used by all
/// implementations so benchmark comparisons isolate data-structure cost,
/// not hash-function cost.
#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Hasher that applies [`mix64`] to a single `u64` key.
/// Only valid for `u64` inputs -- will panic on other write calls.
#[derive(Default)]
pub struct Mix64Hasher {
    value: u64,
}

impl Hasher for Mix64Hasher {
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.value = mix64(i);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.value
    }

    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("Mix64Hasher only supports u64 keys");
    }
}

#[derive(Clone, Default)]
pub struct Mix64BuildHasher;

impl BuildHasher for Mix64BuildHasher {
    type Hasher = Mix64Hasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        Mix64Hasher::default()
    }
}

pub trait IndexTable {
    type KeyIter<'a>: Iterator<Item = u64> where Self: 'a;

    fn insert(&mut self, id: u64) -> bool;
    fn contains(&self, id: u64) -> bool;
    /// Probabilistic membership test — may return false positives but never
    /// false negatives. Implementations that cannot answer probabilistically
    /// should fall back to exact `contains`.
    fn contains_probable(&self, id: u64) -> bool;
    /// Returns the stored key if present. For key-only tables this returns the
    /// key itself; for key-value tables it will return the value.
    fn get(&self, id: u64) -> Option<u64>;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    /// Returns an iterator over all stored keys.
    fn iter(&self) -> Self::KeyIter<'_>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn load_factor(&self) -> f64 {
        let capacity = self.capacity();
        if capacity == 0 {
            0.0
        } else {
            self.len() as f64 / capacity as f64
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    PrefixTooLarge(u8),
    CapacityTooSmall { prefix_bits: u8, arena_capacity: usize },
    ZeroCapacity,
}

#[derive(Debug, Clone, Copy)]
pub struct RadixConfig {
    pub prefix_bits: u8,
    pub arena_capacity: usize,
}
