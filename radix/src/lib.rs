use core::hash::{BuildHasher, Hasher};
use hashbrown::HashSet;

pub mod radix_tree;

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
    fn insert(&mut self, id: u64) -> bool;
    fn contains(&self, id: u64) -> bool;
    /// Returns the stored key if present. For key-only tables this returns the
    /// key itself; for key-value tables it will return the value.
    fn get(&self, id: u64) -> Option<u64>;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;

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

#[derive(Clone)]
pub struct M0Table {
    set: HashSet<u64, Mix64BuildHasher>,
    pub config: RadixConfig,
}

impl M0Table {
    pub fn new(config: RadixConfig) -> Result<Self, BuildError> {
        if config.arena_capacity == 0 {
            return Err(BuildError::ZeroCapacity);
        }

        if config.prefix_bits >= usize::BITS as u8 {
            return Err(BuildError::PrefixTooLarge(config.prefix_bits));
        }

        let min_capacity = 1usize << config.prefix_bits;
        if config.arena_capacity < min_capacity {
            return Err(BuildError::CapacityTooSmall {
                prefix_bits: config.prefix_bits,
                arena_capacity: config.arena_capacity,
            });
        }

        // Request 7/8 of arena_capacity so hashbrown allocates exactly
        // arena_capacity raw slots (hashbrown uses ceil(req * 8/7) rounded
        // to the next power of two as its raw allocation).
        let request = config.arena_capacity * 7 / 8;
        let set = HashSet::with_capacity_and_hasher(request, Mix64BuildHasher);
        debug_assert_eq!(
            set.capacity(),
            request,
            "hashbrown raw allocation mismatch: expected {} usable from {} raw slots, got {}",
            request,
            config.arena_capacity,
            set.capacity(),
        );

        Ok(Self { set, config })
    }
}

impl IndexTable for M0Table {
    fn insert(&mut self, id: u64) -> bool {
        self.set.insert(id)
    }

    fn contains(&self, id: u64) -> bool {
        self.set.contains(&id)
    }

    fn get(&self, id: u64) -> Option<u64> {
        self.set.get(&id).copied()
    }

    fn len(&self) -> usize {
        self.set.len()
    }

    fn capacity(&self) -> usize {
        self.config.arena_capacity
    }
}

#[cfg(test)]
mod tests {
    use super::{BuildError, IndexTable, M0Table, RadixConfig};

    #[test]
    fn rejects_zero_capacity() {
        let result = M0Table::new(RadixConfig {
            prefix_bits: 0,
            arena_capacity: 0,
        });

        assert!(matches!(result, Err(BuildError::ZeroCapacity)));
    }

    #[test]
    fn rejects_incoherent_prefix_and_capacity() {
        let result = M0Table::new(RadixConfig {
            prefix_bits: 8,
            arena_capacity: 64,
        });

        assert!(matches!(
            result,
            Err(BuildError::CapacityTooSmall {
                prefix_bits: 8,
                arena_capacity: 64
            })
        ));
    }

    #[test]
    fn inserts_and_contains() {
        let mut table = M0Table::new(RadixConfig {
            prefix_bits: 4,
            arena_capacity: 512,
        })
        .expect("valid config");

        let values = [1_u64, 2, 5, 42, 8_192, 1_048_576];
        for v in values {
            assert!(table.insert(v));
        }

        for v in values {
            assert!(table.contains(v));
        }

        assert!(!table.contains(9_999_999));
        assert_eq!(table.len(), values.len());
        assert!(table.load_factor() > 0.0);
    }

    #[test]
    fn get_returns_key_when_present() {
        let mut table = M0Table::new(RadixConfig {
            prefix_bits: 4,
            arena_capacity: 512,
        })
        .expect("valid config");

        assert!(table.insert(42));
        assert_eq!(table.get(42), Some(42));
        assert_eq!(table.get(9_999_999), None);
    }

    #[test]
    fn duplicate_insert_is_not_counted_twice() {
        let mut table = M0Table::new(RadixConfig {
            prefix_bits: 4,
            arena_capacity: 512,
        })
        .expect("valid config");

        assert!(table.insert(123));
        assert!(!table.insert(123));
        assert_eq!(table.len(), 1);
    }
}
