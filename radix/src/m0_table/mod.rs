use hashbrown::HashSet;

use crate::{BuildError, IndexTable, Mix64BuildHasher, RadixConfig};

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
    type KeyIter<'a> = std::iter::Copied<hashbrown::hash_set::Iter<'a, u64>>;

    fn insert(&mut self, id: u64) -> bool {
        self.set.insert(id)
    }

    fn contains(&self, id: u64) -> bool {
        self.set.contains(&id)
    }

    // TODO: hashbrown does not expose raw slot-level access for a greedy
    // linear scan. The default trait impl (falls back to `contains`) is used.

    fn contains_probable(&self, id: u64) -> bool {
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

    fn iter(&self) -> Self::KeyIter<'_> {
        self.set.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::M0Table;
    use crate::{BuildError, IndexTable, RadixConfig};

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
