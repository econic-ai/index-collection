use hashbrown::HashSet;

pub mod radix_tree;

pub trait IndexTable {
    fn insert(&mut self, id: u64) -> bool;
    fn lookup(&self, id: u64) -> bool;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;

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
    set: HashSet<u64>,
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

        Ok(Self {
            set: HashSet::with_capacity(config.arena_capacity),
            config,
        })
    }
}

impl IndexTable for M0Table {
    fn insert(&mut self, id: u64) -> bool {
        self.set.insert(id)
    }

    fn lookup(&self, id: u64) -> bool {
        self.set.contains(&id)
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
    fn inserts_and_looks_up() {
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
            assert!(table.lookup(v));
        }

        assert!(!table.lookup(9_999_999));
        assert_eq!(table.len(), values.len());
        assert!(table.load_factor() > 0.0);
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
