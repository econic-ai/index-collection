/// M1/M2 layout uses 64 slots per bucket and always stores fp8.
pub const RADIX_BUCKET_SLOTS: usize = 64;
pub const RADIX_SLOT_BITS: u8 = 6;
pub const RADIX_FP_EMPTY: u8 = 0x00;
pub const RADIX_MAX_PROBE_ROUNDS: u8 = 64;

use crate::IndexTable;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(feature = "radix-debug-flow")]
use std::sync::Arc;
#[cfg(feature = "radix-debug-flow")]
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy)]
pub struct RadixTreeConfig {
    /// Global slot budget: total_slots = 1 << capacity_bits.
    pub capacity_bits: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RadixTreeLayout {
    pub capacity_bits: u8,
    pub total_slots: usize,
    pub bucket_bits: u8,
    pub bucket_count: usize,
    pub bucket_slots: usize,
    pub slot_bits: u8,
}

#[derive(Debug, Clone)]
pub struct RadixTree {
    pub config: RadixTreeConfig,
    pub layout: RadixTreeLayout,
    /// 1 byte per slot. 0x00 means empty.
    pub fingerprints: Vec<u8>,
    /// Full-key confirmation arena.
    pub hashes: Vec<u64>,
    pub len: usize,
    #[cfg(feature = "radix-debug-flow")]
    debug_flow: DebugFlowCounters,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HashParts {
    pub bucket: usize,
    pub preferred_slot: usize,
    pub fingerprint: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadixTreeBuildError {
    CapacityBitsTooSmall(u8),
    CapacityBitsTooLarge(u8),
}

#[cfg(feature = "radix-debug-flow")]
#[derive(Debug, Default)]
struct DebugFlowCounterState {
    simd_scan_calls: AtomicU64,
    scalar_scan_calls: AtomicU64,
    preferred_slot_hits: AtomicU64,
    lookup_rounds: AtomicU64,
    insert_rounds: AtomicU64,
}

#[cfg(feature = "radix-debug-flow")]
#[derive(Debug, Clone, Default)]
struct DebugFlowCounters {
    inner: Arc<DebugFlowCounterState>,
}

#[cfg(feature = "radix-debug-flow")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DebugFlowSnapshot {
    pub simd_scan_calls: u64,
    pub scalar_scan_calls: u64,
    pub preferred_slot_hits: u64,
    pub lookup_rounds: u64,
    pub insert_rounds: u64,
}

impl RadixTree {
    pub fn new(config: RadixTreeConfig) -> Result<Self, RadixTreeBuildError> {
        if config.capacity_bits < RADIX_SLOT_BITS {
            return Err(RadixTreeBuildError::CapacityBitsTooSmall(
                config.capacity_bits,
            ));
        }

        // bucket bits + slot bits + fp bits must fit in a u64 hash.
        if config.capacity_bits > 56 {
            return Err(RadixTreeBuildError::CapacityBitsTooLarge(
                config.capacity_bits,
            ));
        }

        let bucket_bits = config.capacity_bits - RADIX_SLOT_BITS;
        let total_slots = 1usize << config.capacity_bits;
        let bucket_count = 1usize << bucket_bits;
        let layout = RadixTreeLayout {
            capacity_bits: config.capacity_bits,
            total_slots,
            bucket_bits,
            bucket_count,
            bucket_slots: RADIX_BUCKET_SLOTS,
            slot_bits: RADIX_SLOT_BITS,
        };

        Ok(Self {
            config,
            layout,
            fingerprints: vec![RADIX_FP_EMPTY; total_slots],
            hashes: vec![0; total_slots],
            len: 0,
            #[cfg(feature = "radix-debug-flow")]
            debug_flow: DebugFlowCounters::default(),
        })
    }

    pub fn hash_parts(&self, id: u64) -> HashParts {
        self.hash_parts_for_round(mix64(id), 0)
    }

    #[inline]
    fn hash_parts_for_round(&self, mixed_hash: u64, round: u8) -> HashParts {
        let rotated = mixed_hash.rotate_left(round as u32);
        let bucket_bits = self.layout.bucket_bits;
        let slot_shift = 64 - bucket_bits - RADIX_SLOT_BITS;
        let fp_shift = slot_shift - 8;

        let bucket = if bucket_bits == 0 {
            0
        } else {
            (rotated >> (64 - bucket_bits)) as usize
        };
        let preferred_slot = ((rotated >> slot_shift) & ((RADIX_BUCKET_SLOTS - 1) as u64)) as usize;
        let raw_fp = ((rotated >> fp_shift) & 0xff) as u8;
        let fingerprint = if raw_fp == RADIX_FP_EMPTY { 1 } else { raw_fp };

        HashParts {
            bucket,
            preferred_slot,
            fingerprint,
        }
    }

    #[inline]
    fn slot_index(&self, bucket: usize, slot: usize) -> usize {
        bucket * RADIX_BUCKET_SLOTS + slot
    }

    #[inline]
    fn bucket_base_index(&self, bucket: usize) -> usize {
        bucket * RADIX_BUCKET_SLOTS
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    unsafe fn mask_from_cmp16(cmp: uint8x16_t) -> u16 {
        let mut lanes = [0u8; 16];
        // SAFETY: lanes has exactly 16 bytes and cmp is a valid NEON vector.
        unsafe { vst1q_u8(lanes.as_mut_ptr(), cmp) };
        let mut mask = 0u16;
        let mut i = 0usize;
        while i < 16 {
            mask |= (((lanes[i] >> 7) & 1) as u16) << i;
            i += 1;
        }
        mask
    }

    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn scan_bucket_masks_simd(&self, bucket: usize, target_fp: u8) -> (u64, u64) {
        #[cfg(feature = "radix-debug-flow")]
        self.debug_flow
            .inner
            .simd_scan_calls
            .fetch_add(1, Ordering::Relaxed);

        let base = self.bucket_base_index(bucket);
        let bucket_ptr = self.fingerprints[base..].as_ptr();
        let mut empty_mask = 0u64;
        let mut match_mask = 0u64;

        // SAFETY: bucket_ptr points to a contiguous region of at least 64 bytes.
        // NEON loads are unaligned-safe on AArch64.
        unsafe {
            let zero = vdupq_n_u8(0);
            let target = vdupq_n_u8(target_fp);
            let mut chunk = 0usize;
            while chunk < 4 {
                let p = bucket_ptr.add(chunk * 16);
                let vals = vld1q_u8(p);
                let empty_cmp = vceqq_u8(vals, zero);
                let match_cmp = vceqq_u8(vals, target);
                let e = Self::mask_from_cmp16(empty_cmp) as u64;
                let m = Self::mask_from_cmp16(match_cmp) as u64;
                empty_mask |= e << (chunk * 16);
                match_mask |= m << (chunk * 16);
                chunk += 1;
            }
        }

        (empty_mask, match_mask)
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn scan_bucket_masks_simd(&self, bucket: usize, target_fp: u8) -> (u64, u64) {
        self.scan_bucket_masks_scalar(bucket, target_fp)
    }

    #[inline]
    fn scan_bucket_masks(&self, bucket: usize, target_fp: u8) -> (u64, u64) {
        self.scan_bucket_masks_simd(bucket, target_fp)
    }

    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn scan_bucket_masks_scalar(&self, bucket: usize, target_fp: u8) -> (u64, u64) {
        #[cfg(feature = "radix-debug-flow")]
        self.debug_flow
            .inner
            .scalar_scan_calls
            .fetch_add(1, Ordering::Relaxed);

        let mut empty_mask = 0u64;
        let mut match_mask = 0u64;
        let base = self.bucket_base_index(bucket);

        for i in 0..RADIX_BUCKET_SLOTS {
            let fp = self.fingerprints[base + i];
            if fp == RADIX_FP_EMPTY {
                empty_mask |= 1u64 << i;
            }
            if fp == target_fp {
                match_mask |= 1u64 << i;
            }
        }

        (empty_mask, match_mask)
    }

    #[inline]
    fn ordered_slot(preferred_slot: usize, offset: usize) -> usize {
        (preferred_slot + offset) & (RADIX_BUCKET_SLOTS - 1)
    }

    #[cfg(feature = "radix-debug-flow")]
    pub fn debug_flow_snapshot(&self) -> DebugFlowSnapshot {
        DebugFlowSnapshot {
            simd_scan_calls: self.debug_flow.inner.simd_scan_calls.load(Ordering::Relaxed),
            scalar_scan_calls: self.debug_flow.inner.scalar_scan_calls.load(Ordering::Relaxed),
            preferred_slot_hits: self
                .debug_flow
                .inner
                .preferred_slot_hits
                .load(Ordering::Relaxed),
            lookup_rounds: self.debug_flow.inner.lookup_rounds.load(Ordering::Relaxed),
            insert_rounds: self.debug_flow.inner.insert_rounds.load(Ordering::Relaxed),
        }
    }

    #[cfg(feature = "radix-debug-flow")]
    pub fn reset_debug_flow_counters(&self) {
        self.debug_flow.inner.simd_scan_calls.store(0, Ordering::Relaxed);
        self.debug_flow.inner.scalar_scan_calls.store(0, Ordering::Relaxed);
        self.debug_flow
            .inner
            .preferred_slot_hits
            .store(0, Ordering::Relaxed);
        self.debug_flow.inner.lookup_rounds.store(0, Ordering::Relaxed);
        self.debug_flow.inner.insert_rounds.store(0, Ordering::Relaxed);
    }

    #[cfg(feature = "radix-debug-flow")]
    pub fn debug_lookup_flow(&self, id: u64) -> Vec<String> {
        let mut trace = Vec::new();
        let mixed_hash = mix64(id);
        trace.push(format!("lookup id={id} mixed_hash=0x{mixed_hash:016x}"));

        for round in 0..RADIX_MAX_PROBE_ROUNDS {
            let parts = self.hash_parts_for_round(mixed_hash, round);
            let preferred_idx = self.slot_index(parts.bucket, parts.preferred_slot);
            let preferred_fp = self.fingerprints[preferred_idx];
            trace.push(format!(
                "round={round} bucket={} preferred_slot={} preferred_idx={} target_fp=0x{:02x} preferred_fp=0x{:02x}",
                parts.bucket, parts.preferred_slot, preferred_idx, parts.fingerprint, preferred_fp
            ));

            if preferred_fp == parts.fingerprint {
                let preferred_hash = self.hashes[preferred_idx];
                let preferred_hash_match = preferred_hash == id;
                trace.push(format!(
                    "  preferred_fp_match=true preferred_hash=0x{preferred_hash:016x} preferred_hash_match={preferred_hash_match}"
                ));
                if preferred_hash_match {
                    trace.push("  return=true via preferred-slot fast path".to_string());
                    return trace;
                }
            } else {
                trace.push("  preferred_fp_match=false".to_string());
            }

            let (empty_mask, match_mask) = self.scan_bucket_masks(parts.bucket, parts.fingerprint);
            let rot_match = match_mask.rotate_right(parts.preferred_slot as u32);
            let rot_empty = empty_mask.rotate_right(parts.preferred_slot as u32);
            let first_empty = if rot_empty == 0 {
                RADIX_BUCKET_SLOTS
            } else {
                rot_empty.trailing_zeros() as usize
            };
            trace.push(format!(
                "  scan empty_mask=0x{empty_mask:016x} match_mask=0x{match_mask:016x} first_empty_offset={first_empty}"
            ));

            let mut probe_match_mask = if first_empty == 0 {
                0
            } else if first_empty == RADIX_BUCKET_SLOTS {
                rot_match
            } else {
                rot_match & ((1u64 << first_empty) - 1)
            };

            while probe_match_mask != 0 {
                let offset = probe_match_mask.trailing_zeros() as usize;
                let slot = Self::ordered_slot(parts.preferred_slot, offset);
                let idx = self.slot_index(parts.bucket, slot);
                let candidate_hash = self.hashes[idx];
                let candidate_match = candidate_hash == id;
                trace.push(format!(
                    "  candidate offset={offset} slot={slot} idx={idx} candidate_hash=0x{candidate_hash:016x} candidate_match={candidate_match}"
                ));
                if candidate_match {
                    trace.push("  return=true via scanned candidate".to_string());
                    return trace;
                }
                probe_match_mask &= probe_match_mask - 1;
            }
        }

        trace.push("return=false after all rounds".to_string());
        trace
    }
}

impl IndexTable for RadixTree {
    fn insert(&mut self, id: u64) -> bool {
        if self.len == self.layout.total_slots {
            return false;
        }

        let mixed_hash = mix64(id);
        for round in 0..RADIX_MAX_PROBE_ROUNDS {
            #[cfg(feature = "radix-debug-flow")]
            self.debug_flow
                .inner
                .insert_rounds
                .fetch_add(1, Ordering::Relaxed);

            let parts = self.hash_parts_for_round(mixed_hash, round);
            let (empty_mask, match_mask) = self.scan_bucket_masks(parts.bucket, parts.fingerprint);
            let rot_match = match_mask.rotate_right(parts.preferred_slot as u32);
            let rot_empty = empty_mask.rotate_right(parts.preferred_slot as u32);
            let first_empty = if rot_empty == 0 {
                None
            } else {
                Some(rot_empty.trailing_zeros() as usize)
            };

            let mut probe_match_mask = match first_empty {
                Some(limit) if limit == 0 => 0,
                Some(limit) => rot_match & ((1u64 << limit) - 1),
                None => rot_match,
            };

            while probe_match_mask != 0 {
                let offset = probe_match_mask.trailing_zeros() as usize;
                let slot = Self::ordered_slot(parts.preferred_slot, offset);
                let idx = self.slot_index(parts.bucket, slot);
                if self.hashes[idx] == id {
                    return false;
                }
                probe_match_mask &= probe_match_mask - 1;
            }

            if let Some(offset) = first_empty {
                let slot = Self::ordered_slot(parts.preferred_slot, offset);
                let idx = self.slot_index(parts.bucket, slot);
                self.fingerprints[idx] = parts.fingerprint;
                self.hashes[idx] = id;
                self.len += 1;
                return true;
            }
        }

        false
    }

    fn lookup(&self, id: u64) -> bool {
        let mixed_hash = mix64(id);
        for round in 0..RADIX_MAX_PROBE_ROUNDS {
            #[cfg(feature = "radix-debug-flow")]
            self.debug_flow
                .inner
                .lookup_rounds
                .fetch_add(1, Ordering::Relaxed);

            let parts = self.hash_parts_for_round(mixed_hash, round);
            let preferred_idx = self.slot_index(parts.bucket, parts.preferred_slot);
            let preferred_fp = self.fingerprints[preferred_idx];
            #[cfg(feature = "radix-debug-flow")]
            if std::env::var_os("RADIX_PANIC_ON_FP8_MISS").is_some()
                && preferred_fp != parts.fingerprint
            {
                panic!(
                    "preferred fp8 mismatch: id={} round={} bucket={} slot={} expected_fp=0x{:02x} actual_fp=0x{:02x}",
                    id,
                    round,
                    parts.bucket,
                    parts.preferred_slot,
                    parts.fingerprint,
                    preferred_fp
                );
            }
            if preferred_fp == parts.fingerprint && self.hashes[preferred_idx] == id {
                #[cfg(feature = "radix-debug-flow")]
                self.debug_flow
                    .inner
                    .preferred_slot_hits
                    .fetch_add(1, Ordering::Relaxed);
                return true;
            }
            // return true;

            let (empty_mask, match_mask) = self.scan_bucket_masks(parts.bucket, parts.fingerprint);
            let rot_match = match_mask.rotate_right(parts.preferred_slot as u32);
            let rot_empty = empty_mask.rotate_right(parts.preferred_slot as u32);
            let first_empty = if rot_empty == 0 {
                RADIX_BUCKET_SLOTS
            } else {
                rot_empty.trailing_zeros() as usize
            };

            let mut probe_match_mask = if first_empty == 0 {
                0
            } else if first_empty == RADIX_BUCKET_SLOTS {
                rot_match
            } else {
                rot_match & ((1u64 << first_empty) - 1)
            };

            while probe_match_mask != 0 {
                let offset = probe_match_mask.trailing_zeros() as usize;
                let slot = Self::ordered_slot(parts.preferred_slot, offset);
                let idx = self.slot_index(parts.bucket, slot);
                if self.hashes[idx] == id {
                    return true;
                }
                probe_match_mask &= probe_match_mask - 1;
            }
        }

        false
    }

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        self.layout.total_slots
    }
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::{
        RADIX_BUCKET_SLOTS, RADIX_FP_EMPTY, RADIX_MAX_PROBE_ROUNDS, RadixTree,
        RadixTreeBuildError, RadixTreeConfig,
    };
    use crate::IndexTable;

    #[test]
    fn rejects_capacity_bits_below_slot_bits() {
        let t = RadixTree::new(RadixTreeConfig { capacity_bits: 5 });
        assert!(matches!(
            t,
            Err(RadixTreeBuildError::CapacityBitsTooSmall(5))
        ));
    }

    #[test]
    fn builds_layout_from_capacity_bits() {
        let t = RadixTree::new(RadixTreeConfig { capacity_bits: 18 }).expect("build");
        assert_eq!(t.layout.total_slots, 1 << 18);
        assert_eq!(t.layout.bucket_bits, 12);
        assert_eq!(t.layout.bucket_count, 1 << 12);
        assert_eq!(t.layout.bucket_slots, RADIX_BUCKET_SLOTS);
    }

    #[test]
    fn insert_lookup_and_duplicate_behavior() {
        let mut t = RadixTree::new(RadixTreeConfig { capacity_bits: 10 }).expect("build");
        assert!(t.insert(42));
        assert!(t.lookup(42));
        assert!(!t.insert(42));
        assert_eq!(t.len(), 1);
        assert!(!t.lookup(99_999));
    }

    #[test]
    fn fingerprints_start_empty() {
        let t = RadixTree::new(RadixTreeConfig { capacity_bits: 10 }).expect("build");
        assert!(t.fingerprints.iter().all(|fp| *fp == RADIX_FP_EMPTY));
    }

    #[test]
    fn can_fill_to_high_load_with_multi_round_probe() {
        let capacity_bits = 16;
        let mut t = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let target = ((t.capacity() as f64) * 0.95) as usize;
        for i in 0..target {
            assert!(
                t.insert((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
                "failed at insert {} with rounds={}",
                i,
                RADIX_MAX_PROBE_ROUNDS
            );
        }
        assert_eq!(t.len(), target);
    }

    #[cfg(feature = "radix-debug-flow")]
    #[test]
    fn debug_flow_preferred_slot_short_circuit_hit() {
        let mut t = RadixTree::new(RadixTreeConfig { capacity_bits: 10 }).expect("build");
        let key = 42_u64;
        assert!(t.insert(key));
        t.reset_debug_flow_counters();

        assert!(t.lookup(key));
        let snap = t.debug_flow_snapshot();
        assert_eq!(snap.preferred_slot_hits, 1);
        assert_eq!(snap.lookup_rounds, 1);
        assert_eq!(snap.simd_scan_calls + snap.scalar_scan_calls, 0);
    }

    #[cfg(feature = "radix-debug-flow")]
    #[test]
    fn debug_flow_lookup_miss_scans_bucket_path() {
        let mut t = RadixTree::new(RadixTreeConfig { capacity_bits: 10 }).expect("build");
        let present = 123_u64;
        let absent = 999_999_u64;
        assert!(t.insert(present));
        t.reset_debug_flow_counters();

        assert!(!t.lookup(absent));
        let snap = t.debug_flow_snapshot();
        assert!(snap.lookup_rounds >= 1);
        assert!(
            snap.simd_scan_calls + snap.scalar_scan_calls > 0,
            "expected at least one scan call, got simd={} scalar={}",
            snap.simd_scan_calls,
            snap.scalar_scan_calls
        );
    }

    #[cfg(feature = "radix-debug-flow")]
    #[test]
    fn debug_flow_lookup_trace_prints_for_tiny_hit() {
        let mut t = RadixTree::new(RadixTreeConfig { capacity_bits: 10 }).expect("build");
        let key = 42_u64;
        assert!(t.insert(key));
        let trace = t.debug_lookup_flow(key);
        for line in &trace {
            println!("{line}");
        }
        assert!(trace.iter().any(|l| l.contains("return=true via preferred-slot fast path")));
    }
}
