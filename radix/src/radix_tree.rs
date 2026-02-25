/// M1/M2 layout uses 64 slots per bucket and always stores fp8.
pub const RADIX_BUCKET_SLOTS: usize = 64;
pub const RADIX_SLOT_BITS: u8 = 6;
pub const RADIX_FP_EMPTY: u8 = 0x00;
pub const RADIX_MAX_PROBE_ROUNDS: u8 = 64;

use crate::{IndexTable, mix64};
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// 64-byte fingerprint bucket, cache-line-aligned.
/// Each bucket occupies exactly one cache line on typical hardware.
#[derive(Debug, Clone)]
#[repr(C, align(64))]
pub struct FpBucket(pub [u8; RADIX_BUCKET_SLOTS]);

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
    /// 1 byte per slot, grouped into cache-line-aligned buckets.
    pub fingerprints: Vec<FpBucket>,
    /// Full-key confirmation arena.
    pub hashes: Vec<u64>,
    pub len: usize,
    /// Precomputed for the hot path — avoids serial dependency chain.
    slot_shift: u8,
    fp_shift: u8,
    bucket_mask: usize,
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

/// Pack the high bit of each byte in a u64 into the low 8 bits.
/// Input: u64 where each byte is 0x00 or 0xFF (NEON comparison result).
/// Output: u8 bitmask with one bit per byte.
/// Used by scan_bucket_masks (insert path). Contains path uses sparse masks.
#[inline]
fn pack_msbs(v: u64) -> u8 {
    const HI: u64 = 0x8080_8080_8080_8080;
    const MUL: u64 = 0x0002_0408_1020_4081;
    ((v & HI).wrapping_mul(MUL) >> 56) as u8
}

#[allow(dead_code)]
const SPARSE_MSB: u64 = 0x8080_8080_8080_8080;

/// 8-byte group size — matches hashbrown's NEON group width.
const GROUP_BYTES: usize = 8;
/// Number of 8-byte groups per 64-slot bucket.
const GROUPS_PER_BUCKET: usize = RADIX_BUCKET_SLOTS / GROUP_BYTES;


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

        let slot_shift = 64 - bucket_bits - RADIX_SLOT_BITS;
        let fp_shift = slot_shift - 8;
        let bucket_mask = bucket_count.wrapping_sub(1);

        Ok(Self {
            config,
            layout,
            fingerprints: vec![FpBucket([RADIX_FP_EMPTY; RADIX_BUCKET_SLOTS]); bucket_count],
            hashes: vec![0; total_slots],
            len: 0,
            slot_shift,
            fp_shift,
            bucket_mask,
        })
    }

    /// Extract bucket, preferred_slot, and fingerprint from a raw id.
    pub fn hash_parts(&self, id: u64) -> HashParts {
        let h = mix64(id);

        let bucket = (h >> (64 - self.layout.bucket_bits)) as usize & self.bucket_mask;
        let preferred_slot = ((h >> self.slot_shift) & 0x3f) as usize;
        let raw_fp = ((h >> self.fp_shift) & 0xff) as u8;
        let fingerprint = raw_fp | ((raw_fp == 0) as u8);

        HashParts {
            bucket,
            preferred_slot,
            fingerprint,
        }
    }

    // -----------------------------------------------------------------
    // Bucket scanning
    // -----------------------------------------------------------------

    /// NEON: scan 64-byte bucket, returning (empty_mask, match_mask) as
    /// 64-bit bitmasks with one bit per slot.
    #[inline]
    #[cfg(target_arch = "aarch64")]
    fn scan_bucket_masks(&self, bucket: usize, target_fp: u8) -> (u64, u64) {
        let ptr = self.fingerprints[bucket].0.as_ptr();
        let mut empty_mask = 0u64;
        let mut match_mask = 0u64;

        // SAFETY: ptr points to a contiguous region of at least 64 bytes
        // (one full bucket). NEON loads are unaligned-safe on AArch64.
        // Bitmask extraction uses vreinterpret + vget_lane (matching
        // hashbrown's NEON pattern) instead of store-to-stack.
        unsafe {
            let zero = vdupq_n_u8(0);
            let target = vdupq_n_u8(target_fp);
            let mut chunk = 0usize;
            while chunk < 4 {
                let p = ptr.add(chunk * 16);
                let vals = vld1q_u8(p);

                let ecmp = vceqq_u8(vals, zero);
                let mcmp = vceqq_u8(vals, target);

                let e64 = vreinterpretq_u64_u8(ecmp);
                let m64 = vreinterpretq_u64_u8(mcmp);

                let e_lo = pack_msbs(vgetq_lane_u64::<0>(e64));
                let e_hi = pack_msbs(vgetq_lane_u64::<1>(e64));
                let m_lo = pack_msbs(vgetq_lane_u64::<0>(m64));
                let m_hi = pack_msbs(vgetq_lane_u64::<1>(m64));

                let e16 = (e_lo as u64) | ((e_hi as u64) << 8);
                let m16 = (m_lo as u64) | ((m_hi as u64) << 8);

                empty_mask |= e16 << (chunk * 16);
                match_mask |= m16 << (chunk * 16);
                chunk += 1;
            }
        }

        (empty_mask, match_mask)
    }

    /// Scalar fallback for non-AArch64 targets.
    #[inline]
    #[cfg(not(target_arch = "aarch64"))]
    fn scan_bucket_masks(&self, bucket: usize, target_fp: u8) -> (u64, u64) {
        let mut empty_mask = 0u64;
        let mut match_mask = 0u64;

        for i in 0..RADIX_BUCKET_SLOTS {
            let fp = self.fingerprints[bucket].0[i];
            if fp == RADIX_FP_EMPTY {
                empty_mask |= 1u64 << i;
            }
            if fp == target_fp {
                match_mask |= 1u64 << i;
            }
        }

        (empty_mask, match_mask)
    }

    // -----------------------------------------------------------------
    // contains variants — active version is wired in the IndexTable impl
    // -----------------------------------------------------------------

    /// v1: inline preferred-slot check at round 0, cold fallback using the
    /// original rotation-based multi-round probe. Kept for reference.
    #[inline]
    pub fn contains_v1(&self, id: u64) -> bool {
        let h = mix64(id);
        let bucket_bits = self.layout.bucket_bits;
        let slot_shift = 64 - bucket_bits - RADIX_SLOT_BITS;
        let fp_shift = slot_shift - 8;

        let bucket = if bucket_bits == 0 {
            0
        } else {
            (h >> (64 - bucket_bits)) as usize
        };
        let preferred_slot = ((h >> slot_shift) & 0x3f) as usize;
        let raw_fp = ((h >> fp_shift) & 0xff) as u8;
        let fp = if raw_fp == 0 { 1 } else { raw_fp };

        // SAFETY: bucket < bucket_count, preferred_slot < 64.
        unsafe {
            if *self.fingerprints.get_unchecked(bucket).0.get_unchecked(preferred_slot) == fp
                && *self.hashes.get_unchecked(bucket * RADIX_BUCKET_SLOTS + preferred_slot) == id
            {
                return true;
            }
        }

        self.contains_v1_slow(id, h)
    }

    #[cold]
    #[inline(never)]
    fn contains_v1_slow(&self, id: u64, mixed_hash: u64) -> bool {
        for round in 0..RADIX_MAX_PROBE_ROUNDS {
            let rotated = mixed_hash.rotate_left(round as u32);
            let bucket_bits = self.layout.bucket_bits;
            let slot_shift = 64 - bucket_bits - RADIX_SLOT_BITS;
            let fp_shift = slot_shift - 8;
            let bucket = if bucket_bits == 0 {
                0
            } else {
                (rotated >> (64 - bucket_bits)) as usize
            };
            let preferred_slot = ((rotated >> slot_shift) & 0x3f) as usize;
            let raw_fp = ((rotated >> fp_shift) & 0xff) as u8;
            let fp = if raw_fp == 0 { 1 } else { raw_fp };

            let (empty_mask, match_mask) = self.scan_bucket_masks(bucket, fp);
            let rot_match = match_mask.rotate_right(preferred_slot as u32);
            let rot_empty = empty_mask.rotate_right(preferred_slot as u32);
            let first_empty = if rot_empty == 0 {
                RADIX_BUCKET_SLOTS
            } else {
                rot_empty.trailing_zeros() as usize
            };

            let mut probe = if first_empty == 0 {
                0
            } else if first_empty == RADIX_BUCKET_SLOTS {
                rot_match
            } else {
                rot_match & ((1u64 << first_empty) - 1)
            };

            while probe != 0 {
                let offset = probe.trailing_zeros() as usize;
                let slot = (preferred_slot + offset) & (RADIX_BUCKET_SLOTS - 1);
                let idx = bucket * RADIX_BUCKET_SLOTS + slot;
                if self.hashes[idx] == id {
                    return true;
                }
                probe &= probe - 1;
            }
        }

        false
    }

    /// v2: linear bucket advance, early termination on empty, fixed
    /// preferred_slot/fp per key (only bucket changes on overflow).
    #[inline]
    pub fn contains_v2(&self, id: u64) -> bool {
        let h = mix64(id);
        let bucket_bits = self.layout.bucket_bits;
        let slot_shift = 64 - bucket_bits - RADIX_SLOT_BITS;
        let fp_shift = slot_shift - 8;

        let bucket = if bucket_bits == 0 {
            0
        } else {
            (h >> (64 - bucket_bits)) as usize
        };
        let preferred_slot = ((h >> slot_shift) & 0x3f) as usize;
        let raw_fp = ((h >> fp_shift) & 0xff) as u8;
        let fp = if raw_fp == 0 { 1 } else { raw_fp };

        // SAFETY: bucket < bucket_count, preferred_slot < 64.
        unsafe {
            if *self.fingerprints.get_unchecked(bucket).0.get_unchecked(preferred_slot) == fp
                && *self.hashes.get_unchecked(bucket * RADIX_BUCKET_SLOTS + preferred_slot) == id
            {
                return true;
            }
        }

        self.contains_v2_slow(id, bucket, fp)
    }

    /// Slow path: scan bucket for fp matches, confirm against full hash.
    /// Linear advance to next bucket on full bucket. Early exit on empty.
    #[cold]
    #[inline(never)]
    fn contains_v2_slow(&self, id: u64, start_bucket: usize, fp: u8) -> bool {
        let bucket_mask = self.layout.bucket_count.wrapping_sub(1);
        let mut bucket = start_bucket;

        for _ in 0..self.layout.bucket_count {
            let (empty_mask, match_mask) = self.scan_bucket_masks(bucket, fp);

            let mut probe = match_mask;
            while probe != 0 {
                let slot = probe.trailing_zeros() as usize;
                if self.hashes[bucket * RADIX_BUCKET_SLOTS + slot] == id {
                    return true;
                }
                probe &= probe - 1;
            }

            if empty_mask != 0 {
                return false;
            }

            bucket = (bucket + 1) & bucket_mask;
        }

        false
    }

    /// v3: single-function contains — preferred-slot probe per bucket,
    /// then progressive 8-byte group scan, linear bucket advance.
    /// All SIMD logic inlined for full visibility and iteration control.
    #[inline]
    pub fn contains_v3(&self, id: u64) -> bool {
        let h = mix64(id);

        let mut bucket = (h >> (64 - self.layout.bucket_bits)) as usize & self.bucket_mask;
        let preferred_slot = ((h >> self.slot_shift) & 0x3f) as usize;
        let raw_fp = ((h >> self.fp_shift) & 0xff) as u8;
        let fp = raw_fp | ((raw_fp == 0) as u8);

        let fp_ptr = self.fingerprints.as_ptr() as *const u8;
        let hash_ptr = self.hashes.as_ptr();
        let bucket_mask = self.bucket_mask;
        let start_group = preferred_slot / GROUP_BYTES;
        let mut remaining = self.layout.bucket_count;

        loop {
            let base = bucket * RADIX_BUCKET_SLOTS;

            // ── 1. Preferred-slot probe ────────────────────────────
            unsafe {
                let fp_at = *fp_ptr.add(base + preferred_slot);
                if fp_at == fp {
                    if *hash_ptr.add(base + preferred_slot) == id {
                        return true;
                    }
                } else if fp_at == 0 {
                    return false;
                }
            }

            // ── 2. Progressive slot scan (8 groups of 8 bytes) ────
            let mut sg_has_empty = false;

            unsafe {
                for gi in 0..GROUPS_PER_BUCKET as u8 {
                    let g = ((start_group as u8) + gi) & (GROUPS_PER_BUCKET as u8 - 1);
                    let gp = fp_ptr.add(base + (g as usize) * GROUP_BYTES);
                    let gh = hash_ptr.add(base + (g as usize) * GROUP_BYTES);

                    let mut group_has_empty = false;
                    for si in 0..GROUP_BYTES {
                        let b = *gp.add(si);
                        if b == fp {
                            if *gh.add(si) == id {
                                return true;
                            }
                        } else if b == 0 {
                            group_has_empty = true;
                        }
                    }

                    if group_has_empty {
                        if gi == 0 {
                            sg_has_empty = true;
                        } else {
                            return false;
                        }
                    }
                }
            }

            if sg_has_empty {
                return false;
            }

            // SIMD variant (kept for switching back):
            // unsafe {
            //     for gi in 0..GROUPS_PER_BUCKET as u8 {
            //         let g = ((start_group as u8) + gi) & (GROUPS_PER_BUCKET as u8 - 1);
            //         let gp = fp_ptr.add(base + (g as usize) * GROUP_BYTES);
            //         let gh = hash_ptr.add(base + (g as usize) * GROUP_BYTES);
            //         #[cfg(target_arch = "aarch64")]
            //         let (empty, matched) = {
            //             let vals = vld1_u8(gp);
            //             (
            //                 vget_lane_u64::<0>(vreinterpret_u64_u8(vceq_u8(vals, vdup_n_u8(0)))) & SPARSE_MSB,
            //                 vget_lane_u64::<0>(vreinterpret_u64_u8(vceq_u8(vals, vdup_n_u8(fp)))) & SPARSE_MSB,
            //             )
            //         };
            //         #[cfg(not(target_arch = "aarch64"))]
            //         let (empty, matched) = {
            //             let (mut e, mut m) = (0u64, 0u64);
            //             for i in 0..8usize {
            //                 let b = *gp.add(i);
            //                 if b == 0 { e |= 0x80u64 << (i * 8); }
            //                 if b == fp { m |= 0x80u64 << (i * 8); }
            //             }
            //             (e, m)
            //         };
            //         let mut m = matched;
            //         while m != 0 {
            //             let byte_idx = (m.trailing_zeros() >> 3) as usize;
            //             if *gh.add(byte_idx) == id { return true; }
            //             m &= m - 1;
            //         }
            //         if empty != 0 {
            //             if gi == 0 { sg_has_empty = true; }
            //             else { return false; }
            //         }
            //     }
            // }
            // if sg_has_empty { return false; }

            // ── 3. Bucket full — advance ──────────────────────────
            bucket = (bucket + 1) & bucket_mask;
            remaining -= 1;
            if remaining == 0 {
                return false;
            }
        }
    }
}

impl IndexTable for RadixTree {
    /// Insert with linear bucket advance (matches contains_v2 probe sequence).
    #[inline]
    fn insert(&mut self, id: u64) -> bool {
        if self.len == self.layout.total_slots {
            return false;
        }

        let h = mix64(id);

        let start_bucket = (h >> (64 - self.layout.bucket_bits)) as usize & self.bucket_mask;
        let preferred_slot = ((h >> self.slot_shift) & 0x3f) as usize;
        let raw_fp = ((h >> self.fp_shift) & 0xff) as u8;
        let fp = raw_fp | ((raw_fp == 0) as u8);

        let mut bucket = start_bucket;

        for _ in 0..self.layout.bucket_count {
            let (empty_mask, match_mask) = self.scan_bucket_masks(bucket, fp);

            // Dedup: check all fp-matching slots in this bucket.
            let mut probe = match_mask;
            while probe != 0 {
                let slot = probe.trailing_zeros() as usize;
                if self.hashes[bucket * RADIX_BUCKET_SLOTS + slot] == id {
                    return false;
                }
                probe &= probe - 1;
            }

            // Place at nearest empty slot after preferred_slot (wrapping).
            if empty_mask != 0 {
                let rot_empty = empty_mask.rotate_right(preferred_slot as u32);
                let offset = rot_empty.trailing_zeros() as usize;
                let slot = (preferred_slot + offset) & (RADIX_BUCKET_SLOTS - 1);
                self.fingerprints[bucket].0[slot] = fp;
                let idx = bucket * RADIX_BUCKET_SLOTS + slot;
                self.hashes[idx] = id;
                self.len += 1;
                return true;
            }

            bucket = (bucket + 1) & self.bucket_mask;
        }

        false
    }

    /// Active contains — delegates to the current best version.
    #[inline]
    fn contains(&self, id: u64) -> bool {
        self.contains_v3(id)
    }

    #[inline]
    fn get(&self, _id: u64) -> Option<u64> {
        // Placeholder: full key-value retrieval deferred to later milestone.
        todo!("RadixTree::get not yet implemented")
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.layout.total_slots
    }
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
    fn insert_contains_and_duplicate_behavior() {
        let mut t = RadixTree::new(RadixTreeConfig { capacity_bits: 10 }).expect("build");
        assert!(t.insert(42));
        assert!(t.contains(42));
        assert!(!t.insert(42));
        assert_eq!(t.len(), 1);
        assert!(!t.contains(99_999));
    }

    #[test]
    fn fingerprints_start_empty() {
        let t = RadixTree::new(RadixTreeConfig { capacity_bits: 10 }).expect("build");
        assert!(t.fingerprints.iter().all(|b| b.0.iter().all(|&fp| fp == RADIX_FP_EMPTY)));
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

    #[test]
    fn all_inserted_keys_are_found() {
        let capacity_bits = 14;
        let mut t = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let n = ((t.capacity() as f64) * 0.90) as usize;
        let keys: Vec<u64> = (0..n)
            .map(|i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .collect();

        for &k in &keys {
            assert!(t.insert(k));
        }
        for &k in &keys {
            assert!(t.contains(k), "key {} not found after insert", k);
        }
    }

    #[test]
    fn lookup_miss_returns_false() {
        let capacity_bits = 14;
        let mut t = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let n = ((t.capacity() as f64) * 0.75) as usize;
        for i in 0..n {
            t.insert((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        }
        for i in 0..1000u64 {
            let absent = i.wrapping_mul(0xdead_beef_cafe_babe).wrapping_add(0xffff);
            assert!(!t.contains(absent), "false positive for absent key {}", absent);
        }
    }
}
