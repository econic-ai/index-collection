/// Radix tree with 4 × 64-slot groups per bucket.
/// Each group is cache-line-aligned (64 bytes).
pub const GROUP_SLOTS: usize = 64;
pub const SLOT_BITS: u8 = 6;
pub const GROUPS_PER_BUCKET: usize = 4;
pub const GROUP_BITS: u8 = 2;
pub const BUCKET_SLOTS: usize = GROUPS_PER_BUCKET * GROUP_SLOTS;
pub const RADIX_FP_EMPTY: u8 = 0x00;
pub const RADIX_MAX_PROBE_ROUNDS: u8 = 64;

pub mod analysis;

use crate::{IndexTable, mix64};
use core::arch::aarch64::*;
#[cfg(feature = "gpu32")]
use core::ffi::c_void;
#[cfg(feature = "gpu32")]
use std::cell::RefCell;
#[cfg(feature = "gpu32")]
use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};

/// Hint the CPU to prefetch a cache line for reading.
#[inline(always)]
unsafe fn prefetch_read(addr: *const u8) {
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{addr}]",
            addr = in(reg) addr,
            options(nostack, preserves_flags),
        );
    }
}

/// 64-byte fingerprint group, cache-line-aligned.
#[derive(Debug, Clone)]
#[repr(C, align(64))]
pub struct FpGroup(pub [u8; GROUP_SLOTS]);

/// Bucket containing 4 cache-line-aligned fingerprint groups (256 slots).
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Bucket {
    pub groups: [FpGroup; GROUPS_PER_BUCKET],
}

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
    pub group_slots: usize,
    pub slot_bits: u8,
    pub group_bits: u8,
}

pub struct RadixTree {
    pub config: RadixTreeConfig,
    pub layout: RadixTreeLayout,
    /// Fingerprint arena: buckets of 4 cache-line-aligned groups.
    pub fingerprints: Vec<Bucket>,
    /// Full-key confirmation arena (flat, indexed by bucket*256 + group*64 + slot).
    pub hashes: Vec<u64>,
    pub len: usize,
    /// Precomputed for the hot path — avoids serial dependency chain.
    bucket_shift: u8,
    group_slot_shift: u8,
    fp_shift: u8,
    bucket_mask: usize,
    /// Cached heap pointers — stable because Vecs never resize.
    fp_ptr: *const u8,
    hash_ptr: *const u64,
}

// Raw pointers are not Send/Sync by default, but the heap allocations they
// reference are owned by the Vecs in the same struct and never reallocated.
unsafe impl Send for RadixTree {}
unsafe impl Sync for RadixTree {}

impl Clone for RadixTree {
    fn clone(&self) -> Self {
        let fingerprints = self.fingerprints.clone();
        let hashes = self.hashes.clone();
        let fp_ptr = fingerprints.as_ptr() as *const u8;
        let hash_ptr = hashes.as_ptr();
        Self {
            config: self.config,
            layout: self.layout,
            fingerprints,
            hashes,
            len: self.len,
            bucket_shift: self.bucket_shift,
            group_slot_shift: self.group_slot_shift,
            fp_shift: self.fp_shift,
            bucket_mask: self.bucket_mask,
            fp_ptr,
            hash_ptr,
        }
    }
}

impl std::fmt::Debug for RadixTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RadixTree")
            .field("config", &self.config)
            .field("layout", &self.layout)
            .field("len", &self.len)
            .field("bucket_mask", &self.bucket_mask)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadixTreeBuildError {
    CapacityBitsTooSmall(u8),
    CapacityBitsTooLarge(u8),
}

/// Pack the high bit of each byte in a u64 into the low 8 bits.
/// Input: u64 where each byte is 0x00 or 0xFF (NEON comparison result).
/// Output: u8 bitmask with one bit per byte.
#[inline]
fn pack_msbs(v: u64) -> u8 {
    const HI: u64 = 0x8080_8080_8080_8080;
    const MUL: u64 = 0x0002_0408_1020_4081;
    ((v & HI).wrapping_mul(MUL) >> 56) as u8
}

/// Pack a 128-bit NEON byte comparison into a 16-bit mask.
/// Bit i is set when byte i of `cmp` is 0xFF.
#[inline]
unsafe fn pack_neon_16(cmp: uint8x16_t) -> u16 {
    unsafe {
        let u64s = vreinterpretq_u64_u8(cmp);
        let lo = pack_msbs(vgetq_lane_u64::<0>(u64s));
        let hi = pack_msbs(vgetq_lane_u64::<1>(u64s));
        (lo as u16) | ((hi as u16) << 8)
    }
}

/// 16-byte NEON chunk width within a 64-slot group.
const CHUNK_WIDTH: usize = 16;
/// Number of 16-byte chunks per group.
const CHUNKS_PER_GROUP: usize = GROUP_SLOTS / CHUNK_WIDTH;

#[cfg(not(feature = "gpu32"))]
type IterMask = u16;

#[cfg(not(feature = "gpu32"))]
const ITER_CHUNK_WIDTH: usize = 16;

#[cfg(feature = "gpu32")]
struct GpuScanContext {
    device: Device,
    pipeline: ComputePipelineState,
    queue: metal::CommandQueue,
}

#[cfg(feature = "gpu32")]
impl GpuScanContext {
    fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        let src = r#"
#include <metal_stdlib>
using namespace metal;

kernel void mark_occupied(
    const device uchar* fps [[buffer(0)]],
    device uchar* flags [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    flags[gid] = fps[gid] != 0;
}
"#;
        let lib = device.new_library_with_source(src, &opts).ok()?;
        let func = lib.get_function("mark_occupied", None).ok()?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func).ok()?;
        Some(Self {
            device,
            pipeline,
            queue,
        })
    }
}

#[cfg(feature = "gpu32")]
thread_local! {
    static GPU_SCAN_CONTEXT: RefCell<Option<GpuScanContext>> = const { RefCell::new(None) };
}

impl RadixTree {
    pub fn new(config: RadixTreeConfig) -> Result<Self, RadixTreeBuildError> {
        let min_bits = GROUP_BITS + SLOT_BITS;
        if config.capacity_bits < min_bits {
            return Err(RadixTreeBuildError::CapacityBitsTooSmall(
                config.capacity_bits,
            ));
        }

        if config.capacity_bits > 56 {
            return Err(RadixTreeBuildError::CapacityBitsTooLarge(
                config.capacity_bits,
            ));
        }

        let bucket_bits = config.capacity_bits - GROUP_BITS - SLOT_BITS;
        let total_slots = 1usize << config.capacity_bits;
        let bucket_count = 1usize << bucket_bits;
        let layout = RadixTreeLayout {
            capacity_bits: config.capacity_bits,
            total_slots,
            bucket_bits,
            bucket_count,
            bucket_slots: BUCKET_SLOTS,
            group_slots: GROUP_SLOTS,
            slot_bits: SLOT_BITS,
            group_bits: GROUP_BITS,
        };

        // Hash layout (left to right):
        //   [bucket_bits | group_bits(2) | slot_bits(6) | fp_bits(8) | ...]
        let bucket_shift = if bucket_bits > 0 { 64 - bucket_bits } else { 0 };
        let group_slot_shift = 64 - bucket_bits - GROUP_BITS - SLOT_BITS;
        let fp_shift = group_slot_shift - 8;
        let bucket_mask = bucket_count.wrapping_sub(1);

        let empty_bucket = Bucket {
            groups: [
                FpGroup([RADIX_FP_EMPTY; GROUP_SLOTS]),
                FpGroup([RADIX_FP_EMPTY; GROUP_SLOTS]),
                FpGroup([RADIX_FP_EMPTY; GROUP_SLOTS]),
                FpGroup([RADIX_FP_EMPTY; GROUP_SLOTS]),
            ],
        };

        let fingerprints = vec![empty_bucket; bucket_count];
        let hashes = vec![0u64; total_slots];
        let fp_ptr = fingerprints.as_ptr() as *const u8;
        let hash_ptr = hashes.as_ptr();

        Ok(Self {
            config,
            layout,
            fingerprints,
            hashes,
            len: 0,
            bucket_shift,
            group_slot_shift,
            fp_shift,
            bucket_mask,
            fp_ptr,
            hash_ptr,
        })
    }

}

#[cfg(feature = "gpu32")]
impl RadixTree {
    fn collect_iter_keys_gpu(&self) -> Vec<u64> {
        let total = self.layout.total_slots;
        if total == 0 {
            return Vec::new();
        }

        let mut occupied = vec![0u8; total];

        GPU_SCAN_CONTEXT.with(|ctx_cell| {
            if ctx_cell.borrow().is_none() {
                *ctx_cell.borrow_mut() = GpuScanContext::new();
            }

            let ctx_borrow = ctx_cell.borrow();
            let Some(ctx) = ctx_borrow.as_ref() else {
                panic!("gpu32 enabled but no Metal device/pipeline available");
            };

            let fp_buf = ctx.device.new_buffer_with_data(
                self.fp_ptr.cast::<c_void>(),
                total as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let flag_buf = ctx
                .device
                .new_buffer(total as u64, MTLResourceOptions::StorageModeShared);

            let cmd = ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.pipeline);
            enc.set_buffer(0, Some(&fp_buf), 0);
            enc.set_buffer(1, Some(&flag_buf), 0);

            let grid = MTLSize {
                width: total as u64,
                height: 1,
                depth: 1,
            };
            let tg_width = ctx.pipeline.thread_execution_width().max(1) as u64;
            let threadgroup = MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, threadgroup);
            enc.end_encoding();

            cmd.commit();
            cmd.wait_until_completed();

            unsafe {
                let src = std::slice::from_raw_parts(flag_buf.contents().cast::<u8>(), total);
                occupied.copy_from_slice(src);
            }
        });

        let mut keys = Vec::with_capacity(self.len);
        for (slot, &is_set) in occupied.iter().enumerate() {
            if is_set != 0 {
                keys.push(unsafe { *self.hash_ptr.add(slot) });
            }
        }
        keys
    }
}

// ── Iteration ──────────────────────────────────────────────────────

/// NEON-accelerated iterator over stored keys.
/// Loads 16 fingerprint bytes at a time, compares against zero to build
/// an occupied bitmask, then iterates set bits to yield hashes.
#[cfg(not(feature = "gpu32"))]
pub struct Iter<'a> {
    fp_ptr: *const u8,
    hash_ptr: *const u64,
    /// Absolute slot position of the current 16-byte chunk start.
    chunk_base: usize,
    /// Remaining occupied-bits mask for the current iterator chunk.
    mask: IterMask,
    total: usize,
    _marker: std::marker::PhantomData<&'a ()>,
}

#[cfg(not(feature = "gpu32"))]
impl<'a> Iter<'a> {
    /// Advance to the next iterator chunk that has at least one occupied slot.
    #[inline]
    fn advance_chunk(&mut self) -> bool {
        while self.chunk_base < self.total {
            #[cfg(feature = "gpu32")]
            let occupied = unsafe {
                let ptr = self.fp_ptr.add(self.chunk_base);
                let zero = vdupq_n_u8(0);
                let lo = vld1q_u8(ptr);
                let hi = vld1q_u8(ptr.add(16));
                let lo_empty = pack_neon_16(vceqq_u8(lo, zero)) as u32;
                let hi_empty = pack_neon_16(vceqq_u8(hi, zero)) as u32;
                let empty = lo_empty | (hi_empty << 16);
                !empty
            };            
            #[cfg(not(feature = "gpu32"))]
            let occupied = unsafe {
                // Keep the default 16B path identical to the original hot loop.
                let ptr = self.fp_ptr.add(self.chunk_base);
                let chunk = vld1q_u8(ptr);
                let zero = vdupq_n_u8(0);
                let eq = vceqq_u8(chunk, zero);
                let empty_mask = pack_neon_16(eq);
                !empty_mask & 0xFFFF
            };

            if occupied != 0 {
                self.mask = occupied;
                return true;
            }
            self.chunk_base += ITER_CHUNK_WIDTH;
        }
        false
    }
}

#[cfg(not(feature = "gpu32"))]
impl<'a> Iterator for Iter<'a> {
    type Item = u64;
    #[inline]
    fn next(&mut self) -> Option<u64> {
        loop {
            if self.mask != 0 {
                let bit = self.mask.trailing_zeros() as usize;
                self.mask &= self.mask - 1;
                let slot = self.chunk_base + bit;
                return Some(unsafe { *self.hash_ptr.add(slot) });
            }
            self.chunk_base += ITER_CHUNK_WIDTH;
            if !self.advance_chunk() {
                return None;
            }
        }
    }
}

#[cfg(feature = "gpu32")]
pub struct Iter<'a> {
    inner: std::vec::IntoIter<u64>,
    _marker: std::marker::PhantomData<&'a ()>,
}

#[cfg(feature = "gpu32")]
impl<'a> Iterator for Iter<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<u64> {
        self.inner.next()
    }
}

impl IndexTable for RadixTree {
    type KeyIter<'a> = Iter<'a>;

    /// Scalar probe at preferred_offset with bulk hash-line prefetch for
    /// the NEON fallback. The preferred_offset bit is masked out of the
    /// NEON match scan to avoid duplicate work.
    #[inline]
    fn insert(&mut self, id: u64) -> bool {
        if self.len == self.layout.total_slots {
            return false;
        }

        let h = mix64(id);

        let start_bucket = (h >> self.bucket_shift) as usize & self.bucket_mask;
        let group_slot = ((h >> self.group_slot_shift) & 0xFF) as u8;
        let group = (group_slot >> 6) as usize;
        let preferred_slot = (group_slot & 0x3F) as usize;
        let raw_fp = ((h >> self.fp_shift) & 0xFF) as u8;
        let fp = raw_fp | ((raw_fp == 0) as u8);

        let start_chunk = preferred_slot >> 4;
        let preferred_offset = preferred_slot & 0xF;
        let pref_bit = 1u16 << preferred_offset;
        let mut bucket = start_bucket;
        let hash_ptr = self.hash_ptr;

        for _ in 0..self.layout.bucket_count {
            let group_ptr = self.fingerprints[bucket].groups[group].0.as_ptr();
            let hash_base = bucket * BUCKET_SLOTS + group * GROUP_SLOTS;

            unsafe {
                for ci in 0..CHUNKS_PER_GROUP {
                    let c = (start_chunk + ci) & (CHUNKS_PER_GROUP - 1);
                    let chunk_offset = c * CHUNK_WIDTH;
                    let pref_addr = chunk_offset + preferred_offset;

                    // Prefetch both hash cache lines for this chunk (16 hashes = 128 bytes)
                    prefetch_read(hash_ptr.add(hash_base + chunk_offset) as *const u8);
                    prefetch_read(hash_ptr.add(hash_base + chunk_offset + 8) as *const u8);

                    // Scalar check at preferred offset
                    let pref_byte = *group_ptr.add(pref_addr);
                    if pref_byte == fp {
                        if self.hashes[hash_base + pref_addr] == id {
                            return false;
                        }
                    } else if pref_byte == 0 {
                        self.fingerprints[bucket].groups[group].0[pref_addr] = fp;
                        self.hashes[hash_base + pref_addr] = id;
                        self.len += 1;
                        return true;
                    }

                    // NEON scan — preferred_offset excluded via pref_bit
                    let zero = vdupq_n_u8(0);
                    let target = vdupq_n_u8(fp);
                    let vals = vld1q_u8(group_ptr.add(chunk_offset));
                    let match_mask = pack_neon_16(vceqq_u8(vals, target)) & !pref_bit;
                    let empty_mask = pack_neon_16(vceqq_u8(vals, zero));

                    let mut m = match_mask;
                    while m != 0 {
                        let pos = m.trailing_zeros() as usize;
                        if self.hashes[hash_base + chunk_offset + pos] == id {
                            return false;
                        }
                        m &= m - 1;
                    }

                    // Place at first reachable empty (displacement order)
                    if empty_mask != 0 {
                        let shifted = empty_mask.rotate_right(
                            ((preferred_offset + 1) & 0xF) as u32,
                        );
                        let dist = shifted.trailing_zeros() as usize;
                        let pos = (preferred_offset + 1 + dist) & 0xF;
                        let slot = chunk_offset + pos;
                        self.fingerprints[bucket].groups[group].0[slot] = fp;
                        self.hashes[hash_base + slot] = id;
                        self.len += 1;
                        return true;
                    }
                }
            }

            // Group full — same group, next bucket.
            bucket = (bucket + 1) & self.bucket_mask;
        }

        false
    }

    /// Scalar probe at preferred_offset with bulk hash-line prefetch for
    /// the NEON fallback. The preferred_offset bit is masked out of the
    /// NEON match scan to avoid duplicate work.
    #[inline]
    fn contains(&self, id: u64) -> bool {
        let h = mix64(id);

        let mut bucket = (h >> self.bucket_shift) as usize & self.bucket_mask;
        let group_slot = ((h >> self.group_slot_shift) & 0xFF) as u8;
        let group = (group_slot >> 6) as usize;
        let preferred_slot = (group_slot & 0x3F) as usize;
        let raw_fp = ((h >> self.fp_shift) & 0xFF) as u8;
        let fp = raw_fp | ((raw_fp == 0) as u8);

        let fp_ptr = self.fp_ptr;
        let hash_ptr = self.hash_ptr;
        let group_offset = group * GROUP_SLOTS;
        let start_chunk = preferred_slot >> 4;
        let preferred_offset = preferred_slot & 0xF;
        let pref_bit = 1u16 << preferred_offset;

        unsafe {
            let bucket_mask = self.bucket_mask;
            let mut remaining = self.layout.bucket_count;

            loop {
                let group_base = bucket * BUCKET_SLOTS + group_offset;
                let zero = vdupq_n_u8(0);
                let target = vdupq_n_u8(fp);

                for ci in 0..CHUNKS_PER_GROUP {
                    let c = (start_chunk + ci) & (CHUNKS_PER_GROUP - 1);
                    let chunk_base = group_base + c * CHUNK_WIDTH;

                    // Prefetch both hash cache lines for this chunk (16 hashes = 128 bytes)
                    prefetch_read(hash_ptr.add(chunk_base) as *const u8);
                    prefetch_read(hash_ptr.add(chunk_base + 8) as *const u8);

                    // Scalar check at preferred offset
                    let pref_byte = *fp_ptr.add(chunk_base + preferred_offset);
                    if pref_byte == fp {
                        if *hash_ptr.add(chunk_base + preferred_offset) == id {
                            return true;
                        }
                    } else if pref_byte == 0 {
                        return false;
                    }

                    // NEON scan — preferred_offset excluded via pref_bit
                    let vals = vld1q_u8(fp_ptr.add(chunk_base));
                    let match_mask = pack_neon_16(vceqq_u8(vals, target)) & !pref_bit;
                    let empty_mask = pack_neon_16(vceqq_u8(vals, zero));

                    let mut m = match_mask;
                    while m != 0 {
                        let pos = m.trailing_zeros() as usize;
                        if *hash_ptr.add(chunk_base + pos) == id {
                            return true;
                        }
                        m &= m - 1;
                    }

                    if empty_mask != 0 {
                        return false;
                    }
                }

                bucket = (bucket + 1) & bucket_mask;
                remaining -= 1;
                if remaining == 0 {
                    return false;
                }
            }
        }
    }

    fn contains_probable(&self, _id: u64) -> bool {
        todo!("RadixTree::contains_probable — fingerprint-only check without hash confirmation")
    }

    #[inline]
    fn get(&self, _id: u64) -> Option<u64> {
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

    fn iter(&self) -> Self::KeyIter<'_> {
        #[cfg(feature = "gpu32")]
        {
            let keys = self.collect_iter_keys_gpu();
            return Iter {
                inner: keys.into_iter(),
                _marker: std::marker::PhantomData,
            };
        }
        #[cfg(not(feature = "gpu32"))]
        {
        let mut it = Iter {
            fp_ptr: self.fp_ptr,
            hash_ptr: self.hash_ptr,
            chunk_base: 0,
            mask: 0,
            total: self.layout.total_slots,
            _marker: std::marker::PhantomData,
        };
        it.advance_chunk();
        it
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BUCKET_SLOTS, GROUP_SLOTS, RADIX_FP_EMPTY, RADIX_MAX_PROBE_ROUNDS, RadixTree,
        RadixTreeBuildError, RadixTreeConfig,
    };
    use crate::IndexTable;

    #[test]
    fn rejects_capacity_bits_below_minimum() {
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
        assert_eq!(t.layout.bucket_bits, 10);
        assert_eq!(t.layout.bucket_count, 1 << 10);
        assert_eq!(t.layout.bucket_slots, BUCKET_SLOTS);
        assert_eq!(t.layout.group_slots, GROUP_SLOTS);
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
        assert!(t.fingerprints.iter().all(|b| {
            b.groups
                .iter()
                .all(|g| g.0.iter().all(|&fp| fp == RADIX_FP_EMPTY))
        }));
    }

    #[test]
    fn can_fill_to_high_load() {
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

    #[test]
    fn iter_yields_all_inserted_keys() {
        let mut t = RadixTree::new(RadixTreeConfig { capacity_bits: 14 }).expect("build");
        let n = (t.layout.total_slots as f64 * 0.50) as usize;
        let mut expected: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let key = crate::mix64(i as u64 + 1);
            t.insert(key);
            expected.push(key);
        }
        assert_eq!(t.len(), n);

        let mut got: Vec<u64> = t.iter().collect();
        assert_eq!(got.len(), n);

        expected.sort();
        got.sort();
        assert_eq!(got, expected);
    }
}
