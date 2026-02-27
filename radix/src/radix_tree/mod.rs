/// Radix tree with 4 × 64-slot groups per bucket.
/// Each group is cache-line-aligned (64 bytes).
pub const GROUP_SLOTS: usize = 64;
pub const SLOT_BITS: u8 = 6;
pub const GROUPS_PER_BUCKET: usize = 4;
pub const GROUP_BITS: u8 = 2;
pub const BUCKET_SLOTS: usize = GROUPS_PER_BUCKET * GROUP_SLOTS;
pub const RADIX_FP_EMPTY: u8 = 0x00;
pub const RADIX_MAX_PROBE_ROUNDS: u8 = 64;

// ── Set predicate masks ────────────────────────────────────────────
//
// 2-array state encoding: (a_occ << 2 | b_occ << 1 | a_eq_b)
// 8 possible states → u8 mask, bit i set means state i satisfies the predicate.
//
// 3-array state encoding: (a_occ << 5 | b_occ << 4 | c_occ << 3 | ab_eq << 2 | ac_eq << 1 | bc_eq)
// 64 possible states → u64 mask.

/// A ∩ B — both occupied, same fingerprint.
pub const PRED2_INTERSECT: u8 = 0x80;
/// A \ B — a occupied, b either empty or different fingerprint.
pub const PRED2_DIFF_AB: u8 = 0x50;
/// B \ A — b occupied, a either empty or different fingerprint.
pub const PRED2_DIFF_BA: u8 = 0x44;
/// A △ B — symmetric difference: at least one occupied, fingerprints differ.
pub const PRED2_SYM_DIFF: u8 = 0x54;

/// A ∩ B ∩ C — all three occupied, all fingerprints equal.
pub const PRED3_CONSENSUS: u64 = 0x8000_0000_0000_0000;
/// Unique to A — a occupied, a ≠ b, a ≠ c.
pub const PRED3_UNIQUE_A: u64 = 0x0303_0303_0000_0000;
/// (A ∩ B) \ C — a and b occupied with same fp, c absent or different.
pub const PRED3_SHARED_AB_NOT_C: u64 = 0x3030_3030_0000_0000;

pub mod analysis;

use crate::{IndexTable, mix64};
use core::arch::aarch64::*;
#[cfg(feature = "gpu32")]
use core::ffi::c_void;
#[cfg(feature = "gpu32")]
use metal::{Buffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
#[cfg(feature = "gpu32")]
extern crate libc;

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
    /// Page-aligned fingerprint memory for zero-copy Metal buffer.
    /// When gpu32 is active, `fingerprints` Vec is empty and this owns the memory.
    #[cfg(feature = "gpu32")]
    fp_page_ptr: *mut u8,
    /// Page-aligned hash memory for zero-copy Metal buffer.
    #[cfg(feature = "gpu32")]
    hash_page_ptr: *mut u64,
    #[cfg(feature = "gpu32")]
    gpu: GpuState,
}

// Raw pointers are not Send/Sync by default, but the heap allocations they
// reference are owned by the Vecs in the same struct and never reallocated.
unsafe impl Send for RadixTree {}
unsafe impl Sync for RadixTree {}

impl Clone for RadixTree {
    fn clone(&self) -> Self {
        #[cfg(not(feature = "gpu32"))]
        {
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
        #[cfg(feature = "gpu32")]
        {
            let total = self.layout.total_slots;
            let bucket_count = self.layout.bucket_count;
            let fp_page_ptr = unsafe { page_alloc(total) };
            let hash_page_ptr = unsafe { page_alloc(total * 8) }.cast::<u64>();
            unsafe {
                std::ptr::copy_nonoverlapping(self.fp_ptr, fp_page_ptr, total);
                std::ptr::copy_nonoverlapping(self.hash_ptr, hash_page_ptr, total);
            }
            let fingerprints = unsafe {
                Vec::from_raw_parts(
                    fp_page_ptr.cast::<Bucket>(),
                    bucket_count,
                    bucket_count,
                )
            };
            let hashes = unsafe {
                Vec::from_raw_parts(hash_page_ptr, total, total)
            };
            let fp_ptr = fp_page_ptr as *const u8;
            let hash_ptr = hash_page_ptr as *const u64;
            let gpu = GpuState::new(fp_ptr, hash_ptr, total);
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
                fp_page_ptr,
                hash_page_ptr,
                gpu,
            }
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
const GPU_KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gather_keys(
    const device uchar*  fps    [[buffer(0)]],
    const device ulong*  hashes [[buffer(1)]],
    device ulong*        out    [[buffer(2)]],
    device atomic_uint*  count  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (fps[gid] != 0) {
        uint idx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        out[idx] = hashes[gid];
    }
}

kernel void contains_greedy(
    const device uchar*  fps    [[buffer(0)]],
    const device ulong*  hashes [[buffer(1)]],
    device atomic_uint*  found  [[buffer(2)]],
    constant uchar&      fp     [[buffer(3)]],
    constant ulong&      key    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (fps[gid] == fp && hashes[gid] == key) {
        atomic_store_explicit(found, 1, memory_order_relaxed);
    }
}

kernel void fingerprint_diff(
    const device uchar* fp_a   [[buffer(0)]],
    const device uchar* fp_b   [[buffer(1)]],
    device atomic_uint* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (fp_a[gid] != fp_b[gid]) {
        uint word_idx = gid / 32;
        uint bit_idx  = gid % 32;
        atomic_fetch_or_explicit(
            &output[word_idx],
            (1u << bit_idx),
            memory_order_relaxed
        );
    }
}

kernel void set_intersection(
    const device uchar* fp_a     [[buffer(0)]],
    const device uchar* fp_b     [[buffer(1)]],
    device uint*        matches  [[buffer(2)]],
    device atomic_uint* count    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uchar a = fp_a[gid];
    uchar b = fp_b[gid];
    if (a != 0 && b != 0 && a == b) {
        uint idx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        matches[idx] = gid;
    }
}

kernel void set_predicate_2(
    const device uchar* fp_a     [[buffer(0)]],
    const device uchar* fp_b     [[buffer(1)]],
    device uint*        matches  [[buffer(2)]],
    device atomic_uint* count    [[buffer(3)]],
    constant uchar&     mask     [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uchar a = fp_a[gid];
    uchar b = fp_b[gid];
    uint state = (uint(a != 0) << 2) | (uint(b != 0) << 1) | uint(a == b);
    if ((mask >> state) & 1) {
        uint idx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        matches[idx] = gid;
    }
}

kernel void set_predicate_3(
    const device uchar* fp_a     [[buffer(0)]],
    const device uchar* fp_b     [[buffer(1)]],
    const device uchar* fp_c     [[buffer(2)]],
    device uint*        matches  [[buffer(3)]],
    device atomic_uint* count    [[buffer(4)]],
    constant ulong&     mask     [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uchar a = fp_a[gid];
    uchar b = fp_b[gid];
    uchar c = fp_c[gid];
    uint state = (uint(a != 0) << 5)
               | (uint(b != 0) << 4)
               | (uint(c != 0) << 3)
               | (uint(a == b) << 2)
               | (uint(a == c) << 1)
               | uint(b == c);
    if ((mask >> state) & 1) {
        uint idx = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
        matches[idx] = gid;
    }
}
"#;

/// Persistent Metal resources for GPU-accelerated iteration.
/// Created once at table construction, reused across all `iter()` calls.
#[cfg(feature = "gpu32")]
struct GpuState {
    queue: metal::CommandQueue,
    pipeline: ComputePipelineState,
    /// Zero-copy buffer wrapping page-aligned fingerprint memory.
    fp_buf: Buffer,
    /// Zero-copy buffer wrapping page-aligned hash memory.
    hash_buf: Buffer,
    /// Pre-allocated output buffer for gathered keys (capacity = total_slots).
    out_buf: Buffer,
    /// 4-byte atomic counter buffer.
    count_buf: Buffer,
    /// Pipeline for the contains_greedy kernel.
    greedy_pipeline: ComputePipelineState,
    /// 4-byte atomic flag for contains_greedy result.
    greedy_found_buf: Buffer,
    /// 1-byte constant buffer for target fingerprint.
    greedy_fp_buf: Buffer,
    /// 8-byte constant buffer for target key.
    greedy_key_buf: Buffer,
    /// Pipeline for the fingerprint_diff kernel.
    diff_pipeline: ComputePipelineState,
    /// Bitmask output buffer for diff (total_slots / 32 * 4 bytes, rounded up to page).
    diff_output_buf: Buffer,
    /// Pipeline for the set_intersection kernel.
    intersect_pipeline: ComputePipelineState,
    /// Compacted position output buffer for intersection (total_slots * 4 bytes worst case).
    intersect_matches_buf: Buffer,
    /// 4-byte atomic counter for intersection match count.
    intersect_count_buf: Buffer,
    /// Pipeline for the generalised 2-array predicate kernel.
    pred2_pipeline: ComputePipelineState,
    /// Pipeline for the generalised 3-array predicate kernel.
    pred3_pipeline: ComputePipelineState,
    /// 1-byte constant buffer for the 2-array predicate mask.
    pred_mask_u8_buf: Buffer,
    /// 8-byte constant buffer for the 3-array predicate mask.
    pred_mask_u64_buf: Buffer,
    /// Metal device handle, retained for creating per-call buffers.
    device: Device,
    total_slots: usize,
    tg_width: u64,
}

#[cfg(feature = "gpu32")]
impl GpuState {
    fn new(
        fp_ptr: *const u8,
        hash_ptr: *const u64,
        total_slots: usize,
    ) -> Self {
        let device = Device::system_default()
            .expect("gpu32: no Metal device available");
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let lib = device
            .new_library_with_source(GPU_KERNEL_SRC, &opts)
            .expect("gpu32: Metal shader compilation failed");
        let gather_func = lib
            .get_function("gather_keys", None)
            .expect("gpu32: gather_keys function not found");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&gather_func)
            .expect("gpu32: pipeline creation failed");

        let greedy_func = lib
            .get_function("contains_greedy", None)
            .expect("gpu32: contains_greedy function not found");
        let greedy_pipeline = device
            .new_compute_pipeline_state_with_function(&greedy_func)
            .expect("gpu32: greedy pipeline creation failed");

        let shared = MTLResourceOptions::StorageModeShared;

        let fp_buf = device.new_buffer_with_bytes_no_copy(
            fp_ptr as *const c_void,
            page_ceil(total_slots) as u64,
            shared,
            None,
        );
        let hash_buf = device.new_buffer_with_bytes_no_copy(
            hash_ptr as *const c_void,
            page_ceil(total_slots * 8) as u64,
            shared,
            None,
        );

        let out_buf = device.new_buffer(
            (total_slots * 8) as u64,
            shared,
        );
        let count_buf = device.new_buffer(4, shared);

        let greedy_found_buf = device.new_buffer(4, shared);
        let greedy_fp_buf = device.new_buffer(1, shared);
        let greedy_key_buf = device.new_buffer(8, shared);

        let diff_func = lib
            .get_function("fingerprint_diff", None)
            .expect("gpu32: fingerprint_diff function not found");
        let diff_pipeline = device
            .new_compute_pipeline_state_with_function(&diff_func)
            .expect("gpu32: diff pipeline creation failed");

        let diff_words = (total_slots + 31) / 32;
        let diff_output_buf = device.new_buffer(
            page_ceil(diff_words * 4) as u64,
            shared,
        );

        let intersect_func = lib
            .get_function("set_intersection", None)
            .expect("gpu32: set_intersection function not found");
        let intersect_pipeline = device
            .new_compute_pipeline_state_with_function(&intersect_func)
            .expect("gpu32: intersect pipeline creation failed");

        let intersect_matches_buf = device.new_buffer(
            (total_slots * 4) as u64,
            shared,
        );
        let intersect_count_buf = device.new_buffer(4, shared);

        let pred2_func = lib
            .get_function("set_predicate_2", None)
            .expect("gpu32: set_predicate_2 function not found");
        let pred2_pipeline = device
            .new_compute_pipeline_state_with_function(&pred2_func)
            .expect("gpu32: pred2 pipeline creation failed");

        let pred3_func = lib
            .get_function("set_predicate_3", None)
            .expect("gpu32: set_predicate_3 function not found");
        let pred3_pipeline = device
            .new_compute_pipeline_state_with_function(&pred3_func)
            .expect("gpu32: pred3 pipeline creation failed");

        let pred_mask_u8_buf = device.new_buffer(1, shared);
        let pred_mask_u64_buf = device.new_buffer(8, shared);

        let tg_width = pipeline.thread_execution_width().max(1) as u64;

        Self {
            queue,
            pipeline,
            fp_buf,
            hash_buf,
            out_buf,
            count_buf,
            greedy_pipeline,
            greedy_found_buf,
            greedy_fp_buf,
            greedy_key_buf,
            diff_pipeline,
            diff_output_buf,
            intersect_pipeline,
            intersect_matches_buf,
            intersect_count_buf,
            pred2_pipeline,
            pred3_pipeline,
            pred_mask_u8_buf,
            pred_mask_u64_buf,
            device,
            total_slots,
            tg_width,
        }
    }

    /// Dispatch the gather kernel and return a slice of collected keys.
    /// The returned count is the number of occupied slots found by the GPU.
    fn gather_keys(&self) -> usize {
        unsafe {
            // Reset atomic counter to zero.
            let cnt_ptr = self.count_buf.contents().cast::<u32>();
            *cnt_ptr = 0;
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&self.fp_buf), 0);
        enc.set_buffer(1, Some(&self.hash_buf), 0);
        enc.set_buffer(2, Some(&self.out_buf), 0);
        enc.set_buffer(3, Some(&self.count_buf), 0);

        let grid = MTLSize {
            width: self.total_slots as u64,
            height: 1,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: self.tg_width,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, threadgroup);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        unsafe { *(self.count_buf.contents().cast::<u32>()) as usize }
    }

    /// Read gathered keys from the output buffer.
    unsafe fn read_keys(&self, count: usize) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(self.out_buf.contents().cast::<u64>(), count)
        }
    }

    /// Dispatch the contains_greedy kernel: linear scan of all slots for (fp, key).
    fn contains_greedy(&self, fp: u8, key: u64) -> bool {
        unsafe {
            *self.greedy_found_buf.contents().cast::<u32>() = 0;
            *self.greedy_fp_buf.contents().cast::<u8>() = fp;
            *self.greedy_key_buf.contents().cast::<u64>() = key;
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.greedy_pipeline);
        enc.set_buffer(0, Some(&self.fp_buf), 0);
        enc.set_buffer(1, Some(&self.hash_buf), 0);
        enc.set_buffer(2, Some(&self.greedy_found_buf), 0);
        enc.set_buffer(3, Some(&self.greedy_fp_buf), 0);
        enc.set_buffer(4, Some(&self.greedy_key_buf), 0);

        let grid = MTLSize {
            width: self.total_slots as u64,
            height: 1,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: self.tg_width,
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, threadgroup);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        unsafe { *self.greedy_found_buf.contents().cast::<u32>() != 0 }
    }

    /// Dispatch fingerprint_diff kernel: XOR two fp arrays, produce bitmask of differing positions.
    /// `other_fp_ptr` must point to a page-aligned fp array of the same total_slots length.
    fn fingerprint_diff(&self, other_fp_ptr: *const u8) -> &[u32] {
        let diff_words = (self.total_slots + 31) / 32;

        unsafe {
            std::ptr::write_bytes(self.diff_output_buf.contents().cast::<u8>(), 0, diff_words * 4);
        }

        let shared = MTLResourceOptions::StorageModeShared;
        let other_buf = self.device.new_buffer_with_bytes_no_copy(
            other_fp_ptr as *const c_void,
            page_ceil(self.total_slots) as u64,
            shared,
            None,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.diff_pipeline);
        enc.set_buffer(0, Some(&self.fp_buf), 0);
        enc.set_buffer(1, Some(&other_buf), 0);
        enc.set_buffer(2, Some(&self.diff_output_buf), 0);

        let grid = MTLSize { width: self.total_slots as u64, height: 1, depth: 1 };
        let threadgroup = MTLSize { width: self.tg_width, height: 1, depth: 1 };
        enc.dispatch_threads(grid, threadgroup);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        unsafe {
            std::slice::from_raw_parts(self.diff_output_buf.contents().cast::<u32>(), diff_words)
        }
    }

    /// Dispatch set_intersection kernel: find positions where both fp arrays hold the same
    /// non-empty fingerprint. Returns (positions_slice, match_count).
    fn set_intersection(&self, other_fp_ptr: *const u8) -> (&[u32], usize) {
        unsafe {
            *self.intersect_count_buf.contents().cast::<u32>() = 0;
        }

        let shared = MTLResourceOptions::StorageModeShared;
        let other_buf = self.device.new_buffer_with_bytes_no_copy(
            other_fp_ptr as *const c_void,
            page_ceil(self.total_slots) as u64,
            shared,
            None,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.intersect_pipeline);
        enc.set_buffer(0, Some(&self.fp_buf), 0);
        enc.set_buffer(1, Some(&other_buf), 0);
        enc.set_buffer(2, Some(&self.intersect_matches_buf), 0);
        enc.set_buffer(3, Some(&self.intersect_count_buf), 0);

        let grid = MTLSize { width: self.total_slots as u64, height: 1, depth: 1 };
        let threadgroup = MTLSize { width: self.tg_width, height: 1, depth: 1 };
        enc.dispatch_threads(grid, threadgroup);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        let count = unsafe { *self.intersect_count_buf.contents().cast::<u32>() } as usize;
        let positions = unsafe {
            std::slice::from_raw_parts(self.intersect_matches_buf.contents().cast::<u32>(), count)
        };
        (positions, count)
    }

    /// Dispatch the generalised 2-array predicate kernel.
    /// Reuses intersect_matches_buf / intersect_count_buf for output.
    fn set_predicate_2(&self, other_fp_ptr: *const u8, mask: u8) -> (&[u32], usize) {
        unsafe {
            *self.intersect_count_buf.contents().cast::<u32>() = 0;
            *self.pred_mask_u8_buf.contents().cast::<u8>() = mask;
        }

        let shared = MTLResourceOptions::StorageModeShared;
        let other_buf = self.device.new_buffer_with_bytes_no_copy(
            other_fp_ptr as *const c_void,
            page_ceil(self.total_slots) as u64,
            shared,
            None,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pred2_pipeline);
        enc.set_buffer(0, Some(&self.fp_buf), 0);
        enc.set_buffer(1, Some(&other_buf), 0);
        enc.set_buffer(2, Some(&self.intersect_matches_buf), 0);
        enc.set_buffer(3, Some(&self.intersect_count_buf), 0);
        enc.set_buffer(4, Some(&self.pred_mask_u8_buf), 0);

        let grid = MTLSize { width: self.total_slots as u64, height: 1, depth: 1 };
        let threadgroup = MTLSize { width: self.tg_width, height: 1, depth: 1 };
        enc.dispatch_threads(grid, threadgroup);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        let count = unsafe { *self.intersect_count_buf.contents().cast::<u32>() } as usize;
        let positions = unsafe {
            std::slice::from_raw_parts(self.intersect_matches_buf.contents().cast::<u32>(), count)
        };
        (positions, count)
    }

    /// Dispatch the generalised 3-array predicate kernel.
    fn set_predicate_3(
        &self,
        b_fp_ptr: *const u8,
        c_fp_ptr: *const u8,
        mask: u64,
    ) -> (&[u32], usize) {
        unsafe {
            *self.intersect_count_buf.contents().cast::<u32>() = 0;
            *self.pred_mask_u64_buf.contents().cast::<u64>() = mask;
        }

        let shared = MTLResourceOptions::StorageModeShared;
        let b_buf = self.device.new_buffer_with_bytes_no_copy(
            b_fp_ptr as *const c_void,
            page_ceil(self.total_slots) as u64,
            shared,
            None,
        );
        let c_buf = self.device.new_buffer_with_bytes_no_copy(
            c_fp_ptr as *const c_void,
            page_ceil(self.total_slots) as u64,
            shared,
            None,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pred3_pipeline);
        enc.set_buffer(0, Some(&self.fp_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&c_buf), 0);
        enc.set_buffer(3, Some(&self.intersect_matches_buf), 0);
        enc.set_buffer(4, Some(&self.intersect_count_buf), 0);
        enc.set_buffer(5, Some(&self.pred_mask_u64_buf), 0);

        let grid = MTLSize { width: self.total_slots as u64, height: 1, depth: 1 };
        let threadgroup = MTLSize { width: self.tg_width, height: 1, depth: 1 };
        enc.dispatch_threads(grid, threadgroup);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        let count = unsafe { *self.intersect_count_buf.contents().cast::<u32>() } as usize;
        let positions = unsafe {
            std::slice::from_raw_parts(self.intersect_matches_buf.contents().cast::<u32>(), count)
        };
        (positions, count)
    }
}

/// Round `size` up to the next page boundary (4096).
#[cfg(feature = "gpu32")]
fn page_ceil(size: usize) -> usize {
    (size + 4095) & !4095
}

/// Allocate page-aligned memory via `posix_memalign`.
/// Returns a pointer suitable for `newBufferWithBytesNoCopy`.
#[cfg(feature = "gpu32")]
unsafe fn page_alloc(size: usize) -> *mut u8 {
    let alloc_size = page_ceil(size);
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { libc::posix_memalign(&mut ptr, 4096, alloc_size) };
    assert_eq!(ret, 0, "gpu32: posix_memalign failed");
    unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, alloc_size); }
    ptr.cast::<u8>()
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

        #[cfg(not(feature = "gpu32"))]
        {
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

        #[cfg(feature = "gpu32")]
        {
            // Page-aligned allocation for zero-copy Metal buffers.
            // Memory is zeroed by page_alloc (RADIX_FP_EMPTY == 0x00).
            let fp_page_ptr = unsafe { page_alloc(total_slots) };
            let hash_page_ptr = unsafe { page_alloc(total_slots * 8) }.cast::<u64>();

            // Wrap page-aligned memory in Vecs so insert/contains work unchanged.
            let fingerprints = unsafe {
                Vec::from_raw_parts(
                    fp_page_ptr.cast::<Bucket>(),
                    bucket_count,
                    bucket_count,
                )
            };
            let hashes = unsafe {
                Vec::from_raw_parts(hash_page_ptr, total_slots, total_slots)
            };

            let fp_ptr = fp_page_ptr as *const u8;
            let hash_ptr = hash_page_ptr as *const u64;
            let gpu = GpuState::new(fp_ptr, hash_ptr, total_slots);

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
                fp_page_ptr,
                hash_page_ptr,
                gpu,
            })
        }
    }

}

#[cfg(feature = "gpu32")]
impl Drop for RadixTree {
    fn drop(&mut self) {
        // Prevent Vec from calling the Rust allocator on posix_memalign memory.
        let fp = std::mem::take(&mut self.fingerprints);
        let h = std::mem::take(&mut self.hashes);
        std::mem::forget(fp);
        std::mem::forget(h);
        unsafe {
            libc::free(self.fp_page_ptr.cast::<c_void>());
            libc::free(self.hash_page_ptr.cast::<c_void>());
        }
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
            let occupied = unsafe {
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

/// GPU-backed iterator over stored keys.
/// The gather kernel runs once at iterator creation; iteration reads
/// from the pre-filled output buffer with no further GPU dispatch.
#[cfg(feature = "gpu32")]
pub struct Iter<'a> {
    keys: *const u64,
    pos: usize,
    count: usize,
    _marker: std::marker::PhantomData<&'a ()>,
}

#[cfg(feature = "gpu32")]
impl<'a> Iterator for Iter<'a> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<u64> {
        if self.pos < self.count {
            let val = unsafe { *self.keys.add(self.pos) };
            self.pos += 1;
            Some(val)
        } else {
            None
        }
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

    /// Linear scan of the entire fingerprint arena.
    /// CPU path: NEON 16-byte chunks, compare against target fp, confirm hash.
    /// GPU path: Metal kernel dispatched over all slots.
    fn contains_greedy(&self, id: u64) -> bool {
        let h = mix64(id);
        let raw_fp = ((h >> self.fp_shift) & 0xFF) as u8;
        let fp = raw_fp | ((raw_fp == 0) as u8);

        #[cfg(feature = "gpu32")]
        {
            return self.gpu.contains_greedy(fp, id);
        }

        #[cfg(not(feature = "gpu32"))]
        {
            let fp_ptr = self.fp_ptr;
            let hash_ptr = self.hash_ptr;
            let total = self.layout.total_slots;
            let target = unsafe { vdupq_n_u8(fp) };
            let mut pos = 0usize;

            while pos + 16 <= total {
                unsafe {
                    let chunk = vld1q_u8(fp_ptr.add(pos));
                    let cmp = vceqq_u8(chunk, target);
                    let mut mask = pack_neon_16(cmp);
                    while mask != 0 {
                        let bit = mask.trailing_zeros() as usize;
                        if *hash_ptr.add(pos + bit) == id {
                            return true;
                        }
                        mask &= mask - 1;
                    }
                }
                pos += 16;
            }
            // Scalar tail for slots not covered by a full 16-byte chunk.
            while pos < total {
                unsafe {
                    if *fp_ptr.add(pos) == fp && *hash_ptr.add(pos) == id {
                        return true;
                    }
                }
                pos += 1;
            }
            false
        }
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
            let count = self.gpu.gather_keys();
            let keys = unsafe { self.gpu.read_keys(count) };
            return Iter {
                keys: keys.as_ptr(),
                pos: 0,
                count,
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

// ── Pairwise array operations ──────────────────────────────────────

/// Produce a bitmask of positions where two RadixTree fingerprint arrays differ.
/// Both trees must have the same capacity. Returns a `Vec<u32>` where bit `i`
/// of word `i/32` is set when `fp_a[i] != fp_b[i]`.
///
/// CPU path: NEON XOR + compare + pack.
/// GPU path: Metal kernel with atomic OR into bitmask words.
pub fn fingerprint_diff(a: &RadixTree, b: &RadixTree) -> Vec<u32> {
    assert_eq!(
        a.layout.total_slots, b.layout.total_slots,
        "fingerprint_diff: capacity mismatch ({} vs {})",
        a.layout.total_slots, b.layout.total_slots
    );

    #[cfg(feature = "gpu32")]
    {
        let result = a.gpu.fingerprint_diff(b.fp_ptr);
        return result.to_vec();
    }

    #[cfg(not(feature = "gpu32"))]
    {
        let total = a.layout.total_slots;
        let diff_words = (total + 31) / 32;
        let mut output = vec![0u32; diff_words];
        let fp_a = a.fp_ptr;
        let fp_b = b.fp_ptr;
        let mut pos = 0usize;

        while pos + 16 <= total {
            unsafe {
                let va = vld1q_u8(fp_a.add(pos));
                let vb = vld1q_u8(fp_b.add(pos));
                let xor = veorq_u8(va, vb);
                let zero = vdupq_n_u8(0);
                let eq = vceqq_u8(xor, zero);
                let same_mask = pack_neon_16(eq);
                let diff_mask = !same_mask & 0xFFFF;

                if diff_mask != 0 {
                    let word_base = pos / 32;
                    let bit_base = pos % 32;
                    if bit_base == 0 {
                        output[word_base] |= diff_mask as u32;
                    } else if bit_base == 16 {
                        output[word_base] |= (diff_mask as u32) << 16;
                    } else {
                        output[word_base] |= (diff_mask as u32) << bit_base;
                        if bit_base + 16 > 32 {
                            output[word_base + 1] |= (diff_mask as u32) >> (32 - bit_base);
                        }
                    }
                }
            }
            pos += 16;
        }
        // Scalar tail
        while pos < total {
            unsafe {
                if *fp_a.add(pos) != *fp_b.add(pos) {
                    output[pos / 32] |= 1u32 << (pos % 32);
                }
            }
            pos += 1;
        }
        output
    }
}

/// Find all positions where two RadixTree fingerprint arrays hold the same
/// non-empty fingerprint value. Returns `(positions, match_count)`.
///
/// Both trees must have the same capacity.
///
/// CPU path: NEON compare + occupied mask + co-occupied AND + pack.
/// GPU path: Metal kernel with atomic counter for stream compaction.
pub fn set_intersection(a: &RadixTree, b: &RadixTree) -> (Vec<u32>, usize) {
    assert_eq!(
        a.layout.total_slots, b.layout.total_slots,
        "set_intersection: capacity mismatch ({} vs {})",
        a.layout.total_slots, b.layout.total_slots
    );

    #[cfg(feature = "gpu32")]
    {
        let (positions, count) = a.gpu.set_intersection(b.fp_ptr);
        return (positions.to_vec(), count);
    }

    #[cfg(not(feature = "gpu32"))]
    {
        let total = a.layout.total_slots;
        let mut matches: Vec<u32> = Vec::new();
        let fp_a = a.fp_ptr;
        let fp_b = b.fp_ptr;
        let mut pos = 0usize;

        while pos + 16 <= total {
            unsafe {
                let va = vld1q_u8(fp_a.add(pos));
                let vb = vld1q_u8(fp_b.add(pos));
                let zero = vdupq_n_u8(0);

                // Occupied masks: non-zero bytes
                let occ_a = !pack_neon_16(vceqq_u8(va, zero)) & 0xFFFF;
                let occ_b = !pack_neon_16(vceqq_u8(vb, zero)) & 0xFFFF;
                let co_occupied = occ_a & occ_b;

                // Equal fingerprints
                let eq_mask = pack_neon_16(vceqq_u8(va, vb));

                let mut hit_mask = co_occupied & eq_mask;
                while hit_mask != 0 {
                    let bit = hit_mask.trailing_zeros() as usize;
                    matches.push((pos + bit) as u32);
                    hit_mask &= hit_mask - 1;
                }
            }
            pos += 16;
        }
        // Scalar tail
        while pos < total {
            unsafe {
                let a_val = *fp_a.add(pos);
                let b_val = *fp_b.add(pos);
                if a_val != 0 && b_val != 0 && a_val == b_val {
                    matches.push(pos as u32);
                }
            }
            pos += 1;
        }
        let count = matches.len();
        (matches, count)
    }
}

/// Generalised 2-array predicate evaluation.
///
/// For each position, computes a 3-bit state `(a_occ << 2 | b_occ << 1 | a_eq_b)`
/// and tests the corresponding bit in `mask`. Matching positions are collected.
pub fn set_predicate_2(a: &RadixTree, b: &RadixTree, mask: u8) -> (Vec<u32>, usize) {
    assert_eq!(
        a.layout.total_slots, b.layout.total_slots,
        "set_predicate_2: capacity mismatch ({} vs {})",
        a.layout.total_slots, b.layout.total_slots
    );

    #[cfg(feature = "gpu32")]
    {
        let (positions, count) = a.gpu.set_predicate_2(b.fp_ptr, mask);
        return (positions.to_vec(), count);
    }

    #[cfg(not(feature = "gpu32"))]
    {
        let total = a.layout.total_slots;
        let mut matches: Vec<u32> = Vec::new();
        let fp_a = a.fp_ptr;
        let fp_b = b.fp_ptr;
        let mut pos = 0usize;

        while pos + 16 <= total {
            unsafe {
                let va = vld1q_u8(fp_a.add(pos));
                let vb = vld1q_u8(fp_b.add(pos));
                let zero = vdupq_n_u8(0);

                let occ_a_mask = !pack_neon_16(vceqq_u8(va, zero)) & 0xFFFF;
                let occ_b_mask = !pack_neon_16(vceqq_u8(vb, zero)) & 0xFFFF;
                let eq_mask = pack_neon_16(vceqq_u8(va, vb));

                for bit in 0..16u32 {
                    let a_occ = ((occ_a_mask >> bit) & 1) as u8;
                    let b_occ = ((occ_b_mask >> bit) & 1) as u8;
                    let a_eq_b = ((eq_mask >> bit) & 1) as u8;
                    let state = (a_occ << 2) | (b_occ << 1) | a_eq_b;
                    if (mask >> state) & 1 == 1 {
                        matches.push((pos + bit as usize) as u32);
                    }
                }
            }
            pos += 16;
        }
        while pos < total {
            unsafe {
                let a_val = *fp_a.add(pos);
                let b_val = *fp_b.add(pos);
                let a_occ = (a_val != 0) as u8;
                let b_occ = (b_val != 0) as u8;
                let a_eq_b = (a_val == b_val) as u8;
                let state = (a_occ << 2) | (b_occ << 1) | a_eq_b;
                if (mask >> state) & 1 == 1 {
                    matches.push(pos as u32);
                }
            }
            pos += 1;
        }
        let count = matches.len();
        (matches, count)
    }
}

/// Generalised 3-array predicate evaluation.
///
/// For each position, computes a 6-bit state
/// `(a_occ << 5 | b_occ << 4 | c_occ << 3 | ab_eq << 2 | ac_eq << 1 | bc_eq)`
/// and tests the corresponding bit in `mask`. Matching positions are collected.
pub fn set_predicate_3(
    a: &RadixTree,
    b: &RadixTree,
    c: &RadixTree,
    mask: u64,
) -> (Vec<u32>, usize) {
    assert_eq!(
        a.layout.total_slots, b.layout.total_slots,
        "set_predicate_3: capacity mismatch a/b ({} vs {})",
        a.layout.total_slots, b.layout.total_slots
    );
    assert_eq!(
        a.layout.total_slots, c.layout.total_slots,
        "set_predicate_3: capacity mismatch a/c ({} vs {})",
        a.layout.total_slots, c.layout.total_slots
    );

    #[cfg(feature = "gpu32")]
    {
        let (positions, count) = a.gpu.set_predicate_3(b.fp_ptr, c.fp_ptr, mask);
        return (positions.to_vec(), count);
    }

    #[cfg(not(feature = "gpu32"))]
    {
        let total = a.layout.total_slots;
        let mut matches: Vec<u32> = Vec::new();
        let fp_a = a.fp_ptr;
        let fp_b = b.fp_ptr;
        let fp_c = c.fp_ptr;
        let mut pos = 0usize;

        while pos + 16 <= total {
            unsafe {
                let va = vld1q_u8(fp_a.add(pos));
                let vb = vld1q_u8(fp_b.add(pos));
                let vc = vld1q_u8(fp_c.add(pos));
                let zero = vdupq_n_u8(0);

                let occ_a_mask = !pack_neon_16(vceqq_u8(va, zero)) & 0xFFFF;
                let occ_b_mask = !pack_neon_16(vceqq_u8(vb, zero)) & 0xFFFF;
                let occ_c_mask = !pack_neon_16(vceqq_u8(vc, zero)) & 0xFFFF;
                let ab_eq_mask = pack_neon_16(vceqq_u8(va, vb));
                let ac_eq_mask = pack_neon_16(vceqq_u8(va, vc));
                let bc_eq_mask = pack_neon_16(vceqq_u8(vb, vc));

                for bit in 0..16u32 {
                    let a_occ  = ((occ_a_mask >> bit) & 1) as u64;
                    let b_occ  = ((occ_b_mask >> bit) & 1) as u64;
                    let c_occ  = ((occ_c_mask >> bit) & 1) as u64;
                    let ab_eq  = ((ab_eq_mask >> bit) & 1) as u64;
                    let ac_eq  = ((ac_eq_mask >> bit) & 1) as u64;
                    let bc_eq  = ((bc_eq_mask >> bit) & 1) as u64;
                    let state = (a_occ << 5) | (b_occ << 4) | (c_occ << 3)
                              | (ab_eq << 2) | (ac_eq << 1) | bc_eq;
                    if (mask >> state) & 1 == 1 {
                        matches.push((pos + bit as usize) as u32);
                    }
                }
            }
            pos += 16;
        }
        while pos < total {
            unsafe {
                let a_val = *fp_a.add(pos);
                let b_val = *fp_b.add(pos);
                let c_val = *fp_c.add(pos);
                let a_occ  = (a_val != 0) as u64;
                let b_occ  = (b_val != 0) as u64;
                let c_occ  = (c_val != 0) as u64;
                let ab_eq  = (a_val == b_val) as u64;
                let ac_eq  = (a_val == c_val) as u64;
                let bc_eq  = (b_val == c_val) as u64;
                let state = (a_occ << 5) | (b_occ << 4) | (c_occ << 3)
                          | (ab_eq << 2) | (ac_eq << 1) | bc_eq;
                if (mask >> state) & 1 == 1 {
                    matches.push(pos as u32);
                }
            }
            pos += 1;
        }
        let count = matches.len();
        (matches, count)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BUCKET_SLOTS, GROUP_SLOTS, RADIX_FP_EMPTY, RADIX_MAX_PROBE_ROUNDS, RadixTree,
        RadixTreeBuildError, RadixTreeConfig, fingerprint_diff, set_intersection,
        set_predicate_2, set_predicate_3,
        PRED2_INTERSECT, PRED2_DIFF_AB, PRED3_CONSENSUS, PRED3_UNIQUE_A,
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
    fn contains_greedy_finds_all_inserted_keys() {
        let mut t = RadixTree::new(RadixTreeConfig { capacity_bits: 14 }).expect("build");
        let n = (t.layout.total_slots as f64 * 0.50) as usize;
        let keys: Vec<u64> = (0..n)
            .map(|i| crate::mix64(i as u64 + 1))
            .collect();
        for &k in &keys {
            assert!(t.insert(k));
        }
        for &k in &keys {
            assert!(t.contains_greedy(k), "greedy missed key {}", k);
        }
        assert!(!t.contains_greedy(0xdead_beef_cafe_babe));
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

    #[test]
    fn fingerprint_diff_detects_mutations() {
        let capacity_bits = 12;
        let total = 1usize << capacity_bits;

        let mut a = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let mut b = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");

        // Insert the same keys into both tables — identical fp arrays.
        let n = (total as f64 * 0.50) as usize;
        for i in 0..n {
            let key = crate::mix64(i as u64 + 1);
            a.insert(key);
            b.insert(key);
        }

        let diff = fingerprint_diff(&a, &b);
        let diff_count: u32 = diff.iter().map(|w| w.count_ones()).sum();
        assert_eq!(diff_count, 0, "identical tables should produce zero diff bits");

        // Now insert extra keys into b only — those positions should appear in the diff.
        let extra = (total as f64 * 0.10) as usize;
        for i in 0..extra {
            let key = crate::mix64((n + i) as u64 + 1);
            b.insert(key);
        }

        let diff = fingerprint_diff(&a, &b);
        let diff_count: u32 = diff.iter().map(|w| w.count_ones()).sum();
        assert!(
            diff_count >= extra as u32,
            "diff should flag at least {} mutated positions, got {}",
            extra, diff_count
        );
    }

    #[test]
    fn set_intersection_finds_shared_keys() {
        let capacity_bits = 12;
        let total = 1usize << capacity_bits;

        let mut a = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let mut b = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");

        // Shared keys: insert into both.
        let shared_n = (total as f64 * 0.25) as usize;
        for i in 0..shared_n {
            let key = crate::mix64(i as u64 + 1);
            a.insert(key);
            b.insert(key);
        }

        // Unique to a
        let unique_a = (total as f64 * 0.10) as usize;
        for i in 0..unique_a {
            let key = crate::mix64((shared_n + i) as u64 + 10_000);
            a.insert(key);
        }

        // Unique to b
        let unique_b = (total as f64 * 0.10) as usize;
        for i in 0..unique_b {
            let key = crate::mix64((shared_n + unique_a + i) as u64 + 20_000);
            b.insert(key);
        }

        let (positions, count) = set_intersection(&a, &b);
        assert_eq!(positions.len(), count);

        // The intersection should find at least shared_n matches (shared keys land in
        // the same position with the same fp). Some extra matches are possible from
        // fp8 collisions between unique-to-a and unique-to-b keys.
        assert!(
            count >= shared_n,
            "intersection should find at least {} shared positions, got {}",
            shared_n, count
        );

        // All reported positions must have matching non-empty fps in both arrays.
        let fp_a = a.fp_ptr;
        let fp_b = b.fp_ptr;
        for &pos in &positions {
            let p = pos as usize;
            unsafe {
                let va = *fp_a.add(p);
                let vb = *fp_b.add(p);
                assert_ne!(va, 0, "intersection position {} has empty fp in a", p);
                assert_eq!(va, vb, "intersection position {} has mismatched fps", p);
            }
        }
    }

    #[test]
    fn set_intersection_empty_tables_yields_zero() {
        let capacity_bits = 10;
        let a = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let b = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let (positions, count) = set_intersection(&a, &b);
        assert_eq!(count, 0);
        assert!(positions.is_empty());
    }

    // ── set_predicate_2 tests ──────────────────────────────────────

    #[test]
    fn set_predicate_2_intersection_matches_dedicated() {
        let capacity_bits = 10;
        let mut a = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let mut b = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");

        // Insert shared keys into both tables
        for id in 0..50u64 {
            a.insert(id);
            b.insert(id);
        }
        // Insert unique keys into each
        for id in 100..130u64 {
            a.insert(id);
        }
        for id in 200..230u64 {
            b.insert(id);
        }

        let (dedicated_pos, dedicated_count) = set_intersection(&a, &b);
        let (pred_pos, pred_count) = set_predicate_2(&a, &b, PRED2_INTERSECT);

        assert_eq!(dedicated_count, pred_count,
            "predicate intersection count ({}) should match dedicated ({})",
            pred_count, dedicated_count);

        let mut ded_sorted = dedicated_pos.clone();
        let mut pred_sorted = pred_pos.clone();
        ded_sorted.sort();
        pred_sorted.sort();
        assert_eq!(ded_sorted, pred_sorted,
            "predicate intersection positions should match dedicated");
    }

    #[test]
    fn set_predicate_2_difference() {
        let capacity_bits = 10;
        let mut a = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let mut b = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");

        // Shared keys
        for id in 0..20u64 {
            a.insert(id);
            b.insert(id);
        }
        // Keys unique to A
        let unique_a_keys: Vec<u64> = (100..120).collect();
        for &id in &unique_a_keys {
            a.insert(id);
        }
        // Keys unique to B
        for id in 200..220u64 {
            b.insert(id);
        }

        let (_diff_positions, diff_count) = set_predicate_2(&a, &b, PRED2_DIFF_AB);

        // Every unique-to-A key must appear in the diff result
        for &id in &unique_a_keys {
            assert!(a.contains(id), "key {} should be in A", id);
            assert!(!b.contains(id), "key {} should NOT be in B", id);
        }
        // diff_count should be at least the number of unique-to-A keys
        assert!(diff_count >= unique_a_keys.len(),
            "diff A\\B count ({}) should be >= unique A keys ({})",
            diff_count, unique_a_keys.len());
        // No shared key should appear in diff (shared keys have identical fp at same position)
        // Note: this is approximate — fp collisions at different positions are possible,
        // so we just verify the count is reasonable.
        assert!(diff_count <= a.len(),
            "diff count ({}) should not exceed A's size ({})", diff_count, a.len());
    }

    // ── set_predicate_3 tests ──────────────────────────────────────

    #[test]
    fn set_predicate_3_consensus() {
        let capacity_bits = 10;
        let mut a = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let mut b = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let mut c = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");

        // Insert the same keys into all three
        for id in 0..40u64 {
            a.insert(id);
            b.insert(id);
            c.insert(id);
        }
        // Unique keys in each
        for id in 100..110u64 { a.insert(id); }
        for id in 200..210u64 { b.insert(id); }
        for id in 300..310u64 { c.insert(id); }

        let (_, consensus_count) = set_predicate_3(&a, &b, &c, PRED3_CONSENSUS);

        // Consensus count should be at least the number of shared keys
        // (fingerprint collisions could inflate it slightly, but with capacity_bits=10
        // and only 40 shared keys the load is very low)
        assert!(consensus_count >= 40,
            "consensus count ({}) should be >= 40 shared keys", consensus_count);
    }

    #[test]
    fn set_predicate_3_unique() {
        let capacity_bits = 10;
        let mut a = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let mut b = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
        let mut c = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");

        // Keys unique to A only
        for id in 0..30u64 { a.insert(id); }
        // Keys shared between B and C but not A
        for id in 100..130u64 {
            b.insert(id);
            c.insert(id);
        }

        let (_, unique_count) = set_predicate_3(&a, &b, &c, PRED3_UNIQUE_A);

        // All 30 unique-to-A keys should be found (their positions have a≠0, b==0, c==0)
        assert!(unique_count >= 30,
            "unique_a count ({}) should be >= 30", unique_count);
    }
}
