# GPU-Accelerated Iteration — `gpu32` Feature

## Context

The radix tree iterator uses NEON SIMD to scan 16-byte fingerprint chunks, building occupied bitmasks and yielding hashes for set bits. This achieves ~1.5 Gelem/s at 25% load on Apple Silicon. The question is whether the Apple GPU — sharing unified memory with the CPU — can beat this for full-table iteration.

An initial `gpu32` implementation dispatched a Metal compute kernel per `iter()` call, but introduced seven performance problems that made it ~10× slower than the CPU path. This document records the problems and the fixes applied.

---

## Problem 1: Buffer copy on every `iter()` call

### What was wrong

`new_buffer_with_data` copies the entire fingerprint arena into a new Metal buffer on each `iter()` invocation. On Apple Silicon's unified memory architecture, the fingerprints are already in GPU-accessible DRAM — copying them is pure waste.

### Fix

Use `new_buffer_with_bytes_no_copy` to wrap the existing fingerprint and hash memory directly. Metal maps the pointer into GPU address space with zero copy. This requires the backing memory to be page-aligned (4096 bytes) and the buffer length to be a page multiple.

### Implementation

Fingerprint and hash arenas are allocated via `posix_memalign` with 4096-byte alignment. The allocation size is rounded up to the next page boundary:

```rust
fn page_ceil(size: usize) -> usize {
    (size + 4095) & !4095
}

unsafe fn page_alloc(size: usize) -> *mut u8 {
    let alloc_size = page_ceil(size);
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let ret = libc::posix_memalign(&mut ptr, 4096, alloc_size);
    assert_eq!(ret, 0, "posix_memalign failed");
    std::ptr::write_bytes(ptr.cast::<u8>(), 0, alloc_size);
    ptr.cast::<u8>()
}
```

The page-aligned memory is wrapped in `Vec::from_raw_parts` so that existing `insert`/`contains` code — which indexes into `self.fingerprints[bucket]` and `self.hashes[slot]` — works without modification. A custom `Drop` impl calls `std::mem::forget` on the Vecs before `libc::free` on the raw pointers, preventing the Rust allocator from freeing `posix_memalign` memory.

The Metal buffers are created once at table construction:

```rust
let fp_buf = device.new_buffer_with_bytes_no_copy(
    fp_ptr as *const c_void,
    page_ceil(total_slots) as u64,
    MTLResourceOptions::StorageModeShared,
    None,  // Rust owns the lifetime
);
```

### Considerations

- `StorageModeShared` is correct for UMA — both CPU and GPU access the same physical pages.
- The deallocator is `None` because Rust's `Drop` impl manages the memory lifetime.
- `RADIX_FP_EMPTY == 0x00`, so zeroed pages from `page_alloc` are correctly initialised.

---

## Problem 2: Output buffer allocated per call

### What was wrong

Each `iter()` allocated a fresh Metal buffer for the kernel's output flags. Metal buffer allocation is not free — it involves kernel-space page mapping.

### Fix

Pre-allocate the output buffer at table construction time and reuse it across all `iter()` calls. The output buffer holds up to `total_slots` gathered `u64` keys (worst case: 100% occupancy).

---

## Problem 3: Command buffer + encoder created per call

### What was wrong

Full Metal dispatch ceremony — command buffer creation, encoder setup, pipeline binding, dispatch, commit, synchronous wait — on every `iter()` call. At the benchmark table size (262K slots = 256KB fingerprints), dispatch overhead likely exceeded kernel compute time.

### Fix

The Metal device, command queue, and compute pipeline are created once and stored as persistent state on `RadixTree` in a `GpuState` struct. Only the command buffer and encoder are created per `iter()` call — this is unavoidable in Metal's execution model, but pipeline compilation and queue creation are amortised to zero.

```rust
struct GpuState {
    queue: metal::CommandQueue,
    pipeline: ComputePipelineState,
    fp_buf: Buffer,
    hash_buf: Buffer,
    out_buf: Buffer,
    count_buf: Buffer,
    total_slots: usize,
    tg_width: u64,
}
```

### Considerations

- Metal command buffers are lightweight and designed for per-frame (per-call) creation.
- Pipeline state objects are expensive to create but immutable and reusable.
- The command queue is thread-safe and reusable.

---

## Problem 4: Host-side sequential gather after GPU

### What was wrong

The GPU kernel only marked occupied flags (`u8` per slot). After GPU completion, a scalar CPU loop iterated all flags and gathered hashes — the same work pattern as the CPU iterator but without NEON, without bitmasks, just byte-by-byte branching.

### Fix

Replace the two-phase approach (GPU marks flags, CPU gathers) with a single GPU kernel that does both: check fingerprint occupancy AND write the corresponding hash to a compacted output buffer using an atomic counter.

```metal
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
```

The kernel reads fingerprints and hashes via zero-copy buffers (Problem 1 fix), writes gathered keys to the pre-allocated output buffer (Problem 2 fix), and uses a 4-byte atomic counter for compaction. The host reads the counter after completion to know how many keys were gathered.

### Considerations

- `memory_order_relaxed` is sufficient — output order doesn't matter for iteration, only completeness.
- The atomic counter introduces contention at high occupancy, but Apple GPU hardware handles atomic increments efficiently within a threadgroup and across threadgroups via the L2 cache.
- Output key order is non-deterministic (depends on GPU thread scheduling). This matches the CPU iterator's behaviour — iteration order is not guaranteed.

---

## Problem 5: `Vec<u64>` materialisation + `IntoIter` wrapper

### What was wrong

Every `iter()` call heap-allocated a `Vec<u64>` of all keys, then wrapped it in `IntoIter`. The CPU path yields lazily with zero allocation.

### Fix

The GPU `Iter` struct now holds a raw pointer into the pre-allocated Metal output buffer and a count. No heap allocation occurs during `iter()` — the output buffer is reused across calls.

```rust
#[cfg(feature = "gpu32")]
pub struct Iter<'a> {
    keys: *const u64,
    pos: usize,
    count: usize,
    _marker: std::marker::PhantomData<&'a ()>,
}
```

The `next()` implementation is a simple pointer bump — no bounds checking beyond `pos < count`, no allocation, no indirection.

### Considerations

- The output buffer's contents are valid only until the next `iter()` call (which re-dispatches the kernel). The lifetime marker `'a` tied to the `RadixTree` borrow prevents use-after-invalidation.
- The buffer is `StorageModeShared`, so CPU reads after `wait_until_completed` see coherent data without explicit synchronisation.

---

## Problem 6: Dead code in `advance_chunk`

### What was wrong

A `#[cfg(feature = "gpu32")]` branch existed inside a `#[cfg(not(feature = "gpu32"))]` impl block. The code could never compile or execute — it was unreachable dead code from an earlier iteration that mixed CPU-side 32-byte scanning with the GPU feature flag.

### Fix

Removed the dead branch. The `advance_chunk` method now contains only the original 16-byte NEON scan logic, unconditionally, within its `#[cfg(not(feature = "gpu32"))]` block.

---

## Problem 7: Thread-local lazy init

### What was wrong

`GpuScanContext` was stored in a `thread_local! { RefCell<Option<...>> }`, requiring a `RefCell` borrow check on every `iter()` call plus lazy initialisation logic.

### Fix

`GpuState` is stored directly on `RadixTree`, initialised once in `new()`. No `RefCell`, no `Option`, no lazy init, no thread-local indirection. The Metal resources have the same lifetime as the table.

---

## Compile-time switching

All changes are gated behind `#[cfg(feature = "gpu32")]`. The default build produces identical code to the pre-gpu32 baseline — no runtime branches, no GPU dependencies, no page-aligned allocation.

The `gpu32` feature activates:
- `metal` and `libc` dependencies
- Page-aligned memory allocation in `RadixTree::new()` and `Clone`
- `GpuState` construction and storage on the struct
- Metal-backed `Iter` struct and `iter()` implementation
- Custom `Drop` to free `posix_memalign` memory

The `IndexTable` trait, `insert`, `contains`, and all other operations are unchanged under both configurations.

---

## Benchmark Protocol

Compare CPU and GPU iteration throughput at identical load factors and table sizes:

```
make bench IMPL=radix_tree OP=iter          # CPU baseline
make bench IMPL=radix_tree OP=iter GPU32=1  # GPU path
```

Record throughput in keys/s and effective bandwidth (bytes scanned per second) at load factors: 1%, 25%, 50%, 75%.

Expected GPU overhead profile:
- Fixed cost: command buffer creation + dispatch + wait (~10-50µs)
- Variable cost: kernel execution (proportional to total_slots, not occupied count)
- The GPU path scans all slots unconditionally; the CPU path skips empty 16-byte chunks via bitmask

At small table sizes (256K slots), dispatch overhead may dominate. The crossover point — where GPU throughput exceeds CPU — depends on table size and occupancy. Recording both dimensions is necessary to identify it.
