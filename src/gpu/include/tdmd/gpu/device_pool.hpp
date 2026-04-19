#pragma once

// SPEC: docs/specs/gpu/SPEC.md §5 (memory model, cached device pool)
// Master spec: §14 M6, §15.2
// Exec pack: docs/development/m6_execution_pack.md T6.3, D-M6-12
//
// DevicePool — concrete DeviceAllocator for both device memory
// (cudaMallocAsync-backed) and pinned host memory (cudaMallocHost-backed).
// Single class owns both pools because in M6 there's a 1:1 correspondence
// between a rank and the pair of device+pinned-host pools it uses (no
// multi-tenant sharing).
//
// Size classes (D-M6-12): {4 KiB, 64 KiB, 1 MiB, 16 MiB, 256 MiB}. Requests
// round up to the nearest class. Blocks above 256 MiB fall through to
// direct cudaMallocAsync / cudaMallocHost with no pooling.
//
// LRU eviction policy is deferred (OQ-M6-1) — simple grow-on-demand free
// lists in v1.0. If T6.5 kernel pressure-testing shows pool bloat is a
// problem, SPEC §5.1 will add LRU.
//
// Thread-safety: M6 runs one rank per process, one logical thread per rank
// (no OpenMP inside kernels). DevicePool is **not** thread-safe — external
// synchronisation required if concurrent access is ever needed.

#include "tdmd/gpu/device_allocator.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/types.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace tdmd::gpu {

class DevicePool : public DeviceAllocator {
public:
  // Constructs a pool bound to `cfg.device_id`. Warms up the device pool
  // with `cfg.memory_pool_init_size_mib` MiB of free blocks to avoid
  // cold-start stall on first kernel launch (D-M6-12).
  //
  // Throws std::runtime_error on CPU-only build or CUDA failure.
  explicit DevicePool(const GpuConfig& cfg);

  // Shuts down the pool: frees all outstanding blocks via cudaFreeAsync
  // and cudaFreeHost respectively. Any live DevicePtr<> blocks allocated
  // from this pool are orphaned (their deleter context becomes stale) —
  // caller must ensure DevicePool outlives all DevicePtr<> it issued.
  ~DevicePool() override;

  DevicePool(const DevicePool&) = delete;
  DevicePool& operator=(const DevicePool&) = delete;
  DevicePool(DevicePool&&) = delete;
  DevicePool& operator=(DevicePool&&) = delete;

  // DeviceAllocator interface.
  DevicePtr<std::byte> allocate_device(std::size_t nbytes, DeviceStream& stream) override;
  DevicePtr<std::byte> allocate_pinned_host(std::size_t nbytes) override;
  std::size_t bytes_in_use_device() const noexcept override;
  std::size_t bytes_in_use_pinned() const noexcept override;

  // Additional telemetry hooks — used by T6.11 for NVTX + counter export.
  std::size_t device_pool_hits() const noexcept;
  std::size_t device_pool_misses() const noexcept;
  std::size_t pinned_pool_hits() const noexcept;
  std::size_t pinned_pool_misses() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tdmd::gpu
