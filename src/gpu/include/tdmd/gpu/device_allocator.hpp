#pragma once

// SPEC: docs/specs/gpu/SPEC.md §5 (memory model)
// Master spec: §14 M6, §15.2
// Exec pack: docs/development/m6_execution_pack.md T6.2, D-M6-12, D-M6-3
//
// Abstract allocator contract for device + pinned-host memory. T6.3 lands
// the concrete DevicePool (cudaMallocAsync-backed, D-M6-12) and
// PinnedHostPool (cudaMallocHost for MPI staging, D-M6-3). Everything
// downstream (neighbor, EAM, VV kernels) consumes this interface — enables
// injection of mock allocators in unit tests.

#include "tdmd/gpu/types.hpp"

#include <cstddef>

namespace tdmd::gpu {

class DeviceAllocator {
public:
  virtual ~DeviceAllocator();

  DeviceAllocator() = default;
  DeviceAllocator(const DeviceAllocator&) = delete;
  DeviceAllocator& operator=(const DeviceAllocator&) = delete;
  DeviceAllocator(DeviceAllocator&&) = delete;
  DeviceAllocator& operator=(DeviceAllocator&&) = delete;

  // Allocate `nbytes` of device memory, stream-ordered against `stream` when
  // stream.valid(). Returns a DevicePtr<std::byte> whose deleter returns the
  // block to this allocator (pool recycle) or frees via cudaFreeAsync per
  // D-M6-12 policy. Typed cast happens at the call site.
  virtual DevicePtr<std::byte> allocate_device(std::size_t nbytes, DeviceStream& stream) = 0;

  // Allocate pinned (page-locked) host memory for MPI staging per D-M6-3.
  // Deleter releases via cudaFreeHost.
  virtual DevicePtr<std::byte> allocate_pinned_host(std::size_t nbytes) = 0;

  // Observability — telemetry + unit tests assert pool accounting.
  virtual std::size_t bytes_in_use_device() const noexcept = 0;
  virtual std::size_t bytes_in_use_pinned() const noexcept = 0;
};

}  // namespace tdmd::gpu
