#pragma once

// SPEC: docs/specs/gpu/SPEC.md §2 (core types), §3 (streams), §5 (memory)
// Master spec: §14 M6, §15.2, §D.1 Philosophy B
// Exec pack: docs/development/m6_execution_pack.md T6.2
//
// Public types for the GPU primitive layer. Pure C++ — no CUDA headers.
// CUDA-specific state is hidden behind PIMPL per D-M6-17 so this header
// compiles on CPU-only builds (CI compile-only path); the concrete Impl
// struct bodies live in `gpu_types.cpp` (T6.2 stub) and are replaced by
// CUDA-bound versions in T6.3.
//
// M6 scope per D-M6-4: three kernels (neighbor-list, EAM/alloy force,
// VV NVE integrator). Everything else is reserved for M7+.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tdmd::gpu {

// CUDA device ordinal. 0 = default device.
using DeviceId = std::int32_t;

// Per-rank stream index. Range [0, GpuConfig::streams).
// 0 = stream_compute (kernel launches), 1 = stream_mem (H<->D copies).
// See gpu/SPEC §3 + D-M6-13.
using StreamOrdinal = std::uint8_t;

inline constexpr StreamOrdinal kStreamCompute = 0;
inline constexpr StreamOrdinal kStreamMem = 1;

// Capability flags a runtime can advertise; consumed by DevicePool / kernels
// to opt into fast paths when available. Enumeration mirrors comm::BackendCapability.
enum class GpuCapability : std::uint8_t {
  CudaMallocAsync,     // CUDA 11.2+ stream-ordered allocator (D-M6-12 pool backing)
  CooperativeLaunch,   // cross-block synchronization
  TensorCores,         // sm_80+ MMA (reserved for M9+ SNAP/PACE)
  AsyncMemcpy,         // cp.async, sm_80+
  L2CachePersistence,  // stream access policy hint, sm_80+
};

// Device description populated by gpu::probe() at runtime. On CPU-only
// builds probe() returns an empty vector; this struct remains declarable.
struct DeviceInfo {
  DeviceId device_id = 0;
  std::string name;
  std::uint32_t compute_capability_major = 0;
  std::uint32_t compute_capability_minor = 0;
  std::size_t total_global_memory_bytes = 0;
  std::uint32_t multiprocessor_count = 0;
  std::uint32_t warp_size = 32;
  std::uint32_t max_threads_per_block = 0;
  std::vector<GpuCapability> capabilities;
};

// RAII CUDA stream wrapper. Move-only.
//
// T6.2 ships a stub Impl with no members; T6.3 replaces it with a definition
// that owns `cudaStream_t`. Consumers in .cu translation units reach the
// underlying handle via impl()->stream_ once T6.3 lands.
class DeviceStream {
public:
  struct Impl;  // defined in gpu_types.cpp (T6.2 stub) / .cu (T6.3).

  DeviceStream();  // null stream — valid() == false
  explicit DeviceStream(std::unique_ptr<Impl> impl);
  ~DeviceStream();

  DeviceStream(const DeviceStream&) = delete;
  DeviceStream& operator=(const DeviceStream&) = delete;

  DeviceStream(DeviceStream&&) noexcept;
  DeviceStream& operator=(DeviceStream&&) noexcept;

  Impl* impl() noexcept { return impl_.get(); }
  const Impl* impl() const noexcept { return impl_.get(); }
  bool valid() const noexcept { return static_cast<bool>(impl_); }

private:
  std::unique_ptr<Impl> impl_;
};

// RAII CUDA event. Move-only. Same PIMPL pattern as DeviceStream.
class DeviceEvent {
public:
  struct Impl;

  DeviceEvent();
  explicit DeviceEvent(std::unique_ptr<Impl> impl);
  ~DeviceEvent();

  DeviceEvent(const DeviceEvent&) = delete;
  DeviceEvent& operator=(const DeviceEvent&) = delete;

  DeviceEvent(DeviceEvent&&) noexcept;
  DeviceEvent& operator=(DeviceEvent&&) noexcept;

  Impl* impl() noexcept { return impl_.get(); }
  const Impl* impl() const noexcept { return impl_.get(); }
  bool valid() const noexcept { return static_cast<bool>(impl_); }

private:
  std::unique_ptr<Impl> impl_;
};

// Owning handle to device memory. Move-only; deleter supplied by the pool
// that allocated the block (T6.3 DevicePool). For a null handle the deleter
// may be null.
//
// The deleter signature is noexcept to match CUDA free semantics — a pool
// that recycles the block into a free-list is exception-free; a backend
// that calls cudaFreeAsync swallows the error and reports via telemetry.
template <typename T>
class DevicePtr {
public:
  using DeleterFn = void (*)(void* device_ptr, void* context) noexcept;

  DevicePtr() noexcept = default;

  DevicePtr(T* ptr, DeleterFn deleter, void* context) noexcept
      : ptr_(ptr), deleter_(deleter), context_(context) {}

  ~DevicePtr() { reset(); }

  DevicePtr(const DevicePtr&) = delete;
  DevicePtr& operator=(const DevicePtr&) = delete;

  DevicePtr(DevicePtr&& other) noexcept
      : ptr_(other.ptr_), deleter_(other.deleter_), context_(other.context_) {
    other.ptr_ = nullptr;
    other.deleter_ = nullptr;
    other.context_ = nullptr;
  }

  DevicePtr& operator=(DevicePtr&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      deleter_ = other.deleter_;
      context_ = other.context_;
      other.ptr_ = nullptr;
      other.deleter_ = nullptr;
      other.context_ = nullptr;
    }
    return *this;
  }

  T* get() const noexcept { return ptr_; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  [[nodiscard]] T* release() noexcept {
    T* out = ptr_;
    ptr_ = nullptr;
    deleter_ = nullptr;
    context_ = nullptr;
    return out;
  }

  void reset() noexcept {
    if (ptr_ != nullptr && deleter_ != nullptr) {
      deleter_(static_cast<void*>(ptr_), context_);
    }
    ptr_ = nullptr;
    deleter_ = nullptr;
    context_ = nullptr;
  }

private:
  T* ptr_ = nullptr;
  DeleterFn deleter_ = nullptr;
  void* context_ = nullptr;
};

}  // namespace tdmd::gpu
