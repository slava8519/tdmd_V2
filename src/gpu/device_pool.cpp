// SPEC: docs/specs/gpu/SPEC.md §5 (memory model, cached device pool)
// Master spec: §14 M6, §15.2
// Exec pack: docs/development/m6_execution_pack.md T6.3, D-M6-12, D-M6-3
//
// DevicePool — concrete DeviceAllocator backed by cudaMallocAsync for
// device memory (D-M6-12) and cudaMallocHost for pinned host memory
// (D-M6-3, MPI host-staging). v1.0 uses simple grow-on-demand free-lists
// per size class; LRU eviction is deferred (OQ-M6-1) pending T6.5 kernel
// pressure-testing.

#include "tdmd/gpu/device_pool.hpp"

#include "tdmd/telemetry/nvtx.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if TDMD_BUILD_CUDA
#include "cuda_handles.hpp"
#endif

namespace tdmd::gpu {

#if TDMD_BUILD_CUDA

namespace {

// Size classes per D-M6-12. Requests round up to the nearest class;
// requests beyond the largest class take the direct-allocation path.
constexpr std::array<std::size_t, 5> kSizeClasses = {
    std::size_t{4} * 1024,           // 4 KiB
    std::size_t{64} * 1024,          // 64 KiB
    std::size_t{1} * 1024 * 1024,    // 1 MiB
    std::size_t{16} * 1024 * 1024,   // 16 MiB
    std::size_t{256} * 1024 * 1024,  // 256 MiB
};
constexpr std::size_t kNumClasses = kSizeClasses.size();
constexpr std::size_t kDirectClassIdx = kNumClasses;  // sentinel

std::size_t class_for(std::size_t nbytes) noexcept {
  for (std::size_t i = 0; i < kNumClasses; ++i) {
    if (nbytes <= kSizeClasses[i]) {
      return i;
    }
  }
  return kDirectClassIdx;
}

[[noreturn]] void throw_cuda_error(const char* op, cudaError_t err) {
  std::ostringstream oss;
  oss << "gpu::DevicePool::" << op << " failed: " << cudaGetErrorName(err) << " ("
      << cudaGetErrorString(err) << ")";
  throw std::runtime_error(oss.str());
}

}  // namespace

#endif  // TDMD_BUILD_CUDA

#if TDMD_BUILD_CUDA

struct DevicePool::Impl {
  explicit Impl(const GpuConfig& c) : cfg(c) {}

  GpuConfig cfg;

  // Stream used by the destructor to submit cudaFreeAsync ops for pool-class
  // blocks. Caller-supplied streams drive user-facing allocations.
  cudaStream_t cleanup_stream = nullptr;

  // Per-class free lists of recycled blocks.
  std::array<std::vector<void*>, kNumClasses> device_free;
  std::array<std::vector<void*>, kNumClasses> pinned_free;

  // All pool-class blocks we've ever allocated (for destructor sweep),
  // regardless of whether currently on the free list or out in a DevicePtr.
  std::unordered_map<void*, std::size_t> device_tracked;
  std::unordered_map<void*, std::size_t> pinned_tracked;

  // Oversize (direct-alloc) bookkeeping.
  struct DirectEntry {
    std::size_t nbytes = 0;
    cudaStream_t stream = nullptr;  // stream used at allocation; reused for free
  };
  std::unordered_map<void*, DirectEntry> device_direct;
  std::unordered_map<void*, std::size_t> pinned_direct;  // nbytes only

  // Telemetry.
  std::size_t bytes_in_use_device = 0;
  std::size_t bytes_in_use_pinned = 0;
  std::size_t device_pool_hits = 0;
  std::size_t device_pool_misses = 0;
  std::size_t pinned_pool_hits = 0;
  std::size_t pinned_pool_misses = 0;

  // Deleter trampolines — context = Impl*; size class encoded via template.
  template <std::size_t Class>
  static void device_pool_deleter(void* ptr, void* ctx) noexcept {
    auto* impl = static_cast<Impl*>(ctx);
    impl->device_free[Class].push_back(ptr);
    impl->bytes_in_use_device -= kSizeClasses[Class];
  }
  template <std::size_t Class>
  static void pinned_pool_deleter(void* ptr, void* ctx) noexcept {
    auto* impl = static_cast<Impl*>(ctx);
    impl->pinned_free[Class].push_back(ptr);
    impl->bytes_in_use_pinned -= kSizeClasses[Class];
  }
  static void device_direct_deleter(void* ptr, void* ctx) noexcept {
    auto* impl = static_cast<Impl*>(ctx);
    auto it = impl->device_direct.find(ptr);
    if (it == impl->device_direct.end()) {
      return;
    }
    cudaStream_t s = it->second.stream != nullptr ? it->second.stream : impl->cleanup_stream;
    (void) cudaFreeAsync(ptr, s);
    impl->bytes_in_use_device -= it->second.nbytes;
    impl->device_direct.erase(it);
  }
  static void pinned_direct_deleter(void* ptr, void* ctx) noexcept {
    auto* impl = static_cast<Impl*>(ctx);
    auto it = impl->pinned_direct.find(ptr);
    if (it == impl->pinned_direct.end()) {
      return;
    }
    (void) cudaFreeHost(ptr);
    impl->bytes_in_use_pinned -= it->second;
    impl->pinned_direct.erase(it);
  }

  // Per-class deleter tables — one function pointer per kSizeClasses entry.
  // Hand-rolled (instead of index_sequence) for readability; kNumClasses is 5.
  static constexpr std::array<DevicePtr<std::byte>::DeleterFn, kNumClasses> kDeviceDeleters = {
      &device_pool_deleter<0>,
      &device_pool_deleter<1>,
      &device_pool_deleter<2>,
      &device_pool_deleter<3>,
      &device_pool_deleter<4>,
  };
  static constexpr std::array<DevicePtr<std::byte>::DeleterFn, kNumClasses> kPinnedDeleters = {
      &pinned_pool_deleter<0>,
      &pinned_pool_deleter<1>,
      &pinned_pool_deleter<2>,
      &pinned_pool_deleter<3>,
      &pinned_pool_deleter<4>,
  };
};

static_assert(kNumClasses == 5, "deleter tables assume 5 size classes");

DevicePool::DevicePool(const GpuConfig& cfg) : impl_(std::make_unique<Impl>(cfg)) {
  cudaError_t err = cudaSetDevice(cfg.device_id);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaSetDevice", err);
  }

  err = cudaStreamCreateWithFlags(&impl_->cleanup_stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaStreamCreateWithFlags(cleanup)", err);
  }

  // Warm-up: pre-allocate `memory_pool_init_size_mib` MiB worth of 1 MiB
  // blocks (class 2) so the first real kernel launch does not stall on
  // allocator cold-start (D-M6-12). Blocks are immediately pushed onto
  // the free list for class 2 — subsequent allocate_device() of any size
  // ≤ 1 MiB finds them waiting.
  const std::size_t warmup_bytes =
      static_cast<std::size_t>(cfg.memory_pool_init_size_mib) * 1024 * 1024;
  constexpr std::size_t kWarmupClass = 2;  // 1 MiB
  const std::size_t warmup_count = warmup_bytes / kSizeClasses[kWarmupClass];

  for (std::size_t i = 0; i < warmup_count; ++i) {
    void* ptr = nullptr;
    err = cudaMallocAsync(&ptr, kSizeClasses[kWarmupClass], impl_->cleanup_stream);
    if (err != cudaSuccess) {
      throw_cuda_error("cudaMallocAsync(warmup)", err);
    }
    impl_->device_tracked[ptr] = kWarmupClass;
    impl_->device_free[kWarmupClass].push_back(ptr);
  }
  // Synchronise so warmup blocks are materialised before first user request.
  err = cudaStreamSynchronize(impl_->cleanup_stream);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaStreamSynchronize(warmup)", err);
  }
}

DevicePool::~DevicePool() {
  if (impl_ == nullptr) {
    return;
  }

  // Force-wait for all pending GPU work that might still reference our
  // blocks. Contract forbids DevicePtr outliving the pool, but we still
  // want teardown to be race-free if a caller violates that.
  (void) cudaDeviceSynchronize();

  for (const auto& [ptr, class_idx] : impl_->device_tracked) {
    (void) class_idx;
    (void) cudaFreeAsync(ptr, impl_->cleanup_stream);
  }
  for (const auto& [ptr, entry] : impl_->device_direct) {
    (void) entry;
    (void) cudaFreeAsync(ptr, impl_->cleanup_stream);
  }
  for (const auto& [ptr, class_idx] : impl_->pinned_tracked) {
    (void) class_idx;
    (void) cudaFreeHost(ptr);
  }
  for (const auto& [ptr, nbytes] : impl_->pinned_direct) {
    (void) nbytes;
    (void) cudaFreeHost(ptr);
  }

  if (impl_->cleanup_stream != nullptr) {
    (void) cudaStreamSynchronize(impl_->cleanup_stream);
    (void) cudaStreamDestroy(impl_->cleanup_stream);
    impl_->cleanup_stream = nullptr;
  }
}

DevicePtr<std::byte> DevicePool::allocate_device(std::size_t nbytes, DeviceStream& stream) {
  TDMD_NVTX_RANGE("gpu.pool.alloc_device");

  const std::size_t class_idx = class_for(nbytes);
  cudaStream_t cs = raw_stream(stream);

  if (class_idx < kNumClasses) {
    const std::size_t block_size = kSizeClasses[class_idx];
    void* ptr = nullptr;
    if (!impl_->device_free[class_idx].empty()) {
      ptr = impl_->device_free[class_idx].back();
      impl_->device_free[class_idx].pop_back();
      ++impl_->device_pool_hits;
    } else {
      cudaError_t err = cudaMallocAsync(&ptr, block_size, cs);
      if (err != cudaSuccess) {
        throw_cuda_error("cudaMallocAsync", err);
      }
      impl_->device_tracked[ptr] = class_idx;
      ++impl_->device_pool_misses;
    }
    impl_->bytes_in_use_device += block_size;
    return DevicePtr<std::byte>(static_cast<std::byte*>(ptr),
                                Impl::kDeviceDeleters[class_idx],
                                impl_.get());
  }

  // Oversize: direct cudaMallocAsync, no free-list recycling.
  void* ptr = nullptr;
  cudaError_t err = cudaMallocAsync(&ptr, nbytes, cs);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaMallocAsync(direct)", err);
  }
  impl_->device_direct[ptr] = {nbytes, cs};
  impl_->bytes_in_use_device += nbytes;
  ++impl_->device_pool_misses;
  return DevicePtr<std::byte>(static_cast<std::byte*>(ptr),
                              &Impl::device_direct_deleter,
                              impl_.get());
}

DevicePtr<std::byte> DevicePool::allocate_pinned_host(std::size_t nbytes) {
  TDMD_NVTX_RANGE("gpu.pool.alloc_pinned");

  const std::size_t class_idx = class_for(nbytes);

  if (class_idx < kNumClasses) {
    const std::size_t block_size = kSizeClasses[class_idx];
    void* ptr = nullptr;
    if (!impl_->pinned_free[class_idx].empty()) {
      ptr = impl_->pinned_free[class_idx].back();
      impl_->pinned_free[class_idx].pop_back();
      ++impl_->pinned_pool_hits;
    } else {
      cudaError_t err = cudaMallocHost(&ptr, block_size);
      if (err != cudaSuccess) {
        throw_cuda_error("cudaMallocHost", err);
      }
      impl_->pinned_tracked[ptr] = class_idx;
      ++impl_->pinned_pool_misses;
    }
    impl_->bytes_in_use_pinned += block_size;
    return DevicePtr<std::byte>(static_cast<std::byte*>(ptr),
                                Impl::kPinnedDeleters[class_idx],
                                impl_.get());
  }

  void* ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, nbytes);
  if (err != cudaSuccess) {
    throw_cuda_error("cudaMallocHost(direct)", err);
  }
  impl_->pinned_direct[ptr] = nbytes;
  impl_->bytes_in_use_pinned += nbytes;
  ++impl_->pinned_pool_misses;
  return DevicePtr<std::byte>(static_cast<std::byte*>(ptr),
                              &Impl::pinned_direct_deleter,
                              impl_.get());
}

std::size_t DevicePool::bytes_in_use_device() const noexcept {
  return impl_->bytes_in_use_device;
}

std::size_t DevicePool::bytes_in_use_pinned() const noexcept {
  return impl_->bytes_in_use_pinned;
}

std::size_t DevicePool::device_pool_hits() const noexcept {
  return impl_->device_pool_hits;
}

std::size_t DevicePool::device_pool_misses() const noexcept {
  return impl_->device_pool_misses;
}

std::size_t DevicePool::pinned_pool_hits() const noexcept {
  return impl_->pinned_pool_hits;
}

std::size_t DevicePool::pinned_pool_misses() const noexcept {
  return impl_->pinned_pool_misses;
}

#else  // CPU-only build — stub that throws on construction.

struct DevicePool::Impl {};

DevicePool::DevicePool(const GpuConfig& /*cfg*/) : impl_(nullptr) {
  throw std::runtime_error(
      "gpu::DevicePool: CPU-only build (TDMD_BUILD_CUDA=0); no CUDA runtime linked");
}

DevicePool::~DevicePool() = default;

DevicePtr<std::byte> DevicePool::allocate_device(std::size_t /*nbytes*/, DeviceStream& /*stream*/) {
  throw std::runtime_error("gpu::DevicePool::allocate_device: CPU-only build");
}

DevicePtr<std::byte> DevicePool::allocate_pinned_host(std::size_t /*nbytes*/) {
  throw std::runtime_error("gpu::DevicePool::allocate_pinned_host: CPU-only build");
}

std::size_t DevicePool::bytes_in_use_device() const noexcept {
  return 0;
}
std::size_t DevicePool::bytes_in_use_pinned() const noexcept {
  return 0;
}
std::size_t DevicePool::device_pool_hits() const noexcept {
  return 0;
}
std::size_t DevicePool::device_pool_misses() const noexcept {
  return 0;
}
std::size_t DevicePool::pinned_pool_hits() const noexcept {
  return 0;
}
std::size_t DevicePool::pinned_pool_misses() const noexcept {
  return 0;
}

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
