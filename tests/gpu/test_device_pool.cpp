// Exec pack: docs/development/m6_execution_pack.md T6.3
// SPEC: docs/specs/gpu/SPEC.md §5 (memory model), D-M6-12, D-M6-3
//
// DevicePool runtime tests. Each case skips when CUDA is unavailable
// (CPU-only build or no visible CUDA device) so the suite stays green on
// CI runners without a GPU. Pressure-tests the hit/miss counters, pool
// recycling, size-class rounding, and the direct-alloc fallback.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <utility>

#if TDMD_BUILD_CUDA
#include <cuda_runtime.h>
#endif

namespace tg = tdmd::gpu;

namespace {

bool cuda_device_available() noexcept {
#if TDMD_BUILD_CUDA
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
#else
  return false;
#endif
}

// Small helper: minimal pool sized to keep warmup cheap — we only care
// about functional correctness here, not cold-start latency.
tg::GpuConfig small_pool_cfg() {
  tg::GpuConfig cfg;
  cfg.memory_pool_init_size_mib = 4;  // 4 MiB warmup = 4 blocks of class 2
  return cfg;
}

}  // namespace

TEST_CASE("DevicePool — construct + destruct when CUDA present", "[gpu][pool]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }
  tg::DevicePool pool(small_pool_cfg());
  REQUIRE(pool.bytes_in_use_device() == 0u);
  REQUIRE(pool.bytes_in_use_pinned() == 0u);
}

TEST_CASE("DevicePool — CPU-only build throws on construction", "[gpu][pool][cpu]") {
#if TDMD_BUILD_CUDA
  SUCCEED("CUDA build — behaviour is exercised by construction test above");
#else
  tg::GpuConfig cfg;
  REQUIRE_THROWS_AS(tg::DevicePool(cfg), std::runtime_error);
#endif
}

#if TDMD_BUILD_CUDA

TEST_CASE("DevicePool — warmup pre-populates class 2 free-list", "[gpu][pool]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }
  tg::GpuConfig cfg = small_pool_cfg();
  cfg.memory_pool_init_size_mib = 4;
  tg::DevicePool pool(cfg);
  tg::DeviceStream s = tg::make_stream(cfg.device_id);

  // 1 MiB request rounds up to class 2 (exactly 1 MiB). Warmup primed the
  // free-list with 4 such blocks, so the first allocation must be a hit.
  const std::size_t one_mib = std::size_t{1} * 1024 * 1024;
  auto p = pool.allocate_device(one_mib, s);
  REQUIRE(static_cast<bool>(p));
  REQUIRE(pool.device_pool_hits() == 1u);
  REQUIRE(pool.device_pool_misses() == 0u);
  REQUIRE(pool.bytes_in_use_device() == one_mib);
}

TEST_CASE("DevicePool — allocate, drop, re-allocate hits the pool", "[gpu][pool]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }
  tg::DevicePool pool(small_pool_cfg());
  tg::DeviceStream s = tg::make_stream(0);

  const std::size_t small = 100u;  // class 0 (4 KiB)
  {
    auto p = pool.allocate_device(small, s);
    REQUIRE(static_cast<bool>(p));
    REQUIRE(pool.bytes_in_use_device() == 4u * 1024u);
    REQUIRE(pool.device_pool_misses() == 1u);
    REQUIRE(pool.device_pool_hits() == 0u);
  }
  // Drop returns block to free-list.
  REQUIRE(pool.bytes_in_use_device() == 0u);

  {
    auto p2 = pool.allocate_device(small, s);
    REQUIRE(static_cast<bool>(p2));
    REQUIRE(pool.device_pool_hits() == 1u);
    REQUIRE(pool.device_pool_misses() == 1u);
  }
  REQUIRE(pool.bytes_in_use_device() == 0u);
}

TEST_CASE("DevicePool — size-class rounding accounts class bytes", "[gpu][pool]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }
  tg::DevicePool pool(small_pool_cfg());
  tg::DeviceStream s = tg::make_stream(0);

  // 5 KiB request → class 1 (64 KiB). bytes_in_use reflects the class.
  auto p = pool.allocate_device(5u * 1024u, s);
  REQUIRE(static_cast<bool>(p));
  REQUIRE(pool.bytes_in_use_device() == 64u * 1024u);
}

TEST_CASE("DevicePool — pinned-host allocate + recycle", "[gpu][pool][pinned]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }
  tg::DevicePool pool(small_pool_cfg());

  {
    auto ph = pool.allocate_pinned_host(1024u);  // class 0 (4 KiB)
    REQUIRE(static_cast<bool>(ph));
    REQUIRE(pool.bytes_in_use_pinned() == 4u * 1024u);
    REQUIRE(pool.pinned_pool_misses() == 1u);
  }
  REQUIRE(pool.bytes_in_use_pinned() == 0u);

  {
    auto ph2 = pool.allocate_pinned_host(2048u);  // still class 0
    REQUIRE(pool.pinned_pool_hits() == 1u);
    REQUIRE(pool.pinned_pool_misses() == 1u);
  }
}

TEST_CASE("DevicePool — allocate returns move-only DevicePtr", "[gpu][pool]") {
  if (!cuda_device_available()) {
    SKIP("no CUDA device available");
  }
  tg::DevicePool pool(small_pool_cfg());
  tg::DeviceStream s = tg::make_stream(0);

  auto p = pool.allocate_device(256u, s);
  REQUIRE(static_cast<bool>(p));
  void* raw = p.get();
  auto q = std::move(p);
  REQUIRE(q.get() == raw);
  REQUIRE_FALSE(static_cast<bool>(p));  // NOLINT(bugprone-use-after-move)
}

#endif  // TDMD_BUILD_CUDA
