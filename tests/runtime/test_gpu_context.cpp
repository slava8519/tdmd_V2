// SPEC: docs/specs/runtime/SPEC.md §2.3; docs/specs/gpu/SPEC.md §9
// Exec pack: docs/development/m6_execution_pack.md T6.7

#include "tdmd/gpu/factories.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/runtime/gpu_context.hpp"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

namespace {

bool has_cuda_device() {
  try {
    return !tdmd::gpu::probe_devices().empty();
  } catch (...) {
    return false;
  }
}

}  // namespace

TEST_CASE("T6.7 — GpuContext constructs pool + stream on available device",
          "[runtime][gpu][gpu_context]") {
  if (!has_cuda_device()) {
    SKIP("no CUDA device visible");
  }
  tdmd::gpu::GpuConfig cfg{};
  cfg.memory_pool_init_size_mib = 4;  // tiny warm-up for the unit test
  tdmd::runtime::GpuContext ctx(cfg);
  REQUIRE(ctx.device_info().device_id == 0);
  // pool() / compute_stream() must return usable handles; just invoke the
  // accessors to detect any lifetime snag.
  (void) ctx.pool();
  (void) ctx.compute_stream();
}

TEST_CASE("T6.7 — GpuContext throws on out-of-range device_id", "[runtime][gpu][gpu_context]") {
  if (!has_cuda_device()) {
    SKIP("no CUDA device visible");
  }
  tdmd::gpu::GpuConfig cfg{};
  cfg.device_id = 999;
  REQUIRE_THROWS_AS(tdmd::runtime::GpuContext(cfg), std::runtime_error);
}

TEST_CASE("T6.7 — GpuContext throws on CPU-only build", "[runtime][gpu][gpu_context][cpu-only]") {
  if (has_cuda_device()) {
    SKIP("CUDA device present — this case covers the TDMD_BUILD_CUDA=OFF guard");
  }
  tdmd::gpu::GpuConfig cfg{};
  REQUIRE_THROWS_AS(tdmd::runtime::GpuContext(cfg), std::runtime_error);
}
