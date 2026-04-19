#pragma once

// SPEC: docs/specs/runtime/SPEC.md §2.3 (GPU backend wiring);
//       docs/specs/gpu/SPEC.md §9 (engine wire-up), §5 (memory pool), §3 (streams)
// Exec pack: docs/development/m6_execution_pack.md T6.7
// Decisions: D-M6-3 (host-staged MPI), D-M6-12 (single pool), D-M6-13 (2 streams)
//
// `GpuContext` — engine-owned RAII container for the CUDA resources a run
// needs. In T6.7 one `DevicePool` + one `DeviceStream` (the "compute" stream
// from the D-M6-13 pair); the "mem" stream arrives at T6.9 when compute/MPI
// overlap is wired.
//
// Lifetime matches `SimulationEngine::init()` → `SimulationEngine::finalize()`.
// Constructor probes + selects a CUDA device and warms the pool (cold-start
// stall mitigation per D-M6-12). Destructor tears down in reverse order.
//
// CPU-only builds: the constructor throws `std::runtime_error` — the engine
// guards against this by only instantiating `GpuContext` when
// `runtime.backend == gpu`. Any CPU-only unit test touching it must pre-check.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/gpu_config.hpp"
#include "tdmd/gpu/types.hpp"

#include <memory>

namespace tdmd::runtime {

class GpuContext {
public:
  // Probes devices, selects `cfg.device_id`, allocates a DevicePool with
  // `cfg.memory_pool_init_size_mib` warm-up, and creates one compute stream.
  // Throws `std::runtime_error` when no CUDA device is visible, when
  // `cfg.device_id` is out of range, or on a CPU-only build.
  explicit GpuContext(const tdmd::gpu::GpuConfig& cfg);

  ~GpuContext();

  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;
  GpuContext(GpuContext&&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;

  [[nodiscard]] tdmd::gpu::DevicePool& pool() noexcept { return *pool_; }
  [[nodiscard]] tdmd::gpu::DeviceStream& compute_stream() noexcept { return compute_stream_; }
  [[nodiscard]] const tdmd::gpu::DeviceInfo& device_info() const noexcept { return device_info_; }

private:
  tdmd::gpu::DeviceInfo device_info_{};
  std::unique_ptr<tdmd::gpu::DevicePool> pool_;
  tdmd::gpu::DeviceStream compute_stream_{};
};

}  // namespace tdmd::runtime
