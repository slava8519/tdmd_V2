#pragma once

// SPEC: docs/specs/gpu/SPEC.md §3.2 (compute/mem overlap pipeline)
// Master spec: §14 M7 (Pattern 2 two-level K-deep pipeline)
// Exec pack: docs/development/m7_execution_pack.md T7.8
//
// GpuDispatchAdapter — K-deep pipeline orchestrator for EAM GPU dispatch.
// The adapter rotates through K internal slots, each holding its own
// `EamAlloyGpu` instance + the last-in-flight async handle. Callers enqueue
// iteration work via `enqueue_eam(...)` and retrieve results later via
// `drain_eam(slot)` in the same FIFO order.
//
// The overlap win (measured by tests/gpu/test_overlap_budget.cpp): because
// EamAlloyGpu::compute_async() routes H2D/D2H to `mem_stream` and kernels
// to `compute_stream`, K back-to-back enqueues + K back-to-back drains
// produce a timeline in which slot N+1's H2D overlaps slot N's kernel
// execution and slot N-1's D2H (gpu/SPEC §3.2).
//
// Why K slots rather than a single EamAlloyGpu? Each instance owns its own
// per-call device buffer set via its async handle; trying to reuse a single
// EamAlloyGpu across K concurrent dispatches would alias buffers.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/eam_alloy_gpu.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace tdmd::scheduler {

class GpuDispatchAdapter {
public:
  GpuDispatchAdapter(std::size_t pipeline_depth,
                     tdmd::gpu::DevicePool& pool,
                     tdmd::gpu::DeviceStream& compute_stream,
                     tdmd::gpu::DeviceStream& mem_stream);
  ~GpuDispatchAdapter();

  GpuDispatchAdapter(const GpuDispatchAdapter&) = delete;
  GpuDispatchAdapter& operator=(const GpuDispatchAdapter&) = delete;
  GpuDispatchAdapter(GpuDispatchAdapter&&) = delete;
  GpuDispatchAdapter& operator=(GpuDispatchAdapter&&) = delete;

  [[nodiscard]] std::size_t pipeline_depth() const noexcept;

  // Enqueue EAM work on the next free slot; returns the slot id. Caller's
  // host output buffers MUST stay alive until the matching drain_eam()
  // returns. Throws if the slot is already holding an undrained handle —
  // callers must drain in enqueue order.
  [[nodiscard]] std::size_t enqueue_eam(std::size_t n,
                                        const std::uint32_t* host_types,
                                        const double* host_x,
                                        const double* host_y,
                                        const double* host_z,
                                        std::size_t ncells,
                                        const std::uint32_t* host_cell_offsets,
                                        const std::uint32_t* host_cell_atoms,
                                        const tdmd::gpu::BoxParams& params,
                                        const tdmd::gpu::EamAlloyTablesHost& tables,
                                        double* host_fx_out,
                                        double* host_fy_out,
                                        double* host_fz_out);

  // Wait on the slot's D2H event, run host Kahan reduction, return the EAM
  // result. Clears the slot for reuse. Throws if the slot is empty.
  tdmd::gpu::EamAlloyGpuResult drain_eam(std::size_t slot);

private:
  struct Slot {
    tdmd::gpu::EamAlloyGpu eam;
    std::optional<tdmd::gpu::EamAlloyAsyncHandle> pending;
  };

  std::vector<Slot> slots_;
  std::size_t next_slot_ = 0;
  tdmd::gpu::DevicePool& pool_;
  tdmd::gpu::DeviceStream& compute_stream_;
  tdmd::gpu::DeviceStream& mem_stream_;
};

}  // namespace tdmd::scheduler
