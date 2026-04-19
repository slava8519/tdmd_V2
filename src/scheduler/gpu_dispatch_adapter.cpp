#include "tdmd/scheduler/gpu_dispatch_adapter.hpp"

// SPEC: docs/specs/gpu/SPEC.md §3.2 (compute/mem overlap pipeline)
// Exec pack: docs/development/m7_execution_pack.md T7.8

#include <stdexcept>
#include <utility>

namespace tdmd::scheduler {

GpuDispatchAdapter::GpuDispatchAdapter(std::size_t pipeline_depth,
                                       tdmd::gpu::DevicePool& pool,
                                       tdmd::gpu::DeviceStream& compute_stream,
                                       tdmd::gpu::DeviceStream& mem_stream)
    : slots_(pipeline_depth),
      pool_(pool),
      compute_stream_(compute_stream),
      mem_stream_(mem_stream) {
  if (pipeline_depth == 0) {
    throw std::invalid_argument("GpuDispatchAdapter: pipeline_depth must be > 0");
  }
}

GpuDispatchAdapter::~GpuDispatchAdapter() = default;

std::size_t GpuDispatchAdapter::pipeline_depth() const noexcept {
  return slots_.size();
}

std::size_t GpuDispatchAdapter::enqueue_eam(std::size_t n,
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
                                            double* host_fz_out) {
  const std::size_t slot = next_slot_;
  if (slots_[slot].pending.has_value()) {
    throw std::logic_error(
        "GpuDispatchAdapter::enqueue_eam: slot holds an undrained handle — caller must drain in "
        "FIFO order before rotating through all K slots");
  }
  slots_[slot].pending = slots_[slot].eam.compute_async(n,
                                                        host_types,
                                                        host_x,
                                                        host_y,
                                                        host_z,
                                                        ncells,
                                                        host_cell_offsets,
                                                        host_cell_atoms,
                                                        params,
                                                        tables,
                                                        host_fx_out,
                                                        host_fy_out,
                                                        host_fz_out,
                                                        pool_,
                                                        compute_stream_,
                                                        mem_stream_);
  next_slot_ = (next_slot_ + 1) % slots_.size();
  return slot;
}

tdmd::gpu::EamAlloyGpuResult GpuDispatchAdapter::drain_eam(std::size_t slot) {
  if (slot >= slots_.size()) {
    throw std::out_of_range("GpuDispatchAdapter::drain_eam: slot out of range");
  }
  if (!slots_[slot].pending.has_value()) {
    throw std::logic_error("GpuDispatchAdapter::drain_eam: slot has no pending handle");
  }
  auto handle = std::move(*slots_[slot].pending);
  slots_[slot].pending.reset();
  return slots_[slot].eam.finalize_async(std::move(handle));
}

}  // namespace tdmd::scheduler
