#pragma once

// SPEC: docs/specs/gpu/SPEC.md §7.1; docs/specs/neighbor/SPEC.md §4
// Exec pack: docs/development/m6_execution_pack.md T6.4
//
// Thin neighbor-module facade for the GPU half-list builder. Lives under
// src/neighbor/ so that engine wiring (T6.7) can depend on
// tdmd::neighbor::GpuNeighborBuilder without pulling the full src/gpu/
// include path into call sites.
//
// The CUDA-specific work lives in `tdmd::gpu::NeighborListGpu`; this class
// forwards to it. When TDMD_BUILD_CUDA=0 the `build()` call throws so the
// CPU-only build still links cleanly.

#include "tdmd/gpu/neighbor_list_gpu.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <memory>

namespace tdmd::gpu {
class DevicePool;
class DeviceStream;
}  // namespace tdmd::gpu

namespace tdmd::neighbor {

class GpuNeighborBuilder {
public:
  // Takes a pool + stream by reference — both must outlive the builder.
  // The stream is remembered and re-used on every `build()` call.
  GpuNeighborBuilder(tdmd::gpu::DevicePool& pool, tdmd::gpu::DeviceStream& stream);
  ~GpuNeighborBuilder();

  GpuNeighborBuilder(const GpuNeighborBuilder&) = delete;
  GpuNeighborBuilder& operator=(const GpuNeighborBuilder&) = delete;
  GpuNeighborBuilder(GpuNeighborBuilder&&) noexcept;
  GpuNeighborBuilder& operator=(GpuNeighborBuilder&&) noexcept;

  void build(const AtomSoA& atoms,
             const Box& box,
             const CellGrid& grid,
             double cutoff,
             double skin);

  [[nodiscard]] const tdmd::gpu::NeighborListGpu& neighbor_list() const noexcept;
  [[nodiscard]] tdmd::gpu::NeighborListGpu& neighbor_list() noexcept;

private:
  tdmd::gpu::DevicePool* pool_;
  tdmd::gpu::DeviceStream* stream_;
  std::unique_ptr<tdmd::gpu::NeighborListGpu> nl_;
};

}  // namespace tdmd::neighbor
