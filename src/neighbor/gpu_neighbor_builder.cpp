// SPEC: docs/specs/gpu/SPEC.md §7.1; docs/specs/neighbor/SPEC.md §4
// Exec pack: docs/development/m6_execution_pack.md T6.4

#include "tdmd/neighbor/gpu_neighbor_builder.hpp"

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

namespace tdmd::neighbor {

GpuNeighborBuilder::GpuNeighborBuilder(tdmd::gpu::DevicePool& pool, tdmd::gpu::DeviceStream& stream)
    : pool_(&pool), stream_(&stream), nl_(std::make_unique<tdmd::gpu::NeighborListGpu>()) {}

GpuNeighborBuilder::~GpuNeighborBuilder() = default;
GpuNeighborBuilder::GpuNeighborBuilder(GpuNeighborBuilder&&) noexcept = default;
GpuNeighborBuilder& GpuNeighborBuilder::operator=(GpuNeighborBuilder&&) noexcept = default;

void GpuNeighborBuilder::build(const AtomSoA& atoms,
                               const Box& box,
                               const CellGrid& grid,
                               double cutoff,
                               double skin) {
  tdmd::gpu::BoxParams p;
  p.xlo = box.xlo;
  p.ylo = box.ylo;
  p.zlo = box.zlo;
  p.lx = box.lx();
  p.ly = box.ly();
  p.lz = box.lz();
  p.cell_x = grid.cell_x();
  p.cell_y = grid.cell_y();
  p.cell_z = grid.cell_z();
  p.nx = grid.nx();
  p.ny = grid.ny();
  p.nz = grid.nz();
  p.periodic_x = box.periodic_x;
  p.periodic_y = box.periodic_y;
  p.periodic_z = box.periodic_z;
  p.cutoff = cutoff;
  p.skin = skin;

  nl_->build(atoms.size(),
             atoms.x.data(),
             atoms.y.data(),
             atoms.z.data(),
             grid.cell_count(),
             grid.cell_offsets().data(),
             grid.cell_atoms().data(),
             p,
             *pool_,
             *stream_);
}

const tdmd::gpu::NeighborListGpu& GpuNeighborBuilder::neighbor_list() const noexcept {
  return *nl_;
}

tdmd::gpu::NeighborListGpu& GpuNeighborBuilder::neighbor_list() noexcept {
  return *nl_;
}

}  // namespace tdmd::neighbor
