// SPEC: docs/specs/gpu/SPEC.md §7.3, §1.1; docs/specs/potentials/SPEC.md §6
// Exec pack: docs/development/m8_execution_pack.md T8.6a

#include "tdmd/potentials/snap_gpu_adapter.hpp"

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/snap_gpu.hpp"
#include "tdmd/gpu/snap_gpu_mixed.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tdmd::potentials {

SnapGpuAdapter::SnapGpuAdapter(const SnapData& data)
    : data_(&data), gpu_(std::make_unique<tdmd::gpu::SnapGpuActive>()) {
  const std::size_t n_species = data.species.size();
  if (n_species == 0) {
    throw std::invalid_argument("SnapGpuAdapter: SnapData has no species");
  }
  if (!(data.params.rcutfac > 0.0)) {
    throw std::invalid_argument("SnapGpuAdapter: rcutfac must be > 0");
  }
  // M8-scope flag gate — matches SnapPotential / snap_file validation so the
  // GPU path cannot silently permit configurations the CPU oracle rejects.
  if (data.params.chemflag) {
    throw std::invalid_argument("SnapGpuAdapter: chemflag=1 not supported in M8 (T8.6 scope)");
  }
  if (data.params.quadraticflag) {
    throw std::invalid_argument("SnapGpuAdapter: quadraticflag=1 not supported in M8 (T8.6 scope)");
  }
  if (data.params.switchinnerflag) {
    throw std::invalid_argument(
        "SnapGpuAdapter: switchinnerflag=1 not supported in M8 (T8.6 scope)");
  }

  const std::size_t beta_stride = static_cast<std::size_t>(data.k_max) + 1U;
  for (const auto& sp : data.species) {
    if (sp.beta.size() != beta_stride) {
      throw std::invalid_argument(
          "SnapGpuAdapter: per-species β coefficient count does not match k_max + 1");
    }
  }
  if (data.rcut_sq_ab.size() != n_species * n_species) {
    throw std::invalid_argument("SnapGpuAdapter: rcut_sq_ab must be n_species × n_species");
  }

  radius_elem_flat_.resize(n_species);
  weight_elem_flat_.resize(n_species);
  for (std::size_t a = 0; a < n_species; ++a) {
    radius_elem_flat_[a] = data.species[a].radius_elem;
    weight_elem_flat_[a] = data.species[a].weight_elem;
  }

  beta_flat_.resize(n_species * beta_stride);
  for (std::size_t a = 0; a < n_species; ++a) {
    const auto& src = data.species[a].beta;
    double* dst = beta_flat_.data() + a * beta_stride;
    for (std::size_t k = 0; k < beta_stride; ++k) {
      dst[k] = src[k];
    }
  }
}

SnapGpuAdapter::~SnapGpuAdapter() = default;
SnapGpuAdapter::SnapGpuAdapter(SnapGpuAdapter&&) noexcept = default;
SnapGpuAdapter& SnapGpuAdapter::operator=(SnapGpuAdapter&&) noexcept = default;

std::uint64_t SnapGpuAdapter::compute_version() const noexcept {
  return gpu_ ? gpu_->compute_version() : 0;
}

ForceResult SnapGpuAdapter::compute(AtomSoA& atoms,
                                    const Box& box,
                                    const CellGrid& grid,
                                    tdmd::gpu::DevicePool& pool,
                                    tdmd::gpu::DeviceStream& stream) {
  ForceResult result;
  const std::size_t n = atoms.size();
  if (n == 0) {
    return result;
  }

  const std::size_t n_species = data_->species.size();
  const std::size_t beta_stride = static_cast<std::size_t>(data_->k_max) + 1U;
  const double max_cutoff = data_->max_pairwise_cutoff();

  tdmd::gpu::BoxParams bp;
  bp.xlo = box.xlo;
  bp.ylo = box.ylo;
  bp.zlo = box.zlo;
  bp.lx = box.lx();
  bp.ly = box.ly();
  bp.lz = box.lz();
  bp.cell_x = grid.cell_x();
  bp.cell_y = grid.cell_y();
  bp.cell_z = grid.cell_z();
  bp.nx = grid.nx();
  bp.ny = grid.ny();
  bp.nz = grid.nz();
  bp.periodic_x = box.periodic_x;
  bp.periodic_y = box.periodic_y;
  bp.periodic_z = box.periodic_z;
  bp.cutoff = max_cutoff;
  bp.skin = 0.0;  // SNAP GPU walks its own stencil; no skin needed.

  tdmd::gpu::SnapTablesHost tables;
  tables.twojmax = data_->params.twojmax;
  tables.rcutfac = data_->params.rcutfac;
  tables.rfac0 = data_->params.rfac0;
  tables.rmin0 = data_->params.rmin0;
  tables.switchflag = data_->params.switchflag ? 1 : 0;
  tables.bzeroflag = data_->params.bzeroflag ? 1 : 0;
  tables.bnormflag = data_->params.bnormflag ? 1 : 0;
  tables.wselfallflag = data_->params.wselfallflag ? 1 : 0;
  tables.k_max = data_->k_max;
  // idxb_max / idxu_max / idxz_max are computed by SnaEngine::init() in the
  // CPU path. T8.6b will re-derive them inside SnapGpu::Impl; in T8.6a the
  // adapter passes 0 since compute() throws before consuming them.
  tables.idxb_max = 0;
  tables.idxu_max = 0;
  tables.idxz_max = 0;
  tables.n_species = n_species;
  tables.beta_stride = beta_stride;
  tables.radius_elem = radius_elem_flat_.data();
  tables.weight_elem = weight_elem_flat_.data();
  tables.beta_coefficients = beta_flat_.data();
  tables.rcut_sq_ab = data_->rcut_sq_ab.data();

  tdmd::gpu::SnapGpuResult r = gpu_->compute(n,
                                             atoms.type.data(),
                                             atoms.x.data(),
                                             atoms.y.data(),
                                             atoms.z.data(),
                                             grid.cell_count(),
                                             grid.cell_offsets().data(),
                                             grid.cell_atoms().data(),
                                             bp,
                                             tables,
                                             atoms.fx.data(),
                                             atoms.fy.data(),
                                             atoms.fz.data(),
                                             pool,
                                             stream);

  result.potential_energy = r.potential_energy;
  for (std::size_t k = 0; k < 6; ++k) {
    result.virial[k] = r.virial[k];
  }
  return result;
}

}  // namespace tdmd::potentials
