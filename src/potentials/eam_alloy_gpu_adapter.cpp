// SPEC: docs/specs/gpu/SPEC.md §7.2, §1.1; docs/specs/potentials/SPEC.md §4.4
// Exec pack: docs/development/m6_execution_pack.md T6.5

#include "tdmd/potentials/eam_alloy_gpu_adapter.hpp"

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/eam_alloy_gpu.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tdmd::potentials {

namespace {

// Flatten a single TabulatedFunction's n·7-coefficient table into the tail
// of `out`. LAMMPS cell indexing is 1-based, TDMD storage is 0-based; we
// copy all n_ cells including the sentinel so `locate_dev` never walks off
// the end. Output layout: row-major 7 doubles per cell.
void flatten_tab_into(const TabulatedFunction& tab, std::vector<double>& out) {
  const std::size_t n = tab.size();
  out.reserve(out.size() + n * TabulatedFunction::kCoefficientsPerCell);
  for (std::size_t i = 1; i <= n; ++i) {
    const auto& c = tab.coeffs(i);
    for (double v : c) {
      out.push_back(v);
    }
  }
}

}  // namespace

EamAlloyGpuAdapter::EamAlloyGpuAdapter(const EamAlloyData& data)
    : data_(&data), gpu_(std::make_unique<tdmd::gpu::EamAlloyGpu>()) {
  const std::size_t n_species = data.species_names.size();
  if (n_species == 0) {
    throw std::invalid_argument("EamAlloyGpuAdapter: EamAlloyData has zero species");
  }
  if (data.F_rho.size() != n_species || data.rho_r.size() != n_species) {
    throw std::invalid_argument(
        "EamAlloyGpuAdapter: F_rho / rho_r table counts do not match N_species");
  }
  const std::size_t npairs = n_species * (n_species + 1) / 2;
  if (data.z2r.size() != npairs) {
    throw std::invalid_argument("EamAlloyGpuAdapter: z2r table count != N·(N+1)/2");
  }
  if (!(data.cutoff > 0.0)) {
    throw std::invalid_argument("EamAlloyGpuAdapter: cutoff must be strictly positive");
  }

  // Validate uniform table widths — all F_rho share nrho, all rho_r / z2r
  // share nr. The spline constructor already guarantees consistency within a
  // file, but cross-table checks guard against hand-built test fixtures.
  const std::size_t nrho = data.F_rho[0].size();
  const std::size_t nr = data.rho_r[0].size();
  for (const auto& t : data.F_rho) {
    if (t.size() != nrho) {
      throw std::invalid_argument("EamAlloyGpuAdapter: F_rho tables have inconsistent nrho");
    }
  }
  for (const auto& t : data.rho_r) {
    if (t.size() != nr) {
      throw std::invalid_argument("EamAlloyGpuAdapter: rho_r tables have inconsistent nr");
    }
  }
  for (const auto& t : data.z2r) {
    if (t.size() != nr) {
      throw std::invalid_argument("EamAlloyGpuAdapter: z2r tables have inconsistent nr");
    }
  }

  F_coeffs_flat_.clear();
  F_coeffs_flat_.reserve(n_species * nrho * TabulatedFunction::kCoefficientsPerCell);
  for (const auto& t : data.F_rho) {
    flatten_tab_into(t, F_coeffs_flat_);
  }

  rho_coeffs_flat_.clear();
  rho_coeffs_flat_.reserve(n_species * nr * TabulatedFunction::kCoefficientsPerCell);
  for (const auto& t : data.rho_r) {
    flatten_tab_into(t, rho_coeffs_flat_);
  }

  z2r_coeffs_flat_.clear();
  z2r_coeffs_flat_.reserve(npairs * nr * TabulatedFunction::kCoefficientsPerCell);
  for (const auto& t : data.z2r) {
    flatten_tab_into(t, z2r_coeffs_flat_);
  }
}

EamAlloyGpuAdapter::~EamAlloyGpuAdapter() = default;
EamAlloyGpuAdapter::EamAlloyGpuAdapter(EamAlloyGpuAdapter&&) noexcept = default;
EamAlloyGpuAdapter& EamAlloyGpuAdapter::operator=(EamAlloyGpuAdapter&&) noexcept = default;

std::uint64_t EamAlloyGpuAdapter::compute_version() const noexcept {
  return gpu_ ? gpu_->compute_version() : 0;
}

ForceResult EamAlloyGpuAdapter::compute(AtomSoA& atoms,
                                        const Box& box,
                                        const CellGrid& grid,
                                        tdmd::gpu::DevicePool& pool,
                                        tdmd::gpu::DeviceStream& stream) {
  ForceResult result;
  const std::size_t n = atoms.size();
  if (n == 0) {
    return result;
  }

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
  bp.cutoff = data_->cutoff;
  bp.skin = 0.0;  // EAM walks its own stencil; no skin needed.

  tdmd::gpu::EamAlloyTablesHost tables;
  tables.n_species = data_->species_names.size();
  tables.nrho = data_->F_rho[0].size();
  tables.nr = data_->rho_r[0].size();
  tables.npairs = tables.n_species * (tables.n_species + 1) / 2;
  tables.F_x0 = data_->F_rho[0].x0();
  tables.F_dx = data_->F_rho[0].dx();
  tables.r_x0 = data_->rho_r[0].x0();
  tables.r_dx = data_->rho_r[0].dx();
  tables.cutoff = data_->cutoff;
  tables.F_coeffs = F_coeffs_flat_.data();
  tables.rho_coeffs = rho_coeffs_flat_.data();
  tables.z2r_coeffs = z2r_coeffs_flat_.data();

  tdmd::gpu::EamAlloyGpuResult r = gpu_->compute(n,
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
