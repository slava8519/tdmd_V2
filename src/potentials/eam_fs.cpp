#include "tdmd/potentials/eam_fs.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

// Reference: LAMMPS `pair_eam_fs.cpp::compute` — fresh rewrite (no copied
// source), kept operation-for-operation aligned so T2.9 differential can
// achieve bit-match. See eam_alloy.cpp for the shared derivation.

namespace tdmd {

EamFsPotential::EamFsPotential(potentials::EamFsData data) : data_(std::move(data)) {
  if (data_.species_names.empty()) {
    throw std::invalid_argument("EamFsPotential: EamFsData has zero species");
  }
  const std::size_t n_species = data_.species_names.size();
  if (data_.F_rho.size() != n_species) {
    throw std::invalid_argument("EamFsPotential: F_rho count does not match N_species");
  }
  if (data_.rho_ij.size() != n_species * n_species) {
    throw std::invalid_argument("EamFsPotential: rho_ij count != N² (row-major)");
  }
  if (data_.z2r.size() != n_species * (n_species + 1) / 2) {
    throw std::invalid_argument("EamFsPotential: z2r table count != N·(N+1)/2");
  }
  if (!(data_.cutoff > 0.0)) {
    throw std::invalid_argument("EamFsPotential: cutoff must be strictly positive");
  }
}

ForceResult EamFsPotential::compute(AtomSoA& atoms, const NeighborList& neighbors, const Box& box) {
  ForceResult result;
  const std::size_t n = atoms.size();
  if (n == 0) {
    return result;
  }

  density_.assign(n, 0.0);
  dF_drho_.resize(n);

  const double* __restrict__ x_ptr = atoms.x.data();
  const double* __restrict__ y_ptr = atoms.y.data();
  const double* __restrict__ z_ptr = atoms.z.data();
  double* __restrict__ fx_ptr = atoms.fx.data();
  double* __restrict__ fy_ptr = atoms.fy.data();
  double* __restrict__ fz_ptr = atoms.fz.data();
  const SpeciesId* __restrict__ type_ptr = atoms.type.data();

  const auto& offsets = neighbors.page_offsets();
  const auto& ids = neighbors.neigh_ids();
  const double cutoff_sq = data_.cutoff * data_.cutoff;
  const std::size_t N = data_.species_names.size();

  // --- Pass 1: density (FS — per-ordered-pair table).
  //
  // `ρ[i]` gets the density seen at species `α=type_i` from a neighbour of
  // species `β=type_j`, which in the FS layout is `rho_ij[α·N + β]`.
  // `ρ[j]` gets the symmetric lookup with the roles reversed:
  // `rho_ij[β·N + α]`. In general `rho_ij[α·N+β] ≠ rho_ij[β·N+α]`.
  for (std::size_t i = 0; i < n; ++i) {
    const auto type_i = static_cast<std::size_t>(type_ptr[i]);
    const std::uint64_t begin = offsets[i];
    const std::uint64_t end = offsets[i + 1];
    for (std::uint64_t k = begin; k < end; ++k) {
      const std::uint32_t j = ids[k];
      const auto delta =
          box.unwrap_minimum_image(x_ptr[j] - x_ptr[i], y_ptr[j] - y_ptr[i], z_ptr[j] - z_ptr[i]);
      const double r2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
      if (r2 > cutoff_sq) {
        continue;
      }
      const double r = std::sqrt(r2);
      const auto type_j = static_cast<std::size_t>(type_ptr[j]);
      density_[i] += data_.rho_ij[potentials::EamFsData::rho_ij_index(type_i, type_j, N)].eval(r);
      density_[j] += data_.rho_ij[potentials::EamFsData::rho_ij_index(type_j, type_i, N)].eval(r);
    }
  }

  double pe = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto type_i = static_cast<std::size_t>(type_ptr[i]);
    const auto& F_tab = data_.F_rho[type_i];
    pe += F_tab.eval(density_[i]);
    dF_drho_[i] = F_tab.derivative(density_[i]);
  }

  // --- Pass 2: forces. Same scalar `dE/dr` as alloy, but the density-rate
  // lookups use the FS ordered-pair tables.
  double v_xx = 0.0;
  double v_yy = 0.0;
  double v_zz = 0.0;
  double v_xy = 0.0;
  double v_xz = 0.0;
  double v_yz = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const auto type_i = static_cast<std::size_t>(type_ptr[i]);
    const double dF_i = dF_drho_[i];
    const std::uint64_t begin = offsets[i];
    const std::uint64_t end = offsets[i + 1];
    for (std::uint64_t k = begin; k < end; ++k) {
      const std::uint32_t j = ids[k];
      const auto delta =
          box.unwrap_minimum_image(x_ptr[j] - x_ptr[i], y_ptr[j] - y_ptr[i], z_ptr[j] - z_ptr[i]);
      const double dx = delta[0];
      const double dy = delta[1];
      const double dz = delta[2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff_sq) {
        continue;
      }
      const double r = std::sqrt(r2);
      const double inv_r = 1.0 / r;
      const auto type_j = static_cast<std::size_t>(type_ptr[j]);

      const double drho_j_dr =
          data_.rho_ij[potentials::EamFsData::rho_ij_index(type_i, type_j, N)].derivative(r);
      const double drho_i_dr =
          data_.rho_ij[potentials::EamFsData::rho_ij_index(type_j, type_i, N)].derivative(r);
      const double dF_j = dF_drho_[j];

      const auto& z2r_tab = data_.z2r[potentials::EamFsData::pair_index(type_i, type_j)];
      const double z_val = z2r_tab.eval(r);
      const double z_deriv = z2r_tab.derivative(r);
      const double phi = z_val * inv_r;
      const double phi_prime = (z_deriv - phi) * inv_r;

      const double dE_dr = dF_i * drho_j_dr + dF_j * drho_i_dr + phi_prime;
      const double fscalar = dE_dr * inv_r;

      const double fx = fscalar * dx;
      const double fy = fscalar * dy;
      const double fz = fscalar * dz;

      fx_ptr[i] += fx;
      fy_ptr[i] += fy;
      fz_ptr[i] += fz;
      fx_ptr[j] -= fx;
      fy_ptr[j] -= fy;
      fz_ptr[j] -= fz;

      pe += phi;

      v_xx += fx * dx;
      v_yy += fy * dy;
      v_zz += fz * dz;
      v_xy += fx * dy;
      v_xz += fx * dz;
      v_yz += fy * dz;
    }
  }

  result.potential_energy = pe;
  result.virial[0] = v_xx;
  result.virial[1] = v_yy;
  result.virial[2] = v_zz;
  result.virial[3] = v_xy;
  result.virial[4] = v_xz;
  result.virial[5] = v_yz;
  return result;
}

}  // namespace tdmd
