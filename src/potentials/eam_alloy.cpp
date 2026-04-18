#include "tdmd/potentials/eam_alloy.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

// Reference: LAMMPS `pair_eam.cpp::compute` and `pair_eam_alloy.cpp` describe
// the same two-pass algorithm and per-pair FP operation sequence. The TDMD
// implementation is a fresh rewrite — no source is copied. LAMMPS is used
// here only as an algorithmic / numerical reference, preserved so that the
// T2.9 differential harness can achieve `1e-10` relative force agreement.

namespace tdmd {

EamAlloyPotential::EamAlloyPotential(potentials::EamAlloyData data) : data_(std::move(data)) {
  if (data_.species_names.empty()) {
    throw std::invalid_argument("EamAlloyPotential: EamAlloyData has zero species");
  }
  const std::size_t n_species = data_.species_names.size();
  if (data_.F_rho.size() != n_species || data_.rho_r.size() != n_species) {
    throw std::invalid_argument(
        "EamAlloyPotential: F_rho / rho_r table counts do not match N_species");
  }
  if (data_.z2r.size() != n_species * (n_species + 1) / 2) {
    throw std::invalid_argument("EamAlloyPotential: z2r table count != N·(N+1)/2");
  }
  if (!(data_.cutoff > 0.0)) {
    throw std::invalid_argument("EamAlloyPotential: cutoff must be strictly positive");
  }
}

ForceResult EamAlloyPotential::compute(AtomSoA& atoms,
                                       const NeighborList& neighbors,
                                       const Box& box) {
  ForceResult result;
  const std::size_t n = atoms.size();
  if (n == 0) {
    return result;
  }

  // Grow-once / zero scratch. `assign` overwrites prior step's density so
  // Pass 1 can `+=` directly.
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

  // --- Pass 1: density accumulation.
  //
  // Half-list gives each pair once with `j > i` (newton on). For this pair:
  //   ρ[i] += ρ_{β}(r)   β = species of j
  //   ρ[j] += ρ_{α}(r)   α = species of i
  // The two table lookups are identical only when type_i == type_j; single-
  // species builds degenerate to one lookup per pair, multi-species costs
  // two. This mirrors LAMMPS's `rhor_spline[type2rhor[type[j]][type[i]]]`
  // indexing.
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
      density_[i] += data_.rho_r[type_j].eval(r);
      density_[j] += data_.rho_r[type_i].eval(r);
    }
  }

  // --- Transition: embedding PE + F'(ρ) cache.
  //
  // Split from Pass 2 so `dF_drho_[j]` is valid for every atom `j` when the
  // force-pair loop reads it — the pair loop accesses both endpoints of
  // each pair. Per-atom embedding PE summed here once so the Pass-2 loop
  // accumulates only the pair-part PE.
  double pe = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto type_i = static_cast<std::size_t>(type_ptr[i]);
    const auto& F_tab = data_.F_rho[type_i];
    pe += F_tab.eval(density_[i]);
    dF_drho_[i] = F_tab.derivative(density_[i]);
  }

  // --- Pass 2: per-pair forces + pair PE + virial.
  //
  // Per SPEC §4.3 / LAMMPS convention the scalar radial derivative of the
  // total pair energy is
  //   dE/dr = F'(ρ_i) · ρ_β'(r) + F'(ρ_j) · ρ_α'(r) + φ'(r)
  // with φ(r) = z2r(r)/r, φ'(r) = (z2r'(r) − φ) / r. The vector force on
  // atom i from pair (i, j) is then
  //   F_i = (dE/dr) · Δ / r,    Δ = r_j − r_i
  // which matches Morse's `factor = g/r; fx = factor · dx` convention.
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

      // Density derivatives: `drho_j_dr` is the rate at which ρ_i changes
      // as r_ij varies (neighbor of species β pushing density into i);
      // `drho_i_dr` is the symmetric rate into ρ_j. Names follow the
      // "affects atom X" convention, not the species label.
      const double drho_j_dr = data_.rho_r[type_j].derivative(r);
      const double drho_i_dr = data_.rho_r[type_i].derivative(r);
      const double dF_j = dF_drho_[j];

      // Pair part via z2r: φ = z/r, φ' = (z' − φ)/r.
      const auto& z2r_tab = data_.z2r[potentials::EamAlloyData::pair_index(type_i, type_j)];
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

      // Clausius virial. Force on i paired with vector i→j; identical to
      // Morse (see morse.cpp for the ij vs ji sign rationale).
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
