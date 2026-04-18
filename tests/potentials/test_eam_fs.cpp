// SPEC: docs/specs/potentials/SPEC.md §4.2 (Finnis-Sinclair asymmetry),
// §4.3 (two-pass). Exec pack: docs/development/m2_execution_pack.md T2.7.
//
// Tests for EamFsPotential. Focus on the one numerical difference from the
// alloy variant — the ordered-pair ρ_{αβ}(r) density functions — so any
// row/column mis-indexing surfaces immediately. Toy tables are linear so the
// Hermite spline returns closed-form values at interior grid nodes.

#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/potentials/eam_fs.hpp"
#include "tdmd/potentials/tabulated.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using tdmd::AtomSoA;
using tdmd::Box;
using tdmd::CellGrid;
using tdmd::EamFsPotential;
using tdmd::ForceResult;
using tdmd::NeighborList;
using tdmd::potentials::EamFsData;
using tdmd::potentials::TabulatedFunction;

namespace {

constexpr int kNrho = 11;
constexpr double kDrho = 0.1;
constexpr int kNr = 11;
constexpr double kDr = 0.5;
constexpr double kCutoff = 5.0;

std::vector<double> linear_r_values(double scale) {
  std::vector<double> y;
  y.reserve(kNr);
  for (int k = 0; k < kNr; ++k) {
    y.push_back(scale * (1.0 - static_cast<double>(k) * kDr / kCutoff));
  }
  return y;
}

std::vector<double> F_values(double slope) {
  std::vector<double> y;
  y.reserve(kNrho);
  for (int k = 0; k < kNrho; ++k) {
    y.push_back(slope * static_cast<double>(k) * kDrho);
  }
  return y;
}

std::vector<double> z2r_values(double scale) {
  std::vector<double> y;
  y.reserve(kNr);
  for (int k = 0; k < kNr; ++k) {
    y.push_back(scale * (kCutoff - static_cast<double>(k) * kDr));
  }
  return y;
}

// Deliberately asymmetric ρ_ij: pick four distinct scales so any swap of
// indices α,β → β,α in either pass produces wrong numbers.
//   ρ_AA(r) = 1.0·(1 − r/5)        scale = 1.0
//   ρ_AB(r) = 3.0·(1 − r/5)        scale = 3.0   (density into A from B)
//   ρ_BA(r) = 7.0·(1 − r/5)        scale = 7.0   (density into B from A)
//   ρ_BB(r) = 11.0·(1 − r/5)       scale = 11.0
EamFsData make_toy_data_two_species() {
  EamFsData d;
  d.species_names = {"A", "B"};
  d.masses = {1.0, 2.0};
  d.nrho = kNrho;
  d.drho = kDrho;
  d.nr = kNr;
  d.dr = kDr;
  d.cutoff = kCutoff;
  d.F_rho.emplace_back(0.0, kDrho, F_values(-1.0));  // F_A = −ρ
  d.F_rho.emplace_back(0.0, kDrho, F_values(-2.0));  // F_B = −2ρ
  // Row-major α·N + β.
  d.rho_ij.emplace_back(0.0, kDr, linear_r_values(1.0));   // [0,0] = ρ_AA
  d.rho_ij.emplace_back(0.0, kDr, linear_r_values(3.0));   // [0,1] = ρ_AB
  d.rho_ij.emplace_back(0.0, kDr, linear_r_values(7.0));   // [1,0] = ρ_BA
  d.rho_ij.emplace_back(0.0, kDr, linear_r_values(11.0));  // [1,1] = ρ_BB
  // z2r lower-triangular: (0,0), (1,0), (1,1).
  d.z2r.emplace_back(0.0, kDr, z2r_values(1.0));
  d.z2r.emplace_back(0.0, kDr, z2r_values(2.0));
  d.z2r.emplace_back(0.0, kDr, z2r_values(3.0));
  return d;
}

EamFsData make_toy_data_single_species() {
  EamFsData d;
  d.species_names = {"A"};
  d.masses = {1.0};
  d.nrho = kNrho;
  d.drho = kDrho;
  d.nr = kNr;
  d.dr = kDr;
  d.cutoff = kCutoff;
  d.F_rho.emplace_back(0.0, kDrho, F_values(-1.0));
  d.rho_ij.emplace_back(0.0, kDr, linear_r_values(1.0));
  d.z2r.emplace_back(0.0, kDr, z2r_values(1.0));
  return d;
}

Box make_cubic_box(double length) {
  Box b;
  b.xhi = length;
  b.yhi = length;
  b.zhi = length;
  b.periodic_x = true;
  b.periodic_y = true;
  b.periodic_z = true;
  return b;
}

struct Dimer {
  AtomSoA atoms;
  Box box;
  CellGrid grid;
  NeighborList list;
};

Dimer make_dimer(tdmd::SpeciesId type_a, tdmd::SpeciesId type_b, double separation) {
  Dimer d;
  d.box = make_cubic_box(30.0);
  const double cx = 0.5 * d.box.lx();
  const double cy = 0.5 * d.box.ly();
  const double cz = 0.5 * d.box.lz();
  d.atoms.add_atom(type_a, cx - 0.5 * separation, cy, cz);
  d.atoms.add_atom(type_b, cx + 0.5 * separation, cy, cz);
  const double skin = 0.3;
  d.grid.build(d.box, kCutoff, skin);
  d.grid.bin(d.atoms);
  d.list.build(d.atoms, d.box, d.grid, kCutoff, skin);
  return d;
}

void zero_forces(AtomSoA& atoms) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.fx[i] = 0.0;
    atoms.fy[i] = 0.0;
    atoms.fz[i] = 0.0;
  }
}

}  // namespace

TEST_CASE("EamFsPotential ctor rejects inconsistent data", "[potentials][eam][fs][ctor]") {
  REQUIRE_NOTHROW(EamFsPotential(make_toy_data_single_species()));

  {
    auto d = make_toy_data_single_species();
    d.species_names.clear();
    REQUIRE_THROWS_AS(EamFsPotential(std::move(d)), std::invalid_argument);
  }
  {
    auto d = make_toy_data_single_species();
    d.rho_ij.emplace_back(0.0, kDr, linear_r_values(1.0));  // N² = 1 expected
    REQUIRE_THROWS_AS(EamFsPotential(std::move(d)), std::invalid_argument);
  }
  {
    auto d = make_toy_data_two_species();
    d.rho_ij.pop_back();  // would break N² packing
    REQUIRE_THROWS_AS(EamFsPotential(std::move(d)), std::invalid_argument);
  }
  {
    auto d = make_toy_data_single_species();
    d.cutoff = -1.0;
    REQUIRE_THROWS_AS(EamFsPotential(std::move(d)), std::invalid_argument);
  }
}

TEST_CASE("EamFsPotential: two-atom A-B picks up ordered-pair densities",
          "[potentials][eam][fs][unit]") {
  // A–B pair at r = 3.0 (interior grid node).
  //
  // Expected lookups (rho_ij indexed α·N + β):
  //   ρ_i (i=A) gets ρ_{A,B}(r=3) = 3.0·(1 − 3/5) = 3.0·0.4 = 1.2
  //   ρ_j (j=B) gets ρ_{B,A}(r=3) = 7.0·(1 − 3/5) = 7.0·0.4 = 2.8
  //   ρ_j = 2.8 exceeds the F-grid (rho_max = 1.0) — clamp would hit.
  //   → scale F tables only weakly so ρ stays in range: use smaller r.
  //
  // r = 4.5 (grid node k=9): ρ_{AB}(4.5) = 3.0·(1 − 0.9) = 0.3
  //                          ρ_{BA}(4.5) = 7.0·0.1 = 0.7
  // Both < 1.0, safe.
  //
  //   ρ_i = 0.3, ρ_j = 0.7
  //   F_A(0.3) = −0.3, F_B(0.7) = −1.4, embedding PE = −1.7
  //   dF_i = −1, dF_j = −2
  //   ρ_{A,B}'(r) = −3.0/5 = −0.6,  ρ_{B,A}'(r) = −7.0/5 = −1.4
  //   drho_j_dr = ρ_{A,B}'(r) = −0.6  (density into i=A from j=B as r varies)
  //   drho_i_dr = ρ_{B,A}'(r) = −1.4  (density into j=B from i=A)
  //   z2r_{AB}(r) = 2.0·(5 − 4.5) = 1.0
  //   z2r'(r) = −2.0
  //   φ = 1/4.5 = 2/9;  φ' = (−2 − 2/9)/4.5 = (−20/9)/(9/2) = −40/81
  //   pair PE = φ = 2/9
  //   dE/dr = (−1)·(−0.6) + (−2)·(−1.4) + (−40/81)
  //         = 0.6 + 2.8 − 40/81 = 3.4 − 0.49382716…
  //   fscalar = dE/dr / 4.5
  //   F_x on i = fscalar · (r_j − r_i)_x = fscalar · 4.5 = dE/dr
  const double r_sep = 4.5;
  const double rho_i = 0.3;
  const double rho_j = 0.7;
  const double dF_i = -1.0;
  const double dF_j = -2.0;
  const double drho_AB_dr = -3.0 / kCutoff;  // −0.6
  const double drho_BA_dr = -7.0 / kCutoff;  // −1.4
  const double z_val = 2.0 * (kCutoff - r_sep);
  const double z_deriv = -2.0;
  const double phi = z_val / r_sep;
  const double phi_prime = (z_deriv - phi) / r_sep;
  const double expected_total_pe = -rho_i + -2.0 * rho_j + phi;
  const double dE_dr = dF_i * drho_AB_dr + dF_j * drho_BA_dr + phi_prime;
  const double fscalar = dE_dr / r_sep;
  const double expected_fx_i = fscalar * r_sep;

  EamFsPotential pot(make_toy_data_two_species());
  Dimer d = make_dimer(0, 1, r_sep);
  zero_forces(d.atoms);
  const ForceResult r = pot.compute(d.atoms, d.list, d.box);

  constexpr double kTol = 1e-12;
  REQUIRE(pot.density()[0] == Catch::Approx(rho_i).margin(kTol));
  REQUIRE(pot.density()[1] == Catch::Approx(rho_j).margin(kTol));
  REQUIRE(pot.dF_drho()[0] == Catch::Approx(dF_i).margin(kTol));
  REQUIRE(pot.dF_drho()[1] == Catch::Approx(dF_j).margin(kTol));
  REQUIRE(r.potential_energy == Catch::Approx(expected_total_pe).margin(kTol));
  REQUIRE(d.atoms.fx[0] == Catch::Approx(expected_fx_i).margin(kTol));
  REQUIRE(d.atoms.fx[1] == Catch::Approx(-expected_fx_i).margin(kTol));
  // Confirm the asymmetry by swapping types: if the kernel used ρ_{B,A}
  // for atom i (instead of ρ_{A,B}), we'd get density = 0.7 instead of 0.3.
  REQUIRE(pot.density()[0] != Catch::Approx(0.7));
}

TEST_CASE("EamFsPotential: single-species degenerates to alloy on same data",
          "[potentials][eam][fs][unit]") {
  // ρ_{A,A} is just ρ_A; with one species FS and alloy must give identical
  // forces. This guards against any FS-only indexing bug affecting the
  // symmetric case.
  EamFsPotential pot(make_toy_data_single_species());
  Dimer d = make_dimer(0, 0, 1.0);
  zero_forces(d.atoms);
  const ForceResult r = pot.compute(d.atoms, d.list, d.box);

  // Numeric expectations mirror the alloy single-species test
  // (ρ_A(1)=0.8, F=−0.8, z2r(1)=4, φ=4, φ'=−5, dE/dr=−4.6).
  constexpr double kTol = 1e-12;
  REQUIRE(pot.density()[0] == Catch::Approx(0.8).margin(kTol));
  REQUIRE(r.potential_energy == Catch::Approx(-1.6 + 4.0).margin(kTol));
  REQUIRE(d.atoms.fx[0] == Catch::Approx(-4.6).margin(kTol));
  REQUIRE(d.atoms.fx[1] == Catch::Approx(4.6).margin(kTol));
}

TEST_CASE("EamFsPotential: Newton's 3rd law over ≥10⁴ random configurations",
          "[potentials][eam][fs][property]") {
  EamFsPotential pot(make_toy_data_two_species());

  std::mt19937 rng(0xc0ffee);
  std::uniform_real_distribution<double> pos_dist(0.0, 20.0);
  std::uniform_int_distribution<unsigned> type_dist(0, 1);

  constexpr std::size_t kConfigs = 10'000;
  constexpr std::size_t kNAtoms = 16;
  constexpr double kBoxLen = 20.0;
  constexpr double kSkin = 0.3;

  Box box = make_cubic_box(kBoxLen);

  double max_residual = 0.0;
  for (std::size_t cfg = 0; cfg < kConfigs; ++cfg) {
    AtomSoA atoms;
    for (std::size_t i = 0; i < kNAtoms; ++i) {
      const tdmd::SpeciesId t = type_dist(rng);
      atoms.add_atom(t, pos_dist(rng), pos_dist(rng), pos_dist(rng));
    }
    CellGrid grid;
    NeighborList list;
    grid.build(box, kCutoff, kSkin);
    grid.bin(atoms);
    list.build(atoms, box, grid, kCutoff, kSkin);
    zero_forces(atoms);
    (void) pot.compute(atoms, list, box);

    double sum_fx = 0.0;
    double sum_fy = 0.0;
    double sum_fz = 0.0;
    double max_abs_f = 0.0;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      sum_fx += atoms.fx[i];
      sum_fy += atoms.fy[i];
      sum_fz += atoms.fz[i];
      max_abs_f = std::max(max_abs_f,
                           std::abs(atoms.fx[i]) + std::abs(atoms.fy[i]) + std::abs(atoms.fz[i]));
    }
    const double residual = std::abs(sum_fx) + std::abs(sum_fy) + std::abs(sum_fz);
    const double rel = (max_abs_f > 0.0) ? residual / max_abs_f : residual;
    max_residual = std::max(max_residual, rel);
  }
  REQUIRE(max_residual < 1e-12);
}
