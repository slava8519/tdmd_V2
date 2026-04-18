// SPEC: docs/specs/potentials/SPEC.md §4.1–§4.3 (EAM/alloy form, two-pass).
// Exec pack: docs/development/m2_execution_pack.md T2.7.
//
// Unit + property tests for EamAlloyPotential. Uses hand-authored linear
// lookup tables so eval/derivative return analytic values at interior grid
// nodes — the Hermite cubic spline is exact for polynomials of degree ≤ 3
// (see T2.5 test suite for the underlying demonstration). This lets us
// verify F_i, F_j, ρ_i, ρ_j, and E against closed-form expressions without
// spline-interpolation error entering the assertions.

#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/potentials/eam_alloy.hpp"
#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/potentials/tabulated.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

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
using tdmd::EamAlloyPotential;
using tdmd::ForceResult;
using tdmd::NeighborList;
using tdmd::potentials::EamAlloyData;
using tdmd::potentials::TabulatedFunction;

namespace {

// --- Toy linear tables ----------------------------------------------------
//
// On uniform grids Hermite-cubic-with-4th-order-FD reproduces linear input
// exactly (value and derivative) — tests below depend on this.

constexpr int kNrho = 11;  // rho ∈ [0, 1.0]
constexpr double kDrho = 0.1;
constexpr int kNr = 11;  // r ∈ [0, 5.0]
constexpr double kDr = 0.5;
constexpr double kCutoff = 5.0;

// ρ(r) = 1 − r/cutoff — decays linearly to zero at cutoff.
std::vector<double> rho_of_r_values() {
  std::vector<double> y;
  y.reserve(kNr);
  for (int k = 0; k < kNr; ++k) {
    y.push_back(1.0 - static_cast<double>(k) * kDr / kCutoff);
  }
  return y;
}

// ρ_B(r) = 2·(1 − r/cutoff) — distinct from ρ_A so a species swap bug
// surfaces immediately (ρ_i and ρ_j pick up visibly different values).
std::vector<double> rho_of_r_values_b() {
  std::vector<double> y;
  y.reserve(kNr);
  for (int k = 0; k < kNr; ++k) {
    y.push_back(2.0 * (1.0 - static_cast<double>(k) * kDr / kCutoff));
  }
  return y;
}

// F_A(ρ) = −ρ — linear embedding, F'(ρ) = −1 everywhere.
std::vector<double> F_of_rho_values_a() {
  std::vector<double> y;
  y.reserve(kNrho);
  for (int k = 0; k < kNrho; ++k) {
    y.push_back(-static_cast<double>(k) * kDrho);
  }
  return y;
}

// F_B(ρ) = −2·ρ — distinct embedding slope.
std::vector<double> F_of_rho_values_b() {
  std::vector<double> y;
  y.reserve(kNrho);
  for (int k = 0; k < kNrho; ++k) {
    y.push_back(-2.0 * static_cast<double>(k) * kDrho);
  }
  return y;
}

// z2r(r) = cutoff − r — linear, so φ(r) = (cutoff − r)/r.
std::vector<double> z2r_values(double scale) {
  std::vector<double> y;
  y.reserve(kNr);
  for (int k = 0; k < kNr; ++k) {
    y.push_back(scale * (kCutoff - static_cast<double>(k) * kDr));
  }
  return y;
}

EamAlloyData make_toy_data_single_species() {
  EamAlloyData d;
  d.species_names = {"A"};
  d.masses = {1.0};
  d.nrho = kNrho;
  d.drho = kDrho;
  d.nr = kNr;
  d.dr = kDr;
  d.cutoff = kCutoff;
  d.F_rho.emplace_back(0.0, kDrho, F_of_rho_values_a());
  d.rho_r.emplace_back(0.0, kDr, rho_of_r_values());
  d.z2r.emplace_back(0.0, kDr, z2r_values(1.0));
  return d;
}

EamAlloyData make_toy_data_two_species() {
  EamAlloyData d;
  d.species_names = {"A", "B"};
  d.masses = {1.0, 2.0};
  d.nrho = kNrho;
  d.drho = kDrho;
  d.nr = kNr;
  d.dr = kDr;
  d.cutoff = kCutoff;
  d.F_rho.emplace_back(0.0, kDrho, F_of_rho_values_a());
  d.F_rho.emplace_back(0.0, kDrho, F_of_rho_values_b());
  d.rho_r.emplace_back(0.0, kDr, rho_of_r_values());
  d.rho_r.emplace_back(0.0, kDr, rho_of_r_values_b());
  // 3 z2r entries for N=2: (0,0), (1,0), (1,1) — scale factors distinguish them.
  d.z2r.emplace_back(0.0, kDr, z2r_values(1.0));  // AA
  d.z2r.emplace_back(0.0, kDr, z2r_values(2.0));  // BA
  d.z2r.emplace_back(0.0, kDr, z2r_values(3.0));  // BB
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

TEST_CASE("EamAlloyPotential ctor rejects inconsistent data", "[potentials][eam][alloy][ctor]") {
  REQUIRE_NOTHROW(EamAlloyPotential(make_toy_data_single_species()));

  {
    auto d = make_toy_data_single_species();
    d.species_names.clear();
    REQUIRE_THROWS_AS(EamAlloyPotential(std::move(d)), std::invalid_argument);
  }
  {
    auto d = make_toy_data_single_species();
    d.F_rho.emplace_back(0.0, kDrho, F_of_rho_values_b());  // extra F_rho
    REQUIRE_THROWS_AS(EamAlloyPotential(std::move(d)), std::invalid_argument);
  }
  {
    auto d = make_toy_data_single_species();
    d.z2r.emplace_back(0.0, kDr, z2r_values(1.0));  // should have exactly 1
    REQUIRE_THROWS_AS(EamAlloyPotential(std::move(d)), std::invalid_argument);
  }
  {
    auto d = make_toy_data_single_species();
    d.cutoff = 0.0;
    REQUIRE_THROWS_AS(EamAlloyPotential(std::move(d)), std::invalid_argument);
  }
}

TEST_CASE("EamAlloyPotential accessors", "[potentials][eam][alloy][ctor]") {
  EamAlloyPotential pot(make_toy_data_single_species());
  REQUIRE(pot.cutoff() == kCutoff);
  REQUIRE(pot.name() == "eam/alloy");
  REQUIRE(pot.effective_skin() == Catch::Approx(0.05 * kCutoff));
  REQUIRE(pot.data().species_names.size() == 1);
  REQUIRE(pot.density().empty());  // no compute yet
}

TEST_CASE("EamAlloyPotential: single atom has ρ=0 and F(0) contributes no force",
          "[potentials][eam][alloy][unit]") {
  // ρ_i = 0 (no neighbours), F(0) = 0 with our toy F(ρ) = −ρ, so PE = 0 and
  // forces must be exactly zero.
  EamAlloyPotential pot(make_toy_data_single_species());
  AtomSoA atoms;
  Box box = make_cubic_box(30.0);
  atoms.add_atom(0, 15.0, 15.0, 15.0);
  CellGrid grid;
  NeighborList list;
  grid.build(box, kCutoff, 0.3);
  grid.bin(atoms);
  list.build(atoms, box, grid, kCutoff, 0.3);

  zero_forces(atoms);
  const ForceResult r = pot.compute(atoms, list, box);

  REQUIRE(pot.density().size() == 1);
  REQUIRE(pot.density()[0] == 0.0);
  REQUIRE(pot.dF_drho()[0] == -1.0);  // F'(0) = −1 for F(ρ) = −ρ
  REQUIRE(r.potential_energy == 0.0);
  REQUIRE(atoms.fx[0] == 0.0);
  REQUIRE(atoms.fy[0] == 0.0);
  REQUIRE(atoms.fz[0] == 0.0);
  for (const double w : r.virial) {
    REQUIRE(w == 0.0);
  }
}

TEST_CASE("EamAlloyPotential: two-atom same-species closed-form forces",
          "[potentials][eam][alloy][unit]") {
  // Place A–A pair at r = 1.0 (interior grid node for dr = 0.5).
  // Analytic expectations (using linear tables that the spline reproduces
  // exactly at grid nodes):
  //   ρ_A(1.0) = 1 − 1/5 = 0.8   ρ_i = ρ_j = 0.8
  //   F_A(0.8) = −0.8            → embedding PE = −1.6
  //   F_A'(ρ)  = −1              → dF_i = dF_j = −1
  //   ρ_A'(r)  = −1/5 = −0.2
  //   z2r(1.0) = 4.0             φ = z/r = 4.0
  //   z2r'(r)  = −1              φ' = (z' − φ)/r = (−1 − 4.0)/1 = −5.0
  //   dE/dr    = dF_i·ρ' + dF_j·ρ' + φ' = (−1)(−0.2) + (−1)(−0.2) + (−5.0)
  //            = 0.4 − 5.0 = −4.6
  //   fscalar  = dE/dr / r = −4.6
  //   F_i = (−4.6, 0, 0), F_j = (4.6, 0, 0); total E = −1.6 + 4.0 = 2.4
  EamAlloyPotential pot(make_toy_data_single_species());
  Dimer d = make_dimer(0, 0, 1.0);
  zero_forces(d.atoms);
  const ForceResult r = pot.compute(d.atoms, d.list, d.box);

  constexpr double kTol = 1e-12;
  REQUIRE(pot.density()[0] == Catch::Approx(0.8).margin(kTol));
  REQUIRE(pot.density()[1] == Catch::Approx(0.8).margin(kTol));
  REQUIRE(pot.dF_drho()[0] == Catch::Approx(-1.0).margin(kTol));
  REQUIRE(r.potential_energy == Catch::Approx(2.4).margin(kTol));
  REQUIRE(d.atoms.fx[0] == Catch::Approx(-4.6).margin(kTol));
  REQUIRE(d.atoms.fx[1] == Catch::Approx(4.6).margin(kTol));
  REQUIRE(d.atoms.fy[0] == Catch::Approx(0.0).margin(kTol));
  REQUIRE(d.atoms.fz[0] == Catch::Approx(0.0).margin(kTol));

  // Newton's 3rd law.
  REQUIRE(d.atoms.fx[0] + d.atoms.fx[1] == Catch::Approx(0.0).margin(kTol));

  // Virial: only W_xx is non-zero (pair along +x axis).
  // W_xx = fx_i · dx = (−4.6)·(1.0) = −4.6 where dx = x_j − x_i = +1.0.
  REQUIRE(r.virial[0] == Catch::Approx(-4.6).margin(kTol));
  REQUIRE(r.virial[1] == Catch::Approx(0.0).margin(kTol));
  REQUIRE(r.virial[2] == Catch::Approx(0.0).margin(kTol));
  REQUIRE(r.virial[3] == Catch::Approx(0.0).margin(kTol));
  REQUIRE(r.virial[4] == Catch::Approx(0.0).margin(kTol));
  REQUIRE(r.virial[5] == Catch::Approx(0.0).margin(kTol));
}

TEST_CASE("EamAlloyPotential: two-atom A-B asymmetric species", "[potentials][eam][alloy][unit]") {
  // A–B pair at r = 1.0.
  //   ρ_A(r) = 1 − r/5;  ρ_B(r) = 2(1 − r/5).   At r=1: ρ_A = 0.8, ρ_B = 1.6.
  //   But our grid rho only extends to rho=1.0 (nrho=11, drho=0.1 → max ρ=1.0).
  //   ρ_j = ρ_B at B-→A contribution... wait careful:
  //     ρ_i (i=A) += ρ_B(r) = 1.6   ← B is neighbour, so B's density function.
  //     ρ_j (j=B) += ρ_A(r) = 0.8
  //   ρ_i = 1.6 exceeds the F-grid (rho_max = 1.0). TabulatedFunction clamps
  //   to boundary, so F_A(1.6) → F_A at rho_max = −1.0, F_A'(1.6) → slope at
  //   the end cell at p = 1 (endpoint clamp): for a linear F this is just the
  //   last-cell interior derivative = −1 (linear slope is exact at all cells).
  // To avoid the clamp, choose smaller separation so ρ stays in range.
  //
  // At r = 3.0 (grid node k=6, interior): ρ_A(3.0) = 1 − 0.6 = 0.4;
  //   ρ_B(3.0) = 0.8. Both in range.
  //   ρ_i (A) += ρ_B(3.0) = 0.8   ρ_j (B) += ρ_A(3.0) = 0.4
  //   F_A(0.8) = −0.8   F_B(0.4) = −0.8   embedding PE = −1.6
  //   F_A'(ρ) = −1   F_B'(ρ) = −2
  //   ρ_A'(r) = −0.2   ρ_B'(r) = −0.4
  //   drho_j_dr = ρ_B'(r) = −0.4  (density from j=B affecting i=A)
  //   drho_i_dr = ρ_A'(r) = −0.2  (density from i=A affecting j=B)
  //   z2r for pair (A,B) with lower-tri packing: pair_index(0,1) = 1.
  //     z2r_{AB}(r) = 2.0·(cutoff − r),  z2r_{AB}(3.0) = 2.0·2.0 = 4.0
  //     z2r_{AB}'(r) = −2.0
  //     φ = z/r = 4/3;  φ' = (z' − φ)/r = (−2 − 4/3) / 3 = −10/9
  //   pair PE = φ = 4/3
  //   dE/dr = dF_i · drho_j_dr + dF_j · drho_i_dr + φ'
  //         = (−1)·(−0.4) + (−2)·(−0.2) + (−10/9)
  //         = 0.4 + 0.4 − 10/9 = 0.8 − 1.111… = −0.311…
  //   fscalar = dE/dr / r = −0.311…/3 ≈ −0.1037…
  //   F_i = fscalar · Δ where Δ = (3, 0, 0): F_i,x = 3·fscalar ≈ −0.3111…
  //
  // Express expectation symbolically to avoid manual rounding error.
  const double r_sep = 3.0;
  const double rho_i = 0.8;         // ρ_A sees ρ_B(3) = 0.8 (from toy tables)
  const double rho_j = 0.4;         // ρ_B sees ρ_A(3) = 0.4
  const double F_i = -rho_i;        // F_A = −ρ
  const double F_j = -2.0 * rho_j;  // F_B = −2ρ
  const double dF_i = -1.0;
  const double dF_j = -2.0;
  const double drho_A_dr = -0.2;
  const double drho_B_dr = -0.4;
  const double z_val = 2.0 * (kCutoff - r_sep);
  const double z_deriv = -2.0;
  const double phi = z_val / r_sep;
  const double phi_prime = (z_deriv - phi) / r_sep;
  const double pair_pe = phi;
  const double dE_dr = dF_i * drho_B_dr + dF_j * drho_A_dr + phi_prime;
  const double fscalar = dE_dr / r_sep;
  const double expected_total_pe = F_i + F_j + pair_pe;
  const double expected_fx_i = fscalar * r_sep;  // Δ = (r_sep, 0, 0)

  EamAlloyPotential pot(make_toy_data_two_species());
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
  REQUIRE(r.virial[0] == Catch::Approx(expected_fx_i * r_sep).margin(kTol));
}

TEST_CASE("EamAlloyPotential: Newton's 3rd law over ≥10⁴ random configurations",
          "[potentials][eam][alloy][property]") {
  // Σ F over all atoms must be zero to FP noise. The half-list kernel
  // applies F_i += Δ, F_j -= Δ per pair, so the sum telescopes to zero
  // algebraically. We verify this against accumulated rounding error only.
  EamAlloyPotential pot(make_toy_data_two_species());

  std::mt19937 rng(0xdeadbeef);
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
  // Floating-point telescoping error scales with N·pairs·ε — empirically
  // well under 1e-12 for our config sizes.
  REQUIRE(max_residual < 1e-12);
}
