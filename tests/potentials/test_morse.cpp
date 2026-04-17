// SPEC: docs/specs/potentials/SPEC.md §3 (Morse), §2.4 (cutoff)
// Exec pack: docs/development/m1_execution_pack.md T1.8
//
// Unit / property tests for MorsePotential. Two strategies are exercised:
//   - HardCutoff (Strategy A) for the exec-pack invariant "F(r₀) = 0 within 1e-12"
//     and for the ≥10³ property test `F = -dE/dr`. With Strategy A, U is the
//     raw Morse energy so the derivative is clean.
//   - ShiftedForce (Strategy C) for the SPEC §2.4.2 production default; we check
//     continuity at r_c (E → 0 and F → 0 from the inside).
//
// Strategy B (shifted-energy, LAMMPS `pair_style morse` default) is intentionally
// NOT implemented in M1; OQ-M1-1 will be revisited at T1.11 when the LAMMPS diff
// harness is wired up.

#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

// Girifalco-Weizer Al (1959) parameters — the fixture carried in
// tests/potentials/fixtures/morse_al.yaml. Kept as constants here so the test
// executable stays hermetic (no YAML dep on this layer yet — arrives with T1.4).
constexpr double kD = 0.2703;      // eV
constexpr double kAlpha = 1.1646;  // 1/Å
constexpr double kR0 = 3.253;      // Å
constexpr double kCutoff = 8.0;    // Å
constexpr double kSkin = 0.3;      // Å — same default as T1.6 neighbor list

tdmd::Box make_cubic_box(double length) {
  tdmd::Box box;
  box.xhi = length;
  box.yhi = length;
  box.zhi = length;
  box.periodic_x = true;
  box.periodic_y = true;
  box.periodic_z = true;
  return box;
}

struct Dimer {
  tdmd::AtomSoA atoms;
  tdmd::Box box;
  tdmd::CellGrid grid;
  tdmd::NeighborList list;
};

// Builds a two-atom periodic system where the pair is aligned along +x at the
// requested separation. Box is big enough that the minimum image is the
// intended direct pair (30 Å > 2·(cutoff + skin) + margin).
Dimer make_dimer(double separation, double cutoff = kCutoff, double skin = kSkin) {
  Dimer s;
  s.box = make_cubic_box(30.0);
  const double cx = 0.5 * s.box.lx();
  const double cy = 0.5 * s.box.ly();
  const double cz = 0.5 * s.box.lz();
  s.atoms.add_atom(0, cx - 0.5 * separation, cy, cz);
  s.atoms.add_atom(0, cx + 0.5 * separation, cy, cz);
  s.grid.build(s.box, cutoff, skin);
  s.grid.bin(s.atoms);
  s.list.build(s.atoms, s.box, s.grid, cutoff, skin);
  return s;
}

void zero_forces(tdmd::AtomSoA& atoms) {
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atoms.fx[i] = 0.0;
    atoms.fy[i] = 0.0;
    atoms.fz[i] = 0.0;
  }
}

tdmd::MorsePotential make_pot(
    tdmd::MorsePotential::CutoffStrategy s = tdmd::MorsePotential::CutoffStrategy::ShiftedForce) {
  return tdmd::MorsePotential(tdmd::MorsePotential::PairParams{kD, kAlpha, kR0, kCutoff}, s);
}

}  // namespace

TEST_CASE("MorsePotential constructor rejects invalid parameters", "[potentials][morse][ctor]") {
  using MP = tdmd::MorsePotential;
  const MP::PairParams good{kD, kAlpha, kR0, kCutoff};
  REQUIRE_NOTHROW(MP{good});

  auto mutated = [&](auto mut) {
    MP::PairParams p = good;
    mut(p);
    return p;
  };

  REQUIRE_THROWS_AS(MP{mutated([](auto& p) { p.D = 0.0; })}, std::invalid_argument);
  REQUIRE_THROWS_AS(MP{mutated([](auto& p) { p.D = -1.0; })}, std::invalid_argument);
  REQUIRE_THROWS_AS(MP{mutated([](auto& p) { p.alpha = 0.0; })}, std::invalid_argument);
  REQUIRE_THROWS_AS(MP{mutated([](auto& p) { p.alpha = -0.5; })}, std::invalid_argument);
  REQUIRE_THROWS_AS(MP{mutated([](auto& p) { p.r0 = 0.0; })}, std::invalid_argument);
  REQUIRE_THROWS_AS(MP{mutated([](auto& p) { p.cutoff = p.r0; })}, std::invalid_argument);
  REQUIRE_THROWS_AS(MP{mutated([](auto& p) { p.cutoff = p.r0 - 0.1; })}, std::invalid_argument);
  REQUIRE_THROWS_AS(MP{mutated([](auto& p) { p.D = std::numeric_limits<double>::quiet_NaN(); })},
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      MP{mutated([](auto& p) { p.cutoff = std::numeric_limits<double>::infinity(); })},
      std::invalid_argument);
}

TEST_CASE("MorsePotential accessors expose construction state", "[potentials][morse][ctor]") {
  using MP = tdmd::MorsePotential;
  const MP::PairParams params{kD, kAlpha, kR0, kCutoff};
  const MP pot_default{params};
  REQUIRE(pot_default.strategy() == MP::CutoffStrategy::ShiftedForce);
  REQUIRE(pot_default.cutoff() == kCutoff);
  REQUIRE(pot_default.params().D == kD);
  REQUIRE(pot_default.name() == "morse");
  REQUIRE(pot_default.effective_skin() == Catch::Approx(0.05 * kCutoff));

  const MP pot_hard{params, MP::CutoffStrategy::HardCutoff};
  REQUIRE(pot_hard.strategy() == MP::CutoffStrategy::HardCutoff);
}

TEST_CASE("MorsePotential HardCutoff: force is exactly zero at r = r0",
          "[potentials][morse][hard]") {
  // Exec pack invariant: with a hard cutoff the well bottom is sharp — F(r₀) = 0
  // exactly (analytically), and numerically must be < 1e-12.
  auto s = make_dimer(kR0);
  auto pot = make_pot(tdmd::MorsePotential::CutoffStrategy::HardCutoff);
  zero_forces(s.atoms);
  const auto result = pot.compute(s.atoms, s.list, s.box);

  REQUIRE(std::abs(s.atoms.fx[0]) < 1e-12);
  REQUIRE(std::abs(s.atoms.fy[0]) < 1e-12);
  REQUIRE(std::abs(s.atoms.fz[0]) < 1e-12);
  REQUIRE(std::abs(s.atoms.fx[1]) < 1e-12);
  REQUIRE(std::abs(s.atoms.fy[1]) < 1e-12);
  REQUIRE(std::abs(s.atoms.fz[1]) < 1e-12);
  // E_pair(r₀) = D · 0² - D = -D (well depth, raw Morse with -D offset).
  REQUIRE(result.potential_energy == Catch::Approx(-kD).margin(1e-14));
}

TEST_CASE("MorsePotential HardCutoff: energy is minimum at r = r0",
          "[potentials][morse][hard][minimum]") {
  // Numerically confirm the energy minimum is at r=r₀ by scanning a neighborhood
  // and checking argmin coincides with r₀ within grid spacing.
  auto pot = make_pot(tdmd::MorsePotential::CutoffStrategy::HardCutoff);
  double best_r = 0.0;
  double best_e = std::numeric_limits<double>::infinity();
  for (int k = -200; k <= 200; ++k) {
    const double r = kR0 + 0.001 * k;
    auto s = make_dimer(r);
    zero_forces(s.atoms);
    const auto result = pot.compute(s.atoms, s.list, s.box);
    if (result.potential_energy < best_e) {
      best_e = result.potential_energy;
      best_r = r;
    }
  }
  REQUIRE(best_r == Catch::Approx(kR0).margin(1e-3));
  REQUIRE(best_e == Catch::Approx(-kD).margin(1e-12));
}

TEST_CASE("MorsePotential: Newton's third law (pair forces sum to zero)",
          "[potentials][morse][newton]") {
  // Attractive regime (r > r₀), repulsive regime (r < r₀), and the well bottom.
  for (double r : {2.5, kR0, 4.0, 5.5, 7.5}) {
    auto s = make_dimer(r);
    auto pot = make_pot();
    zero_forces(s.atoms);
    (void) pot.compute(s.atoms, s.list, s.box);

    const double sum_x = s.atoms.fx[0] + s.atoms.fx[1];
    const double sum_y = s.atoms.fy[0] + s.atoms.fy[1];
    const double sum_z = s.atoms.fz[0] + s.atoms.fz[1];
    REQUIRE(std::abs(sum_x) < 1e-12);
    REQUIRE(std::abs(sum_y) < 1e-12);
    REQUIRE(std::abs(sum_z) < 1e-12);
  }
}

TEST_CASE("MorsePotential HardCutoff: force matches -dE/dr (≥10³ property cases)",
          "[potentials][morse][property][fd]") {
  // Sample 1024 separations. HardCutoff makes U(r) the clean Morse well so the
  // FD target is exact. Uses a 4-point O(h⁴) stencil so truncation (~1e-16·U⁽⁵⁾)
  // is dwarfed by the 1e-8 tolerance; roundoff at h=1e-4 is ~eps·U/h ≈ 3e-13.
  auto pot = make_pot(tdmd::MorsePotential::CutoffStrategy::HardCutoff);
  constexpr double h = 1e-4;
  constexpr int kCases = 1024;
  const double r_min = 1.5;
  const double r_max = kCutoff - 0.05;

  std::mt19937_64 rng(0x5eedcafedeadbeefULL);
  std::uniform_real_distribution<double> dist(r_min, r_max);

  double max_abs_err = 0.0;
  for (int c = 0; c < kCases; ++c) {
    const double r = dist(rng);
    auto spp = make_dimer(r + 2.0 * h);
    auto sp = make_dimer(r + h);
    auto sm = make_dimer(r - h);
    auto smm = make_dimer(r - 2.0 * h);
    auto s0 = make_dimer(r);

    zero_forces(spp.atoms);
    const auto epp = pot.compute(spp.atoms, spp.list, spp.box).potential_energy;
    zero_forces(sp.atoms);
    const auto ep = pot.compute(sp.atoms, sp.list, sp.box).potential_energy;
    zero_forces(sm.atoms);
    const auto em = pot.compute(sm.atoms, sm.list, sm.box).potential_energy;
    zero_forces(smm.atoms);
    const auto emm = pot.compute(smm.atoms, smm.list, smm.box).potential_energy;
    zero_forces(s0.atoms);
    (void) pot.compute(s0.atoms, s0.list, s0.box);

    // Atoms aligned along +x with atom 1 to the right; displacing atom 1 by +dr
    // along +x changes the pair separation by +dr, so dU/dr = dU/dx₁.
    // F_x on atom 1 = -dU/dx₁ = -dU/dr.
    const double dUdr = (emm - 8.0 * em + 8.0 * ep - epp) / (12.0 * h);
    const double fx_analytic = s0.atoms.fx[1];
    max_abs_err = std::max(max_abs_err, std::abs(fx_analytic - (-dUdr)));
  }
  REQUIRE(max_abs_err < 1e-8);
}

TEST_CASE("MorsePotential ShiftedForce: continuity at r_c", "[potentials][morse][shifted]") {
  // Strategy C is designed so that F_shifted(r_c) = 0 and E_shifted(r_c) = 0
  // exactly, with linear/quadratic decay respectively as r → r_c from below:
  //   F(r_c - ε) ≈ ε · |G'(r_c)|   (O(ε))
  //   E(r_c - ε) ≈ ½ · ε² · |G'(r_c)|  (O(ε²))
  // Using ε = 1e-10 → F ≈ 3e-13, E ≈ 1e-23 — both well within machine noise.
  auto pot = make_pot(tdmd::MorsePotential::CutoffStrategy::ShiftedForce);
  auto s = make_dimer(kCutoff - 1e-10);
  zero_forces(s.atoms);
  const auto result = pot.compute(s.atoms, s.list, s.box);
  REQUIRE(std::abs(s.atoms.fx[0]) < 1e-10);
  REQUIRE(std::abs(s.atoms.fx[1]) < 1e-10);
  REQUIRE(std::abs(s.atoms.fy[0]) < 1e-14);
  REQUIRE(std::abs(s.atoms.fz[0]) < 1e-14);
  REQUIRE(std::abs(result.potential_energy) < 1e-14);
}

TEST_CASE("MorsePotential: zero contribution past cutoff", "[potentials][morse][cutoff]") {
  // A pair just outside the cutoff (accepted by the neighbor skin) must produce
  // no force and no energy — the r² guard inside compute() should skip it.
  auto pot = make_pot(tdmd::MorsePotential::CutoffStrategy::ShiftedForce);
  auto s = make_dimer(kCutoff + 0.2 * kSkin);  // within skin — in neighbor list
  zero_forces(s.atoms);
  const auto result = pot.compute(s.atoms, s.list, s.box);
  REQUIRE(s.atoms.fx[0] == 0.0);
  REQUIRE(s.atoms.fx[1] == 0.0);
  REQUIRE(result.potential_energy == 0.0);
}

TEST_CASE("MorsePotential: determinism — two runs byte-identical",
          "[potentials][morse][determinism]") {
  auto run_once = [](double r) {
    auto s = make_dimer(r);
    auto pot = make_pot();
    zero_forces(s.atoms);
    const auto result = pot.compute(s.atoms, s.list, s.box);
    return std::make_tuple(s.atoms.fx[0],
                           s.atoms.fy[0],
                           s.atoms.fz[0],
                           s.atoms.fx[1],
                           s.atoms.fy[1],
                           s.atoms.fz[1],
                           result.potential_energy,
                           result.virial[0],
                           result.virial[1],
                           result.virial[2]);
  };
  const auto a = run_once(3.5);
  const auto b = run_once(3.5);
  REQUIRE(a == b);
}

TEST_CASE("MorsePotential: virial Voigt matches pair F·r", "[potentials][morse][virial]") {
  // For a dimer aligned along +x at separation r: v_xx = F_pair_x · dx = f·r (where
  // f is the force on atom i along +x, dx = +r). Off-diagonal and v_yy/v_zz are 0.
  auto pot = make_pot();
  auto s = make_dimer(3.8);
  zero_forces(s.atoms);
  const auto result = pot.compute(s.atoms, s.list, s.box);

  const double dx = s.atoms.x[1] - s.atoms.x[0];
  const double expected_vxx = s.atoms.fx[0] * dx;
  REQUIRE(result.virial[0] == Catch::Approx(expected_vxx).margin(1e-14));
  REQUIRE(result.virial[1] == Catch::Approx(0.0).margin(1e-14));
  REQUIRE(result.virial[2] == Catch::Approx(0.0).margin(1e-14));
  REQUIRE(result.virial[3] == Catch::Approx(0.0).margin(1e-14));
  REQUIRE(result.virial[4] == Catch::Approx(0.0).margin(1e-14));
  REQUIRE(result.virial[5] == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("MorsePotential: zero-atom system returns zero result", "[potentials][morse][edge]") {
  tdmd::AtomSoA atoms;
  auto box = make_cubic_box(30.0);
  tdmd::CellGrid grid;
  grid.build(box, kCutoff, kSkin);
  grid.bin(atoms);
  tdmd::NeighborList list;
  list.build(atoms, box, grid, kCutoff, kSkin);

  auto pot = make_pot();
  const auto result = pot.compute(atoms, list, box);
  REQUIRE(result.potential_energy == 0.0);
  for (double v : result.virial) {
    REQUIRE(v == 0.0);
  }
}
