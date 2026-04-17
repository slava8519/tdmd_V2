#include "tdmd/neighbor/displacement_tracker.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <random>

namespace {

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

tdmd::AtomSoA make_atoms_grid(int n = 8) {
  tdmd::AtomSoA atoms;
  for (int i = 0; i < n; ++i) {
    atoms.add_atom(0, static_cast<double>(i), static_cast<double>(i) * 0.5, 1.0);
  }
  return atoms;
}

}  // namespace

TEST_CASE("DisplacementTracker default-constructs empty with zero state",
          "[neighbor][tracker][smoke]") {
  tdmd::DisplacementTracker tr;
  REQUIRE(tr.empty());
  REQUIRE(tr.size() == 0);
  REQUIRE(tr.max_displacement() == 0.0);
  REQUIRE(tr.threshold() == 0.0);
  REQUIRE_FALSE(tr.needs_rebuild());
}

TEST_CASE("DisplacementTracker reset baselines positions and zeroes displacement",
          "[neighbor][tracker]") {
  auto atoms = make_atoms_grid();
  tdmd::DisplacementTracker tr;
  tr.set_threshold(0.15);
  tr.reset(atoms);

  REQUIRE(tr.size() == atoms.size());
  REQUIRE(tr.max_displacement() == 0.0);
  REQUIRE_FALSE(tr.needs_rebuild());
}

TEST_CASE("DisplacementTracker::update returns 0 when nothing moved", "[neighbor][tracker]") {
  auto atoms = make_atoms_grid();
  const auto box = make_cubic_box(20.0);
  tdmd::DisplacementTracker tr;
  tr.set_threshold(0.15);
  tr.reset(atoms);
  tr.update(atoms, box);
  REQUIRE(tr.max_displacement() == 0.0);
}

TEST_CASE("DisplacementTracker reports max displacement exactly", "[neighbor][tracker]") {
  auto atoms = make_atoms_grid();
  const auto box = make_cubic_box(20.0);
  tdmd::DisplacementTracker tr;
  tr.set_threshold(0.2);
  tr.reset(atoms);

  atoms.x[3] += 0.1;
  atoms.y[5] += 0.05;
  atoms.z[7] += 0.09;
  tr.update(atoms, box);
  REQUIRE(std::abs(tr.max_displacement() - 0.1) < 1e-14);
  REQUIRE_FALSE(tr.needs_rebuild());
}

TEST_CASE("DisplacementTracker triggers rebuild above threshold, not below",
          "[neighbor][tracker][trigger]") {
  auto atoms = make_atoms_grid();
  const auto box = make_cubic_box(20.0);
  tdmd::DisplacementTracker tr;
  tr.set_threshold(0.15);  // skin/2 for skin=0.3
  tr.reset(atoms);

  atoms.x[0] += 0.14;
  tr.update(atoms, box);
  REQUIRE_FALSE(tr.needs_rebuild());

  atoms.x[0] += 0.02;  // now displaced by 0.16, above 0.15
  tr.update(atoms, box);
  REQUIRE(tr.needs_rebuild());
}

TEST_CASE("DisplacementTracker uses minimum-image for periodic wraps", "[neighbor][tracker][pbc]") {
  const double L = 10.0;
  tdmd::AtomSoA atoms;
  atoms.add_atom(0, 0.05, 5.0, 5.0);

  const auto box = make_cubic_box(L);
  tdmd::DisplacementTracker tr;
  tr.set_threshold(0.5);
  tr.reset(atoms);

  atoms.x[0] = L - 0.02;
  tr.update(atoms, box);
  REQUIRE(std::abs(tr.max_displacement() - 0.07) < 1e-14);
  REQUIRE_FALSE(tr.needs_rebuild());
}

TEST_CASE("DisplacementTracker::update rejects size mismatch", "[neighbor][tracker][error]") {
  auto atoms = make_atoms_grid();
  tdmd::DisplacementTracker tr;
  tr.set_threshold(0.1);
  tr.reset(atoms);

  atoms.add_atom(0, 0.0, 0.0, 0.0);
  const auto box = make_cubic_box(20.0);
  REQUIRE_THROWS_AS(tr.update(atoms, box), std::logic_error);
}

TEST_CASE("DisplacementTracker set_threshold rejects negatives", "[neighbor][tracker][error]") {
  tdmd::DisplacementTracker tr;
  REQUIRE_THROWS_AS(tr.set_threshold(-0.01), std::invalid_argument);
  REQUIRE_NOTHROW(tr.set_threshold(0.0));
}

TEST_CASE("DisplacementTracker is conservative on random walks (10⁴ trials)",
          "[neighbor][tracker][property]") {
  std::mt19937_64 rng(0xC0D1FAC3U);
  std::uniform_real_distribution<double> step(-0.02, 0.02);

  const double skin = 0.3;
  const double threshold = 0.5 * skin;
  const double L = 20.0;
  const auto box = make_cubic_box(L);

  constexpr int kTrials = 10000;
  for (int t = 0; t < kTrials; ++t) {
    tdmd::AtomSoA atoms;
    atoms.add_atom(0, 5.0, 5.0, 5.0);
    atoms.add_atom(0, 7.0, 8.0, 9.0);

    tdmd::DisplacementTracker tr;
    tr.set_threshold(threshold);
    tr.reset(atoms);

    double exp_max = 0.0;
    int triggered_step = -1;
    for (int s = 1; s <= 30; ++s) {
      atoms.x[0] += step(rng);
      atoms.y[0] += step(rng);
      atoms.z[0] += step(rng);
      atoms.x[1] += step(rng);
      atoms.y[1] += step(rng);
      atoms.z[1] += step(rng);
      tr.update(atoms, box);
      exp_max = std::max(exp_max, tr.max_displacement());
      if (tr.needs_rebuild() && triggered_step == -1) {
        triggered_step = s;
        REQUIRE(tr.max_displacement() > threshold);
      }
    }
    if (triggered_step == -1) {
      REQUIRE(exp_max <= threshold);
    }
  }
}
