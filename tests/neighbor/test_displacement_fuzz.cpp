// SPEC: docs/specs/neighbor/SPEC.md §5, §6
// Exec pack: docs/development/m3_execution_pack.md T3.8, D-M3-8
//
// ≥10⁴ fuzz cases covering random velocities, skin ratios, and periodic
// boundary crossings. The tests verify the tracker's behavioural contract
// rather than a specific numeric outcome:
//
//   P1. `skin_exceeded()` NEVER returns true when the real maximum
//        per-atom displacement is still ≤ threshold. (Conservative bound
//        of SPEC §5.2.)
//   P2. `execute_rebuild()` always clears skin_exceeded + rebuild_pending
//        and bumps build_version by exactly 1, regardless of hysteresis.
//   P3. After an explicit `request_rebuild(reason)`, `rebuild_pending`
//        stays true until the next `execute_rebuild` — independent of
//        whether the skin continues to exceed or not.
//   P4. Under periodic boundaries, the tracker's computed displacement
//        matches the hand-computed minimum-image distance to within 1e-10
//        (reference-profile bit-match budget).
//
// Seed is frozen — D-M3-8 reproducibility. Any regression that changes
// generator output should either be rolled back or documented as an
// explicit baseline bump in this file's header.

#include "tdmd/neighbor/displacement_tracker.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr std::uint64_t kFixedSeed = 0xD15B1ACEC0DEBEEFuLL;
constexpr int kCases = 10000;  // D-M3-8 lower bound for displacement fuzz.

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

// Hand-computed minimum-image distance for a single atom, used as the
// ground truth for P4. Matches tdmd::Box::unwrap_minimum_image's formula
// but phrased inline so a regression there shows up here.
double mi_distance(double dx, double dy, double dz, double L) {
  auto wrap = [L](double v) {
    while (v > 0.5 * L)
      v -= L;
    while (v < -0.5 * L)
      v += L;
    return v;
  };
  dx = wrap(dx);
  dy = wrap(dy);
  dz = wrap(dz);
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace

TEST_CASE("Tracker fuzz — skin bound + rebuild hygiene over 1e4 cases",
          "[neighbor][tracker][fuzz][property]") {
  std::mt19937_64 rng(kFixedSeed);
  std::uniform_real_distribution<double> skin_dist(0.1, 2.0);
  std::uniform_real_distribution<double> dt_dist(0.0005, 0.002);  // ps
  std::uniform_real_distribution<double> v_max_dist(0.1, 10.0);   // Å/ps
  std::uniform_int_distribution<int> n_atoms_dist(2, 32);
  std::uniform_int_distribution<int> n_steps_dist(5, 40);
  std::uniform_real_distribution<double> pos_dist(0.0, 1.0);
  std::uniform_real_distribution<double> v_component(-1.0, 1.0);

  const double L = 20.0;
  const auto box = make_cubic_box(L);

  int cases_with_trigger = 0;
  int cases_without_trigger = 0;

  for (int c = 0; c < kCases; ++c) {
    const double skin = skin_dist(rng);
    const double threshold = 0.5 * skin;
    const double dt = dt_dist(rng);
    const double v_max = v_max_dist(rng);
    const int n_atoms = n_atoms_dist(rng);
    const int n_steps = n_steps_dist(rng);

    tdmd::AtomSoA atoms;
    std::vector<double> vx(static_cast<std::size_t>(n_atoms));
    std::vector<double> vy(static_cast<std::size_t>(n_atoms));
    std::vector<double> vz(static_cast<std::size_t>(n_atoms));
    for (int i = 0; i < n_atoms; ++i) {
      atoms.add_atom(0, pos_dist(rng) * L, pos_dist(rng) * L, pos_dist(rng) * L);
      vx[static_cast<std::size_t>(i)] = v_max * v_component(rng);
      vy[static_cast<std::size_t>(i)] = v_max * v_component(rng);
      vz[static_cast<std::size_t>(i)] = v_max * v_component(rng);
    }

    tdmd::DisplacementTracker tr;
    tr.set_threshold(threshold);
    tr.init(atoms);
    const std::vector<double> x0(atoms.x.begin(), atoms.x.end());
    const std::vector<double> y0(atoms.y.begin(), atoms.y.end());
    const std::vector<double> z0(atoms.z.begin(), atoms.z.end());

    bool triggered = false;
    for (int s = 0; s < n_steps; ++s) {
      for (int i = 0; i < n_atoms; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        double nx = atoms.x[ui] + vx[ui] * dt;
        double ny = atoms.y[ui] + vy[ui] * dt;
        double nz = atoms.z[ui] + vz[ui] * dt;
        // Wrap into box so the tracker sees a real periodic-crossing
        // case (not just a steadily growing coordinate).
        while (nx >= L)
          nx -= L;
        while (nx < 0)
          nx += L;
        while (ny >= L)
          ny -= L;
        while (ny < 0)
          ny += L;
        while (nz >= L)
          nz -= L;
        while (nz < 0)
          nz += L;
        atoms.x[ui] = nx;
        atoms.y[ui] = ny;
        atoms.z[ui] = nz;
      }

      tr.update_displacement(atoms, box);

      // P4: tracker's max_d must equal the hand-computed minimum-image max.
      double truth_max = 0.0;
      for (int i = 0; i < n_atoms; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const double d =
            mi_distance(atoms.x[ui] - x0[ui], atoms.y[ui] - y0[ui], atoms.z[ui] - z0[ui], L);
        truth_max = std::max(truth_max, d);
      }
      REQUIRE(std::abs(tr.max_displacement() - truth_max) < 1e-10);

      // P1: skin_exceeded must imply truth_max > threshold.
      if (tr.skin_exceeded()) {
        REQUIRE(truth_max > threshold);
      }

      if (tr.skin_exceeded() && !triggered) {
        triggered = true;
        tr.request_rebuild("skin fuzz");
        REQUIRE(tr.rebuild_pending());
        REQUIRE(tr.rebuild_reason() == "skin fuzz");

        // P2: execute_rebuild resets and bumps build_version.
        const auto version_before = tr.build_version();
        tr.execute_rebuild(atoms);
        REQUIRE(tr.build_version() == version_before + 1);
        REQUIRE(tr.max_displacement() == 0.0);
        REQUIRE_FALSE(tr.skin_exceeded());
        REQUIRE_FALSE(tr.rebuild_pending());
        // P3 partial: request after execute comes up clean.
        tr.request_rebuild("external trigger");
        REQUIRE(tr.rebuild_pending());
        REQUIRE(tr.rebuild_reason() == "external trigger");
        tr.execute_rebuild(atoms);
        REQUIRE_FALSE(tr.rebuild_pending());
        break;
      }
    }

    if (triggered) {
      ++cases_with_trigger;
    } else {
      ++cases_without_trigger;
    }
  }

  // Sanity check that the fuzz corpus exercises both branches — if
  // every case triggered or none did, the parameter ranges drifted and
  // the coverage interpretation breaks. Permissive bounds so minor
  // generator tweaks don't flake CI.
  INFO("cases with rebuild=" << cases_with_trigger << " / without=" << cases_without_trigger);
  REQUIRE(cases_with_trigger > kCases / 20);     // > 5% exercised the rebuild path
  REQUIRE(cases_without_trigger > kCases / 20);  // > 5% stayed below threshold
}
