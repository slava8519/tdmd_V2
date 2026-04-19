// T5.9 — Linear1D scheme acceptance: anchor-test geometry (10⁶ Al FCC
// atoms) partitioned into 8 Z-slabs must yield uniform atom counts within
// ±5 %. Canonical order monotonic [0, 1, ..., N_z-1]. Hilbert M3 default
// remains untouched (verified by the existing test_linear1d.cpp + the
// DefaultZoningPlanner selection tests — this file only exercises the
// forced-Linear1D path the anchor test relies on).
//
// SPEC:
//   zoning/SPEC.md §3.1 (Linear1D scheme), §4.2 (canonical order)
//   master spec §13.3 (anchor-test premise — Andreev §2.2 1D case)
// Exec pack: docs/development/m5_execution_pack.md T5.9

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/default_planner.hpp"
#include "tdmd/zoning/linear1d.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace tz = tdmd::zoning;

namespace {

// Al FCC: lattice constant 4.05 Å, 4 atoms per unit cell. The anchor test
// references a ~10⁶-atom box; we deliberately set Z to admit exactly 8
// slabs at the M5 cutoff (8 Å cutoff + 2 Å skin = 10 Å zone width) so the
// uniformity check has an integer reference value (125 000 atoms/slab).
constexpr double kCutoff = 8.0;
constexpr double kSkin = 2.0;
constexpr double kZoneWidth = kCutoff + kSkin;  // 10 Å
constexpr std::uint32_t kTargetNZ = 8;
constexpr double kBoxZ = kZoneWidth * kTargetNZ;  // 80 Å — fits 8 slabs exactly

// x, y kept short (single zone each) so `choose_axis` picks Z unambiguously.
constexpr double kBoxShort = kZoneWidth;  // 10 Å → 1 zone on x/y

// 10⁶ synthetic atoms — uniform draws in the box. The real anchor-test
// uses an Al FCC lattice which is also uniform at the ~10 Å zone scale,
// so the uniformity assertion carries over.
constexpr std::size_t kNAtoms = 1'000'000;
constexpr std::size_t kExpectedPerSlab = kNAtoms / kTargetNZ;  // 125 000
constexpr double kTolerance = 0.05;                            // ±5 %

tdmd::Box make_anchor_box() {
  tdmd::Box b;
  b.xlo = 0.0;
  b.xhi = kBoxShort;
  b.ylo = 0.0;
  b.yhi = kBoxShort;
  b.zlo = 0.0;
  b.zhi = kBoxZ;
  b.periodic_x = b.periodic_y = b.periodic_z = true;
  return b;
}

}  // namespace

TEST_CASE("T5.9 — Linear1D on 10⁶-atom anchor box: 8 Z-slabs uniform within ±5%",
          "[zoning][linear1d][T5.9][anchor]") {
  tz::Linear1DZoningPlanner p;
  tz::PerformanceHint hint;
  const auto box = make_anchor_box();
  const auto plan = p.plan(box, kCutoff, kSkin, /*n_ranks=*/4, hint);

  REQUIRE(plan.scheme == tz::ZoningScheme::Linear1D);
  REQUIRE(plan.n_zones[0] == 1);
  REQUIRE(plan.n_zones[1] == 1);
  REQUIRE(plan.n_zones[2] == kTargetNZ);
  REQUIRE(plan.total_zones() == kTargetNZ);
  REQUIRE(plan.canonical_order.size() == kTargetNZ);
  for (std::uint32_t i = 0; i < kTargetNZ; ++i) {
    REQUIRE(plan.canonical_order[i] == static_cast<tz::ZoneId>(i));
  }

  const double slab_width = plan.zone_size[2];
  CHECK(slab_width == Catch::Approx(kZoneWidth).epsilon(1e-12));

  // Deterministic uniform draw — same seed across platforms so the
  // ±5 % budget is evaluated against a reproducible histogram. std::mt19937
  // + uniform_real_distribution is portable enough for this coarse check.
  std::mt19937 rng(0xDEADBEEFU);
  std::uniform_real_distribution<double> uz(0.0, kBoxZ);
  std::array<std::size_t, kTargetNZ> counts{};
  for (std::size_t i = 0; i < kNAtoms; ++i) {
    const double z = uz(rng);
    auto slab = static_cast<std::size_t>(z / slab_width);
    if (slab >= kTargetNZ) {
      // Handles the z == kBoxZ edge from the exclusive upper bound rounding.
      slab = kTargetNZ - 1;
    }
    ++counts[slab];
  }

  const double lo = kExpectedPerSlab * (1.0 - kTolerance);
  const double hi = kExpectedPerSlab * (1.0 + kTolerance);
  for (std::size_t s = 0; s < kTargetNZ; ++s) {
    CAPTURE(s, counts[s], kExpectedPerSlab);
    CHECK(static_cast<double>(counts[s]) >= lo);
    CHECK(static_cast<double>(counts[s]) <= hi);
  }
}

TEST_CASE("T5.9 — DefaultZoningPlanner.plan_with_scheme honors forced Linear1D",
          "[zoning][default_planner][T5.9]") {
  // YAML `zoning.scheme: linear_1d` routes through plan_with_scheme; this
  // is the critical path the SimulationEngine takes for the anchor test,
  // so verify the dispatch does not fall back to the §3.4 auto-select
  // tree (which on a cubic box would pick Hilbert3D and produce a
  // different canonical_order).
  tz::DefaultZoningPlanner planner;
  tz::PerformanceHint hint;
  // Cubic geometry — auto-select would pick Hilbert3D here.
  tdmd::Box cube;
  cube.xlo = cube.ylo = cube.zlo = 0.0;
  cube.xhi = cube.yhi = cube.zhi = 80.0;
  cube.periodic_x = cube.periodic_y = cube.periodic_z = true;

  const auto hilbert_plan =
      planner.plan_with_scheme(cube, kCutoff, kSkin, tz::ZoningScheme::Hilbert3D, hint);
  CHECK(hilbert_plan.scheme == tz::ZoningScheme::Hilbert3D);

  const auto linear_plan =
      planner.plan_with_scheme(cube, kCutoff, kSkin, tz::ZoningScheme::Linear1D, hint);
  CHECK(linear_plan.scheme == tz::ZoningScheme::Linear1D);
  // On a cube Linear1D ties break to axis 0 (x) per §4.3, not Z — that's
  // still a valid Linear1D layout. Both x/y/z eligible with equal zone
  // counts here, so assert the tie-break invariant rather than hardcoding
  // an axis.
  const std::uint32_t nx = linear_plan.n_zones[0];
  const std::uint32_t ny = linear_plan.n_zones[1];
  const std::uint32_t nz = linear_plan.n_zones[2];
  CHECK(((nx > 1 && ny == 1 && nz == 1) || (nx == 1 && ny > 1 && nz == 1) ||
         (nx == 1 && ny == 1 && nz > 1)));
  CHECK(linear_plan.canonical_order.size() == linear_plan.total_zones());
  for (std::size_t i = 0; i < linear_plan.canonical_order.size(); ++i) {
    CHECK(linear_plan.canonical_order[i] == static_cast<tz::ZoneId>(i));
  }
}

TEST_CASE("T5.9 — Hilbert remains the auto-select default on cubic boxes",
          "[zoning][default_planner][T5.9][regression]") {
  // Regression guard: nothing in T5.9 may perturb the M3 §3.4 decision
  // tree. On a 80×80×80 Å cube with the M5 cutoff+skin Linear1D must NOT
  // be the auto-selected scheme — otherwise an M1/M2/M3 smoke that leaves
  // `zoning:` unset would start using a different scheme than M3 shipped.
  tdmd::Box cube;
  cube.xlo = cube.ylo = cube.zlo = 0.0;
  cube.xhi = cube.yhi = cube.zhi = 80.0;
  cube.periodic_x = cube.periodic_y = cube.periodic_z = true;
  CHECK(tz::DefaultZoningPlanner::select_scheme(cube, kCutoff, kSkin) ==
        tz::ZoningScheme::Hilbert3D);
}
