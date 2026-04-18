// SPEC: docs/specs/zoning/SPEC.md §3.1, §4.2 (Linear1D canonical order), §8
// Exec pack: docs/development/m3_execution_pack.md T3.3

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/linear1d.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <vector>

namespace tz = tdmd::zoning;

namespace {

tdmd::Box make_box(double lx, double ly, double lz) {
  tdmd::Box b;
  b.xlo = 0.0;
  b.xhi = lx;
  b.ylo = 0.0;
  b.yhi = ly;
  b.zlo = 0.0;
  b.zhi = lz;
  b.periodic_x = b.periodic_y = b.periodic_z = true;
  return b;
}

}  // namespace

TEST_CASE("Linear1D — N_min is always 2 per Andreev eq. 35", "[zoning][linear1d]") {
  tz::Linear1DZoningPlanner p;
  tz::PerformanceHint hint;
  // cutoff+skin=3 so zone counts are predictable
  const double cutoff = 2.5;
  const double skin = 0.5;

  // Short x/y (length 4 → 1 zone each at w=3) so z is the unique longest.
  for (double lz : {6.0, 12.0, 24.0, 48.0, 192.0}) {
    auto plan = p.plan(make_box(4.0, 4.0, lz), cutoff, skin, 4, hint);
    REQUIRE(plan.scheme == tz::ZoningScheme::Linear1D);
    REQUIRE(plan.n_min_per_rank == 2);
    // lz / 3 zones along z
    const auto expected = static_cast<std::uint32_t>(lz / 3.0);
    REQUIRE(plan.n_zones[2] == expected);
    REQUIRE(plan.n_zones[0] == 1);
    REQUIRE(plan.n_zones[1] == 1);
    REQUIRE(plan.canonical_order.size() == expected);
    REQUIRE(plan.optimal_rank_count == expected / 2u);
  }
}

TEST_CASE("Linear1D — picks longest axis (thin-slab geometry)", "[zoning][linear1d]") {
  tz::Linear1DZoningPlanner p;
  tz::PerformanceHint hint;
  const double cutoff = 2.5, skin = 0.5;  // w=3
  const double long_ax = 60.0, short_ax = 6.0;

  SECTION("long on x") {
    auto plan = p.plan(make_box(long_ax, short_ax, short_ax), cutoff, skin, 4, hint);
    REQUIRE(plan.n_zones[0] == 20);
    REQUIRE(plan.n_zones[1] == 1);
    REQUIRE(plan.n_zones[2] == 1);
  }
  SECTION("long on y") {
    auto plan = p.plan(make_box(short_ax, long_ax, short_ax), cutoff, skin, 4, hint);
    REQUIRE(plan.n_zones[0] == 1);
    REQUIRE(plan.n_zones[1] == 20);
    REQUIRE(plan.n_zones[2] == 1);
  }
  SECTION("long on z") {
    auto plan = p.plan(make_box(short_ax, short_ax, long_ax), cutoff, skin, 4, hint);
    REQUIRE(plan.n_zones[0] == 1);
    REQUIRE(plan.n_zones[1] == 1);
    REQUIRE(plan.n_zones[2] == 20);
  }
}

TEST_CASE("Linear1D — canonical_order is sequential [0..N-1] permutation", "[zoning][linear1d]") {
  tz::Linear1DZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(6, 6, 30), 2.5, 0.5, 4, hint);
  REQUIRE(plan.canonical_order.size() == 10);
  for (std::size_t i = 0; i < plan.canonical_order.size(); ++i) {
    REQUIRE(plan.canonical_order[i] == static_cast<tz::ZoneId>(i));
  }

  // Permutation property (trivial here but let's prove it).
  std::vector<tz::ZoneId> sorted = plan.canonical_order;
  std::sort(sorted.begin(), sorted.end());
  std::vector<tz::ZoneId> expected(sorted.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<tz::ZoneId>(i);
  }
  REQUIRE(sorted == expected);
}

TEST_CASE("Linear1D — axis tie-break picks lowest axis index", "[zoning][linear1d]") {
  // All three axes admit the same number of zones → choose x.
  tz::Linear1DZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(12, 12, 12), 2.5, 0.5, 4, hint);
  REQUIRE(plan.n_zones[0] == 4);
  REQUIRE(plan.n_zones[1] == 1);
  REQUIRE(plan.n_zones[2] == 1);
}

TEST_CASE("Linear1D — rejects box with fewer than 2 zones", "[zoning][linear1d]") {
  tz::Linear1DZoningPlanner p;
  tz::PerformanceHint hint;
  // Each side ~3 Å → 1 zone at w=3 → cannot TD.
  REQUIRE_THROWS_AS(p.plan(make_box(3, 3, 3), 2.5, 0.5, 1, hint), tz::ZoningPlanError);
}

TEST_CASE("Linear1D — rejects bad cutoff/skin", "[zoning][linear1d]") {
  tz::Linear1DZoningPlanner p;
  tz::PerformanceHint hint;
  REQUIRE_THROWS_AS(p.plan(make_box(60, 6, 6), 0.0, 0.5, 1, hint), tz::ZoningPlanError);
  REQUIRE_THROWS_AS(p.plan(make_box(60, 6, 6), 2.5, -0.1, 1, hint), tz::ZoningPlanError);
}

TEST_CASE("Linear1D — zone_size[axis] >= cutoff+skin invariant", "[zoning][linear1d]") {
  tz::Linear1DZoningPlanner p;
  tz::PerformanceHint hint;
  // Box that's just a shade above the two-zone boundary (6.4 → 2 zones of 3.2).
  // x/y short so z wins axis selection unambiguously.
  auto plan = p.plan(make_box(4, 4, 6.4), 2.5, 0.5, 1, hint);
  REQUIRE(plan.n_zones[2] == 2);
  REQUIRE(plan.zone_size[2] >= 3.0);
}

TEST_CASE("Linear1D — plan_with_scheme rejects non-Linear1D", "[zoning][linear1d]") {
  tz::Linear1DZoningPlanner p;
  tz::PerformanceHint hint;
  REQUIRE_THROWS_AS(
      p.plan_with_scheme(make_box(60, 6, 6), 2.5, 0.5, tz::ZoningScheme::Decomp2D, hint),
      tz::ZoningPlanError);
  // Same-scheme passes through.
  auto plan = p.plan_with_scheme(make_box(60, 6, 6), 2.5, 0.5, tz::ZoningScheme::Linear1D, hint);
  REQUIRE(plan.scheme == tz::ZoningScheme::Linear1D);
  REQUIRE(plan.n_zones[0] == 20);
}

TEST_CASE("Linear1D — estimate_* match plan() output", "[zoning][linear1d]") {
  tz::Linear1DZoningPlanner p;
  auto box = make_box(60, 6, 6);
  REQUIRE(p.estimate_n_min(tz::ZoningScheme::Linear1D, box, 2.5, 0.5) == 2);
  REQUIRE(p.estimate_optimal_ranks(tz::ZoningScheme::Linear1D, box, 2.5, 0.5) == 10);

  REQUIRE_THROWS_AS(p.estimate_n_min(tz::ZoningScheme::Decomp2D, box, 2.5, 0.5),
                    tz::ZoningPlanError);
}

TEST_CASE("Linear1D — validate_manual_plan is stubbed per OQ-M3-2", "[zoning][linear1d]") {
  tz::Linear1DZoningPlanner p;
  tz::ZoningPlan plan;
  std::string reason;
  REQUIRE_FALSE(p.validate_manual_plan(plan, reason));
  REQUIRE(reason.find("not implemented") != std::string::npos);
}
