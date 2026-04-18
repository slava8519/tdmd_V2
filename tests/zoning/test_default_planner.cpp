// SPEC: docs/specs/zoning/SPEC.md §3.4
// Exec pack: docs/development/m3_execution_pack.md T3.6

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/default_planner.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>

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

TEST_CASE("Default — needle geometry (aspect > 10, min_ax < 4) → Linear1D", "[zoning][default]") {
  // 60 × 3 × 3 → n = (20, 1, 1); aspect = 20, min_ax = 1 → Linear1D.
  const auto scheme = tz::DefaultZoningPlanner::select_scheme(make_box(60, 3, 3), 2.5, 0.5);
  REQUIRE(scheme == tz::ZoningScheme::Linear1D);

  tz::DefaultZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(60, 3, 3), 2.5, 0.5, 1, hint);
  REQUIRE(plan.scheme == tz::ZoningScheme::Linear1D);
  REQUIRE(plan.n_zones[0] == 20);
}

TEST_CASE("Default — thin slab (aspect > 3) → Decomp2D", "[zoning][default]") {
  // 60 × 60 × 3 → n = (20, 20, 1); aspect = 20, min_ax = 1. The
  // Linear1D predicate checks `min_ax < 4` — passes; aspect > 10 —
  // passes. So this geometry actually triggers Linear1D, not
  // Decomp2D. Use a less extreme slab: 30 × 15 × 3 → n = (10, 5, 1),
  // aspect = 10/1 = 10; min_ax=1 <4 but aspect !>10 → falls through
  // to Decomp2D (aspect > 3 branch).
  const auto scheme = tz::DefaultZoningPlanner::select_scheme(make_box(30, 15, 3), 2.5, 0.5);
  REQUIRE(scheme == tz::ZoningScheme::Decomp2D);
  tz::DefaultZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(30, 15, 3), 2.5, 0.5, 1, hint);
  REQUIRE(plan.scheme == tz::ZoningScheme::Decomp2D);
  REQUIRE(plan.n_zones[0] == 10);
  REQUIRE(plan.n_zones[1] == 5);
  REQUIRE(plan.n_zones[2] == 1);
}

TEST_CASE("Default — small xy-plane triggers Decomp2D via (Nx·Ny)<16 clause", "[zoning][default]") {
  // Cube 12 × 12 × 12 → n = (4, 4, 4), Nx·Ny = 16 (NOT < 16), aspect = 1
  // → Hilbert3D. Shrink one axis: 9 × 9 × 12 → n = (3, 3, 4); xy = 9 < 16
  // → Decomp2D even though aspect is mild.
  const auto scheme = tz::DefaultZoningPlanner::select_scheme(make_box(9, 9, 12), 2.5, 0.5);
  REQUIRE(scheme == tz::ZoningScheme::Decomp2D);
}

TEST_CASE("Default — cube (aspect ≤ 3 and xy ≥ 16) → Hilbert3D", "[zoning][default]") {
  // 12 × 12 × 12 → n = (4, 4, 4), aspect = 1, xy = 16 → Hilbert3D.
  const auto scheme = tz::DefaultZoningPlanner::select_scheme(make_box(12, 12, 12), 2.5, 0.5);
  REQUIRE(scheme == tz::ZoningScheme::Hilbert3D);

  tz::DefaultZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(12, 12, 12), 2.5, 0.5, 1, hint);
  REQUIRE(plan.scheme == tz::ZoningScheme::Hilbert3D);
  REQUIRE(plan.canonical_order.size() == 64);
}

TEST_CASE("Default — rejects box with < 3 total zones", "[zoning][default]") {
  tz::DefaultZoningPlanner p;
  tz::PerformanceHint hint;
  REQUIRE_THROWS_AS(p.plan(make_box(3, 3, 3), 2.5, 0.5, 1, hint), tz::ZoningPlanError);
}

TEST_CASE("Default — rejects bad cutoff/skin", "[zoning][default]") {
  tz::DefaultZoningPlanner p;
  tz::PerformanceHint hint;
  REQUIRE_THROWS_AS(p.plan(make_box(60, 3, 3), 0.0, 0.5, 1, hint), tz::ZoningPlanError);
  REQUIRE_THROWS_AS(p.plan(make_box(60, 3, 3), 2.5, -0.1, 1, hint), tz::ZoningPlanError);
}

TEST_CASE("Default — advisory fires when n_ranks > 1.2·n_opt", "[zoning][default]") {
  tz::DefaultZoningPlanner p;
  tz::PerformanceHint hint;
  // Linear1D on 33×3×3 → n = (11,1,1), aspect = 11 > 10. n_opt = 11/2 = 5.
  // 30×3×3 would give aspect == 10 exactly, which fails the strict
  // `aspect > 10` check and drops to Decomp2D (which then rejects a
  // single-long-axis box). Use 33 to stay unambiguously in Linear1D.
  auto plan_many = p.plan(make_box(33, 3, 3), 2.5, 0.5, 10, hint);
  REQUIRE(plan_many.scheme == tz::ZoningScheme::Linear1D);
  REQUIRE(plan_many.optimal_rank_count == 5);
  REQUIRE(plan_many.advisories.size() == 1);
  REQUIRE(plan_many.advisories[0].find("exceeds 1.2·n_opt") != std::string::npos);

  // Request 5 ranks → within threshold → no advisory.
  auto plan_ok = p.plan(make_box(33, 3, 3), 2.5, 0.5, 5, hint);
  REQUIRE(plan_ok.advisories.empty());
}

TEST_CASE("Default — plan_with_scheme dispatches to the requested planner", "[zoning][default]") {
  tz::DefaultZoningPlanner p;
  tz::PerformanceHint hint;
  auto box = make_box(30, 3, 3);

  auto lin = p.plan_with_scheme(box, 2.5, 0.5, tz::ZoningScheme::Linear1D, hint);
  REQUIRE(lin.scheme == tz::ZoningScheme::Linear1D);

  // Decomp2D on this box would throw (only one non-trivial axis).
  REQUIRE_THROWS_AS(p.plan_with_scheme(box, 2.5, 0.5, tz::ZoningScheme::Decomp2D, hint),
                    tz::ZoningPlanError);

  // Manual is stubbed.
  REQUIRE_THROWS_AS(p.plan_with_scheme(box, 2.5, 0.5, tz::ZoningScheme::Manual, hint),
                    tz::ZoningPlanError);
}

TEST_CASE("Default — estimate_* dispatch to concrete planners", "[zoning][default]") {
  tz::DefaultZoningPlanner p;
  auto box = make_box(60, 6, 6);
  REQUIRE(p.estimate_n_min(tz::ZoningScheme::Linear1D, box, 2.5, 0.5) == 2);
  REQUIRE(p.estimate_optimal_ranks(tz::ZoningScheme::Linear1D, box, 2.5, 0.5) == 10);

  auto cube = make_box(12, 12, 12);
  REQUIRE(p.estimate_n_min(tz::ZoningScheme::Hilbert3D, cube, 2.5, 0.5) == 4ull * 16);
  REQUIRE(p.estimate_optimal_ranks(tz::ZoningScheme::Hilbert3D, cube, 2.5, 0.5) == 1u);
}

TEST_CASE("Default — validate_manual_plan is stubbed per OQ-M3-2", "[zoning][default]") {
  tz::DefaultZoningPlanner p;
  tz::ZoningPlan plan;
  std::string reason;
  REQUIRE_FALSE(p.validate_manual_plan(plan, reason));
  REQUIRE(reason.find("not implemented") != std::string::npos);
}
