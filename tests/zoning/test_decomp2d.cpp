// SPEC: docs/specs/zoning/SPEC.md §3.2 (N_min formula), §4.2 (zigzag)
// Exec pack: docs/development/m3_execution_pack.md T3.4

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/decomp2d.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <set>
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

TEST_CASE("Decomp2D — N_min = 2·(N_inner+1) per Andreev eq. 43", "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  const double c = 2.5, s = 0.5;  // w=3

  // Cartesian grid of (N_y_box, N_z_box) in [2..8] × [2..8]. Outer axis
  // is the larger of the two; inner is the smaller. Trivial x = 1 zone.
  for (std::uint32_t Ny = 2; Ny <= 8; ++Ny) {
    for (std::uint32_t Nz = 2; Nz <= 8; ++Nz) {
      const double ly = Ny * 3.0 + 0.1;  // a hair above to avoid floor quirks
      const double lz = Nz * 3.0 + 0.1;
      auto plan = p.plan(make_box(3.0 + 0.1, ly, lz), c, s, 4, hint);

      const auto n_outer = std::max(Ny, Nz);
      const auto n_inner = std::min(Ny, Nz);
      REQUIRE(plan.scheme == tz::ZoningScheme::Decomp2D);
      REQUIRE(plan.n_min_per_rank == 2ull * (n_inner + 1ull));
      REQUIRE(plan.canonical_order.size() == n_outer * n_inner);
      const auto expected_opt =
          (static_cast<std::uint64_t>(n_outer) * n_inner) / (2ull * (n_inner + 1ull));
      REQUIRE(plan.optimal_rank_count == (expected_opt == 0 ? 1 : expected_opt));
    }
  }
}

TEST_CASE("Decomp2D — hand-worked zigzag trace N_y=3, N_z=2 → [0,1,2,5,4,3]",
          "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  // Box dims chosen so: x axis trivial (3.1/3 → 1 zone), y=3 zones,
  // z=2 zones. outer = y (3 zones), inner = z (2 zones).
  // But the SPEC §4.2 pseudocode writes zigzag with OUTER = Z and INNER = Y.
  // Our planner picks OUTER = axis-with-more-zones. So for this case
  // outer=y (Ny=3), inner=z (Nz=2). Flat index = ix + iy·Nx + iz·Nx·Ny
  // with Nx=1, Ny=3, Nz=2: flat = iy + 3·iz.
  // Zigzag over (outer=iy ∈ 0..2, inner=iz ∈ 0..1):
  //   iy=0 (even, fwd): iz=0,1 → flat = 0+0=0, 0+3=3
  //   iy=1 (odd, rev):  iz=1,0 → flat = 1+3=4, 1+0=1
  //   iy=2 (even, fwd): iz=0,1 → flat = 2+0=2, 2+3=5
  // Order: [0, 3, 4, 1, 2, 5]
  auto plan = p.plan(make_box(3.1, 9.1, 6.1), 2.5, 0.5, 1, hint);
  REQUIRE(plan.n_zones[0] == 1);
  REQUIRE(plan.n_zones[1] == 3);  // outer
  REQUIRE(plan.n_zones[2] == 2);  // inner
  const std::vector<tz::ZoneId> expected = {0, 3, 4, 1, 2, 5};
  REQUIRE(plan.canonical_order == expected);
}

TEST_CASE("Decomp2D — zigzag matches SPEC §4.2 when outer=Z, inner=Y", "[zoning][decomp2d]") {
  // SPEC §4.2 pseudocode uses outer=Z, inner=Y and gives for Ny=3,Nz=2
  // the order [0, 1, 2, 5, 4, 3]. Reproduce that exact configuration:
  // x trivial (1 zone), y=3, z=2, and z must be the OUTER axis (more
  // zones than y)... but here Ny=3 > Nz=2, so y would be outer. Swap:
  // use Ny=2, Nz=3 so z=3 zones (outer), y=2 zones (inner). Then:
  //   iz=0 (fwd): iy=0,1 → flat=0+0=0, 1+0=1
  //   iz=1 (rev): iy=1,0 → flat=1+2=3 wait, Nx·Ny stride=1·2=2...
  // Actually SPEC §4.2 writes flat = z·N_y + y. For our encoding
  // flat = ix + iy·Nx + iz·Nx·Ny with Nx=1, Ny=2: flat = iy + 2·iz.
  //   iz=0 fwd: iy=0,1 → 0, 1
  //   iz=1 rev: iy=1,0 → 3, 2
  //   iz=2 fwd: iy=0,1 → 4, 5
  // Order: [0, 1, 3, 2, 4, 5]. This matches the SPEC §4.2 formula
  // `z·N_y + y` once we interpret it as flat lex ID on a (Nx=1,Ny,Nz)
  // grid with zigzag over z.
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(3.1, 6.1, 9.1), 2.5, 0.5, 1, hint);
  REQUIRE(plan.n_zones[0] == 1);
  REQUIRE(plan.n_zones[1] == 2);  // inner
  REQUIRE(plan.n_zones[2] == 3);  // outer
  const std::vector<tz::ZoneId> expected = {0, 1, 3, 2, 4, 5};
  REQUIRE(plan.canonical_order == expected);
}

TEST_CASE("Decomp2D — canonical_order is a permutation of [0..N_total-1]", "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(3.1, 24.1, 15.1), 2.5, 0.5, 4, hint);  // 1×8×5
  REQUIRE(plan.n_zones[0] == 1);
  REQUIRE(plan.n_zones[1] == 8);
  REQUIRE(plan.n_zones[2] == 5);
  REQUIRE(plan.canonical_order.size() == 40);
  std::set<tz::ZoneId> unique(plan.canonical_order.begin(), plan.canonical_order.end());
  REQUIRE(unique.size() == 40);
  REQUIRE(*unique.begin() == 0u);
  // Max flat index with Nx=1, Ny=8: flat_max = iy_max + 8·iz_max = 7 + 8·4 = 39.
  REQUIRE(*unique.rbegin() == 39u);
}

TEST_CASE("Decomp2D — axis-pair selection picks 2 largest axes", "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  // Box 60×30×3 → n=(20, 10, 1). Active: x, y. Outer=x (20), inner=y (10).
  auto plan = p.plan(make_box(60, 30, 3), 2.5, 0.5, 1, hint);
  REQUIRE(plan.n_zones[0] == 20);                   // outer
  REQUIRE(plan.n_zones[1] == 10);                   // inner
  REQUIRE(plan.n_zones[2] == 1);                    // trivial
  REQUIRE(plan.n_min_per_rank == 2ull * (10 + 1));  // 22
}

TEST_CASE("Decomp2D — rejects box with fewer than 2 non-trivial axes", "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  // Only z has ≥ 2 zones → Decomp2D refuses.
  REQUIRE_THROWS_AS(p.plan(make_box(3, 3, 30), 2.5, 0.5, 1, hint), tz::ZoningPlanError);
  // All three trivial.
  REQUIRE_THROWS_AS(p.plan(make_box(3, 3, 3), 2.5, 0.5, 1, hint), tz::ZoningPlanError);
}

TEST_CASE("Decomp2D — rejects bad cutoff/skin", "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  REQUIRE_THROWS_AS(p.plan(make_box(60, 30, 3), 0.0, 0.5, 1, hint), tz::ZoningPlanError);
  REQUIRE_THROWS_AS(p.plan(make_box(60, 30, 3), 2.5, -0.1, 1, hint), tz::ZoningPlanError);
}

TEST_CASE("Decomp2D — zone_size >= cutoff+skin invariant on active axes", "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(6.4, 6.4, 3.0), 2.5, 0.5, 1, hint);
  REQUIRE(plan.n_zones[0] == 2);
  REQUIRE(plan.n_zones[1] == 2);
  REQUIRE(plan.n_zones[2] == 1);
  REQUIRE(plan.zone_size[0] >= 3.0);
  REQUIRE(plan.zone_size[1] >= 3.0);
}

TEST_CASE("Decomp2D — axis tie-break picks lowest-index pair", "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  // All three axes equal → active = (x, y), trivial = z.
  auto plan = p.plan(make_box(12, 12, 12), 2.5, 0.5, 1, hint);
  REQUIRE(plan.n_zones[0] == 4);  // outer = x
  REQUIRE(plan.n_zones[1] == 4);  // inner = y
  REQUIRE(plan.n_zones[2] == 1);  // trivial = z
}

TEST_CASE("Decomp2D — plan_with_scheme rejects non-Decomp2D", "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  REQUIRE_THROWS_AS(
      p.plan_with_scheme(make_box(60, 30, 3), 2.5, 0.5, tz::ZoningScheme::Linear1D, hint),
      tz::ZoningPlanError);
  auto plan = p.plan_with_scheme(make_box(60, 30, 3), 2.5, 0.5, tz::ZoningScheme::Decomp2D, hint);
  REQUIRE(plan.scheme == tz::ZoningScheme::Decomp2D);
}

TEST_CASE("Decomp2D — estimate_* match plan() output", "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  auto box = make_box(60, 30, 3);
  REQUIRE(p.estimate_n_min(tz::ZoningScheme::Decomp2D, box, 2.5, 0.5) == 2ull * (10 + 1));
  // 20·10 / 22 = 9
  REQUIRE(p.estimate_optimal_ranks(tz::ZoningScheme::Decomp2D, box, 2.5, 0.5) == 9);

  REQUIRE_THROWS_AS(p.estimate_n_min(tz::ZoningScheme::Linear1D, box, 2.5, 0.5),
                    tz::ZoningPlanError);
}

TEST_CASE("Decomp2D — validate_manual_plan is stubbed per OQ-M3-2", "[zoning][decomp2d]") {
  tz::Decomp2DZoningPlanner p;
  tz::ZoningPlan plan;
  std::string reason;
  REQUIRE_FALSE(p.validate_manual_plan(plan, reason));
  REQUIRE(reason.find("not implemented") != std::string::npos);
}

TEST_CASE("Decomp2D — 16×5 dissertation anchor (OQ-M3-5 pending)", "[zoning][decomp2d][!mayfail]") {
  // SPEC §8.3 claims n_opt=13 for 2D 16×5; §3.2 formula 2·(N_inner+1)
  // yields N_min=12, n_opt=floor(80/12)=6. Discrepancy tracked as
  // OQ-M3-5; resolution (SPEC delta) is T3.7 work. This test documents
  // the formula-consistent value and is tagged [!mayfail] so the
  // eventual SPEC delta can flip the expected number without churn.
  tz::Decomp2DZoningPlanner p;
  tz::PerformanceHint hint;
  // 16×5 zones: box 48×15 with z trivial.
  auto plan = p.plan(make_box(48.1, 15.1, 3.1), 2.5, 0.5, 1, hint);
  REQUIRE(plan.n_zones[0] == 16);
  REQUIRE(plan.n_zones[1] == 5);
  REQUIRE(plan.n_zones[2] == 1);
  REQUIRE(plan.n_min_per_rank == 12);
  REQUIRE(plan.optimal_rank_count == 6);  // per §3.2 formula
  // CHECK (soft) — if SPEC delta flips, this is the target value.
  CHECK(plan.optimal_rank_count == 13);  // §8.3 anchor (currently fails)
}
