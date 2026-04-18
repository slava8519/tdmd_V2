// SPEC: docs/specs/zoning/SPEC.md §3.3, §3.5, §4.2
// Exec pack: docs/development/m3_execution_pack.md T3.5

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/hilbert3d.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
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

TEST_CASE("Hilbert3D — N_min within [3N², 6N²] envelope for cubic boxes", "[zoning][hilbert3d]") {
  tz::Hilbert3DZoningPlanner p;
  tz::PerformanceHint hint;
  for (std::uint32_t N : {4u, 8u, 16u}) {
    const double L = N * 3.0 + 0.1;
    auto plan = p.plan(make_box(L, L, L), 2.5, 0.5, 1, hint);
    REQUIRE(plan.n_zones[0] == N);
    REQUIRE(plan.n_zones[1] == N);
    REQUIRE(plan.n_zones[2] == N);
    const std::uint64_t N2 = static_cast<std::uint64_t>(N) * N;
    REQUIRE(plan.n_min_per_rank >= 3ull * N2);
    REQUIRE(plan.n_min_per_rank <= 6ull * N2);
  }
}

TEST_CASE("Hilbert3D — canonical_order is a permutation of [0..total-1]", "[zoning][hilbert3d]") {
  tz::Hilbert3DZoningPlanner p;
  tz::PerformanceHint hint;
  // Non-power-of-2 sizes exercise the boundary-filter path.
  auto plan = p.plan(make_box(18.1, 12.1, 9.1), 2.5, 0.5, 1, hint);
  REQUIRE(plan.n_zones[0] == 6);
  REQUIRE(plan.n_zones[1] == 4);
  REQUIRE(plan.n_zones[2] == 3);
  const std::size_t total = 6 * 4 * 3;
  REQUIRE(plan.canonical_order.size() == total);
  std::set<tz::ZoneId> unique(plan.canonical_order.begin(), plan.canonical_order.end());
  REQUIRE(unique.size() == total);
  REQUIRE(*unique.begin() == 0u);
  REQUIRE(*unique.rbegin() == static_cast<tz::ZoneId>(total - 1));
}

TEST_CASE("Hilbert3D — locality: mean L1 distance between consecutive zones is low",
          "[zoning][hilbert3d]") {
  // Hilbert's defining virtue is that consecutive zones are spatially
  // close. We use a simple bound: the average L1 box-coord distance
  // between consecutive canonical_order entries should be small
  // (≤ 2.0 for an N³ cube, since Hilbert walks unit-distance steps
  // within a level and occasionally jumps between sublevels).
  tz::Hilbert3DZoningPlanner p;
  tz::PerformanceHint hint;
  auto plan = p.plan(make_box(24.1, 24.1, 24.1), 2.5, 0.5, 1, hint);
  const std::uint32_t Nx = plan.n_zones[0];
  const std::uint32_t Ny = plan.n_zones[1];
  REQUIRE(Nx == 8);

  const auto to_xyz = [Nx, Ny](tz::ZoneId id) {
    const std::uint32_t x = id % Nx;
    const std::uint32_t y = (id / Nx) % Ny;
    const std::uint32_t z = id / (Nx * Ny);
    return std::array<std::uint32_t, 3>{x, y, z};
  };

  double total_l1 = 0.0;
  for (std::size_t i = 1; i < plan.canonical_order.size(); ++i) {
    const auto a = to_xyz(plan.canonical_order[i - 1]);
    const auto b = to_xyz(plan.canonical_order[i]);
    total_l1 += std::abs(static_cast<int>(a[0]) - static_cast<int>(b[0])) +
                std::abs(static_cast<int>(a[1]) - static_cast<int>(b[1])) +
                std::abs(static_cast<int>(a[2]) - static_cast<int>(b[2]));
  }
  const double mean = total_l1 / static_cast<double>(plan.canonical_order.size() - 1);
  // Lex order on the same grid gives mean ≈ 1.5 for the inner axis but
  // up to Nx for wrap-around rows — mean > 2.0 easily. Hilbert keeps
  // the mean low.
  REQUIRE(mean <= 2.0);
}

TEST_CASE("Hilbert3D — 2×2×2 corner case", "[zoning][hilbert3d]") {
  tz::Hilbert3DZoningPlanner p;
  tz::PerformanceHint hint;
  // 6.1 / 3 = 2 zones per axis → 8 total zones.
  auto plan = p.plan(make_box(6.1, 6.1, 6.1), 2.5, 0.5, 1, hint);
  REQUIRE(plan.n_zones[0] == 2);
  REQUIRE(plan.n_zones[1] == 2);
  REQUIRE(plan.n_zones[2] == 2);
  REQUIRE(plan.canonical_order.size() == 8);
  // N_min = 4·max(4,4,4) = 16 > total=8 → optimal_rank_count clamped to 1.
  REQUIRE(plan.n_min_per_rank == 16u);
  REQUIRE(plan.optimal_rank_count == 1u);
}

TEST_CASE("Hilbert3D — 16³ cube matches §3.3 formula", "[zoning][hilbert3d]") {
  // §3.3: n_opt ≈ min(N)/4. For 16³ → 4. (OQ-M3-6 tracks the
  // separate "target n_opt=64" claim in §8.3 which misreads the row
  // header.) 4·16² = 1024; 4096/1024 = 4 exactly.
  tz::Hilbert3DZoningPlanner p;
  tz::PerformanceHint hint;
  const double L = 16 * 3.0 + 0.1;
  auto plan = p.plan(make_box(L, L, L), 2.5, 0.5, 1, hint);
  REQUIRE(plan.n_zones[0] == 16);
  REQUIRE(plan.n_zones[1] == 16);
  REQUIRE(plan.n_zones[2] == 16);
  REQUIRE(plan.n_min_per_rank == 1024u);
  REQUIRE(plan.optimal_rank_count == 4u);
}

TEST_CASE("Hilbert3D — rejects bad cutoff/skin", "[zoning][hilbert3d]") {
  tz::Hilbert3DZoningPlanner p;
  tz::PerformanceHint hint;
  REQUIRE_THROWS_AS(p.plan(make_box(12, 12, 12), 0.0, 0.5, 1, hint), tz::ZoningPlanError);
  REQUIRE_THROWS_AS(p.plan(make_box(12, 12, 12), 2.5, -0.1, 1, hint), tz::ZoningPlanError);
}

TEST_CASE("Hilbert3D — rejects a single-zone box", "[zoning][hilbert3d]") {
  tz::Hilbert3DZoningPlanner p;
  tz::PerformanceHint hint;
  REQUIRE_THROWS_AS(p.plan(make_box(3, 3, 3), 2.5, 0.5, 1, hint), tz::ZoningPlanError);
}

TEST_CASE("Hilbert3D — plan_with_scheme rejects non-Hilbert3D", "[zoning][hilbert3d]") {
  tz::Hilbert3DZoningPlanner p;
  tz::PerformanceHint hint;
  REQUIRE_THROWS_AS(
      p.plan_with_scheme(make_box(12, 12, 12), 2.5, 0.5, tz::ZoningScheme::Linear1D, hint),
      tz::ZoningPlanError);
  auto plan = p.plan_with_scheme(make_box(12, 12, 12), 2.5, 0.5, tz::ZoningScheme::Hilbert3D, hint);
  REQUIRE(plan.scheme == tz::ZoningScheme::Hilbert3D);
}

TEST_CASE("Hilbert3D — estimate_* match plan() output", "[zoning][hilbert3d]") {
  tz::Hilbert3DZoningPlanner p;
  auto box = make_box(24.1, 24.1, 24.1);
  REQUIRE(p.estimate_n_min(tz::ZoningScheme::Hilbert3D, box, 2.5, 0.5) == 4ull * 8 * 8);
  REQUIRE(p.estimate_optimal_ranks(tz::ZoningScheme::Hilbert3D, box, 2.5, 0.5) == 2u);
  REQUIRE_THROWS_AS(p.estimate_n_min(tz::ZoningScheme::Decomp2D, box, 2.5, 0.5),
                    tz::ZoningPlanError);
}

TEST_CASE("Hilbert3D — validate_manual_plan is stubbed per OQ-M3-2", "[zoning][hilbert3d]") {
  tz::Hilbert3DZoningPlanner p;
  tz::ZoningPlan plan;
  std::string reason;
  REQUIRE_FALSE(p.validate_manual_plan(plan, reason));
  REQUIRE(reason.find("not implemented") != std::string::npos);
}
