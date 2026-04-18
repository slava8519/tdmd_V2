// SPEC: docs/specs/zoning/SPEC.md §8.2 (property tests)
// Exec pack: docs/development/m3_execution_pack.md T3.7, D-M3-8
//
// ≥10⁵ fuzz cases per scheme. The fuzzer biases into the geometric region
// each concrete planner actually handles, but we exercise the *concrete*
// planners (Linear1DZoningPlanner, Decomp2DZoningPlanner,
// Hilbert3DZoningPlanner) directly rather than going through Default —
// the property tests verify formula correctness, not selection logic.
//
// Invariants checked (per SPEC §8.2):
//   1. plan.n_min_per_rank >= 1
//   2. plan.optimal_rank_count >= 1
//   3. plan.canonical_order.size() == product(plan.n_zones)
//   4. is_permutation(plan.canonical_order)  — every ZoneId in
//      [0, total) appears exactly once
//   5. zone_size[i] >= cutoff + skin  — except when the box admits only
//      1 zone on that axis, in which case the planner may stretch it
//   6. Scheme-specific N_min formula:
//        Linear1D  → N_min == 2
//        Decomp2D  → N_min == 2·(N_inner+1)
//        Hilbert3D → N_min == 4·max(Nx·Ny, Ny·Nz, Nx·Nz)
//   7. Determinism — same inputs twice produce a bit-exact plan.

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/decomp2d.hpp"
#include "tdmd/zoning/hilbert3d.hpp"
#include "tdmd/zoning/linear1d.hpp"
#include "tdmd/zoning/zoning.hpp"

#include "fuzz_corpus.hpp"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

namespace tz = tdmd::zoning;
namespace tf = tdmd::zoning::fuzz;

namespace {

// Count of fuzz cases per scheme. D-M3-8 demands ≥10⁵ per scheme; we
// run exactly that and rely on the fuzz generator biasing to give broad
// coverage without runaway runtime (empirically < 1.5 s total on a
// 2020-era x86_64 debug build).
constexpr std::size_t kCasesPerScheme = 100000;

bool is_permutation_of_0_to_n(const std::vector<tz::ZoneId>& v) {
  std::vector<tz::ZoneId> sorted(v);
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i] != static_cast<tz::ZoneId>(i)) {
      return false;
    }
  }
  return true;
}

void check_core_invariants(const tz::ZoningPlan& plan, double cutoff, double skin) {
  REQUIRE(plan.n_min_per_rank >= 1);
  REQUIRE(plan.optimal_rank_count >= 1);
  const std::uint64_t total = plan.total_zones();
  REQUIRE(plan.canonical_order.size() == total);
  REQUIRE(is_permutation_of_0_to_n(plan.canonical_order));
  const double w = cutoff + skin;
  for (int ax = 0; ax < 3; ++ax) {
    // Planners enforce `zone_size >= w` whenever that axis carries ≥1
    // zones; the only escape hatch is the length < w degenerate path
    // which throws before we see a plan.
    CHECK(plan.zone_size[static_cast<std::size_t>(ax)] >= w - 1e-12);
  }
  REQUIRE(plan.cutoff == cutoff);
  REQUIRE(plan.skin == skin);
}

}  // namespace

TEST_CASE("Property — Linear1D invariants over 1e5 fuzz cases", "[zoning][property][linear1d]") {
  tf::FuzzGenerator gen;
  tz::Linear1DZoningPlanner planner;
  tz::PerformanceHint hint;
  std::size_t checked = 0;
  for (std::size_t i = 0; i < kCasesPerScheme; ++i) {
    const auto tc = gen.next_linear1d();
    tz::ZoningPlan plan;
    try {
      plan = planner.plan(tc.box, tc.cutoff, tc.skin, tc.n_ranks, hint);
    } catch (const tz::ZoningPlanError&) {
      continue;  // Edge case — generator can produce boxes too small; skip.
    }
    check_core_invariants(plan, tc.cutoff, tc.skin);
    REQUIRE(plan.scheme == tz::ZoningScheme::Linear1D);
    REQUIRE(plan.n_min_per_rank == 2);  // Andreev eq. 35
    ++checked;
  }
  // Generator is tuned to near-100% success; sanity check we didn't
  // silently skip everything due to a regression in Linear1D's throw logic.
  REQUIRE(checked > kCasesPerScheme * 9 / 10);
}

TEST_CASE("Property — Decomp2D invariants over 1e5 fuzz cases", "[zoning][property][decomp2d]") {
  tf::FuzzGenerator gen;
  tz::Decomp2DZoningPlanner planner;
  tz::PerformanceHint hint;
  std::size_t checked = 0;
  for (std::size_t i = 0; i < kCasesPerScheme; ++i) {
    const auto tc = gen.next_decomp2d();
    tz::ZoningPlan plan;
    try {
      plan = planner.plan(tc.box, tc.cutoff, tc.skin, tc.n_ranks, hint);
    } catch (const tz::ZoningPlanError&) {
      continue;
    }
    check_core_invariants(plan, tc.cutoff, tc.skin);
    REQUIRE(plan.scheme == tz::ZoningScheme::Decomp2D);
    const std::uint32_t a = plan.n_zones[0];
    const std::uint32_t b = plan.n_zones[1];
    const std::uint32_t c = plan.n_zones[2];
    // Find the two non-trivial axes; inner = smaller, outer = larger.
    std::array<std::uint32_t, 3> n = {a, b, c};
    std::sort(n.begin(), n.end());  // [trivial=1, inner, outer]
    const std::uint64_t n_inner = n[1];
    REQUIRE(plan.n_min_per_rank == 2ull * (n_inner + 1ull));  // Andreev eq. 43
    ++checked;
  }
  REQUIRE(checked > kCasesPerScheme * 9 / 10);
}

TEST_CASE("Property — Hilbert3D invariants over 1e5 fuzz cases", "[zoning][property][hilbert3d]") {
  tf::FuzzGenerator gen;
  tz::Hilbert3DZoningPlanner planner;
  tz::PerformanceHint hint;
  std::size_t checked = 0;
  for (std::size_t i = 0; i < kCasesPerScheme; ++i) {
    const auto tc = gen.next_hilbert3d();
    tz::ZoningPlan plan;
    try {
      plan = planner.plan(tc.box, tc.cutoff, tc.skin, tc.n_ranks, hint);
    } catch (const tz::ZoningPlanError&) {
      continue;
    }
    check_core_invariants(plan, tc.cutoff, tc.skin);
    REQUIRE(plan.scheme == tz::ZoningScheme::Hilbert3D);
    const std::uint64_t nx = plan.n_zones[0];
    const std::uint64_t ny = plan.n_zones[1];
    const std::uint64_t nz = plan.n_zones[2];
    const std::uint64_t xy = nx * ny;
    const std::uint64_t yz = ny * nz;
    const std::uint64_t xz = nx * nz;
    const std::uint64_t face_max = std::max({xy, yz, xz});
    REQUIRE(plan.n_min_per_rank == 4ull * face_max);  // SPEC §3.3
    ++checked;
  }
  REQUIRE(checked > kCasesPerScheme * 9 / 10);
}

TEST_CASE("Property — determinism (same input → bit-exact plan)",
          "[zoning][property][determinism]") {
  // Sample 50 cases per scheme; for each, plan twice and check every
  // field of the ZoningPlan bit-matches. 50 is enough: determinism is a
  // structural property — if it fails at all it fails immediately.
  auto check_det = [](auto&& planner_factory, auto&& case_factory, const char* scheme_name) {
    INFO("determinism for scheme " << scheme_name);
    tf::FuzzGenerator gen;
    tz::PerformanceHint hint;
    for (std::size_t i = 0; i < 50; ++i) {
      const auto tc = case_factory(gen);
      auto planner1 = planner_factory();
      auto planner2 = planner_factory();
      try {
        auto p1 = planner1.plan(tc.box, tc.cutoff, tc.skin, tc.n_ranks, hint);
        auto p2 = planner2.plan(tc.box, tc.cutoff, tc.skin, tc.n_ranks, hint);
        REQUIRE(p1.scheme == p2.scheme);
        REQUIRE(p1.n_zones == p2.n_zones);
        REQUIRE(p1.zone_size == p2.zone_size);
        REQUIRE(p1.n_min_per_rank == p2.n_min_per_rank);
        REQUIRE(p1.optimal_rank_count == p2.optimal_rank_count);
        REQUIRE(p1.canonical_order == p2.canonical_order);
      } catch (const tz::ZoningPlanError&) {
        continue;
      }
    }
  };

  check_det([]() { return tz::Linear1DZoningPlanner{}; },
            [](tf::FuzzGenerator& g) { return g.next_linear1d(); },
            "Linear1D");
  check_det([]() { return tz::Decomp2DZoningPlanner{}; },
            [](tf::FuzzGenerator& g) { return g.next_decomp2d(); },
            "Decomp2D");
  check_det([]() { return tz::Hilbert3DZoningPlanner{}; },
            [](tf::FuzzGenerator& g) { return g.next_hilbert3d(); },
            "Hilbert3D");
}
