// SPEC: docs/specs/zoning/SPEC.md §3.4 (selection algorithm)
// Exec pack: docs/development/m3_execution_pack.md T3.6

#include "tdmd/zoning/default_planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>

namespace tdmd::zoning {

namespace {

std::uint32_t n_zones_along(double length, double cutoff, double skin) noexcept {
  const double w = cutoff + skin;
  if (w <= 0.0 || length < w) {
    return 0;
  }
  return static_cast<std::uint32_t>(std::floor(length / w));
}

}  // namespace

ZoningScheme DefaultZoningPlanner::select_scheme(const tdmd::Box& box, double cutoff, double skin) {
  if (cutoff <= 0.0 || skin < 0.0) {
    throw ZoningPlanError("DefaultZoningPlanner: cutoff must be > 0 and skin >= 0");
  }
  const std::array<std::uint32_t, 3> n = {
      n_zones_along(box.lx(), cutoff, skin),
      n_zones_along(box.ly(), cutoff, skin),
      n_zones_along(box.lz(), cutoff, skin),
  };
  const std::uint64_t total = static_cast<std::uint64_t>(n[0]) * n[1] * n[2];
  if (total < 3) {
    throw ZoningPlanError(
        "DefaultZoningPlanner: box admits fewer than 3 zones total — "
        "TD is not useful here; run a single-rank SD-vacuum path instead");
  }

  const std::uint32_t max_ax = *std::max_element(n.begin(), n.end());
  // `min_ax` used in the §3.4 decision tree is the smallest zone count
  // among the three axes. For thin-slab geometries this drops to 1.
  const std::uint32_t min_ax = *std::min_element(n.begin(), n.end());

  // §3.4 decision tree, verbatim:
  //   max/min > 10 and min < 4 → Linear1D
  //   max/min > 3  or  (Nx·Ny) < 16 → Decomp2D
  //   else → Hilbert3D
  // The `(Nx·Ny) < 16` clause deliberately preserves the SPEC wording
  // — it's a slightly axis-specific heuristic that T3.7 may revisit
  // if fuzz coverage flags weird choices.
  const double aspect = static_cast<double>(max_ax) / static_cast<double>(min_ax);
  const std::uint64_t xy = static_cast<std::uint64_t>(n[0]) * n[1];

  if (aspect > 10.0 && min_ax < 4) {
    return ZoningScheme::Linear1D;
  }
  if (aspect > 3.0 || xy < 16) {
    return ZoningScheme::Decomp2D;
  }
  return ZoningScheme::Hilbert3D;
}

ZoningPlan DefaultZoningPlanner::plan(const tdmd::Box& box,
                                      double cutoff,
                                      double skin,
                                      std::uint64_t n_ranks,
                                      const PerformanceHint& hint) const {
  const ZoningScheme scheme = select_scheme(box, cutoff, skin);
  ZoningPlan plan;
  switch (scheme) {
    case ZoningScheme::Linear1D:
      plan = linear1d_.plan(box, cutoff, skin, n_ranks, hint);
      break;
    case ZoningScheme::Decomp2D:
      plan = decomp2d_.plan(box, cutoff, skin, n_ranks, hint);
      break;
    case ZoningScheme::Hilbert3D:
      plan = hilbert3d_.plan(box, cutoff, skin, n_ranks, hint);
      break;
    case ZoningScheme::Manual:
      throw ZoningPlanError(
          "DefaultZoningPlanner::plan reached Manual scheme — impossible by selection");
  }

  // Advisory: n_ranks too high. SPEC §3.4 threshold = 1.2·n_opt. The
  // runtime (M4) will elevate this to a proper warning; for M3 we
  // attach a string to the plan so `tdmd explain --zoning` can show it.
  if (n_ranks > 0 &&
      static_cast<double>(n_ranks) > 1.2 * static_cast<double>(plan.optimal_rank_count)) {
    std::ostringstream msg;
    msg << "n_ranks=" << n_ranks << " exceeds 1.2·n_opt (n_opt=" << plan.optimal_rank_count
        << ") — consider Pattern 2 (M7+) or a larger box";
    plan.advisories.push_back(msg.str());
  }

  return plan;
}

ZoningPlan DefaultZoningPlanner::plan_with_scheme(const tdmd::Box& box,
                                                  double cutoff,
                                                  double skin,
                                                  ZoningScheme forced_scheme,
                                                  const PerformanceHint& hint) const {
  switch (forced_scheme) {
    case ZoningScheme::Linear1D:
      return linear1d_.plan_with_scheme(box, cutoff, skin, forced_scheme, hint);
    case ZoningScheme::Decomp2D:
      return decomp2d_.plan_with_scheme(box, cutoff, skin, forced_scheme, hint);
    case ZoningScheme::Hilbert3D:
      return hilbert3d_.plan_with_scheme(box, cutoff, skin, forced_scheme, hint);
    case ZoningScheme::Manual:
      throw ZoningPlanError(
          "DefaultZoningPlanner: Manual scheme is stubbed (OQ-M3-2 / SPEC §10 OQ-5)");
  }
  throw ZoningPlanError("DefaultZoningPlanner: unknown ZoningScheme enumerator");
}

std::uint64_t DefaultZoningPlanner::estimate_n_min(ZoningScheme scheme,
                                                   const tdmd::Box& box,
                                                   double cutoff,
                                                   double skin) const {
  switch (scheme) {
    case ZoningScheme::Linear1D:
      return linear1d_.estimate_n_min(scheme, box, cutoff, skin);
    case ZoningScheme::Decomp2D:
      return decomp2d_.estimate_n_min(scheme, box, cutoff, skin);
    case ZoningScheme::Hilbert3D:
      return hilbert3d_.estimate_n_min(scheme, box, cutoff, skin);
    case ZoningScheme::Manual:
      throw ZoningPlanError("estimate_n_min: Manual scheme unsupported in M3");
  }
  throw ZoningPlanError("estimate_n_min: unknown ZoningScheme enumerator");
}

std::uint64_t DefaultZoningPlanner::estimate_optimal_ranks(ZoningScheme scheme,
                                                           const tdmd::Box& box,
                                                           double cutoff,
                                                           double skin) const {
  switch (scheme) {
    case ZoningScheme::Linear1D:
      return linear1d_.estimate_optimal_ranks(scheme, box, cutoff, skin);
    case ZoningScheme::Decomp2D:
      return decomp2d_.estimate_optimal_ranks(scheme, box, cutoff, skin);
    case ZoningScheme::Hilbert3D:
      return hilbert3d_.estimate_optimal_ranks(scheme, box, cutoff, skin);
    case ZoningScheme::Manual:
      throw ZoningPlanError("estimate_optimal_ranks: Manual scheme unsupported in M3");
  }
  throw ZoningPlanError("estimate_optimal_ranks: unknown ZoningScheme enumerator");
}

bool DefaultZoningPlanner::validate_manual_plan(const ZoningPlan& /*plan*/,
                                                std::string& reason_if_invalid) const {
  reason_if_invalid = "validate_manual_plan: not implemented in M3 (SPEC §10 OQ-5)";
  return false;
}

}  // namespace tdmd::zoning
