// SPEC: docs/specs/zoning/SPEC.md §3.1
// Exec pack: docs/development/m3_execution_pack.md T3.3

#include "tdmd/zoning/linear1d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace tdmd::zoning {

namespace {

// N_zones along an axis: floor(L / (r_c + r_skin)). SPEC §3.4 makes this
// the canonical "how many zones fit" formula for all schemes.
std::uint32_t n_zones_along(double length, double cutoff, double skin) noexcept {
  const double w = cutoff + skin;
  if (w <= 0.0 || length < w) {
    return 0;
  }
  const auto n = static_cast<std::uint32_t>(std::floor(length / w));
  return n;
}

}  // namespace

int Linear1DZoningPlanner::choose_axis(const tdmd::Box& box, double cutoff, double skin) noexcept {
  // Zone count per axis — not raw length — so a very long axis whose
  // cutoff would place only one zone there is not preferred over a
  // shorter but multi-zone one.
  const std::array<std::uint32_t, 3> n = {
      n_zones_along(box.lx(), cutoff, skin),
      n_zones_along(box.ly(), cutoff, skin),
      n_zones_along(box.lz(), cutoff, skin),
  };
  // Pick the axis with the most zones; ties break to lowest axis index
  // (stable, deterministic — part of the canonical-order contract §4.3).
  std::size_t best = 0;
  for (std::size_t ax = 1; ax < 3; ++ax) {
    if (n[ax] > n[best]) {
      best = ax;
    }
  }
  return static_cast<int>(best);
}

ZoningPlan Linear1DZoningPlanner::plan(const tdmd::Box& box,
                                       double cutoff,
                                       double skin,
                                       std::uint64_t n_ranks,
                                       const PerformanceHint& /*hint*/) const {
  if (cutoff <= 0.0 || skin < 0.0) {
    throw ZoningPlanError("Linear1D: cutoff must be > 0 and skin >= 0");
  }
  const int axis = choose_axis(box, cutoff, skin);
  const double length = box.length(axis);
  const std::uint32_t n_along = n_zones_along(length, cutoff, skin);
  if (n_along < 2) {
    throw ZoningPlanError(
        "Linear1D: chosen axis admits fewer than 2 zones — box is too small for TD");
  }

  ZoningPlan plan;
  plan.scheme = ZoningScheme::Linear1D;
  plan.n_zones = {1, 1, 1};
  plan.n_zones[static_cast<std::size_t>(axis)] = n_along;

  // Equal-width zones along the chosen axis; the remainder goes into the
  // last zone (at most w — a few %). Non-chosen axes report the box
  // extent as a single zone so downstream consumers see a consistent
  // "zone size on axis" field.
  const double w = cutoff + skin;
  plan.zone_size[0] = box.lx();
  plan.zone_size[1] = box.ly();
  plan.zone_size[2] = box.lz();
  plan.zone_size[static_cast<std::size_t>(axis)] = length / n_along;
  // `zone_size[axis] = length/n_along` may dip slightly below `w` if the
  // caller passed a length just above `w`; enforce the invariant by
  // nudging up. This is correct because `n_along = floor(length/w)` so
  // `length/n_along >= w` always — but floating-point rounding on the
  // division can under-shoot. Documented here rather than silently
  // widening the zone.
  if (plan.zone_size[static_cast<std::size_t>(axis)] < w) {
    plan.zone_size[static_cast<std::size_t>(axis)] = w;
  }

  plan.n_min_per_rank = 2;  // Andreev eq. 35 — invariant for Linear1D.
  plan.optimal_rank_count = static_cast<std::uint64_t>(n_along / 2u);
  if (plan.optimal_rank_count == 0) {
    plan.optimal_rank_count = 1;
  }

  // canonical_order = [0, 1, ..., n_along - 1]. The non-chosen axes
  // contribute a single zone each so total_zones == n_along.
  plan.canonical_order.reserve(n_along);
  for (std::uint32_t i = 0; i < n_along; ++i) {
    plan.canonical_order.push_back(static_cast<ZoneId>(i));
  }

  plan.buffer_width = {skin, skin, skin};
  plan.cutoff = cutoff;
  plan.skin = skin;

  // n_ranks is advisory here — the planner doesn't refuse bad counts, the
  // runtime (M4+) decides whether to downgrade or go Pattern-2. The
  // warning path lives in DefaultZoningPlanner (T3.6) to keep this
  // concrete planner pure.
  (void) n_ranks;

  return plan;
}

ZoningPlan Linear1DZoningPlanner::plan_with_scheme(const tdmd::Box& box,
                                                   double cutoff,
                                                   double skin,
                                                   ZoningScheme forced_scheme,
                                                   const PerformanceHint& hint) const {
  if (forced_scheme != ZoningScheme::Linear1D) {
    throw ZoningPlanError(
        "Linear1DZoningPlanner cannot satisfy non-Linear1D forced scheme — "
        "use DefaultZoningPlanner to dispatch across schemes");
  }
  return plan(box, cutoff, skin, /*n_ranks=*/1, hint);
}

std::uint64_t Linear1DZoningPlanner::estimate_n_min(ZoningScheme scheme,
                                                    const tdmd::Box& /*box*/,
                                                    double /*cutoff*/,
                                                    double /*skin*/) const {
  if (scheme != ZoningScheme::Linear1D) {
    throw ZoningPlanError("Linear1DZoningPlanner::estimate_n_min called with other scheme");
  }
  return 2;
}

std::uint64_t Linear1DZoningPlanner::estimate_optimal_ranks(ZoningScheme scheme,
                                                            const tdmd::Box& box,
                                                            double cutoff,
                                                            double skin) const {
  if (scheme != ZoningScheme::Linear1D) {
    throw ZoningPlanError("Linear1DZoningPlanner::estimate_optimal_ranks called with other scheme");
  }
  const int axis = choose_axis(box, cutoff, skin);
  const std::uint32_t n_along = n_zones_along(box.length(axis), cutoff, skin);
  const auto opt = static_cast<std::uint64_t>(n_along / 2u);
  return opt == 0 ? 1 : opt;
}

bool Linear1DZoningPlanner::validate_manual_plan(const ZoningPlan& /*plan*/,
                                                 std::string& reason_if_invalid) const {
  // Per OQ-M3-2 decision: stubbed in M3; real implementation lands at M9+
  // alongside adaptive re-zoning.
  reason_if_invalid = "validate_manual_plan: not implemented in M3 (SPEC §10 OQ-5)";
  return false;
}

}  // namespace tdmd::zoning
