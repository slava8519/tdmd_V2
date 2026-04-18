// SPEC: docs/specs/zoning/SPEC.md §3.2, §4.2
// Exec pack: docs/development/m3_execution_pack.md T3.4
//
// Known open question — OQ-M3-5: dissertation §8.3 anchor table lists
// `2D 16×5 → n_opt=13`; the §3.2 formula `N_min = 2·(N_inner+1)` yields
// N_min=12, n_opt=6 for that configuration. Either §3.2 or §8.3 is
// mis-calibrated. T3.4 implements §3.2 as written; T3.7 reconciles via
// SPEC delta (the formula governs; the anchor value is secondary and
// will be updated to match the formula, or vice versa pending dissertation
// re-read).

#include "tdmd/zoning/decomp2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

Decomp2DZoningPlanner::AxisAssignment Decomp2DZoningPlanner::choose_axes(const tdmd::Box& box,
                                                                         double cutoff,
                                                                         double skin) {
  const std::array<std::uint32_t, 3> n = {
      n_zones_along(box.lx(), cutoff, skin),
      n_zones_along(box.ly(), cutoff, skin),
      n_zones_along(box.lz(), cutoff, skin),
  };
  // Sort axis indices by zone count descending; stable tie-break by
  // axis index ascending. `order[0]` → outer (most zones), `order[1]`
  // → inner, `order[2]` → trivial. Manual 3-element sort so the
  // tie-break rule is unambiguous.
  std::array<int, 3> order = {0, 1, 2};
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return n[static_cast<std::size_t>(a)] > n[static_cast<std::size_t>(b)];
  });
  // Need at least two axes with ≥ 2 zones for zigzag to have > 1
  // inner-loop step. A single-column pipeline collapses to Linear1D
  // (and its N_min = 2 is lower anyway).
  if (n[static_cast<std::size_t>(order[0])] < 2 || n[static_cast<std::size_t>(order[1])] < 2) {
    throw ZoningPlanError(
        "Decomp2D: box admits fewer than two axes with ≥ 2 zones — "
        "use Linear1D or enlarge the box");
  }
  return AxisAssignment{order[0], order[1], order[2]};
}

ZoningPlan Decomp2DZoningPlanner::plan(const tdmd::Box& box,
                                       double cutoff,
                                       double skin,
                                       std::uint64_t n_ranks,
                                       const PerformanceHint& /*hint*/) const {
  if (cutoff <= 0.0 || skin < 0.0) {
    throw ZoningPlanError("Decomp2D: cutoff must be > 0 and skin >= 0");
  }
  const auto axes = choose_axes(box, cutoff, skin);
  const auto n_outer = n_zones_along(box.length(axes.outer), cutoff, skin);
  const auto n_inner = n_zones_along(box.length(axes.inner), cutoff, skin);

  ZoningPlan plan;
  plan.scheme = ZoningScheme::Decomp2D;
  plan.n_zones = {1, 1, 1};
  plan.n_zones[static_cast<std::size_t>(axes.outer)] = n_outer;
  plan.n_zones[static_cast<std::size_t>(axes.inner)] = n_inner;
  // Trivial axis stays at 1.

  const double w = cutoff + skin;
  plan.zone_size[0] = box.lx();
  plan.zone_size[1] = box.ly();
  plan.zone_size[2] = box.lz();
  plan.zone_size[static_cast<std::size_t>(axes.outer)] = box.length(axes.outer) / n_outer;
  plan.zone_size[static_cast<std::size_t>(axes.inner)] = box.length(axes.inner) / n_inner;
  // Floating-point rounding safety net — same rationale as Linear1D.
  for (int ax : {axes.outer, axes.inner}) {
    if (plan.zone_size[static_cast<std::size_t>(ax)] < w) {
      plan.zone_size[static_cast<std::size_t>(ax)] = w;
    }
  }

  // N_min formula per Andreev eq. 43 as transcribed in SPEC §3.2.
  plan.n_min_per_rank = 2ull * (static_cast<std::uint64_t>(n_inner) + 1ull);
  const std::uint64_t total = static_cast<std::uint64_t>(n_outer) * n_inner;
  plan.optimal_rank_count = total / plan.n_min_per_rank;
  if (plan.optimal_rank_count == 0) {
    plan.optimal_rank_count = 1;
  }

  // Zigzag canonical order over (outer, inner). The flat ZoneId uses
  // the shared lex convention `flat = ix + iy·Nx + iz·Nx·Ny`. With the
  // trivial axis pinned to index 0, only the active axes contribute;
  // `stride[ax]` below gives the lex multiplier for each box axis.
  const std::array<std::uint64_t, 3> stride = {
      1ull,
      static_cast<std::uint64_t>(plan.n_zones[0]),
      static_cast<std::uint64_t>(plan.n_zones[0]) * plan.n_zones[1],
  };
  plan.canonical_order.reserve(total);
  for (std::uint32_t io = 0; io < n_outer; ++io) {
    const bool reverse = (io % 2u) != 0u;
    for (std::uint32_t k = 0; k < n_inner; ++k) {
      const std::uint32_t ii = reverse ? (n_inner - 1u - k) : k;
      const std::uint64_t flat = io * stride[static_cast<std::size_t>(axes.outer)] +
                                 ii * stride[static_cast<std::size_t>(axes.inner)];
      plan.canonical_order.push_back(static_cast<ZoneId>(flat));
    }
  }

  plan.buffer_width = {skin, skin, skin};
  plan.cutoff = cutoff;
  plan.skin = skin;

  (void) n_ranks;  // advisory only — warning lives in DefaultZoningPlanner.
  return plan;
}

ZoningPlan Decomp2DZoningPlanner::plan_with_scheme(const tdmd::Box& box,
                                                   double cutoff,
                                                   double skin,
                                                   ZoningScheme forced_scheme,
                                                   const PerformanceHint& hint) const {
  if (forced_scheme != ZoningScheme::Decomp2D) {
    throw ZoningPlanError("Decomp2DZoningPlanner cannot satisfy non-Decomp2D forced scheme");
  }
  return plan(box, cutoff, skin, /*n_ranks=*/1, hint);
}

std::uint64_t Decomp2DZoningPlanner::estimate_n_min(ZoningScheme scheme,
                                                    const tdmd::Box& box,
                                                    double cutoff,
                                                    double skin) const {
  if (scheme != ZoningScheme::Decomp2D) {
    throw ZoningPlanError("Decomp2DZoningPlanner::estimate_n_min called with other scheme");
  }
  const auto axes = choose_axes(box, cutoff, skin);
  const auto n_inner = n_zones_along(box.length(axes.inner), cutoff, skin);
  return 2ull * (static_cast<std::uint64_t>(n_inner) + 1ull);
}

std::uint64_t Decomp2DZoningPlanner::estimate_optimal_ranks(ZoningScheme scheme,
                                                            const tdmd::Box& box,
                                                            double cutoff,
                                                            double skin) const {
  if (scheme != ZoningScheme::Decomp2D) {
    throw ZoningPlanError("Decomp2DZoningPlanner::estimate_optimal_ranks called with other scheme");
  }
  const auto axes = choose_axes(box, cutoff, skin);
  const auto n_outer = n_zones_along(box.length(axes.outer), cutoff, skin);
  const auto n_inner = n_zones_along(box.length(axes.inner), cutoff, skin);
  const std::uint64_t total = static_cast<std::uint64_t>(n_outer) * n_inner;
  const std::uint64_t nmin = 2ull * (static_cast<std::uint64_t>(n_inner) + 1ull);
  const auto opt = total / nmin;
  return opt == 0 ? 1ull : opt;
}

bool Decomp2DZoningPlanner::validate_manual_plan(const ZoningPlan& /*plan*/,
                                                 std::string& reason_if_invalid) const {
  reason_if_invalid = "validate_manual_plan: not implemented in M3 (SPEC §10 OQ-5)";
  return false;
}

}  // namespace tdmd::zoning
