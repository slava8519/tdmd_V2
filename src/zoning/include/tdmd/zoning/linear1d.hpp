#pragma once

// SPEC: docs/specs/zoning/SPEC.md §3.1 (Scheme A — Linear1D)
// Exec pack: docs/development/m3_execution_pack.md T3.3
// Andreev dissertation §2.2, eq. 35
//
// Scheme A — single-axis decomposition. The simplest of the three TDMD
// schemes and the one that underpins the M5 anchor-test: Andreev's own
// 10⁶-atom Al FCC run was Linear1D. `N_min = 2` regardless of system
// size; `canonical_order` is sequential [0, 1, ..., N-1].
//
// The planner picks the longest axis automatically (ties break on axis
// index, lowest first). Callers who want a specific axis can still force
// it via `DefaultZoningPlanner::plan_with_scheme` (M3.6).

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/planner.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <cstdint>
#include <string>

namespace tdmd::zoning {

class Linear1DZoningPlanner final : public ZoningPlanner {
public:
  ZoningPlan plan(const tdmd::Box& box,
                  double cutoff,
                  double skin,
                  std::uint64_t n_ranks,
                  const PerformanceHint& hint) const override;

  ZoningPlan plan_with_scheme(const tdmd::Box& box,
                              double cutoff,
                              double skin,
                              ZoningScheme forced_scheme,
                              const PerformanceHint& hint) const override;

  [[nodiscard]] std::uint64_t estimate_n_min(ZoningScheme scheme,
                                             const tdmd::Box& box,
                                             double cutoff,
                                             double skin) const override;

  [[nodiscard]] std::uint64_t estimate_optimal_ranks(ZoningScheme scheme,
                                                     const tdmd::Box& box,
                                                     double cutoff,
                                                     double skin) const override;

  bool validate_manual_plan(const ZoningPlan& plan, std::string& reason_if_invalid) const override;

  // Expose the axis picker for the M3.6 DefaultZoningPlanner to avoid
  // duplicating the aspect-ratio tie-break logic. Return 0/1/2 for x/y/z.
  [[nodiscard]] static int choose_axis(const tdmd::Box& box, double cutoff, double skin) noexcept;
};

}  // namespace tdmd::zoning
