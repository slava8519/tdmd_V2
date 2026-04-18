#pragma once

// SPEC: docs/specs/zoning/SPEC.md §3.4 (scheme selection)
// Exec pack: docs/development/m3_execution_pack.md T3.6
//
// The default ZoningPlanner — picks scheme by box aspect ratio
// following SPEC §3.4's decision tree, then delegates to the matching
// concrete planner (`Linear1DZoningPlanner`, `Decomp2DZoningPlanner`,
// `Hilbert3DZoningPlanner`). Emits a soft advisory when
// `n_ranks > 1.2·n_opt` (fed back through `ZoningPlan::advisories`);
// real telemetry hookup lands at M4 when the scheduler consumes plans.

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/decomp2d.hpp"
#include "tdmd/zoning/hilbert3d.hpp"
#include "tdmd/zoning/linear1d.hpp"
#include "tdmd/zoning/planner.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <cstdint>
#include <string>

namespace tdmd::zoning {

class DefaultZoningPlanner final : public ZoningPlanner {
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

  // Expose the scheme-selection decision so `tdmd explain --zoning`
  // (T3.9) can report "chose Decomp2D because max_ax/min_ax = 4" to
  // users without re-running the dispatch logic.
  [[nodiscard]] static ZoningScheme select_scheme(const tdmd::Box& box, double cutoff, double skin);

private:
  Linear1DZoningPlanner linear1d_;
  Decomp2DZoningPlanner decomp2d_;
  Hilbert3DZoningPlanner hilbert3d_;
};

}  // namespace tdmd::zoning
