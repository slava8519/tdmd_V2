#pragma once

// SPEC: docs/specs/zoning/SPEC.md §3.2 (Scheme B — Decomp2D), §4.2 (zigzag)
// Exec pack: docs/development/m3_execution_pack.md T3.4
// Andreev dissertation §2.4, eq. 43 (N_min), eq. 44 (n_opt), eq. 45 (anchor)
//
// Scheme B — two-axis decomposition with zigzag (snake) traversal.
// Picks the two box axes with the most zones; the third collapses to a
// single zone. Of the two active axes, the *outer* (slower) axis is the
// one with more zones, the *inner* (faster) axis the one with fewer.
// This gives the tightest N_min = 2·(N_inner + 1) per Andreev eq. 43.
//
// canonical_order is the zigzag snake: outer rows fill left-to-right,
// right-to-left, left-to-right, ... . ZoneId encoding follows the lex
// convention shared with Linear1D: `flat = ix + iy·Nx + iz·Nx·Ny`.

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/planner.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <cstdint>
#include <string>

namespace tdmd::zoning {

class Decomp2DZoningPlanner final : public ZoningPlanner {
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

  // Axis-triple chosen by the planner. `outer` and `inner` index into
  // {0,1,2} (x/y/z); `trivial` is the axis collapsed to a single zone.
  // Exposed for DefaultZoningPlanner (T3.6) and tests to inspect without
  // re-deriving from n_zones.
  struct AxisAssignment {
    int outer;
    int inner;
    int trivial;
  };

  [[nodiscard]] static AxisAssignment choose_axes(const tdmd::Box& box, double cutoff, double skin);
};

}  // namespace tdmd::zoning
