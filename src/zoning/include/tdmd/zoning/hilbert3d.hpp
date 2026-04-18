#pragma once

// SPEC: docs/specs/zoning/SPEC.md §3.3 (Scheme C — Hilbert3D), §4.2
// Exec pack: docs/development/m3_execution_pack.md T3.5
//
// Scheme C — 3D space-filling curve via Skilling 2004 Hilbert. The
// planner pads the box's (Nx, Ny, Nz) zone grid up to the next power
// of 2, walks the Hilbert curve on that padded cube, and emits a
// `canonical_order` containing only the in-box flat ZoneIds (lex
// encoded as `ix + iy·Nx + iz·Nx·Ny`). Boundary-filter preserves
// locality at the expense of some fragmentation at padding edges.
//
// N_min formula per §3.3: `4·max(Nx·Ny, Ny·Nz, Nx·Nz)`. This is the
// empirical envelope for standard Hilbert walks on non-trivially-sized
// 3D grids.

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/planner.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <cstdint>
#include <string>

namespace tdmd::zoning {

class Hilbert3DZoningPlanner final : public ZoningPlanner {
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
};

}  // namespace tdmd::zoning
