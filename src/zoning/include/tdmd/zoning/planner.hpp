#pragma once

// SPEC: docs/specs/zoning/SPEC.md §2.2 (main interface), §2.3 (helper queries)
// Master spec: §12.3
// Exec pack: docs/development/m3_execution_pack.md T3.2
//
// Abstract `ZoningPlanner` contract. Implementations land in T3.3–T3.6:
//
//   T3.3 — Linear1DZoningPlanner
//   T3.4 — Decomp2DZoningPlanner
//   T3.5 — Hilbert3DZoningPlanner
//   T3.6 — DefaultZoningPlanner (auto-selects between the above three)
//
// The interface is deliberately side-effect-free: `plan()` is a pure
// function of inputs, producing one `ZoningPlan` and throwing on
// impossible-to-satisfy constraints. No caching, no RNG, no I/O.

#include "tdmd/state/box.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tdmd::zoning {

// Thrown when the planner cannot produce a valid plan — e.g. the requested
// box is smaller than three zones along every axis (SPEC §3.4: "box too
// small for TD; use SD-vacuum mode").
class ZoningPlanError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class ZoningPlanner {
public:
  virtual ~ZoningPlanner() = default;

  // Produce a complete plan. `hint` is advisory (see `PerformanceHint`).
  // `n_ranks` is the user-requested parallelism; when `n_ranks > n_opt`
  // the planner emits an advisory warning but still returns a valid plan
  // — the runtime decides whether to downgrade to Pattern 2 (M7+).
  virtual ZoningPlan plan(const tdmd::Box& box,
                          double cutoff,
                          double skin,
                          std::uint64_t n_ranks,
                          const PerformanceHint& hint) const = 0;

  // Force a specific scheme; useful for testing, benchmarking, and future
  // user-driven overrides in `tdmd.yaml`. The concrete planner is free to
  // fall back to another scheme if the forced one cannot be satisfied on
  // the given box — but it MUST document the fallback via stderr warning.
  virtual ZoningPlan plan_with_scheme(const tdmd::Box& box,
                                      double cutoff,
                                      double skin,
                                      ZoningScheme forced_scheme,
                                      const PerformanceHint& hint) const = 0;

  // Predict-only queries — cheap enough to call from `tdmd explain --zoning`
  // without materialising the full `canonical_order` vector.
  [[nodiscard]] virtual std::uint64_t estimate_n_min(ZoningScheme scheme,
                                                     const tdmd::Box& box,
                                                     double cutoff,
                                                     double skin) const = 0;

  [[nodiscard]] virtual std::uint64_t estimate_optimal_ranks(ZoningScheme scheme,
                                                             const tdmd::Box& box,
                                                             double cutoff,
                                                             double skin) const = 0;

  // v9+ stub — validates a caller-supplied `ZoningPlan`. M3 implementations
  // throw "not implemented" per SPEC §10 OQ-5 decision (OQ-M3-2); M9+
  // implements for adaptive re-zoning.
  virtual bool validate_manual_plan(const ZoningPlan& plan,
                                    std::string& reason_if_invalid) const = 0;
};

}  // namespace tdmd::zoning
