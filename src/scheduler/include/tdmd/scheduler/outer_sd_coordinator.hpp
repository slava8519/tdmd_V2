#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §2.4 (OuterSdCoordinator contract +
//       OC-1..OC-6), §4.6 (HaloSnapshot archive + HA-1..HA-5)
// Master spec: §12.7a
// Exec pack: docs/development/m7_execution_pack.md T7.6
//
// Pure-virtual outer-level coordinator for Pattern 2 (M7+). Owned by
// `SimulationEngine` in Pattern 2; `nullptr` in Pattern 1 / 3 — the inner
// `TdScheduler` accepts the null state via `attach_outer_coordinator`.
//
// This header replaces the M4 stub that lived inline in `td_scheduler.hpp`
// (D-M4-2 placeholder). The forward declaration in `td_scheduler.hpp`
// keeps `attach_outer_coordinator(nullptr)` callable without forcing the
// inner scheduler header to pull in `tdmd::state` for the SubdomainGrid.

#include "tdmd/scheduler/halo_snapshot.hpp"
#include "tdmd/scheduler/subdomain_grid.hpp"
#include "tdmd/scheduler/types.hpp"

#include <chrono>
#include <cstdint>
#include <optional>

namespace tdmd::scheduler {

class OuterSdCoordinator {
public:
  virtual ~OuterSdCoordinator() = default;

  virtual void initialize(const SubdomainGrid& grid, std::uint32_t k_max) = 0;

  // OC-1. Non-blocking, idempotent. Returns true ⟺ the archive holds a
  // snapshot at `target_level - 1` for every peer subdomain whose halo
  // intersects `local_zone`. In Pattern 1 this method is never called.
  [[nodiscard]] virtual bool can_advance_boundary_zone(ZoneId local_zone,
                                                       TimeLevel target_level) = 0;

  // OC-2. Phase B hook: register this subdomain's outgoing snapshot.
  // Exactly one snapshot per (zone, level); double-register is a hard
  // error in Reference, advisory log in Production / Fast.
  virtual void register_boundary_snapshot(ZoneId local_zone,
                                          TimeLevel level,
                                          const HaloSnapshot& snap) = 0;

  // OC-4. Non-blocking pull side. Returns std::nullopt when the requested
  // peer snapshot has not (yet) arrived.
  [[nodiscard]] virtual std::optional<HaloSnapshot>
  fetch_peer_snapshot(std::uint32_t peer_subdomain, ZoneId peer_zone, TimeLevel level) = 0;

  // OC-6. Boundary-specific watchdog, separate from the inner deadlock
  // detector (§8). Emits diagnostic dumps for boundary zones stalled
  // longer than `T_stall_max`.
  virtual void check_stall_boundaries(std::chrono::milliseconds t_stall_max) = 0;

  // OC-5. Monotonic non-decreasing — consolidated across subdomains via
  // outer-comm collectives at end-of-iteration.
  [[nodiscard]] virtual TimeLevel global_frontier_min() const = 0;
  [[nodiscard]] virtual TimeLevel global_frontier_max() const = 0;
};

}  // namespace tdmd::scheduler
