#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §2.2 (main interface)
// Master spec: §12.4
// Exec pack: docs/development/m4_execution_pack.md T4.2
//
// Abstract `TdScheduler`. Mirrors the SPEC signature verbatim; concrete
// `CausalWavefrontScheduler` lands in T4.5+. `ZoningPlan` and
// `OuterSdCoordinator` are forward-declared so this header stays free of
// build-time deps on zoning/ and comm/ (the former lands via
// td_scheduler.cpp implementations, the latter is Pattern 2 / M7+ scope).

#include "tdmd/scheduler/types.hpp"

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

// Forward decls — see above.
namespace tdmd::zoning {
struct ZoningPlan;
}

namespace tdmd::scheduler {

// Pattern 2 coordinator. M4 carried only a virtual-dtor stub here so that
// `attach_outer_coordinator(nullptr)` was a legal call without pulling in
// a comm / state dependency. T7.6 lands the full pure-virtual interface
// in `outer_sd_coordinator.hpp`. The forward declaration here keeps the
// public TdScheduler header free of `tdmd::state` / SubdomainGrid; clients
// that actually construct or call methods on the coordinator must include
// `tdmd/scheduler/outer_sd_coordinator.hpp` directly.
class OuterSdCoordinator;

class TdScheduler {
public:
  virtual ~TdScheduler() = default;

  // Lifecycle:
  virtual void initialize(const tdmd::zoning::ZoningPlan& plan) = 0;
  virtual void attach_outer_coordinator(OuterSdCoordinator* coord) = 0;  // nullable
  virtual void shutdown() = 0;

  // Certificate management:
  virtual void refresh_certificates() = 0;
  virtual void invalidate_certificates_for(ZoneId zone) = 0;
  virtual void invalidate_all_certificates(const std::string& reason) = 0;

  // Task selection (one iteration):
  [[nodiscard]] virtual std::vector<ZoneTask> select_ready_tasks() = 0;

  // Task lifecycle callbacks:
  virtual void mark_computing(const ZoneTask& task) = 0;
  virtual void mark_completed(const ZoneTask& task) = 0;
  virtual void mark_packed(const ZoneTask& task) = 0;
  virtual void mark_inflight(const ZoneTask& task) = 0;
  virtual void mark_committed(const ZoneTask& task) = 0;

  // Commit protocol (two-phase, §6):
  virtual void commit_completed() = 0;

  // Introspection:
  [[nodiscard]] virtual bool finished() const = 0;
  [[nodiscard]] virtual std::size_t min_zones_per_rank() const = 0;
  [[nodiscard]] virtual std::size_t optimal_rank_count(std::size_t total_zones) const = 0;
  [[nodiscard]] virtual std::size_t current_pipeline_depth() const = 0;
  [[nodiscard]] virtual TimeLevel local_frontier_min() const = 0;
  [[nodiscard]] virtual TimeLevel local_frontier_max() const = 0;

  // Events from comm / neighbor (§10.1):
  virtual void on_zone_data_arrived(ZoneId zone, TimeLevel step, Version version) = 0;
  virtual void on_halo_arrived(std::uint32_t peer_subdomain, TimeLevel step) = 0;
  virtual void on_neighbor_rebuild_completed(const std::vector<ZoneId>& affected) = 0;

  // Watchdog (§8):
  virtual void check_deadlock(std::chrono::milliseconds t_watchdog) = 0;
};

}  // namespace tdmd::scheduler
