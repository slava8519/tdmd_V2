#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §2.3 (canonical realization), §9 (queues)
// Master spec: §6.3, §6.6, §12.4
// Exec pack: docs/development/m4_execution_pack.md T4.5 (core), T4.6+ (select/commit/watchdog)
//
// Concrete `TdScheduler` realization. T4.5 lands:
//   - lifecycle (initialize / attach_outer_coordinator / shutdown)
//   - canonical_order copy from ZoningPlan
//   - spatial dependency DAG (from zone_dag.hpp)
//   - proactive refresh_certificates producing one cert per
//     (zone, meta.time_level + 1)
//   - event handlers (on_zone_data_arrived, on_neighbor_rebuild_completed)
//   - mark_* forwards to ZoneStateMachine
//   - introspection (finished / frontier / pipeline depth / zone counts)
//
// T4.5 leaves as minimal stubs (bodies grow in T4.6–T4.8):
//   - select_ready_tasks — returns empty
//   - commit_completed — Pattern 1 sweep (commit_completed_no_peer on each
//     Completed zone). This is correct for M4 single-rank (D-M4-6) but the
//     retry/I5 hardening and explicit Phase-B arrive in T4.7.
//   - check_deadlock — records a progress timestamp, never throws.

#include "tdmd/scheduler/certificate_input_source.hpp"
#include "tdmd/scheduler/certificate_store.hpp"
#include "tdmd/scheduler/diagnostic_dump.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/scheduler/retry_state.hpp"
#include "tdmd/scheduler/safety_certificate.hpp"
#include "tdmd/scheduler/td_scheduler.hpp"
#include "tdmd/scheduler/types.hpp"
#include "tdmd/scheduler/zone_dag.hpp"
#include "tdmd/scheduler/zone_meta.hpp"
#include "tdmd/scheduler/zone_state_machine.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tdmd::scheduler {

class CausalWavefrontScheduler final : public TdScheduler {
public:
  explicit CausalWavefrontScheduler(SchedulerPolicy policy);
  ~CausalWavefrontScheduler() override;

  CausalWavefrontScheduler(const CausalWavefrontScheduler&) = delete;
  CausalWavefrontScheduler& operator=(const CausalWavefrontScheduler&) = delete;
  CausalWavefrontScheduler(CausalWavefrontScheduler&&) = delete;
  CausalWavefrontScheduler& operator=(CausalWavefrontScheduler&&) = delete;

  // --- TdScheduler overrides ---------------------------------------------

  void initialize(const tdmd::zoning::ZoningPlan& plan) override;
  void attach_outer_coordinator(OuterSdCoordinator* coord) override;
  void shutdown() override;

  void refresh_certificates() override;
  void invalidate_certificates_for(ZoneId zone) override;
  void invalidate_all_certificates(const std::string& reason) override;

  [[nodiscard]] std::vector<ZoneTask> select_ready_tasks() override;

  void mark_computing(const ZoneTask& task) override;
  void mark_completed(const ZoneTask& task) override;
  void mark_packed(const ZoneTask& task) override;
  void mark_inflight(const ZoneTask& task) override;
  void mark_committed(const ZoneTask& task) override;

  void commit_completed() override;

  [[nodiscard]] bool finished() const override;
  [[nodiscard]] std::size_t min_zones_per_rank() const override;
  [[nodiscard]] std::size_t optimal_rank_count(std::size_t total_zones) const override;
  [[nodiscard]] std::size_t current_pipeline_depth() const override;
  [[nodiscard]] TimeLevel local_frontier_min() const override;
  [[nodiscard]] TimeLevel local_frontier_max() const override;

  void on_zone_data_arrived(ZoneId zone, TimeLevel step, Version version) override;
  void on_halo_arrived(std::uint32_t peer_subdomain, TimeLevel step) override;
  void on_neighbor_rebuild_completed(const std::vector<ZoneId>& affected) override;

  void check_deadlock(std::chrono::milliseconds t_watchdog) override;

  // --- T4.5 extensions (not in abstract base) ---------------------------
  //
  // These are the test / engine-wiring surface. They do not appear in
  // `TdScheduler` because they are implementation details of the
  // CausalWavefrontScheduler policy; other future implementations (e.g.
  // an Observed-Time scheduler in M8+) would have different knobs.

  // Read-only views of internal state. Pointers returned from ZoneMeta&
  // stay valid until the next shutdown() / initialize().
  [[nodiscard]] const std::vector<ZoneId>& canonical_order() const noexcept;
  [[nodiscard]] ZoneDepMask spatial_dep_mask(ZoneId zone) const;
  [[nodiscard]] const ZoneMeta& zone_meta(ZoneId zone) const;
  [[nodiscard]] std::size_t total_zones() const noexcept;
  [[nodiscard]] const CertificateStore& cert_store() const noexcept;
  [[nodiscard]] const SchedulerPolicy& policy() const noexcept;
  [[nodiscard]] bool initialized() const noexcept;
  [[nodiscard]] OuterSdCoordinator* outer_coordinator() const noexcept;
  [[nodiscard]] const RetryTracker& retry_tracker() const noexcept;

  // Compose a DiagnosticReport from the current state (zone histogram,
  // frontier, retry budget, last kEventRingCapacity events). Callers
  // typically don't need this — the watchdog invokes it internally —
  // but T4.10/T4.11 acceptance tests assert fields directly.
  [[nodiscard]] DiagnosticReport make_diagnostic_report() const;

  // Configuration hooks. T4.9 (SimulationEngine wiring) uses these; tests
  // use them to stub physics inputs and drive finished().
  void set_certificate_input_source(const CertificateInputSource* src) noexcept;
  void set_target_time_level(TimeLevel target) noexcept;
  [[nodiscard]] TimeLevel target_time_level() const noexcept;

  // Release every Committed zone back to Empty — the engine's cycle reset
  // between Phase B (commit_completed) and the next `on_zone_data_arrived`.
  // Kept out of the abstract interface because the engine loop (T4.9) owns
  // this orchestration; tests call it to drive synthetic lifecycles.
  void release_committed();

private:
  void require_initialized(const char* op) const;

  SchedulerPolicy policy_;
  ZoneStateMachine state_machine_;
  CertificateStore cert_store_;
  RetryTracker retry_tracker_;

  // Populated by initialize(); cleared by shutdown().
  bool initialized_ = false;
  std::size_t total_zones_ = 0;
  std::vector<ZoneMeta> metas_;                // index = scheduler::ZoneId
  std::vector<ZoneId> canonical_order_;        // size == total_zones_
  std::vector<ZoneDepMask> spatial_dep_mask_;  // size == total_zones_

  const CertificateInputSource* cert_source_ = nullptr;
  OuterSdCoordinator* outer_coord_ = nullptr;

  TimeLevel target_time_level_ = 0;

  // Cached from ZoningPlan at initialize(); scheduler itself doesn't
  // retain a reference to the plan.
  std::uint64_t min_zones_per_rank_ = 1;
  std::uint64_t optimal_rank_count_cached_ = 1;

  // Deadlock bookkeeping. SPEC §8.2 defines progress as exactly one of:
  //   - dispatch (Ready → Computing, i.e. mark_computing)
  //   - inflight → Committed (mark_committed) or Pattern 1 commit
  //   - zone event processed (arrived / halo / neighbor_rebuild)
  //   - frontier_min increase
  // Cert refresh + invalidations deliberately do NOT bump the watchdog —
  // a retry storm should trigger the watchdog just like any other stall.
  std::chrono::steady_clock::time_point last_progress_{std::chrono::steady_clock::now()};
  TimeLevel last_frontier_min_{0};

  // Event ring buffer for diagnostic_dump. kEventRingCapacity matches
  // SPEC §8.3's "last 100 events". Implemented as a fixed array + head
  // pointer to avoid allocations during steady-state scheduling.
  static constexpr std::size_t kEventRingCapacity = 100;
  std::array<EventRecord, kEventRingCapacity> event_ring_{};
  std::size_t event_ring_head_ = 0;   // next write slot
  std::size_t event_ring_count_ = 0;  // live entries, capped at capacity

  // Record an event into the ring. Oldest-first order is reconstructed at
  // dump time by walking from (head - count) mod capacity. All scheduler
  // event handlers funnel through this helper.
  void record_event(SchedulerEvent kind,
                    ZoneId zone_id = 0xFFFFFFFFU,
                    TimeLevel time_level = 0,
                    std::uint32_t count = 0);

  // If local_frontier_min() > last_frontier_min_, bump last_progress_ and
  // update last_frontier_min_. Called after each event-processing entry.
  void maybe_update_frontier_progress();
};

}  // namespace tdmd::scheduler
