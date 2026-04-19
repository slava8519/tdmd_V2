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
#include "tdmd/scheduler/event_log.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/scheduler/retry_state.hpp"
#include "tdmd/scheduler/safety_certificate.hpp"
#include "tdmd/scheduler/td_scheduler.hpp"
#include "tdmd/scheduler/types.hpp"
#include "tdmd/scheduler/zone_dag.hpp"
#include "tdmd/scheduler/zone_meta.hpp"
#include "tdmd/scheduler/zone_state_machine.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Forward declare CommBackend — scheduler only holds a pointer, so a full
// include is unnecessary here and keeps the scheduler/comm dependency
// one-directional (scheduler dispatches into comm, never the reverse).
namespace tdmd::comm {
class CommBackend;
}  // namespace tdmd::comm

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
  // frontier, retry budget, last 100 events from the EventLog tail).
  // Callers typically don't need this — the watchdog invokes it
  // internally — but T4.10/T4.11 acceptance tests assert fields directly.
  [[nodiscard]] DiagnosticReport make_diagnostic_report() const;

  // Full event history. T4.10 determinism tests snapshot this to compare
  // two runs byte-for-byte. Buffer holds up to EventLog::kCapacity (1024)
  // events; diagnostic dumps surface only the last 100 per SPEC §8.3.
  [[nodiscard]] const EventLog& event_log() const noexcept { return events_; }

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

  // --- T5.7 peer dispatch ----------------------------------------------
  //
  // `set_comm_backend(b)` injects the transport used by `commit_completed`
  // to move a zone's state to its downstream peer. Nullptr disables peer
  // dispatch and restores the M4 Pattern 1 short-circuit (every Completed
  // zone commits via `commit_completed_no_peer`). Ownership stays with the
  // caller — the engine owns the backend and outlives the scheduler.
  //
  // `set_peer_routing(r)` sets `r[zone_id] = dest_rank` for zones that
  // must forward their state on commit (or -1 for zones that stay local).
  // The vector must have size == total_zones() at call time; a mismatched
  // size or use of negative values other than -1 throws.
  //
  // `poll_arrivals()` is the receiver-side pump — it calls
  // `backend->progress()` then drains every arrived `TemporalPacket` and
  // fires `on_zone_data_arrived` for the matching zone. CRC / protocol-
  // version drops are counted by the backend; cert-hash drops are counted
  // by the scheduler (they trigger a retry via `invalidate_certificates_for`).
  void set_comm_backend(comm::CommBackend* backend) noexcept;
  void set_peer_routing(std::vector<int> routing);
  void poll_arrivals();

  [[nodiscard]] std::uint64_t dropped_cert_hash_count() const noexcept {
    return dropped_cert_hash_;
  }

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

  // T5.7 peer dispatch. `peer_routing_[z] = -1` means zone z has no
  // off-rank downstream peer and commits via the Pattern 1 short-circuit.
  // Any non-negative entry names the dest MPI rank the scheduler sends to
  // when the zone enters Completed.
  comm::CommBackend* comm_backend_ = nullptr;
  std::vector<int> peer_routing_;
  std::uint64_t dropped_cert_hash_ = 0;

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

  // Event log (EventLog::kCapacity = 1024, per OQ-M4-4). DiagnosticReport
  // surfaces only the last 100 — the extra buffer is headroom for T4.10
  // determinism tests and post-mortem tooling.
  EventLog events_{};

  // Record an event via the EventLog. All scheduler event handlers funnel
  // through this helper so cross-run byte-exactness is a property of the
  // call sequence alone (timestamps excepted — ignored by determinism tests).
  void record_event(SchedulerEvent kind,
                    ZoneId zone_id = 0xFFFFFFFFU,
                    TimeLevel time_level = 0,
                    std::uint32_t count = 0);

  // If local_frontier_min() > last_frontier_min_, bump last_progress_ and
  // update last_frontier_min_. Called after each event-processing entry.
  void maybe_update_frontier_progress();
};

}  // namespace tdmd::scheduler
