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
#include "tdmd/scheduler/policy.hpp"
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

  // Configuration hooks. T4.9 (SimulationEngine wiring) uses these; tests
  // use them to stub physics inputs and drive finished().
  void set_certificate_input_source(const CertificateInputSource* src) noexcept;
  void set_target_time_level(TimeLevel target) noexcept;
  [[nodiscard]] TimeLevel target_time_level() const noexcept;

private:
  void require_initialized(const char* op) const;

  SchedulerPolicy policy_;
  ZoneStateMachine state_machine_;
  CertificateStore cert_store_;

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

  // Deadlock bookkeeping (T4.8 will grow this). The progress timestamp is
  // advanced by mark_completed / commit_completed / refresh_certificates /
  // on_zone_data_arrived / on_neighbor_rebuild_completed.
  std::chrono::steady_clock::time_point last_progress_{std::chrono::steady_clock::now()};
};

}  // namespace tdmd::scheduler
