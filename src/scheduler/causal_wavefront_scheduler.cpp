// SPEC: docs/specs/scheduler/SPEC.md §2.3 (canonical realization), §4.4, §9, §10
// Master spec: §6.3, §6.6, §12.4
// Exec pack: docs/development/m4_execution_pack.md T4.5

#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"

#include "tdmd/scheduler/queues.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tdmd::scheduler {

CausalWavefrontScheduler::CausalWavefrontScheduler(SchedulerPolicy policy)
    : policy_(std::move(policy)), cert_store_(policy_.mode_policy_tag) {}

CausalWavefrontScheduler::~CausalWavefrontScheduler() = default;

void CausalWavefrontScheduler::require_initialized(const char* op) const {
  if (!initialized_) {
    throw std::logic_error(std::string{"CausalWavefrontScheduler::"} + op +
                           ": scheduler not initialized");
  }
}

void CausalWavefrontScheduler::initialize(const tdmd::zoning::ZoningPlan& plan) {
  if (initialized_) {
    throw std::logic_error(
        "CausalWavefrontScheduler::initialize: already initialized; call shutdown() first");
  }

  const auto total = plan.total_zones();
  if (total == 0) {
    throw std::logic_error("CausalWavefrontScheduler::initialize: plan has zero zones");
  }
  if (plan.canonical_order.size() != total) {
    throw std::logic_error("CausalWavefrontScheduler::initialize: canonical_order size mismatch");
  }

  // zone_dag enforces the >64 ceiling; do it here too so callers see a
  // scheduler-scoped message.
  if (total > 64) {
    throw std::logic_error(
        "CausalWavefrontScheduler::initialize: >64 zones not supported in M4 "
        "(see OQ-M4-1)");
  }

  total_zones_ = static_cast<std::size_t>(total);
  metas_.assign(total_zones_, ZoneMeta{});
  canonical_order_ = plan.canonical_order;  // copy by value (invariant)

  // Every zone id in the canonical order must be in [0, total).
  for (const auto z : canonical_order_) {
    if (static_cast<std::uint64_t>(z) >= total) {
      throw std::logic_error(
          "CausalWavefrontScheduler::initialize: canonical_order contains out-of-range ZoneId");
    }
  }

  const double radius = plan.cutoff + plan.skin;
  spatial_dep_mask_ = compute_spatial_dependencies(plan, radius);

  // Cache planner-derived introspection inputs. We don't keep a reference
  // to `plan` itself — the scheduler must not outlive a stale plan shape.
  min_zones_per_rank_ = plan.n_min_per_rank;
  optimal_rank_count_cached_ = plan.optimal_rank_count;

  initialized_ = true;
  last_progress_ = std::chrono::steady_clock::now();
}

void CausalWavefrontScheduler::attach_outer_coordinator(OuterSdCoordinator* coord) {
  // Pattern 1 single-rank (M4) — nullptr is the expected argument and the
  // only legal value until M7 (D-M4-2). We still accept a non-null pointer
  // for forward compatibility with M7 test doubles, but M4 never dispatches
  // to it.
  outer_coord_ = coord;
}

void CausalWavefrontScheduler::shutdown() {
  // Idempotent: a shutdown on an uninitialized scheduler is a no-op, not
  // an error — mirrors how SimulationEngine drivers pair RAII guards.
  cert_store_.invalidate_all("shutdown");
  metas_.clear();
  canonical_order_.clear();
  spatial_dep_mask_.clear();
  cert_source_ = nullptr;
  outer_coord_ = nullptr;
  total_zones_ = 0;
  target_time_level_ = 0;
  min_zones_per_rank_ = 1;
  optimal_rank_count_cached_ = 1;
  initialized_ = false;
}

void CausalWavefrontScheduler::refresh_certificates() {
  require_initialized("refresh_certificates");

  for (const ZoneId z : canonical_order_) {
    const auto& meta = metas_[z];
    const TimeLevel target = meta.time_level + 1;

    CertificateInputs in{};
    in.zone_id = z;
    in.time_level = target;
    in.version = meta.version;

    if (cert_source_ != nullptr) {
      cert_source_->fill_inputs(z, target, in);
      // The store stamps mode_policy_tag authoritatively; the source may
      // populate physics/neighbor fields and the validity windows.
      in.zone_id = z;
      in.time_level = target;
    }

    (void) cert_store_.build(in);
  }

  last_progress_ = std::chrono::steady_clock::now();
}

void CausalWavefrontScheduler::invalidate_certificates_for(ZoneId zone) {
  require_initialized("invalidate_certificates_for");
  cert_store_.invalidate_for(zone);

  // If the zone was Ready (cert_id pointing at a now-removed entry), roll
  // the state machine back to ResidentPrev. For Computing / Completed /
  // later states the T4.7 commit hardening defines the abort path; T4.5
  // limits itself to the cheap Ready case.
  if (zone < metas_.size() && metas_[zone].state == ZoneState::Ready) {
    state_machine_.cert_invalidated(metas_[zone]);
  }
  last_progress_ = std::chrono::steady_clock::now();
}

void CausalWavefrontScheduler::invalidate_all_certificates(const std::string& reason) {
  require_initialized("invalidate_all_certificates");
  cert_store_.invalidate_all(reason);
  for (auto& meta : metas_) {
    if (meta.state == ZoneState::Ready) {
      state_machine_.cert_invalidated(meta);
    }
  }
  last_progress_ = std::chrono::steady_clock::now();
}

std::vector<ZoneTask> CausalWavefrontScheduler::select_ready_tasks() {
  // SPEC §5.1 pseudocode; §5.2 tie-break; §5.3 max_tasks_per_iteration.
  // Pure-filter + tie-break ordering + capped prefix; the selected zones
  // are then transitioned ResidentPrev → Ready via state_machine.mark_ready
  // inside the cap loop (so callers can go straight to mark_computing).
  require_initialized("select_ready_tasks");

  if (total_zones_ == 0) {
    return {};
  }

  const TimeLevel frontier_min_t = local_frontier_min();
  const std::uint64_t k_max = policy_.k_max_pipeline_depth;

  // Map ZoneId → canonical-order position for the Reference tie-break
  // (SPEC §5.2). Cached per call rather than at initialize() — total_zones_
  // is ≤ 64 in M4 (OQ-M4-1) so this is O(Z) and keeps the header free of
  // std::vector fields dedicated to an occasional hot-path helper.
  std::vector<std::size_t> canonical_index(total_zones_);
  for (std::size_t i = 0; i < canonical_order_.size(); ++i) {
    canonical_index[canonical_order_[i]] = i;
  }

  std::vector<TaskCandidate> cands;
  cands.reserve(total_zones_);

  for (const ZoneId z : canonical_order_) {
    const auto& meta = metas_[z];
    // Only ResidentPrev is selectable. Ready already has a cert locked in;
    // Computing/Completed/... are in flight. Empty has no data.
    if (meta.state != ZoneState::ResidentPrev) {
      continue;
    }

    const TimeLevel min_level = meta.time_level + 1;
    const TimeLevel k_span = k_max == 0 ? 0 : k_max - 1;
    const TimeLevel max_level_by_pipeline = min_level + k_span;
    const TimeLevel max_level_by_frontier = frontier_min_t + k_max;
    const TimeLevel max_level = std::min(max_level_by_pipeline, max_level_by_frontier);

    if (min_level > max_level) {
      continue;  // I6 frontier guard shuts this zone out at the current level
    }

    for (TimeLevel t = min_level; t <= max_level; ++t) {
      // Cert presence + safety (SPEC §4.1).
      const auto* cert = cert_store_.get(z, t);
      if (cert == nullptr || !cert->safe) {
        continue;
      }
      // Neighbor validity window (SPEC §4.2, §5.1 bullet 5).
      if (cert->neighbor_valid_until_step < t) {
        continue;
      }

      // Spatial peer readiness: every peer must be at or past t-1 and
      // must have actual data (state ≠ Empty). Pure read; no mutation.
      const ZoneDepMask dm = spatial_dep_mask_[z];
      bool peers_ok = true;
      if (dm != 0) {
        for (ZoneId p = 0; p < total_zones_; ++p) {
          if ((dm & (ZoneDepMask{1} << p)) == 0) {
            continue;
          }
          const auto& pm = metas_[p];
          if (pm.state == ZoneState::Empty) {
            peers_ok = false;
            break;
          }
          if (t > 0 && pm.time_level + 1 < t) {  // pm.time_level < t - 1
            peers_ok = false;
            break;
          }
        }
      }
      if (!peers_ok) {
        continue;
      }

      // Pattern 2 boundary gate — M4 is Pattern 1 only (D-M4-2); a
      // non-null outer_coord_ is accepted for M7 test doubles but never
      // consulted here.
      // if (outer_coord_ != nullptr && is_boundary_zone(z) &&
      //     !outer_coord_->can_advance_boundary_zone(z, t)) continue;

      TaskCandidate tc{};
      tc.zone_id = z;
      tc.time_level = t;
      tc.canonical_index = canonical_index[z];
      tc.version = meta.version;
      tc.cert_id = cert->cert_id;
      cands.push_back(tc);
      break;  // §5.1: one task per zone per iteration
    }
  }

  // §5.2 Reference tie-break: (time_level_asc, canonical_index_asc,
  // version_asc). std::sort is stable for strict-weak orderings; the
  // candidate list is already in canonical order by construction but may
  // mix time_levels, so sort is required for byte-stability.
  std::sort(cands.begin(), cands.end(), ReferenceTaskCompare{});

  // §5.3 cap.
  const std::size_t cap = std::min<std::size_t>(cands.size(), policy_.max_tasks_per_iteration);

  std::vector<ZoneTask> out;
  out.reserve(cap);
  for (std::size_t i = 0; i < cap; ++i) {
    const auto& c = cands[i];
    auto& meta = metas_[c.zone_id];

    // ResidentPrev → Ready. I2 (cert_id ≠ 0) is enforced by mark_ready;
    // our cert_id was allocated from the store so it is always nonzero.
    state_machine_.mark_ready(meta, c.cert_id);

    ZoneTask task{};
    task.zone_id = c.zone_id;
    task.time_level = c.time_level;
    task.local_state_version = meta.version;
    task.dep_mask = spatial_dep_mask_[c.zone_id];
    task.certificate_version = c.cert_id;
    task.priority = 0;
    task.mode_policy_tag = static_cast<std::uint32_t>(policy_.mode_policy_tag);
    out.push_back(task);
  }

  if (!out.empty()) {
    last_progress_ = std::chrono::steady_clock::now();
  }
  return out;
}

void CausalWavefrontScheduler::mark_computing(const ZoneTask& task) {
  require_initialized("mark_computing");
  state_machine_.mark_computing(metas_.at(task.zone_id));
  last_progress_ = std::chrono::steady_clock::now();
}

void CausalWavefrontScheduler::mark_completed(const ZoneTask& task) {
  require_initialized("mark_completed");
  state_machine_.mark_completed(metas_.at(task.zone_id));
  last_progress_ = std::chrono::steady_clock::now();
}

void CausalWavefrontScheduler::mark_packed(const ZoneTask& task) {
  require_initialized("mark_packed");
  state_machine_.mark_packed(metas_.at(task.zone_id));
  last_progress_ = std::chrono::steady_clock::now();
}

void CausalWavefrontScheduler::mark_inflight(const ZoneTask& task) {
  require_initialized("mark_inflight");
  state_machine_.mark_inflight(metas_.at(task.zone_id));
  last_progress_ = std::chrono::steady_clock::now();
}

void CausalWavefrontScheduler::mark_committed(const ZoneTask& task) {
  require_initialized("mark_committed");
  state_machine_.mark_committed(metas_.at(task.zone_id));
  last_progress_ = std::chrono::steady_clock::now();
}

void CausalWavefrontScheduler::commit_completed() {
  // Pattern 1 single-rank (D-M4-6): every Completed zone commits directly
  // to ResidentPrev via the state machine's no-peer shortcut. T4.7 hardens
  // this with retry semantics and peer-aware Phase B for M5.
  require_initialized("commit_completed");
  for (auto& meta : metas_) {
    if (meta.state == ZoneState::Completed) {
      state_machine_.commit_completed_no_peer(meta);
    }
  }
  last_progress_ = std::chrono::steady_clock::now();
}

bool CausalWavefrontScheduler::finished() const {
  // Fresh or shut-down scheduler has no pending work by definition — this
  // matches the watchdog contract in §8.1: `not finished()` implies "real
  // work is expected". Uninitialized = nothing expected.
  if (!initialized_) {
    return true;
  }
  for (const auto& meta : metas_) {
    if (meta.time_level < target_time_level_) {
      return false;
    }
  }
  return true;
}

std::size_t CausalWavefrontScheduler::min_zones_per_rank() const {
  return min_zones_per_rank_;
}

std::size_t CausalWavefrontScheduler::optimal_rank_count(std::size_t total_zones) const {
  // Returns the cached plan value when the caller passes the plan's
  // total_zones; otherwise falls back to `floor(total_zones /
  // min_zones_per_rank)`. The SPEC doesn't specify the argument's role
  // precisely — treating it as a "given this many zones, what's the
  // optimum?" query is the shape M7 will use.
  if (total_zones == total_zones_) {
    return optimal_rank_count_cached_;
  }
  const auto min_zpr = min_zones_per_rank_ == 0 ? 1 : min_zones_per_rank_;
  const auto result = total_zones / min_zpr;
  return result == 0 ? 1 : result;
}

std::size_t CausalWavefrontScheduler::current_pipeline_depth() const {
  if (!initialized_ || metas_.empty()) {
    return 0;
  }
  return static_cast<std::size_t>(local_frontier_max() - local_frontier_min());
}

TimeLevel CausalWavefrontScheduler::local_frontier_min() const {
  if (!initialized_ || metas_.empty()) {
    return 0;
  }
  TimeLevel m = metas_[0].time_level;
  for (std::size_t i = 1; i < metas_.size(); ++i) {
    m = std::min(m, metas_[i].time_level);
  }
  return m;
}

TimeLevel CausalWavefrontScheduler::local_frontier_max() const {
  if (!initialized_ || metas_.empty()) {
    return 0;
  }
  TimeLevel m = metas_[0].time_level;
  for (std::size_t i = 1; i < metas_.size(); ++i) {
    m = std::max(m, metas_[i].time_level);
  }
  return m;
}

void CausalWavefrontScheduler::on_zone_data_arrived(ZoneId zone, TimeLevel step, Version version) {
  require_initialized("on_zone_data_arrived");
  auto& meta = metas_.at(zone);
  state_machine_.on_zone_data_arrived(meta);  // Empty → ResidentPrev (throws otherwise)
  meta.time_level = step;
  meta.version = version;
  last_progress_ = std::chrono::steady_clock::now();
}

void CausalWavefrontScheduler::on_halo_arrived(std::uint32_t /*peer_subdomain*/,
                                               TimeLevel /*step*/) {
  // Pattern 2 only; M4 Pattern 1 has no halo peers (D-M4-2). Recording
  // progress would be misleading — ignore entirely.
}

void CausalWavefrontScheduler::on_neighbor_rebuild_completed(const std::vector<ZoneId>& affected) {
  require_initialized("on_neighbor_rebuild_completed");
  for (const auto zone : affected) {
    cert_store_.invalidate_for(zone);
    if (zone < metas_.size() && metas_[zone].state == ZoneState::Ready) {
      state_machine_.cert_invalidated(metas_[zone]);
    }
  }
  last_progress_ = std::chrono::steady_clock::now();
}

void CausalWavefrontScheduler::check_deadlock(std::chrono::milliseconds /*t_watchdog*/) {
  // T4.8 scope — full watchdog + diagnostic dump. T4.5 is a placeholder
  // that records progress without evaluating the predicate; tests that
  // want the M4 semantics will arrive in T4.8.
}

// --- Extension surface ----------------------------------------------------

const std::vector<ZoneId>& CausalWavefrontScheduler::canonical_order() const noexcept {
  return canonical_order_;
}

ZoneDepMask CausalWavefrontScheduler::spatial_dep_mask(ZoneId zone) const {
  if (zone >= spatial_dep_mask_.size()) {
    throw std::out_of_range("CausalWavefrontScheduler::spatial_dep_mask: zone out of range");
  }
  return spatial_dep_mask_[zone];
}

const ZoneMeta& CausalWavefrontScheduler::zone_meta(ZoneId zone) const {
  return metas_.at(zone);
}

std::size_t CausalWavefrontScheduler::total_zones() const noexcept {
  return total_zones_;
}

const CertificateStore& CausalWavefrontScheduler::cert_store() const noexcept {
  return cert_store_;
}

const SchedulerPolicy& CausalWavefrontScheduler::policy() const noexcept {
  return policy_;
}

bool CausalWavefrontScheduler::initialized() const noexcept {
  return initialized_;
}

OuterSdCoordinator* CausalWavefrontScheduler::outer_coordinator() const noexcept {
  return outer_coord_;
}

void CausalWavefrontScheduler::set_certificate_input_source(
    const CertificateInputSource* src) noexcept {
  cert_source_ = src;
}

void CausalWavefrontScheduler::set_target_time_level(TimeLevel target) noexcept {
  target_time_level_ = target;
}

TimeLevel CausalWavefrontScheduler::target_time_level() const noexcept {
  return target_time_level_;
}

}  // namespace tdmd::scheduler
