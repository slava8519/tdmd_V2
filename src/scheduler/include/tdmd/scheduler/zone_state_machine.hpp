#pragma once

// SPEC: docs/specs/scheduler/SPEC.md §3 (state machine §3.1, I1-I5 §3.2)
// Master spec: §6.2, §13.4
// Exec pack: docs/development/m4_execution_pack.md T4.4
//
// ZoneStateMachine — one method per event from the §3.1 diagram. Each
// method validates the current state (+ relevant flags and cert_id),
// mutates ZoneMeta in place, and throws StateMachineError on violation.
// Throws are transactional: on rejection ZoneMeta is unchanged.
//
// In M4 Reference profile, illegal transitions always throw. Production /
// FastExperimental will later downgrade some rejections to asserts; the
// policy hook will be wired in T4.5.

#include "tdmd/scheduler/types.hpp"
#include "tdmd/scheduler/zone_meta.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tdmd::scheduler {

class StateMachineError : public std::logic_error {
public:
  using std::logic_error::logic_error;
};

class ZoneStateMachine {
public:
  // Empty → ResidentPrev. Receiver-side event (§3.1). Rejects if the zone
  // is already in any non-Empty state — the scheduler must release() first.
  void on_zone_data_arrived(ZoneMeta& m) const;

  // ResidentPrev → Ready. `cert_id` must be non-zero (I2 setup); rejects
  // if the zone is already queued (I4 zone-level dedup).
  void mark_ready(ZoneMeta& m, std::uint64_t cert_id) const;

  // Ready → Computing. Requires state == Ready, cert_id != 0 (I2), and
  // in_ready_queue == true. Clears in_ready_queue on success.
  void mark_computing(ZoneMeta& m) const;

  // Computing → Completed. Bumps ZoneMeta::version (master spec §6.1 Phase A).
  void mark_completed(ZoneMeta& m) const;

  // Completed → PackedForSend. Used when a spatial peer still needs this
  // zone's data (Pattern 2 / M5+); a no-op in single-rank Pattern 1 where
  // commit_completed_no_peer is used instead.
  void mark_packed(ZoneMeta& m) const;

  // PackedForSend → InFlight. Sets in_inflight_queue = true.
  void mark_inflight(ZoneMeta& m) const;

  // InFlight → Committed. Clears in_inflight_queue.
  void mark_committed(ZoneMeta& m) const;

  // Completed → Committed (Pattern 1 commit short-circuit, SPEC §6.2 bullet 2).
  // Called by the scheduler's Phase-B when no peer needs the zone's data —
  // the Pattern 1 single-rank case at M4. Clears cert_id; leaves
  // in_inflight_queue at its prior value (false — no InFlight was visited).
  // The engine must call `release` + `on_zone_data_arrived` to start the next
  // time step.
  void commit_completed_no_peer(ZoneMeta& m) const;

  // Ready → ResidentPrev. Rollback when the cert is invalidated before
  // mark_computing. Clears in_ready_queue and cert_id.
  void cert_invalidated(ZoneMeta& m) const;

  // Committed → Empty. Releases the zone back to the pool for the next
  // cycle; the scheduler is expected to bump time_level via a subsequent
  // on_zone_data_arrived + refresh cycle (I1: no shortcut to Ready).
  void release(ZoneMeta& m) const;

  // Invariant checks — pure, do not throw. Fuzzer calls after every event.
  // Public so tests can assert them independently of the internal rejects.
  static bool check_i3_queue_disjoint(const ZoneMeta& m) noexcept;
  static bool check_i2_computing_has_cert(const ZoneMeta& m) noexcept;

private:
  [[noreturn]] static void reject(const std::string& msg);
};

}  // namespace tdmd::scheduler
