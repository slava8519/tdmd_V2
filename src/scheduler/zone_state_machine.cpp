// SPEC: docs/specs/scheduler/SPEC.md §3 (diagram §3.1, invariants §3.2)
// Master spec: §6.2, §13.4 I1-I5
// Exec pack: docs/development/m4_execution_pack.md T4.4

#include "tdmd/scheduler/zone_state_machine.hpp"

#include <string>

namespace tdmd::scheduler {

namespace {

std::string state_name(ZoneState s) {
  switch (s) {
    case ZoneState::Empty:
      return "Empty";
    case ZoneState::ResidentPrev:
      return "ResidentPrev";
    case ZoneState::Ready:
      return "Ready";
    case ZoneState::Computing:
      return "Computing";
    case ZoneState::Completed:
      return "Completed";
    case ZoneState::PackedForSend:
      return "PackedForSend";
    case ZoneState::InFlight:
      return "InFlight";
    case ZoneState::Committed:
      return "Committed";
  }
  return "Unknown";
}

}  // namespace

[[noreturn]] void ZoneStateMachine::reject(const std::string& msg) {
  throw StateMachineError{msg};
}

void ZoneStateMachine::on_zone_data_arrived(ZoneMeta& m) const {
  if (m.state != ZoneState::Empty) {
    reject("on_zone_data_arrived: expected Empty, got " + state_name(m.state));
  }
  m.state = ZoneState::ResidentPrev;
}

void ZoneStateMachine::mark_ready(ZoneMeta& m, std::uint64_t cert_id) const {
  if (m.state != ZoneState::ResidentPrev) {
    reject("mark_ready: expected ResidentPrev, got " + state_name(m.state));
  }
  if (cert_id == 0) {
    reject("mark_ready: cert_id must be non-zero (I2)");
  }
  if (m.in_ready_queue) {
    // I4 zone-level dedup: same zone cannot be queued Ready twice.
    reject("mark_ready: zone already in ready_queue (I4)");
  }
  if (m.in_inflight_queue) {
    // I3 defense: should be impossible from ResidentPrev but hard-fail if
    // external mutation corrupted flags.
    reject("mark_ready: zone in inflight_queue (I3)");
  }
  m.cert_id = cert_id;
  m.in_ready_queue = true;
  m.state = ZoneState::Ready;
}

void ZoneStateMachine::mark_computing(ZoneMeta& m) const {
  if (m.state != ZoneState::Ready) {
    reject("mark_computing: expected Ready, got " + state_name(m.state));
  }
  if (m.cert_id == 0) {
    // I2: Computing requires valid cert. At state-machine level we check the
    // handle; the scheduler proper also checks presence in CertificateStore.
    reject("mark_computing: cert_id == 0 (I2)");
  }
  if (!m.in_ready_queue) {
    reject("mark_computing: zone not in ready_queue");
  }
  m.in_ready_queue = false;
  m.state = ZoneState::Computing;
}

void ZoneStateMachine::mark_completed(ZoneMeta& m) const {
  if (m.state != ZoneState::Computing) {
    reject("mark_completed: expected Computing, got " + state_name(m.state));
  }
  ++m.version;  // Phase A bump (master spec §6.1)
  m.state = ZoneState::Completed;
}

void ZoneStateMachine::mark_packed(ZoneMeta& m) const {
  if (m.state != ZoneState::Completed) {
    reject("mark_packed: expected Completed, got " + state_name(m.state));
  }
  m.state = ZoneState::PackedForSend;
}

void ZoneStateMachine::mark_inflight(ZoneMeta& m) const {
  if (m.state != ZoneState::PackedForSend) {
    reject("mark_inflight: expected PackedForSend, got " + state_name(m.state));
  }
  if (m.in_ready_queue) {
    reject("mark_inflight: zone still in ready_queue (I3)");
  }
  m.in_inflight_queue = true;
  m.state = ZoneState::InFlight;
}

void ZoneStateMachine::mark_committed(ZoneMeta& m) const {
  if (m.state != ZoneState::InFlight) {
    // I5: Completed → Committed is NOT a legal direct transition. Must go
    // through PackedForSend → InFlight. Catches attempts to commit without
    // the explicit two-phase dance.
    reject("mark_committed: expected InFlight, got " + state_name(m.state) + " (I5)");
  }
  m.in_inflight_queue = false;
  m.state = ZoneState::Committed;
}

void ZoneStateMachine::commit_completed_no_peer(ZoneMeta& m) const {
  if (m.state != ZoneState::Completed) {
    reject("commit_completed_no_peer: expected Completed, got " + state_name(m.state));
  }
  // Pattern 1 Phase-B short-circuit (SPEC §6.2 bullet 2): an internal zone
  // with no downstream peer goes Completed → Committed directly — skipping
  // the PackedForSend → InFlight pair that's only meaningful when a peer
  // needs the buffer. The cert is stale (version bumped at mark_completed),
  // so clear it; the engine releases the zone and re-arms data_arrived for
  // the next time step.
  m.cert_id = 0;
  m.state = ZoneState::Committed;
}

void ZoneStateMachine::cert_invalidated(ZoneMeta& m) const {
  if (m.state != ZoneState::Ready) {
    reject("cert_invalidated: expected Ready, got " + state_name(m.state));
  }
  m.cert_id = 0;
  m.in_ready_queue = false;
  m.state = ZoneState::ResidentPrev;
}

void ZoneStateMachine::release(ZoneMeta& m) const {
  if (m.state != ZoneState::Committed) {
    reject("release: expected Committed, got " + state_name(m.state));
  }
  // time_level and version are preserved as historical counters — the next
  // cycle bumps time_level externally before the zone re-enters the cycle.
  m.cert_id = 0;
  m.in_ready_queue = false;
  m.in_inflight_queue = false;
  m.state = ZoneState::Empty;
}

bool ZoneStateMachine::check_i3_queue_disjoint(const ZoneMeta& m) noexcept {
  return !(m.in_ready_queue && m.in_inflight_queue);
}

bool ZoneStateMachine::check_i2_computing_has_cert(const ZoneMeta& m) noexcept {
  if (m.state != ZoneState::Computing) {
    return true;  // vacuously satisfied
  }
  return m.cert_id != 0;
}

}  // namespace tdmd::scheduler
