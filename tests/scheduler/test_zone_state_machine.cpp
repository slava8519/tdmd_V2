// Exec pack: docs/development/m4_execution_pack.md T4.4
// SPEC: docs/specs/scheduler/SPEC.md §3
// Master spec: §6.2, §13.4 I1-I5
//
// Every legal transition from §3.1 gets a unit test. Every illegal
// transition from §3.2 gets a unit test asserting StateMachineError is
// thrown. The randomised I1-I5 fuzzer lives in fuzz_zone_state_invariants.cpp.

#include "tdmd/scheduler/types.hpp"
#include "tdmd/scheduler/zone_meta.hpp"
#include "tdmd/scheduler/zone_state_machine.hpp"

#include <catch2/catch_test_macros.hpp>

namespace ts = tdmd::scheduler;

namespace {

constexpr std::uint64_t kCertId = 42;

ts::ZoneMeta drive_to(ts::ZoneState target) {
  const ts::ZoneStateMachine sm;
  ts::ZoneMeta m;
  if (target == ts::ZoneState::Empty) {
    return m;
  }
  sm.on_zone_data_arrived(m);
  if (target == ts::ZoneState::ResidentPrev) {
    return m;
  }
  sm.mark_ready(m, kCertId);
  if (target == ts::ZoneState::Ready) {
    return m;
  }
  sm.mark_computing(m);
  if (target == ts::ZoneState::Computing) {
    return m;
  }
  sm.mark_completed(m);
  if (target == ts::ZoneState::Completed) {
    return m;
  }
  sm.mark_packed(m);
  if (target == ts::ZoneState::PackedForSend) {
    return m;
  }
  sm.mark_inflight(m);
  if (target == ts::ZoneState::InFlight) {
    return m;
  }
  sm.mark_committed(m);
  return m;  // Committed
}

}  // namespace

// ----------------------------------------------------------------------------
// Legal transitions — one test each (SPEC §3.1 diagram, 10 edges)
// ----------------------------------------------------------------------------

TEST_CASE("legal — Empty → ResidentPrev (on_zone_data_arrived)", "[scheduler][state][legal]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m;
  REQUIRE(m.state == ts::ZoneState::Empty);
  sm.on_zone_data_arrived(m);
  REQUIRE(m.state == ts::ZoneState::ResidentPrev);
}

TEST_CASE("legal — ResidentPrev → Ready (mark_ready)", "[scheduler][state][legal]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::ResidentPrev);
  sm.mark_ready(m, kCertId);
  REQUIRE(m.state == ts::ZoneState::Ready);
  REQUIRE(m.cert_id == kCertId);
  REQUIRE(m.in_ready_queue);
}

TEST_CASE("legal — Ready → Computing (mark_computing)", "[scheduler][state][legal]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::Ready);
  sm.mark_computing(m);
  REQUIRE(m.state == ts::ZoneState::Computing);
  REQUIRE_FALSE(m.in_ready_queue);
  REQUIRE(m.cert_id == kCertId);
}

TEST_CASE("legal — Computing → Completed bumps version", "[scheduler][state][legal]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::Computing);
  const ts::Version before = m.version;
  sm.mark_completed(m);
  REQUIRE(m.state == ts::ZoneState::Completed);
  REQUIRE(m.version == before + 1);
}

TEST_CASE("legal — Completed → PackedForSend", "[scheduler][state][legal]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::Completed);
  sm.mark_packed(m);
  REQUIRE(m.state == ts::ZoneState::PackedForSend);
}

TEST_CASE("legal — PackedForSend → InFlight", "[scheduler][state][legal]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::PackedForSend);
  sm.mark_inflight(m);
  REQUIRE(m.state == ts::ZoneState::InFlight);
  REQUIRE(m.in_inflight_queue);
}

TEST_CASE("legal — InFlight → Committed", "[scheduler][state][legal]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::InFlight);
  sm.mark_committed(m);
  REQUIRE(m.state == ts::ZoneState::Committed);
  REQUIRE_FALSE(m.in_inflight_queue);
}

TEST_CASE("legal — Completed → Committed (Pattern 1 commit, no peer)",
          "[scheduler][state][legal]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::Completed);
  sm.commit_completed_no_peer(m);
  REQUIRE(m.state == ts::ZoneState::Committed);
  REQUIRE(m.cert_id == 0);
  REQUIRE_FALSE(m.in_inflight_queue);
  REQUIRE_FALSE(m.in_ready_queue);
}

TEST_CASE("legal — Ready → ResidentPrev (cert_invalidated rollback)", "[scheduler][state][legal]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::Ready);
  sm.cert_invalidated(m);
  REQUIRE(m.state == ts::ZoneState::ResidentPrev);
  REQUIRE(m.cert_id == 0);
  REQUIRE_FALSE(m.in_ready_queue);
}

TEST_CASE("legal — Committed → Empty (release)", "[scheduler][state][legal]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::Committed);
  const ts::TimeLevel tl_before = m.time_level;
  const ts::Version v_before = m.version;
  sm.release(m);
  REQUIRE(m.state == ts::ZoneState::Empty);
  REQUIRE(m.cert_id == 0);
  REQUIRE_FALSE(m.in_ready_queue);
  REQUIRE_FALSE(m.in_inflight_queue);
  // Counters preserved — bumped externally before next cycle.
  REQUIRE(m.time_level == tl_before);
  REQUIRE(m.version == v_before);
}

// ----------------------------------------------------------------------------
// Illegal transitions — I1-I5 enforcement
// ----------------------------------------------------------------------------

TEST_CASE("I1 — Committed → Ready direct is illegal", "[scheduler][state][illegal][I1]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::Committed);
  // mark_ready expects ResidentPrev → throws because state == Committed.
  REQUIRE_THROWS_AS(sm.mark_ready(m, kCertId), ts::StateMachineError);
  // State unchanged (transactional reject).
  REQUIRE(m.state == ts::ZoneState::Committed);
}

TEST_CASE("I2 — Empty → Computing direct is illegal", "[scheduler][state][illegal][I2]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m;  // Empty
  REQUIRE_THROWS_AS(sm.mark_computing(m), ts::StateMachineError);
  REQUIRE(m.state == ts::ZoneState::Empty);
}

TEST_CASE("I2 — mark_ready with cert_id=0 is illegal", "[scheduler][state][illegal][I2]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::ResidentPrev);
  REQUIRE_THROWS_AS(sm.mark_ready(m, 0), ts::StateMachineError);
  REQUIRE(m.state == ts::ZoneState::ResidentPrev);
  REQUIRE(m.cert_id == 0);
  REQUIRE_FALSE(m.in_ready_queue);
}

TEST_CASE("I2 — Computing with cert_id=0 check (post-hoc invariant)",
          "[scheduler][state][invariant][I2]") {
  ts::ZoneMeta m;
  m.state = ts::ZoneState::Computing;
  m.cert_id = 0;
  REQUIRE_FALSE(ts::ZoneStateMachine::check_i2_computing_has_cert(m));
  m.cert_id = 5;
  REQUIRE(ts::ZoneStateMachine::check_i2_computing_has_cert(m));
}

TEST_CASE("I3 — queue disjointness check", "[scheduler][state][invariant][I3]") {
  ts::ZoneMeta m;
  REQUIRE(ts::ZoneStateMachine::check_i3_queue_disjoint(m));
  m.in_ready_queue = true;
  REQUIRE(ts::ZoneStateMachine::check_i3_queue_disjoint(m));
  m.in_inflight_queue = true;
  REQUIRE_FALSE(ts::ZoneStateMachine::check_i3_queue_disjoint(m));
}

TEST_CASE("I4 — double mark_ready on same zone is rejected", "[scheduler][state][illegal][I4]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::Ready);
  // Now it's Ready already; mark_ready would fail on state check regardless.
  // Simulate the scheduler-level bug: state coerced back to ResidentPrev
  // but in_ready_queue flag was not cleared → I4 violation.
  m.state = ts::ZoneState::ResidentPrev;
  REQUIRE_THROWS_AS(sm.mark_ready(m, kCertId + 1), ts::StateMachineError);
  // in_ready_queue is unchanged (transactional).
  REQUIRE(m.in_ready_queue);
}

TEST_CASE("I5 — Completed → Committed direct is illegal", "[scheduler][state][illegal][I5]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m = drive_to(ts::ZoneState::Completed);
  // mark_committed expects InFlight — Completed → Committed shortcut rejected.
  REQUIRE_THROWS_AS(sm.mark_committed(m), ts::StateMachineError);
  REQUIRE(m.state == ts::ZoneState::Completed);
}

// Sampling of other illegal transitions — not every (from, event) pair but
// enough to demonstrate every event properly guards its precondition.
TEST_CASE("illegal — on_zone_data_arrived on non-Empty zone", "[scheduler][state][illegal]") {
  ts::ZoneStateMachine sm;
  for (auto s : {ts::ZoneState::ResidentPrev,
                 ts::ZoneState::Ready,
                 ts::ZoneState::Computing,
                 ts::ZoneState::Completed,
                 ts::ZoneState::PackedForSend,
                 ts::ZoneState::InFlight,
                 ts::ZoneState::Committed}) {
    ts::ZoneMeta m;
    m.state = s;
    REQUIRE_THROWS_AS(sm.on_zone_data_arrived(m), ts::StateMachineError);
    REQUIRE(m.state == s);
  }
}

TEST_CASE("illegal — mark_completed outside Computing", "[scheduler][state][illegal]") {
  ts::ZoneStateMachine sm;
  for (auto s : {ts::ZoneState::Empty,
                 ts::ZoneState::ResidentPrev,
                 ts::ZoneState::Ready,
                 ts::ZoneState::Completed,
                 ts::ZoneState::PackedForSend,
                 ts::ZoneState::InFlight,
                 ts::ZoneState::Committed}) {
    ts::ZoneMeta m;
    m.state = s;
    m.cert_id = kCertId;  // ensure I2 is not what trips it
    REQUIRE_THROWS_AS(sm.mark_completed(m), ts::StateMachineError);
  }
}

TEST_CASE("illegal — release outside Committed", "[scheduler][state][illegal]") {
  ts::ZoneStateMachine sm;
  for (auto s : {ts::ZoneState::Empty,
                 ts::ZoneState::ResidentPrev,
                 ts::ZoneState::Ready,
                 ts::ZoneState::Computing,
                 ts::ZoneState::Completed,
                 ts::ZoneState::PackedForSend,
                 ts::ZoneState::InFlight}) {
    ts::ZoneMeta m;
    m.state = s;
    REQUIRE_THROWS_AS(sm.release(m), ts::StateMachineError);
  }
}

TEST_CASE("illegal — cert_invalidated outside Ready", "[scheduler][state][illegal]") {
  ts::ZoneStateMachine sm;
  for (auto s : {ts::ZoneState::Empty,
                 ts::ZoneState::ResidentPrev,
                 ts::ZoneState::Computing,
                 ts::ZoneState::Completed,
                 ts::ZoneState::PackedForSend,
                 ts::ZoneState::InFlight,
                 ts::ZoneState::Committed}) {
    ts::ZoneMeta m;
    m.state = s;
    REQUIRE_THROWS_AS(sm.cert_invalidated(m), ts::StateMachineError);
  }
}

TEST_CASE("illegal — commit_completed_no_peer outside Completed", "[scheduler][state][illegal]") {
  ts::ZoneStateMachine sm;
  for (auto s : {ts::ZoneState::Empty,
                 ts::ZoneState::ResidentPrev,
                 ts::ZoneState::Ready,
                 ts::ZoneState::Computing,
                 ts::ZoneState::PackedForSend,
                 ts::ZoneState::InFlight,
                 ts::ZoneState::Committed}) {
    ts::ZoneMeta m;
    m.state = s;
    REQUIRE_THROWS_AS(sm.commit_completed_no_peer(m), ts::StateMachineError);
  }
}

// ----------------------------------------------------------------------------
// Full cycle walk — one complete lifecycle Pattern 1 and Pattern 2 path
// ----------------------------------------------------------------------------

TEST_CASE("full cycle — Pattern 1 (no peer): Empty → Committed → release → Empty",
          "[scheduler][state][cycle]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m;

  sm.on_zone_data_arrived(m);
  sm.mark_ready(m, 1);
  sm.mark_computing(m);
  sm.mark_completed(m);
  sm.commit_completed_no_peer(m);

  REQUIRE(m.state == ts::ZoneState::Committed);
  REQUIRE(m.version == 1);
  REQUIRE(m.cert_id == 0);
  REQUIRE_FALSE(m.in_ready_queue);
  REQUIRE_FALSE(m.in_inflight_queue);

  // Engine closes the cycle with a release, re-enabling on_zone_data_arrived.
  sm.release(m);
  REQUIRE(m.state == ts::ZoneState::Empty);
}

TEST_CASE("full cycle — Pattern 2 (peer path): Empty → Empty via release",
          "[scheduler][state][cycle]") {
  ts::ZoneStateMachine sm;
  ts::ZoneMeta m;

  sm.on_zone_data_arrived(m);
  sm.mark_ready(m, 7);
  sm.mark_computing(m);
  sm.mark_completed(m);
  sm.mark_packed(m);
  sm.mark_inflight(m);
  sm.mark_committed(m);
  sm.release(m);

  REQUIRE(m.state == ts::ZoneState::Empty);
  REQUIRE(m.version == 1);
}
