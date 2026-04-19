// Exec pack: docs/development/m4_execution_pack.md T4.2
// SPEC: docs/specs/scheduler/SPEC.md §2.1, §11.1
//
// T4.2 lands only the scheduler types + abstract interface + Reference
// policy. The tests here are narrow — they verify the types are
// default-constructible, have the shape the SPEC requires, and the
// reference PolicyFactory emits the canonical fingerprint. Behavioural
// tests (state-machine transitions, cert math, select_ready_tasks) land
// in T4.3+.

#include "tdmd/scheduler/outer_sd_coordinator.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/scheduler/td_scheduler.hpp"
#include "tdmd/scheduler/types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ts = tdmd::scheduler;

TEST_CASE("ZoneId / TimeLevel / Version have the canonical widths", "[scheduler][types]") {
  STATIC_REQUIRE(sizeof(ts::ZoneId) == 4);
  STATIC_REQUIRE(std::is_unsigned_v<ts::ZoneId>);
  STATIC_REQUIRE(sizeof(ts::TimeLevel) == 8);
  STATIC_REQUIRE(std::is_unsigned_v<ts::TimeLevel>);
  STATIC_REQUIRE(sizeof(ts::Version) == 8);
  STATIC_REQUIRE(std::is_unsigned_v<ts::Version>);
}

TEST_CASE("ZoneState enum — all eight states distinct", "[scheduler][types]") {
  // Smoke check against accidental collapse (e.g. duplicate enumerators).
  constexpr ts::ZoneState all[] = {
      ts::ZoneState::Empty,
      ts::ZoneState::ResidentPrev,
      ts::ZoneState::Ready,
      ts::ZoneState::Computing,
      ts::ZoneState::Completed,
      ts::ZoneState::PackedForSend,
      ts::ZoneState::InFlight,
      ts::ZoneState::Committed,
  };
  for (std::size_t i = 0; i < std::size(all); ++i) {
    for (std::size_t j = i + 1; j < std::size(all); ++j) {
      REQUIRE(static_cast<std::uint8_t>(all[i]) != static_cast<std::uint8_t>(all[j]));
    }
  }
}

TEST_CASE("SafetyCertificate — default-constructible, safe=false", "[scheduler][types]") {
  ts::SafetyCertificate c;
  REQUIRE_FALSE(c.safe);
  REQUIRE(c.cert_id == 0);
  REQUIRE(c.zone_id == 0);
  REQUIRE(c.time_level == 0);
  REQUIRE(c.version == 0);
  REQUIRE(c.v_max_zone == 0.0);
  REQUIRE(c.a_max_zone == 0.0);
  REQUIRE(c.dt_candidate == 0.0);
  REQUIRE(c.displacement_bound == 0.0);
  REQUIRE(c.buffer_width == 0.0);
  REQUIRE(c.skin_remaining == 0.0);
  REQUIRE(c.frontier_margin == 0.0);
  REQUIRE(c.neighbor_valid_until_step == 0);
  REQUIRE(c.halo_valid_until_step == 0);
  REQUIRE(c.mode_policy_tag == 0);
}

TEST_CASE("ZoneTask — default-constructible, value-type layout", "[scheduler][types]") {
  ts::ZoneTask t;
  REQUIRE(t.zone_id == 0);
  REQUIRE(t.time_level == 0);
  REQUIRE(t.local_state_version == 0);
  REQUIRE(t.dep_mask == 0);
  REQUIRE(t.certificate_version == 0);
  REQUIRE(t.priority == 0);
  REQUIRE(t.mode_policy_tag == 0);
  STATIC_REQUIRE(std::is_trivially_copyable_v<ts::ZoneTask>);
}

TEST_CASE("PolicyFactory::for_reference — canonical fields", "[scheduler][policy]") {
  const auto p = ts::PolicyFactory::for_reference();

  // Scope invariants from D-M4-1/3/7/13:
  REQUIRE(p.k_max_pipeline_depth == 1);
  REQUIRE(p.max_tasks_per_iteration == 1);
  REQUIRE(p.use_canonical_tie_break);
  REQUIRE_FALSE(p.allow_task_stealing);
  REQUIRE_FALSE(p.allow_adaptive_buffer);
  REQUIRE(p.deterministic_reduction_cert);
  REQUIRE(p.t_watchdog == std::chrono::milliseconds{30'000});
  REQUIRE(p.max_retries_per_task == 3);
  REQUIRE_FALSE(p.exponential_backoff);
  REQUIRE(p.two_phase_commit);
  REQUIRE(p.mode_policy_tag != 0);  // must be a live fingerprint
}

TEST_CASE("PolicyFactory — Production / Fast stubs throw in M4", "[scheduler][policy]") {
  // D-M4-3: only Reference is live in M4. Non-Reference factories throw so
  // silent promotion of an unsupported profile is impossible.
  REQUIRE_THROWS_AS(ts::PolicyFactory::for_production(), std::logic_error);
  REQUIRE_THROWS_AS(ts::PolicyFactory::for_fast_experimental(), std::logic_error);
}

TEST_CASE("TdScheduler — abstract interface is not instantiable", "[scheduler][abi]") {
  // Purely a shape check — the interface compiles and has the expected
  // methods. Instantiation is impossible (pure-virtual); construction
  // happens in T4.5 via CausalWavefrontScheduler.
  STATIC_REQUIRE(std::is_abstract_v<ts::TdScheduler>);
  STATIC_REQUIRE(std::is_polymorphic_v<ts::TdScheduler>);
  // OuterSdCoordinator — T7.6 promoted from M4 stub (D-M4-2) to the full
  // pure-virtual contract from master §12.7a + scheduler/SPEC §2.4. The
  // class is now abstract; only concrete implementations such as
  // ConcreteOuterSdCoordinator instantiate.
  STATIC_REQUIRE(std::is_abstract_v<ts::OuterSdCoordinator>);
  STATIC_REQUIRE(std::is_polymorphic_v<ts::OuterSdCoordinator>);
  STATIC_REQUIRE(std::has_virtual_destructor_v<ts::OuterSdCoordinator>);
}
