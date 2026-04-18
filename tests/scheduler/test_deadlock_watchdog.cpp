// Exec pack: docs/development/m4_execution_pack.md T4.8
// SPEC: docs/specs/scheduler/SPEC.md §8 (deadlock watchdog + diagnostic dump)
//
// Tests for CausalWavefrontScheduler's watchdog + DiagnosticReport surface:
//
//   1. Uninitialized scheduler never throws.
//   2. Finished scheduler never throws (no pending work).
//   3. Idle scheduler with pending work throws after > t_watchdog.
//   4. Progress events (mark_computing, on_zone_data_arrived) reset the timer.
//   5. Refresh + invalidate do NOT reset the timer (SPEC §8.2 narrows this).
//   6. DiagnosticReport state histogram + frontier + event ring population.
//   7. to_string() contains grep-friendly labels the operator tooling expects.
//   8. Event ring saturates at kEventRingCapacity (oldest dropped FIFO).
//   9. DeadlockError inherits std::runtime_error and carries the dump.

#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/diagnostic_dump.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>

namespace ts = tdmd::scheduler;
namespace tz = tdmd::zoning;

namespace {

tz::ZoningPlan make_plan(std::uint32_t nx = 2,
                         std::uint32_t ny = 1,
                         std::uint32_t nz = 1,
                         double zone_size = 1.0,
                         double cutoff = 0.5,
                         double skin = 0.5) {
  tz::ZoningPlan plan;
  plan.scheme = tz::ZoningScheme::Linear1D;
  plan.n_zones = {nx, ny, nz};
  plan.zone_size = {zone_size, zone_size, zone_size};
  plan.cutoff = cutoff;
  plan.skin = skin;
  plan.buffer_width = {skin, skin, skin};
  const auto total = static_cast<tz::ZoneId>(nx * ny * nz);
  plan.canonical_order.reserve(total);
  for (tz::ZoneId z = 0; z < total; ++z) {
    plan.canonical_order.push_back(z);
  }
  plan.n_min_per_rank = 1;
  plan.optimal_rank_count = total;
  return plan;
}

struct SafeSource : ts::CertificateInputSource {
  void fill_inputs(ts::ZoneId zone,
                   ts::TimeLevel time_level,
                   ts::CertificateInputs& out) const override {
    out.zone_id = zone;
    out.time_level = time_level;
    out.v_max_zone = 0.1;
    out.a_max_zone = 0.2;
    out.dt_candidate = 0.001;
    out.buffer_width = 1.0;
    out.skin_remaining = 1.0;
    out.frontier_margin = 1.0;
    out.neighbor_valid_until_step = time_level + 100;
    out.halo_valid_until_step = time_level + 100;
  }
};

ts::SchedulerPolicy policy_all_ready(std::uint32_t total_zones) {
  auto p = ts::PolicyFactory::for_reference();
  p.max_tasks_per_iteration = total_zones;
  p.k_max_pipeline_depth = 1;
  return p;
}

}  // namespace

// 1. Uninitialized scheduler never throws — finished() returns true.
TEST_CASE("watchdog — uninitialized scheduler never fires", "[scheduler][watchdog]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  REQUIRE(sched.finished());
  REQUIRE_NOTHROW(sched.check_deadlock(std::chrono::milliseconds{0}));
  REQUIRE_NOTHROW(sched.check_deadlock(std::chrono::milliseconds{1000}));
}

// 2. Finished scheduler never throws — no pending work.
TEST_CASE("watchdog — finished scheduler never fires", "[scheduler][watchdog]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));
  sched.set_target_time_level(0);  // all zones already at or past target
  REQUIRE(sched.finished());
  // Even after an impossibly small timeout, no throw.
  std::this_thread::sleep_for(std::chrono::milliseconds{5});
  REQUIRE_NOTHROW(sched.check_deadlock(std::chrono::milliseconds{1}));
}

// 3. Idle scheduler with pending work throws after idle > t_watchdog.
TEST_CASE("watchdog — fires on genuine idle", "[scheduler][watchdog][deadlock]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(2)};
  sched.initialize(make_plan(2, 1, 1));
  sched.set_target_time_level(10);

  SafeSource src;
  sched.set_certificate_input_source(&src);

  // Deliberately don't prime any zones — every zone is Empty, pipeline
  // is stuck forever. refresh_certificates is not progress.
  sched.refresh_certificates();

  REQUIRE_FALSE(sched.finished());
  std::this_thread::sleep_for(std::chrono::milliseconds{15});
  REQUIRE_THROWS_AS(sched.check_deadlock(std::chrono::milliseconds{5}), ts::DeadlockError);
}

// 4. Progress events (mark_computing) reset the timer.
TEST_CASE("watchdog — mark_computing is progress (§8.2 bullet 1)",
          "[scheduler][watchdog][progress]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));
  sched.set_target_time_level(10);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);
  sched.refresh_certificates();

  auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);

  std::this_thread::sleep_for(std::chrono::milliseconds{10});
  sched.mark_computing(tasks[0]);  // bumps last_progress_

  // Using t_watchdog = 20ms: 10ms already passed pre-mark_computing, but
  // the bump reset the timer. Immediate check should not fire.
  REQUIRE_NOTHROW(sched.check_deadlock(std::chrono::milliseconds{20}));
}

// 5. refresh_certificates + invalidate are NOT progress events (§8.2).
TEST_CASE("watchdog — refresh/invalidate storm still trips the watchdog",
          "[scheduler][watchdog][retry-storm]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));
  sched.set_target_time_level(10);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);

  // Simulate a retry storm: refresh + invalidate + refresh + invalidate
  // without any compute progress. None of this should reset the timer.
  std::this_thread::sleep_for(std::chrono::milliseconds{10});
  sched.refresh_certificates();
  sched.invalidate_certificates_for(0);  // Ready → ResidentPrev, charge 1 retry
  sched.refresh_certificates();
  sched.invalidate_all_certificates("rebuild");  // zone still ResidentPrev → no rollback

  REQUIRE_FALSE(sched.finished());
  // last_progress_ was set at on_zone_data_arrived; the 10ms sleep means
  // idle > 5ms even though refresh + invalidate happened after.
  REQUIRE_THROWS_AS(sched.check_deadlock(std::chrono::milliseconds{5}), ts::DeadlockError);
}

// 6. DiagnosticReport state histogram + frontier + event ring.
TEST_CASE("diagnostic_dump — state histogram + frontier + events populated",
          "[scheduler][watchdog][diagnostic]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(3)};
  sched.initialize(make_plan(3, 1, 1));
  sched.set_target_time_level(10);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);
  sched.on_zone_data_arrived(1, 0, 0);
  sched.on_zone_data_arrived(2, 0, 0);
  sched.refresh_certificates();

  auto tasks = sched.select_ready_tasks();
  REQUIRE_FALSE(tasks.empty());
  sched.mark_computing(tasks[0]);  // one zone Computing

  const auto report = sched.make_diagnostic_report();
  REQUIRE(report.total_zones == 3);

  const auto residentprev_idx = static_cast<std::size_t>(ts::ZoneState::ResidentPrev);
  const auto ready_idx = static_cast<std::size_t>(ts::ZoneState::Ready);
  const auto computing_idx = static_cast<std::size_t>(ts::ZoneState::Computing);
  const auto empty_idx = static_cast<std::size_t>(ts::ZoneState::Empty);

  // Exactly one zone should be Computing; the other two are either Ready
  // (if max_tasks_per_iteration let them in) or ResidentPrev.
  REQUIRE(report.state_counts[computing_idx] == 1);
  REQUIRE(report.state_counts[empty_idx] == 0);
  const auto rp_plus_ready = report.state_counts[residentprev_idx] + report.state_counts[ready_idx];
  REQUIRE(rp_plus_ready == 2);

  REQUIRE(report.frontier_min == 0);
  REQUIRE(report.frontier_max == 0);

  // Event ring must contain at least the events we generated this test.
  REQUIRE(report.recent_events.size() >= 7);  // init + 3 arrivals + refresh + select + compute
  // Newest event is MarkComputing.
  REQUIRE(report.recent_events.back().kind == ts::SchedulerEvent::MarkComputing);
  // Oldest event is Initialize (only ~10 events so we're below the 100 cap).
  REQUIRE(report.recent_events.front().kind == ts::SchedulerEvent::Initialize);
}

// 7. to_string() contains the grep-friendly operator labels.
TEST_CASE("diagnostic_dump — to_string() labels are grep-friendly",
          "[scheduler][watchdog][diagnostic][dump-string]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(2)};
  sched.initialize(make_plan(2, 1, 1));
  sched.set_target_time_level(5);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);
  sched.refresh_certificates();

  auto report = sched.make_diagnostic_report();
  report.idle_duration = std::chrono::milliseconds{1234};
  report.t_watchdog = std::chrono::milliseconds{500};
  const auto dump = report.to_string();

  REQUIRE(dump.find("deadlock:") != std::string::npos);
  REQUIRE(dump.find("idle_for=1234ms") != std::string::npos);
  REQUIRE(dump.find("t_watchdog=500ms") != std::string::npos);
  REQUIRE(dump.find("frontier_min=") != std::string::npos);
  REQUIRE(dump.find("frontier_max=") != std::string::npos);
  REQUIRE(dump.find("zones:") != std::string::npos);
  REQUIRE(dump.find("queues:") != std::string::npos);
  REQUIRE(dump.find("events:") != std::string::npos);
  REQUIRE(dump.find("advice:") != std::string::npos);
}

// 8. Event ring saturates at kEventRingCapacity.
TEST_CASE("diagnostic_dump — event ring caps at 100 entries",
          "[scheduler][watchdog][diagnostic][event-ring]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));

  // Generate > 100 events via refresh_certificates in a tight loop. Each
  // refresh_certificates is one event (the loop body uses cert_store, not
  // record_event).
  for (int i = 0; i < 150; ++i) {
    sched.refresh_certificates();
  }
  const auto report = sched.make_diagnostic_report();
  REQUIRE(report.recent_events.size() == 100);
  // The oldest surviving event must be a RefreshCertificates — Initialize
  // got pushed out, as did the first ~50 RefreshCertificates entries.
  REQUIRE(report.recent_events.front().kind == ts::SchedulerEvent::RefreshCertificates);
  REQUIRE(report.recent_events.back().kind == ts::SchedulerEvent::RefreshCertificates);
}

// 9. DeadlockError is catchable as std::runtime_error and carries dump.
TEST_CASE("watchdog — DeadlockError.what() contains dump payload",
          "[scheduler][watchdog][deadlock][what]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));
  sched.set_target_time_level(5);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds{15});
  try {
    sched.check_deadlock(std::chrono::milliseconds{5});
    FAIL("expected DeadlockError");
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    REQUIRE(msg.find("deadlock:") != std::string::npos);
    REQUIRE(msg.find("frontier_min=") != std::string::npos);
    REQUIRE(msg.find("events:") != std::string::npos);
  }
}

// 10. Full lifecycle + re-arrive: frontier bumps and timer stays fresh.
TEST_CASE("watchdog — full lifecycle keeps watchdog happy",
          "[scheduler][watchdog][frontier-progress]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));
  sched.set_target_time_level(10);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);
  sched.refresh_certificates();

  auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  sched.mark_computing(tasks[0]);
  sched.mark_completed(tasks[0]);
  sched.commit_completed();
  sched.release_committed();  // zone → Empty

  std::this_thread::sleep_for(std::chrono::milliseconds{10});
  sched.on_zone_data_arrived(0, 1, 1);  // Empty → ResidentPrev at t=1

  // frontier advanced from 0 to 1; last_progress_ was bumped. No throw.
  REQUIRE_NOTHROW(sched.check_deadlock(std::chrono::milliseconds{20}));
}
