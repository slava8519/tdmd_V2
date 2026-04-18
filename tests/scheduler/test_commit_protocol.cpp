// Exec pack: docs/development/m4_execution_pack.md T4.7
// SPEC: docs/specs/scheduler/SPEC.md §6 (two-phase commit), §13.4 I5
//
// Tests for CausalWavefrontScheduler's two-phase commit protocol:
//
//   1. Phase A only: mark_completed leaves zone at Completed (NOT Committed)
//   2. Phase A + Phase B: commit_completed drains all Completed → Committed
//   3. I5 direct regression: mark_committed from Completed raises (state
//      machine enforces this — Completed only exits to PackedForSend or
//      Committed via commit_completed_no_peer, never via mark_committed)
//   4. commit_completed is selective — leaves Ready / Computing zones alone
//   5. release_committed drains Committed → Empty without touching others
//   6. Full 10-zone lifecycle: Empty → ResidentPrev → Ready → Computing →
//      Completed → Committed → Empty → ResidentPrev (next step)
//   7. version monotonicity through a full cycle
//
// These tests drive the scheduler directly through its public mark_* +
// commit_completed + release_committed + on_zone_data_arrived surface —
// the same surface T4.9's SimulationEngine will use.

#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/scheduler/zone_state_machine.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

namespace ts = tdmd::scheduler;
namespace tz = tdmd::zoning;

namespace {

tz::ZoningPlan make_plan(std::uint32_t nx,
                         std::uint32_t ny,
                         std::uint32_t nz,
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
  p.max_tasks_per_iteration = total_zones;  // let everyone advance in one pass
  p.k_max_pipeline_depth = 1;
  return p;
}

void prime_all_zones(ts::CausalWavefrontScheduler& sched,
                     ts::TimeLevel level,
                     ts::Version version = 0) {
  for (std::size_t z = 0; z < sched.total_zones(); ++z) {
    sched.on_zone_data_arrived(static_cast<ts::ZoneId>(z), level, version);
  }
}

}  // namespace

// 1. Phase A alone — mark_completed leaves the zone at Completed.
TEST_CASE("commit protocol — Phase A only: mark_completed → Completed (I5)",
          "[scheduler][commit][phaseA][I5]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);

  sched.mark_computing(tasks[0]);
  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Computing);

  const auto version_before_completed = sched.zone_meta(0).version;
  sched.mark_completed(tasks[0]);

  // I5 (direct): Completed ≠ Committed. Version bumped (Phase A).
  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Completed);
  REQUIRE(sched.zone_meta(0).version == version_before_completed + 1);
}

// 2. Phase A + Phase B → zone lands at Committed.
TEST_CASE("commit protocol — Phase A + Phase B: commit_completed → Committed",
          "[scheduler][commit][phaseB]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(3)};
  sched.initialize(make_plan(3, 1, 1));

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 3);

  for (const auto& t : tasks) {
    sched.mark_computing(t);
    sched.mark_completed(t);
  }
  // All three at Completed.
  for (std::size_t z = 0; z < 3; ++z) {
    REQUIRE(sched.zone_meta(static_cast<ts::ZoneId>(z)).state == ts::ZoneState::Completed);
  }

  sched.commit_completed();
  for (std::size_t z = 0; z < 3; ++z) {
    const auto& m = sched.zone_meta(static_cast<ts::ZoneId>(z));
    REQUIRE(m.state == ts::ZoneState::Committed);
    REQUIRE(m.cert_id == 0);  // cleared by commit_completed_no_peer
    REQUIRE_FALSE(m.in_ready_queue);
    REQUIRE_FALSE(m.in_inflight_queue);
  }
}

// 3. I5 regression — mark_committed from Completed is illegal (state machine
// throws). This is enforced at the state-machine layer; the scheduler
// forwards mark_committed directly, so the throw propagates.
TEST_CASE("commit protocol — I5 regression: mark_committed from Completed raises",
          "[scheduler][commit][I5]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  sched.mark_computing(tasks[0]);
  sched.mark_completed(tasks[0]);
  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Completed);

  // I5: no direct shortcut to Committed — only commit_completed or the
  // full mark_packed/mark_inflight/mark_committed peer chain is legal.
  REQUIRE_THROWS_AS(sched.mark_committed(tasks[0]), ts::StateMachineError);
  // Transactional reject: state unchanged.
  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Completed);
}

// 4. commit_completed only touches Completed zones; Ready and Computing
// entries stay put.
TEST_CASE("commit protocol — commit_completed is selective (Ready/Computing untouched)",
          "[scheduler][commit][selective]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(4)};
  sched.initialize(make_plan(4, 1, 1));

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 4);

  // Drive zone 0 all the way to Completed.
  sched.mark_computing(tasks[0]);
  sched.mark_completed(tasks[0]);
  // Zone 1 gets stuck at Computing.
  sched.mark_computing(tasks[1]);
  // Zones 2, 3 stay at Ready.

  sched.commit_completed();

  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Committed);
  REQUIRE(sched.zone_meta(1).state == ts::ZoneState::Computing);
  REQUIRE(sched.zone_meta(2).state == ts::ZoneState::Ready);
  REQUIRE(sched.zone_meta(3).state == ts::ZoneState::Ready);
}

// 5. release_committed drains Committed → Empty without touching other
// states.
TEST_CASE("commit protocol — release_committed drains Committed → Empty",
          "[scheduler][commit][release]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(3)};
  sched.initialize(make_plan(3, 1, 1));

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 3);

  // Zones 0, 1 commit; zone 2 stays at Ready.
  for (std::size_t i = 0; i < 2; ++i) {
    sched.mark_computing(tasks[i]);
    sched.mark_completed(tasks[i]);
  }
  sched.commit_completed();

  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Committed);
  REQUIRE(sched.zone_meta(1).state == ts::ZoneState::Committed);
  REQUIRE(sched.zone_meta(2).state == ts::ZoneState::Ready);

  sched.release_committed();

  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Empty);
  REQUIRE(sched.zone_meta(1).state == ts::ZoneState::Empty);
  REQUIRE(sched.zone_meta(2).state == ts::ZoneState::Ready);  // not touched
  // time_level preserved on Committed → Empty (release is non-bumping).
  REQUIRE(sched.zone_meta(0).time_level == 0);
}

// 6. Full 10-zone lifecycle through two time steps — the canonical shape
// of what T4.9's engine loop will drive.
TEST_CASE("commit protocol — 10-zone two-step full lifecycle", "[scheduler][commit][cycle]") {
  constexpr std::uint32_t N = 10;
  // Chain geometry — zone i neighbors zone i+1. Use a non-blocking plan by
  // making the geometry sparse (radius < zone_size), so peer checks don't
  // gate advance.
  auto plan = make_plan(N, 1, 1, /*size=*/1.0, /*cut=*/0.3, /*skin=*/0.3);
  ts::CausalWavefrontScheduler sched{policy_all_ready(N)};
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);

  std::vector<ts::Version> v0(N);
  for (std::uint32_t z = 0; z < N; ++z) {
    v0[z] = sched.zone_meta(z).version;
  }

  // Step 1: refresh → select → compute → complete → commit → release.
  sched.refresh_certificates();
  auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == N);
  for (const auto& t : tasks) {
    sched.mark_computing(t);
    sched.mark_completed(t);
  }
  sched.commit_completed();
  sched.release_committed();

  for (std::uint32_t z = 0; z < N; ++z) {
    const auto& m = sched.zone_meta(z);
    REQUIRE(m.state == ts::ZoneState::Empty);
    REQUIRE(m.version == v0[z] + 1);  // bumped by mark_completed
    REQUIRE(m.cert_id == 0);
  }

  // Step 2: engine re-arms at the next time_level.
  for (std::uint32_t z = 0; z < N; ++z) {
    const auto new_version = sched.zone_meta(z).version;
    sched.on_zone_data_arrived(z, /*step=*/1, new_version);
  }
  for (std::uint32_t z = 0; z < N; ++z) {
    const auto& m = sched.zone_meta(z);
    REQUIRE(m.state == ts::ZoneState::ResidentPrev);
    REQUIRE(m.time_level == 1);
  }

  sched.refresh_certificates();
  tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == N);
  for (const auto& t : tasks) {
    REQUIRE(t.time_level == 2);  // next step is L+1 = 2
    sched.mark_computing(t);
    sched.mark_completed(t);
  }
  sched.commit_completed();
  sched.release_committed();

  for (std::uint32_t z = 0; z < N; ++z) {
    const auto& m = sched.zone_meta(z);
    REQUIRE(m.state == ts::ZoneState::Empty);
    REQUIRE(m.version == v0[z] + 2);  // two Phase A bumps
  }
}

// 7. commit_completed called with no Completed zones is a no-op.
TEST_CASE("commit protocol — commit_completed with no completed zones is a no-op",
          "[scheduler][commit][noop]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(2)};
  sched.initialize(make_plan(2, 1, 1));

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  REQUIRE_NOTHROW(sched.commit_completed());
  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::ResidentPrev);
  REQUIRE(sched.zone_meta(1).state == ts::ZoneState::ResidentPrev);
}

// 8. uninitialized guards (same shape as T4.5 tests, added for T4.7
// extensions).
TEST_CASE("commit protocol — uninitialized commit_completed / release_committed throw",
          "[scheduler][commit][guards]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  REQUIRE_THROWS_AS(sched.commit_completed(), std::logic_error);
  REQUIRE_THROWS_AS(sched.release_committed(), std::logic_error);
}
