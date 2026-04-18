// Exec pack: docs/development/m4_execution_pack.md T4.7
// SPEC: docs/specs/scheduler/SPEC.md §7 (retry policy), §11.1 (D-M4-13)
//
// Tests for CausalWavefrontScheduler's RetryTracker wiring:
//
//   1. Basic increment / reset / overflow (RetryTracker unit)
//   2. reset_for drops every counter for a zone
//   3. cert invalidation increments retry for the Ready zone
//   4. Successful mark_completed resets (zone, time_level) counter
//   5. on_zone_data_arrived resets all counters for the zone (new cycle)
//   6. Retry budget exhausted → RetryExhaustedError propagates
//   7. Retry canonical ≥100 random failure-injection sequences — same seed →
//      same retry_count trajectory across runs (Reference profile determinism)
//
// The retry-fuzz case models the scheduler as a driver that tries to
// commit every zone but suffers random cert invalidations mid-flight.
// Because Reference profile forbids rand-based backoff (SPEC §7.2), the
// retry_count at every step is a pure function of (seed, event sequence).

#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/scheduler/retry_state.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <vector>

namespace ts = tdmd::scheduler;
namespace tz = tdmd::zoning;

namespace {

tz::ZoningPlan make_plan(std::uint32_t nx,
                         std::uint32_t ny,
                         std::uint32_t nz,
                         double zone_size = 1.0,
                         double cutoff = 0.2,
                         double skin = 0.2) {
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

ts::SchedulerPolicy policy_all_ready(std::uint32_t cap, std::uint32_t max_retries = 3) {
  auto p = ts::PolicyFactory::for_reference();
  p.max_tasks_per_iteration = cap;
  p.k_max_pipeline_depth = 1;
  p.max_retries_per_task = max_retries;
  return p;
}

}  // namespace

// 1. Unit — increment / reset / overflow.
TEST_CASE("RetryTracker — basic increment / reset / overflow", "[scheduler][retry][unit]") {
  ts::RetryTracker rt{3};
  REQUIRE(rt.max_retries() == 3);
  REQUIRE(rt.count_of(5, 7) == 0);

  REQUIRE(rt.increment(5, 7) == 1);
  REQUIRE(rt.increment(5, 7) == 2);
  REQUIRE(rt.increment(5, 7) == 3);
  REQUIRE(rt.count_of(5, 7) == 3);

  REQUIRE_THROWS_AS(rt.increment(5, 7), ts::RetryExhaustedError);

  rt.reset(5, 7);
  REQUIRE(rt.count_of(5, 7) == 0);

  // A fresh key starts at 0.
  REQUIRE(rt.increment(5, 8) == 1);
  REQUIRE(rt.count_of(5, 7) == 0);
  REQUIRE(rt.count_of(5, 8) == 1);
}

// 2. reset_for drops every counter for a zone across levels.
TEST_CASE("RetryTracker — reset_for zone drops all levels", "[scheduler][retry][unit]") {
  ts::RetryTracker rt{5};
  rt.increment(1, 10);
  rt.increment(1, 11);
  rt.increment(1, 12);
  rt.increment(2, 10);
  REQUIRE(rt.tracked_count() == 4);

  rt.reset_for(1);
  REQUIRE(rt.count_of(1, 10) == 0);
  REQUIRE(rt.count_of(1, 11) == 0);
  REQUIRE(rt.count_of(1, 12) == 0);
  REQUIRE(rt.count_of(2, 10) == 1);  // other zone preserved
  REQUIRE(rt.tracked_count() == 1);
}

// 3. Scheduler-level: cert invalidation on a Ready zone increments retry.
TEST_CASE("retry wiring — invalidate_certificates_for increments retry (Ready zone)",
          "[scheduler][retry][wiring]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Ready);
  REQUIRE(sched.retry_tracker().count_of(0, 1) == 0);

  sched.invalidate_certificates_for(0);
  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::ResidentPrev);
  REQUIRE(sched.retry_tracker().count_of(0, 1) == 1);
}

// 4. Successful mark_completed resets the (zone, time_level) counter.
TEST_CASE("retry wiring — mark_completed resets retry counter for (zone, level)",
          "[scheduler][retry][wiring]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);

  // Simulate one cert invalidation before a successful compute.
  sched.refresh_certificates();
  auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  sched.invalidate_certificates_for(0);
  REQUIRE(sched.retry_tracker().count_of(0, 1) == 1);

  // Re-refresh, re-select, drive to completion.
  sched.refresh_certificates();
  tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  sched.mark_computing(tasks[0]);
  sched.mark_completed(tasks[0]);
  REQUIRE(sched.retry_tracker().count_of(0, 1) == 0);  // reset on success
}

// 5. on_zone_data_arrived clears all counters for the zone (fresh cycle).
TEST_CASE("retry wiring — on_zone_data_arrived clears retry budget for zone",
          "[scheduler][retry][wiring]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1)};
  sched.initialize(make_plan(1, 1, 1));

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);

  sched.refresh_certificates();
  auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  sched.invalidate_certificates_for(0);
  REQUIRE(sched.retry_tracker().count_of(0, 1) == 1);

  // Drive the zone through a full cycle back to Empty, then re-arrive.
  sched.refresh_certificates();
  tasks = sched.select_ready_tasks();
  sched.mark_computing(tasks[0]);
  sched.mark_completed(tasks[0]);
  sched.commit_completed();
  sched.release_committed();
  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Empty);

  sched.on_zone_data_arrived(0, 1, sched.zone_meta(0).version);
  // Any residual counter is wiped by on_zone_data_arrived's reset_for.
  REQUIRE(sched.retry_tracker().count_of(0, 1) == 0);
  REQUIRE(sched.retry_tracker().tracked_count() == 0);
}

// 6. Retry budget exhausted — the 4th invalidation throws.
TEST_CASE("retry wiring — budget exhausted after max_retries_per_task",
          "[scheduler][retry][wiring][overflow]") {
  ts::CausalWavefrontScheduler sched{policy_all_ready(1, /*max_retries=*/3)};
  sched.initialize(make_plan(1, 1, 1));

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);

  // Three invalidations absorbed, fourth throws.
  for (int i = 1; i <= 3; ++i) {
    sched.refresh_certificates();
    const auto tasks = sched.select_ready_tasks();
    REQUIRE(tasks.size() == 1);
    sched.invalidate_certificates_for(0);
    REQUIRE(sched.retry_tracker().count_of(0, 1) == static_cast<std::uint32_t>(i));
  }

  sched.refresh_certificates();
  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  REQUIRE_THROWS_AS(sched.invalidate_certificates_for(0), ts::RetryExhaustedError);
}

// 7. Retry canonical: random failure injection is deterministic per seed.
// The scheduler state (post-run retry counters per zone) must be identical
// across runs sharing the same seed. SPEC §7.2 Reference requires this.
TEST_CASE("retry canonical — same seed → same retry trajectory",
          "[scheduler][retry][canonical][fuzz]") {
  constexpr std::uint32_t Z = 4;
  constexpr int kIters = 200;
  constexpr std::uint64_t kSeedBase = 0x4D345F53434845ULL ^ 0xC3B2'11A5'7F0E'9D40ULL;

  auto run = [&](std::uint64_t seed) {
    ts::CausalWavefrontScheduler sched{policy_all_ready(Z, /*max_retries=*/1'000'000)};
    // Geometry: 4 independent zones (non-neighbors → peer checks pass).
    sched.initialize(make_plan(Z, 1, 1, /*size=*/1.0, /*cut=*/0.1, /*skin=*/0.1));

    SafeSource src;
    sched.set_certificate_input_source(&src);
    for (std::uint32_t z = 0; z < Z; ++z) {
      sched.on_zone_data_arrived(z, 0, 0);
    }

    std::mt19937_64 rng{seed};
    for (int i = 0; i < kIters; ++i) {
      sched.refresh_certificates();
      const auto tasks = sched.select_ready_tasks();
      // For each task, coin-flip: 30% invalidate (retry++), else complete.
      for (const auto& t : tasks) {
        const auto coin = std::uniform_int_distribution<int>{0, 99}(rng);
        if (coin < 30) {
          sched.invalidate_certificates_for(t.zone_id);
        } else {
          sched.mark_computing(t);
          sched.mark_completed(t);
        }
      }
      // Commit + release any Completed zones so the next iteration can
      // re-arrive them at the next step.
      sched.commit_completed();
      sched.release_committed();
      for (std::uint32_t z = 0; z < Z; ++z) {
        if (sched.zone_meta(z).state == ts::ZoneState::Empty) {
          const auto new_level = sched.zone_meta(z).time_level + 1;
          sched.on_zone_data_arrived(z, new_level, sched.zone_meta(z).version);
        }
      }
    }

    std::vector<std::uint32_t> snapshot;
    snapshot.reserve(Z * 4);
    for (std::uint32_t z = 0; z < Z; ++z) {
      for (ts::TimeLevel l = 0; l <= sched.zone_meta(z).time_level + 1; ++l) {
        snapshot.push_back(sched.retry_tracker().count_of(z, l));
      }
      snapshot.push_back(static_cast<std::uint32_t>(sched.zone_meta(z).time_level));
      snapshot.push_back(static_cast<std::uint32_t>(sched.zone_meta(z).version));
    }
    return snapshot;
  };

  for (std::uint64_t twist = 0; twist < 100; ++twist) {
    const auto seed = kSeedBase ^ twist;
    const auto a = run(seed);
    const auto b = run(seed);
    REQUIRE(a == b);  // canonical: same seed → byte-identical trajectory
  }
}
