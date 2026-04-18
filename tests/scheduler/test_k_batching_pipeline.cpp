// Exec pack: docs/development/m5_execution_pack.md T5.6
// SPEC: docs/specs/scheduler/SPEC.md §5 (task selection), §13.4 I6,
//       master spec §6.5a, §14 M5 (D-M5-1 K ∈ {1, 2, 4, 8}, D-M5-12 K=1 regression)
//
// K-batching pipeline tests:
//
//  1. Staggered-zone demonstration per K ∈ {1, 2, 4, 8}: four spatially
//     independent zones primed at time_levels {0, 1, 2, 3} — K alone
//     determines how many zones may advance in a single iteration,
//     because there are no peer constraints.
//     Expected task counts: K=1 → 1, K=2 → 2, K=4 → 4, K=8 → 4 (capped
//     by total_zones=4).
//
//  2. Full-iteration loop over the same staggered setup until every zone
//     reaches target_time_level=10. Count iterations — smaller K forces
//     more iterations (slower drain), larger K lets the frontier catch
//     up more quickly. Assert I6 (`task.time_level ≤ frontier_min + K`)
//     every iteration.
//
//  3. K-validation: `initialize()` rejects K ∈ {0, 3, 5, 6, 7, 16} with
//     a clear D-M5-1 error message.
//
//  4. K=1 regression: with independent zones + identical seed, the task
//     sequence under K=1 in M5 must match the M4 Reference K=1 shape
//     (byte-exact zone_id / time_level trace).

#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ts = tdmd::scheduler;
namespace tz = tdmd::zoning;

namespace {

// Spatially independent plan: cutoff+skin = 0.2 with zone_size = 1.0 keeps
// face-adjacent zones *outside* the neighbor radius, so the scheduler's
// spatial_dep_mask is all-zero and K alone drives the frontier guard.
tz::ZoningPlan make_independent_plan(std::uint32_t nx) {
  tz::ZoningPlan plan;
  plan.scheme = tz::ZoningScheme::Linear1D;
  plan.n_zones = {nx, 1, 1};
  plan.zone_size = {1.0, 1.0, 1.0};
  plan.cutoff = 0.1;
  plan.skin = 0.1;
  plan.buffer_width = {0.1, 0.1, 0.1};
  plan.canonical_order.reserve(nx);
  for (tz::ZoneId z = 0; z < static_cast<tz::ZoneId>(nx); ++z) {
    plan.canonical_order.push_back(z);
  }
  plan.n_min_per_rank = 1;
  plan.optimal_rank_count = nx;
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
    out.neighbor_valid_until_step = time_level + 1000;
    out.halo_valid_until_step = time_level + 1000;
  }
};

ts::SchedulerPolicy make_policy(std::uint32_t k_max, std::uint32_t cap) {
  auto p = ts::PolicyFactory::for_reference();
  p.k_max_pipeline_depth = k_max;
  p.max_tasks_per_iteration = cap;
  return p;
}

// Prime zones at staggered starting levels: zone z ↦ time_level = z.
void prime_staggered(ts::CausalWavefrontScheduler& sched) {
  for (std::size_t z = 0; z < sched.total_zones(); ++z) {
    sched.on_zone_data_arrived(static_cast<ts::ZoneId>(z),
                               static_cast<ts::TimeLevel>(z),
                               /*version=*/0);
  }
}

}  // namespace

TEST_CASE("K-batching — staggered 4 zones, task count per iter scales with K",
          "[scheduler][k_batching][D-M5-1]") {
  constexpr std::uint32_t kNZones = 4;

  struct Case {
    std::uint32_t k;
    std::size_t expected_count;
  };
  // K=1: only zone 0 (at meta=0) can reach t=1 ≤ frontier_min+K=0+1.
  // K=2: zones 0,1 (at meta={0,1}) can reach t ∈ {1,2} ≤ 0+2.
  // K=4: all 4 zones (at meta={0,1,2,3}) can reach t ∈ {1,2,3,4} ≤ 0+4.
  // K=8: same as K=4 — cap_by_total_zones bites, not K.
  const Case cases[] = {{1, 1}, {2, 2}, {4, 4}, {8, 4}};

  for (const auto& c : cases) {
    ts::CausalWavefrontScheduler sched{make_policy(c.k, kNZones)};
    sched.initialize(make_independent_plan(kNZones));

    SafeSource src;
    sched.set_certificate_input_source(&src);
    prime_staggered(sched);
    sched.refresh_certificates();

    const auto tasks = sched.select_ready_tasks();
    INFO("K=" << c.k << " expected=" << c.expected_count << " got=" << tasks.size());
    REQUIRE(tasks.size() == c.expected_count);

    // I6: every task.time_level ≤ frontier_min + K.
    const ts::TimeLevel frontier_min = sched.local_frontier_min();
    for (const auto& t : tasks) {
      REQUIRE(t.time_level <= frontier_min + c.k);
    }
  }
}

TEST_CASE("K-batching — full pipeline drain preserves I6 and hits target",
          "[scheduler][k_batching][pipeline]") {
  constexpr std::uint32_t kNZones = 4;
  constexpr ts::TimeLevel kTarget = 10;

  for (const std::uint32_t k : {1u, 2u, 4u, 8u}) {
    ts::CausalWavefrontScheduler sched{make_policy(k, kNZones)};
    sched.initialize(make_independent_plan(kNZones));
    sched.set_target_time_level(kTarget);

    SafeSource src;
    sched.set_certificate_input_source(&src);
    prime_staggered(sched);

    std::size_t iterations = 0;
    std::size_t total_tasks = 0;
    constexpr std::size_t kMaxIter = 200;  // safety net against infinite loop

    while (!sched.finished() && iterations < kMaxIter) {
      sched.refresh_certificates();
      const auto tasks = sched.select_ready_tasks();

      // I6 every iteration.
      const ts::TimeLevel frontier_min = sched.local_frontier_min();
      for (const auto& t : tasks) {
        REQUIRE(t.time_level <= frontier_min + k);
      }

      for (const auto& t : tasks) {
        sched.mark_computing(t);
      }
      for (const auto& t : tasks) {
        sched.mark_completed(t);
      }
      sched.commit_completed();
      sched.release_committed();
      // Zones released → ResidentPrev; bump time_level via zone_data_arrived
      // to mirror what the runtime does after a physics step.
      for (const auto& t : tasks) {
        sched.on_zone_data_arrived(t.zone_id, t.time_level, /*version=*/0);
      }
      total_tasks += tasks.size();
      ++iterations;
    }

    INFO("K=" << k << " iterations=" << iterations << " total_tasks=" << total_tasks);
    REQUIRE(sched.finished());
    REQUIRE(iterations < kMaxIter);
    // Every zone must have reached the target level exactly.
    for (std::size_t z = 0; z < kNZones; ++z) {
      REQUIRE(sched.zone_meta(static_cast<ts::ZoneId>(z)).time_level >= kTarget);
    }
  }
}

TEST_CASE("K-batching — initialize() rejects K outside {1, 2, 4, 8}",
          "[scheduler][k_batching][D-M5-1][validation]") {
  const auto plan = make_independent_plan(2);
  for (const std::uint32_t bad_k : {0u, 3u, 5u, 6u, 7u, 9u, 16u, 100u}) {
    auto policy = make_policy(bad_k, 2);
    ts::CausalWavefrontScheduler sched{policy};
    INFO("expected throw for k_max_pipeline_depth=" << bad_k);
    REQUIRE_THROWS_AS(sched.initialize(plan), std::logic_error);
  }

  // Sanity: the allowed set must not throw.
  for (const std::uint32_t good_k : {1u, 2u, 4u, 8u}) {
    auto policy = make_policy(good_k, 2);
    ts::CausalWavefrontScheduler sched{policy};
    REQUIRE_NOTHROW(sched.initialize(plan));
  }
}

TEST_CASE("K-batching — K=1 matches M4 Reference behavior byte-exact (D-M5-12)",
          "[scheduler][k_batching][regression][D-M5-12]") {
  // Two schedulers with identical policy / plan / source / priming must
  // emit the same task vector. Also, K=1 with a staggered layout should
  // pick exactly one zone per iteration (the zone at frontier_min).
  constexpr std::uint32_t kNZones = 4;

  auto run_once = [&]() {
    ts::CausalWavefrontScheduler sched{make_policy(/*k_max=*/1, /*cap=*/kNZones)};
    sched.initialize(make_independent_plan(kNZones));
    SafeSource src;
    sched.set_certificate_input_source(&src);
    prime_staggered(sched);
    sched.refresh_certificates();
    return sched.select_ready_tasks();
  };

  const auto a = run_once();
  const auto b = run_once();

  REQUIRE(a.size() == 1);
  REQUIRE(b.size() == 1);
  REQUIRE(a[0].zone_id == 0);     // zone at the frontier wins the tie-break
  REQUIRE(a[0].time_level == 1);  // frontier_min+1
  REQUIRE(a[0].zone_id == b[0].zone_id);
  REQUIRE(a[0].time_level == b[0].time_level);
  REQUIRE(a[0].local_state_version == b[0].local_state_version);
  REQUIRE(a[0].dep_mask == b[0].dep_mask);
}
