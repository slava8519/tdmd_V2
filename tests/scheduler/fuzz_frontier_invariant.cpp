// Exec pack: docs/development/m4_execution_pack.md T4.6, D-M4-8
// SPEC: docs/specs/scheduler/SPEC.md §5 (task selection), §13.4 I6
// Master spec: §6.7, §13.4
//
// Property fuzzer for I6 (frontier guard): after every call to
// select_ready_tasks(), every returned task must satisfy
//
//     task.time_level ≤ local_frontier_min + K_max
//
// Strategy per sequence:
//
//   1. Random plan geometry (1×1×1 .. 4×2×2, keeping total_zones ≤ 16).
//   2. Random K_max in [1, 4], random cap in [1, 8].
//   3. Per zone: random starting time_level in [0, 7]; optionally leave
//      the zone Empty (roughly 1/5 probability) to exercise peer gates.
//   4. refresh_certificates with a stub source that may randomly mark the
//      cert unsafe (20% of zones) or expire neighbor_valid_until_step
//      (20% disjoint).
//   5. Call select_ready_tasks, compute frontier_min, assert I6 for every
//      returned task. Also assert canonical-order tie-break holds (no
//      returned task violates the strict-weak ordering vs the one before).
//
// Reproducibility contract: `{seed_base ^ twist_T46 ^ seq}` fully
// determines the sequence; two passes with the same seed must produce
// identical tasks vectors.

#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/scheduler/queues.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace ts = tdmd::scheduler;
namespace tz = tdmd::zoning;

namespace {

constexpr std::uint64_t kFuzzSeedBase = 0x4D345F53434845ULL;       // "M4_SCHE"
constexpr std::uint64_t kT46SeedTwist = 0x27B1'AD4C'F619'0873ULL;  // T4.6 offset
constexpr std::size_t kSeqs = 250'000;

// Random source that occasionally produces unsafe certs or expired neighbor
// windows. `rng` is mutated on each fill_inputs → deterministic only if the
// scheduler iterates zones in a fixed order (it does: canonical_order). The
// source therefore wraps its own rng and we reseed it before every fuzz
// sequence so refresh_certificates is reproducible from (seed, seq).
struct NoisySource : ts::CertificateInputSource {
  mutable std::mt19937_64 rng;
  double unsafe_prob = 0.2;
  double window_expired_prob = 0.2;

  explicit NoisySource(std::uint64_t seed) : rng{seed} {}

  void fill_inputs(ts::ZoneId zone,
                   ts::TimeLevel time_level,
                   ts::CertificateInputs& out) const override {
    std::uniform_real_distribution<double> u{0.0, 1.0};
    const bool force_unsafe = u(rng) < unsafe_prob;
    const bool force_window_expired = !force_unsafe && u(rng) < window_expired_prob;

    out.zone_id = zone;
    out.time_level = time_level;
    out.v_max_zone = 0.1;
    out.a_max_zone = 0.2;
    out.dt_candidate = 0.001;
    if (force_unsafe) {
      out.buffer_width = 0.0;  // δ > 0 but buffer = 0 → unsafe
      out.skin_remaining = 0.0;
      out.frontier_margin = 0.0;
    } else {
      out.buffer_width = 1.0;
      out.skin_remaining = 1.0;
      out.frontier_margin = 1.0;
    }
    if (force_window_expired) {
      out.neighbor_valid_until_step = time_level == 0 ? 0 : time_level - 1;
    } else {
      out.neighbor_valid_until_step = time_level + 100;
    }
    out.halo_valid_until_step = time_level + 1000;
  }
};

tz::ZoningPlan make_random_plan(std::mt19937_64& rng) {
  // Keep total_zones small (≤ 16) so the OQ-M4-1 ceiling (64) is never hit
  // and the fuzzer stays under its CI wall-time budget.
  std::uniform_int_distribution<std::uint32_t> d_nx{1, 4};
  std::uniform_int_distribution<std::uint32_t> d_ny{1, 2};
  std::uniform_int_distribution<std::uint32_t> d_nz{1, 2};
  const auto nx = d_nx(rng);
  const auto ny = d_ny(rng);
  const auto nz = d_nz(rng);

  tz::ZoningPlan plan;
  plan.scheme = tz::ZoningScheme::Linear1D;
  plan.n_zones = {nx, ny, nz};
  plan.zone_size = {1.0, 1.0, 1.0};
  plan.cutoff = 0.4;
  plan.skin = 0.4;
  plan.buffer_width = {0.4, 0.4, 0.4};
  const auto total = static_cast<tz::ZoneId>(nx * ny * nz);
  plan.canonical_order.reserve(total);
  for (tz::ZoneId z = 0; z < total; ++z) {
    plan.canonical_order.push_back(z);
  }
  plan.n_min_per_rank = 1;
  plan.optimal_rank_count = total;
  return plan;
}

}  // namespace

TEST_CASE("select_ready_tasks — I6 frontier invariant fuzzer (≥250k sequences)",
          "[scheduler][select][fuzzer][I6]") {
  std::size_t total_calls = 0;
  std::size_t total_tasks_emitted = 0;
  std::size_t nonempty_selections = 0;

  for (std::size_t seq = 0; seq < kSeqs; ++seq) {
    const std::uint64_t seed = kFuzzSeedBase ^ kT46SeedTwist ^ seq;
    std::mt19937_64 rng{seed};

    // Build plan + policy.
    const auto plan = make_random_plan(rng);
    const std::uint32_t k_max = std::uniform_int_distribution<std::uint32_t>{1, 4}(rng);
    const std::uint32_t cap = std::uniform_int_distribution<std::uint32_t>{1, 8}(rng);
    auto policy = ts::PolicyFactory::for_reference();
    policy.k_max_pipeline_depth = k_max;
    policy.max_tasks_per_iteration = cap;

    ts::CausalWavefrontScheduler sched{policy};
    sched.initialize(plan);

    // Seed the noisy source from the same sequence seed so cert choices
    // are reproducible.
    NoisySource src{seed};
    sched.set_certificate_input_source(&src);

    // Randomly prime each zone at a random time_level, or leave it Empty.
    std::uniform_int_distribution<int> d_empty{0, 4};  // 1/5 → leave Empty
    std::uniform_int_distribution<ts::TimeLevel> d_tl{0, 7};
    for (std::size_t z = 0; z < sched.total_zones(); ++z) {
      if (d_empty(rng) == 0) {
        continue;  // leave Empty
      }
      const auto tl = d_tl(rng);
      sched.on_zone_data_arrived(static_cast<ts::ZoneId>(z), tl, 0);
    }

    sched.refresh_certificates();

    const auto tasks = sched.select_ready_tasks();
    ++total_calls;
    total_tasks_emitted += tasks.size();
    if (!tasks.empty()) {
      ++nonempty_selections;
    }

    // Compute frontier_min across non-Empty metas only — the scheduler's
    // local_frontier_min() reports over all zones regardless of state, which
    // is the guard's reference value. Use the scheduler-observed value for
    // the assertion (that is the contract).
    const ts::TimeLevel frontier_min = sched.local_frontier_min();

    // I6: every returned task satisfies time_level ≤ frontier_min + k_max.
    for (const auto& t : tasks) {
      REQUIRE(t.time_level <= frontier_min + k_max);
      // Cap respected.
    }
    REQUIRE(tasks.size() <= cap);

    // Tie-break ordering: the returned vector must be non-decreasing under
    // ReferenceTaskCompare (we only have zone_id/time_level/version, not
    // canonical_index — in this test canonical_order is identity so
    // canonical_index == zone_id).
    for (std::size_t i = 1; i < tasks.size(); ++i) {
      const auto& prev = tasks[i - 1];
      const auto& cur = tasks[i];
      ts::TaskCandidate a{
          .zone_id = prev.zone_id,
          .time_level = prev.time_level,
          .canonical_index = prev.zone_id,
          .version = prev.local_state_version,
      };
      ts::TaskCandidate b{
          .zone_id = cur.zone_id,
          .time_level = cur.time_level,
          .canonical_index = cur.zone_id,
          .version = cur.local_state_version,
      };
      REQUIRE_FALSE(ts::ReferenceTaskCompare{}(b, a));  // not out-of-order
    }

    // Scheduler-level assertion: every returned task's zone is now Ready.
    for (const auto& t : tasks) {
      REQUIRE(sched.zone_meta(t.zone_id).state == ts::ZoneState::Ready);
    }
  }

  // Coverage floors: if the fuzzer never produces a non-empty selection,
  // we aren't actually exercising the I6 post-condition.
  REQUIRE(total_calls == kSeqs);
  REQUIRE(nonempty_selections > kSeqs / 4);  // ≥ 25 % non-empty
  REQUIRE(total_tasks_emitted > kSeqs / 2);  // ≥ 0.5 tasks/seq on avg
}

TEST_CASE("select_ready_tasks — fuzzer reproducibility", "[scheduler][select][fuzzer][repro]") {
  // Two identical-seed runs of the fuzzer core must produce identical
  // task outputs. Small sample — scaffolding sanity test.
  auto one_run = [](std::uint64_t seed) {
    std::mt19937_64 rng{seed};
    const auto plan = make_random_plan(rng);
    const std::uint32_t k_max = std::uniform_int_distribution<std::uint32_t>{1, 4}(rng);
    const std::uint32_t cap = std::uniform_int_distribution<std::uint32_t>{1, 8}(rng);
    auto policy = ts::PolicyFactory::for_reference();
    policy.k_max_pipeline_depth = k_max;
    policy.max_tasks_per_iteration = cap;

    ts::CausalWavefrontScheduler sched{policy};
    sched.initialize(plan);
    NoisySource src{seed};
    sched.set_certificate_input_source(&src);

    std::uniform_int_distribution<int> d_empty{0, 4};
    std::uniform_int_distribution<ts::TimeLevel> d_tl{0, 7};
    for (std::size_t z = 0; z < sched.total_zones(); ++z) {
      if (d_empty(rng) == 0) {
        continue;
      }
      const auto tl = d_tl(rng);
      sched.on_zone_data_arrived(static_cast<ts::ZoneId>(z), tl, 0);
    }
    sched.refresh_certificates();
    return sched.select_ready_tasks();
  };

  const auto a = one_run(kFuzzSeedBase ^ kT46SeedTwist ^ 12345ULL);
  const auto b = one_run(kFuzzSeedBase ^ kT46SeedTwist ^ 12345ULL);
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i].zone_id == b[i].zone_id);
    REQUIRE(a[i].time_level == b[i].time_level);
    REQUIRE(a[i].local_state_version == b[i].local_state_version);
    REQUIRE(a[i].dep_mask == b[i].dep_mask);
  }
}
