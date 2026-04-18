// Exec pack: docs/development/m5_execution_pack.md T5.6
// SPEC: docs/specs/scheduler/SPEC.md §5, §13.4 I6; master spec §6.5a
//
// Property fuzzer for I6 under K ∈ {1, 2, 4, 8} (D-M5-1). Complements
// fuzz_frontier_invariant.cpp (T4.6, random K ∈ [1, 4]) by pinning K to
// the M5 legal set and driving ≥ 100 000 sequences. Every returned task
// must satisfy:
//
//   task.time_level ≤ local_frontier_min + k_max_pipeline_depth
//
// Seed scheme `{kFuzzSeedBase ^ kT56SeedTwist ^ seq}` fully determines
// each sequence — two identical-seed runs produce byte-identical task
// vectors (see `reproducibility` case below).

#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace ts = tdmd::scheduler;
namespace tz = tdmd::zoning;

namespace {

constexpr std::uint64_t kFuzzSeedBase = 0x4D355F4B42544348ULL;  // "M5_KBTCH"
constexpr std::uint64_t kT56SeedTwist = 0xE31A'55D7'0F92'43B7ULL;
constexpr std::size_t kSeqs = 120'000;  // > 100k per T5.6 budget

constexpr std::array<std::uint32_t, 4> kLegalK = {1u, 2u, 4u, 8u};

struct NoisySource : ts::CertificateInputSource {
  mutable std::mt19937_64 rng;
  double unsafe_prob = 0.15;
  double window_expired_prob = 0.15;

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
      out.buffer_width = 0.0;
      out.skin_remaining = 0.0;
      out.frontier_margin = 0.0;
    } else {
      out.buffer_width = 1.0;
      out.skin_remaining = 1.0;
      out.frontier_margin = 1.0;
    }
    out.neighbor_valid_until_step =
        force_window_expired ? (time_level == 0 ? 0 : time_level - 1) : time_level + 100;
    out.halo_valid_until_step = time_level + 1000;
  }
};

tz::ZoningPlan make_random_plan(std::mt19937_64& rng) {
  // Keep total_zones ≤ 16 → well below the OQ-M4-1 ceiling (64) and under
  // the CI wall-time budget for 120k sequences.
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

std::vector<ts::ZoneTask> run_sequence(std::uint64_t seed, std::uint32_t& k_out) {
  std::mt19937_64 rng{seed};
  const auto plan = make_random_plan(rng);

  // Draw K from the legal M5 set — D-M5-1.
  std::uniform_int_distribution<std::size_t> d_k_idx{0, kLegalK.size() - 1};
  const std::uint32_t k = kLegalK[d_k_idx(rng)];
  k_out = k;

  // Cap is also drawn — gives coverage of partial-drain vs drain-all behavior.
  std::uniform_int_distribution<std::uint32_t> d_cap{1, 8};
  const std::uint32_t cap = d_cap(rng);

  auto policy = ts::PolicyFactory::for_reference();
  policy.k_max_pipeline_depth = k;
  policy.max_tasks_per_iteration = cap;

  ts::CausalWavefrontScheduler sched{policy};
  sched.initialize(plan);

  NoisySource src{seed};
  sched.set_certificate_input_source(&src);

  std::uniform_int_distribution<int> d_empty{0, 4};  // 1/5 leave Empty
  std::uniform_int_distribution<ts::TimeLevel> d_tl{0, 15};
  for (std::size_t z = 0; z < sched.total_zones(); ++z) {
    if (d_empty(rng) == 0) {
      continue;
    }
    sched.on_zone_data_arrived(static_cast<ts::ZoneId>(z), d_tl(rng), 0);
  }
  sched.refresh_certificates();
  return sched.select_ready_tasks();
}

}  // namespace

TEST_CASE("K-batching fuzzer — I6 holds for K ∈ {1, 2, 4, 8} across 120k sequences",
          "[scheduler][k_batching][fuzzer][I6]") {
  std::size_t total_calls = 0;
  std::size_t nonempty = 0;
  std::size_t total_tasks_emitted = 0;
  std::array<std::size_t, 4> k_histogram{};

  for (std::size_t seq = 0; seq < kSeqs; ++seq) {
    const std::uint64_t seed = kFuzzSeedBase ^ kT56SeedTwist ^ seq;

    std::uint32_t k = 0;
    const auto tasks = run_sequence(seed, k);
    ++total_calls;
    total_tasks_emitted += tasks.size();
    if (!tasks.empty()) {
      ++nonempty;
    }

    // Re-initialize an identical scheduler to query frontier_min — the one
    // used inside run_sequence is scoped to that function. We regenerate
    // the state by replaying the same seed through an independent session
    // so the assertion is made against the same scheduler snapshot that
    // produced the tasks.
    std::mt19937_64 rng{seed};
    const auto plan = make_random_plan(rng);
    std::uniform_int_distribution<std::size_t> d_k_idx{0, kLegalK.size() - 1};
    const std::uint32_t k_replay = kLegalK[d_k_idx(rng)];
    REQUIRE(k_replay == k);  // reproducibility signal
    std::uniform_int_distribution<std::uint32_t> d_cap{1, 8};
    const std::uint32_t cap = d_cap(rng);
    auto policy = ts::PolicyFactory::for_reference();
    policy.k_max_pipeline_depth = k;
    policy.max_tasks_per_iteration = cap;

    ts::CausalWavefrontScheduler sched{policy};
    sched.initialize(plan);
    NoisySource src{seed};
    sched.set_certificate_input_source(&src);
    std::uniform_int_distribution<int> d_empty{0, 4};
    std::uniform_int_distribution<ts::TimeLevel> d_tl{0, 15};
    for (std::size_t z = 0; z < sched.total_zones(); ++z) {
      if (d_empty(rng) == 0) {
        continue;
      }
      sched.on_zone_data_arrived(static_cast<ts::ZoneId>(z), d_tl(rng), 0);
    }
    sched.refresh_certificates();
    const auto tasks2 = sched.select_ready_tasks();

    // Reproducibility contract: two identical-seed replays emit identical
    // task vectors. If this ever drifts, the seeding convention is broken.
    REQUIRE(tasks2.size() == tasks.size());
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      REQUIRE(tasks2[i].zone_id == tasks[i].zone_id);
      REQUIRE(tasks2[i].time_level == tasks[i].time_level);
    }

    const ts::TimeLevel frontier_min = sched.local_frontier_min();
    for (const auto& t : tasks2) {
      REQUIRE(t.time_level <= frontier_min + k);
    }
    REQUIRE(tasks2.size() <= cap);

    for (std::size_t i = 0; i < kLegalK.size(); ++i) {
      if (kLegalK[i] == k) {
        ++k_histogram[i];
        break;
      }
    }
  }

  REQUIRE(total_calls == kSeqs);
  // Coverage floors: each legal K was drawn at least once (uniform over 4
  // values, so 120 k / 4 ≈ 30 k per bucket — plenty of headroom).
  for (std::size_t i = 0; i < kLegalK.size(); ++i) {
    INFO("K=" << kLegalK[i] << " draws=" << k_histogram[i]);
    REQUIRE(k_histogram[i] > kSeqs / 10);  // ≥ 10 % per bucket
  }
  // Non-empty floor — if the fuzzer emits nothing, I6 is vacuously true and
  // the assertions above are a no-op.
  REQUIRE(nonempty > kSeqs / 4);
  REQUIRE(total_tasks_emitted > kSeqs / 2);
}
