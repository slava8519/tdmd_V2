// Exec pack: docs/development/m4_execution_pack.md T4.6
// SPEC: docs/specs/scheduler/SPEC.md §5 (task selection), §9 (queue semantics),
//       §13.4 I6 (frontier guard)
//
// Unit cases for CausalWavefrontScheduler::select_ready_tasks():
//
//   1. 2-zone chain — single task emitted with Reference policy (cap=1)
//   2. 4-zone 2×2 — one task emitted (cap=1), canonical_index breaks ties
//   3. All-ready with raised cap — N tasks, ordered (t, canonical_index, ver)
//   4. All-blocked (all zones Empty) — empty output, no throw
//   5. Frontier saturation — advanced zone blocked by lagging peer (I6)
//   6. Tie-break determinism — same snapshot → byte-identical output ×100
//   7. Cert missing — no candidate for that zone
//   8. Cert unsafe — no candidate for that zone (stub returns unsafe)
//   9. Peer-block — peer at t-2 blocks selection at t; lowering peer barrier unblocks
//  10. Cap respected — cap=2 on 4 eligible zones returns exactly 2
//  11. Already-Ready zone — not re-selected (would trip I4 in mark_ready)
//  12. Neighbor validity window expired — cert.neighbor_valid_until_step < t skips
//
// Plus a direct test of ReferenceTaskCompare (queues.hpp) to lock the
// strict-weak ordering in byte-stable shape for T4.10 determinism tests.

#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/scheduler/queues.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

namespace ts = tdmd::scheduler;
namespace tz = tdmd::zoning;

namespace {

// Default (cutoff + skin) = 1.0 matches zone_size = 1.0, so face-adjacent
// zones are neighbors in the scheduler's spatial DAG (radius == centre
// distance at the face). Tests that want isolated zones override the last
// two arguments explicitly.
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

// Predictable, always-safe input source. Neighbor validity window is
// parameterized so tests can force it to expire (set `expire_neighbor_at`
// to any value below the candidate time_level).
struct SafeSource : ts::CertificateInputSource {
  ts::TimeLevel neighbor_until_delta = 10;
  bool force_expired_neighbor = false;
  ts::TimeLevel expired_neighbor_until = 0;

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
    out.neighbor_valid_until_step =
        force_expired_neighbor ? expired_neighbor_until : time_level + neighbor_until_delta;
    out.halo_valid_until_step = time_level + 1000;
  }
};

// Always-unsafe: frontier_margin = 0 drives δ < min(...) false.
struct UnsafeSource : ts::CertificateInputSource {
  void fill_inputs(ts::ZoneId zone,
                   ts::TimeLevel time_level,
                   ts::CertificateInputs& out) const override {
    out.zone_id = zone;
    out.time_level = time_level;
    out.v_max_zone = 1.0;
    out.a_max_zone = 10.0;
    out.dt_candidate = 0.1;
    out.buffer_width = 0.001;  // tiny buffer → δ > buffer → unsafe
    out.skin_remaining = 0.001;
    out.frontier_margin = 0.001;
    out.neighbor_valid_until_step = time_level + 100;
    out.halo_valid_until_step = time_level + 100;
  }
};

// Policy helper — Reference with a bumped cap so we can test multi-task
// selection cases without hitting the Reference default of 1.
ts::SchedulerPolicy policy_with_cap(std::uint32_t cap, std::uint32_t k_max = 1) {
  auto p = ts::PolicyFactory::for_reference();
  p.max_tasks_per_iteration = cap;
  p.k_max_pipeline_depth = k_max;
  return p;
}

// Place every zone into ResidentPrev at a given starting time_level.
// Uses on_zone_data_arrived (Empty → ResidentPrev; sets time_level).
void prime_all_zones(ts::CausalWavefrontScheduler& sched,
                     ts::TimeLevel start_level,
                     ts::Version start_version = 0) {
  for (std::size_t z = 0; z < sched.total_zones(); ++z) {
    sched.on_zone_data_arrived(static_cast<ts::ZoneId>(z), start_level, start_version);
  }
}

}  // namespace

TEST_CASE("select_ready_tasks — 2-zone chain emits one task (cap=1)",
          "[scheduler][select][chain]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(2, 1, 1);
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, /*start_level=*/0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  // Canonical index 0 wins the tie-break: zone_id = 0, time_level = 1.
  REQUIRE(tasks[0].zone_id == 0);
  REQUIRE(tasks[0].time_level == 1);
  // ZoneMeta for zone 0 transitioned ResidentPrev → Ready.
  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Ready);
  REQUIRE(sched.zone_meta(1).state == ts::ZoneState::ResidentPrev);
}

TEST_CASE("select_ready_tasks — 4-zone 2×2, cap=1, tie-break by canonical_index",
          "[scheduler][select][tiebreak]") {
  // Scramble canonical_order so zone_id != canonical_index and verify the
  // tie-break uses canonical_index (not zone_id).
  auto plan = make_plan(2, 2, 1);
  plan.canonical_order = {2, 0, 3, 1};  // zone 2 wins
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  REQUIRE(tasks[0].zone_id == 2);  // first in canonical_order
  REQUIRE(tasks[0].time_level == 1);
}

TEST_CASE("select_ready_tasks — all-ready with raised cap returns N ordered tasks",
          "[scheduler][select][multi]") {
  ts::CausalWavefrontScheduler sched{policy_with_cap(/*cap=*/4)};
  const auto plan = make_plan(2, 2, 1);  // 4 zones
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 4);
  // All at time_level 1, canonical_order = identity → zone_id ascending.
  for (std::size_t i = 0; i < 4; ++i) {
    REQUIRE(tasks[i].zone_id == static_cast<ts::ZoneId>(i));
    REQUIRE(tasks[i].time_level == 1);
  }
  // All 4 zones moved to Ready.
  for (ts::ZoneId z = 0; z < 4; ++z) {
    REQUIRE(sched.zone_meta(z).state == ts::ZoneState::Ready);
  }
}

TEST_CASE("select_ready_tasks — all zones Empty → empty output", "[scheduler][select][empty]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(3, 1, 1);
  sched.initialize(plan);
  // No on_zone_data_arrived, no refresh: zones stay Empty.
  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.empty());
}

TEST_CASE("select_ready_tasks — frontier saturation (I6) blocks advanced zone",
          "[scheduler][select][frontier][I6]") {
  // Zone 0 primed at time_level=5, zone 1 primed at time_level=0. With
  // K_max=1, frontier_min=0 → max by frontier = 1. Zone 0 wants t ≥ 6 but
  // the I6 guard caps at frontier_min + K_max = 1 ⇒ no candidate.
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(2, 1, 1);
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, /*step=*/5, 0);
  sched.on_zone_data_arrived(1, /*step=*/0, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 1);
  // Only zone 1 (time_level=0, wants t=1) qualifies; zone 0 is frontier-blocked.
  REQUIRE(tasks[0].zone_id == 1);
  REQUIRE(tasks[0].time_level == 1);
}

TEST_CASE("select_ready_tasks — I6 post-condition on returned tasks", "[scheduler][select][I6]") {
  ts::CausalWavefrontScheduler sched{policy_with_cap(/*cap=*/16, /*k_max=*/1)};
  const auto plan = make_plan(4, 1, 1);  // 4 zones
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  const auto frontier_min = sched.local_frontier_min();
  const std::uint64_t k_max = 1;
  for (const auto& t : tasks) {
    REQUIRE(t.time_level <= frontier_min + k_max);
  }
}

TEST_CASE("select_ready_tasks — deterministic across 100 calls",
          "[scheduler][select][determinism]") {
  // Run select_ready_tasks on identical snapshots 100 times; all outputs
  // must be byte-identical. Each run is a fresh scheduler (select mutates
  // state by transitioning ResidentPrev → Ready).
  std::vector<std::vector<ts::ZoneTask>> runs;
  runs.reserve(100);

  for (int i = 0; i < 100; ++i) {
    ts::CausalWavefrontScheduler sched{policy_with_cap(/*cap=*/4)};
    const auto plan = make_plan(2, 2, 1);
    sched.initialize(plan);
    SafeSource src;
    sched.set_certificate_input_source(&src);
    prime_all_zones(sched, 0);
    sched.refresh_certificates();
    runs.push_back(sched.select_ready_tasks());
  }

  // All runs equal run 0 field-by-field. cert_ids are unequal across runs
  // (fresh scheduler → fresh cert_id counter), but zone_id / time_level /
  // local_state_version / dep_mask / priority / mode_policy_tag must match.
  REQUIRE(runs[0].size() == 4);
  for (std::size_t i = 1; i < runs.size(); ++i) {
    REQUIRE(runs[i].size() == runs[0].size());
    for (std::size_t j = 0; j < runs[i].size(); ++j) {
      REQUIRE(runs[i][j].zone_id == runs[0][j].zone_id);
      REQUIRE(runs[i][j].time_level == runs[0][j].time_level);
      REQUIRE(runs[i][j].local_state_version == runs[0][j].local_state_version);
      REQUIRE(runs[i][j].dep_mask == runs[0][j].dep_mask);
      REQUIRE(runs[i][j].priority == runs[0][j].priority);
      REQUIRE(runs[i][j].mode_policy_tag == runs[0][j].mode_policy_tag);
    }
  }
}

TEST_CASE("select_ready_tasks — missing cert excludes zone", "[scheduler][select][cert]") {
  ts::CausalWavefrontScheduler sched{policy_with_cap(/*cap=*/4)};
  const auto plan = make_plan(3, 1, 1);
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  // Surgically remove zone 1's cert.
  sched.invalidate_certificates_for(1);

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 2);
  for (const auto& t : tasks) {
    REQUIRE(t.zone_id != 1);
  }
}

TEST_CASE("select_ready_tasks — unsafe cert excludes zone", "[scheduler][select][cert][unsafe]") {
  ts::CausalWavefrontScheduler sched{policy_with_cap(/*cap=*/4)};
  const auto plan = make_plan(2, 1, 1);
  sched.initialize(plan);

  UnsafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  // Both certs exist but are marked unsafe.
  REQUIRE(sched.cert_store().size() == 2);
  const auto* c0 = sched.cert_store().get(0, 1);
  REQUIRE(c0 != nullptr);
  REQUIRE_FALSE(c0->safe);

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.empty());
}

TEST_CASE("select_ready_tasks — peer-block when peer lags by more than 1 step",
          "[scheduler][select][peer]") {
  // Zone 0 at time_level=5, zone 1 at time_level=3. K_max large enough
  // to admit any t; zone 0 wants t=6 but peer (zone 1) is only at 3,
  // 3+1 = 4 < 6 → peer-blocked. Zone 1 wants t=4 (since its min_level=4);
  // peer (zone 0) is at 5 ≥ 3 → peer OK.
  auto policy = policy_with_cap(/*cap=*/2, /*k_max=*/10);
  ts::CausalWavefrontScheduler sched{policy};
  const auto plan = make_plan(2, 1, 1);
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 5, 0);
  sched.on_zone_data_arrived(1, 3, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  // Only zone 1 admissible: t = 4 is within [3+1, min(3+1+9, 3+10)] = [4, 13].
  // Peer (zone 0) at time_level=5 ≥ t-1=3 ✓, cert safe, neighbor window OK.
  // Zone 0 candidate at t=6 requires peer ≥ 5; zone 1 at 3 fails.
  REQUIRE(tasks.size() == 1);
  REQUIRE(tasks[0].zone_id == 1);
  REQUIRE(tasks[0].time_level == 4);
}

TEST_CASE("select_ready_tasks — Empty peer blocks (state != Empty required)",
          "[scheduler][select][peer][empty]") {
  // If a peer hasn't received data yet (Empty), we cannot advance past t=1
  // even though time_level arithmetic would allow it. Zone 0 primed, zone 1
  // left Empty.
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(2, 1, 1);
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  sched.on_zone_data_arrived(0, 0, 0);  // zone 1 stays Empty
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.empty());
}

TEST_CASE("select_ready_tasks — max_tasks_per_iteration cap respected",
          "[scheduler][select][cap]") {
  ts::CausalWavefrontScheduler sched{policy_with_cap(/*cap=*/2)};
  const auto plan = make_plan(4, 1, 1);  // 4 zones, all eligible
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.size() == 2);
  // First 2 by canonical order.
  REQUIRE(tasks[0].zone_id == 0);
  REQUIRE(tasks[1].zone_id == 1);
}

TEST_CASE("select_ready_tasks — already-Ready zone not re-selected", "[scheduler][select][dedup]") {
  // After the first select, zones moved to Ready. A second select without
  // any intervening state change must not return the same zones again (they
  // would trip I4 in mark_ready). Two consecutive calls → second is empty.
  ts::CausalWavefrontScheduler sched{policy_with_cap(/*cap=*/4)};
  const auto plan = make_plan(2, 1, 1);
  sched.initialize(plan);

  SafeSource src;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto first = sched.select_ready_tasks();
  REQUIRE(first.size() == 2);

  // Second call: all ResidentPrev zones are now Ready → filter rejects them.
  const auto second = sched.select_ready_tasks();
  REQUIRE(second.empty());
}

TEST_CASE("select_ready_tasks — expired neighbor validity window excludes",
          "[scheduler][select][neighbor]") {
  // Force cert.neighbor_valid_until_step = 0 regardless of candidate
  // time_level. Any candidate t ≥ 1 will fail the `neighbor_valid_until_step
  // < t` check.
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(2, 1, 1);
  sched.initialize(plan);

  SafeSource src;
  src.force_expired_neighbor = true;
  src.expired_neighbor_until = 0;
  sched.set_certificate_input_source(&src);
  prime_all_zones(sched, 0);
  sched.refresh_certificates();

  const auto tasks = sched.select_ready_tasks();
  REQUIRE(tasks.empty());
}

TEST_CASE("ReferenceTaskCompare — strict weak ordering by (t, idx, ver)",
          "[scheduler][queues][compare]") {
  const ts::ReferenceTaskCompare cmp;

  const ts::TaskCandidate a{.zone_id = 0, .time_level = 1, .canonical_index = 2, .version = 3};
  const ts::TaskCandidate b{.zone_id = 1, .time_level = 1, .canonical_index = 2, .version = 3};

  // Equal: neither less than the other.
  REQUIRE_FALSE(cmp(a, b));
  REQUIRE_FALSE(cmp(b, a));

  // Time level dominates.
  const ts::TaskCandidate lo_t{.zone_id = 9, .time_level = 0, .canonical_index = 9, .version = 9};
  REQUIRE(cmp(lo_t, a));
  REQUIRE_FALSE(cmp(a, lo_t));

  // Within same t, canonical_index dominates.
  const ts::TaskCandidate lo_idx{.zone_id = 9, .time_level = 1, .canonical_index = 0, .version = 9};
  REQUIRE(cmp(lo_idx, a));
  REQUIRE_FALSE(cmp(a, lo_idx));

  // Within same t and idx, version dominates.
  const ts::TaskCandidate lo_ver{.zone_id = 0, .time_level = 1, .canonical_index = 2, .version = 0};
  REQUIRE(cmp(lo_ver, a));
  REQUIRE_FALSE(cmp(a, lo_ver));
}
