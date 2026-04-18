// Exec pack: docs/development/m4_execution_pack.md T4.5
// SPEC: docs/specs/scheduler/SPEC.md §2.3, §4.4, §10
//
// Core tests for CausalWavefrontScheduler: lifecycle, canonical order echo,
// spatial DAG shape on three small geometries, refresh_certificates count
// + ordering, invalidation semantics, and event handler forwards.
//
// Plans are hand-constructed rather than running the zoning planner — this
// keeps the test independent of zoning implementation details and makes
// the grid-to-ZoneId mapping explicit.

#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/scheduler/zone_dag.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace ts = tdmd::scheduler;
namespace tz = tdmd::zoning;

namespace {

// Build a ZoningPlan with nx × ny × nz zones of uniform `zone_size`, using
// the same row-major ZoneId convention that compute_spatial_dependencies
// assumes. canonical_order is the identity permutation — Hilbert shuffles
// are tested separately in the zoning module.
tz::ZoningPlan make_plan(std::uint32_t nx,
                         std::uint32_t ny,
                         std::uint32_t nz,
                         double zone_size,
                         double cutoff,
                         double skin) {
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

std::size_t popcount_u64(std::uint64_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<std::size_t>(__builtin_popcountll(x));
#else
  std::size_t c = 0;
  while (x) {
    x &= x - 1;
    ++c;
  }
  return c;
#endif
}

}  // namespace

TEST_CASE("CausalWavefrontScheduler — lifecycle", "[scheduler][core][lifecycle]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};

  REQUIRE_FALSE(sched.initialized());
  REQUIRE(sched.total_zones() == 0);

  const auto plan = make_plan(2, 2, 2, /*size=*/3.0, /*cut=*/1.0, /*skin=*/1.0);
  REQUIRE_NOTHROW(sched.initialize(plan));
  REQUIRE(sched.initialized());
  REQUIRE(sched.total_zones() == 8);

  // Pattern 1 attach is nullptr.
  REQUIRE_NOTHROW(sched.attach_outer_coordinator(nullptr));
  REQUIRE(sched.outer_coordinator() == nullptr);

  // Double-initialize rejected.
  REQUIRE_THROWS_AS(sched.initialize(plan), std::logic_error);

  REQUIRE_NOTHROW(sched.shutdown());
  REQUIRE_FALSE(sched.initialized());
  REQUIRE(sched.total_zones() == 0);

  // Shutdown is idempotent.
  REQUIRE_NOTHROW(sched.shutdown());
}

TEST_CASE("CausalWavefrontScheduler — uninitialized operations throw",
          "[scheduler][core][guards]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  REQUIRE_THROWS_AS(sched.refresh_certificates(), std::logic_error);
  REQUIRE_THROWS_AS(sched.invalidate_certificates_for(0), std::logic_error);
  REQUIRE_THROWS_AS(sched.invalidate_all_certificates("x"), std::logic_error);
  REQUIRE_THROWS_AS(sched.select_ready_tasks(), std::logic_error);
}

TEST_CASE("CausalWavefrontScheduler — rejects >64 zones (OQ-M4-1)", "[scheduler][core][limits]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(5, 5, 4, 1.0, 0.5, 0.5);  // 100 zones
  REQUIRE_THROWS_AS(sched.initialize(plan), std::logic_error);
}

TEST_CASE("CausalWavefrontScheduler — canonical_order echoed verbatim",
          "[scheduler][core][canonical]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  auto plan = make_plan(2, 3, 1, 1.0, 0.5, 0.5);
  // Scramble the canonical order to a non-identity permutation.
  plan.canonical_order = {3, 0, 4, 1, 5, 2};
  REQUIRE_NOTHROW(sched.initialize(plan));

  const auto& echoed = sched.canonical_order();
  REQUIRE(echoed.size() == 6);
  for (std::size_t i = 0; i < 6; ++i) {
    REQUIRE(echoed[i] == plan.canonical_order[i]);
  }

  // Stored by value: mutating the plan post-initialize doesn't affect the
  // scheduler's copy.
  plan.canonical_order[0] = 99;
  REQUIRE(sched.canonical_order()[0] == 3);
}

TEST_CASE("Spatial DAG — face-adjacency metric on uniform grids", "[scheduler][core][dag]") {
  SECTION("1×1×1 — lone zone has no neighbours") {
    const auto plan = make_plan(1, 1, 1, 1.0, 0.5, 0.5);
    const auto masks = ts::compute_spatial_dependencies(plan, plan.cutoff + plan.skin);
    REQUIRE(masks.size() == 1);
    REQUIRE(masks[0] == 0);
  }

  SECTION("3×1×1 — chain: ends have 1, middle has 2") {
    const auto plan = make_plan(3, 1, 1, 1.0, 0.5, 0.5);
    const auto masks = ts::compute_spatial_dependencies(plan, plan.cutoff + plan.skin);
    REQUIRE(masks.size() == 3);
    REQUIRE(popcount_u64(masks[0]) == 1);  // only (1,0,0)
    REQUIRE(popcount_u64(masks[1]) == 2);  // (0,0,0) and (2,0,0)
    REQUIRE(popcount_u64(masks[2]) == 1);  // only (1,0,0)
    // Symmetry.
    REQUIRE((masks[0] & (1ULL << 1)) != 0);
    REQUIRE((masks[1] & (1ULL << 0)) != 0);
    REQUIRE((masks[1] & (1ULL << 2)) != 0);
    REQUIRE((masks[2] & (1ULL << 1)) != 0);
  }

  SECTION("2×2×1 — each zone has 2 face neighbours (1 in x + 1 in y)") {
    const auto plan = make_plan(2, 2, 1, 1.0, 0.5, 0.5);
    const auto masks = ts::compute_spatial_dependencies(plan, plan.cutoff + plan.skin);
    REQUIRE(masks.size() == 4);
    for (const auto m : masks) {
      REQUIRE(popcount_u64(m) == 2);
    }
    // Symmetry: (0) ↔ (1), (0) ↔ (2), (1) ↔ (3), (2) ↔ (3) — plus the
    // diagonals should be excluded.
    REQUIRE((masks[0] & (1ULL << 1)) != 0);
    REQUIRE((masks[0] & (1ULL << 2)) != 0);
    REQUIRE((masks[0] & (1ULL << 3)) == 0);  // diagonal excluded
    REQUIRE((masks[3] & (1ULL << 0)) == 0);
  }

  SECTION("2×2×2 — each zone has 3 face-adjacent neighbours") {
    const auto plan = make_plan(2, 2, 2, 1.0, 0.5, 0.5);
    const auto masks = ts::compute_spatial_dependencies(plan, plan.cutoff + plan.skin);
    REQUIRE(masks.size() == 8);
    for (const auto m : masks) {
      REQUIRE(popcount_u64(m) == 3);
    }
    // Symmetric across all pairs.
    for (std::size_t a = 0; a < 8; ++a) {
      for (std::size_t b = 0; b < 8; ++b) {
        const bool ab = (masks[a] & (1ULL << b)) != 0;
        const bool ba = (masks[b] & (1ULL << a)) != 0;
        REQUIRE(ab == ba);
      }
    }
  }
}

TEST_CASE("Spatial DAG — unravel matches row-major convention", "[scheduler][core][dag]") {
  const std::array<std::uint32_t, 3> n{3, 2, 4};
  // Reverse mapping (x + nx·y + nx·ny·z) must roundtrip.
  for (std::uint32_t id = 0; id < 24; ++id) {
    const auto [x, y, z] = ts::unravel_zone_index(id, n);
    const std::uint32_t rebuilt = x + n[0] * y + n[0] * n[1] * z;
    REQUIRE(rebuilt == id);
  }
}

TEST_CASE("CausalWavefrontScheduler — spatial_dep_mask exposes DAG", "[scheduler][core][dag]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(3, 1, 1, 1.0, 0.5, 0.5);
  sched.initialize(plan);

  REQUIRE(sched.spatial_dep_mask(0) == (1ULL << 1));
  REQUIRE(sched.spatial_dep_mask(1) == ((1ULL << 0) | (1ULL << 2)));
  REQUIRE(sched.spatial_dep_mask(2) == (1ULL << 1));

  REQUIRE_THROWS_AS(sched.spatial_dep_mask(3), std::out_of_range);
}

TEST_CASE("CausalWavefrontScheduler — refresh_certificates creates N certs in order",
          "[scheduler][core][refresh]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(2, 2, 1, 1.0, 0.5, 0.5);  // 4 zones
  sched.initialize(plan);

  REQUIRE(sched.cert_store().size() == 0);
  REQUIRE(sched.cert_store().last_cert_id() == 0);

  sched.refresh_certificates();
  REQUIRE(sched.cert_store().size() == 4);
  // cert_id monotonic from 1, one per zone in canonical_order (identity here).
  REQUIRE(sched.cert_store().last_cert_id() == 4);

  // Each zone has exactly one cert at time_level 1 (zone starts at 0).
  for (ts::ZoneId z = 0; z < 4; ++z) {
    const auto* cert = sched.cert_store().get(z, /*time_level=*/1);
    REQUIRE(cert != nullptr);
    REQUIRE(cert->zone_id == z);
    REQUIRE(cert->time_level == 1);
  }

  // Second refresh replaces (not duplicates). Size unchanged, cert_id advances.
  sched.refresh_certificates();
  REQUIRE(sched.cert_store().size() == 4);
  REQUIRE(sched.cert_store().last_cert_id() == 8);
}

TEST_CASE("CausalWavefrontScheduler — refresh_certificates honours canonical_order",
          "[scheduler][core][refresh][canonical]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  auto plan = make_plan(2, 2, 1, 1.0, 0.5, 0.5);
  plan.canonical_order = {2, 0, 3, 1};
  sched.initialize(plan);

  sched.refresh_certificates();
  // cert_id 1 went to zone 2 (first in canonical_order), 2 to zone 0, etc.
  const auto* c_first = sched.cert_store().get(2, 1);
  const auto* c_second = sched.cert_store().get(0, 1);
  const auto* c_third = sched.cert_store().get(3, 1);
  const auto* c_fourth = sched.cert_store().get(1, 1);
  REQUIRE(c_first);
  REQUIRE(c_second);
  REQUIRE(c_third);
  REQUIRE(c_fourth);
  REQUIRE(c_first->cert_id == 1);
  REQUIRE(c_second->cert_id == 2);
  REQUIRE(c_third->cert_id == 3);
  REQUIRE(c_fourth->cert_id == 4);
}

namespace {

// Stub provider that fills physics inputs with predictable values so we
// can assert they ride through refresh.
struct StubSource : ts::CertificateInputSource {
  mutable std::size_t calls = 0;
  void fill_inputs(ts::ZoneId zone,
                   ts::TimeLevel time_level,
                   ts::CertificateInputs& out) const override {
    ++calls;
    out.zone_id = zone;
    out.time_level = time_level;
    out.v_max_zone = 0.1;
    out.a_max_zone = 0.2;
    out.dt_candidate = 0.001;
    out.buffer_width = 1.0;
    out.skin_remaining = 1.0;
    out.frontier_margin = 1.0;
    out.neighbor_valid_until_step = time_level + 10;
    out.halo_valid_until_step = time_level + 100;
  }
};

}  // namespace

TEST_CASE("CausalWavefrontScheduler — cert input source is consulted",
          "[scheduler][core][refresh][source]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(3, 1, 1, 1.0, 0.5, 0.5);
  sched.initialize(plan);

  StubSource src;
  sched.set_certificate_input_source(&src);

  sched.refresh_certificates();
  REQUIRE(src.calls == 3);

  const auto* c = sched.cert_store().get(0, 1);
  REQUIRE(c != nullptr);
  REQUIRE(c->v_max_zone == 0.1);
  REQUIRE(c->a_max_zone == 0.2);
  REQUIRE(c->dt_candidate == 0.001);
  REQUIRE(c->safe);  // with v=0.1, a=0.2, dt=0.001 → δ≈1e-4 < 1.0 → safe
  REQUIRE(c->neighbor_valid_until_step == 11);
  REQUIRE(c->halo_valid_until_step == 101);
}

TEST_CASE("CausalWavefrontScheduler — invalidate_certificates_for",
          "[scheduler][core][invalidate]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(3, 1, 1, 1.0, 0.5, 0.5);
  sched.initialize(plan);
  sched.refresh_certificates();

  REQUIRE(sched.cert_store().size() == 3);
  sched.invalidate_certificates_for(1);
  REQUIRE(sched.cert_store().size() == 2);
  REQUIRE(sched.cert_store().get(1, 1) == nullptr);
  REQUIRE(sched.cert_store().get(0, 1) != nullptr);
  REQUIRE(sched.cert_store().get(2, 1) != nullptr);
}

TEST_CASE("CausalWavefrontScheduler — invalidate_all_certificates",
          "[scheduler][core][invalidate]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(2, 1, 1, 1.0, 0.5, 0.5);
  sched.initialize(plan);
  sched.refresh_certificates();
  REQUIRE(sched.cert_store().size() == 2);
  sched.invalidate_all_certificates("test");
  REQUIRE(sched.cert_store().size() == 0);
}

TEST_CASE("CausalWavefrontScheduler — on_zone_data_arrived transitions Empty→ResidentPrev",
          "[scheduler][core][events]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(2, 1, 1, 1.0, 0.5, 0.5);
  sched.initialize(plan);

  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::Empty);
  REQUIRE(sched.zone_meta(0).time_level == 0);

  sched.on_zone_data_arrived(0, /*step=*/5, /*version=*/2);
  REQUIRE(sched.zone_meta(0).state == ts::ZoneState::ResidentPrev);
  REQUIRE(sched.zone_meta(0).time_level == 5);
  REQUIRE(sched.zone_meta(0).version == 2);

  // Second call rejects (already ResidentPrev) — state_machine throws,
  // time_level/version remain unchanged.
  REQUIRE_THROWS_AS(sched.on_zone_data_arrived(0, 99, 99), ts::StateMachineError);
  REQUIRE(sched.zone_meta(0).time_level == 5);
  REQUIRE(sched.zone_meta(0).version == 2);
}

TEST_CASE("CausalWavefrontScheduler — on_neighbor_rebuild_completed invalidates affected",
          "[scheduler][core][events]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(3, 1, 1, 1.0, 0.5, 0.5);
  sched.initialize(plan);
  sched.refresh_certificates();
  REQUIRE(sched.cert_store().size() == 3);

  sched.on_neighbor_rebuild_completed({0, 2});
  REQUIRE(sched.cert_store().get(0, 1) == nullptr);
  REQUIRE(sched.cert_store().get(1, 1) != nullptr);
  REQUIRE(sched.cert_store().get(2, 1) == nullptr);
}

TEST_CASE("CausalWavefrontScheduler — introspection", "[scheduler][core][introspection]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};

  // Uninitialized: finished true, frontiers zero, pipeline zero.
  REQUIRE(sched.finished());
  REQUIRE(sched.local_frontier_min() == 0);
  REQUIRE(sched.local_frontier_max() == 0);
  REQUIRE(sched.current_pipeline_depth() == 0);

  auto plan = make_plan(2, 2, 1, 1.0, 0.5, 0.5);
  plan.n_min_per_rank = 2;
  plan.optimal_rank_count = 2;
  sched.initialize(plan);

  REQUIRE(sched.min_zones_per_rank() == 2);
  REQUIRE(sched.optimal_rank_count(4) == 2);  // matches plan: cached value
  REQUIRE(sched.optimal_rank_count(8) == 4);  // fallback: 8 / 2
  REQUIRE(sched.optimal_rank_count(1) == 1);  // floor would be 0, clamped to 1

  REQUIRE(sched.finished());  // target is 0, all zones at 0
  sched.set_target_time_level(3);
  REQUIRE_FALSE(sched.finished());
  REQUIRE(sched.target_time_level() == 3);
}

TEST_CASE("CausalWavefrontScheduler — select_ready_tasks / check_deadlock stubs (T4.6+T4.8)",
          "[scheduler][core][stubs]") {
  ts::CausalWavefrontScheduler sched{ts::PolicyFactory::for_reference()};
  const auto plan = make_plan(2, 1, 1, 1.0, 0.5, 0.5);
  sched.initialize(plan);

  // T4.5 select returns empty.
  REQUIRE(sched.select_ready_tasks().empty());
  // T4.5 check_deadlock is a no-op that never throws.
  REQUIRE_NOTHROW(sched.check_deadlock(std::chrono::milliseconds{1}));
}
