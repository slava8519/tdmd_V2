// Exec pack: docs/development/m3_execution_pack.md T3.2
// SPEC: docs/specs/zoning/SPEC.md §2.1
//
// T3.2 lands only the types + abstract interface — no scheme implementation
// exists yet, so the tests here are narrow: they check that the types are
// default-constructible and have the shape the SPEC requires. Real
// behavioural tests arrive in T3.3 (Linear1D) and onwards.

#include "tdmd/zoning/planner.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <type_traits>

namespace tz = tdmd::zoning;

TEST_CASE("ZoningPlan — default-constructible and total_zones correct", "[zoning][types]") {
  tz::ZoningPlan plan;
  REQUIRE(plan.scheme == tz::ZoningScheme::Linear1D);
  REQUIRE(plan.n_zones[0] == 0);
  REQUIRE(plan.n_zones[1] == 0);
  REQUIRE(plan.n_zones[2] == 0);
  REQUIRE(plan.n_min_per_rank == 1);
  REQUIRE(plan.optimal_rank_count == 1);
  REQUIRE(plan.canonical_order.empty());
  REQUIRE_FALSE(plan.subdomain_box.has_value());
  REQUIRE(plan.total_zones() == 0);

  plan.n_zones = {2, 3, 4};
  REQUIRE(plan.total_zones() == 24);
}

TEST_CASE("PerformanceHint — default stubbed for M3", "[zoning][types]") {
  tz::PerformanceHint hint;
  // M3 does not populate cost/bandwidth (D-M3-3 / OQ-M3-1): callers may
  // pass a zero-initialised hint; schemes must tolerate that.
  REQUIRE(hint.cost_per_force_evaluation_seconds == 0.0);
  REQUIRE(hint.bandwidth_peer_to_peer_bytes_per_sec == 0.0);
  REQUIRE(hint.atom_record_size_bytes == 32.0);
  REQUIRE(hint.preferred_K_pipeline == 1);
}

TEST_CASE("ZoneId — 32-bit unsigned integer", "[zoning][types]") {
  STATIC_REQUIRE(sizeof(tz::ZoneId) == 4);
  STATIC_REQUIRE(std::is_unsigned_v<tz::ZoneId>);
}

TEST_CASE("ZoningPlanner — abstract contract", "[zoning][types]") {
  // We can't instantiate the abstract class but we can at least confirm
  // it has a virtual destructor (inheriting safely from it).
  STATIC_REQUIRE(std::has_virtual_destructor_v<tz::ZoningPlanner>);
}

TEST_CASE("ZoningPlanError — is a std::runtime_error", "[zoning][types]") {
  STATIC_REQUIRE(std::is_base_of_v<std::runtime_error, tz::ZoningPlanError>);
  REQUIRE_THROWS_AS(throw tz::ZoningPlanError("bad box"), tz::ZoningPlanError);
  REQUIRE_THROWS_AS(throw tz::ZoningPlanError("bad box"), std::runtime_error);
}
