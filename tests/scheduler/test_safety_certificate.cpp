// Exec pack: docs/development/m4_execution_pack.md T4.3
// SPEC: docs/specs/scheduler/SPEC.md §4
// Master spec: §6.4
//
// Unit tests for SafetyCertificate math + CertificateStore. I7 monotonicity
// is tested here manually over 100+ hand-picked near-threshold cases. The
// randomised property fuzzer lives in fuzz_cert_monotonicity.cpp (≥250k).

#include "tdmd/scheduler/certificate_store.hpp"
#include "tdmd/scheduler/safety_certificate.hpp"
#include "tdmd/scheduler/types.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>

namespace ts = tdmd::scheduler;

namespace {

// Canonical inputs — all values safely under thresholds. Used as a clean
// baseline from which tests nudge one field at a time.
constexpr double kVClean = 5.0;         // Å / ps
constexpr double kAClean = 2.0;         // Å / ps²
constexpr double kDtClean = 0.001;      // ps
constexpr double kBufferClean = 0.5;    // Å
constexpr double kSkinClean = 0.5;      // Å
constexpr double kFrontierClean = 1.0;  // Å

ts::CertificateInputs clean_inputs() {
  ts::CertificateInputs in;
  in.zone_id = 7;
  in.time_level = 42;
  in.version = 3;
  in.v_max_zone = kVClean;
  in.a_max_zone = kAClean;
  in.dt_candidate = kDtClean;
  in.buffer_width = kBufferClean;
  in.skin_remaining = kSkinClean;
  in.frontier_margin = kFrontierClean;
  in.neighbor_valid_until_step = 100;
  in.halo_valid_until_step = 100;
  in.mode_policy_tag = 0xAABBCCDDEEFF1122ULL;
  return in;
}

}  // namespace

// ----------------------------------------------------------------------------
// displacement_bound: pure math
// ----------------------------------------------------------------------------

TEST_CASE("compute_displacement_bound — canonical formula", "[scheduler][cert][math]") {
  // δ(dt) = v·dt + 0.5·a·dt². For v=10, a=20, dt=0.1 →
  //   10·0.1 + 0.5·20·0.01 = 1.0 + 0.1 = 1.1
  const double d = ts::compute_displacement_bound(10.0, 20.0, 0.1);
  REQUIRE_THAT(d, Catch::Matchers::WithinAbs(1.1, 1e-12));
}

TEST_CASE("compute_displacement_bound — dt=0 yields zero", "[scheduler][cert][math]") {
  REQUIRE(ts::compute_displacement_bound(100.0, 100.0, 0.0) == 0.0);
}

TEST_CASE("compute_displacement_bound — v=a=0 yields zero", "[scheduler][cert][math]") {
  REQUIRE(ts::compute_displacement_bound(0.0, 0.0, 0.5) == 0.0);
}

TEST_CASE("compute_displacement_bound — purely linear regime", "[scheduler][cert][math]") {
  // a=0 → δ = v·dt (to FP precision; 0.1 isn't exact in binary).
  REQUIRE_THAT(ts::compute_displacement_bound(7.0, 0.0, 0.1),
               Catch::Matchers::WithinAbs(0.7, 1e-12));
}

TEST_CASE("compute_displacement_bound — purely quadratic regime", "[scheduler][cert][math]") {
  // v=0 → δ = 0.5·a·dt²
  REQUIRE_THAT(ts::compute_displacement_bound(0.0, 8.0, 0.5),
               Catch::Matchers::WithinAbs(1.0, 1e-12));
}

TEST_CASE("compute_displacement_bound — NaN in any arg → NaN", "[scheduler][cert][math]") {
  const double n = std::numeric_limits<double>::quiet_NaN();
  REQUIRE(std::isnan(ts::compute_displacement_bound(n, 1.0, 0.1)));
  REQUIRE(std::isnan(ts::compute_displacement_bound(1.0, n, 0.1)));
  REQUIRE(std::isnan(ts::compute_displacement_bound(1.0, 1.0, n)));
}

// ----------------------------------------------------------------------------
// compute_safe: predicate + defensive policy
// ----------------------------------------------------------------------------

TEST_CASE("compute_safe — clean inputs are safe", "[scheduler][cert][safe]") {
  REQUIRE(ts::compute_safe(kVClean, kAClean, kDtClean, kBufferClean, kSkinClean, kFrontierClean));
}

TEST_CASE("compute_safe — δ just below min threshold is safe", "[scheduler][cert][safe]") {
  // Construct δ ≈ 0.4999; min(buffer,skin,frontier)=0.5. Should be safe.
  // v=499.9, a=0, dt=0.001 → δ = 0.4999
  REQUIRE(ts::compute_safe(499.9, 0.0, 0.001, 0.5, 1.0, 1.0));
}

TEST_CASE("compute_safe — δ equal to min threshold is NOT safe (strict <)",
          "[scheduler][cert][safe]") {
  // δ must be strictly less than min. δ = 0.5 equals threshold → unsafe.
  // v=500, a=0, dt=0.001 → δ = 0.5
  REQUIRE_FALSE(ts::compute_safe(500.0, 0.0, 0.001, 0.5, 1.0, 1.0));
}

TEST_CASE("compute_safe — buffer clamps min even with room elsewhere", "[scheduler][cert][safe]") {
  // buffer is the tightest of the three → that's what governs.
  REQUIRE_FALSE(ts::compute_safe(100.0, 0.0, 0.01, 0.5, 10.0, 10.0));
  REQUIRE(ts::compute_safe(40.0, 0.0, 0.01, 0.5, 10.0, 10.0));
}

TEST_CASE("compute_safe — skin clamps min when tightest", "[scheduler][cert][safe]") {
  REQUIRE_FALSE(ts::compute_safe(100.0, 0.0, 0.01, 10.0, 0.5, 10.0));
  REQUIRE(ts::compute_safe(40.0, 0.0, 0.01, 10.0, 0.5, 10.0));
}

TEST_CASE("compute_safe — frontier clamps min when tightest", "[scheduler][cert][safe]") {
  REQUIRE_FALSE(ts::compute_safe(100.0, 0.0, 0.01, 10.0, 10.0, 0.5));
  REQUIRE(ts::compute_safe(40.0, 0.0, 0.01, 10.0, 10.0, 0.5));
}

TEST_CASE("compute_safe — zero buffer makes it unsafe", "[scheduler][cert][safe][boundary]") {
  // threshold min = 0 ⇒ δ < 0 impossible for finite non-negative v,a,dt.
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.001, 0.0, 10.0, 10.0));
}

TEST_CASE("compute_safe — zero skin makes it unsafe", "[scheduler][cert][safe][boundary]") {
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.001, 10.0, 0.0, 10.0));
}

TEST_CASE("compute_safe — zero frontier makes it unsafe", "[scheduler][cert][safe][boundary]") {
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.001, 10.0, 10.0, 0.0));
}

TEST_CASE("compute_safe — dt=0 is degenerate → unsafe", "[scheduler][cert][safe][boundary]") {
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.0, 1.0, 1.0, 1.0));
}

TEST_CASE("compute_safe — negative v → unsafe (defensive)", "[scheduler][cert][safe][defensive]") {
  REQUIRE_FALSE(ts::compute_safe(-1.0, 1.0, 0.001, 1.0, 1.0, 1.0));
}

TEST_CASE("compute_safe — negative a → unsafe (defensive)", "[scheduler][cert][safe][defensive]") {
  REQUIRE_FALSE(ts::compute_safe(1.0, -1.0, 0.001, 1.0, 1.0, 1.0));
}

TEST_CASE("compute_safe — negative dt → unsafe (defensive)", "[scheduler][cert][safe][defensive]") {
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, -0.001, 1.0, 1.0, 1.0));
}

TEST_CASE("compute_safe — negative threshold field → unsafe (defensive)",
          "[scheduler][cert][safe][defensive]") {
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.001, -1.0, 1.0, 1.0));
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.001, 1.0, -1.0, 1.0));
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.001, 1.0, 1.0, -1.0));
}

TEST_CASE("compute_safe — NaN anywhere → unsafe", "[scheduler][cert][safe][defensive]") {
  const double n = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_FALSE(ts::compute_safe(n, 1.0, 0.001, 1.0, 1.0, 1.0));
  REQUIRE_FALSE(ts::compute_safe(1.0, n, 0.001, 1.0, 1.0, 1.0));
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, n, 1.0, 1.0, 1.0));
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.001, n, 1.0, 1.0));
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.001, 1.0, n, 1.0));
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.001, 1.0, 1.0, n));
}

TEST_CASE("compute_safe — +Inf anywhere → unsafe", "[scheduler][cert][safe][defensive]") {
  const double inf = std::numeric_limits<double>::infinity();
  REQUIRE_FALSE(ts::compute_safe(inf, 1.0, 0.001, 1.0, 1.0, 1.0));
  REQUIRE_FALSE(ts::compute_safe(1.0, inf, 0.001, 1.0, 1.0, 1.0));
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, inf, 1.0, 1.0, 1.0));
  REQUIRE_FALSE(ts::compute_safe(1.0, 1.0, 0.001, inf, 1.0, 1.0));
}

// ----------------------------------------------------------------------------
// build_certificate: population + safe/displacement are precomputed
// ----------------------------------------------------------------------------

TEST_CASE("build_certificate — every field copied", "[scheduler][cert][build]") {
  const auto in = clean_inputs();
  const auto c = ts::build_certificate(0xDEADBEEF, in);

  REQUIRE(c.safe);
  REQUIRE(c.cert_id == 0xDEADBEEF);
  REQUIRE(c.zone_id == in.zone_id);
  REQUIRE(c.time_level == in.time_level);
  REQUIRE(c.version == in.version);
  REQUIRE(c.v_max_zone == in.v_max_zone);
  REQUIRE(c.a_max_zone == in.a_max_zone);
  REQUIRE(c.dt_candidate == in.dt_candidate);
  REQUIRE(c.buffer_width == in.buffer_width);
  REQUIRE(c.skin_remaining == in.skin_remaining);
  REQUIRE(c.frontier_margin == in.frontier_margin);
  REQUIRE(c.neighbor_valid_until_step == in.neighbor_valid_until_step);
  REQUIRE(c.halo_valid_until_step == in.halo_valid_until_step);
  REQUIRE(c.mode_policy_tag == in.mode_policy_tag);

  const double expected_delta =
      in.v_max_zone * in.dt_candidate + 0.5 * in.a_max_zone * in.dt_candidate * in.dt_candidate;
  REQUIRE(c.displacement_bound == expected_delta);
}

TEST_CASE("build_certificate — unsafe inputs produce safe=false cert", "[scheduler][cert][build]") {
  auto in = clean_inputs();
  in.buffer_width = 0.0;  // guarantees unsafe
  const auto c = ts::build_certificate(1, in);
  REQUIRE_FALSE(c.safe);
  REQUIRE(c.cert_id == 1);
  REQUIRE(c.buffer_width == 0.0);
}

// ----------------------------------------------------------------------------
// I7 monotonicity — 100 manual near-threshold cases
// ----------------------------------------------------------------------------
//
// Property: safe(C[dt₂]) ∧ dt₁ < dt₂  ⟹  safe(C[dt₁]).
// Symbolic proof: δ is monotone non-decreasing in dt for v,a,dt ≥ 0, so
//   safe(C[dt₂]) ⇒ δ(dt₂) < T ⇒ δ(dt₁) ≤ δ(dt₂) < T ⇒ safe(C[dt₁]).
// These hand-picked cases exercise every regime (linear-dominant, quadratic-
// dominant, threshold edges) before the random fuzzer.

namespace {

// Asserts I7 for a specific (v, a, buffer, skin, frontier) triple over a
// grid of dt values. Checks: for every pair (dt_i, dt_j) with dt_i < dt_j,
// if safe at dt_j then safe at dt_i.
void assert_i7_over_dt_grid(double v, double a, double buf, double skin, double front) {
  constexpr double kDts[] = {0.0001,
                             0.00025,
                             0.0005,
                             0.00075,
                             0.001,
                             0.0025,
                             0.005,
                             0.0075,
                             0.01,
                             0.025,
                             0.05,
                             0.075,
                             0.1};
  constexpr std::size_t kN = sizeof(kDts) / sizeof(kDts[0]);
  for (std::size_t j = 0; j < kN; ++j) {
    const bool safe_j = ts::compute_safe(v, a, kDts[j], buf, skin, front);
    if (!safe_j)
      continue;
    for (std::size_t i = 0; i < j; ++i) {
      const bool safe_i = ts::compute_safe(v, a, kDts[i], buf, skin, front);
      REQUIRE(safe_i);  // I7
    }
  }
}

}  // namespace

TEST_CASE("I7 monotonicity — linear-dominant regime, buffer-bound",
          "[scheduler][cert][I7][manual]") {
  // v=50, a=0: δ=50·dt, grows linearly. buffer=0.05 → dt < 0.001 safe.
  assert_i7_over_dt_grid(50.0, 0.0, 0.05, 1.0, 1.0);
  assert_i7_over_dt_grid(50.0, 0.0, 0.25, 1.0, 1.0);
  assert_i7_over_dt_grid(100.0, 0.0, 1.0, 1.0, 1.0);
}

TEST_CASE("I7 monotonicity — quadratic-dominant regime", "[scheduler][cert][I7][manual]") {
  // v=0, a=1000: δ=500·dt². quadratic. buffer=0.05 → dt < ~0.01 safe.
  assert_i7_over_dt_grid(0.0, 1000.0, 0.05, 1.0, 1.0);
  assert_i7_over_dt_grid(0.0, 1000.0, 1.0, 1.0, 1.0);
  assert_i7_over_dt_grid(0.0, 2000.0, 0.5, 1.0, 1.0);
}

TEST_CASE("I7 monotonicity — mixed regime, skin-bound", "[scheduler][cert][I7][manual]") {
  // Mixed v and a, skin is the tightest. Typical realistic metal-unit range.
  assert_i7_over_dt_grid(10.0, 100.0, 10.0, 0.1, 10.0);
  assert_i7_over_dt_grid(5.0, 50.0, 10.0, 0.05, 10.0);
}

TEST_CASE("I7 monotonicity — mixed regime, frontier-bound", "[scheduler][cert][I7][manual]") {
  assert_i7_over_dt_grid(10.0, 100.0, 10.0, 10.0, 0.1);
  assert_i7_over_dt_grid(5.0, 50.0, 10.0, 10.0, 0.05);
}

TEST_CASE("I7 monotonicity — all three thresholds equal", "[scheduler][cert][I7][manual]") {
  assert_i7_over_dt_grid(10.0, 100.0, 0.5, 0.5, 0.5);
  assert_i7_over_dt_grid(1.0, 10.0, 0.01, 0.01, 0.01);
}

TEST_CASE("I7 monotonicity — zero velocity (pure quadratic)", "[scheduler][cert][I7][manual]") {
  assert_i7_over_dt_grid(0.0, 10.0, 0.5, 0.5, 0.5);
  assert_i7_over_dt_grid(0.0, 100.0, 0.5, 0.5, 0.5);
}

TEST_CASE("I7 monotonicity — zero acceleration (pure linear)", "[scheduler][cert][I7][manual]") {
  assert_i7_over_dt_grid(10.0, 0.0, 0.5, 0.5, 0.5);
  assert_i7_over_dt_grid(100.0, 0.0, 0.5, 0.5, 0.5);
}

TEST_CASE("I7 monotonicity — small values near denormals", "[scheduler][cert][I7][manual]") {
  assert_i7_over_dt_grid(1e-10, 1e-10, 1e-8, 1e-8, 1e-8);
  assert_i7_over_dt_grid(1e-15, 1e-15, 1e-12, 1e-12, 1e-12);
}

// ----------------------------------------------------------------------------
// CertificateStore: CRUD + invalidate semantics
// ----------------------------------------------------------------------------

TEST_CASE("CertificateStore — empty at construction", "[scheduler][store]") {
  ts::CertificateStore s{0xABCDEF};
  REQUIRE(s.size() == 0);
  REQUIRE(s.last_cert_id() == 0);
  REQUIRE(s.mode_policy_tag() == 0xABCDEF);
  REQUIRE(s.get(0, 0) == nullptr);
}

TEST_CASE("CertificateStore — build allocates monotonic cert_id", "[scheduler][store][build]") {
  ts::CertificateStore s{0xAABB};
  auto in = clean_inputs();
  const auto c1 = s.build(in);
  in.time_level = 43;
  const auto c2 = s.build(in);
  in.zone_id = 8;
  in.time_level = 42;
  const auto c3 = s.build(in);

  REQUIRE(c1.cert_id == 1);
  REQUIRE(c2.cert_id == 2);
  REQUIRE(c3.cert_id == 3);
  REQUIRE(s.size() == 3);
  REQUIRE(s.last_cert_id() == 3);
}

TEST_CASE("CertificateStore — build stamps mode_policy_tag unconditionally",
          "[scheduler][store][build]") {
  ts::CertificateStore s{0xCAFEBABE};
  auto in = clean_inputs();
  in.mode_policy_tag = 0x1234;  // caller's value — must be overridden
  const auto c = s.build(in);
  REQUIRE(c.mode_policy_tag == 0xCAFEBABE);

  const auto* stored = s.get(in.zone_id, in.time_level);
  REQUIRE(stored != nullptr);
  REQUIRE(stored->mode_policy_tag == 0xCAFEBABE);
}

TEST_CASE("CertificateStore — get returns nullptr when missing", "[scheduler][store][get]") {
  ts::CertificateStore s{0};
  REQUIRE(s.get(5, 5) == nullptr);
  auto in = clean_inputs();
  s.build(in);
  REQUIRE(s.get(in.zone_id, in.time_level) != nullptr);
  REQUIRE(s.get(in.zone_id, in.time_level + 1) == nullptr);
  REQUIRE(s.get(in.zone_id + 1, in.time_level) == nullptr);
}

TEST_CASE("CertificateStore — rebuild replaces existing cert", "[scheduler][store][build]") {
  ts::CertificateStore s{0};
  auto in = clean_inputs();
  const auto c1 = s.build(in);
  in.version = 99;
  const auto c2 = s.build(in);

  REQUIRE(c1.cert_id == 1);
  REQUIRE(c2.cert_id == 2);
  REQUIRE(s.size() == 1);  // same key → replaced, not appended
  const auto* stored = s.get(in.zone_id, in.time_level);
  REQUIRE(stored != nullptr);
  REQUIRE(stored->cert_id == 2);
  REQUIRE(stored->version == 99);
}

TEST_CASE("CertificateStore — invalidate_for removes all time_levels for zone",
          "[scheduler][store][invalidate]") {
  ts::CertificateStore s{0};
  auto in = clean_inputs();
  for (ts::TimeLevel t = 0; t < 5; ++t) {
    in.zone_id = 7;
    in.time_level = t;
    s.build(in);
  }
  for (ts::TimeLevel t = 0; t < 3; ++t) {
    in.zone_id = 8;
    in.time_level = t;
    s.build(in);
  }
  REQUIRE(s.size() == 8);

  const std::size_t removed = s.invalidate_for(7);
  REQUIRE(removed == 5);
  REQUIRE(s.size() == 3);
  REQUIRE(s.get(7, 0) == nullptr);
  REQUIRE(s.get(7, 4) == nullptr);
  REQUIRE(s.get(8, 0) != nullptr);
  REQUIRE(s.get(8, 2) != nullptr);
}

TEST_CASE("CertificateStore — invalidate_for on missing zone returns 0",
          "[scheduler][store][invalidate]") {
  ts::CertificateStore s{0};
  auto in = clean_inputs();
  s.build(in);
  REQUIRE(s.invalidate_for(999) == 0);
  REQUIRE(s.size() == 1);
}

TEST_CASE("CertificateStore — invalidate_all wipes everything", "[scheduler][store][invalidate]") {
  ts::CertificateStore s{0};
  auto in = clean_inputs();
  for (ts::TimeLevel t = 0; t < 10; ++t) {
    in.time_level = t;
    s.build(in);
  }
  REQUIRE(s.size() == 10);
  const std::size_t removed = s.invalidate_all(std::string_view{"neighbor_rebuild"});
  REQUIRE(removed == 10);
  REQUIRE(s.size() == 0);
  REQUIRE(s.get(in.zone_id, 0) == nullptr);
  // Counter NOT reset — cert_ids stay monotone across invalidation.
  REQUIRE(s.last_cert_id() == 10);
  in.time_level = 0;
  const auto c = s.build(in);
  REQUIRE(c.cert_id == 11);
}

TEST_CASE("CertificateStore — invalidate_all on empty store returns 0",
          "[scheduler][store][invalidate]") {
  ts::CertificateStore s{0};
  REQUIRE(s.invalidate_all(std::string_view{"noop"}) == 0);
}
