// Exec pack: docs/development/m4_execution_pack.md T4.3, D-M4-8
// SPEC: docs/specs/scheduler/SPEC.md §12.2 (property fuzzer hard gate)
// Master spec: §6.4 (I7 monotonicity statement)
//
// Property fuzzer for I7:
//   ∀ (v, a, buffer, skin, frontier), ∀ dt₁ < dt₂:
//     safe(C[dt₂])  ⟹  safe(C[dt₁]).
//
// Corpus strategy:
// — 80% normal range:  v ∈ [0, 100] Å/ps, a ∈ [0, 100] Å/ps²,
//                       dt ∈ [1e-6, 1e-1] ps, thresholds ∈ [1e-3, 5] Å.
//                       Log-uniform on dt and thresholds; ensures we hit
//                       the decision boundary from both sides.
// — 10% edge cases:    each numeric input independently replaced with 0.0
//                       with probability 0.1.
// — 5% denormal range:  |x| ∈ [1e-310, 1e-290] for a randomly-chosen field.
// — 5% pathological:    NaN / +Inf injected into one randomly-chosen field
//                       (v/a/dt/buffer/skin/frontier).
//
// In the pathological branch safe(C[·]) is always false by defense
// (compute_safe guards); I7 is vacuously satisfied. We still assert that.
//
// Seed: 0x4D345F53434845ULL ("M4_SCHE"), per D-M4-8.
// Case count: ≥250 000 per D-M4-8 + exec pack T4.3 acceptance criterion.

#include "tdmd/scheduler/safety_certificate.hpp"

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>

namespace ts = tdmd::scheduler;

namespace {

constexpr std::uint64_t kFuzzSeed = 0x4D345F53434845ULL;  // "M4_SCHE"
constexpr std::size_t kFuzzCases = 250'000;

// Log-uniform sampler on [lo, hi] (both > 0).
double log_uniform(std::mt19937_64& rng, double lo, double hi) {
  std::uniform_real_distribution<double> dist{std::log(lo), std::log(hi)};
  return std::exp(dist(rng));
}

struct Sample {
  double v = 0.0;
  double a = 0.0;
  double dt1 = 0.0;
  double dt2 = 0.0;
  double buffer = 0.0;
  double skin = 0.0;
  double frontier = 0.0;
  bool pathological = false;  // expect safe=false unconditionally
};

Sample draw(std::mt19937_64& rng) {
  Sample s;

  std::uniform_real_distribution<double> u01{0.0, 1.0};
  const double roll = u01(rng);

  if (roll < 0.05) {
    // Pathological: NaN/Inf in a random field.
    s.v = log_uniform(rng, 1e-3, 100.0);
    s.a = log_uniform(rng, 1e-3, 100.0);
    const double dta = log_uniform(rng, 1e-6, 0.1);
    const double dtb = log_uniform(rng, 1e-6, 0.1);
    s.dt1 = std::min(dta, dtb);
    s.dt2 = std::max(dta, dtb);
    s.buffer = log_uniform(rng, 1e-3, 5.0);
    s.skin = log_uniform(rng, 1e-3, 5.0);
    s.frontier = log_uniform(rng, 1e-3, 5.0);

    const double poison = (u01(rng) < 0.5) ? std::numeric_limits<double>::quiet_NaN()
                                           : std::numeric_limits<double>::infinity();
    // Poison a field shared by both compute_safe(dt1) and compute_safe(dt2),
    // so the invariant "both calls collapse to false" holds.
    switch (std::uniform_int_distribution<int>{0, 4}(rng)) {
      case 0:
        s.v = poison;
        break;
      case 1:
        s.a = poison;
        break;
      case 2:
        s.buffer = poison;
        break;
      case 3:
        s.skin = poison;
        break;
      default:
        s.frontier = poison;
        break;
    }
    s.pathological = true;
    return s;
  }

  if (roll < 0.10) {
    // Denormal injection: finite but very small magnitude in one field.
    s.v = log_uniform(rng, 1e-3, 100.0);
    s.a = log_uniform(rng, 1e-3, 100.0);
    const double dta = log_uniform(rng, 1e-6, 0.1);
    const double dtb = log_uniform(rng, 1e-6, 0.1);
    s.dt1 = std::min(dta, dtb);
    s.dt2 = std::max(dta, dtb);
    s.buffer = log_uniform(rng, 1e-3, 5.0);
    s.skin = log_uniform(rng, 1e-3, 5.0);
    s.frontier = log_uniform(rng, 1e-3, 5.0);

    // Sample denormal from [1e-310, 1e-290] — well inside subnormal range.
    std::uniform_real_distribution<double> denormal_exp{-310.0, -290.0};
    const double denorm = std::pow(10.0, denormal_exp(rng));
    switch (std::uniform_int_distribution<int>{0, 5}(rng)) {
      case 0:
        s.v = denorm;
        break;
      case 1:
        s.a = denorm;
        break;
      case 2:
        s.dt1 = denorm;
        s.dt2 = std::max(s.dt2, 2.0 * denorm);
        break;
      case 3:
        s.buffer = denorm;
        break;
      case 4:
        s.skin = denorm;
        break;
      default:
        s.frontier = denorm;
        break;
    }
    // Denormals → physically legitimate values; I7 must hold.
    s.pathological = false;
    return s;
  }

  if (roll < 0.20) {
    // Zero-injection: one field pinned to 0, rest normal.
    s.v = log_uniform(rng, 1e-3, 100.0);
    s.a = log_uniform(rng, 1e-3, 100.0);
    const double dta = log_uniform(rng, 1e-6, 0.1);
    const double dtb = log_uniform(rng, 1e-6, 0.1);
    s.dt1 = std::min(dta, dtb);
    s.dt2 = std::max(dta, dtb);
    s.buffer = log_uniform(rng, 1e-3, 5.0);
    s.skin = log_uniform(rng, 1e-3, 5.0);
    s.frontier = log_uniform(rng, 1e-3, 5.0);

    switch (std::uniform_int_distribution<int>{0, 5}(rng)) {
      case 0:
        s.v = 0.0;
        break;
      case 1:
        s.a = 0.0;
        break;
      case 2:
        s.dt1 = 0.0;
        break;  // dt1=0 → safe(C[dt1])=false by defense
      case 3:
        s.buffer = 0.0;
        break;
      case 4:
        s.skin = 0.0;
        break;
      default:
        s.frontier = 0.0;
        break;
    }
    s.pathological = false;
    return s;
  }

  // Normal range (80% of the corpus).
  s.v = log_uniform(rng, 1e-3, 100.0);
  s.a = log_uniform(rng, 1e-3, 100.0);
  const double dta = log_uniform(rng, 1e-6, 0.1);
  const double dtb = log_uniform(rng, 1e-6, 0.1);
  s.dt1 = std::min(dta, dtb);
  s.dt2 = std::max(dta, dtb);
  s.buffer = log_uniform(rng, 1e-3, 5.0);
  s.skin = log_uniform(rng, 1e-3, 5.0);
  s.frontier = log_uniform(rng, 1e-3, 5.0);
  s.pathological = false;
  return s;
}

}  // namespace

TEST_CASE("I7 monotonicity fuzzer — ≥250k random cases", "[scheduler][cert][I7][fuzzer]") {
  std::mt19937_64 rng{kFuzzSeed};

  std::size_t safe2_count = 0;
  std::size_t safe2_implies_safe1_count = 0;
  std::size_t pathological_count = 0;

  for (std::size_t i = 0; i < kFuzzCases; ++i) {
    const Sample s = draw(rng);

    const bool safe_dt2 = ts::compute_safe(s.v, s.a, s.dt2, s.buffer, s.skin, s.frontier);
    const bool safe_dt1 = ts::compute_safe(s.v, s.a, s.dt1, s.buffer, s.skin, s.frontier);

    if (s.pathological) {
      // Defensive guards must collapse to false for ANY pathological input.
      REQUIRE_FALSE(safe_dt2);
      REQUIRE_FALSE(safe_dt1);
      ++pathological_count;
      continue;
    }

    // I7: safe(C[dt₂]) ∧ dt₁ < dt₂ ⟹ safe(C[dt₁]).
    // Note: dt₁ can equal 0 (zero-injection branch), in which case
    // safe(C[dt₁]) = false by defense — this does NOT break I7 because I7's
    // premise excludes dt₁ ≤ 0 by the compute_safe defensive policy. What
    // we really assert: if dt₁ > 0 AND safe(C[dt₂]), then safe(C[dt₁]).
    if (safe_dt2) {
      ++safe2_count;
      if (s.dt1 > 0.0) {
        REQUIRE(safe_dt1);
        ++safe2_implies_safe1_count;
      }
    }
  }

  // Sanity: we should have hit meaningful coverage — non-trivial share
  // of cases must land on either branch. Otherwise the fuzzer is broken
  // (e.g. always producing unsafe cases → I7 never tested).
  REQUIRE(safe2_count > 10'000);
  REQUIRE(safe2_implies_safe1_count > 10'000);
  REQUIRE(pathological_count > 5'000);  // ~5% of 250k = 12.5k expected
  REQUIRE(pathological_count < 20'000);
}

namespace {

// Bit-level equality on doubles. NaN is not self-equal under ==, so a
// reproducibility check that draws NaN values via the pathological branch
// cannot use operator==; compare the raw bit patterns instead.
bool bit_equal(double x, double y) noexcept {
  return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y);
}

}  // namespace

TEST_CASE("I7 fuzzer is reproducible — same seed twice → same pass/fail verdict",
          "[scheduler][cert][I7][fuzzer][repro]") {
  // Small smoke that seed discipline works. If mt19937_64 produces different
  // streams for the same seed (it can't — standard mandates it — but the
  // scaffolding around it might), this catches it before the big run.
  std::mt19937_64 a{kFuzzSeed};
  std::mt19937_64 b{kFuzzSeed};
  for (int i = 0; i < 1000; ++i) {
    const Sample sa = draw(a);
    const Sample sb = draw(b);
    REQUIRE(bit_equal(sa.v, sb.v));
    REQUIRE(bit_equal(sa.a, sb.a));
    REQUIRE(bit_equal(sa.dt1, sb.dt1));
    REQUIRE(bit_equal(sa.dt2, sb.dt2));
    REQUIRE(bit_equal(sa.buffer, sb.buffer));
    REQUIRE(bit_equal(sa.skin, sb.skin));
    REQUIRE(bit_equal(sa.frontier, sb.frontier));
    REQUIRE(sa.pathological == sb.pathological);
  }
}
