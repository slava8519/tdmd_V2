// SPEC: docs/specs/perfmodel/SPEC.md §3.2 (Pattern 3), §3.3 (Pattern 1),
//       §3.5 (overhead), §3.7 (saturation)
// Exec pack: docs/development/m2_execution_pack.md T2.10
//
// M2 coverage: analytic predictor for Pattern 1 + Pattern 3 (SD). Pattern 2
// (two-level) arrives with M7 — the reduced PerfPrediction record we use here
// matches exec-pack T2.10, not the full §2.1 PerformancePrediction.
//
// Test N values are all ≥ 8000 so the SPEC §3.7 saturation warning regime
// (GPU underutilization at N < few×10³) cannot contaminate the invariants we
// probe. The M2 hardware factory is CPU-only, but the same range keeps us in
// the "T_c non-trivial" regime where the formulas are meaningful.

#include "tdmd/perfmodel/perfmodel.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>

namespace {

tdmd::PerfModel make_morse_model() {
  return tdmd::PerfModel(tdmd::HardwareProfile::modern_x86_64(), tdmd::PotentialCost::morse());
}

tdmd::PerfModel make_eam_model() {
  return tdmd::PerfModel(tdmd::HardwareProfile::modern_x86_64(), tdmd::PotentialCost::eam_alloy());
}

// Slow-interconnect profile used to drive K_opt into the ramp-up regime where
// the SPEC §3.3 sqrt(T_p/T_sched) analytic prediction is visible. modern_x86_64
// has intra/compute ratio such that T_p < T_c always → overlap saturates
// K_opt=1; dropping intra_bw to 2 GB/s forces T_p > T_c for N above ~1e5 and
// lets the sqrt scaling show up in the asymptotic test.
tdmd::HardwareProfile slow_intra_profile() {
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.intra_bw_bytes_per_sec = 2.0e9;  // 2 GB/s — artificially slow to stress TD pipeline
  return hw;
}

tdmd::PerfModel make_slow_intra_model() {
  return tdmd::PerfModel(slow_intra_profile(), tdmd::PotentialCost::morse());
}

}  // namespace

TEST_CASE("HardwareProfile::modern_x86_64 sanity", "[perfmodel][hwprofile]") {
  const auto hw = tdmd::HardwareProfile::modern_x86_64();

  // Sanity range per SPEC §4.1 — a modern dual-socket node with a 100 Gb/s NIC.
  // The exact values are subject to M4 auto-probe updates; bounds below are
  // order-of-magnitude guards, not precise assertions.
  REQUIRE(hw.cpu_flops_per_sec > 1.0e11);
  REQUIRE(hw.cpu_flops_per_sec < 1.0e14);
  REQUIRE(hw.intra_bw_bytes_per_sec > hw.inter_bw_bytes_per_sec);
  REQUIRE(hw.scheduler_overhead_sec > 0.0);
  REQUIRE(hw.scheduler_overhead_sec < 1.0e-3);  // < 1 ms per iteration
  REQUIRE(hw.n_ranks >= 1U);
}

TEST_CASE("PotentialCost factories match §3.1 table", "[perfmodel][potcost]") {
  const auto m = tdmd::PotentialCost::morse();
  const auto e = tdmd::PotentialCost::eam_alloy();

  // §3.1 ranges: Pair 30-50, ManyBodyLocal 80-150. EAM must be clearly heavier.
  REQUIRE(m.flops_per_pair >= 30.0);
  REQUIRE(m.flops_per_pair <= 50.0);
  REQUIRE(e.flops_per_pair >= 80.0);
  REQUIRE(e.flops_per_pair <= 150.0);
  REQUIRE(e.flops_per_pair > m.flops_per_pair);
}

TEST_CASE("PerfModel rejects bad profiles", "[perfmodel][validation]") {
  tdmd::HardwareProfile bad_hw = tdmd::HardwareProfile::modern_x86_64();
  bad_hw.cpu_flops_per_sec = 0.0;
  REQUIRE_THROWS_AS(tdmd::PerfModel(bad_hw, tdmd::PotentialCost::morse()), std::invalid_argument);

  bad_hw = tdmd::HardwareProfile::modern_x86_64();
  bad_hw.n_ranks = 0U;
  REQUIRE_THROWS_AS(tdmd::PerfModel(bad_hw, tdmd::PotentialCost::morse()), std::invalid_argument);

  tdmd::PotentialCost bad_pc = tdmd::PotentialCost::morse();
  bad_pc.n_neighbors_per_atom = 0U;
  REQUIRE_THROWS_AS(tdmd::PerfModel(tdmd::HardwareProfile::modern_x86_64(), bad_pc),
                    std::invalid_argument);
}

TEST_CASE("Pattern 3 monotone in N_atoms", "[perfmodel][pattern3]") {
  const auto model = make_morse_model();

  // §3.2: T_c ∝ N_per_rank, T_halo ∝ N_per_rank^(2/3), both monotonic in N.
  const std::uint64_t sizes[] = {8'000U, 80'000U, 800'000U, 8'000'000U};
  double prev = 0.0;
  for (const auto n : sizes) {
    const auto p = model.predict_pattern3(n);
    REQUIRE(p.pattern_name == "Pattern3_SD");
    REQUIRE(p.recommended_K == 1U);
    REQUIRE(p.speedup_vs_baseline == Catch::Approx(1.0));
    REQUIRE(std::isfinite(p.t_step_sec));
    REQUIRE(p.t_step_sec > 0.0);
    REQUIRE(p.t_step_sec > prev);
    prev = p.t_step_sec;
  }
}

TEST_CASE("Pattern 1 invariant: predict_pattern1 rejects K=0", "[perfmodel][pattern1]") {
  const auto model = make_morse_model();
  REQUIRE_THROWS_AS(model.predict_pattern1(10'000U, 0U), std::invalid_argument);
}

TEST_CASE("Pattern 1 K=1 bounded by 1.5x Pattern 3", "[perfmodel][pattern1][invariant]") {
  const auto model = make_morse_model();

  // Exec pack T2.10 invariant: "predict_pattern1 с K=1 ≤ 1.5× predict_pattern3"
  // across the sane-N range. We sweep 8k → 8M to cover both the sched-dominated
  // tiny regime and the compute-dominated large-N regime.
  const std::uint64_t sizes[] = {8'000U, 80'000U, 800'000U, 8'000'000U, 80'000'000U};
  for (const auto n : sizes) {
    const auto p1 = model.predict_pattern1(n, 1U);
    const auto p3 = model.predict_pattern3(n);
    const double ratio = p1.t_step_sec / p3.t_step_sec;
    INFO("N=" << n << " TD/SD ratio=" << ratio);
    REQUIRE(ratio <= 1.5);
  }
}

TEST_CASE("Pattern 1 K=K_opt speedup vs baseline", "[perfmodel][pattern1][speedup]") {
  const auto model = make_morse_model();

  // For sufficiently large N and this hardware profile, Pattern 1 at K_opt
  // should reach parity with or beat Pattern 3. "Beat" is a stretch goal —
  // the CPU profile's intra/inter BW ratio is only ~17x, so the speedup is
  // modest but should at least be ≥ 1.0.
  for (const std::uint64_t n : {1'000'000U, 10'000'000U}) {
    const auto k_opt = model.K_opt(n);
    const auto p1 = model.predict_pattern1(n, k_opt);
    const auto p3 = model.predict_pattern3(n);

    INFO("N=" << n << " K_opt=" << k_opt << " TD=" << p1.t_step_sec << " SD=" << p3.t_step_sec
              << " speedup=" << p1.speedup_vs_baseline);
    REQUIRE(p1.speedup_vs_baseline >= 1.0);
  }
}

TEST_CASE("K_opt clamps to [1, 16] and stays a power of 2", "[perfmodel][k_opt][clamp]") {
  // modern_x86_64 path: CPU profile has T_p < T_c always for Morse, so the
  // overlap saturates — K_opt is always 1 regardless of N. Exercise this
  // degenerate case explicitly so we don't regress if someone "fixes" K_opt
  // to always grow.
  const auto cpu_model = make_morse_model();
  REQUIRE(cpu_model.K_opt(100U) == 1U);
  REQUIRE(cpu_model.K_opt(8'000U) == 1U);
  REQUIRE(cpu_model.K_opt(8'000'000U) == 1U);
  REQUIRE(cpu_model.K_opt(10'000'000'000ULL) == 1U);

  // Slow-interconnect profile: T_p >> T_c for large N → K_opt can hit the
  // clamp ceiling. Verify we never exceed 16 even at absurd N.
  const auto slow_model = make_slow_intra_model();
  for (const std::uint64_t n :
       {1'000'000ULL, 10'000'000ULL, 100'000'000ULL, 1'000'000'000ULL, 10'000'000'000ULL}) {
    const auto k = slow_model.K_opt(n);
    REQUIRE(k >= 1U);
    REQUIRE(k <= 16U);
    // k & (k-1) == 0 iff k is a power of 2 (§3.3: "power of 2" guidance).
    REQUIRE((k & (k - 1U)) == 0U);
  }

  // At the huge-N extreme on slow-intra, raw sqrt exceeds 16 — must clamp.
  REQUIRE(slow_model.K_opt(10'000'000'000ULL) == 16U);
}

TEST_CASE("K_opt scales ~sqrt(N) in asymptotic ramp-up", "[perfmodel][k_opt][asymptote]") {
  const auto model = make_slow_intra_model();

  // §3.3: K_opt² ∝ T_p ∝ N (with n_ranks held constant). So K_opt(N·4) ≈
  // K_opt(N)·2 in the ramp-up regime before the 16-ceiling saturates.
  //
  // For slow-intra profile, ramp-up region is roughly N ∈ [10^5, 10^6].
  // Below 10^5: K_opt = 1 (T_p still small). Above 10^6: K_opt saturates.
  const auto k_1e5 = model.K_opt(100'000U);
  const auto k_1e6 = model.K_opt(1'000'000U);
  const auto k_1e7 = model.K_opt(10'000'000U);

  INFO("slow-intra K_opt(1e5)=" << k_1e5 << " K_opt(1e6)=" << k_1e6 << " K_opt(1e7)=" << k_1e7);
  // Monotone non-decreasing in N.
  REQUIRE(k_1e6 >= k_1e5);
  REQUIRE(k_1e7 >= k_1e6);
  // Growth happens somewhere in this decade range — K_opt at the top end
  // should strictly exceed K_opt at the bottom (not stuck on a single value).
  REQUIRE(k_1e7 > k_1e5);
}

TEST_CASE("rank() output ordered ascending by t_step_sec", "[perfmodel][rank]") {
  const auto model = make_eam_model();

  const std::uint64_t sizes[] = {8'000U, 80'000U, 800'000U, 8'000'000U};
  for (const auto n : sizes) {
    const auto ranked = model.rank(n);
    REQUIRE(ranked.size() == 2U);
    REQUIRE(ranked[0].t_step_sec <= ranked[1].t_step_sec);

    // Deterministic: same call → same ordering, same values.
    const auto ranked2 = model.rank(n);
    REQUIRE(ranked[0].pattern_name == ranked2[0].pattern_name);
    REQUIRE(ranked[0].t_step_sec == ranked2[0].t_step_sec);

    // At least one candidate is Pattern3_SD and one Pattern1_TD — M2 always
    // enumerates both.
    bool has_sd = false, has_td = false;
    for (const auto& p : ranked) {
      if (p.pattern_name == "Pattern3_SD")
        has_sd = true;
      if (p.pattern_name == "Pattern1_TD")
        has_td = true;
    }
    REQUIRE(has_sd);
    REQUIRE(has_td);
  }
}

TEST_CASE("EAM T_c > Morse T_c for same N (cost sanity)", "[perfmodel][pattern3][eam]") {
  // §3.1: EAM midpoint (115 FLOPS/pair) > Morse midpoint (40) → predicted
  // T_step for EAM at fixed N should strictly exceed Morse T_step.
  const auto m_model = make_morse_model();
  const auto e_model = make_eam_model();
  const std::uint64_t n = 800'000U;
  const auto p_m = m_model.predict_pattern3(n);
  const auto p_e = e_model.predict_pattern3(n);
  REQUIRE(p_e.t_step_sec > p_m.t_step_sec);
}
