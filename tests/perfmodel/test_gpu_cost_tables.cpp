// SPEC: docs/specs/perfmodel/SPEC.md §3.1, §4.2, §9 roadmap M6
// Exec pack: docs/development/m6_execution_pack.md T6.11
//
// Validates the GPU cost-table infrastructure that T6.11 introduces:
//   (1) linear-model math: cost(N) = a + b·N evaluated correctly;
//   (2) structural sanity of the committed reference + mixed tables
//       (positive, finite, ordering of MixedFast vs Reference);
//   (3) PerfModel::predict_step_gpu_sec wiring through HardwareProfile::n_ranks.
//
// The ±20 % accuracy gate vs measured micro-bench data is scoped as T6.11b
// follow-up — it requires a Nsight-profiled calibration run on a specific
// GPU, which the CI pipeline (Option A, no self-hosted runner) cannot do.
// When T6.11b lands, a new TEST_CASE here will load measured coefficients
// from a JSON fixture and compare `predict_step_gpu_sec` against them.

#include "tdmd/perfmodel/gpu_cost_tables.hpp"
#include "tdmd/perfmodel/perfmodel.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>

namespace {

bool is_finite_positive(double x) noexcept {
  return std::isfinite(x) && x > 0.0;
}

void require_kernel_cost_sane(const tdmd::GpuKernelCost& k) {
  REQUIRE(is_finite_positive(k.a_sec));
  REQUIRE(is_finite_positive(k.b_sec_per_atom));
  // Sanity band: launch overhead 1 μs – 1 ms; per-atom 100 ps – 10 μs.
  // Widely generous so T6.11b measured values pass without adjustment.
  REQUIRE(k.a_sec >= 1.0e-6);
  REQUIRE(k.a_sec <= 1.0e-3);
  REQUIRE(k.b_sec_per_atom >= 1.0e-10);
  REQUIRE(k.b_sec_per_atom <= 1.0e-5);
}

}  // namespace

TEST_CASE("GpuKernelCost::predict — linear model math", "[perfmodel][gpu]") {
  tdmd::GpuKernelCost k{/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/5.0e-9};

  // Zero atoms — pure launch cost.
  REQUIRE(k.predict(0U) == Catch::Approx(10.0e-6));

  // 1 M atoms — 10 μs + 5 ns · 10^6 = 10 μs + 5 ms.
  REQUIRE(k.predict(1'000'000U) == Catch::Approx(1.0e-5 + 5.0e-3));

  // Monotone in N.
  REQUIRE(k.predict(100U) < k.predict(1000U));
  REQUIRE(k.predict(1000U) < k.predict(10'000U));
}

TEST_CASE("GpuCostTables::step_total_sec — sum of all stages", "[perfmodel][gpu]") {
  tdmd::GpuCostTables t;
  t.h2d_atom = {1.0e-6, 1.0e-9};
  t.nl_build = {2.0e-6, 2.0e-9};
  t.eam_force = {3.0e-6, 3.0e-9};
  t.vv_pre = {4.0e-6, 4.0e-9};
  t.vv_post = {5.0e-6, 5.0e-9};
  t.d2h_force = {6.0e-6, 6.0e-9};

  // At N=1000: a-sum = 21 μs, b-sum = 21 ns/atom × 1000 = 21 μs. Total = 42 μs.
  REQUIRE(t.step_total_sec(1000U) == Catch::Approx(42.0e-6));
}

TEST_CASE("gpu_cost_tables_fp64_reference — structural sanity", "[perfmodel][gpu]") {
  const auto t = tdmd::gpu_cost_tables_fp64_reference();

  require_kernel_cost_sane(t.nl_build);
  require_kernel_cost_sane(t.eam_force);
  require_kernel_cost_sane(t.vv_pre);
  require_kernel_cost_sane(t.vv_post);
  require_kernel_cost_sane(t.h2d_atom);
  require_kernel_cost_sane(t.d2h_force);

  REQUIRE(!t.provenance.empty());

  // EAM force dominates per-atom cost at steady state (60 neighbours, FP64).
  // If a future refactor drops EAM below NL per-atom, flag it — likely a bug.
  REQUIRE(t.eam_force.b_sec_per_atom > t.nl_build.b_sec_per_atom);
  REQUIRE(t.eam_force.b_sec_per_atom > t.vv_pre.b_sec_per_atom);
}

TEST_CASE("gpu_cost_tables_mixed_fast — structural sanity", "[perfmodel][gpu]") {
  const auto t = tdmd::gpu_cost_tables_mixed_fast();

  require_kernel_cost_sane(t.nl_build);
  require_kernel_cost_sane(t.eam_force);
  require_kernel_cost_sane(t.vv_pre);
  require_kernel_cost_sane(t.vv_post);
  require_kernel_cost_sane(t.h2d_atom);
  require_kernel_cost_sane(t.d2h_force);

  REQUIRE(!t.provenance.empty());
}

TEST_CASE("MixedFast EAM ≤ Reference EAM per-atom cost", "[perfmodel][gpu]") {
  // Philosophy B: FP32 pair math trades precision for speed. Per-atom cost
  // must be ≤ reference; equality is legal (e.g. bandwidth-bound workloads).
  const auto ref = tdmd::gpu_cost_tables_fp64_reference();
  const auto mix = tdmd::gpu_cost_tables_mixed_fast();

  REQUIRE(mix.eam_force.b_sec_per_atom <= ref.eam_force.b_sec_per_atom);

  // The other stages are math-identical between flavors (only EAM changes).
  // Asserting equality here catches accidental drift in the starter tables.
  REQUIRE(mix.nl_build.a_sec == Catch::Approx(ref.nl_build.a_sec));
  REQUIRE(mix.nl_build.b_sec_per_atom == Catch::Approx(ref.nl_build.b_sec_per_atom));
  REQUIRE(mix.vv_pre.a_sec == Catch::Approx(ref.vv_pre.a_sec));
  REQUIRE(mix.vv_post.a_sec == Catch::Approx(ref.vv_post.a_sec));
  REQUIRE(mix.h2d_atom.a_sec == Catch::Approx(ref.h2d_atom.a_sec));
  REQUIRE(mix.d2h_force.a_sec == Catch::Approx(ref.d2h_force.a_sec));
}

TEST_CASE("PerfModel::predict_step_gpu_sec — single-rank baseline", "[perfmodel][gpu]") {
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 1U;  // single-rank so n_per_rank == n_atoms

  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());
  const auto tables = tdmd::gpu_cost_tables_fp64_reference();

  const double t = pm.predict_step_gpu_sec(10'000U, tables);

  // Must equal table total + scheduler overhead at n_ranks=1.
  const double expected = tables.step_total_sec(10'000U) + hw.scheduler_overhead_sec;
  REQUIRE(t == Catch::Approx(expected));
  REQUIRE(is_finite_positive(t));
}

TEST_CASE("PerfModel::predict_step_gpu_sec — n_ranks divides the work", "[perfmodel][gpu]") {
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 8U;

  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());
  const auto tables = tdmd::gpu_cost_tables_fp64_reference();

  // 80 k total with 8 ranks → 10 k per rank — same wall-clock as the single-
  // rank 10 k case above (modulo scheduler overhead, which is per-iter not
  // per-rank so it's identical).
  const double t_eight = pm.predict_step_gpu_sec(80'000U, tables);
  const double t_one_per_rank = tables.step_total_sec(10'000U) + hw.scheduler_overhead_sec;

  REQUIRE(t_eight == Catch::Approx(t_one_per_rank));
}

TEST_CASE("PerfModel::predict_step_gpu_sec — Reference ≥ MixedFast", "[perfmodel][gpu]") {
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 1U;

  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());
  const auto ref = tdmd::gpu_cost_tables_fp64_reference();
  const auto mix = tdmd::gpu_cost_tables_mixed_fast();

  // Large N so the per-atom cost diff between FP64 and Mixed EAM dominates
  // the identical per-launch a-term. At 100 k atoms the ~2 ns/atom diff on
  // EAM = 200 μs — comfortably above noise.
  const double t_ref = pm.predict_step_gpu_sec(100'000U, ref);
  const double t_mix = pm.predict_step_gpu_sec(100'000U, mix);

  REQUIRE(t_mix <= t_ref);
}
