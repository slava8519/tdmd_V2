// SPEC: docs/specs/perfmodel/SPEC.md §3.4 (Pattern 2 hybrid), §11.5 (T7.10)
// Exec pack: docs/development/m7_execution_pack.md T7.10
//
// Validates the Pattern 2 (two-level hybrid TD × SD) cost-prediction surface
// added in T7.10:
//   (1) `face_neighbors_count` geometry helper for periodic [Nx,Ny,Nz] grids;
//   (2) `predict_step_hybrid_seconds` linear math + halo scaling with neighbors;
//   (3) `recommend_pattern2` decision logic — degenerate [1,1,1], small N
//       (Pattern 1 wins), large N (Pattern 2 wins), 5% margin threshold;
//   (4) Pattern 1 path regression — `predict_step_gpu_sec` unchanged.
//
// All numbers come from the placeholder coefficients shipped in
// `gpu_cost_tables.cpp` (T7.10 starter estimates). Calibration and the ±25%
// gate vs measured Pattern 2 hybrid wall-time live in T7.13.

#include "tdmd/perfmodel/gpu_cost_tables.hpp"
#include "tdmd/perfmodel/perfmodel.hpp"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>

namespace {

bool is_finite_positive(double x) noexcept {
  return std::isfinite(x) && x > 0.0;
}

}  // namespace

TEST_CASE("face_neighbors_count — geometry helper", "[perfmodel][pattern2][t710]") {
  using A = std::array<std::uint32_t, 3>;

  // Pattern 1 collapse: no axis partitioned → no peer subdomains.
  REQUIRE(tdmd::face_neighbors_count(A{1U, 1U, 1U}) == 0U);

  // Single-axis Pattern 2 (canonical [2,1,1]): 2 face neighbors along x only.
  REQUIRE(tdmd::face_neighbors_count(A{2U, 1U, 1U}) == 2U);
  REQUIRE(tdmd::face_neighbors_count(A{1U, 2U, 1U}) == 2U);
  REQUIRE(tdmd::face_neighbors_count(A{1U, 1U, 2U}) == 2U);

  // Two-axis: 4 face neighbors. Three-axis: 6 (full 3D periodic interior).
  REQUIRE(tdmd::face_neighbors_count(A{2U, 2U, 1U}) == 4U);
  REQUIRE(tdmd::face_neighbors_count(A{2U, 2U, 2U}) == 6U);
  REQUIRE(tdmd::face_neighbors_count(A{4U, 4U, 4U}) == 6U);  // count, not magnitude
}

TEST_CASE("GpuCostTables — Pattern 2 stages structural sanity",
          "[perfmodel][pattern2][gpu][t710]") {
  const auto t = tdmd::gpu_cost_tables_fp64_reference();

  // All four new stages finite + non-negative; provenance preserved.
  REQUIRE(is_finite_positive(t.halo_pack.a_sec));
  REQUIRE(is_finite_positive(t.halo_pack.b_sec_per_atom));
  REQUIRE(is_finite_positive(t.halo_send_outer.a_sec));
  REQUIRE(is_finite_positive(t.halo_send_outer.b_sec_per_atom));
  REQUIRE(is_finite_positive(t.halo_unpack.a_sec));
  REQUIRE(is_finite_positive(t.halo_unpack.b_sec_per_atom));
  REQUIRE(is_finite_positive(t.nccl_allreduce_inner.a_sec));

  // halo_send_outer dominates the round-trip per-atom cost (NIC bandwidth
  // bottleneck): D2H + MPI + H2D > on-device pack/unpack at the per-atom
  // marginal. If a future redesign breaks this it likely means we forgot the
  // host-staging path is no longer the bottleneck.
  REQUIRE(t.halo_send_outer.b_sec_per_atom >= t.halo_pack.b_sec_per_atom);
  REQUIRE(t.halo_send_outer.b_sec_per_atom >= t.halo_unpack.b_sec_per_atom);
}

TEST_CASE("GpuCostTables::outer_halo_sec_per_neighbor — sum of 3 stages",
          "[perfmodel][pattern2][gpu][t710]") {
  tdmd::GpuCostTables t;
  t.halo_pack = {1.0e-6, 1.0e-9};
  t.halo_send_outer = {2.0e-6, 2.0e-9};
  t.halo_unpack = {3.0e-6, 3.0e-9};

  // At N=1000: a-sum = 6 μs, b-sum = 6 ns/atom × 1000 = 6 μs. Total = 12 μs.
  REQUIRE(t.outer_halo_sec_per_neighbor(1000U) == Catch::Approx(12.0e-6));

  // Zero halo atoms — pure launch overhead sum.
  REQUIRE(t.outer_halo_sec_per_neighbor(0U) == Catch::Approx(6.0e-6));
}

TEST_CASE("predict_step_hybrid_seconds — collapses to inner+reduction at 0 neighbors",
          "[perfmodel][pattern2][t710]") {
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 1U;
  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());
  const auto tables = tdmd::gpu_cost_tables_fp64_reference();

  tdmd::Pattern2CostInputs in;
  in.n_atoms_per_subdomain = 50'000U;
  in.n_face_neighbors = 0U;  // Pattern 1 collapse — no halo cost

  const double t = pm.predict_step_hybrid_seconds(in, tables);
  const double expected = tables.step_total_sec(50'000U) + tables.nccl_allreduce_inner.predict(0U) +
                          hw.scheduler_overhead_sec;
  REQUIRE(t == Catch::Approx(expected));
}

TEST_CASE("predict_step_hybrid_seconds — outer halo scales linearly with n_face_neighbors",
          "[perfmodel][pattern2][t710]") {
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 1U;
  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());
  const auto tables = tdmd::gpu_cost_tables_fp64_reference();

  tdmd::Pattern2CostInputs in;
  in.n_atoms_per_subdomain = 100'000U;

  in.n_face_neighbors = 0U;
  const double t0 = pm.predict_step_hybrid_seconds(in, tables);
  in.n_face_neighbors = 2U;
  const double t2 = pm.predict_step_hybrid_seconds(in, tables);
  in.n_face_neighbors = 6U;
  const double t6 = pm.predict_step_hybrid_seconds(in, tables);

  // Per-neighbor cost is the same; total is monotone increasing.
  REQUIRE(t2 > t0);
  REQUIRE(t6 > t2);

  // Linear: (t6 - t0) == 3 × (t2 - t0) within FP epsilon. The Pattern 1 inner
  // + reduction + scheduler overhead cancel out in the difference.
  const double per_neighbor_2 = (t2 - t0) / 2.0;
  const double per_neighbor_6 = (t6 - t0) / 6.0;
  REQUIRE(per_neighbor_2 == Catch::Approx(per_neighbor_6));
}

TEST_CASE("recommend_pattern2 — degenerate [1,1,1] returns Pattern1",
          "[perfmodel][pattern2][t710]") {
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 1U;
  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());
  const auto tables = tdmd::gpu_cost_tables_fp64_reference();

  const auto rec = pm.recommend_pattern2(100'000U, {1U, 1U, 1U}, tables);

  REQUIRE(rec.recommended_pattern == "Pattern1");
  REQUIRE(rec.margin_fraction == Catch::Approx(0.0));
  REQUIRE(is_finite_positive(rec.t_pattern1_sec));
  // Pattern 2 timing equals Pattern 1 when no partitioning happened — caller
  // can still display "no Pattern 2 candidate" without inventing Inf or NaN.
  REQUIRE(rec.t_pattern2_sec == Catch::Approx(rec.t_pattern1_sec));
}

TEST_CASE("recommend_pattern2 — small system favors Pattern1", "[perfmodel][pattern2][t710]") {
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 1U;
  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());
  const auto tables = tdmd::gpu_cost_tables_fp64_reference();

  // 1k atoms with [2,1,1]: 500 atoms per subdomain — halo + reduction
  // overhead dominates the inner compute saving. Pattern 1 stays default.
  const auto rec = pm.recommend_pattern2(1'000U, {2U, 1U, 1U}, tables);

  REQUIRE(rec.recommended_pattern == "Pattern1");
}

TEST_CASE("recommend_pattern2 — large system + multi-axis favors Pattern2",
          "[perfmodel][pattern2][t710]") {
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 1U;
  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());
  const auto tables = tdmd::gpu_cost_tables_fp64_reference();

  // 8M atoms split 8-way across [2,2,2] — per-subdomain compute saving
  // (8× less inner work, ~8× shorter step) easily clears the 5% margin
  // threshold even with full 6 face-neighbor halo overhead.
  const auto rec = pm.recommend_pattern2(8'000'000U, {2U, 2U, 2U}, tables);

  REQUIRE(rec.recommended_pattern == "Pattern2");
  REQUIRE(rec.margin_fraction >= 0.05);
  REQUIRE(rec.t_pattern2_sec < rec.t_pattern1_sec);
}

TEST_CASE("recommend_pattern2 — deterministic for identical inputs",
          "[perfmodel][pattern2][t710]") {
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 1U;
  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());
  const auto tables = tdmd::gpu_cost_tables_fp64_reference();

  const auto rec1 = pm.recommend_pattern2(500'000U, {2U, 2U, 1U}, tables);
  const auto rec2 = pm.recommend_pattern2(500'000U, {2U, 2U, 1U}, tables);

  REQUIRE(rec1.recommended_pattern == rec2.recommended_pattern);
  REQUIRE(rec1.t_pattern1_sec == rec2.t_pattern1_sec);
  REQUIRE(rec1.t_pattern2_sec == rec2.t_pattern2_sec);
  REQUIRE(rec1.margin_fraction == rec2.margin_fraction);
}

TEST_CASE("Pattern 1 regression — predict_step_gpu_sec unchanged", "[perfmodel][pattern2][t710]") {
  // Sanity: T7.10 must be additive. The exact T6.11 acceptance test pattern
  // (n_atoms=10k, single rank, FP64 reference) yields the same number.
  auto hw = tdmd::HardwareProfile::modern_x86_64();
  hw.n_ranks = 1U;
  tdmd::PerfModel pm(hw, tdmd::PotentialCost::eam_alloy());
  const auto tables = tdmd::gpu_cost_tables_fp64_reference();

  const double t = pm.predict_step_gpu_sec(10'000U, tables);
  const double expected = tables.step_total_sec(10'000U) + hw.scheduler_overhead_sec;
  REQUIRE(t == Catch::Approx(expected));
}
