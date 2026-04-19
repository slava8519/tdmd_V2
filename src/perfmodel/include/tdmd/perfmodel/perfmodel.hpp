#pragma once

// SPEC: docs/specs/perfmodel/SPEC.md §2.2 (PerfModel), §3.2 (Pattern 3),
//       §3.3 (Pattern 1), §3.5 (overhead), §6.1 (explain format)
// Exec pack: docs/development/m2_execution_pack.md T2.10
//
// Analytic time-to-step predictor for Pattern 1 (pure TD, single-subdomain)
// and Pattern 3 (pure SD, MPI-only halo). Pattern 2 (two-level hybrid) is
// deferred to M7 per SPEC §9 roadmap; the full `PerformancePrediction` of
// §2.1 (with `t_step_pattern2_seconds` etc.) extends this M2-reduced struct
// additively when M7 lands.
//
// The M2 design aim is: enough of the formula surface to make
// `tdmd explain --perf` (T2.11) and the M2 integration smoke (T2.13)
// meaningful, without stubbing Pattern 2 fields that would leave NaN / Inf
// traps for callers.

#include "tdmd/perfmodel/gpu_cost_tables.hpp"
#include "tdmd/perfmodel/hardware_profile.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tdmd {

// One prediction record. Field set matches exec pack T2.10 (subset of master
// SPEC §2.1 `PerformancePrediction` — the full record arrives with M7 Pattern 2
// support). See the comment on `PerfModel` below for the roadmap rationale.
struct PerfPrediction {
  // "Pattern1_TD" or "Pattern3_SD". Internal naming matches SPEC §3 headings;
  // the CLI `explain` command (T2.11) maps these to user-friendly labels per
  // SPEC §6.1.
  std::string pattern_name;

  // Wall-clock seconds per MD step, after overhead corrections from §3.5.
  // Strictly positive and finite for any `n_atoms ≥ 10`.
  double t_step_sec = 0.0;

  // Pipeline depth `K` (§3.3). `K=1` for Pattern 3 by convention — the SD
  // pattern has no TD pipeline, so the field records "no batching".
  std::uint32_t recommended_K = 1;

  // `T_step_SD / T_step_pattern` — dimensionless. Pattern 3 self-reports 1.0;
  // Pattern 1 reports the speedup of TD over SD for the same `n_atoms`.
  double speedup_vs_baseline = 1.0;
};

// Analytic predictor. One instance per (hardware, potential) pair.
//
// The §2.2 SPEC sketch defines a `predict(ZoningPlan, PotentialModel, ...)`
// virtual; M2 ships the concrete numeric core as a plain class because
// `ZoningPlan` / `PotentialModel` are scheduler-owned types that land with
// M3+. Wiring this concrete model behind the virtual interface is a trivial
// adapter when those types exist.
class PerfModel {
public:
  PerfModel(HardwareProfile hw, PotentialCost potential);

  // Pattern 3: pure spatial decomposition (SPEC §3.2).
  //   T_step_SD = T_c + T_halo_SD
  //   T_halo_SD = 2 · (N_atoms_per_rank)^(2/3) · atom_record_size / B_inter
  // Plus §3.5 corrections: `T_sched` only (neighbor rebuild / migration
  // amortizations need live telemetry which lands with T2.12).
  [[nodiscard]] PerfPrediction predict_pattern3(std::uint64_t n_atoms) const;

  // Pattern 1: pure time decomposition (SPEC §3.3).
  //   T_step_TD(K) = T_c + T_comm_inner(K)
  //   T_comm_inner(K) = max(0, T_p / K - T_c_overlap)
  // with `T_c_overlap` taken as `T_c` (full overlap possible per the §3.3
  // "at K=1, T_comm_inner may almost fully overlap" note). `K` is the caller's
  // choice; see `rank()` for the auto-K path.
  [[nodiscard]] PerfPrediction predict_pattern1(std::uint64_t n_atoms, std::uint32_t K) const;

  // Optimal `K` for Pattern 1 per SPEC §3.3:
  //   K_opt = sqrt(T_p / T_c_startup)
  // rounded up to the nearest power of 2, then clamped to [1, 16] per the
  // master-spec §6.5a guidance ("K_opt ∈ {1, 2, 4, 8}" typical, {16, 32, 64}
  // for very large / slow network). The M2 clamp to 16 matches the `rank()`
  // sweep upper bound; larger K is unused until M5 anchor tests.
  [[nodiscard]] std::uint32_t K_opt(std::uint64_t n_atoms) const;

  // Full set of candidates, sorted ascending by `t_step_sec`. At M2:
  // {Pattern3_SD, Pattern1_TD @ K_opt}. M7 adds Pattern 2 sweep entries.
  // Deterministic ordering on ties: Pattern 3 first (lower `pattern_name`
  // string by lex sort, used as the tiebreaker).
  [[nodiscard]] std::vector<PerfPrediction> rank(std::uint64_t n_atoms) const;

  // GPU per-step time, consuming pre-calibrated `GpuCostTables` (Reference
  // or MixedFast). Returns wall-clock seconds for a single MD step on one
  // rank owning `n_atoms`. Adds `hw_.scheduler_overhead_sec` per §3.5 — the
  // dispatch / runtime overhead is paid regardless of CPU vs GPU backend.
  //
  // Used by `tdmd explain --perf --backend gpu` (CLI landing in T6.12+) and
  // by the M7 Pattern 2 cost estimator when the inner subdomain runs on GPU.
  // **Not** wired into `rank()` yet — ranker comparisons are CPU-only until
  // the heterogeneous model lands (M10+, see perfmodel/SPEC §9 roadmap).
  [[nodiscard]] double predict_step_gpu_sec(std::uint64_t n_atoms,
                                            const GpuCostTables& tables) const noexcept;

  // T7.10 — Pattern 2 (two-level hybrid TD × SD) per-step prediction.
  // Cost model (master spec §12.7, perfmodel/SPEC §3.4 + §11.5):
  //   t_hybrid = t_inner_TD + t_outer_halo + t_reduction
  //   t_inner_TD   = step_total_sec(n_atoms_per_subdomain)        — T6.11 path
  //   t_outer_halo = n_face_neighbors · outer_halo_sec_per_neighbor(n_halo)
  //                  with n_halo ≈ n_atoms_per_subdomain^(2/3)    — §3.2 face area
  //   t_reduction  = nccl_allreduce_inner.predict(0)              — fixed payload
  //
  // Pure cost — caller computes geometry. Returns 0 if `n_face_neighbors == 0`
  // for the outer term, collapsing to `predict_step_gpu_sec(n_atoms_per_subdomain)`
  // plus the fixed reduction overhead. Adds `hw_.scheduler_overhead_sec` once
  // per step (matches T6.11 Pattern 1 path).
  [[nodiscard]] double predict_step_hybrid_seconds(const struct Pattern2CostInputs& inputs,
                                                   const GpuCostTables& tables) const noexcept;

  // T7.10 — Pattern recommendation. Compares the Pattern 1 single-subdomain
  // estimate against the Pattern 2 estimate for the given subdomain grid;
  // recommends Pattern 2 only when it wins by >= 5% (OQ-M7-9 resolution —
  // matches dissertation efficiency-tolerance precedent). Pattern 1 is the
  // default tiebreaker when Pattern 2 is within noise.
  [[nodiscard]] struct Pattern2Recommendation recommend_pattern2(
      std::uint64_t n_atoms_total,
      const std::array<std::uint32_t, 3>& subdomains,
      const GpuCostTables& tables) const noexcept;

  [[nodiscard]] const HardwareProfile& hardware() const noexcept { return hw_; }
  [[nodiscard]] const PotentialCost& potential() const noexcept { return potential_; }

private:
  HardwareProfile hw_;
  PotentialCost potential_;

  // Shared compute time per step per rank — §3.1 numerator. Extracted to
  // avoid repeating `N · C_force / FLOPS` across the three predictors.
  [[nodiscard]] double compute_time_sec(std::uint64_t n_atoms_per_rank) const noexcept;
};

// T7.10 — input bundle for `predict_step_hybrid_seconds()`. Caller computes
// the per-subdomain partition + face-neighbor count from the engine-side
// `zoning.subdomains` array. See `face_neighbors_count()` helper below for
// the canonical geometry formula (periodic interior subdomain).
struct Pattern2CostInputs {
  // Per-subdomain atom count = N_total / product(subdomains). Caller does
  // the division so PerfModel doesn't need the full grid struct.
  std::uint64_t n_atoms_per_subdomain = 0;

  // Number of face-adjacent peer subdomains (0..6). Computed from the grid
  // by counting axes with N_axis >= 2; periodic boundaries assumed (M7
  // default). 0 collapses Pattern 2 to the Pattern 1 inner-only cost.
  std::uint32_t n_face_neighbors = 0;
};

// T7.10 — output of `recommend_pattern2()`. Carries both candidate timings
// for explainability (`tdmd explain --perf` consumer in M9+) plus the
// dimensionless margin so CLI / telemetry can decide whether the prediction
// is "Pattern 2 wins decisively" or "noise-limited tossup".
struct Pattern2Recommendation {
  // "Pattern1" or "Pattern2" — uses the same naming convention as
  // `PerfPrediction::pattern_name` for downstream consistency.
  std::string recommended_pattern;

  // Pattern 1 estimate: all atoms on one subdomain, one device. Computed as
  // `tables.step_total_sec(n_atoms_total) + scheduler_overhead`.
  double t_pattern1_sec = 0.0;

  // Pattern 2 estimate from `predict_step_hybrid_seconds()` for the
  // requested subdomain grid.
  double t_pattern2_sec = 0.0;

  // (t_pattern1 - t_pattern2) / t_pattern1. Positive → Pattern 2 faster.
  // `recommended_pattern == "Pattern2"` iff margin >= 0.05 (OQ-M7-9).
  double margin_fraction = 0.0;
};

// T7.10 — geometry helper: count face-neighbors of an interior subdomain in
// a periodic [Nx, Ny, Nz] grid. Each axis with N_axis >= 2 contributes 2
// face neighbors (one per face along that axis); axes with N_axis == 1
// contribute 0. Range: 0 (Pattern 1, [1,1,1]) to 6 (full 3D Pattern 2).
[[nodiscard]] std::uint32_t face_neighbors_count(
    const std::array<std::uint32_t, 3>& subdomains) noexcept;

}  // namespace tdmd
