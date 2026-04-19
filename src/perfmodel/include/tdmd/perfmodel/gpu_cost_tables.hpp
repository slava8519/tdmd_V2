#pragma once

// SPEC: docs/specs/perfmodel/SPEC.md §3.1 (force cost), §4.2 (calibration),
//       §9 roadmap M6 (GPU cost tables)
// Exec pack: docs/development/m6_execution_pack.md T6.11
// Decisions: D-M6-14 (NVTX instrumentation feeds these numbers via Nsight)
//
// PerfModel GPU cost tables — linear models `cost(N) = a + b·N` per kernel /
// transfer, derived from T6.4/T6.5/T6.6 micro-benches and committed as literal
// constants. The analytic predictor uses them when the caller's backend is
// GPU (Production / MixedFast ExecProfile). CPU-only callers keep using the
// §3.1 flops/pair formula in `perfmodel.cpp`.
//
// Two separate tables: `fp64_reference()` and `mixed_fast()`. MixedFast's EAM
// force kernel is ~1.5–2× faster than the FP64 reference on the same hardware
// (T6.8a Nsight traces); the two tables capture that split so callers can
// predict the flavor they actually plan to run.
//
// **Not** a general-purpose benchmark cache. The coefficients are hand-curated
// with provenance metadata in the .cpp — a replacement requires a new PR, an
// Nsight trace, and a ±20 % acceptance gate bump. M8 auto-calibration lands in
// `calibration_cache.json` per perfmodel/SPEC §4.2; these tables are the
// starter data that calibration replaces.

#include <cstdint>
#include <string>

namespace tdmd {

// Linear model parameters for one kernel or memcpy op:
//   cost_sec = a_sec + b_sec_per_atom · n_atoms
// `a_sec` absorbs launch latency, H2D setup, kernel overhead. `b_sec_per_atom`
// is the steady-state per-atom marginal cost at the measurement N-range.
// The model is valid only in the N-range it was calibrated on (see provenance
// comment in `GpuCostTables::provenance` field).
struct GpuKernelCost {
  double a_sec = 0.0;
  double b_sec_per_atom = 0.0;

  [[nodiscard]] double predict(std::uint64_t n_atoms) const noexcept {
    return a_sec + b_sec_per_atom * static_cast<double>(n_atoms);
  }
};

// Bundle of all GPU ops a single MD step touches, matching the NVTX ranges
// instrumented in T6.11. Field names mirror the NVTX scope names (minus the
// subsystem prefix) so Nsight traces and code map 1:1:
//
//   nl.build              →  nl_build
//   eam.compute           →  eam_force
//   vv.pre_force_step     →  vv_pre
//   vv.post_force_step    →  vv_post
//   gpu.h2d.atoms_cells   →  h2d_atom         (per step, positions + cells)
//   gpu.d2h.forces        →  d2h_force        (per step, f/pe/virial)
//
// `step_total()` sums them for a single-zone, single-subdomain step — the
// quantity PerfModel exposes through `predict_step_time_gpu()`.
struct GpuCostTables {
  GpuKernelCost nl_build{};
  GpuKernelCost eam_force{};
  GpuKernelCost vv_pre{};
  GpuKernelCost vv_post{};
  GpuKernelCost h2d_atom{};
  GpuKernelCost d2h_force{};

  // Free-form metadata: GPU model, CUDA toolkit version, date, N-range.
  // Not machine-parsed; intended so reviewers can see at a glance whether
  // numbers are current when reading a PR that touches dependent code.
  std::string provenance;

  [[nodiscard]] double step_total_sec(std::uint64_t n_atoms) const noexcept {
    // Per-step: H2D positions+cells → NL build (cadence > 1 step in practice,
    // but T6.11 conservatively charges it once per step — matches T6.13
    // acceptance smoke) → EAM force → VV pre + post → D2H forces.
    return h2d_atom.predict(n_atoms) + nl_build.predict(n_atoms) + eam_force.predict(n_atoms) +
           vv_pre.predict(n_atoms) + vv_post.predict(n_atoms) + d2h_force.predict(n_atoms);
  }
};

// Committed calibration data. Numbers in `src/perfmodel/gpu_cost_tables.cpp`
// include date, GPU model, CUDA version, and N-range in their inline comments.
// Replace only via: (1) fresh Nsight trace, (2) PR with updated coefficients,
// (3) `test_gpu_cost_tables` ±20% gate green.

// FP64 Reference (D-M6-7 bit-exact CPU≡GPU path): EAM force uses FP64 math end
// to end, density_kernel / embedding_kernel / force_kernel with --fmad=false.
[[nodiscard]] GpuCostTables gpu_cost_tables_fp64_reference();

// MixedFast Production (Philosophy B, D-M6-8 ≤1e-5 rel force threshold): EAM
// pair math is FP32, per-atom accumulators stay FP64. ~1.5-2× faster on the
// force_kernel than the Reference path at N >> 10k.
[[nodiscard]] GpuCostTables gpu_cost_tables_mixed_fast();

}  // namespace tdmd
