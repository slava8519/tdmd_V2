// SPEC: docs/specs/perfmodel/SPEC.md §3.1, §4.2 (calibration), §9 roadmap M6
// Exec pack: docs/development/m6_execution_pack.md T6.11
//
// PerfModel GPU cost-table constants. Coefficients are starter estimates for
// a midrange Ampere/Ada-class consumer GPU (RTX 4080 / RTX A5000 tier) running
// CUDA 13.1 with the tdmd_gpu kernels introduced in T6.4–T6.8a. They are
// replaced by measured values once the T6.11 calibration harness is driven
// from a Nsight Systems session — see the `//  CALIBRATION:` comments below.
//
// Provenance string in every table documents what those numbers correspond
// to; the ±20 % acceptance gate in test_gpu_cost_tables enforces sane values
// and catches drift when the harness reruns it.
//
// Linear model is: cost_sec(N) = a_sec + b_sec_per_atom · N.
// The model is valid only for 10 k ≲ N ≲ 1 M atoms per device — below 10 k
// launch overhead dominates (a-term) and the b-term is noise; above ~1 M the
// kernels saturate shared memory + register pressure and the model under-
// predicts. Callers that exceed the range should consult fresh Nsight data.

#include "tdmd/perfmodel/gpu_cost_tables.hpp"

namespace tdmd {

namespace {

// Keep the provenance string short (~120 chars): GPU class + CUDA tool + date
// + N-range + method. The exact device/toolkit are documented once, avoiding
// per-field repetition.
//
// CALIBRATION:
//   Coefficients below are **order-of-magnitude placeholders** derived from
//   published Ampere consumer-GPU benchmarks (CUDA 13.1 PCIe Gen4 x16,
//   ~25 GB/s H2D, ~50 μs kernel launch-to-completion for small ops). They
//   pass the structural sanity gate in test_gpu_cost_tables but will not
//   meet the ±20 % acceptance threshold on a specific machine until replaced
//   with Nsight-measured values from bench_gpu_cost_calibration (T6.11b).
//
// Once measured coefficients land, the `provenance` string should shift from
// "initial estimate" to the specific GPU model + CUDA version that was on the
// bench at calibration time (e.g. "RTX 4080 / CUDA 13.1.2 / 2026-04-20").
constexpr const char* kReferenceProvenance =
    "initial estimate, Ampere/Ada consumer, CUDA 13.1, 10k-1M N-range, "
    "T6.11 placeholder — replace via calibration harness (M6 exec pack)";

constexpr const char* kMixedFastProvenance =
    "initial estimate, Ampere/Ada consumer, CUDA 13.1, 10k-1M N-range, "
    "FP32-pair kernels (T6.8a), T6.11 placeholder — replace via calibration "
    "harness";

}  // namespace

GpuCostTables gpu_cost_tables_fp64_reference() {
  GpuCostTables t;

  // ---- Memory transfers (PCIe Gen4 x16 ≈ 25 GB/s effective) -------------
  // H2D: positions (3 × 8 B) + types (4 B) + cell CSR (~4 B/atom slot) per
  // step ≈ 32 B/atom. `a` = per-copy launch overhead ~10 μs.
  t.h2d_atom = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/32.0 / 25.0e9};

  // D2H: forces (24 B) + per-atom PE/virial buffers (8 × 7 = 56 B) ≈ 80 B/atom.
  t.d2h_force = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/80.0 / 25.0e9};

  // ---- Kernels -----------------------------------------------------------
  // NL build (T6.4): two passes (count + emit) + host scan. ~30 μs entry
  // cost, ~3 ns/atom steady-state (thread-per-atom, 27-cell stencil walk).
  t.nl_build = {/*a_sec=*/30.0e-6, /*b_sec_per_atom=*/3.0e-9};

  // EAM force FP64 (T6.5): three kernels (density + embed + force). FP64
  // fma throttled by --fmad=false; ~50 μs entry, ~5 ns/atom.
  t.eam_force = {/*a_sec=*/50.0e-6, /*b_sec_per_atom=*/5.0e-9};

  // VV pre/post (T6.6): element-wise thread-per-atom. ~10 μs entry, ~1 ns/atom.
  t.vv_pre = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/1.0e-9};
  t.vv_post = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/1.0e-9};

  t.provenance = kReferenceProvenance;
  return t;
}

GpuCostTables gpu_cost_tables_mixed_fast() {
  GpuCostTables t;

  // Transfers and NL/VV kernels are unchanged — mixed/reference split only
  // affects EAM force math precision (Philosophy B: FP32 pair math, FP64
  // accumulators). T6.8a Nsight traces show identical NL/VV timing.
  t.h2d_atom = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/32.0 / 25.0e9};
  t.d2h_force = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/80.0 / 25.0e9};
  t.nl_build = {/*a_sec=*/30.0e-6, /*b_sec_per_atom=*/3.0e-9};
  t.vv_pre = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/1.0e-9};
  t.vv_post = {/*a_sec=*/10.0e-6, /*b_sec_per_atom=*/1.0e-9};

  // EAM force MixedFast: FP32 pair math ~1.7× faster on per-atom loop
  // (T6.8a traces). Entry cost identical (same 3 kernels).
  t.eam_force = {/*a_sec=*/50.0e-6, /*b_sec_per_atom=*/3.0e-9};

  t.provenance = kMixedFastProvenance;
  return t;
}

}  // namespace tdmd
