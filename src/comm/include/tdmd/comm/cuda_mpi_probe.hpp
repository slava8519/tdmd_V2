#pragma once

// SPEC: docs/specs/comm/SPEC.md §6.2 (GpuAwareMpiBackend probe)
// Master spec: §10 (parallel model), §12.6 (comm interfaces)
// Exec pack: docs/development/m7_execution_pack.md T7.3
//
// Runtime probe for CUDA-aware MPI support. Header-only API: returns a
// boolean answer to the question "can MPI move device pointers without a
// host bounce?". The answer is cached per-process — the underlying probe
// is cheap but the result never changes during a run.
//
// Probe order (first hit wins):
//   1. `MPIX_Query_cuda_support()` — OpenMPI / Spectrum MPI extension
//      returning 1 if MPI was built with CUDA support AND the runtime is
//      currently routing device pointers through that path.
//   2. `OMPI_MCA_opal_cuda_support` env var = "true" — manual override
//      used on systems where the symbol isn't exposed but CUDA support
//      was wired in via mpi_config or environment file.
//   3. `MV2_USE_CUDA` env var = "1" — MVAPICH2-GDR's enable flag.
//   4. Otherwise → false.
//
// The probe NEVER aborts: a missing symbol, missing env var, or even a
// missing MPI library is a clean `false`. Callers that need CUDA-aware
// MPI for correctness (e.g. `GpuAwareMpiBackend` constructor) check the
// result and throw their own descriptive error.
//
// This module compiles with TDMD_ENABLE_MPI=0 too — the probe simply
// returns false in that case so SimulationEngine preflight doesn't have
// to special-case the no-MPI build flavor.

namespace tdmd::comm {

// True iff the current MPI runtime supports passing CUDA device pointers
// directly to MPI_Send / MPI_Recv. Cheap; result is cached after first call.
[[nodiscard]] bool is_cuda_aware_mpi() noexcept;

// Reset the cached probe result. Test-only — production callers must treat
// `is_cuda_aware_mpi()` as a process-lifetime constant. Exposed so unit
// tests can simulate both probe outcomes by toggling env vars between
// invocations.
void reset_cuda_mpi_probe_cache_for_testing() noexcept;

}  // namespace tdmd::comm
