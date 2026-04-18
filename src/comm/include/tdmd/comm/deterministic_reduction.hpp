#pragma once

// SPEC: docs/specs/comm/SPEC.md §7.2 (deterministic reduction)
// Master spec: §7.3 (Level 1 determinism), §D.14 (MPI guarantees)
// Exec pack: docs/development/m5_execution_pack.md T5.4
//
// Deterministic replacement for MPI_Allreduce(SUM, MPI_DOUBLE) in the
// Reference profile (D-M5-9). The native collective sums in an
// implementation-defined order — legal per MPI spec but destroys the
// bitwise "same input → same output" guarantee required by playbook §3.5
// Level 1 determinism.
//
// Strategy: Allgather all local values, then a fixed-order Kahan-compensated
// reduction on every rank. Cost: 2–3× MPI_Allreduce, acceptable for the
// handful of global reductions per step (thermo, drift, energy). Fast
// profile (M8+) may opt into raw MPI_Allreduce.

#include <cstddef>
#include <vector>

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI
#include <mpi.h>
#endif

namespace tdmd::comm {

// Kahan-compensated sum over rank-ordered values. `values` is expected to
// contain one entry per rank, indexed by rank id. Pure CPU helper — callable
// without MPI; used by the backend after MPI_Allgather. Also callable by
// unit tests to compare against the collective path.
double kahan_sum_ordered(const std::vector<double>& values) noexcept;

#if defined(TDMD_ENABLE_MPI) && TDMD_ENABLE_MPI
// Deterministic global sum over `comm`: gather every rank's `local` to all
// ranks, then run kahan_sum_ordered in rank order. Return value is
// bit-identical on every rank provided the inputs are bit-identical across
// ranks at the same logical step.
double deterministic_sum_double(double local, MPI_Comm comm);
#endif

}  // namespace tdmd::comm
