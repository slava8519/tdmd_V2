#pragma once

// SPEC: docs/specs/perfmodel/SPEC.md §2.1 (HardwareProfile), §4.1 (calibration)
// Exec pack: docs/development/m2_execution_pack.md T2.10
//
// Hardware profile used by `PerfModel::predict*`. M2 ships a hand-curated
// `modern_x86_64()` factory with the numbers from perfmodel/SPEC §4.1; M4 will
// replace it with a micro-benchmark probe (`bench_gemm_fp64_small`,
// `bench_mpi_pingpong_minimum`, etc).
//
// M2 reduction of SPEC §2.1: we only carry the fields needed by Pattern 1 + 3
// formulas (§3.2, §3.3). GPU / topology fields (NVLink, PCIe, NCCL probes) land
// with the M6 GPU path. Extending this struct at M6 is additive — existing code
// reads only the subset defined here.
//
// `n_ranks` is explicitly part of the profile per SPEC §4.1 ("rank_count" probe
// result). It determines `P = P_space · P_time` used in §3.1
// `N_atoms_per_rank = N_total / P`. The exec-pack sketch omitted it; adding it
// here is not a SPEC delta (the master §12.7 type carries it).

#include <cstdint>

namespace tdmd {

// Analytic-model hardware description. Numbers reflect a modern x86_64 node
// with a midrange NIC; see `modern_x86_64()` for the canonical defaults.
//
// All bandwidths are in bytes/sec, scheduler overhead in seconds/iteration, and
// FLOPS in FP64 ops/sec. Mixing units here would make the §3.1-§3.3 formulas
// silently wrong, so the constructors validate finiteness but trust caller
// units — the factory is the safe entry point.
struct HardwareProfile {
  // Compute: single-rank peak FP64 throughput — denominator of `T_c` in §3.1.
  double cpu_flops_per_sec = 0.0;

  // Intra-rank temporal packet bandwidth: shared-memory / same-socket transfer
  // for Pattern 1 pipeline stages. Numerator-denominator of `T_p` in §3.3.
  double intra_bw_bytes_per_sec = 0.0;

  // Inter-rank halo bandwidth for Pattern 3 (§3.2 `B_inter_rank`). MPI / NIC
  // effective throughput after protocol overhead.
  double inter_bw_bytes_per_sec = 0.0;

  // Scheduler overhead per TD iteration (§3.5). 10-50 μs typical; grows with
  // zone count but zone scaling lives in M7.
  double scheduler_overhead_sec = 0.0;

  // Number of ranks participating in the run. Master SPEC §12.7 `rank_count`
  // probe result; the exec-pack sketch omitted this but Pattern 3 formula in
  // §3.2 needs `N_atoms_per_rank = N_total / P` and Pattern 1 `T_p`
  // denominator uses atoms-per-zone, also divided by rank count.
  //
  // Default 8 matches the "modern x86_64" node in SPEC §4.1 (single NUMA dual-
  // socket box). Single-rank (`n_ranks=1`) degenerates Pattern 3 to pure
  // compute with `T_halo ≡ 0` — a legitimate edge case for the `explain` CLI
  // on small analysis workstations.
  std::uint32_t n_ranks = 8;

  // Canonical factory: midrange 2024-era dual-socket x86_64 node with 8 ranks
  // and a 100 Gb/s NIC. Exact values below come from perfmodel/SPEC §4.1.
  // Re-measurement lands with the M4 auto-probe.
  [[nodiscard]] static HardwareProfile modern_x86_64();
};

// Per-potential cost constants consumed by `T_c = N · C_force / FLOPS`
// (§3.1). The table in SPEC §3.1 gives ranges per PotentialKind; M2 ships the
// midpoint of each range as `flops_per_pair` (Morse = 40, EAM = 115). M4
// auto-calibration will replace these with measured values per
// `calibration_cache.json` (§4.2).
struct PotentialCost {
  // FLOP count per pair interaction (dimensionless). `N_neighbors_per_atom`
  // multiplied by this yields `C_force_per_atom` in §3.1 notation.
  double flops_per_pair = 0.0;

  // Typical neighbor count per atom for a canonical FCC metal at the cutoff
  // radius used here. The fuzz invariant in §3.1 lists 50-300 as the normal
  // range; both factories below sit in the middle of that band.
  std::uint32_t n_neighbors_per_atom = 0;

  // Midpoint of SPEC §3.1 Pair range (30-50) × 60 neighbors for Al FCC at 8 Å.
  [[nodiscard]] static PotentialCost morse();

  // Midpoint of SPEC §3.1 ManyBodyLocal/EAM range (80-150). Ni-Al Mishin 2004
  // at 6 Å sees ~55 neighbors; we round to 60 for symmetry with `morse()`.
  [[nodiscard]] static PotentialCost eam_alloy();
};

}  // namespace tdmd
