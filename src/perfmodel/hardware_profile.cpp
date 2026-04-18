#include "tdmd/perfmodel/hardware_profile.hpp"

namespace tdmd {

HardwareProfile HardwareProfile::modern_x86_64() {
  // Canonical "M2 reference node" values. Cited from perfmodel/SPEC §4.1 plus
  // the vendor datasheets for the 2024 reference hardware listed there:
  //   cpu_flops_per_sec      — 2 × Xeon Gold 6430 @ 2.1 GHz nominal, AVX-512
  //                            FP64: ~2 TFLOPS/socket × 2 = 4e12
  //   intra_bw_bytes_per_sec — DDR5-4800 dual-channel per socket, ~300 GB/s
  //                            effective for contiguous streaming
  //   inter_bw_bytes_per_sec — 100 Gb/s NIC, ~12 GB/s effective after MPI
  //                            protocol overhead (B_inter in §3.2)
  //   scheduler_overhead_sec — midpoint of "10-50 μs" range in §3.5
  //   n_ranks                — dual-socket 8-rank layout (master SPEC §4.1
  //                            reference config)
  HardwareProfile hw;
  hw.cpu_flops_per_sec = 4.0e12;
  hw.intra_bw_bytes_per_sec = 300.0e9;
  hw.inter_bw_bytes_per_sec = 12.0e9;
  hw.scheduler_overhead_sec = 30.0e-6;
  hw.n_ranks = 8;
  return hw;
}

PotentialCost PotentialCost::morse() {
  // Pair potential row of §3.1 table: 30-50 FLOPS/pair; midpoint = 40.
  // Neighbor count 60 matches Al FCC @ 8 Å cutoff (T1 benchmark density).
  PotentialCost cost;
  cost.flops_per_pair = 40.0;
  cost.n_neighbors_per_atom = 60;
  return cost;
}

PotentialCost PotentialCost::eam_alloy() {
  // ManyBodyLocal/EAM row: 80-150 FLOPS/pair; midpoint = 115. EAM is a
  // two-pass algorithm (embedding density + pairwise), but §3.1 folds both
  // passes into the single per-pair constant — matches the reported numbers
  // from LAMMPS `pair eam/alloy` profiling on Ni-Al (T4 benchmark).
  PotentialCost cost;
  cost.flops_per_pair = 115.0;
  cost.n_neighbors_per_atom = 60;
  return cost;
}

}  // namespace tdmd
