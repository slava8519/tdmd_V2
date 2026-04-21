// SPEC: docs/specs/gpu/SPEC.md §7.5 (SNAP GPU — T8.6b), §1.1 (data-oblivious gpu/)
// Module SPEC: docs/specs/potentials/SPEC.md §6 (SNAP)
// Exec pack: docs/development/m8_execution_pack.md T8.6b
// Decisions: D-M6-17 (PIMPL firewall — this header is PRIVATE under src/gpu/)
//
// Host-side SNAP index-table builder. A standalone port of SnaEngine's
// build_indexlist + init_clebsch_gordan + init_rootpqarray producing flat
// STL arrays ready for device H2D upload. Kept here (src/gpu/, private) so
// the gpu module stays decoupled from src/potentials/snap/sna_engine (module
// SPEC §1.1 data-oblivious invariant).
//
// The port is line-for-line with sna_engine.cpp's lines 233-364 (build_indexlist),
// 1101-1148 (init_clebsch_gordan) and 1153-1159 (init_rootpqarray), just with
// LAMMPS' nested-pointer arrays replaced by flat std::vector storage. Since
// these are pure integer / double recurrences with no FP-accumulation subtlety,
// byte-exactness with the CPU oracle follows by construction.
//
// -----------------------------------------------------------------------------
// LAMMPS GPLv2 ATTRIBUTION (preserved verbatim from sna.cpp upstream header):
//
//   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
//   https://www.lammps.org/, Sandia National Laboratories
//   LAMMPS development team: developers@lammps.org
//
//   Copyright (2003) Sandia Corporation. Under the terms of Contract
//   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
//   certain rights in this software. This software is distributed under
//   the GNU General Public License.
//
//   Contributing authors: Aidan Thompson, Christian Trott, SNL
// -----------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <vector>

namespace tdmd::gpu::snap_detail {

// SnaZIndices packed as 9 ints in contiguous memory (host layout; same on
// device). Matches the CPU `SnaZIndices` struct: j1, j2, j, ma1min, ma2max,
// mb1min, mb2max, na, nb, jju — we store 10 ints per entry (matches struct
// alignment). Kernel reads via `d_idxz + jjz * 10`.
inline constexpr int kSnaZIndicesStride = 10;

// idxb packed as 3 ints per entry (j1, j2, j). Kernel reads via
// `d_idxb + jjb * 3`.
inline constexpr int kSnaBIndicesStride = 3;

// Host-side flat tables. All vectors own their storage. No device pointers
// here; uploading is the caller's responsibility (SnapGpu::Impl ctor).
struct FlatIndexTables {
  int twojmax = 0;
  int jdim = 0;    // == twojmax + 1
  int jdimpq = 0;  // == twojmax + 2

  // Derived maxima. Same semantics as SnaEngine::idxu_max_ / idxz_max_ / etc.
  int idxu_max = 0;
  int idxz_max = 0;
  int idxb_max = 0;
  int idxcg_max = 0;

  // Flat 1D index arrays.
  //   idxu_block[j]         — size jdim        = twojmax + 1
  //   idxcg_block[j1,j2,j]  — size jdim³       (flat: j1*jdim² + j2*jdim + j)
  //   idxz_block[j1,j2,j]   — size jdim³
  //   idxb_block[j1,j2,j]   — size jdim³
  std::vector<int> idxu_block;
  std::vector<int> idxcg_block;
  std::vector<int> idxz_block;
  std::vector<int> idxb_block;

  // Packed struct arrays:
  //   idxz_packed  — idxz_max * 10 ints (SnaZIndices record)
  //   idxb_packed  — idxb_max * 3 ints  (SnaBIndices record)
  std::vector<int> idxz_packed;
  std::vector<int> idxb_packed;

  // Clebsch-Gordan coefficients and Wigner normalisation table.
  //   cglist   — size idxcg_max doubles
  //   rootpq   — size jdimpq² doubles (flat row-major; rootpq[p*jdimpq + q])
  std::vector<double> cglist;
  std::vector<double> rootpq;

  // T-opt-2 (yi_kernel Phase B parallelization): per-jju CSR bucket of jjz
  // indices. Each bucket lists the jjz values that map to this jju
  // (= idxz_packed[jjz*10 + 9]) in ascending jjz order. With this, Phase B
  // can be parallelized as tid-strided over jju (each thread owns one jju's
  // += sequence) — and the ascending-jjz ordering inside each bucket matches
  // the legacy single-lane loop's accumulation order bit-for-bit.
  //   idxz_jju_bucket_begin — size idxu_max + 1 (CSR prefix sum)
  //   idxz_by_jju           — size idxz_max (sorted jjz values)
  std::vector<int> idxz_jju_bucket_begin;
  std::vector<int> idxz_by_jju;
};

// Build a complete FlatIndexTables instance for the given twojmax. Pure host
// function; no CUDA. Throws std::invalid_argument if twojmax is negative or
// odd.
FlatIndexTables build_flat_tables(int twojmax);

}  // namespace tdmd::gpu::snap_detail
