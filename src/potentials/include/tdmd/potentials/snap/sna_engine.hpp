#pragma once

// -----------------------------------------------------------------------------
// SnaEngine — verbatim port of LAMMPS USER-SNAP `SNA` class (ML-SNAP package).
//
//   Original: verify/third_party/lammps/src/ML-SNAP/sna.h + sna.cpp
//   Upstream authors: Aidan Thompson, Christian Trott (Sandia National Labs)
//   Upstream licence: GNU General Public License v2 (GPLv2) — see below.
//
// The port is structural, preserving LAMMPS member names, loop ordering,
// floating-point accumulation sequence, and array layout. The only changes
// are to LAMMPS infrastructure (Pointers base, memory->create/destroy,
// error->one/all, MathConst/MathSpecial namespaces) — replaced with plain
// `new[]/delete[]`, `std::runtime_error`, and M_PI constants. This
// structural fidelity is load-bearing for the M8 D-M8-7 byte-exact
// differential (TDMD Fp64Reference ≡ LAMMPS FP64 ≤ 1e-12 rel). Any
// reorganization of inner-loop accumulation breaks the contract.
//
// SPEC: docs/specs/potentials/SPEC.md §6 (SNAP). Exec pack:
// docs/development/m8_execution_pack.md T8.4a (types/parser, shipped) +
// T8.4b (this file — force body port).
//
// -----------------------------------------------------------------------------
// LAMMPS GPLv2 ATTRIBUTION (preserved verbatim from sna.h upstream header):
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

#include <cstddef>

namespace tdmd::snap_detail {

// VMK half-integer → array-offset index structs (LAMMPS names preserved
// verbatim — see sna.h upstream).
struct SnaZIndices {
  int j1, j2, j, ma1min, ma2max, mb1min, mb2max, na, nb, jju;
};

struct SnaBIndices {
  int j1, j2, j;
};

// The engine holds all per-atom scratch (rij, wj, ...) plus the
// twojmax-derived index blocks (idxcg_block, idxz, idxb, ...). The caller
// fills `rij[jj]` / `wj[jj]` / `rcutij[jj]` / `inside[jj]` for jj ∈ [0, jnum)
// and then drives compute_ui → compute_yi → per-neighbour
// {compute_duidrj, compute_deidrj} per pair_snap.cpp outer loop.
//
// Non-copyable, non-movable: owns raw buffers and internal pointer arithmetic
// references them by address; moving would require reconstructing index
// pointers. `SnapPotential` holds one instance inside a
// `std::unique_ptr<SnaEngine>`.
class SnaEngine {
public:
  // Constructor arguments match LAMMPS `SNA::SNA(LAMMPS*, rfac0, twojmax,
  // rmin0, switchflag, bzeroflag, chemflag, bnormflag, wselfallflag,
  // nelements, switchinnerflag)` minus the LAMMPS* pointer — we keep the
  // same order so a reviewer can cross-check 1:1 against pair_snap.cpp
  // line 410.
  SnaEngine(double rfac0_in,
            int twojmax_in,
            double rmin0_in,
            int switch_flag_in,
            int bzero_flag_in,
            int chem_flag_in,
            int bnorm_flag_in,
            int wselfall_flag_in,
            int nelements_in,
            int switch_inner_flag_in);
  ~SnaEngine();

  SnaEngine(const SnaEngine&) = delete;
  SnaEngine& operator=(const SnaEngine&) = delete;
  SnaEngine(SnaEngine&&) = delete;
  SnaEngine& operator=(SnaEngine&&) = delete;

  void build_indexlist();
  void init();

  int ncoeff = 0;  // number of bispectrum components per atom per triple.

  // Bispectrum pipeline (called in the order Ui → Zi → Bi → Yi → deidrj).
  void compute_ui(int jnum, int ielem);
  void compute_zi();
  void compute_yi(const double* beta);
  void compute_bi(int ielem);

  // Per-neighbour derivative pipeline.
  void compute_duidrj(int jj);
  void compute_deidrj(double* dedr);

  // Smoothing function (outer + optional inner cosine cutoff).
  [[nodiscard]] double compute_sfac(double r, double rcut, double sinner, double dinner) const;
  [[nodiscard]] double compute_dsfac(double r, double rcut, double sinner, double dinner) const;

  // Grow per-neighbour scratch arrays to at least `newnmax`. Idempotent on
  // shrink. Owned internal; SnapPotential calls before the per-atom loop.
  void grow_rij(int newnmax);
  [[nodiscard]] int nmax_capacity() const noexcept { return nmax_; }

  // Public bispectrum outputs.
  double* blist = nullptr;    // size = idxb_max * ntriples
  double** dblist = nullptr;  // dblist[itriple*idxb_max + jjb][k], k ∈ {0,1,2}

  // Per-neighbour scratch (public because pair_snap.cpp fills them directly;
  // we match that pattern for byte-exactness).
  double** rij = nullptr;      // rij[nmax][3]
  int* inside = nullptr;       // inside[nmax]
  double* wj = nullptr;        // wj[nmax]
  double* rcutij = nullptr;    // rcutij[nmax]
  double* sinnerij = nullptr;  // sinnerij[nmax]
  double* dinnerij = nullptr;  // dinnerij[nmax]
  int* element = nullptr;      // element[nmax] (chem_flag only)

  [[nodiscard]] int twojmax() const noexcept { return twojmax_; }
  [[nodiscard]] int idxb_max() const noexcept { return idxb_max_; }
  [[nodiscard]] int ntriples() const noexcept { return ntriples_; }

private:
  // Scalar hyperparameters (LAMMPS names preserved).
  double rfac0_, rmin0_;
  int twojmax_;
  int switch_flag_, switch_inner_flag_;
  int bzero_flag_, chem_flag_, bnorm_flag_, wselfall_flag_;
  double wself_ = 1.0;
  int nelements_;
  int ndoubles_;
  int ntriples_;

  // Derived index-block sizes.
  int idxcg_max_ = 0;
  int idxu_max_ = 0;
  int idxz_max_ = 0;
  int idxb_max_ = 0;

  // Index blocks (allocated by build_indexlist).
  int*** idxcg_block = nullptr;  // [jdim][jdim][jdim]
  int* idxu_block = nullptr;     // [jdim]
  int*** idxz_block = nullptr;   // [jdim][jdim][jdim]
  int*** idxb_block = nullptr;   // [jdim][jdim][jdim]
  SnaZIndices* idxz = nullptr;   // [idxz_max]
  SnaBIndices* idxb = nullptr;   // [idxb_max]

  // Rotation / bispectrum tables.
  double** rootpqarray = nullptr;  // [jdimpq][jdimpq]
  double* cglist = nullptr;        // [idxcg_max]

  // Wigner U-function buffers.
  double** ulist_r_ij = nullptr;  // [nmax][idxu_max]
  double** ulist_i_ij = nullptr;  // [nmax][idxu_max]
  double* ulisttot_r = nullptr;   // [idxu_max * nelements]
  double* ulisttot_i = nullptr;   // [idxu_max * nelements]
  double** dulist_r = nullptr;    // [idxu_max][3]
  double** dulist_i = nullptr;    // [idxu_max][3]

  // Z / Y buffers.
  double* zlist_r = nullptr;  // [idxz_max * ndoubles]
  double* zlist_i = nullptr;
  double* ylist_r = nullptr;  // [idxu_max * nelements]
  double* ylist_i = nullptr;

  // bzero[j] — reference bispectrum subtracted when bzero_flag=1.
  double* bzero = nullptr;

  // compute_duidrj sets elem_duarray before calling compute_duarray; it is
  // read by compute_deidrj / compute_dbidrj. Matches LAMMPS semantics.
  int elem_duarray_ = 0;

  int nmax_ = 0;  // allocated capacity of per-neighbour scratch.

  // Internal helpers (all verbatim from sna.cpp).
  void create_twojmax_arrays();
  void destroy_twojmax_arrays();
  void init_clebsch_gordan();
  void init_rootpqarray();
  void zero_uarraytot(int ielem);
  void add_uarraytot(double r, int jj);
  void compute_uarray(double x, double y, double z, double z0, double r, int jj);
  [[nodiscard]] double deltacg(int j1, int j2, int j) const;
  void compute_ncoeff();
  void compute_duarray(double x,
                       double y,
                       double z,
                       double z0,
                       double r,
                       double dz0dr,
                       double wj_local,
                       double rcut,
                       int jj);
};

}  // namespace tdmd::snap_detail
