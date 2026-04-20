// SPEC: docs/specs/gpu/SPEC.md §7.5 (SNAP GPU — T8.6b), §8.1 (FP64-only),
//       §D.16 (__restrict__ discipline)
// Module SPEC: docs/specs/potentials/SPEC.md §6 (SNAP)
// Exec pack: docs/development/m8_execution_pack.md T8.6b
//
// Device-side SNAP helpers. Port of SnaEngine's per-neighbour primitives:
//
//   compute_uarray_device   — Wigner U-function recurrence (sna.cpp 835-896)
//   compute_duarray_device  — Wigner U-derivative recurrence (sna.cpp 901-1038)
//   compute_sfac_device     — smoothing function (sna.cpp 1189-1213)
//   compute_dsfac_device    — smoothing function derivative (sna.cpp 1215-1257)
//
// The port preserves loop order, FP operand ordering, and array access patterns
// 1:1 with the CPU oracle so Fp64Reference GPU is byte-exact with CPU at the
// 1e-12 relative gate (exercised at T8.7 D-M8-13).
//
// Runtime assumptions (M8 scope — enforced at SnapGpuAdapter ctor):
//   chemflag = 0            ⇒ nelements = 1, elem_duarray = 0
//   switchinnerflag = 0     ⇒ sfac depends only on (r, rcut, rmin0)
//   quadraticflag = 0       ⇒ β coefficients are linear-only
//
// switchflag, bzeroflag, bnormflag, wselfallflag remain runtime-configurable.
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 (M_PI * 0.5)
#endif

#include <cuda_runtime.h>

namespace tdmd::gpu::snap_detail {

// --------------------------------------------------------------------------
// compute_sfac_device — smooth cutoff. Switch-inner path is not reachable in
// M8 (adapter rejects switchinnerflag=1); we still accept the arg for
// signature parity with CPU.
// --------------------------------------------------------------------------
__device__ __forceinline__ double compute_sfac_device(double r,
                                                      double rcut,
                                                      double rmin0,
                                                      int switch_flag) {
  double sfac;
  if (switch_flag == 0) {
    sfac = 1.0;
  } else if (r <= rmin0) {
    sfac = 1.0;
  } else if (r > rcut) {
    sfac = 0.0;
  } else {
    const double rcutfac_local = M_PI / (rcut - rmin0);
    sfac = 0.5 * (cos((r - rmin0) * rcutfac_local) + 1.0);
  }
  return sfac;
}

__device__ __forceinline__ double compute_dsfac_device(double r,
                                                       double rcut,
                                                       double rmin0,
                                                       int switch_flag) {
  double dsfac;
  if (switch_flag == 0) {
    dsfac = 0.0;
  } else if (r <= rmin0) {
    dsfac = 0.0;
  } else if (r > rcut) {
    dsfac = 0.0;
  } else {
    const double rcutfac_local = M_PI / (rcut - rmin0);
    dsfac = -0.5 * sin((r - rmin0) * rcutfac_local) * rcutfac_local;
  }
  return dsfac;
}

// --------------------------------------------------------------------------
// compute_uarray_device — writes ulist_r/i into caller-provided scratch of
// length idxu_max. Mirrors SnaEngine::compute_uarray exactly (same loop
// order, same operand order). Caller must ensure ulist_r/ulist_i have
// idxu_max elements.
//
// rootpq is the flattened rootpqarray stored row-major as a
// (jdimpq × jdimpq) double array; rootpq[p * jdimpq + q] == √(p/q).
// idxu_block is the flat int[twojmax+1] array from FlatIndexTables.
// --------------------------------------------------------------------------
__device__ __forceinline__ void compute_uarray_device(double x,
                                                      double y,
                                                      double z,
                                                      double z0,
                                                      double r,
                                                      const double* __restrict__ rootpq,
                                                      int jdimpq,
                                                      const int* __restrict__ idxu_block,
                                                      int twojmax,
                                                      double* __restrict__ ulist_r,
                                                      double* __restrict__ ulist_i) {
  const double r0inv = 1.0 / sqrt(r * r + z0 * z0);
  const double a_r = r0inv * z0;
  const double a_i = -r0inv * z;
  const double b_r = r0inv * y;
  const double b_i = -r0inv * x;

  ulist_r[0] = 1.0;
  ulist_i[0] = 0.0;

  for (int j = 1; j <= twojmax; ++j) {
    int jju = idxu_block[j];
    int jjup = idxu_block[j - 1];

    // fill in left side of matrix layer from previous layer.
    for (int mb = 0; 2 * mb <= j; ++mb) {
      ulist_r[jju] = 0.0;
      ulist_i[jju] = 0.0;

      for (int ma = 0; ma < j; ++ma) {
        double rootpq_v = rootpq[(j - ma) * jdimpq + (j - mb)];
        ulist_r[jju] += rootpq_v * (a_r * ulist_r[jjup] + a_i * ulist_i[jjup]);
        ulist_i[jju] += rootpq_v * (a_r * ulist_i[jjup] - a_i * ulist_r[jjup]);

        rootpq_v = rootpq[(ma + 1) * jdimpq + (j - mb)];
        ulist_r[jju + 1] = -rootpq_v * (b_r * ulist_r[jjup] + b_i * ulist_i[jjup]);
        ulist_i[jju + 1] = -rootpq_v * (b_r * ulist_i[jjup] - b_i * ulist_r[jjup]);
        jju++;
        jjup++;
      }
      jju++;
    }

    // copy left side to right side with inversion symmetry VMK 4.4(2).
    jju = idxu_block[j];
    jjup = jju + (j + 1) * (j + 1) - 1;
    int mbpar = 1;
    for (int mb = 0; 2 * mb <= j; ++mb) {
      int mapar = mbpar;
      for (int ma = 0; ma <= j; ++ma) {
        if (mapar == 1) {
          ulist_r[jjup] = ulist_r[jju];
          ulist_i[jjup] = -ulist_i[jju];
        } else {
          ulist_r[jjup] = -ulist_r[jju];
          ulist_i[jjup] = ulist_i[jju];
        }
        mapar = -mapar;
        jju++;
        jjup--;
      }
      mbpar = -mbpar;
    }
  }
}

// --------------------------------------------------------------------------
// compute_duarray_device — writes dulist_r/i flat as [u * 3 + k] into scratch
// of size idxu_max × 3. Mirrors SnaEngine::compute_duarray. Requires ulist_r/i
// already populated (same r, z0) by a prior compute_uarray_device call.
// Applies sfac/dsfac scaling at the end (so output is the fully-scaled
// per-neighbour dU/dr · wj, consumable directly by compute_deidrj).
// --------------------------------------------------------------------------
__device__ __forceinline__ void compute_duarray_device(
    double x,
    double y,
    double z,
    double z0,
    double r,
    double dz0dr,
    double wj_local,
    double rcut,
    double rmin0,
    int switch_flag,
    const double* __restrict__ rootpq,
    int jdimpq,
    const int* __restrict__ idxu_block,
    int twojmax,
    const double* __restrict__ ulist_r,
    const double* __restrict__ ulist_i,
    double* __restrict__ dulist_r,  // [idxu_max × 3]
    double* __restrict__ dulist_i) {
  double a_r, a_i, b_r, b_i;
  double da_r[3], da_i[3], db_r[3], db_i[3];
  double dz0[3], dr0inv[3], dr0invdr;
  double rootpq_v;

  const double rinv = 1.0 / r;
  const double ux = x * rinv;
  const double uy = y * rinv;
  const double uz = z * rinv;

  const double r0inv = 1.0 / sqrt(r * r + z0 * z0);
  a_r = z0 * r0inv;
  a_i = -z * r0inv;
  b_r = y * r0inv;
  b_i = -x * r0inv;

  dr0invdr = -pow(r0inv, 3.0) * (r + z0 * dz0dr);

  dr0inv[0] = dr0invdr * ux;
  dr0inv[1] = dr0invdr * uy;
  dr0inv[2] = dr0invdr * uz;

  dz0[0] = dz0dr * ux;
  dz0[1] = dz0dr * uy;
  dz0[2] = dz0dr * uz;

  for (int k = 0; k < 3; ++k) {
    da_r[k] = dz0[k] * r0inv + z0 * dr0inv[k];
    da_i[k] = -z * dr0inv[k];
  }
  da_i[2] += -r0inv;

  for (int k = 0; k < 3; ++k) {
    db_r[k] = y * dr0inv[k];
    db_i[k] = -x * dr0inv[k];
  }
  db_i[0] += -r0inv;
  db_r[1] += r0inv;

  dulist_r[0 * 3 + 0] = 0.0;
  dulist_r[0 * 3 + 1] = 0.0;
  dulist_r[0 * 3 + 2] = 0.0;
  dulist_i[0 * 3 + 0] = 0.0;
  dulist_i[0 * 3 + 1] = 0.0;
  dulist_i[0 * 3 + 2] = 0.0;

  for (int j = 1; j <= twojmax; ++j) {
    int jju = idxu_block[j];
    int jjup = idxu_block[j - 1];
    for (int mb = 0; 2 * mb <= j; ++mb) {
      dulist_r[jju * 3 + 0] = 0.0;
      dulist_r[jju * 3 + 1] = 0.0;
      dulist_r[jju * 3 + 2] = 0.0;
      dulist_i[jju * 3 + 0] = 0.0;
      dulist_i[jju * 3 + 1] = 0.0;
      dulist_i[jju * 3 + 2] = 0.0;

      for (int ma = 0; ma < j; ++ma) {
        rootpq_v = rootpq[(j - ma) * jdimpq + (j - mb)];
        for (int k = 0; k < 3; ++k) {
          dulist_r[jju * 3 + k] +=
              rootpq_v * (da_r[k] * ulist_r[jjup] + da_i[k] * ulist_i[jjup] +
                          a_r * dulist_r[jjup * 3 + k] + a_i * dulist_i[jjup * 3 + k]);
          dulist_i[jju * 3 + k] +=
              rootpq_v * (da_r[k] * ulist_i[jjup] - da_i[k] * ulist_r[jjup] +
                          a_r * dulist_i[jjup * 3 + k] - a_i * dulist_r[jjup * 3 + k]);
        }

        rootpq_v = rootpq[(ma + 1) * jdimpq + (j - mb)];
        for (int k = 0; k < 3; ++k) {
          dulist_r[(jju + 1) * 3 + k] =
              -rootpq_v * (db_r[k] * ulist_r[jjup] + db_i[k] * ulist_i[jjup] +
                           b_r * dulist_r[jjup * 3 + k] + b_i * dulist_i[jjup * 3 + k]);
          dulist_i[(jju + 1) * 3 + k] =
              -rootpq_v * (db_r[k] * ulist_i[jjup] - db_i[k] * ulist_r[jjup] +
                           b_r * dulist_i[jjup * 3 + k] - b_i * dulist_r[jjup * 3 + k]);
        }
        jju++;
        jjup++;
      }
      jju++;
    }

    // copy left side to right side with inversion symmetry VMK 4.4(2).
    jju = idxu_block[j];
    jjup = jju + (j + 1) * (j + 1) - 1;
    int mbpar = 1;
    for (int mb = 0; 2 * mb <= j; ++mb) {
      int mapar = mbpar;
      for (int ma = 0; ma <= j; ++ma) {
        if (mapar == 1) {
          for (int k = 0; k < 3; ++k) {
            dulist_r[jjup * 3 + k] = dulist_r[jju * 3 + k];
            dulist_i[jjup * 3 + k] = -dulist_i[jju * 3 + k];
          }
        } else {
          for (int k = 0; k < 3; ++k) {
            dulist_r[jjup * 3 + k] = -dulist_r[jju * 3 + k];
            dulist_i[jjup * 3 + k] = dulist_i[jju * 3 + k];
          }
        }
        mapar = -mapar;
        jju++;
        jjup--;
      }
      mbpar = -mbpar;
    }
  }

  double sfac = compute_sfac_device(r, rcut, rmin0, switch_flag);
  double dsfac = compute_dsfac_device(r, rcut, rmin0, switch_flag);

  sfac *= wj_local;
  dsfac *= wj_local;
  for (int j = 0; j <= twojmax; ++j) {
    int jju = idxu_block[j];
    for (int mb = 0; 2 * mb <= j; ++mb) {
      for (int ma = 0; ma <= j; ++ma) {
        dulist_r[jju * 3 + 0] = dsfac * ulist_r[jju] * ux + sfac * dulist_r[jju * 3 + 0];
        dulist_i[jju * 3 + 0] = dsfac * ulist_i[jju] * ux + sfac * dulist_i[jju * 3 + 0];
        dulist_r[jju * 3 + 1] = dsfac * ulist_r[jju] * uy + sfac * dulist_r[jju * 3 + 1];
        dulist_i[jju * 3 + 1] = dsfac * ulist_i[jju] * uy + sfac * dulist_i[jju * 3 + 1];
        dulist_r[jju * 3 + 2] = dsfac * ulist_r[jju] * uz + sfac * dulist_r[jju * 3 + 2];
        dulist_i[jju * 3 + 2] = dsfac * ulist_i[jju] * uz + sfac * dulist_i[jju * 3 + 2];
        jju++;
      }
    }
  }
}

}  // namespace tdmd::gpu::snap_detail
