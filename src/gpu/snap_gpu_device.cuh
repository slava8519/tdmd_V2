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

// ==========================================================================
// T8.6c-v4: block-parallel variants of compute_uarray / compute_duarray.
//
// The sequential helpers above run single-lane in snap_ui_kernel and
// snap_deidrj_kernel (which deidrj is ~61.5% of the kernel time). Insight:
// within each outer-j layer, the "fill left side" inner loop writes each
// position P exactly twice — once via `= -rootpq_k·b-term[jjup_init+k-1]`
// (the boundary-write at iteration ma=k-1) and once via `+= rootpq_{j-k}·a-
// term[jjup_init+k]` (the accumulate at iteration ma=k). Both b- and a-terms
// read from layer (j-1) only, not from the current layer. So for each P we
// can compute `v_b + v_a` independently in parallel, with the outer-j loop
// still sequential (synced via __syncthreads between layers to satisfy the
// RAW across layers).
//
// Byte-exactness (D-M8-13, 1e-12 rel):
//   - Expression structure preserved: `real = v_b; real += v_a` matches the
//     original `ulist[P] = v_b; ulist[P] += v_a` write sequence.
//   - For k=0: only a-contribution, `real = 0.0; real += v_a` = `0.0 + v_a`
//     which is bit-identical to v_a.
//   - For k=j: only b-contribution, `real = 0.0; real = v_b` bit-identical.
//   - For 0 < k < j: `real = v_b; real += v_a` = `v_b + v_a` bit-identical
//     to the sequential two-step write.
//   - rootpq indexing and (b_r·u + b_i·ui) parenthesization preserved. No
//     new FMA contraction opportunities; compiler sees the same expression
//     text and makes the same --fmad decision.
//
// The symmetrize (inversion-symmetry copy) loop is also parallelized — each
// (mb, ma) thread writes to a distinct jjup. Byte-exact since copies + sign.
// ==========================================================================

// Parallel compute_uarray. All 128 threads of the block participate. `tid` and
// `stride` = (threadIdx.x, blockDim.x) — caller passes these so the helper
// stays signature-stable if block geometry changes.
__device__ __forceinline__ void compute_uarray_parallel_device(unsigned tid,
                                                               unsigned stride,
                                                               double x,
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
  // Per-thread recomputed scalars (pure arithmetic; bit-identical across
  // threads). Cheap enough to avoid shared-memory traffic.
  const double r0inv = 1.0 / sqrt(r * r + z0 * z0);
  const double a_r = r0inv * z0;
  const double a_i = -r0inv * z;
  const double b_r = r0inv * y;
  const double b_i = -r0inv * x;

  if (tid == 0) {
    ulist_r[0] = 1.0;
    ulist_i[0] = 0.0;
  }

  for (int j = 1; j <= twojmax; ++j) {
    __syncthreads();  // previous layer visible
    const int base_j = idxu_block[j];
    const int base_jj = idxu_block[j - 1];
    const int jlo = j + 1;
    const int mb_count = (j / 2) + 1;
    const int total_fill = jlo * mb_count;

    for (int linear = static_cast<int>(tid); linear < total_fill;
         linear += static_cast<int>(stride)) {
      const int mb = linear / jlo;
      const int k = linear % jlo;
      const int P = base_j + mb * jlo + k;
      const int jjup_base = base_jj + mb * j;

      double real = 0.0;
      double imag = 0.0;

      // "b" contribution from ma=k-1: `= -rootpq[k,(j-mb)] * (b·u_{k-1})`.
      if (k >= 1) {
        const double rootpq_v = rootpq[k * jdimpq + (j - mb)];
        const int jjup = jjup_base + (k - 1);
        real = -rootpq_v * (b_r * ulist_r[jjup] + b_i * ulist_i[jjup]);
        imag = -rootpq_v * (b_r * ulist_i[jjup] - b_i * ulist_r[jjup]);
      }
      // "a" contribution from ma=k: `+= rootpq[j-k,(j-mb)] * (a·u_k)`.
      if (k < j) {
        const double rootpq_v = rootpq[(j - k) * jdimpq + (j - mb)];
        const int jjup = jjup_base + k;
        real += rootpq_v * (a_r * ulist_r[jjup] + a_i * ulist_i[jjup]);
        imag += rootpq_v * (a_r * ulist_i[jjup] - a_i * ulist_r[jjup]);
      }

      ulist_r[P] = real;
      ulist_i[P] = imag;
    }

    __syncthreads();  // fills visible before symmetrize reads

    // Symmetrize (VMK 4.4(2) inversion symmetry): copy left → right with sign.
    const int total_sym = mb_count * jlo;
    for (int linear = static_cast<int>(tid); linear < total_sym;
         linear += static_cast<int>(stride)) {
      const int mb = linear / jlo;
      const int ma = linear % jlo;
      const int jju = base_j + linear;
      const int jjup = base_j + jlo * jlo - 1 - linear;
      const int mapar_is_plus = (((mb + ma) & 1) == 0);
      if (mapar_is_plus) {
        ulist_r[jjup] = ulist_r[jju];
        ulist_i[jjup] = -ulist_i[jju];
      } else {
        ulist_r[jjup] = -ulist_r[jju];
        ulist_i[jjup] = ulist_i[jju];
      }
    }
  }
  __syncthreads();  // finalize: all threads see complete ulist
}

// Parallel compute_duarray. Same layer-sequential + (mb, k) parallel shape.
// Vector k-index (x/y/z) is handled inside the per-position body.
__device__ __forceinline__ void compute_duarray_parallel_device(unsigned tid,
                                                                unsigned stride,
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
                                                                double* __restrict__ dulist_r,
                                                                double* __restrict__ dulist_i) {
  // Recompute per-thread (bit-identical across threads).
  const double rinv = 1.0 / r;
  const double ux = x * rinv;
  const double uy = y * rinv;
  const double uz = z * rinv;

  const double r0inv = 1.0 / sqrt(r * r + z0 * z0);
  const double a_r = z0 * r0inv;
  const double a_i = -z * r0inv;
  const double b_r = y * r0inv;
  const double b_i = -x * r0inv;

  const double dr0invdr = -pow(r0inv, 3.0) * (r + z0 * dz0dr);

  double dr0inv[3];
  double dz0[3];
  dr0inv[0] = dr0invdr * ux;
  dr0inv[1] = dr0invdr * uy;
  dr0inv[2] = dr0invdr * uz;
  dz0[0] = dz0dr * ux;
  dz0[1] = dz0dr * uy;
  dz0[2] = dz0dr * uz;

  double da_r[3], da_i[3], db_r[3], db_i[3];
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

  if (tid == 0) {
    dulist_r[0 * 3 + 0] = 0.0;
    dulist_r[0 * 3 + 1] = 0.0;
    dulist_r[0 * 3 + 2] = 0.0;
    dulist_i[0 * 3 + 0] = 0.0;
    dulist_i[0 * 3 + 1] = 0.0;
    dulist_i[0 * 3 + 2] = 0.0;
  }

  for (int j = 1; j <= twojmax; ++j) {
    __syncthreads();
    const int base_j = idxu_block[j];
    const int base_jj = idxu_block[j - 1];
    const int jlo = j + 1;
    const int mb_count = (j / 2) + 1;
    const int total_fill = jlo * mb_count;

    for (int linear = static_cast<int>(tid); linear < total_fill;
         linear += static_cast<int>(stride)) {
      const int mb = linear / jlo;
      const int kk = linear % jlo;
      const int P = base_j + mb * jlo + kk;
      const int jjup_base = base_jj + mb * j;

      double out_r[3] = {0.0, 0.0, 0.0};
      double out_i[3] = {0.0, 0.0, 0.0};

      // "b" contribution from ma=kk-1 (boundary-write `= -rootpq·b-term`).
      if (kk >= 1) {
        const double rootpq_v = rootpq[kk * jdimpq + (j - mb)];
        const int jjup = jjup_base + (kk - 1);
        const double u_r = ulist_r[jjup];
        const double u_i = ulist_i[jjup];
        for (int k = 0; k < 3; ++k) {
          const double du_r = dulist_r[jjup * 3 + k];
          const double du_i = dulist_i[jjup * 3 + k];
          out_r[k] = -rootpq_v * (db_r[k] * u_r + db_i[k] * u_i + b_r * du_r + b_i * du_i);
          out_i[k] = -rootpq_v * (db_r[k] * u_i - db_i[k] * u_r + b_r * du_i - b_i * du_r);
        }
      }
      // "a" contribution from ma=kk (accumulate `+= rootpq·a-term`).
      if (kk < j) {
        const double rootpq_v = rootpq[(j - kk) * jdimpq + (j - mb)];
        const int jjup = jjup_base + kk;
        const double u_r = ulist_r[jjup];
        const double u_i = ulist_i[jjup];
        for (int k = 0; k < 3; ++k) {
          const double du_r = dulist_r[jjup * 3 + k];
          const double du_i = dulist_i[jjup * 3 + k];
          out_r[k] += rootpq_v * (da_r[k] * u_r + da_i[k] * u_i + a_r * du_r + a_i * du_i);
          out_i[k] += rootpq_v * (da_r[k] * u_i - da_i[k] * u_r + a_r * du_i - a_i * du_r);
        }
      }

      dulist_r[P * 3 + 0] = out_r[0];
      dulist_r[P * 3 + 1] = out_r[1];
      dulist_r[P * 3 + 2] = out_r[2];
      dulist_i[P * 3 + 0] = out_i[0];
      dulist_i[P * 3 + 1] = out_i[1];
      dulist_i[P * 3 + 2] = out_i[2];
    }

    __syncthreads();

    const int total_sym = mb_count * jlo;
    for (int linear = static_cast<int>(tid); linear < total_sym;
         linear += static_cast<int>(stride)) {
      const int mb = linear / jlo;
      const int ma = linear % jlo;
      const int jju = base_j + linear;
      const int jjup = base_j + jlo * jlo - 1 - linear;
      const int mapar_is_plus = (((mb + ma) & 1) == 0);
      if (mapar_is_plus) {
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
    }
  }

  __syncthreads();

  // Parallel sfac/dsfac scaling. All threads compute sfac/dsfac (pure
  // arithmetic, bit-identical); then stride over (j, mb, ma) to scale.
  double sfac = compute_sfac_device(r, rcut, rmin0, switch_flag);
  double dsfac = compute_dsfac_device(r, rcut, rmin0, switch_flag);
  sfac *= wj_local;
  dsfac *= wj_local;

  for (int j = 0; j <= twojmax; ++j) {
    const int base_j = idxu_block[j];
    const int jlo = j + 1;
    const int mb_count = (j / 2) + 1;
    const int total = jlo * mb_count;
    for (int linear = static_cast<int>(tid); linear < total; linear += static_cast<int>(stride)) {
      const int mb = linear / jlo;
      const int ma = linear % jlo;
      const int jju = base_j + mb * jlo + ma;
      const double ur = ulist_r[jju];
      const double ui = ulist_i[jju];
      const double dr0 = dulist_r[jju * 3 + 0];
      const double dr1 = dulist_r[jju * 3 + 1];
      const double dr2 = dulist_r[jju * 3 + 2];
      const double di0 = dulist_i[jju * 3 + 0];
      const double di1 = dulist_i[jju * 3 + 1];
      const double di2 = dulist_i[jju * 3 + 2];
      dulist_r[jju * 3 + 0] = dsfac * ur * ux + sfac * dr0;
      dulist_i[jju * 3 + 0] = dsfac * ui * ux + sfac * di0;
      dulist_r[jju * 3 + 1] = dsfac * ur * uy + sfac * dr1;
      dulist_i[jju * 3 + 1] = dsfac * ui * uy + sfac * di1;
      dulist_r[jju * 3 + 2] = dsfac * ur * uz + sfac * dr2;
      dulist_i[jju * 3 + 2] = dsfac * ui * uz + sfac * di2;
    }
  }
  __syncthreads();
}

}  // namespace tdmd::gpu::snap_detail
