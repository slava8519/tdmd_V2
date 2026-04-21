// SPEC: docs/specs/gpu/SPEC.md §7.5 (SNAP GPU contract reused), §8 (MixedFast
//       policy), §8.3 (D-M8-8 dense-cutoff thresholds), §9 (NVTX every launch)
// Module SPEC: docs/specs/potentials/SPEC.md §6 (SNAP module contract)
// Master spec: §D.1 Philosophy B (FP32 math + FP64 accumulators),
//              §D.11 MixedFastSnapOnlyBuild flavor exception
// Exec pack: docs/development/m8_execution_pack.md T8.9
// Pre-impl:  docs/development/t8.9_pre_impl.md
// Decisions: D-M6-17 (PIMPL firewall), D-M8-4 (SnapOnly flavor uses FP32 SNAP /
//            FP64 EAM), D-M8-8 (≤1e-5 rel force / ≤1e-7 rel PE dense-cutoff
//            thresholds), D-M6-15 (host-side Kahan reductions)
//
// T8.9 Phase A body — Philosophy B narrow FP32 at pair-math only. Mirrors
// `snap_gpu.cu` structure exactly; the ONLY difference from the FP64 sibling
// is `r = sqrtf(rsq)` in the per-pair hot path inside the `ui` and `deidrj`
// kernels. Everything else (theta0, z0, dz0dr, sfac, dsfac) stays FP64.
//
// Scope evolution — this started as a wider FP32 scope (also sincosf, z0,
// sfac, dsfac). Empirical measurement (2000-atom rattled BCC-W, twojmax=8)
// showed force worst rel of 2.1e-5 — above the D-M8-8 1e-5 cap. The root
// cause: FP32 `__sincosf` carries ~1 ULP rel error (~6e-8) which feeds `z0`,
// and `z0` enters the U-recurrence multiplicatively. On a length-O(165)
// bispectrum dot product, that error compounds to ~2e-5 on force. Narrowing
// to `sqrtf` alone keeps `r`'s rel error at ~6e-8 but downstream scalars are
// all FP64, so the compounding is bounded — measured 9.5e-6 force worst
// (well within D-M8-8 1e-5 budget, PE rel 1.7e-9, virial 3.9e-9).
//
// FP32 site (a single scalar op):
//   - r = sqrtf(static_cast<float>(rsq))  — one op per pair in ui + deidrj.
//
// FP64 sites (verbatim from snap_gpu.cu; not listed exhaustively):
//   - theta0, z0, dz0dr, sfac, sincos, all subsequent scalar arithmetic
//   - all ulist/ulisttot/zlist/ylist/dulist/blist arrays
//   - compute_uarray_device, compute_duarray_device
//   - compute_deidrj_device dot product
//   - per-atom fx/fy/fz/pe/virial accumulators
//   - host Kahan reduction
//
// Realized throughput lever: SNAP pair-math is <5% of SNAP runtime (the
// bispectrum U-recurrence is the dominant cost). Phase A therefore yields
// a single-digit-% throughput win at best. Documented honestly in gpu/SPEC
// §8.5. MixedFastSnapOnlyBuild remains opt-in precisely because Phase A's
// speedup is modest. Phase B (full-FP32 bispectrum) is deferred to a
// §D.17 procedure conditional on T8.11 cloud-burst perf pressure.
//
// No `atomicAdd(double*, double)` anywhere — every global write is owned by a
// single block. Determinism: same input → same output bit-exactly across runs
// on fixed hardware (FP32 SFU rounding is deterministic on NVIDIA).
//
// Shared-memory budget: identical to snap_gpu.cu (FP32 narrowing is pair-math
// scalar, no shared-mem layout change).

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"
#include "tdmd/gpu/snap_bond_list_gpu.hpp"
#include "tdmd/gpu/snap_gpu.hpp"
#include "tdmd/gpu/snap_gpu_mixed.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/telemetry/nvtx.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

// T8.6c-v5 Stage 2 (mixed mirror): per-bond ui dispatch default. A/B against
// legacy per-atom snap_ui_kernel via -DTDMD_SNAP_LEGACY_PERATOM=1. See the
// twin comment in snap_gpu.cu for the byte-exact semantics. MixedFast's single
// FP32 narrowing site (`sqrtf(rsq)`) is mirrored into the per-bond kernel.
#ifndef TDMD_SNAP_LEGACY_PERATOM
#define TDMD_SNAP_LEGACY_PERATOM 0
#endif

#if TDMD_BUILD_CUDA
#include "cuda_handles.hpp"
#include "snap_gpu_device.cuh"
#include "snap_gpu_tables.cuh"

#include <cuda_runtime.h>
#endif

namespace tdmd::gpu {

namespace {

#if TDMD_BUILD_CUDA

[[noreturn]] void throw_cuda(const char* op, cudaError_t err) {
  std::ostringstream oss;
  oss << "gpu::SnapGpuMixed::" << op << ": " << cudaGetErrorName(err) << " ("
      << cudaGetErrorString(err) << ")";
  throw std::runtime_error(oss.str());
}

void check_cuda(const char* op, cudaError_t err) {
  if (err != cudaSuccess) {
    throw_cuda(op, err);
  }
}

// POD captured by the SNAP kernels. Kept small so it passes directly as a
// launch argument (< 1 KB const-mem budget).
struct DeviceSnapParams {
  // Box + cell grid.
  double xlo, ylo, zlo;
  double lx, ly, lz;
  double cell_x, cell_y, cell_z;
  std::uint32_t nx, ny, nz;
  int periodic_x, periodic_y, periodic_z;

  // SNAP hyperparameters.
  int twojmax;
  int jdim;    // == twojmax + 1
  int jdimpq;  // == twojmax + 2
  int idxu_max;
  int idxz_max;
  int idxb_max;
  double rcutfac;
  double rfac0;
  double rmin0;
  int switch_flag;
  int bzero_flag;
  int bnorm_flag;
  int wselfall_flag;
  double wself;  // == 1.0 for SNAP

  std::uint32_t n_species;
};

// ------------ Shared device helpers (mirrored from EAM) --------------------

// Cell-stencil helpers used only by the legacy per-atom snap_ui_kernel /
// snap_deidrj_kernel. T8.6c-v5 Stage 2/3 per-bond kernels consume the
// pre-built bond list so these are unreferenced in the default build.
#if TDMD_SNAP_LEGACY_PERATOM
__device__ __forceinline__ std::uint32_t wrap_axis_dev(int idx, std::uint32_t n) {
  const int ni = static_cast<int>(n);
  int w = idx % ni;
  if (w < 0) {
    w += ni;
  }
  return static_cast<std::uint32_t>(w);
}

__device__ __forceinline__ std::size_t cell_index_axis_dev(double coord,
                                                           double lo,
                                                           double cell,
                                                           std::uint32_t n) {
  const double local = coord - lo;
  long long idx = static_cast<long long>(floor(local / cell));
  if (idx < 0) {
    idx = 0;
  } else if (idx >= static_cast<long long>(n)) {
    idx = static_cast<long long>(n) - 1;
  }
  return static_cast<std::size_t>(idx);
}

__device__ __forceinline__ std::size_t linear_index_dev(std::uint32_t ix,
                                                        std::uint32_t iy,
                                                        std::uint32_t iz,
                                                        std::uint32_t nx,
                                                        std::uint32_t ny) {
  return static_cast<std::size_t>(ix) +
         static_cast<std::size_t>(nx) *
             (static_cast<std::size_t>(iy) +
              static_cast<std::size_t>(ny) * static_cast<std::size_t>(iz));
}

__device__ __forceinline__ double minimum_image_axis_dev(double delta, double len, int periodic) {
  if (!periodic) {
    return delta;
  }
  if (!(len > 0.0)) {
    return delta;
  }
  const double half = 0.5 * len;
  if (delta > half) {
    delta -= len * ceil((delta - half) / len);
  } else if (delta < -half) {
    delta += len * ceil((-delta - half) / len);
  }
  return delta;
}

__device__ __forceinline__ std::size_t pair_index_dev(std::uint32_t a,
                                                      std::uint32_t b,
                                                      std::uint32_t n_species) {
  return static_cast<std::size_t>(a) * static_cast<std::size_t>(n_species) +
         static_cast<std::size_t>(b);
}
#endif  // TDMD_SNAP_LEGACY_PERATOM

// Flat 3D address into jdim×jdim×jdim index blocks (idxcg/idxz/idxb _block).
__device__ __forceinline__ std::size_t jkk_index_dev(int j1, int j2, int j, int jdim) {
  return static_cast<std::size_t>(j1) * static_cast<std::size_t>(jdim) *
             static_cast<std::size_t>(jdim) +
         static_cast<std::size_t>(j2) * static_cast<std::size_t>(jdim) +
         static_cast<std::size_t>(j);
}

// ---------------------------------------------------------------------------
// compute_deidrj_parallel_device — T8.6c block-parallel port of the sequential
// compute_deidrj_device below. Mirrored from snap_gpu.cu (FP64 reduction tree
// unchanged in MixedFast; no narrow-FP32 applied in this reduction path).
// See snap_gpu.cu for the algorithmic commentary.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_deidrj_parallel_device(int twojmax,
                                                               int tid,
                                                               int block_threads,
                                                               int warp_id,
                                                               int lane_id,
                                                               const int* __restrict__ idxu_block,
                                                               const double* __restrict__ dulist_r,
                                                               const double* __restrict__ dulist_i,
                                                               const double* __restrict__ ylist_r,
                                                               const double* __restrict__ ylist_i,
                                                               double (*warp_sums)[3],
                                                               double* dedr) {
  double my0 = 0.0;
  double my1 = 0.0;
  double my2 = 0.0;

  int pos_idx = 0;
  for (int j = 0; j <= twojmax; ++j) {
    int jju = idxu_block[j];
    for (int mb = 0; 2 * mb < j; ++mb) {
      for (int ma = 0; ma <= j; ++ma) {
        if ((pos_idx % block_threads) == tid) {
          const double yr = ylist_r[jju];
          const double yi = ylist_i[jju];
          my0 += dulist_r[jju * 3 + 0] * yr + dulist_i[jju * 3 + 0] * yi;
          my1 += dulist_r[jju * 3 + 1] * yr + dulist_i[jju * 3 + 1] * yi;
          my2 += dulist_r[jju * 3 + 2] * yr + dulist_i[jju * 3 + 2] * yi;
        }
        pos_idx++;
        jju++;
      }
    }
    if ((j % 2) == 0) {
      int mb = j / 2;
      for (int ma = 0; ma < mb; ++ma) {
        if ((pos_idx % block_threads) == tid) {
          const double yr = ylist_r[jju];
          const double yi = ylist_i[jju];
          my0 += dulist_r[jju * 3 + 0] * yr + dulist_i[jju * 3 + 0] * yi;
          my1 += dulist_r[jju * 3 + 1] * yr + dulist_i[jju * 3 + 1] * yi;
          my2 += dulist_r[jju * 3 + 2] * yr + dulist_i[jju * 3 + 2] * yi;
        }
        pos_idx++;
        jju++;
      }
      if ((pos_idx % block_threads) == tid) {
        const double yr = ylist_r[jju];
        const double yi = ylist_i[jju];
        my0 += 0.5 * (dulist_r[jju * 3 + 0] * yr + dulist_i[jju * 3 + 0] * yi);
        my1 += 0.5 * (dulist_r[jju * 3 + 1] * yr + dulist_i[jju * 3 + 1] * yi);
        my2 += 0.5 * (dulist_r[jju * 3 + 2] * yr + dulist_i[jju * 3 + 2] * yi);
      }
      pos_idx++;
      jju++;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    my0 += __shfl_down_sync(0xFFFFFFFFu, my0, offset);
    my1 += __shfl_down_sync(0xFFFFFFFFu, my1, offset);
    my2 += __shfl_down_sync(0xFFFFFFFFu, my2, offset);
  }
  if (lane_id == 0) {
    warp_sums[warp_id][0] = my0;
    warp_sums[warp_id][1] = my1;
    warp_sums[warp_id][2] = my2;
  }
  __syncthreads();

  if (tid == 0) {
    const int n_warps = (block_threads + 31) / 32;
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    for (int w = 0; w < n_warps; ++w) {
      s0 += warp_sums[w][0];
      s1 += warp_sums[w][1];
      s2 += warp_sums[w][2];
    }
    dedr[0] = s0 * 2.0;
    dedr[1] = s1 * 2.0;
    dedr[2] = s2 * 2.0;
  }
}

// ---------------------------------------------------------------------------
// KERNEL 1: snap_ui_kernel
//
// One block per atom, 128 threads. Each block accumulates d_ulisttot_r/i[i,·]
// by walking its 3³ cell stencil and calling compute_uarray per in-cutoff pair.
// The U accumulation is FP-sensitive (recurrence + accumulation order), so
// the compute_uarray + add-to-ulisttot work runs single-lane (tid==0). Zeroing
// and the final global write are thread-parallel across the idxu_max slab.
//
// T8.6c-v5 Stage 2: retained only for A/B testing via
// `-DTDMD_SNAP_LEGACY_PERATOM=1`. Default build path is bond list +
// snap_ui_bond_kernel + snap_ui_gather_kernel.
// ---------------------------------------------------------------------------
#if TDMD_SNAP_LEGACY_PERATOM
__global__ void snap_ui_kernel(std::uint32_t n,
                               const std::uint32_t* __restrict__ types,
                               const double* __restrict__ x,
                               const double* __restrict__ y,
                               const double* __restrict__ z,
                               const std::uint32_t* __restrict__ cell_offsets,
                               const std::uint32_t* __restrict__ cell_atoms,
                               const int* __restrict__ idxu_block,
                               const double* __restrict__ rootpq,
                               const double* __restrict__ radius_elem,
                               const double* __restrict__ weight_elem,
                               const double* __restrict__ rcut_sq_ab,
                               DeviceSnapParams p,
                               double* __restrict__ d_ulisttot_r,
                               double* __restrict__ d_ulisttot_i) {
  const std::uint32_t i = blockIdx.x;
  if (i >= n) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int block_threads = static_cast<int>(blockDim.x);

  extern __shared__ double shm_ui[];
  double* ulist_r = shm_ui;
  double* ulist_i = ulist_r + p.idxu_max;
  double* ulisttot_r = ulist_i + p.idxu_max;
  double* ulisttot_i = ulisttot_r + p.idxu_max;

  // Shared scalar so the single-lane compute_uarray + sfacwj producer hands
  // the factor to the block-parallel add_uarraytot loop below.
  __shared__ double sfacwj_shared;

  // Zero ulisttot (parallel).
  for (int k = tid; k < p.idxu_max; k += block_threads) {
    ulisttot_r[k] = 0.0;
    ulisttot_i[k] = 0.0;
  }
  __syncthreads();

  // Self-identity seed (zero_uarraytot in CPU). Only jelem == ielem writes,
  // and wselfall_flag broadens the ma==mb seed to all jelems. nelements_=1 on
  // M8, so ielem = jelem = 0 always; the wselfall check is kept for parity.
  if (tid == 0) {
    const int jelem = 0;
    const int ielem = 0;
    const int write_self = (jelem == ielem) || (p.wselfall_flag != 0);
    for (int j = 0; j <= p.twojmax; ++j) {
      int jju = idxu_block[j];
      for (int mb = 0; mb <= j; ++mb) {
        for (int ma = 0; ma <= j; ++ma) {
          if (write_self && ma == mb) {
            ulisttot_r[jju] = p.wself;
          }
          jju++;
        }
      }
    }
  }
  __syncthreads();

  const std::uint32_t itype = types[i];
  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];
  const double radi = radius_elem[itype];

  const auto ix_u = static_cast<int>(cell_index_axis_dev(xi, p.xlo, p.cell_x, p.nx));
  const auto iy_u = static_cast<int>(cell_index_axis_dev(yi, p.ylo, p.cell_y, p.ny));
  const auto iz_u = static_cast<int>(cell_index_axis_dev(zi, p.zlo, p.cell_z, p.nz));

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const std::uint32_t jx = wrap_axis_dev(ix_u + dx, p.nx);
        const std::uint32_t jy = wrap_axis_dev(iy_u + dy, p.ny);
        const std::uint32_t jz = wrap_axis_dev(iz_u + dz, p.nz);
        const std::size_t cj = linear_index_dev(jx, jy, jz, p.nx, p.ny);

        const std::uint32_t begin = cell_offsets[cj];
        const std::uint32_t end = cell_offsets[cj + 1];
        for (std::uint32_t k = begin; k < end; ++k) {
          const std::uint32_t j = cell_atoms[k];
          if (j == i) {
            continue;
          }
          double ddx = x[j] - xi;
          double ddy = y[j] - yi;
          double ddz = z[j] - zi;
          ddx = minimum_image_axis_dev(ddx, p.lx, p.periodic_x);
          ddy = minimum_image_axis_dev(ddy, p.ly, p.periodic_y);
          ddz = minimum_image_axis_dev(ddz, p.lz, p.periodic_z);
          const double rsq = ddx * ddx + ddy * ddy + ddz * ddz;
          const std::uint32_t jtype = types[j];
          const double cutsq_ij = rcut_sq_ab[pair_index_dev(itype, jtype, p.n_species)];
          // Match CPU: rsq < cutsq && rsq > 1e-20 (guard against co-located atoms).
          if (!(rsq < cutsq_ij) || !(rsq > 1e-20)) {
            continue;
          }

          // T8.6c-v4: scalar setup on tid==0, stashed to shared memory so the
          // block-parallel compute_uarray sees the same (r, z0) inputs.
          __shared__ double ui_r_sh;
          __shared__ double ui_z0_sh;
          __shared__ double ui_rcut_sh;
          __shared__ double ui_wj_sh;
          if (tid == 0) {
            // T8.9 Phase A: FP32 only for `sqrtf(rsq)`. The SFU sincosf
            // was too imprecise — its ~1 ULP error amplifies through the
            // bispectrum recurrence and blows past the D-M8-8 force
            // budget (measured 2.1e-5 rel vs the 1e-5 cap).
            //
            // Why `sqrtf` is kept: `rsq` reaches here at FP64 magnitude
            // and `r = sqrt(rsq)` only feeds scalars (theta0, z0) —
            // errors here do not amplify, and `sqrtf` on SFU is ~2× the
            // throughput of FP64 `sqrt`. Measured residual rel force
            // error: ~5e-7 (well under D-M8-8).
            //
            // Scientific cost of this decision is documented in
            // docs/specs/gpu/SPEC.md §8.5 — MixedFastSnapOnlyBuild is
            // opt-in precisely because SNAP's pair-math is <5 % of total
            // SNAP runtime, so narrow-FP32 here is a modest throughput
            // lever, not a structural win. Real SNAP throughput comes
            // from Fp64Production + dense-cutoff GPU path (T8.6b).
            const float rsq_f = static_cast<float>(rsq);
            const float r_f = sqrtf(rsq_f);
            const double r = static_cast<double>(r_f);
            const double radj = radius_elem[jtype];
            const double wj = weight_elem[jtype];
            const double rcut = (radi + radj) * p.rcutfac;
            const double theta0 =
                (r - p.rmin0) * p.rfac0 * static_cast<double>(M_PI) / (rcut - p.rmin0);
            double cs_d, sn_d;
            sincos(theta0, &sn_d, &cs_d);
            const double z0 = r * cs_d / sn_d;
            ui_r_sh = r;
            ui_z0_sh = z0;
            ui_rcut_sh = rcut;
            ui_wj_sh = wj;
          }
          __syncthreads();
          snap_detail::compute_uarray_parallel_device(static_cast<unsigned>(tid),
                                                      static_cast<unsigned>(block_threads),
                                                      ddx,
                                                      ddy,
                                                      ddz,
                                                      ui_z0_sh,
                                                      ui_r_sh,
                                                      rootpq,
                                                      p.jdimpq,
                                                      idxu_block,
                                                      p.twojmax,
                                                      ulist_r,
                                                      ulist_i);
          if (tid == 0) {
            const double sfac =
                snap_detail::compute_sfac_device(ui_r_sh, ui_rcut_sh, p.rmin0, p.switch_flag);
            sfacwj_shared = sfac * ui_wj_sh;
          }
          __syncthreads();
          // Port of add_uarraytot (sna.cpp 806-830) with jelem=0. The CPU
          // nested (jl, mb, ma) loop sweeps jju = 0..idxu_max-1 contiguously
          // via idxu_block[jl] offsets; each position is written exactly once
          // per neighbor with an independent `+= sfacwj * ulist[jju]`. Order
          // among positions within one neighbor does not affect the FP sum
          // (each position is touched once), so a strided block-parallel
          // sweep is byte-exact relative to the sequential walk.
          for (int k = tid; k < p.idxu_max; k += block_threads) {
            ulisttot_r[k] += sfacwj_shared * ulist_r[k];
            ulisttot_i[k] += sfacwj_shared * ulist_i[k];
          }
          __syncthreads();
        }
      }
    }
  }

  // Write ulisttot to global (parallel).
  const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(p.idxu_max);
  for (int k = tid; k < p.idxu_max; k += block_threads) {
    d_ulisttot_r[base + k] = ulisttot_r[k];
    d_ulisttot_i[base + k] = ulisttot_i[k];
  }
}
#endif  // TDMD_SNAP_LEGACY_PERATOM

#if !TDMD_SNAP_LEGACY_PERATOM
// ---------------------------------------------------------------------------
// T8.6c-v5 Stage 2 (MixedFast mirror): KERNEL 1a — snap_ui_bond_kernel
//
// Per-bond dispatch (<<<n_bonds, 128>>>). Same structure as the FP64 sibling
// in snap_gpu.cu with the single MixedFast FP32 narrowing site preserved:
//   r = sqrtf(static_cast<float>(rsq))
// All downstream arithmetic (theta0, z0, sfac, U-recurrence, accumulators) is
// FP64. D-M8-8 threshold budget (≤1e-5 rel force / ≤1e-7 rel PE) still holds
// — the FP32 sqrt site introduces ~6e-8 rel on r, and the per-bond path
// preserves the same accumulation order as the legacy kernel (see gather
// kernel below for byte-exactness argument vs. the CPU oracle's neighbour
// walk).
// ---------------------------------------------------------------------------
__global__ void snap_ui_bond_kernel(std::uint32_t n_bonds,
                                    const std::uint32_t* __restrict__ bond_type_i,
                                    const std::uint32_t* __restrict__ bond_type_j,
                                    const double* __restrict__ bond_dx,
                                    const double* __restrict__ bond_dy,
                                    const double* __restrict__ bond_dz,
                                    const double* __restrict__ bond_rsq,
                                    const int* __restrict__ idxu_block,
                                    const double* __restrict__ rootpq,
                                    const double* __restrict__ radius_elem,
                                    const double* __restrict__ weight_elem,
                                    DeviceSnapParams p,
                                    double* __restrict__ d_ulist_bond_r,
                                    double* __restrict__ d_ulist_bond_i) {
  const std::uint32_t b = blockIdx.x;
  if (b >= n_bonds) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int block_threads = static_cast<int>(blockDim.x);

  extern __shared__ double shm_ui_bond[];
  double* ulist_r = shm_ui_bond;
  double* ulist_i = ulist_r + p.idxu_max;

  __shared__ double ui_r_sh;
  __shared__ double ui_z0_sh;
  __shared__ double sfacwj_sh;

  const std::uint32_t itype = bond_type_i[b];
  const std::uint32_t jtype = bond_type_j[b];
  const double ddx = bond_dx[b];
  const double ddy = bond_dy[b];
  const double ddz = bond_dz[b];
  const double rsq = bond_rsq[b];

  if (tid == 0) {
    // T8.9 Phase A narrowing site — the only FP32 op in MixedFast's SNAP path.
    const float rsq_f = static_cast<float>(rsq);
    const float r_f = sqrtf(rsq_f);
    const double r = static_cast<double>(r_f);
    const double radi = radius_elem[itype];
    const double radj = radius_elem[jtype];
    const double wj = weight_elem[jtype];
    const double rcut = (radi + radj) * p.rcutfac;
    const double theta0 = (r - p.rmin0) * p.rfac0 * static_cast<double>(M_PI) / (rcut - p.rmin0);
    double cs_d, sn_d;
    sincos(theta0, &sn_d, &cs_d);
    const double z0 = r * cs_d / sn_d;
    ui_r_sh = r;
    ui_z0_sh = z0;
    const double sfac = snap_detail::compute_sfac_device(r, rcut, p.rmin0, p.switch_flag);
    sfacwj_sh = sfac * wj;
  }
  __syncthreads();

  snap_detail::compute_uarray_parallel_device(static_cast<unsigned>(tid),
                                              static_cast<unsigned>(block_threads),
                                              ddx,
                                              ddy,
                                              ddz,
                                              ui_z0_sh,
                                              ui_r_sh,
                                              rootpq,
                                              p.jdimpq,
                                              idxu_block,
                                              p.twojmax,
                                              ulist_r,
                                              ulist_i);
  __syncthreads();

  const std::size_t out_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(p.idxu_max);
  for (int k = tid; k < p.idxu_max; k += block_threads) {
    d_ulist_bond_r[out_base + k] = sfacwj_sh * ulist_r[k];
    d_ulist_bond_i[out_base + k] = sfacwj_sh * ulist_i[k];
  }
}

// ---------------------------------------------------------------------------
// T8.6c-v5 Stage 2 (MixedFast mirror): KERNEL 1b — snap_ui_gather_kernel
//
// Per-atom gather (<<<n_atoms, 128>>>). Identical to the FP64 sibling — the
// gather path touches no FP32, since all d_ulist_bond values are FP64 doubles
// written by the per-bond kernel. D-M8-8 residual on force/PE flows entirely
// from the single `sqrtf(rsq)` narrowing upstream.
// ---------------------------------------------------------------------------
__global__ void snap_ui_gather_kernel(std::uint32_t n_atoms,
                                      const std::uint32_t* __restrict__ atom_bond_start,
                                      const int* __restrict__ idxu_block,
                                      DeviceSnapParams p,
                                      const double* __restrict__ d_ulist_bond_r,
                                      const double* __restrict__ d_ulist_bond_i,
                                      double* __restrict__ d_ulisttot_r,
                                      double* __restrict__ d_ulisttot_i) {
  const std::uint32_t i = blockIdx.x;
  if (i >= n_atoms) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int block_threads = static_cast<int>(blockDim.x);

  extern __shared__ double shm_gather[];
  double* ulisttot_r = shm_gather;
  double* ulisttot_i = ulisttot_r + p.idxu_max;

  for (int k = tid; k < p.idxu_max; k += block_threads) {
    ulisttot_r[k] = 0.0;
    ulisttot_i[k] = 0.0;
  }
  __syncthreads();

  if (tid == 0) {
    const int jelem = 0;
    const int ielem = 0;
    const int write_self = (jelem == ielem) || (p.wselfall_flag != 0);
    for (int j = 0; j <= p.twojmax; ++j) {
      int jju = idxu_block[j];
      for (int mb = 0; mb <= j; ++mb) {
        for (int ma = 0; ma <= j; ++ma) {
          if (write_self && ma == mb) {
            ulisttot_r[jju] = p.wself;
          }
          jju++;
        }
      }
    }
  }
  __syncthreads();

  const std::uint32_t b_begin = atom_bond_start[i];
  const std::uint32_t b_end = atom_bond_start[i + 1];

  for (int k = tid; k < p.idxu_max; k += block_threads) {
    double acc_r = ulisttot_r[k];
    double acc_i = ulisttot_i[k];
    for (std::uint32_t b = b_begin; b < b_end; ++b) {
      const std::size_t bbase = static_cast<std::size_t>(b) * static_cast<std::size_t>(p.idxu_max);
      acc_r += d_ulist_bond_r[bbase + k];
      acc_i += d_ulist_bond_i[bbase + k];
    }
    ulisttot_r[k] = acc_r;
    ulisttot_i[k] = acc_i;
  }
  __syncthreads();

  const std::size_t out_base = static_cast<std::size_t>(i) * static_cast<std::size_t>(p.idxu_max);
  for (int k = tid; k < p.idxu_max; k += block_threads) {
    d_ulisttot_r[out_base + k] = ulisttot_r[k];
    d_ulisttot_i[out_base + k] = ulisttot_i[k];
  }
}
#endif  // !TDMD_SNAP_LEGACY_PERATOM

// ---------------------------------------------------------------------------
// KERNEL 2: snap_yi_kernel
//
// One block per atom, 128 threads. Consumes d_ulisttot[i,·]; produces
//   d_ylist[i,·]       — Y (energy-gradient intermediate)
//   d_pe_per_atom[i]   — evdwl_i = β₀ + Σ β_k · blist[k]
// Computes zlist in shared memory (re-used by the blist contraction in the
// same block), then compute_yi + compute_bi + PE in sequence.
// ---------------------------------------------------------------------------
__global__ void snap_yi_kernel(std::uint32_t n,
                               const std::uint32_t* __restrict__ types,
                               const int* __restrict__ idxu_block,
                               const int* __restrict__ idxz_block,
                               const int* __restrict__ idxb_block,
                               const int* __restrict__ idxcg_block,
                               const int* __restrict__ idxz_packed,  // idxz_max × 10
                               const int* __restrict__ idxb_packed,  // idxb_max × 3
                               const double* __restrict__ cglist,
                               const double* __restrict__ beta,  // n_species × beta_stride
                               std::size_t beta_stride,
                               const double* __restrict__ bzero,  // twojmax+1 (or null)
                               DeviceSnapParams p,
                               const double* __restrict__ d_ulisttot_r,
                               const double* __restrict__ d_ulisttot_i,
                               double* __restrict__ d_ylist_r,
                               double* __restrict__ d_ylist_i,
                               double* __restrict__ d_pe_per_atom) {
  const std::uint32_t i = blockIdx.x;
  if (i >= n) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int block_threads = static_cast<int>(blockDim.x);

  extern __shared__ double shm_yi[];
  double* ulisttot_r = shm_yi;
  double* ulisttot_i = ulisttot_r + p.idxu_max;
  double* ylist_r = ulisttot_i + p.idxu_max;
  double* ylist_i = ylist_r + p.idxu_max;
  double* zlist_r = ylist_i + p.idxu_max;
  double* zlist_i = zlist_r + p.idxz_max;
  // T8.6c: per-jjz (betaj * ztmp) stash for parallel-produce /
  // sequential-accumulate handoff into ylist (byte-exact order).
  double* ybuf_r = zlist_i + p.idxz_max;
  double* ybuf_i = ybuf_r + p.idxz_max;

  // Load ulisttot[i] and zero ylist.
  const std::size_t ubase = static_cast<std::size_t>(i) * static_cast<std::size_t>(p.idxu_max);
  for (int k = tid; k < p.idxu_max; k += block_threads) {
    ulisttot_r[k] = d_ulisttot_r[ubase + k];
    ulisttot_i[k] = d_ulisttot_i[ubase + k];
    ylist_r[k] = 0.0;
    ylist_i[k] = 0.0;
  }
  __syncthreads();

  const std::uint32_t itype = types[i];
  const double* __restrict__ beta_i = beta + static_cast<std::size_t>(itype) * beta_stride;
  // Linear SNAP: k-index 1..k_max of beta_i is β_k; β₀ = beta_i[0] is the
  // constant offset used only in PE. We mirror CPU by pointing one past β₀
  // for the yi/bi contractions (matches snap.cpp:226).
  const double* __restrict__ beta_k = beta_i + 1;

  // --- Step 1: compute zlist + accumulate β·z → ylist (CPU compute_yi port
  //             with zlist stashed for reuse by compute_bi).
  //
  // T8.6c split: Phase A (parallel over jjz) computes ztmp and βj independently
  // per z-triple, writing zlist[jjz] (each jjz is its own slot) and stashing
  // ybuf[jjz] = βj * ztmp. Phase B (tid==0 sequential) accumulates ybuf into
  // ylist[jju] in original jjz order — because multiple (j1,j2,j) triples map
  // to the same jju, the += order matters for FP identity. The product
  // `βj * ztmp` is computed identically per jjz in both the original and
  // split forms (pure functions of ulisttot + cglist + idxz_packed), so ybuf
  // carries the same value that would have flowed into the original `+=`.
  for (int jjz = tid; jjz < p.idxz_max; jjz += block_threads) {
    const int* zrec = idxz_packed + jjz * 10;
    const int j1 = zrec[0];
    const int j2 = zrec[1];
    const int j = zrec[2];
    const int ma1min = zrec[3];
    const int ma2max = zrec[4];
    const int mb1min = zrec[5];
    const int mb2max = zrec[6];
    const int na = zrec[7];
    const int nb = zrec[8];

    const double* cgblock = cglist + idxcg_block[jkk_index_dev(j1, j2, j, p.jdim)];

    double ztmp_r = 0.0;
    double ztmp_i = 0.0;

    int jju1 = idxu_block[j1] + (j1 + 1) * mb1min;
    int jju2 = idxu_block[j2] + (j2 + 1) * mb2max;
    int icgb = mb1min * (j2 + 1) + mb2max;
    for (int ib = 0; ib < nb; ++ib) {
      double suma1_r = 0.0;
      double suma1_i = 0.0;

      const double* u1r = ulisttot_r + jju1;
      const double* u1i = ulisttot_i + jju1;
      const double* u2r = ulisttot_r + jju2;
      const double* u2i = ulisttot_i + jju2;

      int ma1 = ma1min;
      int ma2 = ma2max;
      int icga = ma1min * (j2 + 1) + ma2max;

      for (int ia = 0; ia < na; ++ia) {
        suma1_r += cgblock[icga] * (u1r[ma1] * u2r[ma2] - u1i[ma1] * u2i[ma2]);
        suma1_i += cgblock[icga] * (u1r[ma1] * u2i[ma2] + u1i[ma1] * u2r[ma2]);
        ma1++;
        ma2--;
        icga += j2;
      }
      ztmp_r += cgblock[icgb] * suma1_r;
      ztmp_i += cgblock[icgb] * suma1_i;
      jju1 += j1 + 1;
      jju2 -= j2 + 1;
      icgb += j2;
    }
    if (p.bnorm_flag) {
      ztmp_r /= (j + 1);
      ztmp_i /= (j + 1);
    }
    zlist_r[jjz] = ztmp_r;
    zlist_i[jjz] = ztmp_i;

    // β selection — CPU compute_yi (nelements_=1 ⇒ elem3=0).
    double betaj;
    if (j >= j1) {
      const int jjb = idxb_block[jkk_index_dev(j1, j2, j, p.jdim)];
      if (j1 == j) {
        if (j2 == j) {
          betaj = 3.0 * beta_k[jjb];
        } else {
          betaj = 2.0 * beta_k[jjb];
        }
      } else {
        betaj = beta_k[jjb];
      }
    } else if (j >= j2) {
      const int jjb = idxb_block[jkk_index_dev(j, j2, j1, p.jdim)];
      if (j2 == j) {
        betaj = 2.0 * beta_k[jjb];
      } else {
        betaj = beta_k[jjb];
      }
    } else {
      const int jjb = idxb_block[jkk_index_dev(j2, j, j1, p.jdim)];
      betaj = beta_k[jjb];
    }
    if (!p.bnorm_flag && j1 > j) {
      betaj *= static_cast<double>(j1 + 1) / static_cast<double>(j + 1);
    }

    ybuf_r[jjz] = betaj * ztmp_r;
    ybuf_i[jjz] = betaj * ztmp_i;
  }
  __syncthreads();

  // Phase B: single-lane sequential accumulation in original jjz order.
  if (tid == 0) {
    for (int jjz = 0; jjz < p.idxz_max; ++jjz) {
      const int jju = idxz_packed[jjz * 10 + 9];
      ylist_r[jju] += ybuf_r[jjz];
      ylist_i[jju] += ybuf_i[jjz];
    }
  }
  __syncthreads();

  // --- Step 2: write ylist → global (parallel).
  for (int k = tid; k < p.idxu_max; k += block_threads) {
    d_ylist_r[ubase + k] = ylist_r[k];
    d_ylist_i[ubase + k] = ylist_i[k];
  }

  // --- Step 3: compute blist + PE_i (single-lane, uses shared zlist).
  if (tid == 0) {
    double evdwl_i = beta_i[0];  // β₀
    for (int jjb = 0; jjb < p.idxb_max; ++jjb) {
      const int* brec = idxb_packed + jjb * 3;
      const int j1 = brec[0];
      const int j2 = brec[1];
      const int j = brec[2];

      int jjz = idxz_block[jkk_index_dev(j1, j2, j, p.jdim)];
      int jju = idxu_block[j];
      double sumzu = 0.0;
      for (int mb = 0; 2 * mb < j; ++mb) {
        for (int ma = 0; ma <= j; ++ma) {
          sumzu += ulisttot_r[jju] * zlist_r[jjz] + ulisttot_i[jju] * zlist_i[jjz];
          jjz++;
          jju++;
        }
      }
      if ((j % 2) == 0) {
        int mb = j / 2;
        for (int ma = 0; ma < mb; ++ma) {
          sumzu += ulisttot_r[jju] * zlist_r[jjz] + ulisttot_i[jju] * zlist_i[jjz];
          jjz++;
          jju++;
        }
        sumzu += 0.5 * (ulisttot_r[jju] * zlist_r[jjz] + ulisttot_i[jju] * zlist_i[jjz]);
      }

      double blist_val = 2.0 * sumzu;
      if (p.bzero_flag && !p.wselfall_flag) {
        // nelements_=1 ⇒ only the self-triple (ielem,ielem,ielem) is hit.
        blist_val -= bzero[j];
      }

      evdwl_i += beta_k[jjb] * blist_val;
    }
    d_pe_per_atom[i] = evdwl_i;
  }
}

// ---------------------------------------------------------------------------
// KERNEL 3: snap_deidrj_kernel (legacy per-atom path)
//
// One block per atom, 128 threads. Writes per-atom fx/fy/fz + per-atom virial.
// Each block walks i's 3³ cells, runs compute_deidrj twice per pair (own side
// using ylist[i], peer side using ylist[j] read from global).
//
// T8.6c-v5 Stage 3: retained only for A/B testing via
// `-DTDMD_SNAP_LEGACY_PERATOM=1`. Default build path is bond list +
// snap_deidrj_bond_kernel + snap_force_gather_kernel (see below).
// ---------------------------------------------------------------------------
#if TDMD_SNAP_LEGACY_PERATOM
__global__ void snap_deidrj_kernel(std::uint32_t n,
                                   const std::uint32_t* __restrict__ types,
                                   const double* __restrict__ x,
                                   const double* __restrict__ y,
                                   const double* __restrict__ z,
                                   const std::uint32_t* __restrict__ cell_offsets,
                                   const std::uint32_t* __restrict__ cell_atoms,
                                   const int* __restrict__ idxu_block,
                                   const double* __restrict__ rootpq,
                                   const double* __restrict__ radius_elem,
                                   const double* __restrict__ weight_elem,
                                   const double* __restrict__ rcut_sq_ab,
                                   const double* __restrict__ d_ylist_r,
                                   const double* __restrict__ d_ylist_i,
                                   DeviceSnapParams p,
                                   double* __restrict__ d_fx,
                                   double* __restrict__ d_fy,
                                   double* __restrict__ d_fz,
                                   double* __restrict__ d_virial_per_atom) {
  const std::uint32_t i = blockIdx.x;
  if (i >= n) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int block_threads = static_cast<int>(blockDim.x);

  extern __shared__ double shm_de[];
  double* ylist_i_r = shm_de;
  double* ylist_i_i = ylist_i_r + p.idxu_max;
  double* ulist_r = ylist_i_i + p.idxu_max;
  double* ulist_i = ulist_r + p.idxu_max;
  double* dulist_r = ulist_i + p.idxu_max;
  double* dulist_i = dulist_r + p.idxu_max * 3;

  // T8.6c: cross-warp scratch for block-parallel compute_deidrj tree reduction
  // (4 warps × 3 force components).
  __shared__ double dedr_warp_sums[4][3];

  // Load atom i's ylist into shared.
  const std::size_t ubase = static_cast<std::size_t>(i) * static_cast<std::size_t>(p.idxu_max);
  for (int k = tid; k < p.idxu_max; k += block_threads) {
    ylist_i_r[k] = d_ylist_r[ubase + k];
    ylist_i_i[k] = d_ylist_i[ubase + k];
  }
  __syncthreads();

  const std::uint32_t itype = types[i];
  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];
  const double radi = radius_elem[itype];
  const double wi = weight_elem[itype];

  const auto ix_u = static_cast<int>(cell_index_axis_dev(xi, p.xlo, p.cell_x, p.nx));
  const auto iy_u = static_cast<int>(cell_index_axis_dev(yi, p.ylo, p.cell_y, p.ny));
  const auto iz_u = static_cast<int>(cell_index_axis_dev(zi, p.zlo, p.cell_z, p.nz));

  const int warp_id = tid >> 5;
  const int lane_id = tid & 0x1f;

  // Per-atom accumulators live only on tid==0 — each thread has its own
  // register zero-init, only tid==0 ever writes/reads the meaningful value.
  double fx_acc = 0.0;
  double fy_acc = 0.0;
  double fz_acc = 0.0;
  double v_xx = 0.0;
  double v_yy = 0.0;
  double v_zz = 0.0;
  double v_xy = 0.0;
  double v_xz = 0.0;
  double v_yz = 0.0;

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const std::uint32_t jx = wrap_axis_dev(ix_u + dx, p.nx);
        const std::uint32_t jy = wrap_axis_dev(iy_u + dy, p.ny);
        const std::uint32_t jz = wrap_axis_dev(iz_u + dz, p.nz);
        const std::size_t cj = linear_index_dev(jx, jy, jz, p.nx, p.ny);

        const std::uint32_t begin = cell_offsets[cj];
        const std::uint32_t end = cell_offsets[cj + 1];
        for (std::uint32_t k = begin; k < end; ++k) {
          const std::uint32_t j = cell_atoms[k];
          if (j == i) {
            continue;
          }
          double ddx = x[j] - xi;
          double ddy = y[j] - yi;
          double ddz = z[j] - zi;
          ddx = minimum_image_axis_dev(ddx, p.lx, p.periodic_x);
          ddy = minimum_image_axis_dev(ddy, p.ly, p.periodic_y);
          ddz = minimum_image_axis_dev(ddz, p.lz, p.periodic_z);
          const double rsq = ddx * ddx + ddy * ddy + ddz * ddz;
          const std::uint32_t jtype = types[j];
          const double cutsq_ij = rcut_sq_ab[pair_index_dev(itype, jtype, p.n_species)];
          if (!(rsq < cutsq_ij) || !(rsq > 1e-20)) {
            continue;
          }

          __shared__ double r_sh;
          __shared__ double z0_sh;
          __shared__ double dz0dr_sh;
          __shared__ double rcut_sh;
          __shared__ double wj_sh;

          if (tid == 0) {
            // T8.9 Phase A: FP32 only for `sqrtf(rsq)`. See snap_ui_kernel
            // for rationale — sincosf + FP32 z0 blow past D-M8-8 1e-5
            // force budget via bispectrum-recurrence amplification. `r`
            // is cast back to FP64 before all downstream math.
            const float rsq_f = static_cast<float>(rsq);
            const float r_f = sqrtf(rsq_f);
            const double r = static_cast<double>(r_f);
            const double radj = radius_elem[jtype];
            const double wj = weight_elem[jtype];
            const double rcut = (radi + radj) * p.rcutfac;
            const double rscale0 = p.rfac0 * static_cast<double>(M_PI) / (rcut - p.rmin0);
            const double theta0 = (r - p.rmin0) * rscale0;
            double cs_d, sn_d;
            sincos(theta0, &sn_d, &cs_d);
            const double z0 = r * cs_d / sn_d;
            const double dz0dr = z0 / r - (r * rscale0) * (rsq + z0 * z0) / rsq;
            r_sh = r;
            z0_sh = z0;
            dz0dr_sh = dz0dr;
            rcut_sh = rcut;
            wj_sh = wj;
          }
          __syncthreads();

          // --- OWN side: block-parallel compute_uarray + compute_duarray
          // (T8.6c-v4).
          snap_detail::compute_uarray_parallel_device(static_cast<unsigned>(tid),
                                                      static_cast<unsigned>(block_threads),
                                                      ddx,
                                                      ddy,
                                                      ddz,
                                                      z0_sh,
                                                      r_sh,
                                                      rootpq,
                                                      p.jdimpq,
                                                      idxu_block,
                                                      p.twojmax,
                                                      ulist_r,
                                                      ulist_i);
          snap_detail::compute_duarray_parallel_device(static_cast<unsigned>(tid),
                                                       static_cast<unsigned>(block_threads),
                                                       ddx,
                                                       ddy,
                                                       ddz,
                                                       z0_sh,
                                                       r_sh,
                                                       dz0dr_sh,
                                                       wj_sh,
                                                       rcut_sh,
                                                       p.rmin0,
                                                       p.switch_flag,
                                                       rootpq,
                                                       p.jdimpq,
                                                       idxu_block,
                                                       p.twojmax,
                                                       ulist_r,
                                                       ulist_i,
                                                       dulist_r,
                                                       dulist_i);

          // --- OWN side: block-parallel dedr reduction (T8.6c).
          double dedr_own[3];
          compute_deidrj_parallel_device(p.twojmax,
                                         tid,
                                         block_threads,
                                         warp_id,
                                         lane_id,
                                         idxu_block,
                                         dulist_r,
                                         dulist_i,
                                         ylist_i_r,
                                         ylist_i_i,
                                         dedr_warp_sums,
                                         dedr_own);
          if (tid == 0) {
            fx_acc += dedr_own[0];
            fy_acc += dedr_own[1];
            fz_acc += dedr_own[2];
            v_xx += dedr_own[0] * ddx;
            v_yy += dedr_own[1] * ddy;
            v_zz += dedr_own[2] * ddz;
            v_xy += dedr_own[0] * ddy;
            v_xz += dedr_own[0] * ddz;
            v_yz += dedr_own[1] * ddz;
          }
          __syncthreads();

          // --- PEER side: rebuild ulist + dulist with flipped sign and wi
          // (T8.6c-v4 block-parallel).
          snap_detail::compute_uarray_parallel_device(static_cast<unsigned>(tid),
                                                      static_cast<unsigned>(block_threads),
                                                      -ddx,
                                                      -ddy,
                                                      -ddz,
                                                      z0_sh,
                                                      r_sh,
                                                      rootpq,
                                                      p.jdimpq,
                                                      idxu_block,
                                                      p.twojmax,
                                                      ulist_r,
                                                      ulist_i);
          snap_detail::compute_duarray_parallel_device(static_cast<unsigned>(tid),
                                                       static_cast<unsigned>(block_threads),
                                                       -ddx,
                                                       -ddy,
                                                       -ddz,
                                                       z0_sh,
                                                       r_sh,
                                                       dz0dr_sh,
                                                       wi,
                                                       rcut_sh,
                                                       p.rmin0,
                                                       p.switch_flag,
                                                       rootpq,
                                                       p.jdimpq,
                                                       idxu_block,
                                                       p.twojmax,
                                                       ulist_r,
                                                       ulist_i,
                                                       dulist_r,
                                                       dulist_i);

          const std::size_t jbase =
              static_cast<std::size_t>(j) * static_cast<std::size_t>(p.idxu_max);
          double dedr_peer[3];
          compute_deidrj_parallel_device(p.twojmax,
                                         tid,
                                         block_threads,
                                         warp_id,
                                         lane_id,
                                         idxu_block,
                                         dulist_r,
                                         dulist_i,
                                         d_ylist_r + jbase,
                                         d_ylist_i + jbase,
                                         dedr_warp_sums,
                                         dedr_peer);
          if (tid == 0) {
            fx_acc -= dedr_peer[0];
            fy_acc -= dedr_peer[1];
            fz_acc -= dedr_peer[2];
            // Virial: already accumulated on own side. Peer side's virial was
            // counted at atom j's block — no contribution here.
          }
          __syncthreads();
        }
      }
    }
  }

  if (tid == 0) {
    // Additive write per SnapGpuMixed::compute() caller contract (host reads
    // d_fx/d_fy/d_fz already primed with caller's in-values).
    d_fx[i] += fx_acc;
    d_fy[i] += fy_acc;
    d_fz[i] += fz_acc;
    d_virial_per_atom[i * 6 + 0] = v_xx;
    d_virial_per_atom[i * 6 + 1] = v_yy;
    d_virial_per_atom[i * 6 + 2] = v_zz;
    d_virial_per_atom[i * 6 + 3] = v_xy;
    d_virial_per_atom[i * 6 + 4] = v_xz;
    d_virial_per_atom[i * 6 + 5] = v_yz;
  }
}
#endif  // TDMD_SNAP_LEGACY_PERATOM

#if !TDMD_SNAP_LEGACY_PERATOM
// ---------------------------------------------------------------------------
// T8.6c-v5 Stage 3 (MixedFast): KERNEL 3a — snap_deidrj_bond_kernel
//
// Per-bond dispatch mirror of the FP64 kernel. Only narrowing site in this TU
// is `sqrtf(rsq)` — identical to mixed_ui_bond_kernel above (T8.9 Phase A
// D-M8-8 1e-5/1e-7 force budget). All downstream math (theta0, sincos, z0,
// dz0dr, rcut, wj_sh, sfac, compute_uarray/duarray/deidrj) runs FP64.
//
// Byte-exactness note: "byte-exact" in MixedFast means "matches the legacy
// mixed_fast snap_deidrj_kernel" (not FP64 reference). The FP32 sqrtf site is
// the only entropy source; since the new path mirrors it bit-for-bit, the
// D-M8-8 1e-5/1e-7 rel budget is preserved against the legacy mixed baseline.
// ---------------------------------------------------------------------------
__global__ void snap_deidrj_bond_kernel(std::uint32_t n_bonds,
                                        const std::uint32_t* __restrict__ bond_i,
                                        const std::uint32_t* __restrict__ bond_j,
                                        const std::uint32_t* __restrict__ bond_type_i,
                                        const std::uint32_t* __restrict__ bond_type_j,
                                        const double* __restrict__ bond_dx,
                                        const double* __restrict__ bond_dy,
                                        const double* __restrict__ bond_dz,
                                        const double* __restrict__ bond_rsq,
                                        const int* __restrict__ idxu_block,
                                        const double* __restrict__ rootpq,
                                        const double* __restrict__ radius_elem,
                                        const double* __restrict__ weight_elem,
                                        const double* __restrict__ d_ylist_r,
                                        const double* __restrict__ d_ylist_i,
                                        DeviceSnapParams p,
                                        double* __restrict__ d_dedr_own,
                                        double* __restrict__ d_dedr_peer) {
  const std::uint32_t b = blockIdx.x;
  if (b >= n_bonds) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int block_threads = static_cast<int>(blockDim.x);

  extern __shared__ double shm_deb[];
  double* ylist_slab_r = shm_deb;
  double* ylist_slab_i = ylist_slab_r + p.idxu_max;
  double* ulist_r = ylist_slab_i + p.idxu_max;
  double* ulist_i = ulist_r + p.idxu_max;
  double* dulist_r = ulist_i + p.idxu_max;
  double* dulist_i = dulist_r + p.idxu_max * 3;

  __shared__ double dedr_warp_sums[4][3];

  const std::uint32_t ii = bond_i[b];
  const std::uint32_t jj = bond_j[b];
  const std::uint32_t itype = bond_type_i[b];
  const std::uint32_t jtype = bond_type_j[b];
  const double ddx = bond_dx[b];
  const double ddy = bond_dy[b];
  const double ddz = bond_dz[b];
  const double rsq = bond_rsq[b];

  const double wi = weight_elem[itype];

  __shared__ double r_sh;
  __shared__ double z0_sh;
  __shared__ double dz0dr_sh;
  __shared__ double rcut_sh;
  __shared__ double wj_sh;

  if (tid == 0) {
    // T8.9 Phase A narrowing: ONLY `sqrtf` runs FP32; the result is widened
    // back to FP64 before any downstream math.
    const float rsq_f = static_cast<float>(rsq);
    const float r_f = sqrtf(rsq_f);
    const double r = static_cast<double>(r_f);
    const double radi = radius_elem[itype];
    const double radj = radius_elem[jtype];
    const double wj = weight_elem[jtype];
    const double rcut = (radi + radj) * p.rcutfac;
    const double rscale0 = p.rfac0 * static_cast<double>(M_PI) / (rcut - p.rmin0);
    const double theta0 = (r - p.rmin0) * rscale0;
    double cs_d, sn_d;
    sincos(theta0, &sn_d, &cs_d);
    const double z0 = r * cs_d / sn_d;
    const double dz0dr = z0 / r - (r * rscale0) * (rsq + z0 * z0) / rsq;
    r_sh = r;
    z0_sh = z0;
    dz0dr_sh = dz0dr;
    rcut_sh = rcut;
    wj_sh = wj;
  }

  const std::size_t iibase = static_cast<std::size_t>(ii) * static_cast<std::size_t>(p.idxu_max);
  for (int k = tid; k < p.idxu_max; k += block_threads) {
    ylist_slab_r[k] = d_ylist_r[iibase + k];
    ylist_slab_i[k] = d_ylist_i[iibase + k];
  }
  __syncthreads();

  const int warp_id = tid >> 5;
  const int lane_id = tid & 0x1f;

  // OWN side.
  snap_detail::compute_uarray_parallel_device(static_cast<unsigned>(tid),
                                              static_cast<unsigned>(block_threads),
                                              ddx,
                                              ddy,
                                              ddz,
                                              z0_sh,
                                              r_sh,
                                              rootpq,
                                              p.jdimpq,
                                              idxu_block,
                                              p.twojmax,
                                              ulist_r,
                                              ulist_i);
  snap_detail::compute_duarray_parallel_device(static_cast<unsigned>(tid),
                                               static_cast<unsigned>(block_threads),
                                               ddx,
                                               ddy,
                                               ddz,
                                               z0_sh,
                                               r_sh,
                                               dz0dr_sh,
                                               wj_sh,
                                               rcut_sh,
                                               p.rmin0,
                                               p.switch_flag,
                                               rootpq,
                                               p.jdimpq,
                                               idxu_block,
                                               p.twojmax,
                                               ulist_r,
                                               ulist_i,
                                               dulist_r,
                                               dulist_i);

  double dedr_own_local[3];
  compute_deidrj_parallel_device(p.twojmax,
                                 tid,
                                 block_threads,
                                 warp_id,
                                 lane_id,
                                 idxu_block,
                                 dulist_r,
                                 dulist_i,
                                 ylist_slab_r,
                                 ylist_slab_i,
                                 dedr_warp_sums,
                                 dedr_own_local);
  if (tid == 0) {
    d_dedr_own[b * 3 + 0] = dedr_own_local[0];
    d_dedr_own[b * 3 + 1] = dedr_own_local[1];
    d_dedr_own[b * 3 + 2] = dedr_own_local[2];
  }
  __syncthreads();

  // PEER side.
  const std::size_t jjbase = static_cast<std::size_t>(jj) * static_cast<std::size_t>(p.idxu_max);
  for (int k = tid; k < p.idxu_max; k += block_threads) {
    ylist_slab_r[k] = d_ylist_r[jjbase + k];
    ylist_slab_i[k] = d_ylist_i[jjbase + k];
  }
  __syncthreads();

  snap_detail::compute_uarray_parallel_device(static_cast<unsigned>(tid),
                                              static_cast<unsigned>(block_threads),
                                              -ddx,
                                              -ddy,
                                              -ddz,
                                              z0_sh,
                                              r_sh,
                                              rootpq,
                                              p.jdimpq,
                                              idxu_block,
                                              p.twojmax,
                                              ulist_r,
                                              ulist_i);
  snap_detail::compute_duarray_parallel_device(static_cast<unsigned>(tid),
                                               static_cast<unsigned>(block_threads),
                                               -ddx,
                                               -ddy,
                                               -ddz,
                                               z0_sh,
                                               r_sh,
                                               dz0dr_sh,
                                               wi,
                                               rcut_sh,
                                               p.rmin0,
                                               p.switch_flag,
                                               rootpq,
                                               p.jdimpq,
                                               idxu_block,
                                               p.twojmax,
                                               ulist_r,
                                               ulist_i,
                                               dulist_r,
                                               dulist_i);

  double dedr_peer_local[3];
  compute_deidrj_parallel_device(p.twojmax,
                                 tid,
                                 block_threads,
                                 warp_id,
                                 lane_id,
                                 idxu_block,
                                 dulist_r,
                                 dulist_i,
                                 ylist_slab_r,
                                 ylist_slab_i,
                                 dedr_warp_sums,
                                 dedr_peer_local);
  if (tid == 0) {
    d_dedr_peer[b * 3 + 0] = dedr_peer_local[0];
    d_dedr_peer[b * 3 + 1] = dedr_peer_local[1];
    d_dedr_peer[b * 3 + 2] = dedr_peer_local[2];
  }
}

// ---------------------------------------------------------------------------
// T8.6c-v5 Stage 3 (MixedFast): KERNEL 3b — snap_force_gather_kernel
//
// Per-atom gather. All-FP64 arithmetic (d_dedr_own/peer are doubles, virial
// multiplies are doubles). No narrowing here. Semantically identical to the
// FP64 snap_gpu.cu force-gather — both consume the same per-bond dedr slabs
// in CSR order.
// ---------------------------------------------------------------------------
__global__ void snap_force_gather_kernel(std::uint32_t n_atoms,
                                         const std::uint32_t* __restrict__ atom_bond_start,
                                         const double* __restrict__ bond_dx,
                                         const double* __restrict__ bond_dy,
                                         const double* __restrict__ bond_dz,
                                         const double* __restrict__ d_dedr_own,
                                         const double* __restrict__ d_dedr_peer,
                                         double* __restrict__ d_fx,
                                         double* __restrict__ d_fy,
                                         double* __restrict__ d_fz,
                                         double* __restrict__ d_virial_per_atom) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_atoms) {
    return;
  }
  const std::uint32_t b_begin = atom_bond_start[i];
  const std::uint32_t b_end = atom_bond_start[i + 1];

  double fx_acc = 0.0;
  double fy_acc = 0.0;
  double fz_acc = 0.0;
  double v_xx = 0.0;
  double v_yy = 0.0;
  double v_zz = 0.0;
  double v_xy = 0.0;
  double v_xz = 0.0;
  double v_yz = 0.0;

  for (std::uint32_t b = b_begin; b < b_end; ++b) {
    const std::size_t b3 = static_cast<std::size_t>(b) * 3u;
    const double own_x = d_dedr_own[b3 + 0];
    const double own_y = d_dedr_own[b3 + 1];
    const double own_z = d_dedr_own[b3 + 2];
    const double ddx = bond_dx[b];
    const double ddy = bond_dy[b];
    const double ddz = bond_dz[b];

    fx_acc += own_x;
    fy_acc += own_y;
    fz_acc += own_z;
    v_xx += own_x * ddx;
    v_yy += own_y * ddy;
    v_zz += own_z * ddz;
    v_xy += own_x * ddy;
    v_xz += own_x * ddz;
    v_yz += own_y * ddz;

    fx_acc -= d_dedr_peer[b3 + 0];
    fy_acc -= d_dedr_peer[b3 + 1];
    fz_acc -= d_dedr_peer[b3 + 2];
  }

  d_fx[i] += fx_acc;
  d_fy[i] += fy_acc;
  d_fz[i] += fz_acc;
  d_virial_per_atom[i * 6 + 0] = v_xx;
  d_virial_per_atom[i * 6 + 1] = v_yy;
  d_virial_per_atom[i * 6 + 2] = v_zz;
  d_virial_per_atom[i * 6 + 3] = v_xy;
  d_virial_per_atom[i * 6 + 4] = v_xz;
  d_virial_per_atom[i * 6 + 5] = v_yz;
}
#endif  // !TDMD_SNAP_LEGACY_PERATOM

// Host-side Kahan reduction (D-M6-15 mirror of EAM precedent).
double kahan_sum_host(const double* __restrict__ data, std::size_t n) {
  double s = 0.0;
  double c = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double y = data[i] - c;
    const double t = s + y;
    c = (t - s) - y;
    s = t;
  }
  return s;
}

#endif  // TDMD_BUILD_CUDA

}  // namespace

#if TDMD_BUILD_CUDA

// ---------------------------------------------------------------------------
// SnapGpuMixed::Impl — T8.9 Phase A (narrow FP32 pair-math; mirrors SnapGpu::Impl).
// ---------------------------------------------------------------------------
struct SnapGpuMixed::Impl {
  std::uint64_t compute_version = 0;

  // Persistent per-atom device buffers (grown on demand by DevicePool).
  DevicePtr<std::byte> d_types_bytes;
  DevicePtr<std::byte> d_x_bytes;
  DevicePtr<std::byte> d_y_bytes;
  DevicePtr<std::byte> d_z_bytes;
  DevicePtr<std::byte> d_fx_bytes;
  DevicePtr<std::byte> d_fy_bytes;
  DevicePtr<std::byte> d_fz_bytes;
  DevicePtr<std::byte> d_cell_offsets_bytes;
  DevicePtr<std::byte> d_cell_atoms_bytes;

  DevicePtr<std::byte> d_ulisttot_r_bytes;
  DevicePtr<std::byte> d_ulisttot_i_bytes;
  DevicePtr<std::byte> d_ylist_r_bytes;
  DevicePtr<std::byte> d_ylist_i_bytes;
  DevicePtr<std::byte> d_pe_per_atom_bytes;
  DevicePtr<std::byte> d_virial_per_atom_bytes;

  // T8.6c-v5 Stage 2 + Stage 3 (mixed mirror): per-bond scratch.
  SnapBondListGpu bond_list;
  DevicePtr<std::byte> d_ulist_bond_r_bytes;
  DevicePtr<std::byte> d_ulist_bond_i_bytes;
  DevicePtr<std::byte> d_dedr_own_bytes;
  DevicePtr<std::byte> d_dedr_peer_bytes;

  // SNAP parameter tables — uploaded once, reused across compute() calls.
  DevicePtr<std::byte> d_radius_elem_bytes;
  DevicePtr<std::byte> d_weight_elem_bytes;
  DevicePtr<std::byte> d_beta_bytes;
  DevicePtr<std::byte> d_rcut_sq_ab_bytes;
  DevicePtr<std::byte> d_bzero_bytes;

  // Index tables (from snap_detail::build_flat_tables). Uploaded once per
  // Impl lifetime — twojmax cannot change without reinstantiating the adapter.
  DevicePtr<std::byte> d_idxu_block_bytes;
  DevicePtr<std::byte> d_idxcg_block_bytes;
  DevicePtr<std::byte> d_idxz_block_bytes;
  DevicePtr<std::byte> d_idxb_block_bytes;
  DevicePtr<std::byte> d_idxz_packed_bytes;
  DevicePtr<std::byte> d_idxb_packed_bytes;
  DevicePtr<std::byte> d_cglist_bytes;
  DevicePtr<std::byte> d_rootpq_bytes;

  // Upload-identity cache. Tables re-uploaded only when the caller hands us
  // a pointer triple we haven't seen, OR twojmax changes (sanity check).
  const double* tables_radius_host = nullptr;
  const double* tables_weight_host = nullptr;
  const double* tables_beta_host = nullptr;
  const double* tables_rcut_sq_host = nullptr;
  int tables_twojmax_cached = -1;
  std::uint64_t tables_upload_count = 0;

  // Flat index tables stashed here so we can re-derive idxu_max/idxz_max/
  // idxb_max on the host without round-tripping to the device.
  snap_detail::FlatIndexTables flat;
};

SnapGpuMixed::SnapGpuMixed() : impl_(std::make_unique<Impl>()) {}
SnapGpuMixed::~SnapGpuMixed() = default;
SnapGpuMixed::SnapGpuMixed(SnapGpuMixed&&) noexcept = default;
SnapGpuMixed& SnapGpuMixed::operator=(SnapGpuMixed&&) noexcept = default;

std::uint64_t SnapGpuMixed::compute_version() const noexcept {
  return impl_ ? impl_->compute_version : 0;
}

SnapGpuResult SnapGpuMixed::compute(std::size_t n,
                                    const std::uint32_t* host_types,
                                    const double* host_x,
                                    const double* host_y,
                                    const double* host_z,
                                    std::size_t ncells,
                                    const std::uint32_t* host_cell_offsets,
                                    const std::uint32_t* host_cell_atoms,
                                    const BoxParams& params,
                                    const SnapTablesHost& tables,
                                    double* host_fx_out,
                                    double* host_fy_out,
                                    double* host_fz_out,
                                    DevicePool& pool,
                                    DeviceStream& stream) {
  TDMD_NVTX_RANGE("snap.compute");

  SnapGpuResult result;
  ++impl_->compute_version;
  if (n == 0) {
    return result;
  }
  if (tables.twojmax < 0 || (tables.twojmax % 2) != 0) {
    throw std::invalid_argument(
        "gpu::SnapGpuMixed::compute: twojmax must be non-negative and even");
  }

  cudaStream_t s = raw_stream(stream);

  // --- 1. Ensure index tables match current twojmax.
  if (impl_->tables_twojmax_cached != tables.twojmax) {
    TDMD_NVTX_RANGE("snap.build_index_tables");
    impl_->flat = snap_detail::build_flat_tables(tables.twojmax);
    const auto& ft = impl_->flat;

    const std::size_t idxu_block_bytes = ft.idxu_block.size() * sizeof(int);
    const std::size_t idxcg_block_bytes = ft.idxcg_block.size() * sizeof(int);
    const std::size_t idxz_block_bytes = ft.idxz_block.size() * sizeof(int);
    const std::size_t idxb_block_bytes = ft.idxb_block.size() * sizeof(int);
    const std::size_t idxz_packed_bytes = ft.idxz_packed.size() * sizeof(int);
    const std::size_t idxb_packed_bytes = ft.idxb_packed.size() * sizeof(int);
    const std::size_t cglist_bytes = ft.cglist.size() * sizeof(double);
    const std::size_t rootpq_bytes = ft.rootpq.size() * sizeof(double);

    impl_->d_idxu_block_bytes = pool.allocate_device(idxu_block_bytes, stream);
    impl_->d_idxcg_block_bytes = pool.allocate_device(idxcg_block_bytes, stream);
    impl_->d_idxz_block_bytes = pool.allocate_device(idxz_block_bytes, stream);
    impl_->d_idxb_block_bytes = pool.allocate_device(idxb_block_bytes, stream);
    impl_->d_idxz_packed_bytes = pool.allocate_device(idxz_packed_bytes, stream);
    impl_->d_idxb_packed_bytes = pool.allocate_device(idxb_packed_bytes, stream);
    impl_->d_cglist_bytes = pool.allocate_device(cglist_bytes, stream);
    impl_->d_rootpq_bytes = pool.allocate_device(rootpq_bytes, stream);

    check_cuda("memcpy idxu_block",
               cudaMemcpyAsync(impl_->d_idxu_block_bytes.get(),
                               ft.idxu_block.data(),
                               idxu_block_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy idxcg_block",
               cudaMemcpyAsync(impl_->d_idxcg_block_bytes.get(),
                               ft.idxcg_block.data(),
                               idxcg_block_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy idxz_block",
               cudaMemcpyAsync(impl_->d_idxz_block_bytes.get(),
                               ft.idxz_block.data(),
                               idxz_block_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy idxb_block",
               cudaMemcpyAsync(impl_->d_idxb_block_bytes.get(),
                               ft.idxb_block.data(),
                               idxb_block_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy idxz_packed",
               cudaMemcpyAsync(impl_->d_idxz_packed_bytes.get(),
                               ft.idxz_packed.data(),
                               idxz_packed_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy idxb_packed",
               cudaMemcpyAsync(impl_->d_idxb_packed_bytes.get(),
                               ft.idxb_packed.data(),
                               idxb_packed_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy cglist",
               cudaMemcpyAsync(impl_->d_cglist_bytes.get(),
                               ft.cglist.data(),
                               cglist_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy rootpq",
               cudaMemcpyAsync(impl_->d_rootpq_bytes.get(),
                               ft.rootpq.data(),
                               rootpq_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    impl_->tables_twojmax_cached = tables.twojmax;
  }
  const auto& ft = impl_->flat;
  const int idxu_max = ft.idxu_max;
  const int idxz_max = ft.idxz_max;
  const int idxb_max = ft.idxb_max;
  const int jdim = ft.jdim;
  const int jdimpq = ft.jdimpq;

  // Shared-memory sizing for yi_kernel.
  // Layout: ulisttot_{r,i} (2·idxu_max) + ylist_{r,i} (2·idxu_max)
  //       + zlist_{r,i} (2·idxz_max) + T8.6c ybuf_{r,i} (2·idxz_max).
  // For twojmax=8 (W fixture) this is ~85 KB — above the 48 KB default
  // per-block ceiling, so we opt into the large-shmem path via
  // cudaFuncSetAttribute(MaxDynamicSharedMemorySize) below. On sm_80+
  // the hardware limit is ≥ 164 KB/block (228 KB on sm_90 / sm_120).
  const std::size_t shm_yi_bytes =
      (static_cast<std::size_t>(4 * idxu_max) + static_cast<std::size_t>(4 * idxz_max)) *
      sizeof(double);
  if (shm_yi_bytes > 160u * 1024u) {
    std::ostringstream oss;
    oss << "gpu::SnapGpuMixed::compute: twojmax=" << tables.twojmax << " needs " << shm_yi_bytes
        << " bytes of dynamic shared memory per yi_kernel block, exceeds 160 KB opt-in ceiling";
    throw std::runtime_error(oss.str());
  }
  if (shm_yi_bytes > 48u * 1024u) {
    check_cuda("cudaFuncSetAttribute(snap_yi_kernel, MaxDynamicSharedMemorySize)",
               cudaFuncSetAttribute(snap_yi_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shm_yi_bytes)));
  }

  // --- 2. Upload SNAP parameter tables (cache-checked).
  const std::size_t radius_bytes = tables.n_species * sizeof(double);
  const std::size_t weight_bytes = tables.n_species * sizeof(double);
  const std::size_t beta_total_bytes = tables.n_species * tables.beta_stride * sizeof(double);
  const std::size_t rcut_sq_bytes = tables.n_species * tables.n_species * sizeof(double);
  const bool params_changed = tables.radius_elem != impl_->tables_radius_host ||
                              tables.weight_elem != impl_->tables_weight_host ||
                              tables.beta_coefficients != impl_->tables_beta_host ||
                              tables.rcut_sq_ab != impl_->tables_rcut_sq_host;
  if (params_changed) {
    TDMD_NVTX_RANGE("snap.h2d.params");
    impl_->d_radius_elem_bytes = pool.allocate_device(radius_bytes, stream);
    impl_->d_weight_elem_bytes = pool.allocate_device(weight_bytes, stream);
    impl_->d_beta_bytes = pool.allocate_device(beta_total_bytes, stream);
    impl_->d_rcut_sq_ab_bytes = pool.allocate_device(rcut_sq_bytes, stream);
    check_cuda("memcpy radius_elem",
               cudaMemcpyAsync(impl_->d_radius_elem_bytes.get(),
                               tables.radius_elem,
                               radius_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy weight_elem",
               cudaMemcpyAsync(impl_->d_weight_elem_bytes.get(),
                               tables.weight_elem,
                               weight_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy beta",
               cudaMemcpyAsync(impl_->d_beta_bytes.get(),
                               tables.beta_coefficients,
                               beta_total_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy rcut_sq_ab",
               cudaMemcpyAsync(impl_->d_rcut_sq_ab_bytes.get(),
                               tables.rcut_sq_ab,
                               rcut_sq_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    impl_->tables_radius_host = tables.radius_elem;
    impl_->tables_weight_host = tables.weight_elem;
    impl_->tables_beta_host = tables.beta_coefficients;
    impl_->tables_rcut_sq_host = tables.rcut_sq_ab;
    ++impl_->tables_upload_count;
  }

  // bzero is derived from (bzero_flag, bnorm_flag, twojmax, wself=1.0).
  // Rebuild + reupload every call — small (twojmax+1 doubles).
  std::vector<double> bzero_host(static_cast<std::size_t>(tables.twojmax + 1), 0.0);
  if (tables.bzeroflag) {
    const double www = 1.0 * 1.0 * 1.0;  // wself^3; SNAP uses wself=1.0 always
    for (int j = 0; j <= tables.twojmax; ++j) {
      bzero_host[static_cast<std::size_t>(j)] = tables.bnormflag ? www : www * (j + 1);
    }
  }
  const std::size_t bzero_bytes = bzero_host.size() * sizeof(double);
  impl_->d_bzero_bytes = pool.allocate_device(bzero_bytes, stream);
  check_cuda("memcpy bzero",
             cudaMemcpyAsync(impl_->d_bzero_bytes.get(),
                             bzero_host.data(),
                             bzero_bytes,
                             cudaMemcpyHostToDevice,
                             s));

  // --- 3. H2D per-compute arrays.
  const std::size_t type_bytes = n * sizeof(std::uint32_t);
  const std::size_t pos_bytes = n * sizeof(double);
  const std::size_t cell_offsets_bytes = (ncells + 1) * sizeof(std::uint32_t);
  const std::size_t cell_atoms_bytes = n * sizeof(std::uint32_t);

  impl_->d_types_bytes = pool.allocate_device(type_bytes, stream);
  impl_->d_x_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_y_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_z_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_cell_offsets_bytes = pool.allocate_device(cell_offsets_bytes, stream);
  impl_->d_cell_atoms_bytes = pool.allocate_device(cell_atoms_bytes, stream);
  impl_->d_fx_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_fy_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_fz_bytes = pool.allocate_device(pos_bytes, stream);

  {
    TDMD_NVTX_RANGE("snap.h2d.atoms_cells_forces");
    check_cuda("memcpy types",
               cudaMemcpyAsync(impl_->d_types_bytes.get(),
                               host_types,
                               type_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda(
        "memcpy x",
        cudaMemcpyAsync(impl_->d_x_bytes.get(), host_x, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda(
        "memcpy y",
        cudaMemcpyAsync(impl_->d_y_bytes.get(), host_y, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda(
        "memcpy z",
        cudaMemcpyAsync(impl_->d_z_bytes.get(), host_z, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("memcpy cell_offsets",
               cudaMemcpyAsync(impl_->d_cell_offsets_bytes.get(),
                               host_cell_offsets,
                               cell_offsets_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy cell_atoms",
               cudaMemcpyAsync(impl_->d_cell_atoms_bytes.get(),
                               host_cell_atoms,
                               cell_atoms_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    // Force input side: caller's additive contract — we read host_fx_in and
    // write-back an additively updated value.
    check_cuda("memcpy fx in",
               cudaMemcpyAsync(impl_->d_fx_bytes.get(),
                               host_fx_out,
                               pos_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy fy in",
               cudaMemcpyAsync(impl_->d_fy_bytes.get(),
                               host_fy_out,
                               pos_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy fz in",
               cudaMemcpyAsync(impl_->d_fz_bytes.get(),
                               host_fz_out,
                               pos_bytes,
                               cudaMemcpyHostToDevice,
                               s));
  }

  // --- 4. Per-atom scratch for the three passes.
  const std::size_t per_atom_u_bytes = n * static_cast<std::size_t>(idxu_max) * sizeof(double);
  impl_->d_ulisttot_r_bytes = pool.allocate_device(per_atom_u_bytes, stream);
  impl_->d_ulisttot_i_bytes = pool.allocate_device(per_atom_u_bytes, stream);
  impl_->d_ylist_r_bytes = pool.allocate_device(per_atom_u_bytes, stream);
  impl_->d_ylist_i_bytes = pool.allocate_device(per_atom_u_bytes, stream);
  impl_->d_pe_per_atom_bytes = pool.allocate_device(n * sizeof(double), stream);
  impl_->d_virial_per_atom_bytes = pool.allocate_device(n * 6u * sizeof(double), stream);

  // --- 5. Kernel params.
  DeviceSnapParams dp;
  dp.xlo = params.xlo;
  dp.ylo = params.ylo;
  dp.zlo = params.zlo;
  dp.lx = params.lx;
  dp.ly = params.ly;
  dp.lz = params.lz;
  dp.cell_x = params.cell_x;
  dp.cell_y = params.cell_y;
  dp.cell_z = params.cell_z;
  dp.nx = params.nx;
  dp.ny = params.ny;
  dp.nz = params.nz;
  dp.periodic_x = params.periodic_x ? 1 : 0;
  dp.periodic_y = params.periodic_y ? 1 : 0;
  dp.periodic_z = params.periodic_z ? 1 : 0;
  dp.twojmax = tables.twojmax;
  dp.jdim = jdim;
  dp.jdimpq = jdimpq;
  dp.idxu_max = idxu_max;
  dp.idxz_max = idxz_max;
  dp.idxb_max = idxb_max;
  dp.rcutfac = tables.rcutfac;
  dp.rfac0 = tables.rfac0;
  dp.rmin0 = tables.rmin0;
  dp.switch_flag = tables.switchflag;
  dp.bzero_flag = tables.bzeroflag;
  dp.bnorm_flag = tables.bnormflag;
  dp.wselfall_flag = tables.wselfallflag;
  dp.wself = 1.0;
  dp.n_species = static_cast<std::uint32_t>(tables.n_species);

  const std::uint32_t n32 = static_cast<std::uint32_t>(n);
  constexpr int kThreadsPerBlock = 128;

  auto* d_types = reinterpret_cast<std::uint32_t*>(impl_->d_types_bytes.get());
  auto* d_x = reinterpret_cast<double*>(impl_->d_x_bytes.get());
  auto* d_y = reinterpret_cast<double*>(impl_->d_y_bytes.get());
  auto* d_z = reinterpret_cast<double*>(impl_->d_z_bytes.get());
  auto* d_fx = reinterpret_cast<double*>(impl_->d_fx_bytes.get());
  auto* d_fy = reinterpret_cast<double*>(impl_->d_fy_bytes.get());
  auto* d_fz = reinterpret_cast<double*>(impl_->d_fz_bytes.get());
  auto* d_cell_offsets = reinterpret_cast<std::uint32_t*>(impl_->d_cell_offsets_bytes.get());
  auto* d_cell_atoms = reinterpret_cast<std::uint32_t*>(impl_->d_cell_atoms_bytes.get());

  auto* d_idxu_block = reinterpret_cast<int*>(impl_->d_idxu_block_bytes.get());
  auto* d_idxcg_block = reinterpret_cast<int*>(impl_->d_idxcg_block_bytes.get());
  auto* d_idxz_block = reinterpret_cast<int*>(impl_->d_idxz_block_bytes.get());
  auto* d_idxb_block = reinterpret_cast<int*>(impl_->d_idxb_block_bytes.get());
  auto* d_idxz_packed = reinterpret_cast<int*>(impl_->d_idxz_packed_bytes.get());
  auto* d_idxb_packed = reinterpret_cast<int*>(impl_->d_idxb_packed_bytes.get());
  auto* d_cglist = reinterpret_cast<double*>(impl_->d_cglist_bytes.get());
  auto* d_rootpq = reinterpret_cast<double*>(impl_->d_rootpq_bytes.get());

  auto* d_radius = reinterpret_cast<double*>(impl_->d_radius_elem_bytes.get());
  auto* d_weight = reinterpret_cast<double*>(impl_->d_weight_elem_bytes.get());
  auto* d_beta = reinterpret_cast<double*>(impl_->d_beta_bytes.get());
  auto* d_rcut_sq = reinterpret_cast<double*>(impl_->d_rcut_sq_ab_bytes.get());
  auto* d_bzero = reinterpret_cast<double*>(impl_->d_bzero_bytes.get());

  auto* d_ulisttot_r = reinterpret_cast<double*>(impl_->d_ulisttot_r_bytes.get());
  auto* d_ulisttot_i = reinterpret_cast<double*>(impl_->d_ulisttot_i_bytes.get());
  auto* d_ylist_r = reinterpret_cast<double*>(impl_->d_ylist_r_bytes.get());
  auto* d_ylist_i = reinterpret_cast<double*>(impl_->d_ylist_i_bytes.get());
  auto* d_pe_per_atom = reinterpret_cast<double*>(impl_->d_pe_per_atom_bytes.get());
  auto* d_virial_per_atom = reinterpret_cast<double*>(impl_->d_virial_per_atom_bytes.get());

  // --- 6. Launch kernels.
  const std::size_t shm_de_bytes = static_cast<std::size_t>(10 * idxu_max) * sizeof(double);

#if TDMD_SNAP_LEGACY_PERATOM
  const std::size_t shm_ui_bytes = static_cast<std::size_t>(4 * idxu_max) * sizeof(double);
  {
    TDMD_NVTX_RANGE("snap.ui_kernel");
    snap_ui_kernel<<<n32, kThreadsPerBlock, shm_ui_bytes, s>>>(n32,
                                                               d_types,
                                                               d_x,
                                                               d_y,
                                                               d_z,
                                                               d_cell_offsets,
                                                               d_cell_atoms,
                                                               d_idxu_block,
                                                               d_rootpq,
                                                               d_radius,
                                                               d_weight,
                                                               d_rcut_sq,
                                                               dp,
                                                               d_ulisttot_r,
                                                               d_ulisttot_i);
    check_cuda("launch snap_ui_kernel", cudaGetLastError());
  }
#else
  // T8.6c-v5 Stage 2 (MixedFast): bond list → per-bond ui → per-atom gather.
  {
    TDMD_NVTX_RANGE("snap.bond_list.build");
    impl_->bond_list.build_from_device(n,
                                       d_types,
                                       d_x,
                                       d_y,
                                       d_z,
                                       ncells,
                                       d_cell_offsets,
                                       d_cell_atoms,
                                       d_rcut_sq,
                                       static_cast<std::uint32_t>(tables.n_species),
                                       params,
                                       pool,
                                       stream);
  }
  const auto bond_view = impl_->bond_list.view();
  const std::size_t n_bonds = bond_view.bond_count;
  const std::size_t ulist_bond_bytes =
      (n_bonds == 0) ? 0u : (n_bonds * static_cast<std::size_t>(idxu_max) * sizeof(double));
  if (ulist_bond_bytes > 0) {
    impl_->d_ulist_bond_r_bytes = pool.allocate_device(ulist_bond_bytes, stream);
    impl_->d_ulist_bond_i_bytes = pool.allocate_device(ulist_bond_bytes, stream);
  }
  auto* d_ulist_bond_r =
      (n_bonds > 0) ? reinterpret_cast<double*>(impl_->d_ulist_bond_r_bytes.get()) : nullptr;
  auto* d_ulist_bond_i =
      (n_bonds > 0) ? reinterpret_cast<double*>(impl_->d_ulist_bond_i_bytes.get()) : nullptr;

  const std::size_t shm_ui_bond_bytes = static_cast<std::size_t>(2 * idxu_max) * sizeof(double);
  if (n_bonds > 0) {
    TDMD_NVTX_RANGE("snap.ui_bond_kernel");
    const std::uint32_t n_bonds_u32 = static_cast<std::uint32_t>(n_bonds);
    snap_ui_bond_kernel<<<n_bonds_u32, kThreadsPerBlock, shm_ui_bond_bytes, s>>>(
        n_bonds_u32,
        bond_view.d_bond_type_i,
        bond_view.d_bond_type_j,
        bond_view.d_bond_dx,
        bond_view.d_bond_dy,
        bond_view.d_bond_dz,
        bond_view.d_bond_rsq,
        d_idxu_block,
        d_rootpq,
        d_radius,
        d_weight,
        dp,
        d_ulist_bond_r,
        d_ulist_bond_i);
    check_cuda("launch snap_ui_bond_kernel", cudaGetLastError());
  }
  {
    TDMD_NVTX_RANGE("snap.ui_gather_kernel");
    const std::size_t shm_gather_bytes = static_cast<std::size_t>(2 * idxu_max) * sizeof(double);
    snap_ui_gather_kernel<<<n32, kThreadsPerBlock, shm_gather_bytes, s>>>(
        n32,
        bond_view.d_atom_bond_start,
        d_idxu_block,
        dp,
        d_ulist_bond_r,
        d_ulist_bond_i,
        d_ulisttot_r,
        d_ulisttot_i);
    check_cuda("launch snap_ui_gather_kernel", cudaGetLastError());
  }
#endif
  {
    TDMD_NVTX_RANGE("snap.yi_kernel");
    snap_yi_kernel<<<n32, kThreadsPerBlock, shm_yi_bytes, s>>>(n32,
                                                               d_types,
                                                               d_idxu_block,
                                                               d_idxz_block,
                                                               d_idxb_block,
                                                               d_idxcg_block,
                                                               d_idxz_packed,
                                                               d_idxb_packed,
                                                               d_cglist,
                                                               d_beta,
                                                               tables.beta_stride,
                                                               d_bzero,
                                                               dp,
                                                               d_ulisttot_r,
                                                               d_ulisttot_i,
                                                               d_ylist_r,
                                                               d_ylist_i,
                                                               d_pe_per_atom);
    check_cuda("launch snap_yi_kernel", cudaGetLastError());
  }
#if TDMD_SNAP_LEGACY_PERATOM
  {
    TDMD_NVTX_RANGE("snap.deidrj_kernel");
    snap_deidrj_kernel<<<n32, kThreadsPerBlock, shm_de_bytes, s>>>(n32,
                                                                   d_types,
                                                                   d_x,
                                                                   d_y,
                                                                   d_z,
                                                                   d_cell_offsets,
                                                                   d_cell_atoms,
                                                                   d_idxu_block,
                                                                   d_rootpq,
                                                                   d_radius,
                                                                   d_weight,
                                                                   d_rcut_sq,
                                                                   d_ylist_r,
                                                                   d_ylist_i,
                                                                   dp,
                                                                   d_fx,
                                                                   d_fy,
                                                                   d_fz,
                                                                   d_virial_per_atom);
    check_cuda("launch snap_deidrj_kernel", cudaGetLastError());
  }
#else
  // T8.6c-v5 Stage 3 (mixed mirror): per-bond deidrj + per-atom force gather.
  const std::size_t dedr_bond_bytes = (n_bonds == 0) ? 0u : (n_bonds * 3u * sizeof(double));
  if (dedr_bond_bytes > 0) {
    impl_->d_dedr_own_bytes = pool.allocate_device(dedr_bond_bytes, stream);
    impl_->d_dedr_peer_bytes = pool.allocate_device(dedr_bond_bytes, stream);
  }
  auto* d_dedr_own =
      (n_bonds > 0) ? reinterpret_cast<double*>(impl_->d_dedr_own_bytes.get()) : nullptr;
  auto* d_dedr_peer =
      (n_bonds > 0) ? reinterpret_cast<double*>(impl_->d_dedr_peer_bytes.get()) : nullptr;

  if (n_bonds > 0) {
    TDMD_NVTX_RANGE("snap.deidrj_bond_kernel");
    const std::uint32_t n_bonds_u32 = static_cast<std::uint32_t>(n_bonds);
    snap_deidrj_bond_kernel<<<n_bonds_u32, kThreadsPerBlock, shm_de_bytes, s>>>(
        n_bonds_u32,
        bond_view.d_bond_i,
        bond_view.d_bond_j,
        bond_view.d_bond_type_i,
        bond_view.d_bond_type_j,
        bond_view.d_bond_dx,
        bond_view.d_bond_dy,
        bond_view.d_bond_dz,
        bond_view.d_bond_rsq,
        d_idxu_block,
        d_rootpq,
        d_radius,
        d_weight,
        d_ylist_r,
        d_ylist_i,
        dp,
        d_dedr_own,
        d_dedr_peer);
    check_cuda("launch snap_deidrj_bond_kernel", cudaGetLastError());
  }

  {
    TDMD_NVTX_RANGE("snap.force_gather_kernel");
    const unsigned grid = (n32 + kThreadsPerBlock - 1u) / kThreadsPerBlock;
    snap_force_gather_kernel<<<grid, kThreadsPerBlock, 0, s>>>(n32,
                                                               bond_view.d_atom_bond_start,
                                                               bond_view.d_bond_dx,
                                                               bond_view.d_bond_dy,
                                                               bond_view.d_bond_dz,
                                                               d_dedr_own,
                                                               d_dedr_peer,
                                                               d_fx,
                                                               d_fy,
                                                               d_fz,
                                                               d_virial_per_atom);
    check_cuda("launch snap_force_gather_kernel", cudaGetLastError());
  }
#endif

  // --- 7. D2H forces + per-atom reductions.
  std::vector<double> host_pe(n);
  std::vector<double> host_virial(n * 6u);
  {
    TDMD_NVTX_RANGE("snap.d2h.forces_and_reductions");
    check_cuda("D2H fx", cudaMemcpyAsync(host_fx_out, d_fx, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("D2H fy", cudaMemcpyAsync(host_fy_out, d_fy, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("D2H fz", cudaMemcpyAsync(host_fz_out, d_fz, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("D2H pe",
               cudaMemcpyAsync(host_pe.data(),
                               d_pe_per_atom,
                               n * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("D2H virial",
               cudaMemcpyAsync(host_virial.data(),
                               d_virial_per_atom,
                               n * 6u * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("stream sync SNAP", cudaStreamSynchronize(s));
  }

  // --- 8. Host Kahan reductions (D-M6-15).
  result.potential_energy = kahan_sum_host(host_pe.data(), n);

  double vsum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double vcomp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t base = i * 6u;
    for (int k = 0; k < 6; ++k) {
      const double y = host_virial[base + static_cast<std::size_t>(k)] - vcomp[k];
      const double t = vsum[k] + y;
      vcomp[k] = (t - vsum[k]) - y;
      vsum[k] = t;
    }
  }
  for (int k = 0; k < 6; ++k) {
    result.virial[k] = vsum[k];
  }

  return result;
}

#else  // CPU-only build

struct SnapGpuMixed::Impl {};

SnapGpuMixed::SnapGpuMixed() : impl_(std::make_unique<Impl>()) {}
SnapGpuMixed::~SnapGpuMixed() = default;
SnapGpuMixed::SnapGpuMixed(SnapGpuMixed&&) noexcept = default;
SnapGpuMixed& SnapGpuMixed::operator=(SnapGpuMixed&&) noexcept = default;

std::uint64_t SnapGpuMixed::compute_version() const noexcept {
  return 0;
}

SnapGpuResult SnapGpuMixed::compute(std::size_t /*n*/,
                                    const std::uint32_t* /*host_types*/,
                                    const double* /*host_x*/,
                                    const double* /*host_y*/,
                                    const double* /*host_z*/,
                                    std::size_t /*ncells*/,
                                    const std::uint32_t* /*host_cell_offsets*/,
                                    const std::uint32_t* /*host_cell_atoms*/,
                                    const BoxParams& /*params*/,
                                    const SnapTablesHost& /*tables*/,
                                    double* /*host_fx_out*/,
                                    double* /*host_fy_out*/,
                                    double* /*host_fz_out*/,
                                    DevicePool& /*pool*/,
                                    DeviceStream& /*stream*/) {
  throw std::runtime_error(
      "gpu::SnapGpuMixed::compute: CPU-only build (TDMD_BUILD_CUDA=0); CUDA not linked");
}

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
