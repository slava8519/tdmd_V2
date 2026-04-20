// SPEC: docs/specs/gpu/SPEC.md §7.3 (SNAP GPU kernel contract — T8.6 authors);
//       §6.3 (D-M6-7 bit-exact gate extended to SNAP at T8.7),
//       §8.1 (Reference FP64-only, no atomicAdd(double)), §9 (NVTX on every launch)
// Module SPEC: docs/specs/potentials/SPEC.md §6 (SNAP module contract)
// Exec pack: docs/development/m8_execution_pack.md T8.6b (kernel body)
// Pre-impl:  docs/development/t8.6b_pre_impl.md
// Decisions: D-M6-17 (PIMPL firewall), D-M8-13 (GPU Fp64 ≡ CPU Fp64 ≤ 1e-12
//            rel — exercised at T8.7), D-M6-15 (host-side Kahan reductions)
//
// T8.6b body. Three-kernel per-atom SNAP force path:
//
//   snap_ui_kernel       — Wigner-U accumulation: d_ulisttot_r/i[i, 0..idxu_max)
//                            Each block (one atom) walks its 3×3×3 neighbour
//                            stencil, calls compute_uarray_device per pair,
//                            accumulates sfac·wj·U into a per-atom sum.
//                            Self-identity seeded per CPU zero_uarraytot.
//
//   snap_yi_kernel       — Z-list + Y-list + B-list + per-atom PE.
//                            Mirrors CPU compute_yi (VMK z-contraction) writing
//                            d_ylist_r/i[i,·] and — re-using the same zlist
//                            stashed in shared memory — compute_bi producing
//                            blist, then evdwl_i = β₀ + Σ β_k · blist[k].
//                            One block per atom, purely independent per atom
//                            (ylist[i] uses only ulisttot[i]).
//
//   snap_deidrj_kernel   — Per-neighbour dE/dr + Newton-3 force reassembly.
//                            Block i walks its 3×3×3 neighbours. For each pair
//                            (i,j) it runs compute_deidrj TWICE:
//                              (a) own side: dulist(i→j) with wj_j, contracted
//                                  with d_ylist[i]. Accumulate into F_i, virial.
//                              (b) peer side: dulist(j→i) with wj_i, contracted
//                                  with d_ylist[j] (read from global). Subtract
//                                  from F_i (Newton-3 deposit that CPU would
//                                  have done in atom j's outer iteration).
//                            Virial is accumulated own-side only, matching CPU
//                            convention Σ_i Σ_{j∈ninside(i)} fij_own·(r_j−r_i).
//
// No `atomicAdd(double*, double)` anywhere — every global write is owned by a
// single block. Byte-exactness with CPU at ≤1e-12 rel is the T8.7 gate
// (D-M8-13) and not claimed by T8.6b; functional correctness + stable NVE is
// the T8.6b acceptance (see t8.6b_pre_impl.md §9).
//
// Shared-memory budget (dynamic, per block):
//   ui:     4·idxu_max·8   =   4·165·8 = 5.3 KB (twojmax=6)
//   yi:     (4·idxu_max + 2·idxz_max)·8 = 22.7 KB (twojmax=6)
//   deidrj: 10·idxu_max·8  =  10·165·8 = 13.2 KB (twojmax=6)
// All below 48 KB default. twojmax>8 runtime-rejected (see Impl::compute).
//
// CPU/CUDA symmetric structure (matches EAM precedent): CUDA path inside
// `#if TDMD_BUILD_CUDA`, CPU-only stub in `#else`. Stub throws a
// `std::runtime_error` with "CPU-only build" wording matching EAM.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"
#include "tdmd/gpu/snap_gpu.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/telemetry/nvtx.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

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
  oss << "gpu::SnapGpu::" << op << ": " << cudaGetErrorName(err) << " (" << cudaGetErrorString(err)
      << ")";
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

// Flat 3D address into jdim×jdim×jdim index blocks (idxcg/idxz/idxb _block).
__device__ __forceinline__ std::size_t jkk_index_dev(int j1, int j2, int j, int jdim) {
  return static_cast<std::size_t>(j1) * static_cast<std::size_t>(jdim) *
             static_cast<std::size_t>(jdim) +
         static_cast<std::size_t>(j2) * static_cast<std::size_t>(jdim) +
         static_cast<std::size_t>(j);
}

// ---------------------------------------------------------------------------
// compute_deidrj_device — port of SnaEngine::compute_deidrj (sna.cpp 620-672).
// Single-elem case (nelements_=1): jelem=0, ylist indexed at jju.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_deidrj_device(int twojmax,
                                                      const int* __restrict__ idxu_block,
                                                      const double* __restrict__ dulist_r,
                                                      const double* __restrict__ dulist_i,
                                                      const double* __restrict__ ylist_r,
                                                      const double* __restrict__ ylist_i,
                                                      double* dedr) {
  dedr[0] = 0.0;
  dedr[1] = 0.0;
  dedr[2] = 0.0;
  for (int j = 0; j <= twojmax; ++j) {
    int jju = idxu_block[j];
    for (int mb = 0; 2 * mb < j; ++mb) {
      for (int ma = 0; ma <= j; ++ma) {
        const double yr = ylist_r[jju];
        const double yi = ylist_i[jju];
        dedr[0] += dulist_r[jju * 3 + 0] * yr + dulist_i[jju * 3 + 0] * yi;
        dedr[1] += dulist_r[jju * 3 + 1] * yr + dulist_i[jju * 3 + 1] * yi;
        dedr[2] += dulist_r[jju * 3 + 2] * yr + dulist_i[jju * 3 + 2] * yi;
        jju++;
      }
    }
    if ((j % 2) == 0) {
      int mb = j / 2;
      for (int ma = 0; ma < mb; ++ma) {
        const double yr = ylist_r[jju];
        const double yi = ylist_i[jju];
        dedr[0] += dulist_r[jju * 3 + 0] * yr + dulist_i[jju * 3 + 0] * yi;
        dedr[1] += dulist_r[jju * 3 + 1] * yr + dulist_i[jju * 3 + 1] * yi;
        dedr[2] += dulist_r[jju * 3 + 2] * yr + dulist_i[jju * 3 + 2] * yi;
        jju++;
      }
      const double yr = ylist_r[jju];
      const double yi = ylist_i[jju];
      dedr[0] += (dulist_r[jju * 3 + 0] * yr + dulist_i[jju * 3 + 0] * yi) * 0.5;
      dedr[1] += (dulist_r[jju * 3 + 1] * yr + dulist_i[jju * 3 + 1] * yi) * 0.5;
      dedr[2] += (dulist_r[jju * 3 + 2] * yr + dulist_i[jju * 3 + 2] * yi) * 0.5;
    }
  }
  dedr[0] *= 2.0;
  dedr[1] *= 2.0;
  dedr[2] *= 2.0;
}

// ---------------------------------------------------------------------------
// KERNEL 1: snap_ui_kernel
//
// One block per atom, 128 threads. Each block accumulates d_ulisttot_r/i[i,·]
// by walking its 3³ cell stencil and calling compute_uarray per in-cutoff pair.
// The U accumulation is FP-sensitive (recurrence + accumulation order), so
// the compute_uarray + add-to-ulisttot work runs single-lane (tid==0). Zeroing
// and the final global write are thread-parallel across the idxu_max slab.
// ---------------------------------------------------------------------------
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

          if (tid == 0) {
            const double r = sqrt(rsq);
            const double radj = radius_elem[jtype];
            const double wj = weight_elem[jtype];
            const double rcut = (radi + radj) * p.rcutfac;
            const double theta0 = (r - p.rmin0) * p.rfac0 * M_PI / (rcut - p.rmin0);
            const double z0 = r / tan(theta0);
            snap_detail::compute_uarray_device(ddx,
                                               ddy,
                                               ddz,
                                               z0,
                                               r,
                                               rootpq,
                                               p.jdimpq,
                                               idxu_block,
                                               p.twojmax,
                                               ulist_r,
                                               ulist_i);
            const double sfac = snap_detail::compute_sfac_device(r, rcut, p.rmin0, p.switch_flag);
            const double sfacwj = sfac * wj;
            // Port of add_uarraytot (sna.cpp 806-830) with jelem=0.
            for (int jl = 0; jl <= p.twojmax; ++jl) {
              int jju = idxu_block[jl];
              for (int mb = 0; mb <= jl; ++mb) {
                for (int ma = 0; ma <= jl; ++ma) {
                  ulisttot_r[jju] += sfacwj * ulist_r[jju];
                  ulisttot_i[jju] += sfacwj * ulist_i[jju];
                  jju++;
                }
              }
            }
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
  // Single-lane over jjz. nelements_=1 so we only have elem1=elem2=elem3=0.
  if (tid == 0) {
    for (int jjz = 0; jjz < p.idxz_max; ++jjz) {
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
      const int jju = zrec[9];

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

      ylist_r[jju] += betaj * ztmp_r;
      ylist_i[jju] += betaj * ztmp_i;
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
// KERNEL 3: snap_deidrj_kernel
//
// One block per atom, 128 threads. Writes per-atom fx/fy/fz + per-atom virial.
// Each block walks i's 3³ cells, runs compute_deidrj twice per pair (own side
// using ylist[i], peer side using ylist[j] read from global).
// ---------------------------------------------------------------------------
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

  // Per-atom accumulators — single-lane so we don't need warp reductions.
  // (Fine for T8.6b correctness gate; warp-level parallel path is a T8.6c
  // perf opt once the byte-exact gate lands at T8.7.)
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

          if (tid == 0) {
            const double r = sqrt(rsq);
            const double radj = radius_elem[jtype];
            const double wj = weight_elem[jtype];
            const double rcut = (radi + radj) * p.rcutfac;
            const double rscale0 = p.rfac0 * M_PI / (rcut - p.rmin0);
            const double theta0 = (r - p.rmin0) * rscale0;
            const double cs = cos(theta0);
            const double sn = sin(theta0);
            const double z0 = r * cs / sn;
            const double dz0dr = z0 / r - (r * rscale0) * (rsq + z0 * z0) / rsq;

            // --- OWN side: dulist(i→j) · ylist[i]. Sign: (ddx, ddy, ddz).
            snap_detail::compute_uarray_device(ddx,
                                               ddy,
                                               ddz,
                                               z0,
                                               r,
                                               rootpq,
                                               p.jdimpq,
                                               idxu_block,
                                               p.twojmax,
                                               ulist_r,
                                               ulist_i);
            snap_detail::compute_duarray_device(ddx,
                                                ddy,
                                                ddz,
                                                z0,
                                                r,
                                                dz0dr,
                                                wj,
                                                rcut,
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
            double dedr_own[3];
            compute_deidrj_device(p.twojmax,
                                  idxu_block,
                                  dulist_r,
                                  dulist_i,
                                  ylist_i_r,
                                  ylist_i_i,
                                  dedr_own);
            // CPU: F_i += fij, F_j -= fij, virial += fij · (r_j - r_i).
            // We're atom i accumulating i's F + i's virial share.
            fx_acc += dedr_own[0];
            fy_acc += dedr_own[1];
            fz_acc += dedr_own[2];
            v_xx += dedr_own[0] * ddx;
            v_yy += dedr_own[1] * ddy;
            v_zz += dedr_own[2] * ddz;
            v_xy += dedr_own[0] * ddy;
            v_xz += dedr_own[0] * ddz;
            v_yz += dedr_own[1] * ddz;

            // --- PEER side: from atom j's perspective, pair (j→i).
            // At j's CPU outer loop, rij[jj] = r_i - r_j = -(ddx, ddy, ddz);
            // wj[jj] = weight_elem[itype] = wi; theta0/z0/dz0dr are unchanged
            // (depend only on r and rcut which is symmetric). ylist used is
            // the peer's own ylist = d_ylist[j].
            const std::size_t jbase =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(p.idxu_max);
            snap_detail::compute_uarray_device(-ddx,
                                               -ddy,
                                               -ddz,
                                               z0,
                                               r,
                                               rootpq,
                                               p.jdimpq,
                                               idxu_block,
                                               p.twojmax,
                                               ulist_r,
                                               ulist_i);
            snap_detail::compute_duarray_device(-ddx,
                                                -ddy,
                                                -ddz,
                                                z0,
                                                r,
                                                dz0dr,
                                                wi,
                                                rcut,
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
            double dedr_peer[3];
            compute_deidrj_device(p.twojmax,
                                  idxu_block,
                                  dulist_r,
                                  dulist_i,
                                  d_ylist_r + jbase,
                                  d_ylist_i + jbase,
                                  dedr_peer);
            // CPU at j's outer loop: F_j += dedr_peer, F_i -= dedr_peer.
            // We accumulate the "-= peer" side at atom i.
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
    // Additive write per SnapGpu::compute() caller contract (host reads
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
// SnapGpu::Impl — T8.6b full body.
// ---------------------------------------------------------------------------
struct SnapGpu::Impl {
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

SnapGpu::SnapGpu() : impl_(std::make_unique<Impl>()) {}
SnapGpu::~SnapGpu() = default;
SnapGpu::SnapGpu(SnapGpu&&) noexcept = default;
SnapGpu& SnapGpu::operator=(SnapGpu&&) noexcept = default;

std::uint64_t SnapGpu::compute_version() const noexcept {
  return impl_ ? impl_->compute_version : 0;
}

SnapGpuResult SnapGpu::compute(std::size_t n,
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
    throw std::invalid_argument("gpu::SnapGpu::compute: twojmax must be non-negative and even");
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

  // Shared-memory sanity: enforce twojmax ≤ 8 (keeps yi_kernel shmem < 48 KB).
  // Beyond twojmax=8 the yi_kernel needs dynamic shmem > 48KB and requires
  // cudaFuncSetAttribute opt-in — not wired in T8.6b (M8 scope is twojmax≤6).
  const std::size_t shm_yi_bytes =
      (static_cast<std::size_t>(4 * idxu_max) + static_cast<std::size_t>(2 * idxz_max)) *
      sizeof(double);
  if (shm_yi_bytes > 48u * 1024u) {
    std::ostringstream oss;
    oss << "gpu::SnapGpu::compute: twojmax=" << tables.twojmax << " needs " << shm_yi_bytes
        << " bytes of dynamic shared memory per yi_kernel block, exceeds 48 KB default limit "
           "(T8.6b M8 scope targets twojmax ≤ 8; raise cudaFuncAttributeMax"
           "DynamicSharedMemorySize on Ampere+ to lift this)";
    throw std::runtime_error(oss.str());
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
  const std::size_t shm_ui_bytes = static_cast<std::size_t>(4 * idxu_max) * sizeof(double);
  const std::size_t shm_de_bytes = static_cast<std::size_t>(10 * idxu_max) * sizeof(double);

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

struct SnapGpu::Impl {};

SnapGpu::SnapGpu() : impl_(std::make_unique<Impl>()) {}
SnapGpu::~SnapGpu() = default;
SnapGpu::SnapGpu(SnapGpu&&) noexcept = default;
SnapGpu& SnapGpu::operator=(SnapGpu&&) noexcept = default;

std::uint64_t SnapGpu::compute_version() const noexcept {
  return 0;
}

SnapGpuResult SnapGpu::compute(std::size_t /*n*/,
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
      "gpu::SnapGpu::compute: CPU-only build (TDMD_BUILD_CUDA=0); CUDA not linked");
}

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
