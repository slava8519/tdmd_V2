// SPEC: docs/specs/gpu/SPEC.md §7.2 (EAM three-pass, reused), §8.2
//       (MixedFastBuild — Philosophy B), §6.3 (D-M6-7 does NOT apply to
//       Mixed — D-M6-8 threshold applies instead)
// Master spec: §D.1 Philosophy B — FP32 math pipeline + FP64 accumulators
// Module SPEC: docs/specs/potentials/SPEC.md §4.1–§4.4
// Exec pack: docs/development/m6_execution_pack.md T6.8
// Decisions: D-M6-5 (MixedFast compile-time build flavor),
//            D-M6-8 (rel force ≤ 1e-6, rel energy ≤ 1e-8 vs Fp64Reference GPU),
//            D-M6-15 (canonical reduction order preserved across flavors).
//
// FP32/FP64 split inside the three kernels:
//   FP64 (retained): atom positions x/y/z, per-atom accumulators (ρ, F(ρ),
//   fx/fy/fz, pe, virial), spline coefficient tables in device memory, host
//   Kahan reductions.
//
//   FP32 (Mixed): per-pair delta subtraction promotes the FP64 difference to
//   FP32 only for the r²/r/1/r path; spline locate + Horner eval in FP32
//   (reading FP64 table cells and casting on read); per-pair φ/φ'/dE/dr/
//   fscalar/fij_xyz.
//
// The FP32 pair contributions are cast back to FP64 before adding into the
// per-atom accumulator so summation precision across hundreds of neighbours
// retains 53-bit mantissa. This matches Philosophy B "wide accumulation over
// narrow math".

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/eam_alloy_gpu.hpp"  // reuses EamAlloyTablesHost, EamAlloyGpuResult
#include "tdmd/gpu/eam_alloy_gpu_mixed.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/telemetry/nvtx.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#if TDMD_BUILD_CUDA
#include "cuda_handles.hpp"

#include <cuda_runtime.h>
#endif

namespace tdmd::gpu {

namespace {

#if TDMD_BUILD_CUDA

[[noreturn]] void throw_cuda_mixed(const char* op, cudaError_t err) {
  std::ostringstream oss;
  oss << "gpu::EamAlloyGpuMixed::" << op << ": " << cudaGetErrorName(err) << " ("
      << cudaGetErrorString(err) << ")";
  throw std::runtime_error(oss.str());
}

void check_cuda_mixed(const char* op, cudaError_t err) {
  if (err != cudaSuccess) {
    throw_cuda_mixed(op, err);
  }
}

// POD captured by the mixed kernels. Stores both FP64 and FP32 variants of
// the scalars the kernels touch in the hot path — FP32 versions are
// host-precomputed casts so kernels never need explicit per-call casting.
struct DeviceEamParamsMixed {
  // Box + cell grid (FP64 — used in cell-index math which keeps FP64)
  double xlo, ylo, zlo;
  double lx, ly, lz;
  double cell_x, cell_y, cell_z;
  std::uint32_t nx, ny, nz;
  int periodic_x, periodic_y, periodic_z;
  // FP32 mirrors of the above that the FP32 pair-math path reads:
  float lx_f, ly_f, lz_f;
  // EAM cutoff² — FP32 for comparison with FP32 r²
  float cutoff_sq_f;
  // Spline grid scalars (FP64 for locate(), result cast to FP32 for eval)
  std::uint32_t n_species;
  std::uint32_t nrho;
  std::uint32_t nr;
  double F_x0, F_rdx;
  double r_x0, r_rdx;
};

// ------------ Shared device helpers (FP64 — unchanged vs Reference) --------

__device__ __forceinline__ std::uint32_t wrap_axis_dev_m(int idx, std::uint32_t n) {
  const int ni = static_cast<int>(n);
  int w = idx % ni;
  if (w < 0) {
    w += ni;
  }
  return static_cast<std::uint32_t>(w);
}

__device__ __forceinline__ std::size_t cell_index_axis_dev_m(double coord,
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

__device__ __forceinline__ std::size_t linear_index_dev_m(std::uint32_t ix,
                                                          std::uint32_t iy,
                                                          std::uint32_t iz,
                                                          std::uint32_t nx,
                                                          std::uint32_t ny) {
  return static_cast<std::size_t>(ix) +
         static_cast<std::size_t>(nx) *
             (static_cast<std::size_t>(iy) +
              static_cast<std::size_t>(ny) * static_cast<std::size_t>(iz));
}

// Minimum-image stays FP64: position deltas can be large relative to FP32 ULP
// when crossing a periodic boundary on a 50+ Å box, and the correction term is
// only evaluated once per pair.
__device__ __forceinline__ double minimum_image_axis_dev_m(double delta, double len, int periodic) {
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

// ------------ Spline eval — Horner form in FP64, return cast to FP32 ------
// Coefficients on real EAM potentials (Al z2r, Ni-Al Mishin-2004) span ~3
// decades with alternating signs, so FP32 Horner over 3 cubic terms hits
// catastrophic cancellation and blows the D-M6-8 threshold by ~9×. We keep
// the table read + Horner in FP64 and only downcast at exit; that costs ~16
// FP64 FLOPs per pair vs the pair-math savings (FP32 sqrt + 1/r + fscalar
// + fij_xyz, ~12 FLOPs) so we still keep the bulk of the Mixed speedup. The
// FP32 pair math itself (ddx_f, r_f, inv_r_f, fscalar_f, fij_*_f) is
// preserved — that's where Philosophy B actually pays off throughput-wise.

__device__ __forceinline__ void locate_dev_m(double x,
                                             double x0,
                                             double rdx,
                                             std::uint32_t n,
                                             std::size_t& i_out,
                                             double& p_out) {
  const double p_raw = (x - x0) * rdx + 1.0;
  long long m = static_cast<long long>(p_raw);
  const long long m_min = 1;
  const long long m_max = static_cast<long long>(n) - 1;
  if (m < m_min) {
    m = m_min;
  } else if (m > m_max) {
    m = m_max;
  }
  double p = p_raw - static_cast<double>(m);
  if (p > 1.0) {
    p = 1.0;
  } else if (p < 0.0) {
    p = 0.0;
  }
  i_out = static_cast<std::size_t>(m - 1);
  p_out = p;
}

__device__ __forceinline__ double spline_eval_dev_m(const double* __restrict__ table_coeffs,
                                                    double x,
                                                    double x0,
                                                    double rdx,
                                                    std::uint32_t n) {
  std::size_t i;
  double p;
  locate_dev_m(x, x0, rdx, n, i, p);
  const double* c = table_coeffs + i * 7;
  return ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
}

__device__ __forceinline__ double spline_deriv_dev_m(const double* __restrict__ table_coeffs,
                                                     double x,
                                                     double x0,
                                                     double rdx,
                                                     std::uint32_t n) {
  std::size_t i;
  double p;
  locate_dev_m(x, x0, rdx, n, i, p);
  const double* c = table_coeffs + i * 7;
  return (c[0] * p + c[1]) * p + c[2];
}

__device__ __forceinline__ std::size_t pair_index_dev_m(std::uint32_t a, std::uint32_t b) {
  std::uint32_t hi = a > b ? a : b;
  std::uint32_t lo = a > b ? b : a;
  return static_cast<std::size_t>(hi) * (static_cast<std::size_t>(hi) + 1) / 2 +
         static_cast<std::size_t>(lo);
}

// ------------ Kernel 1: density pass (Mixed) -------------------------------
__global__ void density_kernel_mixed(std::uint32_t n,
                                     const std::uint32_t* __restrict__ types,
                                     const double* __restrict__ x,
                                     const double* __restrict__ y,
                                     const double* __restrict__ z,
                                     const std::uint32_t* __restrict__ cell_offsets,
                                     const std::uint32_t* __restrict__ cell_atoms,
                                     const double* __restrict__ rho_coeffs,
                                     DeviceEamParamsMixed p,
                                     double* __restrict__ rho_out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];

  const auto ix_u = static_cast<int>(cell_index_axis_dev_m(xi, p.xlo, p.cell_x, p.nx));
  const auto iy_u = static_cast<int>(cell_index_axis_dev_m(yi, p.ylo, p.cell_y, p.ny));
  const auto iz_u = static_cast<int>(cell_index_axis_dev_m(zi, p.zlo, p.cell_z, p.nz));

  const std::size_t rho_stride = static_cast<std::size_t>(p.nr) * 7u;

  double rho_i = 0.0;  // FP64 per-atom accumulator
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const std::uint32_t jx = wrap_axis_dev_m(ix_u + dx, p.nx);
        const std::uint32_t jy = wrap_axis_dev_m(iy_u + dy, p.ny);
        const std::uint32_t jz = wrap_axis_dev_m(iz_u + dz, p.nz);
        const std::size_t cj = linear_index_dev_m(jx, jy, jz, p.nx, p.ny);

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
          ddx = minimum_image_axis_dev_m(ddx, p.lx, p.periodic_x);
          ddy = minimum_image_axis_dev_m(ddy, p.ly, p.periodic_y);
          ddz = minimum_image_axis_dev_m(ddz, p.lz, p.periodic_z);
          // FP32 pair-math path
          const float ddx_f = static_cast<float>(ddx);
          const float ddy_f = static_cast<float>(ddy);
          const float ddz_f = static_cast<float>(ddz);
          const float r2_f = ddx_f * ddx_f + ddy_f * ddy_f + ddz_f * ddz_f;
          if (r2_f > p.cutoff_sq_f) {
            continue;
          }
          const double r = static_cast<double>(sqrtf(r2_f));
          const std::uint32_t type_j = types[j];
          const double* rho_tab = rho_coeffs + static_cast<std::size_t>(type_j) * rho_stride;
          const double rho_contrib = spline_eval_dev_m(rho_tab, r, p.r_x0, p.r_rdx, p.nr);
          rho_i += rho_contrib;
        }
      }
    }
  }
  rho_out[i] = rho_i;
}

// ------------ Kernel 2: embedding pass (FP64 — unchanged) ------------------
// Per-atom-only pass has no FLOPs-per-pair bandwidth advantage from FP32;
// keep FP64 to preserve F'(ρ) precision for the force kernel.
__global__ void embedding_kernel_mixed(std::uint32_t n,
                                       const std::uint32_t* __restrict__ types,
                                       const double* __restrict__ rho,
                                       const double* __restrict__ F_coeffs,
                                       DeviceEamParamsMixed p,
                                       double* __restrict__ dFdrho_out,
                                       double* __restrict__ pe_embed_out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const std::size_t F_stride = static_cast<std::size_t>(p.nrho) * 7u;
  const std::uint32_t type_i = types[i];
  const double rho_i = rho[i];
  const double* F_tab = F_coeffs + static_cast<std::size_t>(type_i) * F_stride;

  // Reuse the FP64 locate: the embedding-table argument ρ is a per-atom
  // accumulator and needs full FP64 precision here.
  const double p_raw = (rho_i - p.F_x0) * p.F_rdx + 1.0;
  long long m = static_cast<long long>(p_raw);
  const long long m_min = 1;
  const long long m_max = static_cast<long long>(p.nrho) - 1;
  if (m < m_min) {
    m = m_min;
  } else if (m > m_max) {
    m = m_max;
  }
  double pp = p_raw - static_cast<double>(m);
  if (pp > 1.0) {
    pp = 1.0;
  } else if (pp < 0.0) {
    pp = 0.0;
  }
  const std::size_t cell_i = static_cast<std::size_t>(m - 1);
  const double* c = F_tab + cell_i * 7;
  pe_embed_out[i] = ((c[3] * pp + c[4]) * pp + c[5]) * pp + c[6];
  dFdrho_out[i] = (c[0] * pp + c[1]) * pp + c[2];
}

// ------------ Kernel 3: force pass (Mixed) ---------------------------------
__global__ void force_kernel_mixed(std::uint32_t n,
                                   const std::uint32_t* __restrict__ types,
                                   const double* __restrict__ x,
                                   const double* __restrict__ y,
                                   const double* __restrict__ z,
                                   const std::uint32_t* __restrict__ cell_offsets,
                                   const std::uint32_t* __restrict__ cell_atoms,
                                   const double* __restrict__ dFdrho,
                                   const double* __restrict__ rho_coeffs,
                                   const double* __restrict__ z2r_coeffs,
                                   DeviceEamParamsMixed p,
                                   double* __restrict__ fx_out,
                                   double* __restrict__ fy_out,
                                   double* __restrict__ fz_out,
                                   double* __restrict__ pe_pair_out,
                                   double* __restrict__ virial_out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];
  const std::uint32_t type_i = types[i];
  const double dF_i = dFdrho[i];

  const auto ix_u = static_cast<int>(cell_index_axis_dev_m(xi, p.xlo, p.cell_x, p.nx));
  const auto iy_u = static_cast<int>(cell_index_axis_dev_m(yi, p.ylo, p.cell_y, p.ny));
  const auto iz_u = static_cast<int>(cell_index_axis_dev_m(zi, p.zlo, p.cell_z, p.nz));

  const std::size_t rho_stride = static_cast<std::size_t>(p.nr) * 7u;
  const std::size_t z2r_stride = static_cast<std::size_t>(p.nr) * 7u;

  double fx_acc = 0.0;
  double fy_acc = 0.0;
  double fz_acc = 0.0;
  double pe_acc = 0.0;
  double v_xx = 0.0;
  double v_yy = 0.0;
  double v_zz = 0.0;
  double v_xy = 0.0;
  double v_xz = 0.0;
  double v_yz = 0.0;

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const std::uint32_t jx = wrap_axis_dev_m(ix_u + dx, p.nx);
        const std::uint32_t jy = wrap_axis_dev_m(iy_u + dy, p.ny);
        const std::uint32_t jz = wrap_axis_dev_m(iz_u + dz, p.nz);
        const std::size_t cj = linear_index_dev_m(jx, jy, jz, p.nx, p.ny);

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
          ddx = minimum_image_axis_dev_m(ddx, p.lx, p.periodic_x);
          ddy = minimum_image_axis_dev_m(ddy, p.ly, p.periodic_y);
          ddz = minimum_image_axis_dev_m(ddz, p.lz, p.periodic_z);
          const float ddx_f = static_cast<float>(ddx);
          const float ddy_f = static_cast<float>(ddy);
          const float ddz_f = static_cast<float>(ddz);
          const float r2_f = ddx_f * ddx_f + ddy_f * ddy_f + ddz_f * ddz_f;
          if (r2_f > p.cutoff_sq_f) {
            continue;
          }
          // FP32 sites: sqrt + reciprocal (the bandwidth-friendly bit). Promote
          // to FP64 before any multiplication that feeds per-atom accumulators;
          // the FP32 inv_r still loses ~6e-8 rel precision but spreading that
          // across the full pipeline caps the per-pair fij rel-diff at ~1e-7
          // (vs ~3e-7 when the whole energy chain runs FP32), enough to meet
          // D-M6-8 when summed over ~50 neighbours.
          const float r_f = sqrtf(r2_f);
          const float inv_r_f = 1.0f / r_f;
          const double r = static_cast<double>(r_f);
          const double inv_r = static_cast<double>(inv_r_f);

          const std::uint32_t type_j = types[j];
          const double dF_j = dFdrho[j];

          const double* rho_tab_j = rho_coeffs + static_cast<std::size_t>(type_j) * rho_stride;
          const double drho_j_dr = spline_deriv_dev_m(rho_tab_j, r, p.r_x0, p.r_rdx, p.nr);

          const double* rho_tab_i = rho_coeffs + static_cast<std::size_t>(type_i) * rho_stride;
          const double drho_i_dr = spline_deriv_dev_m(rho_tab_i, r, p.r_x0, p.r_rdx, p.nr);

          const std::size_t pair_k = pair_index_dev_m(type_i, type_j);
          const double* z2r_tab = z2r_coeffs + pair_k * z2r_stride;
          const double z_val = spline_eval_dev_m(z2r_tab, r, p.r_x0, p.r_rdx, p.nr);
          const double z_deriv = spline_deriv_dev_m(z2r_tab, r, p.r_x0, p.r_rdx, p.nr);
          const double phi = z_val * inv_r;
          const double phi_prime = (z_deriv - phi) * inv_r;

          const double dE_dr = dF_i * drho_j_dr + dF_j * drho_i_dr + phi_prime;
          const double fscalar = dE_dr * inv_r;

          const double fij_x = fscalar * ddx;
          const double fij_y = fscalar * ddy;
          const double fij_z = fscalar * ddz;

          fx_acc += fij_x;
          fy_acc += fij_y;
          fz_acc += fij_z;

          pe_acc += phi;

          v_xx += fij_x * ddx;
          v_yy += fij_y * ddy;
          v_zz += fij_z * ddz;
          v_xy += fij_x * ddy;
          v_xz += fij_x * ddz;
          v_yz += fij_y * ddz;
        }
      }
    }
  }

  fx_out[i] += fx_acc;
  fy_out[i] += fy_acc;
  fz_out[i] += fz_acc;
  pe_pair_out[i] = pe_acc;
  const std::size_t v_base = static_cast<std::size_t>(i) * 6u;
  virial_out[v_base + 0] = v_xx;
  virial_out[v_base + 1] = v_yy;
  virial_out[v_base + 2] = v_zz;
  virial_out[v_base + 3] = v_xy;
  virial_out[v_base + 4] = v_xz;
  virial_out[v_base + 5] = v_yz;
}

double kahan_sum_host_m(const double* data, std::size_t n) {
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

struct EamAlloyGpuMixed::Impl {
  std::uint64_t compute_version = 0;

  DevicePtr<std::byte> d_types_bytes;
  DevicePtr<std::byte> d_x_bytes;
  DevicePtr<std::byte> d_y_bytes;
  DevicePtr<std::byte> d_z_bytes;
  DevicePtr<std::byte> d_fx_bytes;
  DevicePtr<std::byte> d_fy_bytes;
  DevicePtr<std::byte> d_fz_bytes;
  DevicePtr<std::byte> d_cell_offsets_bytes;
  DevicePtr<std::byte> d_cell_atoms_bytes;
  DevicePtr<std::byte> d_rho_bytes;
  DevicePtr<std::byte> d_dFdrho_bytes;
  DevicePtr<std::byte> d_pe_embed_bytes;
  DevicePtr<std::byte> d_pe_pair_bytes;
  DevicePtr<std::byte> d_virial_bytes;
  DevicePtr<std::byte> d_F_coeffs_bytes;
  DevicePtr<std::byte> d_rho_coeffs_bytes;
  DevicePtr<std::byte> d_z2r_coeffs_bytes;

  // Spline cache identity (T6.9a) — matches EamAlloyGpu::Impl. Splines are
  // immutable for the lifetime of the owning potential; re-upload only when
  // the caller hands us a different host table triple.
  const double* splines_F_coeffs_host = nullptr;
  const double* splines_rho_coeffs_host = nullptr;
  const double* splines_z2r_coeffs_host = nullptr;
  std::uint64_t splines_upload_count = 0;
};

EamAlloyGpuMixed::EamAlloyGpuMixed() : impl_(std::make_unique<Impl>()) {}
EamAlloyGpuMixed::~EamAlloyGpuMixed() = default;
EamAlloyGpuMixed::EamAlloyGpuMixed(EamAlloyGpuMixed&&) noexcept = default;
EamAlloyGpuMixed& EamAlloyGpuMixed::operator=(EamAlloyGpuMixed&&) noexcept = default;

std::uint64_t EamAlloyGpuMixed::compute_version() const noexcept {
  return impl_ ? impl_->compute_version : 0;
}

std::uint64_t EamAlloyGpuMixed::splines_upload_count() const noexcept {
  return impl_ ? impl_->splines_upload_count : 0;
}

EamAlloyGpuResult EamAlloyGpuMixed::compute(std::size_t n,
                                            const std::uint32_t* host_types,
                                            const double* host_x,
                                            const double* host_y,
                                            const double* host_z,
                                            std::size_t ncells,
                                            const std::uint32_t* host_cell_offsets,
                                            const std::uint32_t* host_cell_atoms,
                                            const BoxParams& params,
                                            const EamAlloyTablesHost& tables,
                                            double* host_fx_out,
                                            double* host_fy_out,
                                            double* host_fz_out,
                                            DevicePool& pool,
                                            DeviceStream& stream) {
  TDMD_NVTX_RANGE("eam_mixed.compute");

  EamAlloyGpuResult result;
  ++impl_->compute_version;
  if (n == 0) {
    return result;
  }

  cudaStream_t s = raw_stream(stream);

  const std::size_t type_bytes = n * sizeof(std::uint32_t);
  const std::size_t pos_bytes = n * sizeof(double);
  impl_->d_types_bytes = pool.allocate_device(type_bytes, stream);
  impl_->d_x_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_y_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_z_bytes = pool.allocate_device(pos_bytes, stream);
  auto* d_types = reinterpret_cast<std::uint32_t*>(impl_->d_types_bytes.get());
  auto* d_x = reinterpret_cast<double*>(impl_->d_x_bytes.get());
  auto* d_y = reinterpret_cast<double*>(impl_->d_y_bytes.get());
  auto* d_z = reinterpret_cast<double*>(impl_->d_z_bytes.get());
  const std::size_t cell_offsets_bytes = (ncells + 1) * sizeof(std::uint32_t);
  const std::size_t cell_atoms_bytes = n * sizeof(std::uint32_t);
  impl_->d_cell_offsets_bytes = pool.allocate_device(cell_offsets_bytes, stream);
  impl_->d_cell_atoms_bytes = pool.allocate_device(cell_atoms_bytes, stream);
  auto* d_cell_offsets = reinterpret_cast<std::uint32_t*>(impl_->d_cell_offsets_bytes.get());
  auto* d_cell_atoms = reinterpret_cast<std::uint32_t*>(impl_->d_cell_atoms_bytes.get());
  {
    TDMD_NVTX_RANGE("eam_mixed.h2d.atoms_and_cells");
    check_cuda_mixed("memcpy types",
                     cudaMemcpyAsync(d_types, host_types, type_bytes, cudaMemcpyHostToDevice, s));
    check_cuda_mixed("memcpy x",
                     cudaMemcpyAsync(d_x, host_x, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda_mixed("memcpy y",
                     cudaMemcpyAsync(d_y, host_y, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda_mixed("memcpy z",
                     cudaMemcpyAsync(d_z, host_z, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda_mixed("memcpy cell_offsets",
                     cudaMemcpyAsync(d_cell_offsets,
                                     host_cell_offsets,
                                     cell_offsets_bytes,
                                     cudaMemcpyHostToDevice,
                                     s));
    check_cuda_mixed("memcpy cell_atoms",
                     cudaMemcpyAsync(d_cell_atoms,
                                     host_cell_atoms,
                                     cell_atoms_bytes,
                                     cudaMemcpyHostToDevice,
                                     s));
  }

  // Tables — cached across compute() calls per T6.9a. Splines are immutable
  // for the lifetime of the owning potential; we re-upload only when the
  // caller hands us a different host table triple.
  const std::size_t F_bytes = tables.n_species * tables.nrho * 7u * sizeof(double);
  const std::size_t rho_bytes_tab = tables.n_species * tables.nr * 7u * sizeof(double);
  const std::size_t z2r_bytes = tables.npairs * tables.nr * 7u * sizeof(double);
  const bool splines_changed = tables.F_coeffs != impl_->splines_F_coeffs_host ||
                               tables.rho_coeffs != impl_->splines_rho_coeffs_host ||
                               tables.z2r_coeffs != impl_->splines_z2r_coeffs_host;
  if (splines_changed) {
    TDMD_NVTX_RANGE("eam_mixed.h2d.splines");
    impl_->d_F_coeffs_bytes = pool.allocate_device(F_bytes, stream);
    impl_->d_rho_coeffs_bytes = pool.allocate_device(rho_bytes_tab, stream);
    impl_->d_z2r_coeffs_bytes = pool.allocate_device(z2r_bytes, stream);
    auto* d_F_upload = reinterpret_cast<double*>(impl_->d_F_coeffs_bytes.get());
    auto* d_rho_upload = reinterpret_cast<double*>(impl_->d_rho_coeffs_bytes.get());
    auto* d_z2r_upload = reinterpret_cast<double*>(impl_->d_z2r_coeffs_bytes.get());
    check_cuda_mixed(
        "memcpy F_coeffs",
        cudaMemcpyAsync(d_F_upload, tables.F_coeffs, F_bytes, cudaMemcpyHostToDevice, s));
    check_cuda_mixed(
        "memcpy rho_coeffs",
        cudaMemcpyAsync(d_rho_upload, tables.rho_coeffs, rho_bytes_tab, cudaMemcpyHostToDevice, s));
    check_cuda_mixed(
        "memcpy z2r_coeffs",
        cudaMemcpyAsync(d_z2r_upload, tables.z2r_coeffs, z2r_bytes, cudaMemcpyHostToDevice, s));
    impl_->splines_F_coeffs_host = tables.F_coeffs;
    impl_->splines_rho_coeffs_host = tables.rho_coeffs;
    impl_->splines_z2r_coeffs_host = tables.z2r_coeffs;
    ++impl_->splines_upload_count;
  }
  auto* d_F_coeffs = reinterpret_cast<double*>(impl_->d_F_coeffs_bytes.get());
  auto* d_rho_coeffs = reinterpret_cast<double*>(impl_->d_rho_coeffs_bytes.get());
  auto* d_z2r_coeffs = reinterpret_cast<double*>(impl_->d_z2r_coeffs_bytes.get());

  const std::size_t rho_buf_bytes = n * sizeof(double);
  impl_->d_rho_bytes = pool.allocate_device(rho_buf_bytes, stream);
  impl_->d_dFdrho_bytes = pool.allocate_device(rho_buf_bytes, stream);
  impl_->d_pe_embed_bytes = pool.allocate_device(rho_buf_bytes, stream);
  impl_->d_pe_pair_bytes = pool.allocate_device(rho_buf_bytes, stream);
  impl_->d_virial_bytes = pool.allocate_device(n * 6u * sizeof(double), stream);
  auto* d_rho = reinterpret_cast<double*>(impl_->d_rho_bytes.get());
  auto* d_dFdrho = reinterpret_cast<double*>(impl_->d_dFdrho_bytes.get());
  auto* d_pe_embed = reinterpret_cast<double*>(impl_->d_pe_embed_bytes.get());
  auto* d_pe_pair = reinterpret_cast<double*>(impl_->d_pe_pair_bytes.get());
  auto* d_virial = reinterpret_cast<double*>(impl_->d_virial_bytes.get());

  impl_->d_fx_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_fy_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_fz_bytes = pool.allocate_device(pos_bytes, stream);
  auto* d_fx = reinterpret_cast<double*>(impl_->d_fx_bytes.get());
  auto* d_fy = reinterpret_cast<double*>(impl_->d_fy_bytes.get());
  auto* d_fz = reinterpret_cast<double*>(impl_->d_fz_bytes.get());
  {
    TDMD_NVTX_RANGE("eam_mixed.h2d.forces_in");
    check_cuda_mixed("memcpy fx in",
                     cudaMemcpyAsync(d_fx, host_fx_out, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda_mixed("memcpy fy in",
                     cudaMemcpyAsync(d_fy, host_fy_out, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda_mixed("memcpy fz in",
                     cudaMemcpyAsync(d_fz, host_fz_out, pos_bytes, cudaMemcpyHostToDevice, s));
  }

  DeviceEamParamsMixed dp;
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
  dp.lx_f = static_cast<float>(params.lx);
  dp.ly_f = static_cast<float>(params.ly);
  dp.lz_f = static_cast<float>(params.lz);
  dp.cutoff_sq_f = static_cast<float>(tables.cutoff * tables.cutoff);
  dp.n_species = static_cast<std::uint32_t>(tables.n_species);
  dp.nrho = static_cast<std::uint32_t>(tables.nrho);
  dp.nr = static_cast<std::uint32_t>(tables.nr);
  dp.F_x0 = tables.F_x0;
  dp.F_rdx = 1.0 / tables.F_dx;
  dp.r_x0 = tables.r_x0;
  dp.r_rdx = 1.0 / tables.r_dx;

  const std::uint32_t n32 = static_cast<std::uint32_t>(n);
  constexpr int kThreadsPerBlock = 128;
  const std::uint32_t nblocks = (n32 + kThreadsPerBlock - 1) / kThreadsPerBlock;

  {
    TDMD_NVTX_RANGE("eam_mixed.density_kernel");
    density_kernel_mixed<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                              d_types,
                                                              d_x,
                                                              d_y,
                                                              d_z,
                                                              d_cell_offsets,
                                                              d_cell_atoms,
                                                              d_rho_coeffs,
                                                              dp,
                                                              d_rho);
    check_cuda_mixed("launch density_kernel_mixed", cudaGetLastError());
  }

  {
    TDMD_NVTX_RANGE("eam_mixed.embedding_kernel");
    embedding_kernel_mixed<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                                d_types,
                                                                d_rho,
                                                                d_F_coeffs,
                                                                dp,
                                                                d_dFdrho,
                                                                d_pe_embed);
    check_cuda_mixed("launch embedding_kernel_mixed", cudaGetLastError());
  }

  {
    TDMD_NVTX_RANGE("eam_mixed.force_kernel");
    force_kernel_mixed<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                            d_types,
                                                            d_x,
                                                            d_y,
                                                            d_z,
                                                            d_cell_offsets,
                                                            d_cell_atoms,
                                                            d_dFdrho,
                                                            d_rho_coeffs,
                                                            d_z2r_coeffs,
                                                            dp,
                                                            d_fx,
                                                            d_fy,
                                                            d_fz,
                                                            d_pe_pair,
                                                            d_virial);
    check_cuda_mixed("launch force_kernel_mixed", cudaGetLastError());
  }

  std::vector<double> host_pe_embed(n);
  std::vector<double> host_pe_pair(n);
  std::vector<double> host_virial(n * 6u);

  {
    TDMD_NVTX_RANGE("eam_mixed.d2h.forces_and_reductions");
    check_cuda_mixed("D2H fx",
                     cudaMemcpyAsync(host_fx_out, d_fx, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda_mixed("D2H fy",
                     cudaMemcpyAsync(host_fy_out, d_fy, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda_mixed("D2H fz",
                     cudaMemcpyAsync(host_fz_out, d_fz, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda_mixed("D2H pe_embed",
                     cudaMemcpyAsync(host_pe_embed.data(),
                                     d_pe_embed,
                                     n * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     s));
    check_cuda_mixed("D2H pe_pair",
                     cudaMemcpyAsync(host_pe_pair.data(),
                                     d_pe_pair,
                                     n * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     s));
    check_cuda_mixed("D2H virial",
                     cudaMemcpyAsync(host_virial.data(),
                                     d_virial,
                                     n * 6u * sizeof(double),
                                     cudaMemcpyDeviceToHost,
                                     s));

    check_cuda_mixed("stream sync EAM mixed", cudaStreamSynchronize(s));
  }

  const double pe_embed_total = kahan_sum_host_m(host_pe_embed.data(), n);
  const double pe_pair_full = kahan_sum_host_m(host_pe_pair.data(), n);
  result.potential_energy = pe_embed_total + 0.5 * pe_pair_full;

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
    result.virial[k] = 0.5 * vsum[k];
  }

  return result;
}

#else  // CPU-only build

struct EamAlloyGpuMixed::Impl {};

EamAlloyGpuMixed::EamAlloyGpuMixed() : impl_(std::make_unique<Impl>()) {}
EamAlloyGpuMixed::~EamAlloyGpuMixed() = default;
EamAlloyGpuMixed::EamAlloyGpuMixed(EamAlloyGpuMixed&&) noexcept = default;
EamAlloyGpuMixed& EamAlloyGpuMixed::operator=(EamAlloyGpuMixed&&) noexcept = default;

std::uint64_t EamAlloyGpuMixed::compute_version() const noexcept {
  return 0;
}

std::uint64_t EamAlloyGpuMixed::splines_upload_count() const noexcept {
  return 0;
}

EamAlloyGpuResult EamAlloyGpuMixed::compute(std::size_t /*n*/,
                                            const std::uint32_t* /*host_types*/,
                                            const double* /*host_x*/,
                                            const double* /*host_y*/,
                                            const double* /*host_z*/,
                                            std::size_t /*ncells*/,
                                            const std::uint32_t* /*host_cell_offsets*/,
                                            const std::uint32_t* /*host_cell_atoms*/,
                                            const BoxParams& /*params*/,
                                            const EamAlloyTablesHost& /*tables*/,
                                            double* /*host_fx_out*/,
                                            double* /*host_fy_out*/,
                                            double* /*host_fz_out*/,
                                            DevicePool& /*pool*/,
                                            DeviceStream& /*stream*/) {
  throw std::runtime_error(
      "gpu::EamAlloyGpuMixed::compute: CPU-only build (TDMD_BUILD_CUDA=0); CUDA not linked");
}

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
