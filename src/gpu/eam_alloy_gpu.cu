// SPEC: docs/specs/gpu/SPEC.md §7.2 (EAM three-pass), §6.3 (D-M6-7),
//       §8.1 (Reference FP64-only), §9 (NVTX — reserved for T6.11)
// Module SPEC: docs/specs/potentials/SPEC.md §4.1–§4.4
// Exec pack: docs/development/m6_execution_pack.md T6.5
//
// Three CUDA kernels implementing EAM/alloy per-atom force:
//   density_kernel   — ρ[i] = Σⱼ rho_β(r_ij)      full-list per-atom sweep
//   embedding_kernel — F(ρ), F'(ρ)                thread per atom, no reduction
//   force_kernel     — f[i] = Σⱼ (dE/dr · Δ / r)  full-list per-atom sweep
//
// Full-list iteration (no j>i filter) eliminates scatter conflicts: every
// thread writes only to its own atom's slot for f/ρ/pe/virial. Host-side
// Kahan reduction of per-atom PE + virial buffers gives the final totals.
//
// The 27-cell stencil walker + cell-index math is a direct mirror of T6.4's
// `scan_neighbors` (see src/gpu/neighbor_list_gpu.cu) but with the
// `j <= i` filter removed. We re-implement it locally rather than share the
// definition across TUs to keep each kernel trivially introspectable.
//
// Device spline eval reproduces TabulatedFunction's Horner form bit-exactly:
//   locate(x) → (cell_idx, p); eval = ((c3·p + c4)·p + c5)·p + c6
//   deriv    =  (c0·p + c1)·p + c2
// Coefficient layout (c0..c6 per cell) matches the CPU struct verbatim.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/eam_alloy_gpu.hpp"
#include "tdmd/gpu/factories.hpp"  // make_event (T7.8 async path)
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

[[noreturn]] void throw_cuda(const char* op, cudaError_t err) {
  std::ostringstream oss;
  oss << "gpu::EamAlloyGpu::" << op << ": " << cudaGetErrorName(err) << " ("
      << cudaGetErrorString(err) << ")";
  throw std::runtime_error(oss.str());
}

void check_cuda(const char* op, cudaError_t err) {
  if (err != cudaSuccess) {
    throw_cuda(op, err);
  }
}

// POD captured by the kernels — mirror of BoxParams plus per-table scalars.
// Kept tiny so it passes as a launch arg.
struct DeviceEamParams {
  // Box + cell grid
  double xlo, ylo, zlo;
  double lx, ly, lz;
  double cell_x, cell_y, cell_z;
  std::uint32_t nx, ny, nz;
  int periodic_x, periodic_y, periodic_z;
  // EAM cutoff (squared)
  double cutoff_sq;
  // Spline grid scalars
  std::uint32_t n_species;
  std::uint32_t nrho;
  std::uint32_t nr;
  double F_x0, F_rdx;
  double r_x0, r_rdx;
};

// ------------ Shared device helpers (mirrored from T6.4) -------------------

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

// ------------ Spline eval — mirror of TabulatedFunction ---------------------

__device__ __forceinline__ void locate_dev(double x,
                                           double x0,
                                           double rdx,
                                           std::uint32_t n,
                                           std::size_t& i_out,
                                           double& p_out) {
  // LAMMPS/CPU form: p_raw = (x - x0)·rdx + 1, m = (long long)p_raw.
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

__device__ __forceinline__ double spline_eval_dev(const double* __restrict__ table_coeffs,
                                                  double x,
                                                  double x0,
                                                  double rdx,
                                                  std::uint32_t n) {
  std::size_t i;
  double p;
  locate_dev(x, x0, rdx, n, i, p);
  const double* c = table_coeffs + i * 7;
  return ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
}

__device__ __forceinline__ double spline_deriv_dev(const double* __restrict__ table_coeffs,
                                                   double x,
                                                   double x0,
                                                   double rdx,
                                                   std::uint32_t n) {
  std::size_t i;
  double p;
  locate_dev(x, x0, rdx, n, i, p);
  const double* c = table_coeffs + i * 7;
  return (c[0] * p + c[1]) * p + c[2];
}

// Symmetric lower-triangular pair index (matches EamAlloyData::pair_index).
__device__ __forceinline__ std::size_t pair_index_dev(std::uint32_t a, std::uint32_t b) {
  std::uint32_t hi = a > b ? a : b;
  std::uint32_t lo = a > b ? b : a;
  return static_cast<std::size_t>(hi) * (static_cast<std::size_t>(hi) + 1) / 2 +
         static_cast<std::size_t>(lo);
}

// ------------ Kernel 1: density pass ---------------------------------------
__global__ void density_kernel(std::uint32_t n,
                               const std::uint32_t* __restrict__ types,
                               const double* __restrict__ x,
                               const double* __restrict__ y,
                               const double* __restrict__ z,
                               const std::uint32_t* __restrict__ cell_offsets,
                               const std::uint32_t* __restrict__ cell_atoms,
                               const double* __restrict__ rho_coeffs,
                               DeviceEamParams p,
                               double* __restrict__ rho_out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];

  const auto ix_u = static_cast<int>(cell_index_axis_dev(xi, p.xlo, p.cell_x, p.nx));
  const auto iy_u = static_cast<int>(cell_index_axis_dev(yi, p.ylo, p.cell_y, p.ny));
  const auto iz_u = static_cast<int>(cell_index_axis_dev(zi, p.zlo, p.cell_z, p.nz));

  const std::size_t rho_stride = static_cast<std::size_t>(p.nr) * 7u;

  double rho_i = 0.0;
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
          const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
          if (r2 > p.cutoff_sq) {
            continue;
          }
          const double r = sqrt(r2);
          const std::uint32_t type_j = types[j];
          const double* rho_tab = rho_coeffs + static_cast<std::size_t>(type_j) * rho_stride;
          rho_i += spline_eval_dev(rho_tab, r, p.r_x0, p.r_rdx, p.nr);
        }
      }
    }
  }
  rho_out[i] = rho_i;
}

// ------------ Kernel 2: embedding pass -------------------------------------
__global__ void embedding_kernel(std::uint32_t n,
                                 const std::uint32_t* __restrict__ types,
                                 const double* __restrict__ rho,
                                 const double* __restrict__ F_coeffs,
                                 DeviceEamParams p,
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
  pe_embed_out[i] = spline_eval_dev(F_tab, rho_i, p.F_x0, p.F_rdx, p.nrho);
  dFdrho_out[i] = spline_deriv_dev(F_tab, rho_i, p.F_x0, p.F_rdx, p.nrho);
}

// ------------ Kernel 3: force pass -----------------------------------------
// Writes per-atom fx/fy/fz, per-atom pair-PE (to be halved on host), per-atom
// 6-component virial contribution (to be halved on host). Full-list per-atom
// — each thread writes only to its own atom slot.
__global__ void force_kernel(std::uint32_t n,
                             const std::uint32_t* __restrict__ types,
                             const double* __restrict__ x,
                             const double* __restrict__ y,
                             const double* __restrict__ z,
                             const std::uint32_t* __restrict__ cell_offsets,
                             const std::uint32_t* __restrict__ cell_atoms,
                             const double* __restrict__ dFdrho,
                             const double* __restrict__ rho_coeffs,
                             const double* __restrict__ z2r_coeffs,
                             DeviceEamParams p,
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

  const auto ix_u = static_cast<int>(cell_index_axis_dev(xi, p.xlo, p.cell_x, p.nx));
  const auto iy_u = static_cast<int>(cell_index_axis_dev(yi, p.ylo, p.cell_y, p.ny));
  const auto iz_u = static_cast<int>(cell_index_axis_dev(zi, p.zlo, p.cell_z, p.nz));

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
          const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
          if (r2 > p.cutoff_sq) {
            continue;
          }
          const double r = sqrt(r2);
          const double inv_r = 1.0 / r;

          const std::uint32_t type_j = types[j];
          const double dF_j = dFdrho[j];

          // ρ'_β(r) — density derivative of species j affecting i.
          const double* rho_tab_j = rho_coeffs + static_cast<std::size_t>(type_j) * rho_stride;
          const double drho_j_dr = spline_deriv_dev(rho_tab_j, r, p.r_x0, p.r_rdx, p.nr);

          // ρ'_α(r) — density derivative of species i affecting j.
          const double* rho_tab_i = rho_coeffs + static_cast<std::size_t>(type_i) * rho_stride;
          const double drho_i_dr = spline_deriv_dev(rho_tab_i, r, p.r_x0, p.r_rdx, p.nr);

          // Pair part via z2r: φ = z/r, φ' = (z' - φ)/r.
          const std::size_t pair_k = pair_index_dev(type_i, type_j);
          const double* z2r_tab = z2r_coeffs + pair_k * z2r_stride;
          const double z_val = spline_eval_dev(z2r_tab, r, p.r_x0, p.r_rdx, p.nr);
          const double z_deriv = spline_deriv_dev(z2r_tab, r, p.r_x0, p.r_rdx, p.nr);
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

          // Pair PE — counted twice in full-list, halved on host.
          pe_acc += phi;

          // Clausius virial — per-pair-from-i contribution; halved on host.
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

// Kahan compensated sum on host. One pass over an arbitrary-length buffer —
// deterministic in the input order.
double kahan_sum_host(const double* data, std::size_t n) {
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

struct EamAlloyGpu::Impl {
  std::uint64_t compute_version = 0;

  // Device buffers held across compute() calls to avoid churn. Re-allocated
  // when the previous block isn't large enough. DevicePool handles recycling
  // via its free-list.
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

  // Spline cache identity — splines are immutable for the lifetime of the
  // owning potential instance. We re-upload only if the caller hands us a
  // different host table (e.g. a different potential bound to the same
  // EamAlloyGpu — not supported today but keeps the invariant honest).
  // Count exposed for T6.9a perf test: after N back-to-back compute() calls
  // with the same tables this must be 1.
  const double* splines_F_coeffs_host = nullptr;
  const double* splines_rho_coeffs_host = nullptr;
  const double* splines_z2r_coeffs_host = nullptr;
  std::uint64_t splines_upload_count = 0;
};

EamAlloyGpu::EamAlloyGpu() : impl_(std::make_unique<Impl>()) {}
EamAlloyGpu::~EamAlloyGpu() = default;
EamAlloyGpu::EamAlloyGpu(EamAlloyGpu&&) noexcept = default;
EamAlloyGpu& EamAlloyGpu::operator=(EamAlloyGpu&&) noexcept = default;

std::uint64_t EamAlloyGpu::compute_version() const noexcept {
  return impl_ ? impl_->compute_version : 0;
}

std::uint64_t EamAlloyGpu::splines_upload_count() const noexcept {
  return impl_ ? impl_->splines_upload_count : 0;
}

EamAlloyGpuResult EamAlloyGpu::compute(std::size_t n,
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
  TDMD_NVTX_RANGE("eam.compute");

  EamAlloyGpuResult result;
  ++impl_->compute_version;
  if (n == 0) {
    return result;
  }

  cudaStream_t s = raw_stream(stream);

  // ---------- 1. H2D copies ----------
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
    TDMD_NVTX_RANGE("eam.h2d.atoms_and_cells");
    check_cuda("memcpy types",
               cudaMemcpyAsync(d_types, host_types, type_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("memcpy x", cudaMemcpyAsync(d_x, host_x, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("memcpy y", cudaMemcpyAsync(d_y, host_y, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("memcpy z", cudaMemcpyAsync(d_z, host_z, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("memcpy cell_offsets",
               cudaMemcpyAsync(d_cell_offsets,
                               host_cell_offsets,
                               cell_offsets_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("memcpy cell_atoms",
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
    TDMD_NVTX_RANGE("eam.h2d.splines");
    impl_->d_F_coeffs_bytes = pool.allocate_device(F_bytes, stream);
    impl_->d_rho_coeffs_bytes = pool.allocate_device(rho_bytes_tab, stream);
    impl_->d_z2r_coeffs_bytes = pool.allocate_device(z2r_bytes, stream);
    auto* d_F_upload = reinterpret_cast<double*>(impl_->d_F_coeffs_bytes.get());
    auto* d_rho_upload = reinterpret_cast<double*>(impl_->d_rho_coeffs_bytes.get());
    auto* d_z2r_upload = reinterpret_cast<double*>(impl_->d_z2r_coeffs_bytes.get());
    check_cuda("memcpy F_coeffs",
               cudaMemcpyAsync(d_F_upload, tables.F_coeffs, F_bytes, cudaMemcpyHostToDevice, s));
    check_cuda(
        "memcpy rho_coeffs",
        cudaMemcpyAsync(d_rho_upload, tables.rho_coeffs, rho_bytes_tab, cudaMemcpyHostToDevice, s));
    check_cuda(
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

  // ---------- 2. Output + scratch buffers ----------
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

  // Force buffers — we read current values (from caller's additive contract),
  // accumulate into them inside the force kernel, then D2H.
  impl_->d_fx_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_fy_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_fz_bytes = pool.allocate_device(pos_bytes, stream);
  auto* d_fx = reinterpret_cast<double*>(impl_->d_fx_bytes.get());
  auto* d_fy = reinterpret_cast<double*>(impl_->d_fy_bytes.get());
  auto* d_fz = reinterpret_cast<double*>(impl_->d_fz_bytes.get());
  {
    TDMD_NVTX_RANGE("eam.h2d.forces_in");
    check_cuda("memcpy fx in",
               cudaMemcpyAsync(d_fx, host_fx_out, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("memcpy fy in",
               cudaMemcpyAsync(d_fy, host_fy_out, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("memcpy fz in",
               cudaMemcpyAsync(d_fz, host_fz_out, pos_bytes, cudaMemcpyHostToDevice, s));
  }

  // ---------- 3. Kernel params ----------
  DeviceEamParams dp;
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
  dp.cutoff_sq = tables.cutoff * tables.cutoff;
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

  // ---------- 4. Kernel 1: density ----------
  {
    TDMD_NVTX_RANGE("eam.density_kernel");
    density_kernel<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                        d_types,
                                                        d_x,
                                                        d_y,
                                                        d_z,
                                                        d_cell_offsets,
                                                        d_cell_atoms,
                                                        d_rho_coeffs,
                                                        dp,
                                                        d_rho);
    check_cuda("launch density_kernel", cudaGetLastError());
  }

  // ---------- 5. Kernel 2: embedding ----------
  {
    TDMD_NVTX_RANGE("eam.embedding_kernel");
    embedding_kernel<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                          d_types,
                                                          d_rho,
                                                          d_F_coeffs,
                                                          dp,
                                                          d_dFdrho,
                                                          d_pe_embed);
    check_cuda("launch embedding_kernel", cudaGetLastError());
  }

  // ---------- 6. Kernel 3: force ----------
  {
    TDMD_NVTX_RANGE("eam.force_kernel");
    force_kernel<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
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
    check_cuda("launch force_kernel", cudaGetLastError());
  }

  // ---------- 7. D2H forces + reduction buffers ----------
  std::vector<double> host_pe_embed(n);
  std::vector<double> host_pe_pair(n);
  std::vector<double> host_virial(n * 6u);

  {
    TDMD_NVTX_RANGE("eam.d2h.forces_and_reductions");
    check_cuda("D2H fx", cudaMemcpyAsync(host_fx_out, d_fx, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("D2H fy", cudaMemcpyAsync(host_fy_out, d_fy, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("D2H fz", cudaMemcpyAsync(host_fz_out, d_fz, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("D2H pe_embed",
               cudaMemcpyAsync(host_pe_embed.data(),
                               d_pe_embed,
                               n * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("D2H pe_pair",
               cudaMemcpyAsync(host_pe_pair.data(),
                               d_pe_pair,
                               n * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("D2H virial",
               cudaMemcpyAsync(host_virial.data(),
                               d_virial,
                               n * 6u * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s));

    check_cuda("stream sync EAM", cudaStreamSynchronize(s));
  }

  // ---------- 8. Host Kahan reductions ----------
  const double pe_embed_total = kahan_sum_host(host_pe_embed.data(), n);
  const double pe_pair_full = kahan_sum_host(host_pe_pair.data(), n);
  result.potential_energy = pe_embed_total + 0.5 * pe_pair_full;

  // Virial: 6 parallel Kahan sums. Arrays are interleaved (per-atom 6-block),
  // so gather into 6 contiguous reductions by stride.
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

// =====================================================================
// T7.8: async dispatch — compute_async + finalize_async.
//
// The async path splits the sync compute() into "enqueue all GPU work" and
// "wait for D2H + reduce on host". Between enqueue and wait, the caller can
// launch *more* compute_async calls on independent handles, pipelining
// H2D/kernels/D2H across iterations per gpu/SPEC §3.2.
//
// Per-call buffers live on the returned handle (not in EamAlloyGpu::Impl),
// so K handles can be in-flight simultaneously. Spline tables stay cached in
// Impl and are shared across async dispatches — first compute_async after a
// pointer change uploads on mem_stream; subsequent calls skip the upload.
// =====================================================================

struct EamAlloyAsyncHandle::Impl {
  DevicePtr<std::byte> d_types;
  DevicePtr<std::byte> d_x;
  DevicePtr<std::byte> d_y;
  DevicePtr<std::byte> d_z;
  DevicePtr<std::byte> d_cell_offsets;
  DevicePtr<std::byte> d_cell_atoms;
  DevicePtr<std::byte> d_fx;
  DevicePtr<std::byte> d_fy;
  DevicePtr<std::byte> d_fz;
  DevicePtr<std::byte> d_rho;
  DevicePtr<std::byte> d_dFdrho;
  DevicePtr<std::byte> d_pe_embed;
  DevicePtr<std::byte> d_pe_pair;
  DevicePtr<std::byte> d_virial;

  DeviceEvent h2d_done_event;
  DeviceEvent kernels_done_event;
  DeviceEvent d2h_done_event;

  // T7.8 overlap: pinned host scratch for D2H destinations. Pageable host
  // memory degrades cudaMemcpyAsync to sync (driver uses internal staging),
  // blocking the host thread and defeating compute/mem overlap.
  DevicePtr<std::byte> h_pe_embed;
  DevicePtr<std::byte> h_pe_pair;
  DevicePtr<std::byte> h_virial;

  double* host_fx_out = nullptr;
  double* host_fy_out = nullptr;
  double* host_fz_out = nullptr;

  // Mem-stream pointer stashed by compute_async so finalize_async knows where
  // to queue D2H. Non-owning; caller guarantees stream liveness through drain.
  DeviceStream* mem_stream = nullptr;

  std::size_t n_atoms = 0;
  bool finalized = false;
};

EamAlloyAsyncHandle::EamAlloyAsyncHandle() noexcept = default;
EamAlloyAsyncHandle::EamAlloyAsyncHandle(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
EamAlloyAsyncHandle::~EamAlloyAsyncHandle() = default;
EamAlloyAsyncHandle::EamAlloyAsyncHandle(EamAlloyAsyncHandle&&) noexcept = default;
EamAlloyAsyncHandle& EamAlloyAsyncHandle::operator=(EamAlloyAsyncHandle&&) noexcept = default;

bool EamAlloyAsyncHandle::valid() const noexcept {
  return static_cast<bool>(impl_);
}

EamAlloyAsyncHandle EamAlloyGpu::compute_async(std::size_t n,
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
                                               DeviceStream& compute_stream,
                                               DeviceStream& mem_stream) {
  TDMD_NVTX_RANGE("eam.compute_async");
  ++impl_->compute_version;

  auto handle_impl = std::make_unique<EamAlloyAsyncHandle::Impl>();
  handle_impl->host_fx_out = host_fx_out;
  handle_impl->host_fy_out = host_fy_out;
  handle_impl->host_fz_out = host_fz_out;
  handle_impl->n_atoms = n;

  if (n == 0) {
    return EamAlloyAsyncHandle(std::move(handle_impl));
  }

  cudaStream_t s_compute = raw_stream(compute_stream);
  cudaStream_t s_mem = raw_stream(mem_stream);
  if (s_compute == nullptr || s_mem == nullptr) {
    throw std::runtime_error(
        "EamAlloyGpu::compute_async: compute_stream and mem_stream must both be valid");
  }

  int device_id = 0;
  check_cuda("cudaGetDevice", cudaGetDevice(&device_id));
  handle_impl->h2d_done_event = make_event(static_cast<DeviceId>(device_id));
  handle_impl->kernels_done_event = make_event(static_cast<DeviceId>(device_id));
  handle_impl->d2h_done_event = make_event(static_cast<DeviceId>(device_id));

  const std::size_t type_bytes = n * sizeof(std::uint32_t);
  const std::size_t pos_bytes = n * sizeof(double);
  const std::size_t cell_offsets_bytes = (ncells + 1) * sizeof(std::uint32_t);
  const std::size_t cell_atoms_bytes = n * sizeof(std::uint32_t);

  // Per-call input buffers — tied to mem_stream (H2D producer) for stream-
  // ordered reuse by the pool.
  handle_impl->d_types = pool.allocate_device(type_bytes, mem_stream);
  handle_impl->d_x = pool.allocate_device(pos_bytes, mem_stream);
  handle_impl->d_y = pool.allocate_device(pos_bytes, mem_stream);
  handle_impl->d_z = pool.allocate_device(pos_bytes, mem_stream);
  handle_impl->d_cell_offsets = pool.allocate_device(cell_offsets_bytes, mem_stream);
  handle_impl->d_cell_atoms = pool.allocate_device(cell_atoms_bytes, mem_stream);
  handle_impl->d_fx = pool.allocate_device(pos_bytes, mem_stream);
  handle_impl->d_fy = pool.allocate_device(pos_bytes, mem_stream);
  handle_impl->d_fz = pool.allocate_device(pos_bytes, mem_stream);

  auto* d_types = reinterpret_cast<std::uint32_t*>(handle_impl->d_types.get());
  auto* d_x = reinterpret_cast<double*>(handle_impl->d_x.get());
  auto* d_y = reinterpret_cast<double*>(handle_impl->d_y.get());
  auto* d_z = reinterpret_cast<double*>(handle_impl->d_z.get());
  auto* d_cell_offsets = reinterpret_cast<std::uint32_t*>(handle_impl->d_cell_offsets.get());
  auto* d_cell_atoms = reinterpret_cast<std::uint32_t*>(handle_impl->d_cell_atoms.get());
  auto* d_fx = reinterpret_cast<double*>(handle_impl->d_fx.get());
  auto* d_fy = reinterpret_cast<double*>(handle_impl->d_fy.get());
  auto* d_fz = reinterpret_cast<double*>(handle_impl->d_fz.get());

  {
    TDMD_NVTX_RANGE("eam.async.h2d");
    check_cuda("memcpy types",
               cudaMemcpyAsync(d_types, host_types, type_bytes, cudaMemcpyHostToDevice, s_mem));
    check_cuda("memcpy x", cudaMemcpyAsync(d_x, host_x, pos_bytes, cudaMemcpyHostToDevice, s_mem));
    check_cuda("memcpy y", cudaMemcpyAsync(d_y, host_y, pos_bytes, cudaMemcpyHostToDevice, s_mem));
    check_cuda("memcpy z", cudaMemcpyAsync(d_z, host_z, pos_bytes, cudaMemcpyHostToDevice, s_mem));
    check_cuda("memcpy cell_offsets",
               cudaMemcpyAsync(d_cell_offsets,
                               host_cell_offsets,
                               cell_offsets_bytes,
                               cudaMemcpyHostToDevice,
                               s_mem));
    check_cuda("memcpy cell_atoms",
               cudaMemcpyAsync(d_cell_atoms,
                               host_cell_atoms,
                               cell_atoms_bytes,
                               cudaMemcpyHostToDevice,
                               s_mem));
    check_cuda("memcpy fx in",
               cudaMemcpyAsync(d_fx, host_fx_out, pos_bytes, cudaMemcpyHostToDevice, s_mem));
    check_cuda("memcpy fy in",
               cudaMemcpyAsync(d_fy, host_fy_out, pos_bytes, cudaMemcpyHostToDevice, s_mem));
    check_cuda("memcpy fz in",
               cudaMemcpyAsync(d_fz, host_fz_out, pos_bytes, cudaMemcpyHostToDevice, s_mem));
  }

  // Spline tables: cached in EamAlloyGpu::Impl and shared across handles.
  // Upload goes on mem_stream too so h2d_done_event covers it.
  const std::size_t F_bytes = tables.n_species * tables.nrho * 7u * sizeof(double);
  const std::size_t rho_bytes_tab = tables.n_species * tables.nr * 7u * sizeof(double);
  const std::size_t z2r_bytes = tables.npairs * tables.nr * 7u * sizeof(double);
  const bool splines_changed = tables.F_coeffs != impl_->splines_F_coeffs_host ||
                               tables.rho_coeffs != impl_->splines_rho_coeffs_host ||
                               tables.z2r_coeffs != impl_->splines_z2r_coeffs_host;
  if (splines_changed) {
    TDMD_NVTX_RANGE("eam.async.h2d.splines");
    impl_->d_F_coeffs_bytes = pool.allocate_device(F_bytes, mem_stream);
    impl_->d_rho_coeffs_bytes = pool.allocate_device(rho_bytes_tab, mem_stream);
    impl_->d_z2r_coeffs_bytes = pool.allocate_device(z2r_bytes, mem_stream);
    auto* d_F_upload = reinterpret_cast<double*>(impl_->d_F_coeffs_bytes.get());
    auto* d_rho_upload = reinterpret_cast<double*>(impl_->d_rho_coeffs_bytes.get());
    auto* d_z2r_upload = reinterpret_cast<double*>(impl_->d_z2r_coeffs_bytes.get());
    check_cuda(
        "memcpy F_coeffs",
        cudaMemcpyAsync(d_F_upload, tables.F_coeffs, F_bytes, cudaMemcpyHostToDevice, s_mem));
    check_cuda("memcpy rho_coeffs",
               cudaMemcpyAsync(d_rho_upload,
                               tables.rho_coeffs,
                               rho_bytes_tab,
                               cudaMemcpyHostToDevice,
                               s_mem));
    check_cuda(
        "memcpy z2r_coeffs",
        cudaMemcpyAsync(d_z2r_upload, tables.z2r_coeffs, z2r_bytes, cudaMemcpyHostToDevice, s_mem));
    impl_->splines_F_coeffs_host = tables.F_coeffs;
    impl_->splines_rho_coeffs_host = tables.rho_coeffs;
    impl_->splines_z2r_coeffs_host = tables.z2r_coeffs;
    ++impl_->splines_upload_count;
  }
  auto* d_F_coeffs = reinterpret_cast<double*>(impl_->d_F_coeffs_bytes.get());
  auto* d_rho_coeffs = reinterpret_cast<double*>(impl_->d_rho_coeffs_bytes.get());
  auto* d_z2r_coeffs = reinterpret_cast<double*>(impl_->d_z2r_coeffs_bytes.get());

  // Signal H2D complete — compute_stream will wait on this before reading.
  check_cuda("record h2d_done_event",
             cudaEventRecord(raw_event(handle_impl->h2d_done_event), s_mem));

  // Scratch/output device buffers — allocated on compute_stream since that is
  // the next producer.
  const std::size_t rho_buf_bytes = n * sizeof(double);
  handle_impl->d_rho = pool.allocate_device(rho_buf_bytes, compute_stream);
  handle_impl->d_dFdrho = pool.allocate_device(rho_buf_bytes, compute_stream);
  handle_impl->d_pe_embed = pool.allocate_device(rho_buf_bytes, compute_stream);
  handle_impl->d_pe_pair = pool.allocate_device(rho_buf_bytes, compute_stream);
  handle_impl->d_virial = pool.allocate_device(n * 6u * sizeof(double), compute_stream);
  auto* d_rho = reinterpret_cast<double*>(handle_impl->d_rho.get());
  auto* d_dFdrho = reinterpret_cast<double*>(handle_impl->d_dFdrho.get());
  auto* d_pe_embed = reinterpret_cast<double*>(handle_impl->d_pe_embed.get());
  auto* d_pe_pair = reinterpret_cast<double*>(handle_impl->d_pe_pair.get());
  auto* d_virial = reinterpret_cast<double*>(handle_impl->d_virial.get());

  // compute_stream waits on mem_stream's H2D event before consuming inputs.
  check_cuda("wait h2d on compute_stream",
             cudaStreamWaitEvent(s_compute, raw_event(handle_impl->h2d_done_event), 0));

  DeviceEamParams dp;
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
  dp.cutoff_sq = tables.cutoff * tables.cutoff;
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
    TDMD_NVTX_RANGE("eam.async.density");
    density_kernel<<<nblocks, kThreadsPerBlock, 0, s_compute>>>(n32,
                                                                d_types,
                                                                d_x,
                                                                d_y,
                                                                d_z,
                                                                d_cell_offsets,
                                                                d_cell_atoms,
                                                                d_rho_coeffs,
                                                                dp,
                                                                d_rho);
    check_cuda("launch density_kernel", cudaGetLastError());
  }
  {
    TDMD_NVTX_RANGE("eam.async.embedding");
    embedding_kernel<<<nblocks, kThreadsPerBlock, 0, s_compute>>>(n32,
                                                                  d_types,
                                                                  d_rho,
                                                                  d_F_coeffs,
                                                                  dp,
                                                                  d_dFdrho,
                                                                  d_pe_embed);
    check_cuda("launch embedding_kernel", cudaGetLastError());
  }
  {
    TDMD_NVTX_RANGE("eam.async.force");
    force_kernel<<<nblocks, kThreadsPerBlock, 0, s_compute>>>(n32,
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
    check_cuda("launch force_kernel", cudaGetLastError());
  }

  // Record kernels-done. D2H deferred to finalize_async so mem_stream isn't
  // blocked behind cross-stream waits during the enqueue phase — this lets K
  // back-to-back H2Ds queue ahead of any D2H (T7.8 overlap contract).
  check_cuda("record kernels_done_event",
             cudaEventRecord(raw_event(handle_impl->kernels_done_event), s_compute));

  // Allocate pinned host scratch now so finalize_async has destinations;
  // device buffers (d_fx, d_fy, d_fz, d_pe_*, d_virial) already held by
  // handle_impl keep their data live until finalize D2Hs it off.
  handle_impl->h_pe_embed = pool.allocate_pinned_host(n * sizeof(double));
  handle_impl->h_pe_pair = pool.allocate_pinned_host(n * sizeof(double));
  handle_impl->h_virial = pool.allocate_pinned_host(n * 6u * sizeof(double));

  // Stash mem_stream for finalize_async to use. The handle doesn't own it;
  // caller guarantees liveness until the adapter drains the slot.
  handle_impl->mem_stream = &mem_stream;

  return EamAlloyAsyncHandle(std::move(handle_impl));
}

EamAlloyGpuResult EamAlloyGpu::finalize_async(EamAlloyAsyncHandle handle) {
  TDMD_NVTX_RANGE("eam.finalize_async");
  if (!handle.valid()) {
    throw std::invalid_argument("EamAlloyGpu::finalize_async: handle is not valid");
  }
  auto* hi = handle.impl();
  if (hi->finalized) {
    throw std::invalid_argument("EamAlloyGpu::finalize_async: handle already finalized");
  }
  hi->finalized = true;

  EamAlloyGpuResult result;
  if (hi->n_atoms == 0) {
    return result;
  }

  const std::size_t n = hi->n_atoms;
  const std::size_t pos_bytes = n * sizeof(double);

  cudaStream_t s_mem = raw_stream(*hi->mem_stream);
  auto* d_fx = reinterpret_cast<double*>(hi->d_fx.get());
  auto* d_fy = reinterpret_cast<double*>(hi->d_fy.get());
  auto* d_fz = reinterpret_cast<double*>(hi->d_fz.get());
  auto* d_pe_embed = reinterpret_cast<double*>(hi->d_pe_embed.get());
  auto* d_pe_pair = reinterpret_cast<double*>(hi->d_pe_pair.get());
  auto* d_virial = reinterpret_cast<double*>(hi->d_virial.get());
  auto* h_pe_embed_ptr = reinterpret_cast<double*>(hi->h_pe_embed.get());
  auto* h_pe_pair_ptr = reinterpret_cast<double*>(hi->h_pe_pair.get());
  auto* h_virial_ptr = reinterpret_cast<double*>(hi->h_virial.get());

  // mem_stream waits for this slot's kernels to finish (event recorded in
  // compute_async), then runs D2H. Split-phase lets K enqueued H2Ds pass
  // through mem_stream back-to-back before any D2H blocks.
  check_cuda("wait kernels on mem_stream",
             cudaStreamWaitEvent(s_mem, raw_event(hi->kernels_done_event), 0));
  {
    TDMD_NVTX_RANGE("eam.async.d2h");
    check_cuda("D2H fx",
               cudaMemcpyAsync(hi->host_fx_out, d_fx, pos_bytes, cudaMemcpyDeviceToHost, s_mem));
    check_cuda("D2H fy",
               cudaMemcpyAsync(hi->host_fy_out, d_fy, pos_bytes, cudaMemcpyDeviceToHost, s_mem));
    check_cuda("D2H fz",
               cudaMemcpyAsync(hi->host_fz_out, d_fz, pos_bytes, cudaMemcpyDeviceToHost, s_mem));
    check_cuda(
        "D2H pe_embed",
        cudaMemcpyAsync(h_pe_embed_ptr, d_pe_embed, pos_bytes, cudaMemcpyDeviceToHost, s_mem));
    check_cuda("D2H pe_pair",
               cudaMemcpyAsync(h_pe_pair_ptr, d_pe_pair, pos_bytes, cudaMemcpyDeviceToHost, s_mem));
    check_cuda("D2H virial",
               cudaMemcpyAsync(h_virial_ptr,
                               d_virial,
                               n * 6u * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s_mem));
  }
  check_cuda("record d2h_done_event", cudaEventRecord(raw_event(hi->d2h_done_event), s_mem));
  check_cuda("event sync d2h", cudaEventSynchronize(raw_event(hi->d2h_done_event)));

  const auto* pe_embed_ptr = reinterpret_cast<const double*>(hi->h_pe_embed.get());
  const auto* pe_pair_ptr = reinterpret_cast<const double*>(hi->h_pe_pair.get());
  const auto* virial_ptr = reinterpret_cast<const double*>(hi->h_virial.get());
  const double pe_embed_total = kahan_sum_host(pe_embed_ptr, n);
  const double pe_pair_full = kahan_sum_host(pe_pair_ptr, n);
  result.potential_energy = pe_embed_total + 0.5 * pe_pair_full;

  double vsum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double vcomp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t base = i * 6u;
    for (int k = 0; k < 6; ++k) {
      const double y = virial_ptr[base + static_cast<std::size_t>(k)] - vcomp[k];
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

struct EamAlloyGpu::Impl {};
struct EamAlloyAsyncHandle::Impl {};

EamAlloyGpu::EamAlloyGpu() : impl_(std::make_unique<Impl>()) {}
EamAlloyGpu::~EamAlloyGpu() = default;
EamAlloyGpu::EamAlloyGpu(EamAlloyGpu&&) noexcept = default;
EamAlloyGpu& EamAlloyGpu::operator=(EamAlloyGpu&&) noexcept = default;

EamAlloyAsyncHandle::EamAlloyAsyncHandle() noexcept = default;
EamAlloyAsyncHandle::EamAlloyAsyncHandle(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
EamAlloyAsyncHandle::~EamAlloyAsyncHandle() = default;
EamAlloyAsyncHandle::EamAlloyAsyncHandle(EamAlloyAsyncHandle&&) noexcept = default;
EamAlloyAsyncHandle& EamAlloyAsyncHandle::operator=(EamAlloyAsyncHandle&&) noexcept = default;
bool EamAlloyAsyncHandle::valid() const noexcept {
  return static_cast<bool>(impl_);
}

std::uint64_t EamAlloyGpu::compute_version() const noexcept {
  return 0;
}

std::uint64_t EamAlloyGpu::splines_upload_count() const noexcept {
  return 0;
}

EamAlloyGpuResult EamAlloyGpu::compute(std::size_t /*n*/,
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
      "gpu::EamAlloyGpu::compute: CPU-only build (TDMD_BUILD_CUDA=0); CUDA not linked");
}

EamAlloyAsyncHandle EamAlloyGpu::compute_async(std::size_t /*n*/,
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
                                               DeviceStream& /*compute_stream*/,
                                               DeviceStream& /*mem_stream*/) {
  throw std::runtime_error("gpu::EamAlloyGpu::compute_async: CPU-only build (TDMD_BUILD_CUDA=0)");
}

EamAlloyGpuResult EamAlloyGpu::finalize_async(EamAlloyAsyncHandle /*handle*/) {
  throw std::runtime_error("gpu::EamAlloyGpu::finalize_async: CPU-only build (TDMD_BUILD_CUDA=0)");
}

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
