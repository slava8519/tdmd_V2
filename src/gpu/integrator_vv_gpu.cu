// SPEC: docs/specs/gpu/SPEC.md §7.3 (VV NVE contract), §6.3 (D-M6-7 gate),
//       §8.1 (Reference FP64), §9 (NVTX — reserved for T6.11)
// Module SPEC: docs/specs/integrator/SPEC.md §3 (VV math), §3.5 (GPU-resident),
//              §8.1 (FP64 Reference precision)
// Exec pack: docs/development/m6_execution_pack.md T6.6
// Decisions: D-M6-4, D-M6-7, D-M6-17
//
// VV NVE GPU kernels. Two entry points mirror CPU
// `VelocityVerletIntegrator::{pre,post}_force_step`. Per-atom thread; no
// reductions, no atomics → deterministic element-wise FP64 math. With
// Reference flavor's `--fmad=false` (cmake/BuildFlavors.cmake §17), each
// kernel produces bit-exact results vs the CPU integrator for identical
// inputs — the D-M6-7 gate requires literal equality on this path.
//
// ftm2v (LAMMPS metal-units conversion factor, 1/1.0364269e-4 ≈ 9648.533)
// is folded into a per-species `accel[s] = ftm2v / mass[s]` table on the
// host side by the adapter; the kernel does a single multiply per axis.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/integrator_vv_gpu.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/telemetry/nvtx.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#if TDMD_BUILD_CUDA
#include "cuda_handles.hpp"

#include <cuda_runtime.h>
#endif

namespace tdmd::gpu {

namespace {

#if TDMD_BUILD_CUDA

[[noreturn]] void throw_cuda(const char* op, cudaError_t err) {
  std::ostringstream oss;
  oss << "gpu::VelocityVerletGpu::" << op << ": " << cudaGetErrorName(err) << " ("
      << cudaGetErrorString(err) << ")";
  throw std::runtime_error(oss.str());
}

void check_cuda(const char* op, cudaError_t err) {
  if (err != cudaSuccess) {
    throw_cuda(op, err);
  }
}

constexpr int kThreadsPerBlock = 128;

// Half-kick + drift. Element-wise, thread-per-atom. Matches CPU
// VelocityVerletIntegrator::pre_force_step operand order exactly:
//   v += accel[type] * f * half_dt
//   x += v * dt
__global__ void pre_force_kernel(std::uint32_t n,
                                 double half_dt,
                                 double dt,
                                 const double* __restrict__ accel_by_species,
                                 const std::uint32_t* __restrict__ type,
                                 const double* __restrict__ fx,
                                 const double* __restrict__ fy,
                                 const double* __restrict__ fz,
                                 double* __restrict__ x,
                                 double* __restrict__ y,
                                 double* __restrict__ z,
                                 double* __restrict__ vx,
                                 double* __restrict__ vy,
                                 double* __restrict__ vz) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double accel = accel_by_species[type[i]];
  double vxi = vx[i] + fx[i] * accel * half_dt;
  double vyi = vy[i] + fy[i] * accel * half_dt;
  double vzi = vz[i] + fz[i] * accel * half_dt;
  vx[i] = vxi;
  vy[i] = vyi;
  vz[i] = vzi;
  x[i] = x[i] + vxi * dt;
  y[i] = y[i] + vyi * dt;
  z[i] = z[i] + vzi * dt;
}

// Half-kick only. Matches CPU post_force_step operand order exactly.
__global__ void post_force_kernel(std::uint32_t n,
                                  double half_dt,
                                  const double* __restrict__ accel_by_species,
                                  const std::uint32_t* __restrict__ type,
                                  const double* __restrict__ fx,
                                  const double* __restrict__ fy,
                                  const double* __restrict__ fz,
                                  double* __restrict__ vx,
                                  double* __restrict__ vy,
                                  double* __restrict__ vz) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double accel = accel_by_species[type[i]];
  vx[i] += fx[i] * accel * half_dt;
  vy[i] += fy[i] * accel * half_dt;
  vz[i] += fz[i] * accel * half_dt;
}

#endif  // TDMD_BUILD_CUDA

}  // namespace

#if TDMD_BUILD_CUDA

struct VelocityVerletGpu::Impl {
  std::uint64_t compute_version = 0;
};

VelocityVerletGpu::VelocityVerletGpu() : impl_(std::make_unique<Impl>()) {}
VelocityVerletGpu::~VelocityVerletGpu() = default;
VelocityVerletGpu::VelocityVerletGpu(VelocityVerletGpu&&) noexcept = default;
VelocityVerletGpu& VelocityVerletGpu::operator=(VelocityVerletGpu&&) noexcept = default;

std::uint64_t VelocityVerletGpu::compute_version() const noexcept {
  return impl_ ? impl_->compute_version : 0;
}

void VelocityVerletGpu::pre_force_step(std::size_t n,
                                       double dt,
                                       std::size_t n_species,
                                       const double* host_accel_by_species,
                                       const std::uint32_t* host_types,
                                       const double* host_fx,
                                       const double* host_fy,
                                       const double* host_fz,
                                       double* host_x,
                                       double* host_y,
                                       double* host_z,
                                       double* host_vx,
                                       double* host_vy,
                                       double* host_vz,
                                       DevicePool& pool,
                                       DeviceStream& stream) {
  TDMD_NVTX_RANGE("vv.pre_force_step");

  ++impl_->compute_version;
  if (n == 0) {
    return;
  }

  cudaStream_t s = raw_stream(stream);

  const std::size_t pos_bytes = n * sizeof(double);
  const std::size_t type_bytes = n * sizeof(std::uint32_t);
  const std::size_t accel_bytes = n_species * sizeof(double);

  DevicePtr<std::byte> d_accel_bytes = pool.allocate_device(accel_bytes, stream);
  DevicePtr<std::byte> d_type_bytes = pool.allocate_device(type_bytes, stream);
  DevicePtr<std::byte> d_fx_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_fy_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_fz_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_x_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_y_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_z_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_vx_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_vy_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_vz_bytes = pool.allocate_device(pos_bytes, stream);

  auto* d_accel = reinterpret_cast<double*>(d_accel_bytes.get());
  auto* d_type = reinterpret_cast<std::uint32_t*>(d_type_bytes.get());
  auto* d_fx = reinterpret_cast<double*>(d_fx_bytes.get());
  auto* d_fy = reinterpret_cast<double*>(d_fy_bytes.get());
  auto* d_fz = reinterpret_cast<double*>(d_fz_bytes.get());
  auto* d_x = reinterpret_cast<double*>(d_x_bytes.get());
  auto* d_y = reinterpret_cast<double*>(d_y_bytes.get());
  auto* d_z = reinterpret_cast<double*>(d_z_bytes.get());
  auto* d_vx = reinterpret_cast<double*>(d_vx_bytes.get());
  auto* d_vy = reinterpret_cast<double*>(d_vy_bytes.get());
  auto* d_vz = reinterpret_cast<double*>(d_vz_bytes.get());

  {
    TDMD_NVTX_RANGE("vv.h2d.pre");
    check_cuda(
        "cudaMemcpyAsync accel",
        cudaMemcpyAsync(d_accel, host_accel_by_species, accel_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync type",
               cudaMemcpyAsync(d_type, host_types, type_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync fx",
               cudaMemcpyAsync(d_fx, host_fx, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync fy",
               cudaMemcpyAsync(d_fy, host_fy, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync fz",
               cudaMemcpyAsync(d_fz, host_fz, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync x",
               cudaMemcpyAsync(d_x, host_x, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync y",
               cudaMemcpyAsync(d_y, host_y, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync z",
               cudaMemcpyAsync(d_z, host_z, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync vx",
               cudaMemcpyAsync(d_vx, host_vx, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync vy",
               cudaMemcpyAsync(d_vy, host_vy, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync vz",
               cudaMemcpyAsync(d_vz, host_vz, pos_bytes, cudaMemcpyHostToDevice, s));
  }

  const double half_dt = 0.5 * dt;
  const std::uint32_t n32 = static_cast<std::uint32_t>(n);
  const std::uint32_t nblocks = (n32 + kThreadsPerBlock - 1) / kThreadsPerBlock;

  {
    TDMD_NVTX_RANGE("vv.pre_force_kernel");
    pre_force_kernel<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                          half_dt,
                                                          dt,
                                                          d_accel,
                                                          d_type,
                                                          d_fx,
                                                          d_fy,
                                                          d_fz,
                                                          d_x,
                                                          d_y,
                                                          d_z,
                                                          d_vx,
                                                          d_vy,
                                                          d_vz);
    check_cuda("launch pre_force_kernel", cudaGetLastError());
  }

  {
    TDMD_NVTX_RANGE("vv.d2h.pre");
    check_cuda("cudaMemcpyAsync D2H x",
               cudaMemcpyAsync(host_x, d_x, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("cudaMemcpyAsync D2H y",
               cudaMemcpyAsync(host_y, d_y, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("cudaMemcpyAsync D2H z",
               cudaMemcpyAsync(host_z, d_z, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("cudaMemcpyAsync D2H vx",
               cudaMemcpyAsync(host_vx, d_vx, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("cudaMemcpyAsync D2H vy",
               cudaMemcpyAsync(host_vy, d_vy, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("cudaMemcpyAsync D2H vz",
               cudaMemcpyAsync(host_vz, d_vz, pos_bytes, cudaMemcpyDeviceToHost, s));

    check_cuda("cudaStreamSynchronize (pre_force)", cudaStreamSynchronize(s));
  }
}

void VelocityVerletGpu::post_force_step(std::size_t n,
                                        double dt,
                                        std::size_t n_species,
                                        const double* host_accel_by_species,
                                        const std::uint32_t* host_types,
                                        const double* host_fx,
                                        const double* host_fy,
                                        const double* host_fz,
                                        double* host_vx,
                                        double* host_vy,
                                        double* host_vz,
                                        DevicePool& pool,
                                        DeviceStream& stream) {
  TDMD_NVTX_RANGE("vv.post_force_step");

  ++impl_->compute_version;
  if (n == 0) {
    return;
  }

  cudaStream_t s = raw_stream(stream);

  const std::size_t pos_bytes = n * sizeof(double);
  const std::size_t type_bytes = n * sizeof(std::uint32_t);
  const std::size_t accel_bytes = n_species * sizeof(double);

  DevicePtr<std::byte> d_accel_bytes = pool.allocate_device(accel_bytes, stream);
  DevicePtr<std::byte> d_type_bytes = pool.allocate_device(type_bytes, stream);
  DevicePtr<std::byte> d_fx_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_fy_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_fz_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_vx_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_vy_bytes = pool.allocate_device(pos_bytes, stream);
  DevicePtr<std::byte> d_vz_bytes = pool.allocate_device(pos_bytes, stream);

  auto* d_accel = reinterpret_cast<double*>(d_accel_bytes.get());
  auto* d_type = reinterpret_cast<std::uint32_t*>(d_type_bytes.get());
  auto* d_fx = reinterpret_cast<double*>(d_fx_bytes.get());
  auto* d_fy = reinterpret_cast<double*>(d_fy_bytes.get());
  auto* d_fz = reinterpret_cast<double*>(d_fz_bytes.get());
  auto* d_vx = reinterpret_cast<double*>(d_vx_bytes.get());
  auto* d_vy = reinterpret_cast<double*>(d_vy_bytes.get());
  auto* d_vz = reinterpret_cast<double*>(d_vz_bytes.get());

  {
    TDMD_NVTX_RANGE("vv.h2d.post");
    check_cuda(
        "cudaMemcpyAsync accel",
        cudaMemcpyAsync(d_accel, host_accel_by_species, accel_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync type",
               cudaMemcpyAsync(d_type, host_types, type_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync fx",
               cudaMemcpyAsync(d_fx, host_fx, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync fy",
               cudaMemcpyAsync(d_fy, host_fy, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync fz",
               cudaMemcpyAsync(d_fz, host_fz, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync vx",
               cudaMemcpyAsync(d_vx, host_vx, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync vy",
               cudaMemcpyAsync(d_vy, host_vy, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync vz",
               cudaMemcpyAsync(d_vz, host_vz, pos_bytes, cudaMemcpyHostToDevice, s));
  }

  const double half_dt = 0.5 * dt;
  const std::uint32_t n32 = static_cast<std::uint32_t>(n);
  const std::uint32_t nblocks = (n32 + kThreadsPerBlock - 1) / kThreadsPerBlock;

  {
    TDMD_NVTX_RANGE("vv.post_force_kernel");
    post_force_kernel<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                           half_dt,
                                                           d_accel,
                                                           d_type,
                                                           d_fx,
                                                           d_fy,
                                                           d_fz,
                                                           d_vx,
                                                           d_vy,
                                                           d_vz);
    check_cuda("launch post_force_kernel", cudaGetLastError());
  }

  {
    TDMD_NVTX_RANGE("vv.d2h.post");
    check_cuda("cudaMemcpyAsync D2H vx",
               cudaMemcpyAsync(host_vx, d_vx, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("cudaMemcpyAsync D2H vy",
               cudaMemcpyAsync(host_vy, d_vy, pos_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("cudaMemcpyAsync D2H vz",
               cudaMemcpyAsync(host_vz, d_vz, pos_bytes, cudaMemcpyDeviceToHost, s));

    check_cuda("cudaStreamSynchronize (post_force)", cudaStreamSynchronize(s));
  }
}

#else  // CPU-only build — stubs that throw, mirroring NL/EAM pattern.

struct VelocityVerletGpu::Impl {};

VelocityVerletGpu::VelocityVerletGpu() : impl_(std::make_unique<Impl>()) {}
VelocityVerletGpu::~VelocityVerletGpu() = default;
VelocityVerletGpu::VelocityVerletGpu(VelocityVerletGpu&&) noexcept = default;
VelocityVerletGpu& VelocityVerletGpu::operator=(VelocityVerletGpu&&) noexcept = default;

std::uint64_t VelocityVerletGpu::compute_version() const noexcept {
  return 0;
}

void VelocityVerletGpu::pre_force_step(std::size_t /*n*/,
                                       double /*dt*/,
                                       std::size_t /*n_species*/,
                                       const double* /*host_accel_by_species*/,
                                       const std::uint32_t* /*host_types*/,
                                       const double* /*host_fx*/,
                                       const double* /*host_fy*/,
                                       const double* /*host_fz*/,
                                       double* /*host_x*/,
                                       double* /*host_y*/,
                                       double* /*host_z*/,
                                       double* /*host_vx*/,
                                       double* /*host_vy*/,
                                       double* /*host_vz*/,
                                       DevicePool& /*pool*/,
                                       DeviceStream& /*stream*/) {
  throw std::runtime_error("VelocityVerletGpu::pre_force_step: CUDA disabled (TDMD_BUILD_CUDA=0)");
}

void VelocityVerletGpu::post_force_step(std::size_t /*n*/,
                                        double /*dt*/,
                                        std::size_t /*n_species*/,
                                        const double* /*host_accel_by_species*/,
                                        const std::uint32_t* /*host_types*/,
                                        const double* /*host_fx*/,
                                        const double* /*host_fy*/,
                                        const double* /*host_fz*/,
                                        double* /*host_vx*/,
                                        double* /*host_vy*/,
                                        double* /*host_vz*/,
                                        DevicePool& /*pool*/,
                                        DeviceStream& /*stream*/) {
  throw std::runtime_error("VelocityVerletGpu::post_force_step: CUDA disabled (TDMD_BUILD_CUDA=0)");
}

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
