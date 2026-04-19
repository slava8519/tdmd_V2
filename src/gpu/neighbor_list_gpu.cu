// SPEC: docs/specs/gpu/SPEC.md §7.1 (kernel contract), §6 (determinism),
//       §9 (NVTX ranges)
// Module SPEC: docs/specs/neighbor/SPEC.md §2.1 (CSR layout), §4 (build)
// Exec pack: docs/development/m6_execution_pack.md T6.4
// Decisions: D-M6-7 (CPU↔GPU bit-exact Reference), D-M6-14 (NVTX),
//            D-M6-16 (SoA half-list), D-M6-17 (PIMPL firewall)
//
// Device-side mirror of `NeighborList::build()`. Iteration order is
// byte-identical to the CPU implementation:
//   for i in [0, N):
//     ci = cell_of(atoms[i])
//     for (dz, dy, dx) in {-1,0,+1}³ (triple-nested, dz outermost):
//       cj = wrap(ci + (dx,dy,dz))
//       for k in cell_atoms[cell_offsets[cj] .. cell_offsets[cj+1]):
//         j = cell_atoms[k]
//         if j <= i: continue
//         r2 = |unwrap_minimum_image(atoms[j] - atoms[i])|²
//         if r2 <= (cutoff+skin)²: emit (j, r2)
//
// Each pass iterates in that order; both passes share a single device
// function template (`scan_neighbors`) so divergence between count + emit
// is ruled out at the source level.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"
#include "tdmd/gpu/types.hpp"

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
  oss << "gpu::NeighborListGpu::" << op << ": " << cudaGetErrorName(err) << " ("
      << cudaGetErrorString(err) << ")";
  throw std::runtime_error(oss.str());
}

void check_cuda(const char* op, cudaError_t err) {
  if (err != cudaSuccess) {
    throw_cuda(op, err);
  }
}

// POD captured by the kernels — kept tiny so it passes as a launch arg.
// Mirror of BoxParams but with integer flags for direct device use.
struct DeviceBoxParams {
  double xlo, ylo, zlo;
  double lx, ly, lz;
  double cell_x, cell_y, cell_z;
  std::uint32_t nx, ny, nz;
  int periodic_x, periodic_y, periodic_z;
  double reach_sq;
};

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

// Walks the 27-cell stencil around atom `i` applying a per-neighbor
// callable `F(j, r2)`. Same loop structure used by both kernels.
template <typename F>
__device__ __forceinline__ void scan_neighbors(std::uint32_t i,
                                               const double* __restrict__ x,
                                               const double* __restrict__ y,
                                               const double* __restrict__ z,
                                               const std::uint32_t* __restrict__ cell_offsets,
                                               const std::uint32_t* __restrict__ cell_atoms,
                                               const DeviceBoxParams& p,
                                               F&& emit) {
  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];

  const std::size_t ci_idx =
      linear_index_dev(static_cast<std::uint32_t>(cell_index_axis_dev(xi, p.xlo, p.cell_x, p.nx)),
                       static_cast<std::uint32_t>(cell_index_axis_dev(yi, p.ylo, p.cell_y, p.ny)),
                       static_cast<std::uint32_t>(cell_index_axis_dev(zi, p.zlo, p.cell_z, p.nz)),
                       p.nx,
                       p.ny);

  const auto iz_u = static_cast<int>(ci_idx / (static_cast<std::size_t>(p.nx) * p.ny));
  const auto rem = ci_idx - static_cast<std::size_t>(iz_u) * p.nx * p.ny;
  const auto iy_u = static_cast<int>(rem / p.nx);
  const auto ix_u = static_cast<int>(rem % p.nx);

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
          if (j <= i) {
            continue;
          }
          double ddx = x[j] - xi;
          double ddy = y[j] - yi;
          double ddz = z[j] - zi;
          ddx = minimum_image_axis_dev(ddx, p.lx, p.periodic_x);
          ddy = minimum_image_axis_dev(ddy, p.ly, p.periodic_y);
          ddz = minimum_image_axis_dev(ddz, p.lz, p.periodic_z);
          const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
          if (r2 <= p.reach_sq) {
            emit(j, r2);
          }
        }
      }
    }
  }
}

__global__ void count_neighbors_kernel(std::uint32_t n,
                                       const double* __restrict__ x,
                                       const double* __restrict__ y,
                                       const double* __restrict__ z,
                                       const std::uint32_t* __restrict__ cell_offsets,
                                       const std::uint32_t* __restrict__ cell_atoms,
                                       DeviceBoxParams p,
                                       std::uint32_t* __restrict__ counts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  std::uint32_t c = 0;
  scan_neighbors(i, x, y, z, cell_offsets, cell_atoms, p, [&c](std::uint32_t, double) { ++c; });
  counts[i] = c;
}

__global__ void emit_neighbors_kernel(std::uint32_t n,
                                      const double* __restrict__ x,
                                      const double* __restrict__ y,
                                      const double* __restrict__ z,
                                      const std::uint32_t* __restrict__ cell_offsets,
                                      const std::uint32_t* __restrict__ cell_atoms,
                                      const std::uint64_t* __restrict__ offsets,
                                      DeviceBoxParams p,
                                      std::uint32_t* __restrict__ ids,
                                      double* __restrict__ r2_out) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  std::uint64_t cursor = offsets[i];
  scan_neighbors(i,
                 x,
                 y,
                 z,
                 cell_offsets,
                 cell_atoms,
                 p,
                 [ids, r2_out, &cursor](std::uint32_t j, double r2) {
                   ids[cursor] = j;
                   r2_out[cursor] = r2;
                   ++cursor;
                 });
}

#endif  // TDMD_BUILD_CUDA

}  // namespace

#if TDMD_BUILD_CUDA

struct NeighborListGpu::Impl {
  std::size_t atom_count = 0;
  std::size_t pair_count = 0;
  std::uint64_t build_version = 0;
  double cutoff = 0.0;
  double skin = 0.0;

  // DevicePool-owned device buffers. Re-allocated each build.
  DevicePtr<std::byte> d_x_bytes;
  DevicePtr<std::byte> d_y_bytes;
  DevicePtr<std::byte> d_z_bytes;
  DevicePtr<std::byte> d_cell_offsets_bytes;
  DevicePtr<std::byte> d_cell_atoms_bytes;
  DevicePtr<std::byte> d_offsets_bytes;
  DevicePtr<std::byte> d_ids_bytes;
  DevicePtr<std::byte> d_r2_bytes;

  double* d_x = nullptr;
  double* d_y = nullptr;
  double* d_z = nullptr;
  std::uint32_t* d_cell_offsets = nullptr;
  std::uint32_t* d_cell_atoms = nullptr;
  std::uint64_t* d_offsets = nullptr;
  std::uint32_t* d_ids = nullptr;
  double* d_r2 = nullptr;
};

NeighborListGpu::NeighborListGpu() : impl_(std::make_unique<Impl>()) {}
NeighborListGpu::~NeighborListGpu() = default;
NeighborListGpu::NeighborListGpu(NeighborListGpu&&) noexcept = default;
NeighborListGpu& NeighborListGpu::operator=(NeighborListGpu&&) noexcept = default;

std::size_t NeighborListGpu::atom_count() const noexcept {
  return impl_ ? impl_->atom_count : 0;
}
std::size_t NeighborListGpu::pair_count() const noexcept {
  return impl_ ? impl_->pair_count : 0;
}
std::uint64_t NeighborListGpu::build_version() const noexcept {
  return impl_ ? impl_->build_version : 0;
}
double NeighborListGpu::cutoff() const noexcept {
  return impl_ ? impl_->cutoff : 0.0;
}
double NeighborListGpu::skin() const noexcept {
  return impl_ ? impl_->skin : 0.0;
}

NeighborListGpuView NeighborListGpu::view() const noexcept {
  NeighborListGpuView v;
  if (!impl_) {
    return v;
  }
  v.atom_count = impl_->atom_count;
  v.pair_count = impl_->pair_count;
  v.d_offsets = impl_->d_offsets;
  v.d_ids = impl_->d_ids;
  v.d_r2 = impl_->d_r2;
  return v;
}

void NeighborListGpu::build(std::size_t n,
                            const double* host_x,
                            const double* host_y,
                            const double* host_z,
                            std::size_t ncells,
                            const std::uint32_t* host_cell_offsets,
                            const std::uint32_t* host_cell_atoms,
                            const BoxParams& params,
                            DevicePool& pool,
                            DeviceStream& stream) {
  impl_->atom_count = n;
  impl_->cutoff = params.cutoff;
  impl_->skin = params.skin;
  ++impl_->build_version;

  cudaStream_t s = raw_stream(stream);

  if (n == 0) {
    impl_->pair_count = 0;
    impl_->d_x_bytes.reset();
    impl_->d_y_bytes.reset();
    impl_->d_z_bytes.reset();
    impl_->d_cell_offsets_bytes.reset();
    impl_->d_cell_atoms_bytes.reset();
    impl_->d_offsets_bytes.reset();
    impl_->d_ids_bytes.reset();
    impl_->d_r2_bytes.reset();
    impl_->d_x = impl_->d_y = impl_->d_z = nullptr;
    impl_->d_cell_offsets = impl_->d_cell_atoms = nullptr;
    impl_->d_offsets = nullptr;
    impl_->d_ids = nullptr;
    impl_->d_r2 = nullptr;
    return;
  }

  // ---------- 1. H2D copies ----------
  const std::size_t pos_bytes = n * sizeof(double);
  impl_->d_x_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_y_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_z_bytes = pool.allocate_device(pos_bytes, stream);
  impl_->d_x = reinterpret_cast<double*>(impl_->d_x_bytes.get());
  impl_->d_y = reinterpret_cast<double*>(impl_->d_y_bytes.get());
  impl_->d_z = reinterpret_cast<double*>(impl_->d_z_bytes.get());
  check_cuda("cudaMemcpyAsync x",
             cudaMemcpyAsync(impl_->d_x, host_x, pos_bytes, cudaMemcpyHostToDevice, s));
  check_cuda("cudaMemcpyAsync y",
             cudaMemcpyAsync(impl_->d_y, host_y, pos_bytes, cudaMemcpyHostToDevice, s));
  check_cuda("cudaMemcpyAsync z",
             cudaMemcpyAsync(impl_->d_z, host_z, pos_bytes, cudaMemcpyHostToDevice, s));

  const std::size_t cell_offsets_bytes = (ncells + 1) * sizeof(std::uint32_t);
  const std::size_t cell_atoms_bytes = n * sizeof(std::uint32_t);
  impl_->d_cell_offsets_bytes = pool.allocate_device(cell_offsets_bytes, stream);
  impl_->d_cell_atoms_bytes = pool.allocate_device(cell_atoms_bytes, stream);
  impl_->d_cell_offsets = reinterpret_cast<std::uint32_t*>(impl_->d_cell_offsets_bytes.get());
  impl_->d_cell_atoms = reinterpret_cast<std::uint32_t*>(impl_->d_cell_atoms_bytes.get());
  check_cuda("cudaMemcpyAsync cell_offsets",
             cudaMemcpyAsync(impl_->d_cell_offsets,
                             host_cell_offsets,
                             cell_offsets_bytes,
                             cudaMemcpyHostToDevice,
                             s));
  check_cuda("cudaMemcpyAsync cell_atoms",
             cudaMemcpyAsync(impl_->d_cell_atoms,
                             host_cell_atoms,
                             cell_atoms_bytes,
                             cudaMemcpyHostToDevice,
                             s));

  // ---------- 2. Pass 1 — counts ----------
  const std::size_t counts_bytes = n * sizeof(std::uint32_t);
  DevicePtr<std::byte> d_counts_bytes = pool.allocate_device(counts_bytes, stream);
  auto* d_counts = reinterpret_cast<std::uint32_t*>(d_counts_bytes.get());

  DeviceBoxParams p;
  p.xlo = params.xlo;
  p.ylo = params.ylo;
  p.zlo = params.zlo;
  p.lx = params.lx;
  p.ly = params.ly;
  p.lz = params.lz;
  p.cell_x = params.cell_x;
  p.cell_y = params.cell_y;
  p.cell_z = params.cell_z;
  p.nx = params.nx;
  p.ny = params.ny;
  p.nz = params.nz;
  p.periodic_x = params.periodic_x ? 1 : 0;
  p.periodic_y = params.periodic_y ? 1 : 0;
  p.periodic_z = params.periodic_z ? 1 : 0;
  const double reach = params.cutoff + params.skin;
  p.reach_sq = reach * reach;

  const std::uint32_t n32 = static_cast<std::uint32_t>(n);
  constexpr int kThreadsPerBlock = 128;
  const std::uint32_t nblocks = (n32 + kThreadsPerBlock - 1) / kThreadsPerBlock;

  count_neighbors_kernel<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                              impl_->d_x,
                                                              impl_->d_y,
                                                              impl_->d_z,
                                                              impl_->d_cell_offsets,
                                                              impl_->d_cell_atoms,
                                                              p,
                                                              d_counts);
  check_cuda("launch count_neighbors_kernel", cudaGetLastError());

  // ---------- 3. Exclusive scan on host ----------
  std::vector<std::uint32_t> host_counts(n);
  check_cuda(
      "cudaMemcpyAsync D2H counts",
      cudaMemcpyAsync(host_counts.data(), d_counts, counts_bytes, cudaMemcpyDeviceToHost, s));
  check_cuda("cudaStreamSynchronize (counts D2H)", cudaStreamSynchronize(s));

  std::vector<std::uint64_t> host_offsets(n + 1);
  host_offsets[0] = 0;
  for (std::size_t i = 0; i < n; ++i) {
    host_offsets[i + 1] = host_offsets[i] + static_cast<std::uint64_t>(host_counts[i]);
  }
  const std::uint64_t total_pairs = host_offsets[n];
  impl_->pair_count = static_cast<std::size_t>(total_pairs);

  const std::size_t offsets_bytes = (n + 1) * sizeof(std::uint64_t);
  impl_->d_offsets_bytes = pool.allocate_device(offsets_bytes, stream);
  impl_->d_offsets = reinterpret_cast<std::uint64_t*>(impl_->d_offsets_bytes.get());
  check_cuda("cudaMemcpyAsync H2D offsets",
             cudaMemcpyAsync(impl_->d_offsets,
                             host_offsets.data(),
                             offsets_bytes,
                             cudaMemcpyHostToDevice,
                             s));

  if (total_pairs == 0) {
    impl_->d_ids_bytes.reset();
    impl_->d_r2_bytes.reset();
    impl_->d_ids = nullptr;
    impl_->d_r2 = nullptr;
    check_cuda("cudaStreamSynchronize (empty NL)", cudaStreamSynchronize(s));
    return;
  }

  // ---------- 4. Pass 2 — emit ----------
  const std::size_t ids_bytes = total_pairs * sizeof(std::uint32_t);
  const std::size_t r2_bytes = total_pairs * sizeof(double);
  impl_->d_ids_bytes = pool.allocate_device(ids_bytes, stream);
  impl_->d_r2_bytes = pool.allocate_device(r2_bytes, stream);
  impl_->d_ids = reinterpret_cast<std::uint32_t*>(impl_->d_ids_bytes.get());
  impl_->d_r2 = reinterpret_cast<double*>(impl_->d_r2_bytes.get());

  emit_neighbors_kernel<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                             impl_->d_x,
                                                             impl_->d_y,
                                                             impl_->d_z,
                                                             impl_->d_cell_offsets,
                                                             impl_->d_cell_atoms,
                                                             impl_->d_offsets,
                                                             p,
                                                             impl_->d_ids,
                                                             impl_->d_r2);
  check_cuda("launch emit_neighbors_kernel", cudaGetLastError());
  check_cuda("cudaStreamSynchronize (NL build)", cudaStreamSynchronize(s));
}

NeighborListHostSnapshot NeighborListGpu::download(DeviceStream& stream) const {
  NeighborListHostSnapshot snap;
  if (!impl_ || impl_->atom_count == 0) {
    return snap;
  }
  cudaStream_t s = raw_stream(stream);
  const std::size_t n = impl_->atom_count;
  const std::size_t np = impl_->pair_count;

  snap.offsets.assign(n + 1, 0);
  if (impl_->d_offsets != nullptr) {
    check_cuda("cudaMemcpyAsync D2H offsets",
               cudaMemcpyAsync(snap.offsets.data(),
                               impl_->d_offsets,
                               (n + 1) * sizeof(std::uint64_t),
                               cudaMemcpyDeviceToHost,
                               s));
  }

  snap.ids.assign(np, 0);
  snap.r2.assign(np, 0.0);
  if (np > 0) {
    check_cuda("cudaMemcpyAsync D2H ids",
               cudaMemcpyAsync(snap.ids.data(),
                               impl_->d_ids,
                               np * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("cudaMemcpyAsync D2H r2",
               cudaMemcpyAsync(snap.r2.data(),
                               impl_->d_r2,
                               np * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s));
  }

  check_cuda("cudaStreamSynchronize (download)", cudaStreamSynchronize(s));
  return snap;
}

#else  // CPU-only build

struct NeighborListGpu::Impl {};

NeighborListGpu::NeighborListGpu() : impl_(std::make_unique<Impl>()) {}
NeighborListGpu::~NeighborListGpu() = default;
NeighborListGpu::NeighborListGpu(NeighborListGpu&&) noexcept = default;
NeighborListGpu& NeighborListGpu::operator=(NeighborListGpu&&) noexcept = default;

std::size_t NeighborListGpu::atom_count() const noexcept {
  return 0;
}
std::size_t NeighborListGpu::pair_count() const noexcept {
  return 0;
}
std::uint64_t NeighborListGpu::build_version() const noexcept {
  return 0;
}
double NeighborListGpu::cutoff() const noexcept {
  return 0.0;
}
double NeighborListGpu::skin() const noexcept {
  return 0.0;
}

NeighborListGpuView NeighborListGpu::view() const noexcept {
  return {};
}

void NeighborListGpu::build(std::size_t /*n*/,
                            const double* /*host_x*/,
                            const double* /*host_y*/,
                            const double* /*host_z*/,
                            std::size_t /*ncells*/,
                            const std::uint32_t* /*host_cell_offsets*/,
                            const std::uint32_t* /*host_cell_atoms*/,
                            const BoxParams& /*params*/,
                            DevicePool& /*pool*/,
                            DeviceStream& /*stream*/) {
  throw std::runtime_error(
      "gpu::NeighborListGpu::build: CPU-only build (TDMD_BUILD_CUDA=0); CUDA not linked");
}

NeighborListHostSnapshot NeighborListGpu::download(DeviceStream& /*stream*/) const {
  return {};
}

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
