// SPEC: docs/specs/gpu/SPEC.md §6.1 (no atomicAdd(double); reduce-then-scatter),
//       §7.5 (T8.6c-v5 per-bond dispatch), §9 (NVTX ranges on every launch).
// Module SPEC: docs/specs/potentials/SPEC.md §6 (SnapPotential).
// Pre-impl:  docs/development/t8.6c-v5_pre_impl.md (Stage 1).
//
// Three-pass CSR bond-list build (same shape as NeighborListGpu):
//   1. H2D copies: types, positions, cell CSR, per-pair cutsq matrix.
//   2. Pass 1 `count_bonds_kernel` — one thread per atom i; walks the 27-cell
//      stencil, counts bonds that pass SNAP's per-pair filter.
//   3. Exclusive scan on host (D2H counts → scan → H2D offsets). Scan buffer
//      is `uint32_t[n+1]`; for expected bond counts ≤ a few million per zone
//      this fits in 16 MB and stays under the device pool budget.
//   4. Pass 2 `emit_bonds_kernel` — identical iteration order, writes bonds
//      into per-atom reserved ranges `[atom_bond_start[i], ...)`.
//
// Determinism: both passes share `scan_snap_bonds` (device template) so the
// loop/filter logic cannot diverge between count and emit at the source level.
// Emission order matches `snap_ui_kernel` exactly (dz-outer, dy, dx, k-inner),
// which is what the Stage 2/3 gather kernels rely on for the ≤ 1e-12 rel T8.7
// byte-exact gate: the gather Kahan accumulator sums bonds in the same order
// CPU `SnaEngine::compute_ui` visits neighbours, preserving the FP64 sum.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/snap_bond_list_gpu.hpp"
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
  oss << "gpu::SnapBondListGpu::" << op << ": " << cudaGetErrorName(err) << " ("
      << cudaGetErrorString(err) << ")";
  throw std::runtime_error(oss.str());
}

void check_cuda(const char* op, cudaError_t err) {
  if (err != cudaSuccess) {
    throw_cuda(op, err);
  }
}

// POD passed by value to the kernels. Mirrors BoxParams with integer periodic
// flags + n_species added for per-pair cutsq indexing.
struct DeviceBondBoxParams {
  double xlo, ylo, zlo;
  double lx, ly, lz;
  double cell_x, cell_y, cell_z;
  std::uint32_t nx, ny, nz;
  int periodic_x, periodic_y, periodic_z;
  std::uint32_t n_species;
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

__device__ __forceinline__ std::size_t pair_index_dev(std::uint32_t a,
                                                      std::uint32_t b,
                                                      std::uint32_t n_species) {
  return static_cast<std::size_t>(a) * static_cast<std::size_t>(n_species) +
         static_cast<std::size_t>(b);
}

// Walks the 3×3×3 stencil around atom `i`, applying the SNAP per-pair filter
// and invoking `emit(j, ddx, ddy, ddz, rsq, itype, jtype)` on every surviving
// bond. Mirrors snap_ui_kernel's nested loop precisely — count + emit use the
// same instance so divergence between passes is ruled out at the source.
template <typename F>
__device__ __forceinline__ void scan_snap_bonds(std::uint32_t i,
                                                const std::uint32_t* __restrict__ types,
                                                const double* __restrict__ x,
                                                const double* __restrict__ y,
                                                const double* __restrict__ z,
                                                const std::uint32_t* __restrict__ cell_offsets,
                                                const std::uint32_t* __restrict__ cell_atoms,
                                                const double* __restrict__ rcut_sq_ab,
                                                const DeviceBondBoxParams& p,
                                                F&& emit) {
  const std::uint32_t itype = types[i];
  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];

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
          // Same guard as snap_ui_kernel: rsq < cutsq and rsq > 1e-20 (prevents
          // co-located atom numerics in the downstream compute_uarray recurrence).
          if (!(rsq < cutsq_ij) || !(rsq > 1e-20)) {
            continue;
          }
          emit(j, ddx, ddy, ddz, rsq, itype, jtype);
        }
      }
    }
  }
}

__global__ void count_bonds_kernel(std::uint32_t n,
                                   const std::uint32_t* __restrict__ types,
                                   const double* __restrict__ x,
                                   const double* __restrict__ y,
                                   const double* __restrict__ z,
                                   const std::uint32_t* __restrict__ cell_offsets,
                                   const std::uint32_t* __restrict__ cell_atoms,
                                   const double* __restrict__ rcut_sq_ab,
                                   DeviceBondBoxParams p,
                                   std::uint32_t* __restrict__ counts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  std::uint32_t c = 0;
  scan_snap_bonds(
      i,
      types,
      x,
      y,
      z,
      cell_offsets,
      cell_atoms,
      rcut_sq_ab,
      p,
      [&c](std::uint32_t, double, double, double, double, std::uint32_t, std::uint32_t) { ++c; });
  counts[i] = c;
}

__global__ void emit_bonds_kernel(std::uint32_t n,
                                  const std::uint32_t* __restrict__ types,
                                  const double* __restrict__ x,
                                  const double* __restrict__ y,
                                  const double* __restrict__ z,
                                  const std::uint32_t* __restrict__ cell_offsets,
                                  const std::uint32_t* __restrict__ cell_atoms,
                                  const double* __restrict__ rcut_sq_ab,
                                  const std::uint32_t* __restrict__ atom_bond_start,
                                  DeviceBondBoxParams p,
                                  std::uint32_t* __restrict__ bond_i,
                                  std::uint32_t* __restrict__ bond_j,
                                  std::uint32_t* __restrict__ bond_type_i,
                                  std::uint32_t* __restrict__ bond_type_j,
                                  double* __restrict__ bond_dx,
                                  double* __restrict__ bond_dy,
                                  double* __restrict__ bond_dz,
                                  double* __restrict__ bond_rsq) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  std::uint32_t cursor = atom_bond_start[i];
  scan_snap_bonds(i,
                  types,
                  x,
                  y,
                  z,
                  cell_offsets,
                  cell_atoms,
                  rcut_sq_ab,
                  p,
                  [&](std::uint32_t j,
                      double ddx,
                      double ddy,
                      double ddz,
                      double rsq,
                      std::uint32_t itype,
                      std::uint32_t jtype) {
                    bond_i[cursor] = i;
                    bond_j[cursor] = j;
                    bond_type_i[cursor] = itype;
                    bond_type_j[cursor] = jtype;
                    bond_dx[cursor] = ddx;
                    bond_dy[cursor] = ddy;
                    bond_dz[cursor] = ddz;
                    bond_rsq[cursor] = rsq;
                    ++cursor;
                  });
}

#endif  // TDMD_BUILD_CUDA

}  // namespace

#if TDMD_BUILD_CUDA

struct SnapBondListGpu::Impl {
  std::size_t atom_count = 0;
  std::size_t bond_count = 0;
  std::uint64_t build_version = 0;

  // Input mirrors (kept resident so the view pointers remain valid between
  // build() calls — same lifetime model as NeighborListGpu::Impl).
  DevicePtr<std::byte> d_types_bytes;
  DevicePtr<std::byte> d_x_bytes;
  DevicePtr<std::byte> d_y_bytes;
  DevicePtr<std::byte> d_z_bytes;
  DevicePtr<std::byte> d_cell_offsets_bytes;
  DevicePtr<std::byte> d_cell_atoms_bytes;
  DevicePtr<std::byte> d_rcut_sq_ab_bytes;

  // Outputs.
  DevicePtr<std::byte> d_atom_bond_start_bytes;
  DevicePtr<std::byte> d_bond_i_bytes;
  DevicePtr<std::byte> d_bond_j_bytes;
  DevicePtr<std::byte> d_bond_type_i_bytes;
  DevicePtr<std::byte> d_bond_type_j_bytes;
  DevicePtr<std::byte> d_bond_dx_bytes;
  DevicePtr<std::byte> d_bond_dy_bytes;
  DevicePtr<std::byte> d_bond_dz_bytes;
  DevicePtr<std::byte> d_bond_rsq_bytes;

  std::uint32_t* d_atom_bond_start = nullptr;
  std::uint32_t* d_bond_i = nullptr;
  std::uint32_t* d_bond_j = nullptr;
  std::uint32_t* d_bond_type_i = nullptr;
  std::uint32_t* d_bond_type_j = nullptr;
  double* d_bond_dx = nullptr;
  double* d_bond_dy = nullptr;
  double* d_bond_dz = nullptr;
  double* d_bond_rsq = nullptr;

  // Shared core used by both build() (after its H2D) and build_from_device()
  // (no H2D). Runs: count_kernel → D2H counts → host exclusive scan → H2D
  // offsets → emit_kernel → stream sync.
  void run_passes_from_device(std::size_t n,
                              const std::uint32_t* d_types,
                              const double* d_x,
                              const double* d_y,
                              const double* d_z,
                              const std::uint32_t* d_cell_offsets,
                              const std::uint32_t* d_cell_atoms,
                              const double* d_rcut_sq_ab,
                              std::uint32_t n_species,
                              const BoxParams& params,
                              DevicePool& pool,
                              DeviceStream& stream);
};

SnapBondListGpu::SnapBondListGpu() : impl_(std::make_unique<Impl>()) {}
SnapBondListGpu::~SnapBondListGpu() = default;
SnapBondListGpu::SnapBondListGpu(SnapBondListGpu&&) noexcept = default;
SnapBondListGpu& SnapBondListGpu::operator=(SnapBondListGpu&&) noexcept = default;

std::size_t SnapBondListGpu::atom_count() const noexcept {
  return impl_ ? impl_->atom_count : 0;
}
std::size_t SnapBondListGpu::bond_count() const noexcept {
  return impl_ ? impl_->bond_count : 0;
}
std::uint64_t SnapBondListGpu::build_version() const noexcept {
  return impl_ ? impl_->build_version : 0;
}

SnapBondListGpuView SnapBondListGpu::view() const noexcept {
  SnapBondListGpuView v;
  if (!impl_) {
    return v;
  }
  v.atom_count = impl_->atom_count;
  v.bond_count = impl_->bond_count;
  v.d_atom_bond_start = impl_->d_atom_bond_start;
  v.d_bond_i = impl_->d_bond_i;
  v.d_bond_j = impl_->d_bond_j;
  v.d_bond_type_i = impl_->d_bond_type_i;
  v.d_bond_type_j = impl_->d_bond_type_j;
  v.d_bond_dx = impl_->d_bond_dx;
  v.d_bond_dy = impl_->d_bond_dy;
  v.d_bond_dz = impl_->d_bond_dz;
  v.d_bond_rsq = impl_->d_bond_rsq;
  return v;
}

void SnapBondListGpu::build(std::size_t n,
                            const std::uint32_t* host_types,
                            const double* host_x,
                            const double* host_y,
                            const double* host_z,
                            std::size_t ncells,
                            const std::uint32_t* host_cell_offsets,
                            const std::uint32_t* host_cell_atoms,
                            const double* host_rcut_sq_ab,
                            std::uint32_t n_species,
                            const BoxParams& params,
                            DevicePool& pool,
                            DeviceStream& stream) {
  TDMD_NVTX_RANGE("snap.bond_list.build");

  impl_->atom_count = n;
  ++impl_->build_version;

  cudaStream_t s = raw_stream(stream);

  if (n == 0) {
    impl_->bond_count = 0;
    impl_->d_types_bytes.reset();
    impl_->d_x_bytes.reset();
    impl_->d_y_bytes.reset();
    impl_->d_z_bytes.reset();
    impl_->d_cell_offsets_bytes.reset();
    impl_->d_cell_atoms_bytes.reset();
    impl_->d_rcut_sq_ab_bytes.reset();
    impl_->d_atom_bond_start_bytes.reset();
    impl_->d_bond_i_bytes.reset();
    impl_->d_bond_j_bytes.reset();
    impl_->d_bond_type_i_bytes.reset();
    impl_->d_bond_type_j_bytes.reset();
    impl_->d_bond_dx_bytes.reset();
    impl_->d_bond_dy_bytes.reset();
    impl_->d_bond_dz_bytes.reset();
    impl_->d_bond_rsq_bytes.reset();
    impl_->d_atom_bond_start = nullptr;
    impl_->d_bond_i = impl_->d_bond_j = nullptr;
    impl_->d_bond_type_i = impl_->d_bond_type_j = nullptr;
    impl_->d_bond_dx = impl_->d_bond_dy = impl_->d_bond_dz = impl_->d_bond_rsq = nullptr;
    return;
  }

  // ---------- 1. H2D copies ----------
  {
    TDMD_NVTX_RANGE("snap.bond_list.h2d");
    const std::size_t pos_bytes = n * sizeof(double);
    const std::size_t types_bytes = n * sizeof(std::uint32_t);
    const std::size_t cell_offsets_bytes = (ncells + 1) * sizeof(std::uint32_t);
    const std::size_t cell_atoms_bytes = n * sizeof(std::uint32_t);
    const std::size_t rcut_bytes = static_cast<std::size_t>(n_species) * n_species * sizeof(double);

    impl_->d_types_bytes = pool.allocate_device(types_bytes, stream);
    impl_->d_x_bytes = pool.allocate_device(pos_bytes, stream);
    impl_->d_y_bytes = pool.allocate_device(pos_bytes, stream);
    impl_->d_z_bytes = pool.allocate_device(pos_bytes, stream);
    impl_->d_cell_offsets_bytes = pool.allocate_device(cell_offsets_bytes, stream);
    impl_->d_cell_atoms_bytes = pool.allocate_device(cell_atoms_bytes, stream);
    impl_->d_rcut_sq_ab_bytes = pool.allocate_device(rcut_bytes, stream);

    check_cuda("cudaMemcpyAsync types",
               cudaMemcpyAsync(impl_->d_types_bytes.get(),
                               host_types,
                               types_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda(
        "cudaMemcpyAsync x",
        cudaMemcpyAsync(impl_->d_x_bytes.get(), host_x, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda(
        "cudaMemcpyAsync y",
        cudaMemcpyAsync(impl_->d_y_bytes.get(), host_y, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda(
        "cudaMemcpyAsync z",
        cudaMemcpyAsync(impl_->d_z_bytes.get(), host_z, pos_bytes, cudaMemcpyHostToDevice, s));
    check_cuda("cudaMemcpyAsync cell_offsets",
               cudaMemcpyAsync(impl_->d_cell_offsets_bytes.get(),
                               host_cell_offsets,
                               cell_offsets_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("cudaMemcpyAsync cell_atoms",
               cudaMemcpyAsync(impl_->d_cell_atoms_bytes.get(),
                               host_cell_atoms,
                               cell_atoms_bytes,
                               cudaMemcpyHostToDevice,
                               s));
    check_cuda("cudaMemcpyAsync rcut_sq_ab",
               cudaMemcpyAsync(impl_->d_rcut_sq_ab_bytes.get(),
                               host_rcut_sq_ab,
                               rcut_bytes,
                               cudaMemcpyHostToDevice,
                               s));
  }

  auto* d_types = reinterpret_cast<std::uint32_t*>(impl_->d_types_bytes.get());
  auto* d_x = reinterpret_cast<double*>(impl_->d_x_bytes.get());
  auto* d_y = reinterpret_cast<double*>(impl_->d_y_bytes.get());
  auto* d_z = reinterpret_cast<double*>(impl_->d_z_bytes.get());
  auto* d_cell_offsets = reinterpret_cast<std::uint32_t*>(impl_->d_cell_offsets_bytes.get());
  auto* d_cell_atoms = reinterpret_cast<std::uint32_t*>(impl_->d_cell_atoms_bytes.get());
  auto* d_rcut_sq_ab = reinterpret_cast<double*>(impl_->d_rcut_sq_ab_bytes.get());

  impl_->run_passes_from_device(n,
                                d_types,
                                d_x,
                                d_y,
                                d_z,
                                d_cell_offsets,
                                d_cell_atoms,
                                d_rcut_sq_ab,
                                n_species,
                                params,
                                pool,
                                stream);
}

void SnapBondListGpu::build_from_device(std::size_t n,
                                        const std::uint32_t* d_types,
                                        const double* d_x,
                                        const double* d_y,
                                        const double* d_z,
                                        std::size_t /*ncells*/,
                                        const std::uint32_t* d_cell_offsets,
                                        const std::uint32_t* d_cell_atoms,
                                        const double* d_rcut_sq_ab,
                                        std::uint32_t n_species,
                                        const BoxParams& params,
                                        DevicePool& pool,
                                        DeviceStream& stream) {
  TDMD_NVTX_RANGE("snap.bond_list.build_from_device");

  impl_->atom_count = n;
  ++impl_->build_version;

  if (n == 0) {
    impl_->bond_count = 0;
    impl_->d_atom_bond_start_bytes.reset();
    impl_->d_bond_i_bytes.reset();
    impl_->d_bond_j_bytes.reset();
    impl_->d_bond_type_i_bytes.reset();
    impl_->d_bond_type_j_bytes.reset();
    impl_->d_bond_dx_bytes.reset();
    impl_->d_bond_dy_bytes.reset();
    impl_->d_bond_dz_bytes.reset();
    impl_->d_bond_rsq_bytes.reset();
    impl_->d_atom_bond_start = nullptr;
    impl_->d_bond_i = impl_->d_bond_j = nullptr;
    impl_->d_bond_type_i = impl_->d_bond_type_j = nullptr;
    impl_->d_bond_dx = impl_->d_bond_dy = impl_->d_bond_dz = impl_->d_bond_rsq = nullptr;
    return;
  }

  impl_->run_passes_from_device(n,
                                d_types,
                                d_x,
                                d_y,
                                d_z,
                                d_cell_offsets,
                                d_cell_atoms,
                                d_rcut_sq_ab,
                                n_species,
                                params,
                                pool,
                                stream);
}

void SnapBondListGpu::Impl::run_passes_from_device(std::size_t n,
                                                   const std::uint32_t* d_types,
                                                   const double* d_x,
                                                   const double* d_y,
                                                   const double* d_z,
                                                   const std::uint32_t* d_cell_offsets,
                                                   const std::uint32_t* d_cell_atoms,
                                                   const double* d_rcut_sq_ab,
                                                   std::uint32_t n_species,
                                                   const BoxParams& params,
                                                   DevicePool& pool,
                                                   DeviceStream& stream) {
  cudaStream_t s = raw_stream(stream);

  DeviceBondBoxParams p;
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
  p.n_species = n_species;

  const std::uint32_t n32 = static_cast<std::uint32_t>(n);
  constexpr int kThreadsPerBlock = 128;
  const std::uint32_t nblocks = (n32 + kThreadsPerBlock - 1) / kThreadsPerBlock;

  // ---------- 2. Pass 1 — counts ----------
  const std::size_t counts_bytes = n * sizeof(std::uint32_t);
  DevicePtr<std::byte> d_counts_bytes = pool.allocate_device(counts_bytes, stream);
  auto* d_counts = reinterpret_cast<std::uint32_t*>(d_counts_bytes.get());
  {
    TDMD_NVTX_RANGE("snap.bond_list.count_kernel");
    count_bonds_kernel<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                            d_types,
                                                            d_x,
                                                            d_y,
                                                            d_z,
                                                            d_cell_offsets,
                                                            d_cell_atoms,
                                                            d_rcut_sq_ab,
                                                            p,
                                                            d_counts);
    check_cuda("launch count_bonds_kernel", cudaGetLastError());
  }

  // ---------- 3. Exclusive scan (host) ----------
  std::vector<std::uint32_t> host_counts(n);
  std::uint64_t total_bonds = 0;
  std::vector<std::uint32_t> host_offsets(n + 1, 0);
  {
    TDMD_NVTX_RANGE("snap.bond_list.host_scan");
    check_cuda(
        "cudaMemcpyAsync D2H counts",
        cudaMemcpyAsync(host_counts.data(), d_counts, counts_bytes, cudaMemcpyDeviceToHost, s));
    check_cuda("cudaStreamSynchronize (counts D2H)", cudaStreamSynchronize(s));

    for (std::size_t i = 0; i < n; ++i) {
      host_offsets[i + 1] = host_offsets[i] + static_cast<std::uint32_t>(host_counts[i]);
    }
    total_bonds = host_offsets[n];
    bond_count = static_cast<std::size_t>(total_bonds);

    const std::size_t offsets_bytes = (n + 1) * sizeof(std::uint32_t);
    d_atom_bond_start_bytes = pool.allocate_device(offsets_bytes, stream);
    d_atom_bond_start = reinterpret_cast<std::uint32_t*>(d_atom_bond_start_bytes.get());
    check_cuda("cudaMemcpyAsync H2D offsets",
               cudaMemcpyAsync(d_atom_bond_start,
                               host_offsets.data(),
                               offsets_bytes,
                               cudaMemcpyHostToDevice,
                               s));
  }

  if (total_bonds == 0) {
    d_bond_i_bytes.reset();
    d_bond_j_bytes.reset();
    d_bond_type_i_bytes.reset();
    d_bond_type_j_bytes.reset();
    d_bond_dx_bytes.reset();
    d_bond_dy_bytes.reset();
    d_bond_dz_bytes.reset();
    d_bond_rsq_bytes.reset();
    d_bond_i = d_bond_j = nullptr;
    d_bond_type_i = d_bond_type_j = nullptr;
    d_bond_dx = d_bond_dy = d_bond_dz = d_bond_rsq = nullptr;
    check_cuda("cudaStreamSynchronize (empty bond list)", cudaStreamSynchronize(s));
    return;
  }

  // ---------- 4. Pass 2 — emit ----------
  {
    TDMD_NVTX_RANGE("snap.bond_list.emit_kernel");
    const std::size_t u32_bytes = total_bonds * sizeof(std::uint32_t);
    const std::size_t f64_bytes = total_bonds * sizeof(double);

    d_bond_i_bytes = pool.allocate_device(u32_bytes, stream);
    d_bond_j_bytes = pool.allocate_device(u32_bytes, stream);
    d_bond_type_i_bytes = pool.allocate_device(u32_bytes, stream);
    d_bond_type_j_bytes = pool.allocate_device(u32_bytes, stream);
    d_bond_dx_bytes = pool.allocate_device(f64_bytes, stream);
    d_bond_dy_bytes = pool.allocate_device(f64_bytes, stream);
    d_bond_dz_bytes = pool.allocate_device(f64_bytes, stream);
    d_bond_rsq_bytes = pool.allocate_device(f64_bytes, stream);

    d_bond_i = reinterpret_cast<std::uint32_t*>(d_bond_i_bytes.get());
    d_bond_j = reinterpret_cast<std::uint32_t*>(d_bond_j_bytes.get());
    d_bond_type_i = reinterpret_cast<std::uint32_t*>(d_bond_type_i_bytes.get());
    d_bond_type_j = reinterpret_cast<std::uint32_t*>(d_bond_type_j_bytes.get());
    d_bond_dx = reinterpret_cast<double*>(d_bond_dx_bytes.get());
    d_bond_dy = reinterpret_cast<double*>(d_bond_dy_bytes.get());
    d_bond_dz = reinterpret_cast<double*>(d_bond_dz_bytes.get());
    d_bond_rsq = reinterpret_cast<double*>(d_bond_rsq_bytes.get());

    emit_bonds_kernel<<<nblocks, kThreadsPerBlock, 0, s>>>(n32,
                                                           d_types,
                                                           d_x,
                                                           d_y,
                                                           d_z,
                                                           d_cell_offsets,
                                                           d_cell_atoms,
                                                           d_rcut_sq_ab,
                                                           d_atom_bond_start,
                                                           p,
                                                           d_bond_i,
                                                           d_bond_j,
                                                           d_bond_type_i,
                                                           d_bond_type_j,
                                                           d_bond_dx,
                                                           d_bond_dy,
                                                           d_bond_dz,
                                                           d_bond_rsq);
    check_cuda("launch emit_bonds_kernel", cudaGetLastError());
    check_cuda("cudaStreamSynchronize (bond list build)", cudaStreamSynchronize(s));
  }
}

SnapBondListHostSnapshot SnapBondListGpu::download(DeviceStream& stream) const {
  TDMD_NVTX_RANGE("snap.bond_list.download");
  SnapBondListHostSnapshot snap;
  if (!impl_ || impl_->atom_count == 0) {
    return snap;
  }
  cudaStream_t s = raw_stream(stream);
  const std::size_t n = impl_->atom_count;
  const std::size_t nb = impl_->bond_count;

  snap.atom_bond_start.assign(n + 1, 0);
  if (impl_->d_atom_bond_start != nullptr) {
    check_cuda("cudaMemcpyAsync D2H atom_bond_start",
               cudaMemcpyAsync(snap.atom_bond_start.data(),
                               impl_->d_atom_bond_start,
                               (n + 1) * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost,
                               s));
  }

  snap.bond_i.assign(nb, 0);
  snap.bond_j.assign(nb, 0);
  snap.bond_type_i.assign(nb, 0);
  snap.bond_type_j.assign(nb, 0);
  snap.bond_dx.assign(nb, 0.0);
  snap.bond_dy.assign(nb, 0.0);
  snap.bond_dz.assign(nb, 0.0);
  snap.bond_rsq.assign(nb, 0.0);
  if (nb > 0) {
    check_cuda("cudaMemcpyAsync D2H bond_i",
               cudaMemcpyAsync(snap.bond_i.data(),
                               impl_->d_bond_i,
                               nb * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("cudaMemcpyAsync D2H bond_j",
               cudaMemcpyAsync(snap.bond_j.data(),
                               impl_->d_bond_j,
                               nb * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("cudaMemcpyAsync D2H bond_type_i",
               cudaMemcpyAsync(snap.bond_type_i.data(),
                               impl_->d_bond_type_i,
                               nb * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("cudaMemcpyAsync D2H bond_type_j",
               cudaMemcpyAsync(snap.bond_type_j.data(),
                               impl_->d_bond_type_j,
                               nb * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("cudaMemcpyAsync D2H bond_dx",
               cudaMemcpyAsync(snap.bond_dx.data(),
                               impl_->d_bond_dx,
                               nb * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("cudaMemcpyAsync D2H bond_dy",
               cudaMemcpyAsync(snap.bond_dy.data(),
                               impl_->d_bond_dy,
                               nb * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("cudaMemcpyAsync D2H bond_dz",
               cudaMemcpyAsync(snap.bond_dz.data(),
                               impl_->d_bond_dz,
                               nb * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s));
    check_cuda("cudaMemcpyAsync D2H bond_rsq",
               cudaMemcpyAsync(snap.bond_rsq.data(),
                               impl_->d_bond_rsq,
                               nb * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               s));
  }
  check_cuda("cudaStreamSynchronize (bond list download)", cudaStreamSynchronize(s));
  return snap;
}

#else  // CPU-only build

struct SnapBondListGpu::Impl {};

SnapBondListGpu::SnapBondListGpu() : impl_(std::make_unique<Impl>()) {}
SnapBondListGpu::~SnapBondListGpu() = default;
SnapBondListGpu::SnapBondListGpu(SnapBondListGpu&&) noexcept = default;
SnapBondListGpu& SnapBondListGpu::operator=(SnapBondListGpu&&) noexcept = default;

std::size_t SnapBondListGpu::atom_count() const noexcept {
  return 0;
}
std::size_t SnapBondListGpu::bond_count() const noexcept {
  return 0;
}
std::uint64_t SnapBondListGpu::build_version() const noexcept {
  return 0;
}
SnapBondListGpuView SnapBondListGpu::view() const noexcept {
  return {};
}

void SnapBondListGpu::build(std::size_t /*n*/,
                            const std::uint32_t* /*host_types*/,
                            const double* /*host_x*/,
                            const double* /*host_y*/,
                            const double* /*host_z*/,
                            std::size_t /*ncells*/,
                            const std::uint32_t* /*host_cell_offsets*/,
                            const std::uint32_t* /*host_cell_atoms*/,
                            const double* /*host_rcut_sq_ab*/,
                            std::uint32_t /*n_species*/,
                            const BoxParams& /*params*/,
                            DevicePool& /*pool*/,
                            DeviceStream& /*stream*/) {
  throw std::runtime_error(
      "gpu::SnapBondListGpu::build: CPU-only build (TDMD_BUILD_CUDA=0); CUDA not linked");
}

SnapBondListHostSnapshot SnapBondListGpu::download(DeviceStream& /*stream*/) const {
  return {};
}

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
