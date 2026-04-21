#pragma once

// SPEC: docs/specs/gpu/SPEC.md §6.1 (no atomicAdd(double); reduce-then-scatter
//       is the permitted alternative for FP64 reductions),
//       §7.5 (T8.6c-v5 per-bond dispatch), §9 (NVTX ranges).
// Module SPEC: docs/specs/potentials/SPEC.md §6 (SnapPotential).
// Pre-impl:  docs/development/t8.6c-v5_pre_impl.md (Stage 1).
//
// SnapBondListGpu — per-neighbour SNAP bond list (full list, not j>i half).
// Each bond is a (i, j) pair that passes SNAP's per-pair cutoff filter
// `rsq < cutsq_ij && rsq > 1e-20 && j != i`. Emission order is byte-identical
// to `snap_ui_kernel`'s 3×3×3 cell-stencil walk:
//
//   for i in [0, N):
//     for dz in {-1, 0, +1}, dy in {-1, 0, +1}, dx in {-1, 0, +1}:
//       cj = wrap(ci + (dx, dy, dz))
//       for k in cell_atoms[cell_offsets[cj] .. cell_offsets[cj+1]):
//         j = cell_atoms[k]
//         if j == i:                       continue
//         Δr = minimum_image(x[j] - x[i])
//         r² = |Δr|²
//         if r² >= cutsq[type_i, type_j]:  continue
//         if r² <= 1e-20:                  continue
//         emit (i, j, Δr.x, Δr.y, Δr.z, r², type_i, type_j)
//
// This is a strict superset of `NeighborListGpu`'s half-list:
//   * SNAP uses per-pair cutoffs (rcut_sq_ab matrix), not a single cutoff.
//   * SNAP needs the full list for Newton-3 double-eval in compute_deidrj.
//   * SNAP filters co-located atoms with `rsq > 1e-20` (guard copied from
//     CPU SnaEngine; avoids the `tan(0)` singularity in compute_uarray).
//
// Used downstream by Stage 2 (`snap_ui_bond_kernel` + `snap_ui_gather_kernel`)
// and Stage 3 (`snap_deidrj_bond_kernel` + `snap_force_gather_kernel`). The
// per-atom gather kernels sum over each atom's bond range in emission order
// via shared-memory Kahan — that is how the ≤ 1e-12 rel T8.7 byte-exact gate
// is preserved despite the atomic-free per-bond dispatch (gpu/SPEC §6.1).

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // reuses BoxParams POD
#include "tdmd/gpu/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tdmd::gpu {

// Zero-copy device-pointer view of the built bond list. Pointers are valid
// only while the owning `SnapBondListGpu` is alive (the DevicePool outlives
// both).
struct SnapBondListGpuView {
  std::size_t atom_count = 0;
  std::size_t bond_count = 0;

  // CSR offsets mapping atom i → bond range [start[i], start[i+1]).
  // Length = atom_count + 1.
  const std::uint32_t* d_atom_bond_start = nullptr;

  // Per-bond SoA. All arrays have length `bond_count`.
  const std::uint32_t* d_bond_i = nullptr;
  const std::uint32_t* d_bond_j = nullptr;
  const std::uint32_t* d_bond_type_i = nullptr;
  const std::uint32_t* d_bond_type_j = nullptr;
  const double* d_bond_dx = nullptr;
  const double* d_bond_dy = nullptr;
  const double* d_bond_dz = nullptr;
  const double* d_bond_rsq = nullptr;
};

// Host-side mirror of the device arrays; for validation tests + differential.
struct SnapBondListHostSnapshot {
  std::vector<std::uint32_t> atom_bond_start;  // atom_count + 1
  std::vector<std::uint32_t> bond_i;
  std::vector<std::uint32_t> bond_j;
  std::vector<std::uint32_t> bond_type_i;
  std::vector<std::uint32_t> bond_type_j;
  std::vector<double> bond_dx;
  std::vector<double> bond_dy;
  std::vector<double> bond_dz;
  std::vector<double> bond_rsq;
};

class SnapBondListGpu {
public:
  SnapBondListGpu();
  ~SnapBondListGpu();

  SnapBondListGpu(const SnapBondListGpu&) = delete;
  SnapBondListGpu& operator=(const SnapBondListGpu&) = delete;
  SnapBondListGpu(SnapBondListGpu&&) noexcept;
  SnapBondListGpu& operator=(SnapBondListGpu&&) noexcept;

  // Build the device-resident bond list from raw host arrays.
  //
  //   n                : number of atoms.
  //   host_types       : n uint32_t (element indices into rcut_sq_ab).
  //   host_x/y/z       : n doubles each (positions inside the box, Å).
  //   ncells           : nx * ny * nz.
  //   host_cell_off    : ncells + 1 uint32_t (CSR prefix sum of cell bins).
  //   host_cell_atoms  : n uint32_t (atom indices binned into cells).
  //   host_rcut_sq_ab  : n_species² doubles (row-major, SNAP per-pair cutsq).
  //   n_species        : size of rcut_sq_ab matrix side.
  //   params           : BoxParams (xlo/ylo/zlo, lx/ly/lz, cell_*, nx/ny/nz,
  //                      periodic flags). `cutoff` and `skin` fields of
  //                      BoxParams are ignored — SNAP uses per-pair cutsq.
  //
  // Throws `std::runtime_error` on CPU-only build or on CUDA failure.
  void build(std::size_t n,
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
             DeviceStream& stream);

  // T8.6c-v5 Stage 2: variant that consumes already-resident device arrays (used
  // from SnapGpu::compute() to avoid redundant H2D, since the same atom data is
  // already on device from SnapGpu's H2D step). `ncells` is only used to size
  // the `d_cell_offsets` array for bounds — device pointer lifetime is the
  // caller's responsibility. Emission order is identical to `build()`.
  //
  // Throws `std::runtime_error` on CPU-only build or on CUDA failure.
  void build_from_device(std::size_t n,
                         const std::uint32_t* d_types,
                         const double* d_x,
                         const double* d_y,
                         const double* d_z,
                         std::size_t ncells,
                         const std::uint32_t* d_cell_offsets,
                         const std::uint32_t* d_cell_atoms,
                         const double* d_rcut_sq_ab,
                         std::uint32_t n_species,
                         const BoxParams& params,
                         DevicePool& pool,
                         DeviceStream& stream);

  // D2H copy of the full bond list. Synchronises on the given stream.
  // Intended for tests + differential validation — not for hot-path code.
  [[nodiscard]] SnapBondListHostSnapshot download(DeviceStream& stream) const;

  [[nodiscard]] SnapBondListGpuView view() const noexcept;
  [[nodiscard]] std::size_t atom_count() const noexcept;
  [[nodiscard]] std::size_t bond_count() const noexcept;
  [[nodiscard]] std::uint64_t build_version() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tdmd::gpu
