#pragma once

// SPEC: docs/specs/gpu/SPEC.md §7.1 (neighbor-list build contract)
// Master spec: §5.2 neighbor build
// Module SPEC: docs/specs/neighbor/SPEC.md §2.1, §4 (CSR half-list)
// Exec pack: docs/development/m6_execution_pack.md T6.4, D-M6-7, D-M6-16
//
// NeighborListGpu — device-resident CSR half-list (j > i). Inputs are raw
// host pointers + a `BoxParams` POD (box extent, cell grid dims, periodic
// flags, cutoff, skin). The src/neighbor/ adapter
// (`GpuNeighborBuilder`) translates from domain types (AtomSoA, Box,
// CellGrid) into these primitives — this keeps gpu/ data-oblivious per
// module SPEC §1.1 and breaks a would-be cyclic dependency with neighbor/.
//
// On-device SoA layout (D-M6-16):
//   offsets: uint64[atom_count + 1]  prefix sum of per-atom neighbor counts
//   ids:     uint32[pair_count]      neighbor atom indices
//   r2:      double[pair_count]      squared minimum-image distances
//
// Algorithm (T6.4 MVP):
//   1. H2D copy of atom positions + cell CSR.
//   2. Kernel `count_neighbors` — one thread per atom i; mirrors the CPU
//      27-cell stencil loop.
//   3. Exclusive prefix scan (host-side; D2H/H2D of N uint32).
//   4. Kernel `emit_neighbors` — identical iteration order; writes the
//      CSR.
//
// Determinism: same iteration order as CPU NeighborList::build(), so the
// D-M6-7 bit-exact gate uses `std::memcmp` over the downloaded CSR.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tdmd::gpu {

// Box + cell-grid parameters needed by the kernels. POD so it can be
// passed by value as a kernel launch argument. Populated by the
// neighbor/ adapter from (Box, CellGrid, cutoff, skin).
struct BoxParams {
  double xlo = 0.0;
  double ylo = 0.0;
  double zlo = 0.0;
  double lx = 0.0;
  double ly = 0.0;
  double lz = 0.0;
  double cell_x = 0.0;
  double cell_y = 0.0;
  double cell_z = 0.0;
  std::uint32_t nx = 0;
  std::uint32_t ny = 0;
  std::uint32_t nz = 0;
  bool periodic_x = false;
  bool periodic_y = false;
  bool periodic_z = false;
  double cutoff = 0.0;
  double skin = 0.0;
};

// Zero-copy device-pointer view of the built CSR. Pointers are only
// valid while the owning NeighborListGpu is alive (DevicePool outlives
// both).
struct NeighborListGpuView {
  std::size_t atom_count = 0;
  std::size_t pair_count = 0;
  const std::uint64_t* d_offsets = nullptr;  // length atom_count + 1
  const std::uint32_t* d_ids = nullptr;      // length pair_count
  const double* d_r2 = nullptr;              // length pair_count
};

// Host-side mirror of the device CSR — shapes + dtypes match CPU
// `NeighborList` exactly so a byte-compare is trivial.
struct NeighborListHostSnapshot {
  std::vector<std::uint64_t> offsets;
  std::vector<std::uint32_t> ids;
  std::vector<double> r2;
};

class NeighborListGpu {
public:
  NeighborListGpu();
  ~NeighborListGpu();

  NeighborListGpu(const NeighborListGpu&) = delete;
  NeighborListGpu& operator=(const NeighborListGpu&) = delete;
  NeighborListGpu(NeighborListGpu&&) noexcept;
  NeighborListGpu& operator=(NeighborListGpu&&) noexcept;

  // Builds the device-resident half-list from raw host arrays:
  //   host_x/y/z     : N doubles each (atom positions in metal Å)
  //   host_cell_off  : ncells + 1 uint32_t (CSR prefix sum)
  //   host_cell_atoms: N uint32_t (atom indices binned into cells)
  //   params         : box + cell grid scalars (nx/ny/nz, cell sizes,
  //                    periodic flags, cutoff, skin)
  //
  // Throws std::runtime_error on CPU-only build or on CUDA failure.
  void build(std::size_t n,
             const double* host_x,
             const double* host_y,
             const double* host_z,
             std::size_t ncells,
             const std::uint32_t* host_cell_offsets,
             const std::uint32_t* host_cell_atoms,
             const BoxParams& params,
             DevicePool& pool,
             DeviceStream& stream);

  // D2H copy of the CSR. Synchronises on the given stream. Intended for
  // verification (tests + differential) — not on hot-path iteration code.
  [[nodiscard]] NeighborListHostSnapshot download(DeviceStream& stream) const;

  [[nodiscard]] NeighborListGpuView view() const noexcept;
  [[nodiscard]] std::size_t atom_count() const noexcept;
  [[nodiscard]] std::size_t pair_count() const noexcept;
  [[nodiscard]] std::uint64_t build_version() const noexcept;
  [[nodiscard]] double cutoff() const noexcept;
  [[nodiscard]] double skin() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tdmd::gpu
