#pragma once

// SPEC: docs/specs/gpu/SPEC.md §1.1 (data-oblivious gpu/), §7.2 (EAM contract);
//       docs/specs/potentials/SPEC.md §4.1–§4.4 (EAM/alloy + TabulatedFunction
//       layout)
// Exec pack: docs/development/m6_execution_pack.md T6.5
//
// `EamAlloyGpuAdapter` — thin potentials-layer facade for
// `tdmd::gpu::EamAlloyGpu`. Bridges domain types (`AtomSoA`, `Box`,
// `CellGrid`, `EamAlloyData`) into the raw-primitives API the gpu/ layer
// exposes (per D-M6-17 gpu/ stays data-oblivious).
//
// Ownership:
//   * `EamAlloyData` is borrowed (const ref at construction); caller must
//     keep it alive for the adapter's lifetime. The coefficient arrays are
//     flattened once at construction into adapter-owned contiguous buffers.
//   * DevicePool + DeviceStream are borrowed at every `compute()` call —
//     adapter does not hold pool/stream pointers.
//
// Force-contract matches CPU `EamAlloyPotential::compute`: forces are
// additively accumulated into the caller's AtomSoA (caller must zero
// beforehand per Potential::compute contract), and the returned
// `ForceResult` carries PE + virial Voigt tensor.

#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/potentials/potential.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace tdmd {
class AtomSoA;
class Box;
class CellGrid;
}  // namespace tdmd

namespace tdmd::gpu {
class DevicePool;
class DeviceStream;
class EamAlloyGpu;
}  // namespace tdmd::gpu

namespace tdmd::potentials {

class EamAlloyGpuAdapter {
public:
  // Flattens `data`'s TabulatedFunction coefficient arrays into contiguous
  // host-visible buffers for H2D transfer. `data` must outlive the adapter.
  explicit EamAlloyGpuAdapter(const EamAlloyData& data);
  ~EamAlloyGpuAdapter();

  EamAlloyGpuAdapter(const EamAlloyGpuAdapter&) = delete;
  EamAlloyGpuAdapter& operator=(const EamAlloyGpuAdapter&) = delete;
  EamAlloyGpuAdapter(EamAlloyGpuAdapter&&) noexcept;
  EamAlloyGpuAdapter& operator=(EamAlloyGpuAdapter&&) noexcept;

  // Runs the three EAM GPU kernels + host reductions. `grid.bin(atoms)` must
  // have been called before entry so that `grid.cell_offsets()` /
  // `cell_atoms()` are fresh. `atoms.fx/fy/fz` are additively updated;
  // caller zeros them first (Potential::compute contract).
  //
  // Returns PE + virial Voigt tensor. Throws `std::runtime_error` on
  // CPU-only build or CUDA failure.
  [[nodiscard]] ForceResult compute(AtomSoA& atoms,
                                    const Box& box,
                                    const CellGrid& grid,
                                    tdmd::gpu::DevicePool& pool,
                                    tdmd::gpu::DeviceStream& stream);

  // Monotone counter forwarded from the underlying `EamAlloyGpu`. Tests use
  // this to verify repeat-call behaviour.
  [[nodiscard]] std::uint64_t compute_version() const noexcept;

private:
  const EamAlloyData* data_;

  // Flattened coefficient buffers — 7 doubles/cell, species-major (F, rho)
  // or pair-major (z2r). Ordering matches the pair_index / species-id
  // conventions documented in `EamAlloyTablesHost`.
  std::vector<double> F_coeffs_flat_;
  std::vector<double> rho_coeffs_flat_;
  std::vector<double> z2r_coeffs_flat_;

  std::unique_ptr<tdmd::gpu::EamAlloyGpu> gpu_;
};

}  // namespace tdmd::potentials
