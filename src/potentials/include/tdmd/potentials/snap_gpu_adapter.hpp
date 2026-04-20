#pragma once

// SPEC: docs/specs/gpu/SPEC.md §1.1 (data-oblivious gpu/), §7.3 (SNAP GPU
//       contract); docs/specs/potentials/SPEC.md §6 (SNAP module contract)
// Exec pack: docs/development/m8_execution_pack.md T8.6a
//
// `SnapGpuAdapter` — thin potentials-layer facade for `tdmd::gpu::SnapGpu`.
// Bridges domain types (`AtomSoA`, `Box`, `CellGrid`, `SnapData`) into the
// raw-primitives API the gpu/ layer exposes. Mirrors `EamAlloyGpuAdapter`
// shape so `SimulationEngine::recompute_forces` can dispatch SNAP the same
// way EAM is dispatched today.
//
// Ownership:
//   * `SnapData` is borrowed (const ref at construction); caller must keep
//     it alive for the adapter's lifetime. Per-species coefficient arrays
//     (β, radius, weight) are flattened once at construction into adapter-
//     owned contiguous buffers ready for H2D transfer.
//   * DevicePool + DeviceStream are borrowed at every `compute()` call —
//     adapter does not hold pool/stream pointers.
//
// Force-contract matches CPU `SnapPotential::compute`: forces are additively
// accumulated into the caller's AtomSoA (caller zeros beforehand per
// Potential::compute contract), and the returned `ForceResult` carries PE +
// virial Voigt tensor.
//
// T8.6a status: the scaffolding compiles + links on all flavors. `compute()`
// propagates the SnapGpu "T8.6b not landed" sentinel so callers can see the
// full error path end-to-end. Full bit-exact force computation lands T8.6b,
// with the D-M8-13 ≤ 1e-12 rel acceptance gate exercised by T8.7.

#include "tdmd/potentials/potential.hpp"
#include "tdmd/potentials/snap_file.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tdmd {
struct AtomSoA;
struct Box;
class CellGrid;
}  // namespace tdmd

namespace tdmd::gpu {
class DevicePool;
class DeviceStream;
class SnapGpu;
class SnapGpuMixed;
// Compile-time dispatch between Fp64 reference and T8.9 Phase A (Philosophy B
// narrow-FP32 pair-math) SNAP backends. Only the `MixedFastSnapOnlyBuild`
// flavor routes SNAP through `SnapGpuMixed`; all other flavors keep SNAP on
// the FP64 `SnapGpu` reference (D-M8-4: MixedFastBuild keeps SNAP FP64 because
// SNAP's dominant cost is a small ML-fit of potentials and per-rank SNAP
// throughput is dominated by the bispectrum, which is scientifically identical
// in both paths; MixedFastSnapOnlyBuild is the opt-in throughput path).
#ifdef TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY
using SnapGpuActive = SnapGpuMixed;
#else
using SnapGpuActive = SnapGpu;
#endif
}  // namespace tdmd::gpu

namespace tdmd::potentials {

class SnapGpuAdapter {
public:
  // Flattens `data`'s per-species coefficient arrays (radius_elem,
  // weight_elem, β) into contiguous host-visible buffers. Validates that
  // M8-scope flags (chemflag, quadraticflag, switchinnerflag) are off —
  // consistent with SnapPotential's CPU-side validation at T8.4a/b.
  //
  // `data` must outlive the adapter. Throws std::invalid_argument on M8-scope
  // violations or any structural mismatch.
  explicit SnapGpuAdapter(const SnapData& data);
  ~SnapGpuAdapter();

  SnapGpuAdapter(const SnapGpuAdapter&) = delete;
  SnapGpuAdapter& operator=(const SnapGpuAdapter&) = delete;
  SnapGpuAdapter(SnapGpuAdapter&&) noexcept;
  SnapGpuAdapter& operator=(SnapGpuAdapter&&) noexcept;

  // Runs the SNAP GPU compute (T8.6b will decompose into three kernels).
  // `grid.bin(atoms)` must have been called before entry so that
  // `grid.cell_offsets()` / `cell_atoms()` are fresh. `atoms.fx/fy/fz` are
  // additively updated; caller zeros them first (Potential::compute contract).
  //
  // Returns PE + virial Voigt tensor.
  //
  // T8.6a: propagates the SnapGpu "T8.6b kernel body not landed" logic error
  // so the preflight → SimulationEngine → adapter → gpu chain can be end-to-
  // end tested without the kernel body. Throws std::runtime_error on CPU-only
  // build or CUDA failure.
  [[nodiscard]] ForceResult compute(AtomSoA& atoms,
                                    const Box& box,
                                    const CellGrid& grid,
                                    tdmd::gpu::DevicePool& pool,
                                    tdmd::gpu::DeviceStream& stream);

  // Monotone counter forwarded from the underlying `SnapGpu`. Tests use this
  // to verify repeat-call behaviour once T8.6b lands. In T8.6a it stays at 0
  // (compute() throws before incrementing).
  [[nodiscard]] std::uint64_t compute_version() const noexcept;

private:
  const SnapData* data_;

  // Flattened per-species buffers. Layout:
  //   radius_elem_flat_   — n_species doubles (index by species id)
  //   weight_elem_flat_   — n_species doubles
  //   beta_flat_          — n_species × (k_max + 1) doubles, species-major.
  //                         β[species=α] starts at beta_flat_.data() + α * (k_max + 1).
  std::vector<double> radius_elem_flat_;
  std::vector<double> weight_elem_flat_;
  std::vector<double> beta_flat_;

  std::unique_ptr<tdmd::gpu::SnapGpuActive> gpu_;
};

}  // namespace tdmd::potentials
