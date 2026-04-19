#pragma once

// SPEC: docs/specs/gpu/SPEC.md §7.2 (EAM force contract), §6.3 (D-M6-7 gate),
//       §8.1 (Reference FP64-only), §1.1 (data-oblivious gpu/)
// Master spec: §5.3 (EAM force math)
// Module SPEC: docs/specs/potentials/SPEC.md §4.1–§4.4 (EAM/alloy form,
//              TabulatedFunction Horner contract)
// Exec pack: docs/development/m6_execution_pack.md T6.5
// Decisions: D-M6-4 (three M6 kernels — NL/EAM/VV), D-M6-7 (bit-exact gate),
//            D-M6-15 (canonical reduction), D-M6-17 (PIMPL firewall)
//
// EamAlloyGpu — device-resident EAM/alloy force kernel. Inputs are raw host
// primitives (positions + atom types + cell CSR + flattened Hermite-cubic
// spline coefficients) and a BoxParams POD; the adapter in src/potentials/
// (`EamAlloyGpuAdapter`) translates from domain types (`AtomSoA`, `Box`,
// `CellGrid`, `EamAlloyData`) into these primitives. This keeps gpu/
// data-oblivious per module SPEC §1.1.
//
// Algorithm (three kernels per compute() call; same iteration order on all
// three):
//   1. density_kernel — thread per atom i walks 27-cell stencil (no j<=i
//      filter → full-list-per-atom), accumulates ρ[i] = Σⱼ rho_r[type_j](r).
//   2. embedding_kernel — thread per atom i computes F(ρ[i]) and F'(ρ[i]).
//   3. force_kernel — thread per atom i re-walks 27-cell stencil, computes
//      dE/dr per pair (i, j) and accumulates force components + per-atom PE
//      and virial contribution.
//
// Reduction: per-atom PE + virial arrays are D2H-copied and summed on host
// with Kahan compensation. Pair contributions are counted twice in full-list
// iteration so the host halves them; embedding contributions are per-atom.
//
// Determinism: full-list cell-sweep order is deterministic across runs on
// the same hardware; inter-run bit-exactness not required by gpu/SPEC §7.2
// (1e-12 rel gate absorbs FP64 reduction-order drift).

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tdmd::gpu {

// Flat table packing matching CPU `TabulatedFunction`'s 7-doubles-per-cell
// layout (potentials/SPEC §4.4). All tables share their respective grid:
//   F-grid — (F_x0, F_dx, nrho) shared across F_coeffs slices;
//   r-grid — (r_x0, r_dx, nr)   shared across rho_coeffs + z2r_coeffs slices.
// Pair indexing for z2r follows `EamAlloyData::pair_index(α, β)` — symmetric,
// lower-triangular packing. Pointer members borrow host memory for the
// duration of `EamAlloyGpu::compute()`; no ownership transfer.
struct EamAlloyTablesHost {
  std::size_t n_species = 0;
  std::size_t nrho = 0;
  std::size_t nr = 0;
  std::size_t npairs = 0;  // == n_species * (n_species + 1) / 2

  double F_x0 = 0.0;
  double F_dx = 0.0;
  double r_x0 = 0.0;
  double r_dx = 0.0;

  double cutoff = 0.0;

  // F_coeffs: n_species × nrho × 7 doubles. Species-major: table for species
  // α starts at F_coeffs + α * nrho * 7.
  const double* F_coeffs = nullptr;

  // rho_coeffs: n_species × nr × 7 doubles. Species α's table at
  // rho_coeffs + α * nr * 7.
  const double* rho_coeffs = nullptr;

  // z2r_coeffs: npairs × nr × 7 doubles. Pair (α, β) via pair_index at
  // z2r_coeffs + pair_index(α, β) * nr * 7.
  const double* z2r_coeffs = nullptr;
};

// Scalar totals returned from `compute()` — per-atom forces are written into
// caller-supplied host arrays. Voigt virial ordering matches CPU
// `ForceResult`: (xx, yy, zz, xy, xz, yz).
struct EamAlloyGpuResult {
  double potential_energy = 0.0;
  double virial[6] = {};
};

class EamAlloyGpu {
public:
  EamAlloyGpu();
  ~EamAlloyGpu();

  EamAlloyGpu(const EamAlloyGpu&) = delete;
  EamAlloyGpu& operator=(const EamAlloyGpu&) = delete;
  EamAlloyGpu(EamAlloyGpu&&) noexcept;
  EamAlloyGpu& operator=(EamAlloyGpu&&) noexcept;

  // One-shot EAM compute: uploads atoms + tables + cell CSR, runs the three
  // kernels on `stream`, D2Hs per-atom forces + PE/virial accumulators, and
  // does the final Kahan reductions on host.
  //
  // Inputs:
  //   n                  : atom count.
  //   host_types         : n uint32_t species ids, [0, n_species).
  //   host_x/y/z         : n doubles each (metal Å).
  //   ncells             : CellGrid::cell_count().
  //   host_cell_offsets  : ncells + 1 uint32_t CSR prefix sum.
  //   host_cell_atoms    : n uint32_t atom indices binned into cells.
  //   params             : box + cell grid scalars; `params.cutoff` MUST equal
  //                        tables.cutoff (adapter enforces).
  //   tables             : flattened EAM tables + grid scalars.
  //
  // Outputs (accumulated — caller must zero first, matching CPU
  // `Potential::compute` contract):
  //   host_fx_out/fy_out/fz_out : n doubles each, additively updated.
  //
  // Returns potential energy + virial Voigt tensor.
  //
  // Throws std::runtime_error on CPU-only build or CUDA failure.
  EamAlloyGpuResult compute(std::size_t n,
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
                            DeviceStream& stream);

  // Monotone counter incremented on every successful compute(). Used by tests
  // to verify repeat-call behaviour without reaching into Impl.
  [[nodiscard]] std::uint64_t compute_version() const noexcept;

  // Counter of actual spline H2D uploads (T6.9a caching). Invariant: after N
  // back-to-back compute() calls with the same host spline table pointers,
  // this equals 1 (the first call). Ship target: zero wasteful re-uploads on
  // steady-state hot loops.
  [[nodiscard]] std::uint64_t splines_upload_count() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tdmd::gpu
