#pragma once

// SPEC: docs/specs/gpu/SPEC.md §7.3 (SNAP GPU contract — T8.6 authors it),
//       §6.3 (D-M6-7 bit-exact gate, extended to SNAP as D-M8-13),
//       §8.1 (Reference FP64-only), §9 (NVTX), §1.1 (data-oblivious gpu/)
// Master spec: §14 M8 SNAP proof-of-value; §D.16 __restrict__
// Module SPEC: docs/specs/potentials/SPEC.md §6 (SNAP module contract)
// Exec pack: docs/development/m8_execution_pack.md T8.6a (scaffolding) /
//            T8.6b (full kernel body)
// Decisions: D-M6-17 (PIMPL firewall), D-M8-7 (CPU byte-exact gate landed
//            T8.5 — GPU reference path MUST match to ≤ 1e-12 rel at T8.7)
//
// SnapGpu — device-resident SNAP force kernel. T8.6a lands the class skeleton
// + PIMPL + CPU/CUDA build guards; `compute()` throws a T8.6b sentinel so
// downstream consumers (SimulationEngine, SnapGpuAdapter, tests) can link
// against the final public ABI without waiting for the 2000-line kernel port.
//
// Inputs are raw host primitives (positions, atom types, cell CSR, flattened
// SNAP parameter tables) and a BoxParams POD; the adapter in src/potentials/
// (`SnapGpuAdapter`) translates from domain types (`AtomSoA`, `Box`,
// `CellGrid`, `SnapData`) into these primitives. This keeps gpu/ data-
// oblivious per module SPEC §1.1.
//
// Algorithm sketch (full body deferred to T8.6b):
//   1. snap_ui_kernel   — Pass 1: Wigner U-functions accumulated to ulisttot
//                          over full-list neighbours (per-atom).
//   2. snap_yi_kernel   — Pass 2: Z list contracted with per-species β → Y.
//   3. snap_deidrj_kernel — Pass 3: per-neighbour dE/dr via CG chain rule,
//                            atomicAdd to forces + virial.
// Host-side Kahan reduction for PE + virial totals (D-M6-15).

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tdmd::gpu {

// Flat SNAP parameter table packing. SNAP's per-species state (radius,
// weight, β coefficients) plus the engine's twojmax-derived index tables are
// uploaded once per adapter lifetime and reused across compute() calls
// (mirrors EAM spline-cache pattern — T6.9a). Pointer members borrow
// adapter-owned buffers for the duration of `SnapGpu::compute()`; no
// ownership transfer.
//
// The member layout is intentionally minimal in T8.6a — the kernel body
// (T8.6b) will extend it with index-table pointers (idxcg, idxu, idxz, idxb,
// Clebsch-Gordan coefficients, root-pq tables). T8.6a consumers only use
// the scalar hyperparameters + per-species arrays.
struct SnapTablesHost {
  // Hyperparameters (SnapData::params mirror).
  int twojmax = 0;
  double rcutfac = 0.0;
  double rfac0 = 0.99363;
  double rmin0 = 0.0;
  int switchflag = 1;
  int bzeroflag = 1;
  int bnormflag = 0;
  int wselfallflag = 0;

  // Derived.
  int k_max = 0;  // number of bispectrum components; matches snap_k_max()
  int idxb_max = 0;
  int idxu_max = 0;
  int idxz_max = 0;

  // Species count + per-species arrays (host-visible, contiguous). All arrays
  // are n_species long unless noted; β is packed species-major with stride
  // `beta_stride` == `k_max + 1` (linear SNAP — M8 scope; quadratic deferred).
  std::size_t n_species = 0;
  std::size_t beta_stride = 0;  // == k_max + 1

  const double* radius_elem = nullptr;        // n_species doubles
  const double* weight_elem = nullptr;        // n_species doubles
  const double* beta_coefficients = nullptr;  // n_species × beta_stride doubles

  // Pairwise squared cutoffs, row-major n_species × n_species (mirrors
  // SnapData::rcut_sq_ab layout).
  const double* rcut_sq_ab = nullptr;
};

// Scalar totals returned from `compute()`. Voigt virial ordering matches CPU
// `ForceResult`: (xx, yy, zz, xy, xz, yz).
struct SnapGpuResult {
  double potential_energy = 0.0;
  double virial[6] = {};
};

class SnapGpu {
public:
  SnapGpu();
  ~SnapGpu();

  SnapGpu(const SnapGpu&) = delete;
  SnapGpu& operator=(const SnapGpu&) = delete;
  SnapGpu(SnapGpu&&) noexcept;
  SnapGpu& operator=(SnapGpu&&) noexcept;

  // One-shot SNAP compute: uploads atoms + tables + cell CSR, runs the three
  // SNAP kernels on `stream`, D2Hs per-atom forces + PE/virial accumulators,
  // and does the final Kahan reductions on host.
  //
  // Inputs:
  //   n                  : atom count.
  //   host_types         : n uint32_t species ids, [0, n_species).
  //   host_x/y/z         : n doubles each (metal Å).
  //   ncells             : CellGrid::cell_count().
  //   host_cell_offsets  : ncells + 1 uint32_t CSR prefix sum.
  //   host_cell_atoms    : n uint32_t atom indices binned into cells.
  //   params             : box + cell grid scalars; `params.cutoff` MUST
  //                        equal max pairwise SNAP cutoff.
  //   tables             : SNAP hyperparameters + per-species arrays.
  //
  // Outputs (accumulated — caller zeros first, matching CPU
  // `Potential::compute` contract):
  //   host_fx_out/fy_out/fz_out : n doubles each, additively updated.
  //
  // Returns potential energy + virial Voigt tensor.
  //
  // T8.6a status: throws std::logic_error with sentinel
  //   "SnapGpu::compute — T8.6b kernel body not landed"
  // so every call-site can link and test the error path. T8.6b replaces the
  // body with the actual three-pass kernel chain.
  //
  // Throws std::runtime_error on CPU-only build (TDMD_BUILD_CUDA=0).
  SnapGpuResult compute(std::size_t n,
                        const std::uint32_t* host_types,
                        const double* host_x,
                        const double* host_y,
                        const double* host_z,
                        std::size_t ncells,
                        const std::uint32_t* host_cell_offsets,
                        const std::uint32_t* host_cell_atoms,
                        const BoxParams& params,
                        const SnapTablesHost& tables,
                        double* host_fx_out,
                        double* host_fy_out,
                        double* host_fz_out,
                        DevicePool& pool,
                        DeviceStream& stream);

  // Monotone counter incremented on every successful compute() (after T8.6b
  // lands). In T8.6a this counter stays at 0 since compute() always throws
  // before reaching the increment point.
  [[nodiscard]] std::uint64_t compute_version() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tdmd::gpu
