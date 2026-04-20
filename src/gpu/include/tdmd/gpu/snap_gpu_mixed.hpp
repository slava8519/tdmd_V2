#pragma once

// SPEC: docs/specs/gpu/SPEC.md §8 (mixed-precision policy), §7.5 (SNAP GPU
//       contract reused), §8.3 (D-M8-8 dense-cutoff thresholds)
// Master spec: §D.1 Philosophy B (FP32 math + FP64 accumulators),
//              §D.11 MixedFastSnapOnlyBuild flavor exception
// Exec pack: docs/development/m8_execution_pack.md T8.9
// Pre-impl: docs/development/t8.9_pre_impl.md
// Decisions: D-M8-4 (SnapOnly flavor uses FP32 SNAP / FP64 EAM), D-M8-8
//            (differential thresholds: per-atom force rel ≤ 1e-5,
//            total PE rel ≤ 1e-7, virial rel-to-max ≤ 5e-6 vs Fp64Ref GPU)
//
// SnapGpuMixed — Philosophy B FP32-pair-math variant of SnapGpu. Public
// interface is byte-identical to `SnapGpu` (same inputs, same SnapGpuResult,
// same additive-accumulation contract) so the `SnapGpuAdapter` can dispatch
// to either at compile time via `TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY`.
//
// FP32 sites (see snap_gpu_mixed.cu for full inventory; pair-math only,
// T8.9 Phase A):
//   - r² computation + cutoff filter
//   - r = sqrtf(rsq_f), 1/r via FP32 SFU reciprocal
//   - theta0, cos/sin (via __cosf/__sinf), z0, dz0dr
//   - sfac, dsfac
//
// FP64 sites (unchanged vs SnapGpu):
//   - atom positions / forces in device memory (FP64 SoA)
//   - per-atom accumulators (fx/fy/fz, pe, virial)
//   - rootpq, cglist, beta coefficient tables (device FP64 memory)
//   - ulist_r/i, ulisttot, dulist_r/i, zlist, ylist, blist (all FP64)
//   - U-recurrence (compute_uarray_device, compute_duarray_device) kept FP64
//   - dE/dr contraction (compute_deidrj_device) kept FP64
//   - Host-side Kahan reductions for PE + virial
//
// T8.9 Phase A acceptance: per-atom force rel ≤ 1e-5, PE rel ≤ 1e-7, virial
// rel-to-max ≤ 5e-6 vs Fp64Reference GPU on T6 tungsten 2000-atom rattled
// BCC fixture (dense-cutoff regime, D-M8-8).
//
// Phase B (full FP32 bispectrum) is deferred: the exec pack's aspirational
// "bispectrum basis in FP32" requires evaluation of the 8-level VMK chain
// in FP32 through dot products with length-O(165) arrays, which risks
// exceeding the 1e-5 rel force ceiling. If T8.10/T8.11 shows Phase A's
// ~2-8% throughput is insufficient, Phase B lands as a follow-up delta with
// its own SPEC procedure.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/snap_gpu.hpp"           // SnapTablesHost + SnapGpuResult
#include "tdmd/gpu/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tdmd::gpu {

// Inputs + outputs identical to SnapGpu — reuses SnapTablesHost +
// SnapGpuResult. Rationale: the adapter in src/potentials/ treats both
// variants interchangeably via compile-time dispatch on
// TDMD_FLAVOR_MIXED_FAST_SNAP_ONLY.
class SnapGpuMixed {
public:
  SnapGpuMixed();
  ~SnapGpuMixed();

  SnapGpuMixed(const SnapGpuMixed&) = delete;
  SnapGpuMixed& operator=(const SnapGpuMixed&) = delete;
  SnapGpuMixed(SnapGpuMixed&&) noexcept;
  SnapGpuMixed& operator=(SnapGpuMixed&&) noexcept;

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

  [[nodiscard]] std::uint64_t compute_version() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tdmd::gpu
