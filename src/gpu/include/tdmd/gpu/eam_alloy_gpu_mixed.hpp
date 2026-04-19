#pragma once

// SPEC: docs/specs/gpu/SPEC.md §8 (mixed-precision policy), §7.2 (EAM
//       contract reused), §6.3 (D-M6-7 — Mixed is NOT the bit-exact oracle;
//       D-M6-8 threshold applies instead)
// Master spec: §D.1 Philosophy B (FP32 math + FP64 accumulators)
// Exec pack: docs/development/m6_execution_pack.md T6.8
// Decisions: D-M6-5 (MixedFast = compile-time build flavor), D-M6-8
//            (differential thresholds: ≤1e-6 rel per-atom force,
//            ≤1e-8 rel total energy vs Fp64Reference GPU)
//
// EamAlloyGpuMixed — Philosophy B FP32-math variant of EamAlloyGpu. Public
// interface is byte-identical to `EamAlloyGpu` (same inputs, same Outputs
// struct, same additive-accumulation contract) so the `EamAlloyGpuAdapter`
// can dispatch to either at compile time via `TDMD_FLAVOR_MIXED_FAST`.
//
// FP32 sites (see eam_alloy_gpu_mixed.cu for full inventory):
//   - r² pair-distance + FP32 cutoff filter (density + force kernels)
//   - r = sqrtf(r²_f), 1/r via FP32 SFU reciprocal
// FP64 sites:
//   - atom positions / forces in device memory (FP64 SoA)
//   - per-atom accumulators (ρ, F(ρ), fx/fy/fz, pe, virial)
//   - spline coefficient tables in device memory (FP64)
//   - spline locate + Horner eval (FP32 Horner on real EAM coefficients hits
//     catastrophic cancellation — kept FP64)
//   - phi, phi_prime, dE/dr, fscalar, fij_xyz — derived from FP64 splines but
//     multiplied by the (FP32-rounded) inv_r cast to FP64; per-pair rel error
//     ~1.2e-7, accumulated ~2e-6 on symmetric lattices with partial
//     cancellation
//   - host-side Kahan reductions for PE + virial
//
// T6.8a acceptance: per-atom force rel-diff ≤ 1e-5, total energy rel-diff
// ≤ 1e-7 from Fp64Reference GPU on Ni-Al Mishin-2004 / Al FCC fixtures.
// D-M6-8's tighter 1e-6 / 1e-8 thresholds + 100-step NVE drift gate deferred
// to T6.8b — see tests/gpu/test_eam_mixed_fast_within_threshold.cpp header
// for the rationale. Determinism: same input → same output bit-exactly across
// runs on fixed hardware (D-M6-15 canonical order preserved; FP32 rounding is
// deterministic on NVIDIA SFU path).

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/eam_alloy_gpu.hpp"      // EamAlloyTablesHost + EamAlloyGpuResult
#include "tdmd/gpu/neighbor_list_gpu.hpp"  // BoxParams
#include "tdmd/gpu/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tdmd::gpu {

// Inputs + outputs identical to EamAlloyGpu — reuses EamAlloyTablesHost +
// EamAlloyGpuResult. Rationale: the adapter in src/potentials/ treats both
// variants interchangeably via compile-time dispatch on TDMD_FLAVOR_MIXED_FAST.
class EamAlloyGpuMixed {
public:
  EamAlloyGpuMixed();
  ~EamAlloyGpuMixed();

  EamAlloyGpuMixed(const EamAlloyGpuMixed&) = delete;
  EamAlloyGpuMixed& operator=(const EamAlloyGpuMixed&) = delete;
  EamAlloyGpuMixed(EamAlloyGpuMixed&&) noexcept;
  EamAlloyGpuMixed& operator=(EamAlloyGpuMixed&&) noexcept;

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

  [[nodiscard]] std::uint64_t compute_version() const noexcept;

  // Counter of actual spline H2D uploads (T6.9a caching). Mirrors
  // `EamAlloyGpu::splines_upload_count()` — after N back-to-back compute()
  // calls with the same host spline table pointers, this equals 1.
  [[nodiscard]] std::uint64_t splines_upload_count() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tdmd::gpu
