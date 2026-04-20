// SPEC: docs/specs/gpu/SPEC.md §7.3 (SNAP GPU kernel contract — T8.6 authors);
//       §6.3 (D-M6-7 bit-exact gate extended to SNAP at T8.7),
//       §8.1 (Reference FP64-only), §9 (NVTX on every launch)
// Module SPEC: docs/specs/potentials/SPEC.md §6 (SNAP module contract)
// Exec pack: docs/development/m8_execution_pack.md T8.6a (scaffolding body) /
//            T8.6b (full kernel translation)
// Decisions: D-M6-17 (PIMPL firewall), D-M8-13 (GPU Fp64 ≡ CPU Fp64 ≤ 1e-12
//            rel — exercised at T8.7).
//
// T8.6a scope. This TU provides:
//   * The SnapGpu::Impl PIMPL body (state fields declared; kernels come T8.6b).
//   * A compute() path that throws a sentinel std::logic_error so the public
//     ABI is link-clean and every consumer (SnapGpuAdapter, SimulationEngine
//     dispatch, test_snap_gpu_plumbing) can exercise the full surface.
//   * CPU-only (#else) branch matching EAM precedent — throws "CPU-only build"
//     exactly as EamAlloyGpu does.
//
// Nothing in this TU issues a `<<<...>>>` kernel launch. That is intentional:
//   * NVTX audit (`test_nvtx_audit.cpp`) stays green because there are no
//     launches to wrap.
//   * No device code means no nvcc compilation stress for T8.6a CI.
//   * T8.6b owns the actual kernel bodies + NVTX wrappers.
//
// The TU follows the EAM precedent layout:
//   #if TDMD_BUILD_CUDA
//     struct Impl { … device state fields … };
//     SnapGpu ctor/dtor/move + compute (throws T8.6b sentinel);
//   #else
//     struct Impl {};
//     SnapGpu ctor/dtor/move + compute (throws CPU-only sentinel);
//   #endif

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/neighbor_list_gpu.hpp"
#include "tdmd/gpu/snap_gpu.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/telemetry/nvtx.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if TDMD_BUILD_CUDA
#include "cuda_handles.hpp"

#include <cuda_runtime.h>
#endif

namespace tdmd::gpu {

#if TDMD_BUILD_CUDA

// ---------------------------------------------------------------------------
// T8.6a PIMPL state. Fields listed here are what the T8.6b kernel body will
// consume; they are declared (not yet used) so the Impl struct size is stable
// across the T8.6a → T8.6b landing. Device buffers are allocated lazily in
// compute() once T8.6b lands. The index-table buffers (Clebsch-Gordan,
// root-pq, idxcg_block, etc.) follow T6.9a spline-cache pattern — uploaded
// once on first compute() with new tables.
// ---------------------------------------------------------------------------
struct SnapGpu::Impl {
  std::uint64_t compute_version = 0;

  // Persistent per-atom device buffers (grown on demand).
  DevicePtr<std::byte> d_types_bytes;
  DevicePtr<std::byte> d_x_bytes;
  DevicePtr<std::byte> d_y_bytes;
  DevicePtr<std::byte> d_z_bytes;
  DevicePtr<std::byte> d_fx_bytes;
  DevicePtr<std::byte> d_fy_bytes;
  DevicePtr<std::byte> d_fz_bytes;
  DevicePtr<std::byte> d_cell_offsets_bytes;
  DevicePtr<std::byte> d_cell_atoms_bytes;

  // Per-atom scratch for bispectrum passes (populated T8.6b).
  DevicePtr<std::byte> d_ulisttot_bytes;  // n × idxu_max × 2 (r,i) doubles
  DevicePtr<std::byte> d_ylist_bytes;     // n × idxu_max × 2 (r,i) doubles
  DevicePtr<std::byte> d_blist_bytes;     // n × idxb_max doubles

  // Host-reduction destinations (D2H targets).
  DevicePtr<std::byte> d_pe_per_atom_bytes;
  DevicePtr<std::byte> d_virial_per_atom_bytes;

  // SNAP parameter tables — uploaded once, reused across compute() calls.
  DevicePtr<std::byte> d_radius_elem_bytes;
  DevicePtr<std::byte> d_weight_elem_bytes;
  DevicePtr<std::byte> d_beta_bytes;
  DevicePtr<std::byte> d_rcut_sq_ab_bytes;

  // Index tables (populated by Impl::init_index_tables() at T8.6b). All are
  // derived from twojmax and uploaded once.
  DevicePtr<std::byte> d_idxcg_block_bytes;
  DevicePtr<std::byte> d_idxu_block_bytes;
  DevicePtr<std::byte> d_idxz_block_bytes;
  DevicePtr<std::byte> d_cg_coefficients_bytes;
  DevicePtr<std::byte> d_rootpq_bytes;

  // Upload-identity cache (same tables pointer ⇒ skip H2D, T6.9a pattern).
  const double* tables_radius_host = nullptr;
  const double* tables_weight_host = nullptr;
  const double* tables_beta_host = nullptr;
  const double* tables_rcut_sq_host = nullptr;
  std::uint64_t tables_upload_count = 0;
};

SnapGpu::SnapGpu() : impl_(std::make_unique<Impl>()) {}
SnapGpu::~SnapGpu() = default;
SnapGpu::SnapGpu(SnapGpu&&) noexcept = default;
SnapGpu& SnapGpu::operator=(SnapGpu&&) noexcept = default;

std::uint64_t SnapGpu::compute_version() const noexcept {
  return impl_ ? impl_->compute_version : 0;
}

SnapGpuResult SnapGpu::compute(std::size_t /*n*/,
                               const std::uint32_t* /*host_types*/,
                               const double* /*host_x*/,
                               const double* /*host_y*/,
                               const double* /*host_z*/,
                               std::size_t /*ncells*/,
                               const std::uint32_t* /*host_cell_offsets*/,
                               const std::uint32_t* /*host_cell_atoms*/,
                               const BoxParams& /*params*/,
                               const SnapTablesHost& /*tables*/,
                               double* /*host_fx_out*/,
                               double* /*host_fy_out*/,
                               double* /*host_fz_out*/,
                               DevicePool& /*pool*/,
                               DeviceStream& /*stream*/) {
  // T8.6a sentinel. See snap_gpu.hpp compute() docstring and
  // m8_execution_pack.md T8.6b for the kernel-body landing plan.
  TDMD_NVTX_RANGE("snap.compute_stub");
  throw std::logic_error(
      "SnapGpu::compute: T8.6b kernel body not landed — "
      "set runtime.backend=cpu or await T8.6b merge");
}

#else  // CPU-only build

struct SnapGpu::Impl {};

SnapGpu::SnapGpu() : impl_(std::make_unique<Impl>()) {}
SnapGpu::~SnapGpu() = default;
SnapGpu::SnapGpu(SnapGpu&&) noexcept = default;
SnapGpu& SnapGpu::operator=(SnapGpu&&) noexcept = default;

std::uint64_t SnapGpu::compute_version() const noexcept {
  return 0;
}

SnapGpuResult SnapGpu::compute(std::size_t /*n*/,
                               const std::uint32_t* /*host_types*/,
                               const double* /*host_x*/,
                               const double* /*host_y*/,
                               const double* /*host_z*/,
                               std::size_t /*ncells*/,
                               const std::uint32_t* /*host_cell_offsets*/,
                               const std::uint32_t* /*host_cell_atoms*/,
                               const BoxParams& /*params*/,
                               const SnapTablesHost& /*tables*/,
                               double* /*host_fx_out*/,
                               double* /*host_fy_out*/,
                               double* /*host_fz_out*/,
                               DevicePool& /*pool*/,
                               DeviceStream& /*stream*/) {
  throw std::runtime_error(
      "gpu::SnapGpu::compute: CPU-only build (TDMD_BUILD_CUDA=0); CUDA not linked");
}

#endif  // TDMD_BUILD_CUDA

}  // namespace tdmd::gpu
