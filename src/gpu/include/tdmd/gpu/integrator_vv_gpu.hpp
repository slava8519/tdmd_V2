#pragma once

// SPEC: docs/specs/gpu/SPEC.md §7.3 (VV NVE contract), §6.3 (D-M6-7 gate),
//       §8.1 (Reference FP64-only), §1.1 (data-oblivious gpu/)
// Master spec: §6.2 (NVE velocity-Verlet scheme)
// Module SPEC: docs/specs/integrator/SPEC.md §3 (math), §3.5 (GPU-resident),
//              §8 (precision)
// Exec pack: docs/development/m6_execution_pack.md T6.6
// Decisions: D-M6-4 (three M6 kernels — NL/EAM/VV), D-M6-7 (bit-exact gate),
//            D-M6-17 (PIMPL firewall)
//
// VelocityVerletGpu — device-resident NVE integrator kernels. Inputs are
// raw host primitives (positions, velocities, forces, atom types, and a
// per-species `ftm2v/m` accel-factor table). The adapter in
// src/integrator/ (`GpuVelocityVerletIntegrator`) translates from domain
// types (`AtomSoA`, `SpeciesRegistry`) into these primitives — keeps gpu/
// data-oblivious per module SPEC §1.1.
//
// Two entry points matching CPU `VelocityVerletIntegrator`:
//   pre_force_step:  v ← v + accel · f · (dt/2);  x ← x + v · dt
//   post_force_step: v ← v + accel · f · (dt/2)
// where `accel[s] = ftm2v / mass[s]` — a per-species scalar precomputed on
// host so the kernel does a single multiply per atom per axis.
//
// Both kernels are pure element-wise (thread per atom). No reductions, no
// atomics — so they are deterministic and the Reference FP64 path is
// bit-exact to the CPU integrator (both use FMAD-disabled FP64 math with
// identical operand order; see `BuildFlavors.cmake _tdmd_apply_fp64_reference`).
//
// Per-call H2D upload + D2H download in M6. The SPEC-mandated
// data-lives-on-GPU pattern (integrator/SPEC §3.5) is wired at T6.7 when
// SimulationEngine keeps atoms resident across iterations.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tdmd::gpu {

class VelocityVerletGpu {
public:
  VelocityVerletGpu();
  ~VelocityVerletGpu();

  VelocityVerletGpu(const VelocityVerletGpu&) = delete;
  VelocityVerletGpu& operator=(const VelocityVerletGpu&) = delete;
  VelocityVerletGpu(VelocityVerletGpu&&) noexcept;
  VelocityVerletGpu& operator=(VelocityVerletGpu&&) noexcept;

  // Half-kick velocities with CURRENT forces, then full drift positions:
  //   v[i] += accel_by_species[type[i]] * f[i] * (dt/2)
  //   x[i] += v[i] * dt
  //
  // Inputs:
  //   n                    : atom count (0 → no-op, version still bumps)
  //   dt                   : timestep (ps; must be finite and positive —
  //                          adapter validates)
  //   n_species            : length of accel_by_species[]
  //   host_accel_by_species: n_species doubles, [s] = ftm2v / mass[s]
  //   host_types           : n uint32_t species ids, each in [0, n_species)
  //   host_fx/fy/fz        : n doubles each, current forces (read-only)
  //   host_x/y/z           : n doubles each, positions (updated in place)
  //   host_vx/vy/vz        : n doubles each, velocities (updated in place)
  //
  // Throws std::runtime_error on CPU-only build or CUDA failure.
  void pre_force_step(std::size_t n,
                      double dt,
                      std::size_t n_species,
                      const double* host_accel_by_species,
                      const std::uint32_t* host_types,
                      const double* host_fx,
                      const double* host_fy,
                      const double* host_fz,
                      double* host_x,
                      double* host_y,
                      double* host_z,
                      double* host_vx,
                      double* host_vy,
                      double* host_vz,
                      DevicePool& pool,
                      DeviceStream& stream);

  // Half-kick velocities with NEW forces (computed by caller after drift):
  //   v[i] += accel_by_species[type[i]] * f[i] * (dt/2)
  //
  // Parameters match pre_force_step except no position arrays.
  void post_force_step(std::size_t n,
                       double dt,
                       std::size_t n_species,
                       const double* host_accel_by_species,
                       const std::uint32_t* host_types,
                       const double* host_fx,
                       const double* host_fy,
                       const double* host_fz,
                       double* host_vx,
                       double* host_vy,
                       double* host_vz,
                       DevicePool& pool,
                       DeviceStream& stream);

  // Monotone counter bumped once per {pre,post}_force_step call. Used by
  // tests to verify repeat-call behaviour without reaching into Impl.
  [[nodiscard]] std::uint64_t compute_version() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tdmd::gpu
