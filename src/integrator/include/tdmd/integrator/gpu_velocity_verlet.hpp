#pragma once

// SPEC: docs/specs/gpu/SPEC.md §7.3 (VV NVE contract), §1.1 (data-oblivious gpu/)
// Module SPEC: docs/specs/integrator/SPEC.md §3 (VV math)
// Exec pack: docs/development/m6_execution_pack.md T6.6
//
// Domain-side facade over `tdmd::gpu::VelocityVerletGpu`. Takes `AtomSoA` +
// `SpeciesRegistry` (M1 interface, same as CPU `VelocityVerletIntegrator`)
// and translates to the data-oblivious gpu/ primitives: a flat
// `accel_by_species[]` table (`ftm2v / mass[s]` per species, precomputed
// once at construction) and raw host pointers for atom arrays.
//
// The adapter does NOT implement the `Integrator` interface yet — that
// virtual surface takes `StateManager` + `ZoneFilter`, neither of which
// has a GPU-ready form in M6. Wiring into scheduler/SimulationEngine is
// T6.7 scope.
//
// Semantics:
//   - Same ftm2v convention as CPU `VelocityVerletIntegrator` (metal units,
//     1/1.0364269e-4 ≈ 9648.533) — see CPU header for the SPEC §3.4 deviation
//     note; GPU path uses the identical constant so Reference paths agree
//     bit-exact under `--fmad=false`.
//   - pre/post kernels are per-call H2D + kernel + D2H. Resident-on-GPU
//     pattern (integrator/SPEC §3.5) arrives at T6.7.

#include "tdmd/gpu/device_pool.hpp"
#include "tdmd/gpu/integrator_vv_gpu.hpp"
#include "tdmd/gpu/types.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/species.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace tdmd {

class GpuVelocityVerletIntegrator {
public:
  // Precomputes `accel_by_species[s] = ftm2v / mass[s]` from the registry.
  // Throws std::invalid_argument if the registry is empty or any species
  // has non-finite / non-positive mass.
  explicit GpuVelocityVerletIntegrator(const SpeciesRegistry& species);

  ~GpuVelocityVerletIntegrator();

  GpuVelocityVerletIntegrator(const GpuVelocityVerletIntegrator&) = delete;
  GpuVelocityVerletIntegrator& operator=(const GpuVelocityVerletIntegrator&) = delete;
  GpuVelocityVerletIntegrator(GpuVelocityVerletIntegrator&&) noexcept;
  GpuVelocityVerletIntegrator& operator=(GpuVelocityVerletIntegrator&&) noexcept;

  // Half-kick velocities using CURRENT forces, then full drift positions.
  // Throws std::invalid_argument on dt ≤ 0 or non-finite; std::runtime_error
  // if any atom carries a species id outside the registry.
  void pre_force_step(AtomSoA& atoms,
                      double dt,
                      tdmd::gpu::DevicePool& pool,
                      tdmd::gpu::DeviceStream& stream);

  // Half-kick velocities using NEW forces (computed after drift).
  void post_force_step(AtomSoA& atoms,
                       double dt,
                       tdmd::gpu::DevicePool& pool,
                       tdmd::gpu::DeviceStream& stream);

  // Monotone counter bumped once per kernel launch (both halves each
  // increment). Mirrors the underlying gpu::VelocityVerletGpu counter.
  [[nodiscard]] std::uint64_t compute_version() const noexcept;

private:
  std::vector<double> accel_by_species_;  // ftm2v / mass[s]
  std::unique_ptr<tdmd::gpu::VelocityVerletGpu> gpu_;
};

}  // namespace tdmd
