#pragma once

// SPEC: docs/specs/integrator/SPEC.md §3 (Velocity-Verlet)
// Exec pack: docs/development/m1_execution_pack.md T1.7
//
// Classical velocity-Verlet NVE integrator. Two-phase per SPEC §3.1:
//   pre_force_step  — half-kick velocities with CURRENT forces + full drift positions
//   [caller: zero forces, then compute new forces at the drifted positions]
//   post_force_step — half-kick velocities with NEW forces
//
// Force zeroing and recomputation between the two phases are caller responsibility
// (exec pack T1.7 scope). End-of-step position wrapping is NOT part of this task
// (it is the caller's / later integration surface's responsibility in M1).
//
// Unit convention — NOTE DEVIATION from SPEC §3.4 "unity" claim.
//
// SPEC §3.4 asserts that in TDMD metal units (mass g/mol, force eV/Å, position Å,
// velocity Å/ps, time ps), the raw quotient `f/m` already carries units of Å/ps²
// with no conversion factor needed. Dimensional analysis and LAMMPS `metal` unit
// convention both contradict this:
//
//   (1 eV/Å) / (1 g/mol)
//     = (1.602176634e-19 J / 1e-10 m) / (1e-3 kg/mol · 1/N_A mol/atom)
//     = 1.602176634e-9 N / 1.66053907e-27 kg
//     = 9.648533e17 m/s²
//     = 9648.533 Å/ps²
//
// i.e. a factor of ~9648.5 is required — this is exactly LAMMPS `force->ftm2v`
// for `units metal` (`1/1.0364269e-4`). Implementing with that factor; will be
// cross-validated at T1.11 against the LAMMPS oracle. If the discrepancy holds
// (which physics requires), a SPEC delta updating §3.4 is to follow.
//
// Signatures deviate slightly from SPEC §2.1's virtual interface
// `(StateManager&, ZoneFilter, dt)` because StateManager arrives in T1.9 and
// ZoneFilter is a TD concept introduced at M4+. Per exec pack T1.7, M1 uses the
// narrower `(AtomSoA&, const SpeciesRegistry&, double)` form. Masses come from
// the species registry rather than a hypothetical `AtomSoA::mass()` method
// (AtomSoA has no per-atom mass field — mass is species-level in state/SPEC §5).

#include "tdmd/integrator/integrator.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/species.hpp"

namespace tdmd {

// LAMMPS `units metal` conversion constants (see header comment for derivation
// and the SPEC §3.4 deviation note).
//
//   ftm2v : multiplier on raw (f/m) to produce acceleration in Å/ps².
//   mvv2e : multiplier on raw (m·v²) to produce energy in eV.
//
// Both are exact inverses of each other (ftm2v · mvv2e = 1); quoted here as
// separate literals for numerical symmetry with LAMMPS source.
inline constexpr double kMetalFtm2v = 1.0 / 1.0364269e-4;  // ≈ 9648.533
inline constexpr double kMetalMvv2e = 1.0364269e-4;

class VelocityVerletIntegrator final : public Integrator {
public:
  VelocityVerletIntegrator() = default;

  // Half-kick velocities with the CURRENT forces, then full drift positions:
  //   v(t + dt/2) = v(t) + (f(t) / m) · ftm2v · dt/2
  //   x(t + dt)   = x(t) + v(t + dt/2) · dt
  //
  // Does NOT recompute forces. The caller must (a) zero atoms.f* and
  // (b) evaluate the force field at the drifted positions before calling
  // `post_force_step`. Throws std::invalid_argument if `dt` is not finite
  // or not strictly positive.
  void pre_force_step(AtomSoA& atoms, const SpeciesRegistry& species, double dt);

  // Half-kick velocities with the NEW forces (computed by caller after drift):
  //   v(t + dt) = v(t + dt/2) + (f(t + dt) / m) · ftm2v · dt/2
  //
  // Throws std::invalid_argument if `dt` is not finite or not strictly positive.
  void post_force_step(AtomSoA& atoms, const SpeciesRegistry& species, double dt);
};

// Total kinetic energy of the system, in eV (metal units):
//   KE = 0.5 · Σ_i m_i · |v_i|² · mvv2e
//
// Pure read-only reduction — a free function because no integrator state is
// involved. Returns 0.0 for an empty system.
[[nodiscard]] double kinetic_energy(const AtomSoA& atoms, const SpeciesRegistry& species);

}  // namespace tdmd
