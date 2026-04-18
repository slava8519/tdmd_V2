#pragma once

// SPEC: TDMD master spec §5.3 (unit system policy).
// Module home: src/runtime/ (see docs/specs/runtime/SPEC.md §2).
//
// Single source of truth for physical constants used across the runtime,
// integrator, and potentials layers. Keeping them in one place avoids the
// class of bugs where a thermostat reads one kB and the KE→T converter
// reads another. If a constant needs to be updated (e.g. a future CODATA
// revision), this is the file to touch.
//
// Values are derived from CODATA 2019 SI exact redefinitions unless noted.
// LAMMPS-derived "convenience" factors (ftm2v / mvv2e) currently live in
// `integrator/velocity_verlet.hpp` because they encode a specific LAMMPS
// convention (see project memory `project_metal_unit_factor.md`); they are
// not moved here to keep that provenance explicit.

namespace tdmd {

// Boltzmann constant in metal units (eV / K).
// Derived from CODATA 2019 exact: kB = 1.380649e-23 J/K, e = 1.602176634e-19 C
//   => kB/e = 8.617333262...e-5 eV/K.
// Diverges from LAMMPS's `boltz = 8.617343e-5` (older value) by ~1.13e-6
// relative. This gap is the documented residual floor in the T1 differential
// harness thresholds; see `verify/thresholds/thresholds.yaml` rationale block.
inline constexpr double kBoltzmann_eV_per_K = 8.617333262e-5;

}  // namespace tdmd
