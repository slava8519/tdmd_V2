#include "tdmd/cli/field_docs.hpp"

namespace tdmd::cli {

const std::map<std::string, std::string>& config_field_descriptions() {
  static const std::map<std::string, std::string> table = {
      {"units",
       "simulation.units selects the unit system the .data file and potential "
       "parameters are in. M1 supports 'metal' (Angstrom / eV / g/mol / ps) "
       "only; 'lj' is reserved for M2. 'real', 'cgs', 'si' are recognised but "
       "not supported in v1 (master spec §5.3)."},
      {"atoms.source",
       "atoms.source chooses how initial positions / velocities are produced. "
       "M1 supports 'lammps_data' (read a LAMMPS `write_data` file) only; "
       "generators are deferred to M2."},
      {"atoms.path",
       "atoms.path is the path to the LAMMPS .data file. Relative paths are "
       "resolved relative to the directory containing the YAML config."},
      {"potential.style",
       "potential.style names the pair/many-body form. M1 supports 'morse' "
       "only; EAM / SNAP / MEAM are M2-M3+."},
      {"potential.params.cutoff_strategy",
       "cutoff_strategy selects how the pair energy/force is handled at the "
       "cutoff radius. 'shifted_force' (default) subtracts a linear ramp so "
       "force is exactly zero at r = cutoff. 'hard_cutoff' truncates with a "
       "step discontinuity — only for bitwise reference comparisons."},
      {"integrator.style",
       "integrator.style chooses the time-integration scheme. M1 supports "
       "'velocity_verlet' (NVE) only; NVT/NPT land in M9."},
      {"integrator.dt",
       "integrator.dt is the timestep in the active unit system's native time "
       "unit (ps for metal, reduced units for lj). Must be > 0 and finite."},
      {"neighbor.skin",
       "neighbor.skin is the extra buffer around each atom's cutoff sphere. A "
       "larger skin rebuilds the list less often but costs more pairwise "
       "distance checks. M1 default is 0.3 A."},
      {"thermo.every",
       "thermo.every is the step interval at which a thermo row is emitted. "
       "Row 0 (initial state) and the final step are always included."},
      {"run.n_steps",
       "run.n_steps is the total number of integration steps to perform. "
       "Must be >= 1."},
  };
  return table;
}

}  // namespace tdmd::cli
