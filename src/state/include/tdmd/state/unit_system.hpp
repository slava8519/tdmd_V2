#pragma once

// Standalone header for the `tdmd::UnitSystem` enum, isolated from
// `runtime/unit_converter.hpp` so modules that only need the tag (e.g. io/)
// do not pull the full UnitConverter interface. The actual conversion class
// still lives in `runtime/`; this header only owns the literal.
//
// Lifted into state/ because UnitSystem is a TYPE (alongside AtomSoA / Box)
// rather than a policy, and state/ is depended on by every module, which
// avoids a runtime↔io include cycle (io/lammps_data_reader.hpp needs the
// tag; runtime/simulation_engine.cpp needs io/yaml_config.hpp).

#include <cstdint>

namespace tdmd {

enum class UnitSystem : std::uint8_t {
  Metal,  // LAMMPS metal: Å, eV, g/mol, ps
  Lj,     // LAMMPS lj:    reduced (sigma, epsilon, mass) — M2 impl
  Real,   // LAMMPS real:  recognized, not supported in v1
  Cgs,    // cgs:          recognized, not supported in v1
  Si,     // SI:           never supported (master spec §5.3 policy)
};

}  // namespace tdmd
