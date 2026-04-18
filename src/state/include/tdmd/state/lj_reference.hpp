#pragma once

// POD describing the reference (σ, ε, m) triple that LAMMPS `units lj` needs
// to interpret a dimensionless lj value as a physical quantity.
//
// Lifted into state/ (next to `unit_system.hpp`) because the triple is a TYPE
// consumed by multiple modules:
//   - io/yaml_config: `simulation.reference` block parses into this struct;
//   - io/preflight: semantic validation of the triple;
//   - runtime/unit_converter: the actual conversion formulas;
//   - runtime/simulation_engine: conversion at the ingest boundary.
//
// Keeping it in state/ avoids an io→runtime include cycle (io cannot depend on
// runtime — io/ is a leaf beneath runtime/ in the dep DAG). The conversion
// formulas themselves still live in runtime/unit_converter; this header only
// owns the parameter bundle.
//
// Identity reference (σ=ε=m=1) makes lj ↔ metal a no-op scaling for length /
// energy / mass and a pure-conversion-factor operation for the derived
// dimensions. All fields must be strictly positive; downstream code (the
// converter or preflight) is responsible for rejecting non-positive values.

namespace tdmd {

struct LjReference {
  double sigma = 1.0;    // Å
  double epsilon = 1.0;  // eV
  double mass = 1.0;     // g/mol
};

}  // namespace tdmd
