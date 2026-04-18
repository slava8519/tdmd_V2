#pragma once

// SPEC: docs/specs/io/SPEC.md §2.2 (LAMMPS data import)
// Exec pack: docs/development/m1_execution_pack.md T1.3
//
// Parser for LAMMPS `write_data` output, `atom_style atomic` only in M1.
// Populates `AtomSoA`, `Box`, and `SpeciesRegistry` from a header + sections
// stream (`Masses`, `Atoms`, and optionally `Velocities`).
//
// Scope (exec pack T1.3):
// - atom_style `atomic` only (M2 will add `charge`, `full`, `molecular`);
// - orthogonal box only (triclinic tilt `xy xz yz` is a hard reject);
// - units parameterized via `Options::units` (default `Metal` per master spec
//   §5.3 native representation). Only `Metal` is accepted in M1 — `Lj` et al.
//   are reserved for M2 once `UnitConverter::*_from_lj` is implemented.
// - streaming parse: never slurps the full file.
//
// Round-trip invariant: `parse → dump → parse` is bit-exact when the same
// unit system is used on both ends (exec pack T1.3 mandatory invariant).

#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"
#include "tdmd/state/species.hpp"
#include "tdmd/state/unit_system.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tdmd::io {

// Result metadata; per-atom data lands in the `AtomSoA` passed to the reader,
// box data in the `Box`, species metadata in the `SpeciesRegistry`.
struct LammpsDataImportResult {
  // `write_data` emits "timestep = N" in its header comment. Preserved when
  // present; absent otherwise.
  std::optional<std::uint64_t> timestep;
  std::size_t atom_count = 0;
  std::size_t atom_types = 0;
  // True iff the file provided a `Velocities` section. When false, all
  // `vx / vy / vz` in the output SoA are zero (default of `add_atom`).
  bool has_velocities = false;
};

// Caller-provided knobs. `units` must match what the `.data` file actually
// contains. The `species_names` / `atomic_numbers` vectors, if non-empty, map
// LAMMPS type_id (1-based) to human-readable species metadata; missing entries
// fall back to an auto-generated `"type_<i>"` name with `atomic_number = 0`.
struct LammpsDataImportOptions {
  UnitSystem units = UnitSystem::Metal;
  std::vector<std::string> species_names;     // index i → type_id (i+1)
  std::vector<std::uint32_t> atomic_numbers;  // index i → type_id (i+1)
};

// Thrown on any parse failure. `line_number` is 1-based and points at the
// offending line (or the last line scanned if EOF was reached unexpectedly).
// `what()` is human-readable and includes `line_number` + the offending token
// where relevant.
class LammpsDataParseError : public std::runtime_error {
public:
  LammpsDataParseError(std::size_t line_number, std::string_view message);

  [[nodiscard]] std::size_t line_number() const noexcept { return line_number_; }

private:
  std::size_t line_number_;
};

// Parse `.data` from `in`. Writes atoms to `out_atoms` (which MUST be empty on
// entry — we append rather than resize to keep AtomId monotonic), box bounds
// to `out_box`, and species to `out_species`. Throws `LammpsDataParseError`
// on malformed input.
LammpsDataImportResult read_lammps_data(std::istream& in,
                                        const LammpsDataImportOptions& options,
                                        AtomSoA& out_atoms,
                                        Box& out_box,
                                        SpeciesRegistry& out_species);

// Convenience wrapper: opens `path` with `std::ifstream` and forwards to the
// stream-based overload. Throws `LammpsDataParseError` if the file cannot be
// opened or parsed.
LammpsDataImportResult read_lammps_data_file(const std::string& path,
                                             const LammpsDataImportOptions& options,
                                             AtomSoA& out_atoms,
                                             Box& out_box,
                                             SpeciesRegistry& out_species);

}  // namespace tdmd::io
