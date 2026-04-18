#pragma once

// SPEC: docs/specs/potentials/SPEC.md §2.1 (abstract PotentialModel interface).
// Exec pack: docs/development/m2_execution_pack.md T2.7.
//
// Abstract base for all interaction potentials. `compute` writes pair forces
// into `atoms.fx/fy/fz` (additive — caller zeroes first) and returns total
// potential energy and virial.
//
// SPEC §2.1 describes a richer interface (ForceRequest/ForceResult structs,
// ComputeMask, PotentialKind, estimated_flops_per_atom, parameter_summary,
// parameter_checksum). Those hooks exist to support scheduler filtering,
// perfmodel calibration, and explain/repro-bundle output. M2 introduces only
// the minimal subset exercised by the current runtime and tests
// (compute, cutoff, name) — alignment with the full SPEC interface is a
// scheduled refactor for the telemetry/explain milestones (T2.11/T2.12) and
// the scheduler milestone (M3+).

#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"

#include <array>
#include <string>

namespace tdmd {

// Aggregate returned from `Potential::compute`. Per-atom forces are written
// in-place into `atoms.fx/fy/fz`; this struct carries only the scalar /
// tensor totals. Voigt virial ordering: (xx, yy, zz, xy, xz, yz).
struct ForceResult {
  double potential_energy = 0.0;      // eV
  std::array<double, 6> virial = {};  // eV·Å, Voigt order
};

class Potential {
public:
  Potential() = default;
  virtual ~Potential() = default;

  Potential(const Potential&) = delete;
  Potential& operator=(const Potential&) = delete;
  Potential(Potential&&) = default;
  Potential& operator=(Potential&&) = default;

  // Accumulates pair forces into `atoms.fx/fy/fz` and returns total PE and
  // virial. Caller MUST zero the force fields beforehand — the potential is
  // additive over pair contributions (matches Morse convention). Non-const:
  // multi-body potentials (EAM) maintain per-compute scratch buffers
  // (density, dF/dρ) as member state to avoid hot-path allocation.
  [[nodiscard]] virtual ForceResult compute(AtomSoA& atoms,
                                            const NeighborList& neighbors,
                                            const Box& box) = 0;

  // Interaction cutoff in Å (or the build's length unit). Caller uses this
  // to size the neighbor list; see neighbor/SPEC.md §3.
  [[nodiscard]] virtual double cutoff() const noexcept = 0;

  // Short human-readable name (matches `pair_style` in LAMMPS output — e.g.
  // "morse", "eam/alloy", "eam/fs"). Used by telemetry and explain output.
  [[nodiscard]] virtual std::string name() const = 0;
};

}  // namespace tdmd
