#include "tdmd/runtime/simulation_engine.hpp"

#include "tdmd/io/lammps_data_reader.hpp"
#include "tdmd/io/yaml_config.hpp"
#include "tdmd/potentials/eam_alloy.hpp"
#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/runtime/physical_constants.hpp"
#include "tdmd/runtime/unit_converter.hpp"
#include "tdmd/state/lj_reference.hpp"
#include "tdmd/telemetry/telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tdmd {

namespace {

// CutoffStrategy translation — YAML enum to MorsePotential::CutoffStrategy.
MorsePotential::CutoffStrategy to_morse_strategy(io::MorseCutoffStrategy s) noexcept {
  switch (s) {
    case io::MorseCutoffStrategy::HardCutoff:
      return MorsePotential::CutoffStrategy::HardCutoff;
    case io::MorseCutoffStrategy::ShiftedForce:
      return MorsePotential::CutoffStrategy::ShiftedForce;
  }
  return MorsePotential::CutoffStrategy::ShiftedForce;
}

std::string resolve_atoms_path(const std::string& atoms_path, const std::string& config_dir) {
  namespace fs = std::filesystem;
  fs::path p(atoms_path);
  if (p.is_absolute() || config_dir.empty()) {
    return atoms_path;
  }
  return (fs::path(config_dir) / p).lexically_normal().string();
}

// Convert every raw-from-LAMMPS value in `atoms`, `box`, and `species` from
// lj units into native metal, in place. Called only when `units=lj` in the
// config; preflight has already guaranteed that `ref` exists and that
// (σ, ε, m) are finite-positive, so `UnitConverter::*_from_lj` cannot throw.
//
// The transform is a pure scaling per dimension:
//   positions   — length:   l_metal = l_lj · σ
//   velocities  — velocity: v_metal = v_lj · sqrt(ε/m) · kLjVelocityFactor
//   box bounds  — length (same as positions)
//   masses      — mass:     m_metal = m_lj · m_ref
// Forces are NOT scaled (the force array is zeroed and recomputed after this
// helper returns, so any transient lj values there are discarded).
//
// SpeciesRegistry is immutable-after-init by contract (state/SPEC §5.1), so
// we rebuild it with converted masses rather than mutating entries in place.
void convert_state_lj_to_metal(AtomSoA& atoms,
                               Box& box,
                               SpeciesRegistry& species,
                               const LjReference& ref) {
  const std::size_t n = atoms.size();
  for (std::size_t i = 0; i < n; ++i) {
    atoms.x[i] = UnitConverter::length_from_lj(atoms.x[i], ref).metal_angstroms;
    atoms.y[i] = UnitConverter::length_from_lj(atoms.y[i], ref).metal_angstroms;
    atoms.z[i] = UnitConverter::length_from_lj(atoms.z[i], ref).metal_angstroms;
    atoms.vx[i] = UnitConverter::velocity_from_lj(atoms.vx[i], ref).metal_A_per_ps;
    atoms.vy[i] = UnitConverter::velocity_from_lj(atoms.vy[i], ref).metal_A_per_ps;
    atoms.vz[i] = UnitConverter::velocity_from_lj(atoms.vz[i], ref).metal_A_per_ps;
  }
  box.xlo = UnitConverter::length_from_lj(box.xlo, ref).metal_angstroms;
  box.xhi = UnitConverter::length_from_lj(box.xhi, ref).metal_angstroms;
  box.ylo = UnitConverter::length_from_lj(box.ylo, ref).metal_angstroms;
  box.yhi = UnitConverter::length_from_lj(box.yhi, ref).metal_angstroms;
  box.zlo = UnitConverter::length_from_lj(box.zlo, ref).metal_angstroms;
  box.zhi = UnitConverter::length_from_lj(box.zhi, ref).metal_angstroms;
  SpeciesRegistry rebuilt;
  for (std::size_t t = 0; t < species.count(); ++t) {
    SpeciesInfo info = species.get_info(static_cast<SpeciesId>(t));
    info.mass = UnitConverter::mass_from_lj(info.mass, ref).metal_g_per_mol;
    (void) rebuilt.register_species(info);
  }
  species = std::move(rebuilt);
}

}  // namespace

SimulationEngine::SimulationEngine() = default;
SimulationEngine::~SimulationEngine() = default;

void SimulationEngine::init(const io::YamlConfig& config, const std::string& config_dir) {
  if (state_ != State::Constructed) {
    throw SimulationEngineStateError("SimulationEngine::init called twice (or after finalize)");
  }

  // --- Load atoms / box / species from the referenced LAMMPS .data file.
  // The reader is unit-agnostic: values are stored exactly as they appear in
  // the file. Lj → metal conversion happens below, once, at this ingest
  // boundary — master spec §5.3 "UnitConverter is called at I/O boundary,
  // every other module works in metal".
  const bool is_lj = config.simulation.units == io::UnitsKind::Lj;
  const std::string atoms_path = resolve_atoms_path(config.atoms.path, config_dir);
  io::LammpsDataImportOptions opts{};
  opts.units = is_lj ? UnitSystem::Lj : UnitSystem::Metal;
  atoms_ = AtomSoA{};
  box_ = Box{};
  species_ = SpeciesRegistry{};
  (void) io::read_lammps_data_file(atoms_path, opts, atoms_, box_, species_);

  // Optional in-place lj → metal conversion. Preflight has already checked
  // that `config.simulation.reference` is present and well-formed, so the
  // `.value()` call cannot throw.
  LjReference lj_ref{};
  if (is_lj) {
    lj_ref = config.simulation.reference.value();
    convert_state_lj_to_metal(atoms_, box_, species_, lj_ref);
  }

  // --- Build the potential. Dispatched on `config.potential.style`:
  //   * morse     — native pair potential, lj→metal conversion done here
  //                 for D (energy), alpha (1/length), r0 / cutoff (length).
  //   * eam/alloy — parses a LAMMPS setfl file at load time. The file is
  //                 always in metal units (LAMMPS convention for
  //                 `pair_style eam/alloy`); the YAML's lj mode is rejected
  //                 with an explicit message because EAM tables are not
  //                 dimensionless in the Andreev/LJ sense.
  switch (config.potential.style) {
    case io::PotentialStyle::Morse: {
      const auto& mp = config.potential.morse;
      MorsePotential::PairParams pp{.D = mp.D, .alpha = mp.alpha, .r0 = mp.r0, .cutoff = mp.cutoff};
      if (is_lj) {
        pp.D = UnitConverter::energy_from_lj(mp.D, lj_ref).metal_eV;
        pp.alpha = mp.alpha / lj_ref.sigma;
        pp.r0 = UnitConverter::length_from_lj(mp.r0, lj_ref).metal_angstroms;
        pp.cutoff = UnitConverter::length_from_lj(mp.cutoff, lj_ref).metal_angstroms;
      }
      potential_ = std::make_unique<MorsePotential>(pp, to_morse_strategy(mp.cutoff_strategy));
      break;
    }
    case io::PotentialStyle::EamAlloy: {
      if (is_lj) {
        throw std::invalid_argument(
            "potential.style=eam/alloy is incompatible with simulation.units=lj "
            "(EAM setfl tables are dimensional by convention)");
      }
      const std::string eam_path = resolve_atoms_path(config.potential.eam_alloy.file, config_dir);
      potentials::EamAlloyData eam = potentials::parse_eam_alloy(eam_path);
      potential_ = std::make_unique<EamAlloyPotential>(std::move(eam));
      break;
    }
  }
  cutoff_ = potential_->cutoff();

  // --- Build the integrator.
  integrator_ = std::make_unique<VelocityVerletIntegrator>();
  dt_ = is_lj ? UnitConverter::time_from_lj(config.integrator.dt, lj_ref).metal_ps
              : config.integrator.dt;

  // --- Neighbor pipeline.
  skin_ = is_lj ? UnitConverter::length_from_lj(config.neighbor.skin, lj_ref).metal_angstroms
                : config.neighbor.skin;
  thermo_every_ = std::max<std::uint64_t>(config.thermo.every, 1U);
  cell_grid_.build(box_, cutoff_, skin_);
  cell_grid_.bin(atoms_);
  if (auto reorder = cell_grid_.compute_stable_reorder(atoms_); !reorder.empty()) {
    apply_reorder(atoms_, reorder);
    cell_grid_.bin(atoms_);
  }
  neighbor_list_.build(atoms_, box_, cell_grid_, cutoff_, skin_);
  displacement_tracker_.set_threshold(0.5 * skin_);
  displacement_tracker_.reset(atoms_);

  // --- Initial force / energy / virial snapshot.
  recompute_forces();

  state_ = State::Initialised;
  current_step_ = 0;
}

ThermoRow SimulationEngine::run(std::uint64_t n_steps, std::ostream* thermo_out) {
  if (state_ != State::Initialised) {
    throw SimulationEngineStateError(
        "SimulationEngine::run called before init (or after finalize)");
  }
  if (thermo_out != nullptr) {
    write_thermo_header(*thermo_out);
  }

  ThermoRow last_row = snapshot_thermo(current_step_);
  if (thermo_out != nullptr) {
    write_thermo_row(*thermo_out, last_row);
  }

  for (std::uint64_t s = 1; s <= n_steps; ++s) {
    // Pre-force: half-kick + drift.
    integrator_->pre_force_step(atoms_, species_, dt_);

    // Neighbor rebuild check — based on displacements since the last build.
    {
      telemetry::ScopedSection neigh(telemetry_, "Neigh");
      displacement_tracker_.update(atoms_, box_);
      if (displacement_tracker_.needs_rebuild()) {
        rebuild_neighbors();
      }
    }

    // New forces at drifted positions. `recompute_forces` already wraps the
    // pair-potential call in a "Pair" scope, so no extra wrapper here.
    recompute_forces();

    // Post-force: half-kick.
    integrator_->post_force_step(atoms_, species_, dt_);

    current_step_ = s;

    if (s % thermo_every_ == 0 || s == n_steps) {
      telemetry::ScopedSection output(telemetry_, "Output");
      last_row = snapshot_thermo(s);
      if (thermo_out != nullptr) {
        write_thermo_row(*thermo_out, last_row);
      }
    }
  }

  return last_row;
}

void SimulationEngine::finalize() {
  // No persistent state in M1; transition the FSM so a post-finalize run()
  // throws the expected SimulationEngineStateError.
  if (state_ == State::Constructed) {
    throw SimulationEngineStateError("SimulationEngine::finalize called before init");
  }
  state_ = State::Finalised;
}

void SimulationEngine::write_thermo_header(std::ostream& out) {
  out << "# step temp pe ke etotal press\n";
}

void SimulationEngine::write_thermo_row(std::ostream& out, const ThermoRow& row) {
  // Fixed-precision output (10 significant digits) so determinism tests can
  // byte-compare the stream. `std::setprecision` with `std::scientific` is
  // locale-independent modulo the global `imbue` — we keep the C locale.
  std::ios::fmtflags saved = out.flags();
  const auto saved_precision = out.precision();
  out << row.step << ' ' << std::setprecision(10) << std::scientific << row.temperature_K << ' '
      << row.potential_energy << ' ' << row.kinetic_energy << ' ' << row.total_energy << ' '
      << row.pressure_ev_A3 << '\n';
  out.flags(saved);
  out.precision(saved_precision);
}

void SimulationEngine::write_dump_frame(std::ostream& out) const {
  const std::size_t n = atoms_.size();

  // LAMMPS header — pp pp pp = fully periodic box (master spec §5.2 assumes
  // periodic in all three axes for M1/M2; open boundaries are post-M2).
  out << "ITEM: TIMESTEP\n" << current_step_ << '\n';
  out << "ITEM: NUMBER OF ATOMS\n" << n << '\n';
  out << "ITEM: BOX BOUNDS pp pp pp\n";

  std::ios::fmtflags saved = out.flags();
  const auto saved_precision = out.precision();
  out << std::setprecision(16) << std::scientific;
  out << box_.xlo << ' ' << box_.xhi << '\n';
  out << box_.ylo << ' ' << box_.yhi << '\n';
  out << box_.zlo << ' ' << box_.zhi << '\n';
  out << "ITEM: ATOMS id type x y z fx fy fz\n";

  // Emit rows in id-ascending order. AtomSoA does not guarantee stable
  // ordering after `compute_stable_reorder` — bins regroup atoms by spatial
  // cell, which is incompatible with LAMMPS's per-atom id. We sort a small
  // index vector and walk it; allocating a size-n std::vector once per dump
  // is acceptable (dump is a rare, off-hot-path operation).
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return atoms_.id[a] < atoms_.id[b];
  });

  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t i = order[k];
    // LAMMPS prints type as 1-based, id is already 1-based in AtomSoA (see
    // io/lammps_data_reader.cpp — ids are read verbatim from the data file,
    // and LAMMPS atom ids start at 1). Species `type` is 0-based internally;
    // +1 on the wire for LAMMPS compatibility.
    out << atoms_.id[i] << ' ' << (static_cast<std::uint32_t>(atoms_.type[i]) + 1U) << ' '
        << atoms_.x[i] << ' ' << atoms_.y[i] << ' ' << atoms_.z[i] << ' ' << atoms_.fx[i] << ' '
        << atoms_.fy[i] << ' ' << atoms_.fz[i] << '\n';
  }
  out.flags(saved);
  out.precision(saved_precision);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void SimulationEngine::zero_forces() {
  const std::size_t n = atoms_.size();
  for (std::size_t i = 0; i < n; ++i) {
    atoms_.fx[i] = 0.0;
    atoms_.fy[i] = 0.0;
    atoms_.fz[i] = 0.0;
  }
}

void SimulationEngine::rebuild_neighbors() {
  // Re-bin + optional reorder. A reorder invalidates the velocity-Verlet
  // intermediate state only if positions are observationally swapped —
  // `apply_reorder` permutes every SoA field including the half-stepped
  // velocities, so the integrator's next step is coherent.
  cell_grid_.bin(atoms_);
  if (auto reorder = cell_grid_.compute_stable_reorder(atoms_); !reorder.empty()) {
    apply_reorder(atoms_, reorder);
    cell_grid_.bin(atoms_);
  }
  neighbor_list_.build(atoms_, box_, cell_grid_, cutoff_, skin_);
  displacement_tracker_.reset(atoms_);
}

void SimulationEngine::recompute_forces() {
  telemetry::ScopedSection pair(telemetry_, "Pair");
  zero_forces();
  const auto result = potential_->compute(atoms_, neighbor_list_, box_);
  last_potential_energy_ = result.potential_energy;
  last_virial_ = result.virial;
}

ThermoRow SimulationEngine::snapshot_thermo(std::uint64_t step) const {
  ThermoRow row{};
  row.step = step;
  row.potential_energy = last_potential_energy_;
  row.kinetic_energy = kinetic_energy(atoms_, species_);
  row.total_energy = row.potential_energy + row.kinetic_energy;

  // Temperature with DOF = 3N−3 (three subtracted for the conserved COM
  // momentum, LAMMPS's default). T1.11 confirms this matches the oracle's
  // "temp" column when the initial configuration has zero COM velocity (the
  // `velocity create ... loop geom` default). Single-atom edge case: fall back
  // to DOF = 3 to avoid division by zero.
  const std::size_t n = atoms_.size();
  if (n > 0) {
    const double dof = n > 1 ? (3.0 * static_cast<double>(n) - 3.0) : 3.0;
    row.temperature_K = 2.0 * row.kinetic_energy / (dof * kBoltzmann_eV_per_K);
  }

  // Pressure from the Clausius virial theorem, expressed in native metal
  // units (eV/Å³):
  //   P = (2·KE − Σα W_αα) / (3·V)
  // where the potentials module stores the pair-virial Σ_pairs F_i_α · r_ij_β
  // with r_ij = r_j − r_i (see potentials/morse.cpp line comment for the
  // convention). The atomic virial that enters the Clausius form is
  // Σ_i r_i·F_i = −Σ_pairs r_ij · F_i, hence the minus sign here. This matches
  // the LAMMPS "press" column bar-for-bar (modulo the trivial bar↔eV/Å³
  // factor the T1.11 harness applies).
  const double volume = (box_.xhi - box_.xlo) * (box_.yhi - box_.ylo) * (box_.zhi - box_.zlo);
  if (volume > 0.0) {
    const double virial_trace = last_virial_[0] + last_virial_[1] + last_virial_[2];
    row.pressure_ev_A3 = (2.0 * row.kinetic_energy - virial_trace) / (3.0 * volume);
  }

  return row;
}

}  // namespace tdmd
