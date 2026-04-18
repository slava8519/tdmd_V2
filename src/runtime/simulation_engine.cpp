#include "tdmd/runtime/simulation_engine.hpp"

#include "tdmd/io/lammps_data_reader.hpp"
#include "tdmd/io/yaml_config.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace tdmd {

namespace {

// Boltzmann constant in metal units (eV / K). Used to convert KE → T.
// Derived from CODATA 2019 exact kB = 1.380649e-23 J/K.
constexpr double kBoltzmann_eV_per_K = 8.617333262e-5;

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

}  // namespace

SimulationEngine::SimulationEngine() = default;
SimulationEngine::~SimulationEngine() = default;

void SimulationEngine::init(const io::YamlConfig& config, const std::string& config_dir) {
  if (state_ != State::Constructed) {
    throw SimulationEngineStateError("SimulationEngine::init called twice (or after finalize)");
  }

  // --- Load atoms / box / species from the referenced LAMMPS .data file.
  const std::string atoms_path = resolve_atoms_path(config.atoms.path, config_dir);
  io::LammpsDataImportOptions opts{};
  // io::UnitSystem and tdmd::UnitSystem are the same type (runtime-owned) — only
  // Metal is supported in M1 (both parser and preflight reject `lj`).
  opts.units = UnitSystem::Metal;
  atoms_ = AtomSoA{};
  box_ = Box{};
  species_ = SpeciesRegistry{};
  (void) io::read_lammps_data_file(atoms_path, opts, atoms_, box_, species_);

  // --- Build the Morse potential from YAML.
  const auto& mp = config.potential.morse;
  MorsePotential::PairParams pp{.D = mp.D, .alpha = mp.alpha, .r0 = mp.r0, .cutoff = mp.cutoff};
  potential_ = std::make_unique<MorsePotential>(pp, to_morse_strategy(mp.cutoff_strategy));
  cutoff_ = mp.cutoff;

  // --- Build the integrator.
  integrator_ = std::make_unique<VelocityVerletIntegrator>();
  dt_ = config.integrator.dt;

  // --- Neighbor pipeline.
  skin_ = config.neighbor.skin;
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
    displacement_tracker_.update(atoms_, box_);
    if (displacement_tracker_.needs_rebuild()) {
      rebuild_neighbors();
    }

    // New forces at drifted positions.
    recompute_forces();

    // Post-force: half-kick.
    integrator_->post_force_step(atoms_, species_, dt_);

    current_step_ = s;

    if (s % thermo_every_ == 0 || s == n_steps) {
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
