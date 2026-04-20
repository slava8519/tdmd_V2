#include "tdmd/runtime/simulation_engine.hpp"

#include "tdmd/comm/comm_backend.hpp"
#include "tdmd/integrator/gpu_velocity_verlet.hpp"
#include "tdmd/io/lammps_data_reader.hpp"
#include "tdmd/io/yaml_config.hpp"
#include "tdmd/potentials/eam_alloy.hpp"
#include "tdmd/potentials/eam_alloy_gpu_adapter.hpp"
#include "tdmd/potentials/eam_file.hpp"
#include "tdmd/potentials/morse.hpp"
#include "tdmd/potentials/snap.hpp"
#include "tdmd/potentials/snap_file.hpp"
#include "tdmd/runtime/gpu_context.hpp"
#include "tdmd/runtime/physical_constants.hpp"
#include "tdmd/runtime/unit_converter.hpp"
#include "tdmd/scheduler/causal_wavefront_scheduler.hpp"
#include "tdmd/scheduler/certificate_input_source.hpp"
#include "tdmd/scheduler/concrete_outer_sd_coordinator.hpp"
#include "tdmd/scheduler/policy.hpp"
#include "tdmd/scheduler/subdomain_grid.hpp"
#include "tdmd/state/lj_reference.hpp"
#include "tdmd/telemetry/telemetry.hpp"
#include "tdmd/zoning/default_planner.hpp"
#include "tdmd/zoning/zoning.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
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

// T7.9 — Tile the whole simulation box into `n_sub[0]*n_sub[1]*n_sub[2]`
// equal orthogonal sub-boxes, lexicographic id layout
// `id = ix + nx*(iy + ny*iz)`. Used only by the Pattern 2 wire; equal-stride
// tiling is deliberate (no load-balancing heuristic yet — fuzzer-aware grids
// land in T7.14 / T7.11). `rank_of_subdomain` is set to the subdomain id
// (D-M7-2 single-process pinning — real multi-rank binding is T7.14).
scheduler::SubdomainGrid make_equal_tile_subdomain_grid(const Box& box,
                                                        const std::array<std::uint32_t, 3>& n_sub) {
  scheduler::SubdomainGrid g{};
  g.n_subdomains = n_sub;
  const double dx = (box.xhi - box.xlo) / static_cast<double>(n_sub[0]);
  const double dy = (box.yhi - box.ylo) / static_cast<double>(n_sub[1]);
  const double dz = (box.zhi - box.zlo) / static_cast<double>(n_sub[2]);
  const std::uint32_t total = n_sub[0] * n_sub[1] * n_sub[2];
  g.subdomain_boxes.reserve(total);
  g.rank_of_subdomain.reserve(total);
  for (std::uint32_t iz = 0; iz < n_sub[2]; ++iz) {
    for (std::uint32_t iy = 0; iy < n_sub[1]; ++iy) {
      for (std::uint32_t ix = 0; ix < n_sub[0]; ++ix) {
        Box sb{};
        sb.xlo = box.xlo + static_cast<double>(ix) * dx;
        sb.xhi = box.xlo + static_cast<double>(ix + 1U) * dx;
        sb.ylo = box.ylo + static_cast<double>(iy) * dy;
        sb.yhi = box.ylo + static_cast<double>(iy + 1U) * dy;
        sb.zlo = box.zlo + static_cast<double>(iz) * dz;
        sb.zhi = box.zlo + static_cast<double>(iz + 1U) * dz;
        g.subdomain_boxes.push_back(sb);
        const std::uint32_t sd_id = ix + n_sub[0] * (iy + n_sub[1] * iz);
        g.rank_of_subdomain.push_back(static_cast<int>(sd_id));
      }
    }
  }
  return g;
}

// T4.9 — CertificateInputSource stub for K=1 single-rank TD mode.
//
// In K=1 the scheduler runs all zones through their full lifecycle in
// lockstep, advancing once per physics step. We don't gate dispatch on
// per-zone safety (the legacy path is what actually produces forces;
// scheduler is an observer). Populate conservative always-safe inputs
// so select_ready_tasks promotes every ResidentPrev zone to Ready
// every step. Multi-rank Pattern 2 (M7+) will replace this with a
// state-+-neighbor-backed adapter that computes per-zone v_max / a_max
// / skin_remaining from live state.
class AlwaysSafeCertificateInputSource final : public scheduler::CertificateInputSource {
public:
  void fill_inputs(scheduler::ZoneId zone,
                   scheduler::TimeLevel time_level,
                   scheduler::CertificateInputs& out) const override {
    out.zone_id = zone;
    out.time_level = time_level;
    out.v_max_zone = 0.0;
    out.a_max_zone = 0.0;
    out.dt_candidate = 1.0;    // unused in K=1 (scheduler doesn't dispatch)
    out.buffer_width = 1.0e6;  // generous margins → always safe
    out.skin_remaining = 1.0e6;
    out.frontier_margin = 1.0e6;
    out.neighbor_valid_until_step = time_level + 1'000'000;
    out.halo_valid_until_step = time_level + 1'000'000;
  }
};

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
    case io::PotentialStyle::Snap: {
      if (is_lj) {
        throw std::invalid_argument(
            "potential.style=snap is incompatible with simulation.units=lj "
            "(SNAP coefficients are dimensional, metal units only)");
      }
      const std::string coeff_path =
          resolve_atoms_path(config.potential.snap.coeff_file, config_dir);
      const std::string param_path =
          resolve_atoms_path(config.potential.snap.param_file, config_dir);
      potentials::SnapData snap = potentials::parse_snap_files(coeff_path, param_path);
      potential_ = std::make_unique<SnapPotential>(std::move(snap));
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
  displacement_tracker_.init(atoms_);

  // --- T6.7: GPU backend wiring (opt-in via runtime.backend=gpu). Build
  // the GpuContext + adapters BEFORE the initial force snapshot so
  // `recompute_forces()` dispatches through GPU from step 0. Preflight has
  // already guaranteed potential.style == eam/alloy when gpu is selected.
  gpu_backend_ = (config.runtime.backend == io::RuntimeBackendKind::Gpu);
  if (gpu_backend_) {
    tdmd::gpu::GpuConfig gpu_cfg{};  // defaults: device 0, 256 MiB warm-up
    gpu_context_ = std::make_unique<runtime::GpuContext>(gpu_cfg);
    auto* eam_cpu = dynamic_cast<EamAlloyPotential*>(potential_.get());
    if (eam_cpu == nullptr) {
      throw std::invalid_argument(
          "SimulationEngine: runtime.backend=gpu requires EAM/alloy potential (T6.7 scope)");
    }
    gpu_potential_ = std::make_unique<potentials::EamAlloyGpuAdapter>(eam_cpu->data());
    gpu_integrator_ = std::make_unique<GpuVelocityVerletIntegrator>(species_);
  }

  // --- Initial force / energy / virial snapshot.
  recompute_forces();

  // --- T4.9: opt-in TD scheduler wiring (D-M4-11). Legacy path is the
  // default; td_mode_ stays false and td_scheduler_ stays null when not
  // requested by YAML.
  td_mode_ = config.scheduler.td_mode;
  td_pipeline_depth_cap_ = config.scheduler.pipeline_depth_cap;
  switch (config.zoning.scheme) {
    case io::ZoningSchemeKind::Auto:
      td_zoning_scheme_override_ = ZoningSchemeOverride::Auto;
      break;
    case io::ZoningSchemeKind::Hilbert:
      td_zoning_scheme_override_ = ZoningSchemeOverride::Hilbert;
      break;
    case io::ZoningSchemeKind::Linear1D:
      td_zoning_scheme_override_ = ZoningSchemeOverride::Linear1D;
      break;
  }
  if (td_mode_) {
    td_initialize_scheduler();
  }

  // T7.9 — Pattern 2 wire. Opt-in via `zoning.subdomains` (product ≥ 2).
  // Preflight has already rejected any axis == 0; `[1,1,1]` (default)
  // skips this branch so Pattern 1 byte-exact regression is preserved.
  //
  // The coordinator is always constructed (regardless of td_mode_) so
  // tests can assert engine ownership independent of the TD path; it
  // only attaches to the inner scheduler when td_mode_ actually produced
  // one, keeping the legacy-loop path undisturbed.
  //
  // CLI-level HybridBackend construction + CUDA-aware-MPI / NCCL probe +
  // MpiHostStaging fallback chain lives in T7.14 (M7 integration smoke).
  {
    const auto& sd = config.zoning.subdomains;
    const std::uint32_t sd_total = sd[0] * sd[1] * sd[2];
    if (sd_total >= 2U) {
      auto grid = make_equal_tile_subdomain_grid(box_, sd);
      auto coord = std::make_unique<scheduler::ConcreteOuterSdCoordinator>(
          scheduler::ConcreteOuterSdCoordinator::Mode::kReference);
      coord->initialize(grid, td_pipeline_depth_cap_);
      outer_coord_ = std::move(coord);
      if (td_scheduler_ != nullptr) {
        td_scheduler_->attach_outer_coordinator(outer_coord_.get());
      }
    }
  }

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
    if (td_mode_) {
      // TD mode wraps the same physics calls in scheduler lifecycle events;
      // force / integrator reduction order is identical (D-M4-9 byte-exact).
      td_step(s, s);
    } else {
      // Pre-force: half-kick + drift. T6.7: GPU path uses the VV GPU kernel
      // (byte-equal to CPU VV under Reference flavor, verified at T6.6).
      if (gpu_backend_) {
        gpu_integrator_->pre_force_step(atoms_,
                                        dt_,
                                        gpu_context_->pool(),
                                        gpu_context_->compute_stream());
      } else {
        integrator_->pre_force_step(atoms_, species_, dt_);
      }

      // Neighbor rebuild check — based on displacements since the last build.
      {
        telemetry::ScopedSection neigh(telemetry_, "Neigh");
        displacement_tracker_.update_displacement(atoms_, box_);
        if (displacement_tracker_.skin_exceeded()) {
          displacement_tracker_.request_rebuild("skin exceeded");
        }
        if (displacement_tracker_.rebuild_pending()) {
          rebuild_neighbors();
        }
      }

      // New forces at drifted positions. `recompute_forces` already wraps the
      // pair-potential call in a "Pair" scope, so no extra wrapper here.
      recompute_forces();

      // Post-force: half-kick.
      if (gpu_backend_) {
        gpu_integrator_->post_force_step(atoms_,
                                         dt_,
                                         gpu_context_->pool(),
                                         gpu_context_->compute_stream());
      } else {
        integrator_->post_force_step(atoms_, species_, dt_);
      }
    }

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
  displacement_tracker_.execute_rebuild(atoms_);
}

void SimulationEngine::recompute_forces() {
  telemetry::ScopedSection pair(telemetry_, "Pair");
  zero_forces();
  ForceResult result{};
  if (gpu_backend_) {
    // GPU EAM adapter walks its own per-thread cell stencil — cell_grid_
    // must be freshly binned (rebuild_neighbors() keeps it so). The CPU
    // `neighbor_list_` is not consulted, but is kept current so thermo /
    // skin / restart bookkeeping remains identical to the CPU path.
    result = gpu_potential_->compute(atoms_,
                                     box_,
                                     cell_grid_,
                                     gpu_context_->pool(),
                                     gpu_context_->compute_stream());
  } else {
    result = potential_->compute(atoms_, neighbor_list_, box_);
  }
  last_potential_energy_ = result.potential_energy;
  last_virial_ = result.virial;
}

ThermoRow SimulationEngine::snapshot_thermo(std::uint64_t step) const {
  ThermoRow row{};
  row.step = step;

  // Raw local aggregates. In Option-B M5 (physics replicated across ranks)
  // each rank's local values already equal the global values; in future work
  // partitioning (T5.11+) these become genuine per-rank partials.
  double pe = last_potential_energy_;
  double ke = kinetic_energy(atoms_, species_);
  double wxx = last_virial_[0];
  double wyy = last_virial_[1];
  double wzz = last_virial_[2];

  // T5.8 — deterministic multi-rank reduction. comm/SPEC §7.2 mandates that
  // global_sum_double be a Kahan-compensated ring reduction in Reference
  // profile (forbidding raw MPI_Allreduce). Divide-before-reduce keeps the
  // contract "partial contribution + global sum → full value" stable when a
  // future zone-owned force loop replaces replicated physics: today each
  // rank contributes `x/nranks`, tomorrow each contributes its owned-zone
  // partial — the sum path is identical. Division by nranks is IEEE-754
  // exact for nranks ∈ {2, 4, 8}, preserving the K=1 P=N bit-exactness gate.
  if (comm_backend_ != nullptr && comm_backend_->nranks() > 1) {
    const double inv_nranks = 1.0 / static_cast<double>(comm_backend_->nranks());
    pe = comm_backend_->global_sum_double(pe * inv_nranks);
    ke = comm_backend_->global_sum_double(ke * inv_nranks);
    wxx = comm_backend_->global_sum_double(wxx * inv_nranks);
    wyy = comm_backend_->global_sum_double(wyy * inv_nranks);
    wzz = comm_backend_->global_sum_double(wzz * inv_nranks);
  }

  row.potential_energy = pe;
  row.kinetic_energy = ke;
  row.total_energy = pe + ke;

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
    const double virial_trace = wxx + wyy + wzz;
    row.pressure_ev_A3 = (2.0 * row.kinetic_energy - virial_trace) / (3.0 * volume);
  }

  return row;
}

void SimulationEngine::set_comm_backend(comm::CommBackend* backend) noexcept {
  comm_backend_ = backend;
}

// --- T4.9 TD-mode scheduler wiring --------------------------------------

void SimulationEngine::td_initialize_scheduler() {
  // Build a ZoningPlan from the current box/cutoff/skin. The default
  // planner selects Linear1D for nearly-cubic single-rank boxes and
  // produces a canonical_order permutation the scheduler consumes
  // verbatim (D-M4-4). 1-rank single-subdomain target (D-M4-2) — no
  // subdomain_box, no halo peers.
  // T5.9: YAML `zoning.scheme` overrides the M3 auto-select tree.
  // `Auto` preserves the M4 byte-exact default; `Hilbert` / `Linear1D`
  // dispatch through plan_with_scheme so the anchor-test can pin
  // Andreev's §2.2 1D-slab layout without touching the Hilbert M3
  // regression path.
  zoning::DefaultZoningPlanner planner;
  zoning::PerformanceHint hint{};
  hint.preferred_K_pipeline = 1;
  zoning::ZoningPlan plan;
  switch (td_zoning_scheme_override_) {
    case ZoningSchemeOverride::Auto:
      plan = planner.plan(box_, cutoff_, skin_, /*n_ranks=*/1, hint);
      break;
    case ZoningSchemeOverride::Hilbert:
      plan = planner.plan_with_scheme(box_, cutoff_, skin_, zoning::ZoningScheme::Hilbert3D, hint);
      break;
    case ZoningSchemeOverride::Linear1D:
      plan = planner.plan_with_scheme(box_, cutoff_, skin_, zoning::ZoningScheme::Linear1D, hint);
      break;
  }
  td_plan_ = std::make_unique<zoning::ZoningPlan>(std::move(plan));

  scheduler::SchedulerPolicy policy = scheduler::PolicyFactory::for_reference();
  policy.k_max_pipeline_depth = td_pipeline_depth_cap_;  // D-M5-1 (K ∈ {1,2,4,8})
  policy.max_tasks_per_iteration =                       // drain all ready zones
      static_cast<std::uint32_t>(td_plan_->total_zones());

  td_scheduler_ = std::make_unique<scheduler::CausalWavefrontScheduler>(policy);
  td_scheduler_->initialize(*td_plan_);
  td_scheduler_->attach_outer_coordinator(nullptr);  // Pattern 1 (D-M4-2)

  td_cert_source_ = std::make_unique<AlwaysSafeCertificateInputSource>();
  td_scheduler_->set_certificate_input_source(td_cert_source_.get());

  // Prime every zone into ResidentPrev at step 0. select_ready_tasks will
  // then find them selectable on the first td_step() iteration.
  const std::uint64_t total = td_plan_->total_zones();
  for (std::uint64_t z = 0; z < total; ++z) {
    td_scheduler_->on_zone_data_arrived(static_cast<scheduler::ZoneId>(z),
                                        /*step=*/0,
                                        /*version=*/0);
  }

  // Target set when run() is called — temporarily point to a large horizon
  // so finished() returns false during early event bookkeeping.
  td_scheduler_->set_target_time_level(std::numeric_limits<scheduler::TimeLevel>::max());
}

void SimulationEngine::td_step(std::uint64_t step, std::uint64_t next_version) {
  // ------------ Phase A: refresh certs + select tasks + mark_computing.
  // In K=1 with the always-safe cert source every ResidentPrev zone lands
  // in `tasks`, so after this block the scheduler has one Computing zone
  // per active zone.
  td_scheduler_->refresh_certificates();
  auto tasks = td_scheduler_->select_ready_tasks();
  for (const auto& t : tasks) {
    td_scheduler_->mark_computing(t);
  }

  // ------------ Physics — byte-for-byte identical to legacy loop. T6.7:
  // GPU backend routes integrator halves through `gpu_integrator_` (byte-
  // equal to CPU VV under Reference) and forces through `gpu_potential_`
  // (≤1e-12 rel to CPU EAM, verified at T6.5).
  if (gpu_backend_) {
    gpu_integrator_->pre_force_step(atoms_,
                                    dt_,
                                    gpu_context_->pool(),
                                    gpu_context_->compute_stream());
  } else {
    integrator_->pre_force_step(atoms_, species_, dt_);
  }
  {
    telemetry::ScopedSection neigh(telemetry_, "Neigh");
    displacement_tracker_.update_displacement(atoms_, box_);
    if (displacement_tracker_.skin_exceeded()) {
      displacement_tracker_.request_rebuild("skin exceeded");
    }
    if (displacement_tracker_.rebuild_pending()) {
      rebuild_neighbors();
    }
  }
  recompute_forces();
  if (gpu_backend_) {
    gpu_integrator_->post_force_step(atoms_,
                                     dt_,
                                     gpu_context_->pool(),
                                     gpu_context_->compute_stream());
  } else {
    integrator_->post_force_step(atoms_, species_, dt_);
  }

  // ------------ Phase A epilogue: mark_completed.
  for (const auto& t : tasks) {
    td_scheduler_->mark_completed(t);
  }

  // ------------ Phase B: commit_completed (Pattern 1 internal zones go
  // Completed → Committed directly per SPEC §6.2 bullet 2), then release
  // Committed → Empty, then re-arrive at the new step number to put every
  // zone at ResidentPrev for the next iteration. This is the "observed"
  // single-rank cycle — the scheduler sees a coherent lifecycle per step
  // without actually gating the physics work.
  td_scheduler_->commit_completed();
  td_scheduler_->release_committed();

  const std::uint64_t total = td_plan_->total_zones();
  for (std::uint64_t z = 0; z < total; ++z) {
    td_scheduler_->on_zone_data_arrived(static_cast<scheduler::ZoneId>(z),
                                        static_cast<scheduler::TimeLevel>(step),
                                        static_cast<scheduler::Version>(next_version));
  }
}

}  // namespace tdmd
