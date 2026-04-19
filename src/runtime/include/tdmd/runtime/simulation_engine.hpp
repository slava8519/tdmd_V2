#pragma once

// SPEC: docs/specs/runtime/SPEC.md §2.2 (SimulationEngine lifecycle),
//       master spec §8.4 (single orchestration point)
// Exec pack: docs/development/m1_execution_pack.md T1.9
//
// Single orchestration point for a TDMD run. In M1 this is a single-rank,
// single-subdomain, NVE-only driver — no TD scheduler, no MPI, no GPU.
// The full lifecycle in M1 is:
//
//   SimulationEngine engine;
//   engine.init(config);      // parse .data, bind state + policies, warm forces
//   engine.run(n_steps, out); // VV integrator loop, periodic thermo rows
//   engine.finalize();        // no-op in M1; reserved for dump flush / restart
//
// Ownership (master spec §8.2): the engine owns AtomSoA, Box, SpeciesRegistry,
// the CellGrid / NeighborList / DisplacementTracker trio, and the potential /
// integrator instances. Everything else borrows via const reference.
//
// io/YamlConfig is forward-declared here to avoid a runtime↔io include cycle —
// io/lammps_data_reader.hpp pulls `tdmd::UnitSystem` from this module, so
// this header keeps its io dependency opaque and the cpp pulls the real type.

#include "tdmd/integrator/velocity_verlet.hpp"
#include "tdmd/neighbor/cell_grid.hpp"
#include "tdmd/neighbor/displacement_tracker.hpp"
#include "tdmd/neighbor/neighbor_list.hpp"
#include "tdmd/potentials/potential.hpp"
#include "tdmd/state/atom_soa.hpp"
#include "tdmd/state/box.hpp"
#include "tdmd/state/species.hpp"
#include "tdmd/telemetry/telemetry.hpp"

#include <array>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string>

// Forward declarations for T4.9 TD-mode wiring. The scheduler + zoning
// dependency is private to simulation_engine.cpp; this header stays lean.
namespace tdmd::scheduler {
class CausalWavefrontScheduler;
class CertificateInputSource;
// T7.9 — Pattern 2 outer-level coordinator (master spec §12.7a). Held as a
// borrowed-from-base unique_ptr so the concrete subclass stays in the .cpp.
class OuterSdCoordinator;
}  // namespace tdmd::scheduler

namespace tdmd::zoning {
struct ZoningPlan;
}  // namespace tdmd::zoning

namespace tdmd::io {
struct YamlConfig;
}  // namespace tdmd::io

// T5.8 — multi-rank transport. The engine only holds a pointer, so a full
// include is unnecessary here and keeps the runtime/comm dependency private
// to the .cpp (runtime already pulls comm transitively via scheduler).
namespace tdmd::comm {
class CommBackend;
}  // namespace tdmd::comm

// T6.7 — GPU backend wiring. The GpuContext + potentials/integrator GPU
// adapters are allocated lazily (`runtime.backend: gpu`), so the engine
// header forward-declares and the .cpp includes the real types. On CPU-only
// builds the pointers stay nullptr and the code paths guarding them never
// fire.
namespace tdmd::runtime {
class GpuContext;
}  // namespace tdmd::runtime

namespace tdmd {
class GpuVelocityVerletIntegrator;
}  // namespace tdmd

namespace tdmd::potentials {
class EamAlloyGpuAdapter;
}  // namespace tdmd::potentials

namespace tdmd {

// Thrown by `init` when the engine is already initialised (double-init) or by
// `run` / `finalize` when they are invoked out of order. The master spec §2.2
// lifecycle is strict; transitions that violate it are programmer errors.
class SimulationEngineStateError : public std::logic_error {
public:
  using std::logic_error::logic_error;
};

// Single-row snapshot of the thermodynamic state — what one row of the
// `thermo` stream represents. Surfaced publicly for tests that want to
// inspect the per-step values without parsing the log format.
struct ThermoRow {
  std::uint64_t step = 0;
  double temperature_K = 0.0;     // 2·KE / (3·N·kB),  K
  double potential_energy = 0.0;  // eV
  double kinetic_energy = 0.0;    // eV
  double total_energy = 0.0;      // eV (PE + KE)
  double pressure_ev_A3 = 0.0;    // eV/Å³ (native metal — bar conv in T1.11)
};

class SimulationEngine {
public:
  SimulationEngine();
  ~SimulationEngine();

  SimulationEngine(const SimulationEngine&) = delete;
  SimulationEngine& operator=(const SimulationEngine&) = delete;
  SimulationEngine(SimulationEngine&&) = delete;
  SimulationEngine& operator=(SimulationEngine&&) = delete;

  // Load state + build policies from a parsed `YamlConfig`. Throws
  // `SimulationEngineStateError` if already initialised, or propagates
  // `LammpsDataParseError` / `std::invalid_argument` if the underlying data
  // file / potential params are bad.
  //
  // The yaml-cpp config's `atoms.path` is resolved relative to `config_dir`
  // when non-empty; the default empty string interprets the path verbatim.
  void init(const io::YamlConfig& config, const std::string& config_dir = "");

  // Run the N-step Velocity-Verlet loop. `thermo_out`, if non-null, receives
  // one whitespace-separated row per `thermo_every` steps (inclusive of step
  // 0 and the final step). Returns the final ThermoRow so callers do not
  // have to re-parse the stream. Throws `SimulationEngineStateError` if the
  // engine is not yet initialised.
  ThermoRow run(std::uint64_t n_steps, std::ostream* thermo_out = nullptr);

  // M1: no persistent state to flush. Defined so the callsite matches the
  // SPEC §2.2 init → run → finalize contract.
  void finalize();

  // Optional telemetry sink. Ownership stays with the caller; pass nullptr
  // (default) to disable timing. Must be set before `run()` to take effect.
  // Single-thread only (M2); M3+ will layer a ring-buffered async sink.
  //
  // INVARIANT (LAMMPS-style run-window): attach telemetry and call
  // `Telemetry::begin_run()` *after* `init()` returns, never before. `init()`
  // warm-starts forces via an internal `recompute_forces()`; including that
  // in the measured window yields Pair > Total in tiny-step runs. See
  // `tests/integration/m2_smoke/run_nial_eam_smoke.sh` and
  // `src/cli/run_command.cpp` for the correct attachment point. Mirrors
  // LAMMPS's `run` command convention: initialisation / neighbor setup are
  // outside the timer; only the step loop is timed.
  void set_telemetry(telemetry::Telemetry* sink) noexcept { telemetry_ = sink; }

  // T5.8 — inject a multi-rank CommBackend. When the injected backend reports
  // `nranks() > 1`, snapshot_thermo() reduces PE / KE / virial across ranks
  // via `global_sum_double` (deterministic Kahan ring reduction, comm/SPEC
  // §7.2). Each rank contributes its local contribution divided by nranks;
  // the sum reproduces the scalar bit-exact when physics is replicated across
  // ranks (K=1 P=N contract). nullptr disables the reduction and restores the
  // single-rank thermo path verbatim. Ownership stays with the caller.
  //
  // The engine must be constructed before init() and the pointer held stable
  // for the lifetime of run() — the CLI layer (src/cli/run_command.cpp) owns
  // the backend and destroys it after engine.finalize() returns.
  void set_comm_backend(comm::CommBackend* backend) noexcept;

  // Read-only accessors — useful for tests that build an engine in-process
  // without invoking the CLI layer.
  [[nodiscard]] const AtomSoA& atoms() const noexcept { return atoms_; }
  [[nodiscard]] const Box& box() const noexcept { return box_; }
  [[nodiscard]] const SpeciesRegistry& species() const noexcept { return species_; }
  [[nodiscard]] bool is_initialised() const noexcept { return state_ == State::Initialised; }
  [[nodiscard]] std::uint64_t thermo_every() const noexcept { return thermo_every_; }
  [[nodiscard]] std::uint64_t current_step() const noexcept { return current_step_; }

  // T4.10 test hook: exposes the TD scheduler (or nullptr when `td_mode: false`)
  // so determinism tests can snapshot the event log across two engine runs.
  // Not part of the stable public API — keep test-only callers explicit.
  [[nodiscard]] const scheduler::CausalWavefrontScheduler* td_scheduler_for_testing()
      const noexcept {
    return td_scheduler_.get();
  }

  // T7.9 test hook: exposes the Pattern 2 outer coordinator (or nullptr when
  // `zoning.subdomains` product == 1). Used by tests/runtime/
  // test_pattern2_engine_wire.cpp to assert (a) Pattern 1 regression default
  // leaves it null, (b) opt-in Pattern 2 constructs a live coordinator. Not
  // part of the stable API.
  [[nodiscard]] const scheduler::OuterSdCoordinator* outer_sd_coordinator_for_testing()
      const noexcept {
    return outer_coord_.get();
  }

  // Emit a LAMMPS-ish header row (`# step temp pe ke etotal press`) to `out`.
  // Idempotent / free function — also used by tests that reconstruct a row
  // stream without invoking the engine.
  static void write_thermo_header(std::ostream& out);

  // Emit a single `ThermoRow` as a whitespace-separated line. Precision is
  // fixed (10 significant digits) so output is deterministic and diffable.
  static void write_thermo_row(std::ostream& out, const ThermoRow& row);

  // Emit the current simulation frame as a LAMMPS-compatible text dump:
  //
  //   ITEM: TIMESTEP
  //   <current_step>
  //   ITEM: NUMBER OF ATOMS
  //   <n>
  //   ITEM: BOX BOUNDS pp pp pp
  //   <xlo> <xhi>
  //   <ylo> <yhi>
  //   <zlo> <zhi>
  //   ITEM: ATOMS id type x y z fx fy fz
  //   1 1 ...
  //   2 1 ...
  //
  // Atom rows are emitted in id-ascending order (1-based ids, matching
  // LAMMPS). Forces are the currently-cached `atoms_.fx/fy/fz` — consumers
  // are expected to call this right after `run()` so they see post-final-
  // step forces. Used by the T2.8 DifferentialRunner; LAMMPS `rerun` can
  // consume the same file bit-for-bit.
  void write_dump_frame(std::ostream& out) const;

private:
  enum class State : std::uint8_t { Constructed, Initialised, Finalised };

  // Core simulation state (owned).
  AtomSoA atoms_{};
  Box box_{};
  SpeciesRegistry species_{};

  // Policies (owned via unique_ptr so the header stays light — potential
  // types may grow richer). `potential_` holds the abstract base; the
  // concrete subclass is selected in `init()` by `PotentialStyle`.
  std::unique_ptr<Potential> potential_;
  std::unique_ptr<VelocityVerletIntegrator> integrator_;

  // Neighbor pipeline.
  CellGrid cell_grid_{};
  NeighborList neighbor_list_{};
  DisplacementTracker displacement_tracker_{};

  // Time integration / bookkeeping.
  double dt_ = 0.0;
  double cutoff_ = 0.0;
  double skin_ = 0.0;
  std::uint64_t thermo_every_ = 100;
  std::uint64_t current_step_ = 0;

  // Cached from the most recent force evaluation so thermo rows are cheap.
  double last_potential_energy_ = 0.0;
  std::array<double, 6> last_virial_{};

  // Optional per-section timing sink. Non-owning pointer; nullptr = off.
  telemetry::Telemetry* telemetry_ = nullptr;

  // T5.8 — non-owning multi-rank transport. nullptr in single-rank / CLI
  // built without MPI; otherwise points at a backend the CLI layer owns.
  comm::CommBackend* comm_backend_ = nullptr;

  // T6.7 — GPU backend wiring. All four fields remain nullptr / false when
  // `runtime.backend: cpu` (the default), so the M1..M5 legacy path is
  // byte-exact unchanged. When `gpu` is requested, `gpu_context_` is built
  // in init() and the potential + integrator point at GPU adapters; the
  // `recompute_forces()` and `run()` loops branch on `gpu_backend_`.
  bool gpu_backend_ = false;
  std::unique_ptr<runtime::GpuContext> gpu_context_;
  std::unique_ptr<potentials::EamAlloyGpuAdapter> gpu_potential_;
  std::unique_ptr<GpuVelocityVerletIntegrator> gpu_integrator_;

  State state_ = State::Constructed;

  // M4 / T4.9 — TD scheduler wiring. Inactive (and nullptr) unless
  // `scheduler.td_mode: true` in YAML. In K=1 single-rank (D-M4-1, D-M4-6)
  // the scheduler *observes* the canonical legacy step: forces and integrator
  // are invoked identically to the legacy loop, preserving neighbor-list
  // reduction order so thermo output is byte-exact across modes (D-M4-9).
  // The scheduler transitions zones through their full lifecycle per step
  // for diagnostic + determinism signal, but does not dispatch physics.
  bool td_mode_ = false;
  // D-M5-1: K-batching depth cap from YAML (scheduler.pipeline_depth_cap);
  // {1, 2, 4, 8} only, default 1 for M4 byte-exact regression.
  std::uint32_t td_pipeline_depth_cap_ = 1;
  // T5.9: zoning scheme override from YAML (zoning.scheme). `Auto` runs
  // the M3 decision tree; explicit values bypass it via
  // DefaultZoningPlanner::plan_with_scheme — anchor-test (T5.11) pins
  // this to Linear1D to reproduce Andreev §2.2.
  enum class ZoningSchemeOverride : std::uint8_t { Auto, Hilbert, Linear1D };
  ZoningSchemeOverride td_zoning_scheme_override_ = ZoningSchemeOverride::Auto;
  std::unique_ptr<zoning::ZoningPlan> td_plan_;
  std::unique_ptr<scheduler::CausalWavefrontScheduler> td_scheduler_;
  std::unique_ptr<scheduler::CertificateInputSource> td_cert_source_;

  // T7.9 — Pattern 2 outer coordinator (master spec §12.8). nullptr whenever
  // `zoning.subdomains` product == 1 (Pattern 1 / 3 default); a live
  // ConcreteOuterSdCoordinator otherwise. Declared after td_scheduler_ so the
  // implicit reverse-order destruction tears the coord down before the
  // scheduler — the scheduler dtor is defaulted and does not dereference its
  // borrowed coord pointer, so either order would be safe, but this matches
  // runtime/SPEC §2.2 declaration order.
  std::unique_ptr<scheduler::OuterSdCoordinator> outer_coord_;

  // Internal helpers.
  void zero_forces();
  void rebuild_neighbors();
  void recompute_forces();  // zero + potential::compute, refresh cached PE/virial.
  [[nodiscard]] ThermoRow snapshot_thermo(std::uint64_t step) const;

  // T4.9: once per step in TD mode — wraps legacy physics in scheduler
  // lifecycle events (refresh / select / mark_computing / mark_completed /
  // commit / release / arrive).
  void td_step(std::uint64_t step, std::uint64_t next_version);
  void td_initialize_scheduler();
};

}  // namespace tdmd
