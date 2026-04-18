#pragma once

// SPEC: docs/specs/io/SPEC.md §3.1–§3.3 (tdmd.yaml format + preflight split)
// Exec pack: docs/development/m1_execution_pack.md T1.4
//
// Minimal M1 YAML config reader. Produces a POD `YamlConfig` that the future
// SimulationEngine (T1.9) and `tdmd validate` CLI (T1.10) consume.
//
// M1 scope (exec pack T1.4):
//   Required:  simulation.units (metal only),
//              atoms.source (lammps_data only) + atoms.path,
//              potential.style (morse only) + params {D, alpha, r0, cutoff},
//              integrator.style (velocity_verlet only) + integrator.dt,
//              run.n_steps.
//   Optional:  simulation.seed (default 12345),
//              neighbor.skin  (default 0.3 Å),
//              thermo.every   (default 100 steps),
//              potential.params.cutoff_strategy (shifted_force | hard_cutoff,
//                                                default shifted_force).
//   Deferred:  runtime / scheduler / comm / dump / checkpoint / telemetry /
//              species / box blocks — M2+ (species + box come from the
//              referenced LAMMPS `.data` file in M1).
//
// Two-stage error model:
//   - `parse_yaml_config` throws `YamlParseError` on syntactic or schema
//     violations (malformed YAML, missing required keys, wrong type, unknown
//     key at a recognised block, unsupported literal). `what()` includes the
//     yaml-cpp line number and the offending key path.
//   - Semantic validation (file-existence, dt > 0, cutoff > r0, …) lives in
//     `preflight()` (see preflight.hpp) and returns an accumulated list — we
//     deliberately do NOT fail-fast on semantics so users see every issue in
//     one pass.

#include "tdmd/state/lj_reference.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tdmd::io {

// Kept as a tagged enum (not `std::string`) so downstream code pattern-matches
// on values, not on typo-prone literals.
enum class UnitsKind : std::uint8_t {
  Metal,
  Lj,  // LAMMPS `units lj` — dimensionless relative to (σ, ε, m); requires
       // `simulation.reference`. Ingest path converts to metal internally.
};

enum class AtomsSource : std::uint8_t {
  LammpsData,
  // Inline / Generate — M2+.
};

enum class PotentialStyle : std::uint8_t {
  Morse,
  EamAlloy,  // LAMMPS `pair_style eam/alloy` — setfl file (T2.7, T2.9).
  // EAM/FS / SNAP / PACE — M2+ / M8.
};

enum class MorseCutoffStrategy : std::uint8_t {
  ShiftedForce,  // Production default (potentials/SPEC §2.4.2, Strategy C).
  HardCutoff,    // Raw (Strategy A) — used by unit tests that need F(r₀) = 0 exactly.
};

enum class IntegratorStyle : std::uint8_t {
  VelocityVerlet,  // == NVE in M1.
  // NVT / NPT — M9.
};

struct MorseParams {
  double D = 0.0;
  double alpha = 0.0;
  double r0 = 0.0;
  double cutoff = 0.0;
  MorseCutoffStrategy cutoff_strategy = MorseCutoffStrategy::ShiftedForce;
};

// LAMMPS-compatible `pair_style eam/alloy` parameters. The `file` path is
// resolved relative to the YAML file's directory (same convention as
// `atoms.path` — see runtime/simulation_engine.cpp::resolve_atoms_path).
// Species ordering inside the setfl file must match the LAMMPS type ordering
// in the companion `.data` file, since EamAlloyPotential uses `AtomSoA::type`
// directly as an index into the per-species tables (see
// potentials/eam_alloy.hpp). An explicit `species` map is deferred until a
// benchmark needs it.
struct EamAlloyParams {
  std::string file;
};

struct SimulationBlock {
  UnitsKind units = UnitsKind::Metal;
  std::uint64_t seed = 12345;
  // (σ, ε, m) reference for `units: lj`. Parser populates this iff the user
  // supplied a `simulation.reference` block; preflight rejects its absence
  // when `units=lj` and warns about its presence when `units=metal`.
  std::optional<LjReference> reference;
};

struct AtomsBlock {
  AtomsSource source = AtomsSource::LammpsData;
  std::string path;
};

struct PotentialBlock {
  PotentialStyle style = PotentialStyle::Morse;
  MorseParams morse{};
  EamAlloyParams eam_alloy{};
};

struct IntegratorBlock {
  IntegratorStyle style = IntegratorStyle::VelocityVerlet;
  double dt = 0.0;  // ps (metal); required field, zero default caught by preflight.
};

struct NeighborBlock {
  double skin = 0.3;  // Å; matches T1.6 default.
};

struct ThermoBlock {
  std::uint64_t every = 100;
};

struct RunBlock {
  std::uint64_t n_steps = 0;  // required; zero default caught by preflight.
};

// SPEC: scheduler/SPEC.md §10, master spec §14 M4 (D-M4-11).
// Opt-in TD scheduler wiring. Default `td_mode=false` keeps M1/M2/M3 smokes
// and regression gates running on the legacy NVE loop bit-for-bit; when set
// true, SimulationEngine::run() drives the K=1 single-rank CausalWavefront
// scheduler around the same physics calls, byte-exact thermo expected
// (D-M4-9 acceptance gate).
struct SchedulerBlock {
  bool td_mode = false;

  // D-M5-1: pipeline depth cap K ∈ {1, 2, 4, 8}. Default 1 preserves M4
  // byte-exact regression (D-M5-12) when the YAML key is absent. The parser
  // rejects any other value with a line-numbered error.
  std::uint32_t pipeline_depth_cap = 1;
};

// Aggregate. Parser produces this, preflight inspects it.
struct YamlConfig {
  SimulationBlock simulation{};
  AtomsBlock atoms{};
  PotentialBlock potential{};
  IntegratorBlock integrator{};
  NeighborBlock neighbor{};
  ThermoBlock thermo{};
  RunBlock run{};
  SchedulerBlock scheduler{};
};

// Thrown by `parse_yaml_config` on syntactic / schema violations. `line` is the
// 1-based yaml-cpp line number of the offending node (0 if unknown / file-level).
class YamlParseError : public std::runtime_error {
public:
  YamlParseError(std::size_t line, std::string_view key_path, std::string_view message);

  [[nodiscard]] std::size_t line() const noexcept { return line_; }
  [[nodiscard]] const std::string& key_path() const noexcept { return key_path_; }

private:
  std::size_t line_;
  std::string key_path_;
};

// Parses `path` (must exist and be readable). Returns a fully populated
// `YamlConfig` with defaults applied for optional fields. Throws
// `YamlParseError` on any syntactic / schema violation, or `std::runtime_error`
// if the file cannot be opened.
[[nodiscard]] YamlConfig parse_yaml_config(const std::string& path);

// Stream-based overload; primarily for tests that construct YAML inline. `source_name`
// is used only to label errors (e.g. `"<inline>"` or a filename) — it has no side effects.
[[nodiscard]] YamlConfig parse_yaml_config_string(std::string_view yaml_content,
                                                  std::string_view source_name = "<string>");

}  // namespace tdmd::io
