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

#include <array>
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
  Snap,      // LAMMPS `pair_style snap` — .snapcoeff + .snapparam (T8.4 / T8.5).
  // EAM/FS / PACE — M9+.
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

// LAMMPS-compatible `pair_style snap` parameters. Both `coeff_file` (.snapcoeff)
// and `param_file` (.snapparam) paths resolve relative to the YAML file
// directory, matching the convention used for atoms.path and EamAlloyParams.file.
// Species ordering inside the .snapcoeff must match the LAMMPS type ordering in
// the companion .data file — SnapPotential indexes data.species[] directly via
// AtomSoA::type (M8 single-species W only; chemflag=1 deferred to M9+).
struct SnapParams {
  std::string coeff_file;
  std::string param_file;
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
  SnapParams snap{};
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

// SPEC: comm/SPEC.md §7.2 (deterministic reduction), master spec §14 M5
// Exec pack: docs/development/m5_execution_pack.md T5.8.
// Multi-rank transport selection. Empty block means "no MPI wiring" — the
// engine stays single-rank even when linked against MPI. Populated only
// when the CLI sees `comm:` in the YAML.
enum class CommBackendKind : std::uint8_t {
  MpiHostStaging,  // default — mesh topology, ANY_SOURCE receives (T5.4)
  Ring,            // ring topology (T5.5), rank r → (r+1) % P
  Hybrid,          // T7.9 — HybridBackend (comm/SPEC §6.4); Pattern 2 only.
                   // Inner/outer transport probing + fallback lands in T7.14.
};

enum class CommTopologyKind : std::uint8_t {
  Mesh,  // default — logical full-mesh, any-rank-to-any-rank
  Ring,  // ring — each rank only talks to (r-1) and (r+1)
};

struct CommBlock {
  CommBackendKind backend = CommBackendKind::MpiHostStaging;
  CommTopologyKind topology = CommTopologyKind::Mesh;
};

// SPEC: zoning/SPEC.md §3.1, master spec §13.3 (anchor-test premise).
// Exec pack: docs/development/m5_execution_pack.md T5.9.
// Opt-in zoning scheme override. `Auto` keeps the M3 behaviour — the
// DefaultZoningPlanner's §3.4 decision tree picks between Linear1D /
// Decomp2D / Hilbert3D based on box aspect ratio. `Hilbert` and
// `Linear1D` force a specific scheme via plan_with_scheme(); the anchor
// test (T5.11) uses Linear1D to reproduce Andreev §2.2 verbatim.
enum class ZoningSchemeKind : std::uint8_t {
  Auto,      // default — DefaultZoningPlanner::select_scheme()
  Hilbert,   // force ZoningScheme::Hilbert3D (M3 fallback default)
  Linear1D,  // force ZoningScheme::Linear1D (anchor-test T5.11)
};

struct ZoningBlock {
  ZoningSchemeKind scheme = ZoningSchemeKind::Auto;

  // T7.9 — Pattern 2 opt-in. `[1, 1, 1]` (default) keeps Pattern 1 byte-exact:
  // SimulationEngine skips the OuterSdCoordinator branch and existing M1..M6
  // gates pass unchanged. Any axis > 1 ⇒ product ≥ 2 ⇒ Pattern 2 (runtime/SPEC
  // §7.1), which constructs a ConcreteOuterSdCoordinator and attaches it to
  // the inner scheduler. Preflight rejects zeros and requires
  // `comm.backend=hybrid` consistency (warning when Pattern 2 runs without it).
  std::array<std::uint32_t, 3> subdomains{1U, 1U, 1U};
};

// SPEC: docs/specs/runtime/SPEC.md §2.3 (GPU backend wiring), docs/specs/gpu/
// SPEC.md §9 (engine wire-up), master spec §14 M6.
// Exec pack: docs/development/m6_execution_pack.md T6.7.
// Opt-in GPU compute path. Default `cpu` preserves M1..M5 byte-exact smokes.
// `gpu` requires the binary to have been built with `TDMD_BUILD_CUDA=ON`; the
// CLI preflight rejects the combination otherwise with a clear error.
enum class RuntimeBackendKind : std::uint8_t {
  Cpu,  // default — M1..M5 legacy path, no GPU dependency
  Gpu,  // D-M6-3 host-staging GPU; MPI transport stays MpiHostStaging
};

struct RuntimeBlock {
  RuntimeBackendKind backend = RuntimeBackendKind::Cpu;
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
  CommBlock comm{};
  ZoningBlock zoning{};
  RuntimeBlock runtime{};
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
