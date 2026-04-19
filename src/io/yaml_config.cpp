#include "tdmd/io/yaml_config.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace tdmd::io {

namespace {

// ---------------------------------------------------------------------------
// Small helpers around YAML::Node. yaml-cpp's public API is convenient but
// a few operations (line number, strict type coercion, typo detection) need
// wrapping so the parse_*() routines below read like prose.
// ---------------------------------------------------------------------------

[[nodiscard]] std::size_t line_of(const YAML::Node& node) noexcept {
  const auto mark = node.Mark();
  if (mark.line < 0) {
    return 0;
  }
  // yaml-cpp's Mark::line is 0-based; convert to the human-friendly 1-based
  // numbering we already use across the io module.
  return static_cast<std::size_t>(mark.line) + 1;
}

[[noreturn]] void throw_parse_error(const YAML::Node& node,
                                    std::string_view key_path,
                                    std::string_view message) {
  throw YamlParseError(line_of(node), key_path, message);
}

[[nodiscard]] std::string join_key(std::string_view parent, std::string_view child) {
  if (parent.empty()) {
    return std::string(child);
  }
  std::string out;
  out.reserve(parent.size() + 1 + child.size());
  out.append(parent);
  out.push_back('.');
  out.append(child);
  return out;
}

// Require `parent[key]` exists and return it; throw otherwise. `parent_path`
// is the dotted path of `parent` (e.g. `"simulation"`); `key` is appended.
[[nodiscard]] YAML::Node require_child(const YAML::Node& parent,
                                       std::string_view parent_path,
                                       const char* key) {
  if (!parent.IsMap()) {
    throw_parse_error(parent, parent_path, "expected a mapping");
  }
  YAML::Node child = parent[key];
  if (!child) {
    throw_parse_error(parent,
                      join_key(parent_path, key),
                      std::string("required key '") + key + "' is missing");
  }
  return child;
}

// Strict scalar → T converter. yaml-cpp's as<T>() already throws, but its
// messages are opaque — we rewrap them so the user sees which field failed.
template <typename T>
[[nodiscard]] T as_scalar(const YAML::Node& node, std::string_view key_path) {
  if (!node.IsScalar()) {
    throw_parse_error(node, key_path, "expected a scalar value");
  }
  try {
    return node.as<T>();
  } catch (const YAML::Exception&) {
    std::string msg = "failed to interpret value '";
    msg.append(node.Scalar());
    msg.append("' as the expected type");
    throw_parse_error(node, key_path, msg);
  }
}

// ---------------------------------------------------------------------------
// Whitelist enforcement: catch typos in recognised blocks. yaml-cpp is
// permissive by default — `simluation: ...` silently does nothing. We iterate
// the actual keys present and reject anything not in the whitelist.
// ---------------------------------------------------------------------------

void reject_unknown_keys(const YAML::Node& map,
                         std::string_view block_path,
                         std::initializer_list<std::string_view> allowed) {
  if (!map.IsMap()) {
    return;  // non-map mis-shapes are caught by the caller's require_child.
  }
  for (const auto& kv : map) {
    const std::string key = kv.first.as<std::string>();
    bool ok = false;
    for (auto a : allowed) {
      if (key == a) {
        ok = true;
        break;
      }
    }
    if (!ok) {
      std::string msg = "unknown key '";
      msg.append(key);
      msg.append("' — accepted keys in this block:");
      for (auto a : allowed) {
        msg.append(" ");
        msg.append(a);
      }
      throw_parse_error(kv.first, join_key(block_path, key), msg);
    }
  }
}

// ---------------------------------------------------------------------------
// Enum-from-string converters. Each returns the enum or throws a YamlParseError
// with an actionable list of accepted values.
// ---------------------------------------------------------------------------

UnitsKind parse_units(const YAML::Node& node, std::string_view key_path) {
  const auto raw = as_scalar<std::string>(node, key_path);
  if (raw == "metal") {
    return UnitsKind::Metal;
  }
  if (raw == "lj") {
    // M2: lj requires a sibling `reference` block (checked by preflight,
    // not here — the schema layer stays independent of cross-field rules).
    return UnitsKind::Lj;
  }
  throw_parse_error(node,
                    key_path,
                    "unsupported units literal '" + raw +
                        "' — accepted: metal, lj (real/cgs/si are not supported)");
}

// Parses a `simulation.reference: { sigma, epsilon, mass }` sub-block into
// `LjReference`. We do NOT enforce σ/ε/m > 0 here — malformed values are a
// *semantic* concern (preflight), not a schema concern, matching the Morse
// params convention where parse_morse_params only checks types.
LjReference parse_reference_block(const YAML::Node& node) {
  constexpr std::string_view path = "simulation.reference";
  reject_unknown_keys(node, path, {"sigma", "epsilon", "mass"});
  LjReference out{};
  out.sigma = as_scalar<double>(require_child(node, path, "sigma"), "simulation.reference.sigma");
  out.epsilon =
      as_scalar<double>(require_child(node, path, "epsilon"), "simulation.reference.epsilon");
  out.mass = as_scalar<double>(require_child(node, path, "mass"), "simulation.reference.mass");
  return out;
}

AtomsSource parse_atoms_source(const YAML::Node& node, std::string_view key_path) {
  const auto raw = as_scalar<std::string>(node, key_path);
  if (raw == "lammps_data") {
    return AtomsSource::LammpsData;
  }
  throw_parse_error(node,
                    key_path,
                    "unsupported atoms.source '" + raw +
                        "' — accepted: lammps_data (inline / generate land in M2)");
}

PotentialStyle parse_potential_style(const YAML::Node& node, std::string_view key_path) {
  const auto raw = as_scalar<std::string>(node, key_path);
  if (raw == "morse") {
    return PotentialStyle::Morse;
  }
  if (raw == "eam/alloy") {
    return PotentialStyle::EamAlloy;
  }
  throw_parse_error(node,
                    key_path,
                    "unsupported potential.style '" + raw +
                        "' — accepted: morse, eam/alloy (eam/fs, snap, pace arrive in M2+)");
}

MorseCutoffStrategy parse_cutoff_strategy(const YAML::Node& node, std::string_view key_path) {
  const auto raw = as_scalar<std::string>(node, key_path);
  if (raw == "shifted_force") {
    return MorseCutoffStrategy::ShiftedForce;
  }
  if (raw == "hard_cutoff") {
    return MorseCutoffStrategy::HardCutoff;
  }
  throw_parse_error(
      node,
      key_path,
      "unsupported cutoff_strategy '" + raw + "' — accepted: shifted_force (default), hard_cutoff");
}

IntegratorStyle parse_integrator_style(const YAML::Node& node, std::string_view key_path) {
  const auto raw = as_scalar<std::string>(node, key_path);
  if (raw == "velocity_verlet") {
    return IntegratorStyle::VelocityVerlet;
  }
  throw_parse_error(node,
                    key_path,
                    "unsupported integrator.style '" + raw +
                        "' — accepted: velocity_verlet (nvt, npt arrive in M9)");
}

// ---------------------------------------------------------------------------
// Per-block parsers. Each one owns its whitelist and pulls required children
// up-front so missing-key errors point at the *block*, not at the first use.
// ---------------------------------------------------------------------------

SimulationBlock parse_simulation_block(const YAML::Node& node) {
  constexpr std::string_view path = "simulation";
  reject_unknown_keys(node, path, {"units", "seed", "reference"});
  SimulationBlock out{};
  out.units = parse_units(require_child(node, path, "units"), "simulation.units");
  if (const auto seed = node["seed"]; seed) {
    out.seed = as_scalar<std::uint64_t>(seed, "simulation.seed");
  }
  if (const auto ref = node["reference"]; ref) {
    out.reference = parse_reference_block(ref);
  }
  return out;
}

AtomsBlock parse_atoms_block(const YAML::Node& node) {
  constexpr std::string_view path = "atoms";
  reject_unknown_keys(node, path, {"source", "path"});
  AtomsBlock out{};
  out.source = parse_atoms_source(require_child(node, path, "source"), "atoms.source");
  // For the M1-only literal `lammps_data`, path is required.
  out.path = as_scalar<std::string>(require_child(node, path, "path"), "atoms.path");
  if (out.path.empty()) {
    throw_parse_error(node["path"], "atoms.path", "atoms.path must not be empty");
  }
  return out;
}

MorseParams parse_morse_params(const YAML::Node& node) {
  constexpr std::string_view path = "potential.params";
  reject_unknown_keys(node, path, {"D", "alpha", "r0", "cutoff", "cutoff_strategy"});
  MorseParams out{};
  out.D = as_scalar<double>(require_child(node, path, "D"), "potential.params.D");
  out.alpha = as_scalar<double>(require_child(node, path, "alpha"), "potential.params.alpha");
  out.r0 = as_scalar<double>(require_child(node, path, "r0"), "potential.params.r0");
  out.cutoff = as_scalar<double>(require_child(node, path, "cutoff"), "potential.params.cutoff");
  if (const auto strat = node["cutoff_strategy"]; strat) {
    out.cutoff_strategy = parse_cutoff_strategy(strat, "potential.params.cutoff_strategy");
  }
  return out;
}

EamAlloyParams parse_eam_alloy_params(const YAML::Node& node) {
  constexpr std::string_view path = "potential.params";
  reject_unknown_keys(node, path, {"file"});
  EamAlloyParams out{};
  out.file = as_scalar<std::string>(require_child(node, path, "file"), "potential.params.file");
  if (out.file.empty()) {
    throw_parse_error(node["file"],
                      "potential.params.file",
                      "potential.params.file must not be empty");
  }
  return out;
}

PotentialBlock parse_potential_block(const YAML::Node& node) {
  constexpr std::string_view path = "potential";
  reject_unknown_keys(node, path, {"style", "params"});
  PotentialBlock out{};
  out.style = parse_potential_style(require_child(node, path, "style"), "potential.style");
  const auto params = require_child(node, path, "params");
  switch (out.style) {
    case PotentialStyle::Morse:
      out.morse = parse_morse_params(params);
      break;
    case PotentialStyle::EamAlloy:
      out.eam_alloy = parse_eam_alloy_params(params);
      break;
  }
  return out;
}

IntegratorBlock parse_integrator_block(const YAML::Node& node) {
  constexpr std::string_view path = "integrator";
  reject_unknown_keys(node, path, {"style", "dt"});
  IntegratorBlock out{};
  out.style = parse_integrator_style(require_child(node, path, "style"), "integrator.style");
  out.dt = as_scalar<double>(require_child(node, path, "dt"), "integrator.dt");
  return out;
}

NeighborBlock parse_neighbor_block(const YAML::Node& node) {
  constexpr std::string_view path = "neighbor";
  reject_unknown_keys(node, path, {"skin"});
  NeighborBlock out{};
  if (const auto skin = node["skin"]; skin) {
    out.skin = as_scalar<double>(skin, "neighbor.skin");
  }
  return out;
}

ThermoBlock parse_thermo_block(const YAML::Node& node) {
  constexpr std::string_view path = "thermo";
  reject_unknown_keys(node, path, {"every"});
  ThermoBlock out{};
  if (const auto every = node["every"]; every) {
    out.every = as_scalar<std::uint64_t>(every, "thermo.every");
  }
  return out;
}

RunBlock parse_run_block(const YAML::Node& node) {
  constexpr std::string_view path = "run";
  reject_unknown_keys(node, path, {"n_steps"});
  RunBlock out{};
  out.n_steps = as_scalar<std::uint64_t>(require_child(node, path, "n_steps"), "run.n_steps");
  return out;
}

SchedulerBlock parse_scheduler_block(const YAML::Node& node) {
  constexpr std::string_view path = "scheduler";
  reject_unknown_keys(node, path, {"td_mode", "pipeline_depth_cap"});
  SchedulerBlock out{};
  if (const auto td = node["td_mode"]; td) {
    out.td_mode = as_scalar<bool>(td, "scheduler.td_mode");
  }
  if (const auto k = node["pipeline_depth_cap"]; k) {
    const auto value = as_scalar<std::uint32_t>(k, "scheduler.pipeline_depth_cap");
    // D-M5-1: {1, 2, 4, 8} only. Reject at parse time so invalid configs
    // never reach scheduler::initialize — the earlier we catch K=3/5/16,
    // the less debugging the scientist has to do.
    if (value != 1u && value != 2u && value != 4u && value != 8u) {
      throw YamlParseError(
          line_of(k),
          "scheduler.pipeline_depth_cap",
          "pipeline_depth_cap must be one of {1, 2, 4, 8} (D-M5-1); got " + std::to_string(value));
    }
    out.pipeline_depth_cap = value;
  }
  return out;
}

// T5.8 — multi-rank comm backend selection. Parsed lazily; the block is
// optional and defaults to {MpiHostStaging, Mesh} — the M5 canonical
// combination. When the YAML omits `comm:` the engine runs single-rank
// (no MPI) regardless of TDMD_ENABLE_MPI.
CommBlock parse_comm_block(const YAML::Node& node) {
  constexpr std::string_view path = "comm";
  reject_unknown_keys(node, path, {"backend", "topology"});
  CommBlock out{};
  if (const auto b = node["backend"]; b) {
    const auto value = as_scalar<std::string>(b, "comm.backend");
    if (value == "mpi_host_staging") {
      out.backend = CommBackendKind::MpiHostStaging;
    } else if (value == "ring") {
      out.backend = CommBackendKind::Ring;
    } else {
      throw YamlParseError(
          line_of(b),
          "comm.backend",
          "comm.backend must be one of {mpi_host_staging, ring}; got '" + value + "'");
    }
  }
  if (const auto t = node["topology"]; t) {
    const auto value = as_scalar<std::string>(t, "comm.topology");
    if (value == "mesh") {
      out.topology = CommTopologyKind::Mesh;
    } else if (value == "ring") {
      out.topology = CommTopologyKind::Ring;
    } else {
      throw YamlParseError(line_of(t),
                           "comm.topology",
                           "comm.topology must be one of {mesh, ring}; got '" + value + "'");
    }
  }
  return out;
}

// T5.9 — opt-in zoning scheme override. When absent the M3 auto-select
// decision tree runs unchanged (Hilbert for cubic/near-cubic, Decomp2D /
// Linear1D for long aspect ratios). `linear_1d` is what the anchor test
// sets to reproduce Andreev §2.2.
ZoningBlock parse_zoning_block(const YAML::Node& node) {
  constexpr std::string_view path = "zoning";
  reject_unknown_keys(node, path, {"scheme"});
  ZoningBlock out{};
  if (const auto s = node["scheme"]; s) {
    const auto value = as_scalar<std::string>(s, "zoning.scheme");
    if (value == "auto") {
      out.scheme = ZoningSchemeKind::Auto;
    } else if (value == "hilbert") {
      out.scheme = ZoningSchemeKind::Hilbert;
    } else if (value == "linear_1d") {
      out.scheme = ZoningSchemeKind::Linear1D;
    } else {
      throw YamlParseError(
          line_of(s),
          "zoning.scheme",
          "zoning.scheme must be one of {auto, hilbert, linear_1d}; got '" + value + "'");
    }
  }
  return out;
}

// Top-level dispatch. Required blocks: simulation, atoms, potential,
// integrator, run. Optional blocks: neighbor, thermo. Any other top-level key
// is rejected so M2's new blocks land with a visible SPEC bump instead of
// silently being ignored.
YamlConfig parse_root(const YAML::Node& root) {
  if (!root.IsMap()) {
    throw YamlParseError(line_of(root),
                         "",
                         "expected the YAML root to be a mapping (key: value pairs)");
  }
  reject_unknown_keys(root,
                      "",
                      {"simulation",
                       "atoms",
                       "potential",
                       "integrator",
                       "run",
                       "neighbor",
                       "thermo",
                       "scheduler",
                       "comm",
                       "zoning"});

  YamlConfig cfg{};
  cfg.simulation = parse_simulation_block(require_child(root, "", "simulation"));
  cfg.atoms = parse_atoms_block(require_child(root, "", "atoms"));
  cfg.potential = parse_potential_block(require_child(root, "", "potential"));
  cfg.integrator = parse_integrator_block(require_child(root, "", "integrator"));
  cfg.run = parse_run_block(require_child(root, "", "run"));
  if (const auto n = root["neighbor"]; n) {
    cfg.neighbor = parse_neighbor_block(n);
  }
  if (const auto t = root["thermo"]; t) {
    cfg.thermo = parse_thermo_block(t);
  }
  if (const auto s = root["scheduler"]; s) {
    cfg.scheduler = parse_scheduler_block(s);
  }
  if (const auto c = root["comm"]; c) {
    cfg.comm = parse_comm_block(c);
  }
  if (const auto z = root["zoning"]; z) {
    cfg.zoning = parse_zoning_block(z);
  }
  return cfg;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

YamlParseError::YamlParseError(std::size_t line,
                               std::string_view key_path,
                               std::string_view message)
    : std::runtime_error([&] {
        std::string buf;
        buf.reserve(message.size() + key_path.size() + 32);
        buf.append("tdmd.yaml parse error");
        if (line != 0) {
          buf.append(" at line ");
          buf.append(std::to_string(line));
        }
        if (!key_path.empty()) {
          buf.append(" (");
          buf.append(key_path);
          buf.append(")");
        }
        buf.append(": ");
        buf.append(message);
        return buf;
      }()),
      line_(line),
      key_path_(key_path) {}

YamlConfig parse_yaml_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("tdmd.yaml parse error: cannot open '" + path + "'");
  }
  std::stringstream buf;
  buf << in.rdbuf();
  return parse_yaml_config_string(buf.str(), path);
}

YamlConfig parse_yaml_config_string(std::string_view yaml_content, std::string_view source_name) {
  YAML::Node root;
  try {
    root = YAML::Load(std::string(yaml_content));
  } catch (const YAML::ParserException& e) {
    // yaml-cpp's `mark.line` on ParserException is 0-based when valid.
    const std::size_t line = e.mark.line < 0 ? 0 : static_cast<std::size_t>(e.mark.line) + 1;
    std::string msg = "malformed YAML in ";
    msg.append(source_name);
    msg.append(": ");
    msg.append(e.msg);
    throw YamlParseError(line, "", msg);
  } catch (const YAML::Exception& e) {
    std::string msg = "yaml-cpp error in ";
    msg.append(source_name);
    msg.append(": ");
    msg.append(e.what());
    throw YamlParseError(0, "", msg);
  }
  if (!root || root.IsNull()) {
    std::string msg = "empty YAML document (";
    msg.append(source_name);
    msg.append(")");
    throw YamlParseError(0, "", msg);
  }
  return parse_root(root);
}

}  // namespace tdmd::io
