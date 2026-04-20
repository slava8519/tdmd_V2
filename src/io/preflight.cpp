#include "tdmd/io/preflight.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

namespace tdmd::io {

namespace {

void push(std::vector<PreflightError>& out,
          PreflightSeverity severity,
          std::string_view key_path,
          std::string message) {
  out.push_back(PreflightError{severity, std::string(key_path), std::move(message)});
}

// Guards against NaN / ±Inf in user-supplied floats. We never accept these:
// no physics field in M1 has a meaningful infinite value, and silently
// propagating NaN makes downstream failures opaque.
[[nodiscard]] bool is_finite_positive(double v) noexcept {
  return std::isfinite(v) && v > 0.0;
}

void check_simulation(const SimulationBlock& sim, std::vector<PreflightError>& out) {
  // Cross-field rules tying `units` to the `reference` block. Schema layer
  // (parse_yaml_config) accepts either shape — preflight enforces the mutual
  // constraint so users see a clear `"units=lj requires reference"` message
  // instead of a cryptic converter error deeper in the ingest path.
  if (sim.units == UnitsKind::Lj) {
    if (!sim.reference.has_value()) {
      push(out,
           PreflightSeverity::Error,
           "simulation.reference",
           "units=lj requires a simulation.reference block with { sigma, "
           "epsilon, mass } (all > 0)");
      return;
    }
    const auto& r = sim.reference.value();
    if (!is_finite_positive(r.sigma)) {
      push(out,
           PreflightSeverity::Error,
           "simulation.reference.sigma",
           "simulation.reference.sigma must be finite and > 0 (got " + std::to_string(r.sigma) +
               ")");
    }
    if (!is_finite_positive(r.epsilon)) {
      push(out,
           PreflightSeverity::Error,
           "simulation.reference.epsilon",
           "simulation.reference.epsilon must be finite and > 0 (got " + std::to_string(r.epsilon) +
               ")");
    }
    if (!is_finite_positive(r.mass)) {
      push(out,
           PreflightSeverity::Error,
           "simulation.reference.mass",
           "simulation.reference.mass must be finite and > 0 (got " + std::to_string(r.mass) + ")");
    }
  } else {  // Metal (only other supported kind in M2).
    if (sim.reference.has_value()) {
      push(out,
           PreflightSeverity::Warning,
           "simulation.reference",
           "simulation.reference is ignored when units=metal — remove it or switch units to lj");
    }
  }
}

void check_atoms(const AtomsBlock& atoms, std::vector<PreflightError>& out) {
  namespace fs = std::filesystem;
  // `source` is always `lammps_data` after parse (M1 enum). Only the file
  // itself can fail here.
  if (atoms.path.empty()) {
    // parse_yaml_config already enforces non-empty; keep as a defensive catch
    // so a directly constructed YamlConfig does not sneak past preflight.
    push(out, PreflightSeverity::Error, "atoms.path", "atoms.path is empty");
    return;
  }
  std::error_code ec;
  const bool exists = fs::exists(atoms.path, ec);
  if (ec || !exists) {
    push(out,
         PreflightSeverity::Error,
         "atoms.path",
         "file '" + atoms.path + "' does not exist or is not readable");
    return;
  }
  const bool is_regular = fs::is_regular_file(atoms.path, ec);
  if (ec || !is_regular) {
    push(out,
         PreflightSeverity::Error,
         "atoms.path",
         "atoms.path '" + atoms.path + "' is not a regular file");
  }
}

void check_potential(const PotentialBlock& pot, std::vector<PreflightError>& out) {
  switch (pot.style) {
    case PotentialStyle::Morse: {
      const auto& m = pot.morse;
      if (!is_finite_positive(m.D)) {
        push(out,
             PreflightSeverity::Error,
             "potential.params.D",
             "Morse well depth D must be finite and > 0 (got " + std::to_string(m.D) + ")");
      }
      if (!is_finite_positive(m.alpha)) {
        push(out,
             PreflightSeverity::Error,
             "potential.params.alpha",
             "Morse alpha must be finite and > 0 (got " + std::to_string(m.alpha) + ")");
      }
      if (!is_finite_positive(m.r0)) {
        push(out,
             PreflightSeverity::Error,
             "potential.params.r0",
             "Morse equilibrium distance r0 must be finite and > 0 (got " + std::to_string(m.r0) +
                 ")");
      }
      if (!is_finite_positive(m.cutoff)) {
        push(out,
             PreflightSeverity::Error,
             "potential.params.cutoff",
             "Morse cutoff must be finite and > 0 (got " + std::to_string(m.cutoff) + ")");
      } else if (std::isfinite(m.r0) && m.cutoff <= m.r0) {
        // Only meaningful when both are finite; avoids a spurious second error when
        // r0 was already rejected as non-finite.
        push(out,
             PreflightSeverity::Error,
             "potential.params.cutoff",
             "Morse cutoff (" + std::to_string(m.cutoff) + ") must be strictly greater than r0 (" +
                 std::to_string(m.r0) + ")");
      }
      break;
    }
    case PotentialStyle::EamAlloy: {
      const auto& ea = pot.eam_alloy;
      if (ea.file.empty()) {
        push(out,
             PreflightSeverity::Error,
             "potential.params.file",
             "EAM/alloy potential.params.file must not be empty");
        break;
      }
      // We do not parse the setfl file here (the parser is in the potentials
      // module and pulls in ~200 KB of text-scanning code); just verify the
      // file is reachable. Format-level errors surface at SimulationEngine::init
      // with a path:line diagnostic via parse_eam_alloy().
      if (!std::filesystem::is_regular_file(ea.file)) {
        push(out,
             PreflightSeverity::Error,
             "potential.params.file",
             "EAM/alloy potential.params.file '" + ea.file + "' is not a regular file");
      }
      break;
    }
    case PotentialStyle::Snap: {
      const auto& sp = pot.snap;
      if (sp.coeff_file.empty()) {
        push(out,
             PreflightSeverity::Error,
             "potential.params.coeff_file",
             "SNAP potential.params.coeff_file must not be empty");
      } else if (!std::filesystem::is_regular_file(sp.coeff_file)) {
        push(out,
             PreflightSeverity::Error,
             "potential.params.coeff_file",
             "SNAP potential.params.coeff_file '" + sp.coeff_file + "' is not a regular file");
      }
      if (sp.param_file.empty()) {
        push(out,
             PreflightSeverity::Error,
             "potential.params.param_file",
             "SNAP potential.params.param_file must not be empty");
      } else if (!std::filesystem::is_regular_file(sp.param_file)) {
        push(out,
             PreflightSeverity::Error,
             "potential.params.param_file",
             "SNAP potential.params.param_file '" + sp.param_file + "' is not a regular file");
      }
      break;
    }
  }
}

void check_integrator(const IntegratorBlock& integ, std::vector<PreflightError>& out) {
  if (!is_finite_positive(integ.dt)) {
    push(out,
         PreflightSeverity::Error,
         "integrator.dt",
         "integrator.dt must be finite and > 0 (got " + std::to_string(integ.dt) + ")");
  }
}

void check_neighbor(const NeighborBlock& nb, std::vector<PreflightError>& out) {
  if (!is_finite_positive(nb.skin)) {
    push(out,
         PreflightSeverity::Error,
         "neighbor.skin",
         "neighbor.skin must be finite and > 0 (got " + std::to_string(nb.skin) + ")");
  }
}

void check_run(const RunBlock& run, std::vector<PreflightError>& out) {
  if (run.n_steps == 0) {
    push(out,
         PreflightSeverity::Error,
         "run.n_steps",
         "run.n_steps must be >= 1 (got 0 — nothing to simulate)");
  }
}

// T6.7 — GPU backend potential compatibility. M6 wires EAM/alloy on GPU only
// (Morse stays CPU-only per exec pack). Build-time CUDA availability and
// runtime device probing are checked at SimulationEngine::init() — preflight
// only enforces the YAML-schema-level invariants and does not pull a CUDA dep.
void check_runtime(const RuntimeBlock& rt,
                   const PotentialBlock& pot,
                   std::vector<PreflightError>& out) {
  if (rt.backend != RuntimeBackendKind::Gpu) {
    return;
  }
  if (pot.style != PotentialStyle::EamAlloy) {
    push(out,
         PreflightSeverity::Error,
         "runtime.backend",
         "runtime.backend='gpu' currently only supports potential.style=eam/alloy (M6 scope)");
  }
}

// T7.9 — Pattern 2 schema consistency. Purely YAML-level:
//   - each `zoning.subdomains` axis must be ≥ 1 (zero is meaningless);
//   - `comm.backend=hybrid` requires Pattern 2 (product ≥ 2) — it's the
//     wrong transport for single-subdomain runs;
//   - Pattern 2 without `comm.backend=hybrid` emits a warning — a scientist
//     may intentionally pin `mpi_host_staging` for debugging, but it's
//     suboptimal for multi-node halo traffic.
// The runtime-level CUDA-aware-MPI / NCCL probe + fallback chain lives in
// T7.14 (M7 integration smoke) where the actual transport is constructed.
void check_zoning_pattern2(const ZoningBlock& z,
                           const CommBlock& c,
                           std::vector<PreflightError>& out) {
  for (std::size_t i = 0; i < 3; ++i) {
    if (z.subdomains[i] == 0U) {
      const char* axis = (i == 0 ? "Nx" : (i == 1 ? "Ny" : "Nz"));
      push(out,
           PreflightSeverity::Error,
           "zoning.subdomains",
           std::string("zoning.subdomains ") + axis + " must be >= 1 (got 0)");
    }
  }
  const std::uint32_t total = z.subdomains[0] * z.subdomains[1] * z.subdomains[2];
  const bool pattern2 = total >= 2U;
  const bool hybrid_requested = (c.backend == CommBackendKind::Hybrid);
  if (hybrid_requested && !pattern2) {
    push(out,
         PreflightSeverity::Error,
         "comm.backend",
         "comm.backend='hybrid' requires zoning.subdomains product >= 2 (Pattern 2)");
  }
  if (pattern2 && !hybrid_requested) {
    push(out,
         PreflightSeverity::Warning,
         "comm.backend",
         "zoning.subdomains product >= 2 implies Pattern 2; consider "
         "comm.backend='hybrid' for multi-node (current setting retains inner-only transport)");
  }
}

}  // namespace

std::vector<PreflightError> preflight(const YamlConfig& config) {
  std::vector<PreflightError> out;
  // Ordering: simulation → atoms → potential → integrator → neighbor → run.
  // Chosen to match the top-to-bottom layout of io/SPEC §3.1 so error output
  // reads in source order. Any future block must insert at the matching
  // position.
  check_simulation(config.simulation, out);
  check_atoms(config.atoms, out);
  check_potential(config.potential, out);
  check_integrator(config.integrator, out);
  check_neighbor(config.neighbor, out);
  check_run(config.run, out);
  check_runtime(config.runtime, config.potential, out);
  check_zoning_pattern2(config.zoning, config.comm, out);
  return out;
}

bool preflight_passes(const std::vector<PreflightError>& errors) noexcept {
  return std::none_of(errors.begin(), errors.end(), [](const PreflightError& e) {
    return e.severity == PreflightSeverity::Error;
  });
}

}  // namespace tdmd::io
