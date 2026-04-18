#include "tdmd/cli/validate_command.hpp"

#include "tdmd/cli/field_docs.hpp"
#include "tdmd/io/preflight.hpp"
#include "tdmd/io/yaml_config.hpp"

#include <cstddef>
#include <cxxopts.hpp>
#include <exception>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace tdmd::cli {

namespace {

cxxopts::Options make_validate_options_spec() {
  cxxopts::Options opts("tdmd validate",
                        "Validate a TDMD YAML config (parse + preflight; no simulation)");
  // clang-format off
  opts.add_options()
      ("h,help", "Print help and exit")
      ("strict", "Treat preflight warnings as errors",
          cxxopts::value<bool>()->default_value("false"))
      ("explain", "Print a short description of a config field and exit",
          cxxopts::value<std::string>())
      ("config", "Path to tdmd YAML config",
          cxxopts::value<std::string>());
  // clang-format on
  opts.parse_positional({"config"});
  opts.positional_help("<config.yaml>");
  opts.show_positional_help();
  return opts;
}

void print_explain(std::ostream& out) {
  out << "tdmd validate --explain <field>\n\n"
      << "Recognised fields:\n";
  for (const auto& [k, _v] : config_field_descriptions()) {
    out << "  " << k << '\n';
  }
}

// Short prefix so a reader scanning `tdmd validate` output at a glance sees
// the severity without reading the rest of the line.
std::string_view severity_prefix(io::PreflightSeverity s, bool strict) {
  if (s == io::PreflightSeverity::Error) {
    return "error: ";
  }
  return strict ? "error (strict): " : "warning: ";
}

void write_summary(std::ostream& out, const io::YamlConfig& config) {
  out << "OK\n"
      << "  units:       ";
  switch (config.simulation.units) {
    case io::UnitsKind::Metal:
      out << "metal";
      break;
    case io::UnitsKind::Lj:
      out << "lj";
      if (config.simulation.reference.has_value()) {
        const auto& r = config.simulation.reference.value();
        out << " (σ=" << r.sigma << ", ε=" << r.epsilon << ", m=" << r.mass << ")";
      }
      break;
  }
  out << "\n  atoms:       "
      << (config.atoms.source == io::AtomsSource::LammpsData ? "lammps_data" : "<unknown>")
      << " from '" << config.atoms.path << "'\n"
      << "  potential:   "
      << (config.potential.style == io::PotentialStyle::Morse ? "morse" : "<unknown>") << '\n'
      << "  integrator:  "
      << (config.integrator.style == io::IntegratorStyle::VelocityVerlet ? "velocity_verlet"
                                                                         : "<unknown>")
      << " dt=" << config.integrator.dt << '\n'
      << "  neighbor:    skin=" << config.neighbor.skin << '\n'
      << "  run:         " << config.run.n_steps << " steps, thermo every " << config.thermo.every
      << '\n';
}

}  // namespace

ValidateParseResult parse_validate_options(const std::vector<std::string>& argv,
                                           ValidateOptions& out_options,
                                           std::ostream& help_out) {
  ValidateParseResult result;

  auto spec = make_validate_options_spec();

  std::vector<std::string> storage;
  storage.reserve(argv.size() + 1);
  storage.emplace_back("tdmd validate");
  for (const auto& a : argv) {
    storage.push_back(a);
  }
  std::vector<char*> cargs;
  cargs.reserve(storage.size() + 1);
  for (auto& s : storage) {
    cargs.push_back(s.data());
  }
  cargs.push_back(nullptr);
  int argc = static_cast<int>(cargs.size() - 1);

  cxxopts::ParseResult parsed;
  try {
    parsed = spec.parse(argc, cargs.data());
  } catch (const cxxopts::exceptions::exception& e) {
    result.error = std::string("argument parse error: ") + e.what();
    return result;
  }

  if (parsed.count("help") > 0) {
    help_out << spec.help();
    result.help_requested = true;
    return result;
  }

  out_options.strict = parsed["strict"].as<bool>();

  if (parsed.count("explain") > 0) {
    out_options.explain_field = parsed["explain"].as<std::string>();
    // config is optional in --explain mode; capture it if present, but don't
    // require it.
    if (parsed.count("config") > 0) {
      out_options.config_path = parsed["config"].as<std::string>();
    }
    return result;
  }

  if (parsed.count("config") == 0) {
    result.error = "missing positional argument: <config.yaml>";
    return result;
  }
  out_options.config_path = parsed["config"].as<std::string>();
  return result;
}

int validate_command(const ValidateOptions& options, const ValidateStreams& streams) {
  std::ostream& out = streams.out != nullptr ? *streams.out : std::cout;
  std::ostream& err = streams.err != nullptr ? *streams.err : std::cerr;

  // --- Explain mode: print a short field description and exit. Orthogonal to
  // validation — deliberately does not require (or consume) a config file.
  if (!options.explain_field.empty()) {
    const auto& table = config_field_descriptions();
    auto it = table.find(options.explain_field);
    if (it == table.end()) {
      err << "tdmd validate --explain: unknown field '" << options.explain_field << "'\n\n";
      print_explain(err);
      return 2;
    }
    out << options.explain_field << ":\n  " << it->second << '\n';
    return 0;
  }

  // --- Parse stage.
  io::YamlConfig config;
  try {
    config = io::parse_yaml_config(options.config_path);
  } catch (const io::YamlParseError& e) {
    err << "config parse error: " << e.what() << '\n';
    return 2;
  } catch (const std::exception& e) {
    err << "failed to read config '" << options.config_path << "': " << e.what() << '\n';
    return 1;
  }

  // --- Resolve relative atoms.path so preflight's file-existence check runs
  // against the same path the engine would see. (Validate never opens the
  // file for parsing, but preflight does stat it.)
  namespace fs = std::filesystem;
  const std::string config_dir =
      fs::path(options.config_path).parent_path().lexically_normal().string();
  if (!config.atoms.path.empty()) {
    fs::path p(config.atoms.path);
    if (!p.is_absolute() && !config_dir.empty()) {
      config.atoms.path = (fs::path(config_dir) / p).lexically_normal().string();
    }
  }

  // --- Preflight.
  const auto issues = io::preflight(config);

  std::size_t error_count = 0;
  std::size_t warning_count = 0;
  for (const auto& e : issues) {
    if (e.severity == io::PreflightSeverity::Error) {
      ++error_count;
    } else {
      ++warning_count;
    }
  }

  // Print every issue in canonical order (preflight already sorts by block).
  for (const auto& e : issues) {
    std::ostream& sink = (e.severity == io::PreflightSeverity::Error || options.strict) ? err : out;
    sink << "  - " << severity_prefix(e.severity, options.strict);
    if (!e.key_path.empty()) {
      sink << e.key_path << ": ";
    }
    sink << e.message << '\n';
  }

  const bool strict_failure = options.strict && warning_count > 0;
  if (error_count > 0 || strict_failure) {
    err << "preflight failed for '" << options.config_path << "' (" << error_count << " error"
        << (error_count == 1 ? "" : "s");
    if (warning_count > 0) {
      err << ", " << warning_count << " warning" << (warning_count == 1 ? "" : "s");
    }
    err << ")\n";
    return 2;
  }

  write_summary(out, config);
  return 0;
}

}  // namespace tdmd::cli
