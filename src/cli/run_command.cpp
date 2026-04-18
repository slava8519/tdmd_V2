#include "tdmd/cli/run_command.hpp"

#include "tdmd/io/preflight.hpp"
#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"

#include <cstddef>
#include <cxxopts.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tdmd::cli {

namespace {

// Build the cxxopts spec once — callers share it between `parse` and `--help`.
cxxopts::Options make_run_options_spec() {
  cxxopts::Options opts("tdmd run", "Run a TDMD simulation from a YAML config");
  // clang-format off
  opts.add_options()
      ("h,help", "Print help and exit")
      ("thermo", "Write thermo output to a file (default: stdout)",
          cxxopts::value<std::string>())
      ("quiet", "Suppress non-thermo stdout messages",
          cxxopts::value<bool>()->default_value("false"))
      ("config", "Path to tdmd YAML config",
          cxxopts::value<std::string>());
  // clang-format on
  opts.parse_positional({"config"});
  opts.positional_help("<config.yaml>");
  opts.show_positional_help();
  return opts;
}

std::string describe_preflight(const std::vector<io::PreflightError>& errors) {
  std::string out;
  for (const auto& e : errors) {
    out += "  - ";
    if (e.severity == io::PreflightSeverity::Error) {
      out += "error: ";
    } else {
      out += "warning: ";
    }
    if (!e.key_path.empty()) {
      out += e.key_path;
      out += ": ";
    }
    out += e.message;
    out += '\n';
  }
  return out;
}

}  // namespace

RunParseResult parse_run_options(const std::vector<std::string>& argv,
                                 RunOptions& out_options,
                                 std::ostream& help_out) {
  RunParseResult result;

  auto spec = make_run_options_spec();

  // cxxopts::parse wants an argv-like vector; build it from the caller's args.
  // argv[0] is conventionally the program name — we synthesise one so callers
  // pass the subcommand arguments only (matching `main`'s behaviour after it
  // strips the "run" dispatch token).
  std::vector<std::string> storage;
  storage.reserve(argv.size() + 1);
  storage.emplace_back("tdmd run");
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

  if (parsed.count("config") == 0) {
    result.error = "missing positional argument: <config.yaml>";
    return result;
  }
  out_options.config_path = parsed["config"].as<std::string>();

  if (parsed.count("thermo") > 0) {
    out_options.thermo_path = parsed["thermo"].as<std::string>();
  } else {
    out_options.thermo_path.clear();
  }
  out_options.quiet = parsed["quiet"].as<bool>();

  return result;
}

int run_command(const RunOptions& options, const RunStreams& streams) {
  std::ostream& out = streams.out != nullptr ? *streams.out : std::cout;
  std::ostream& err = streams.err != nullptr ? *streams.err : std::cerr;

  // --- Parse the YAML config. Syntactic / schema errors land as exceptions.
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

  // Resolve atoms path relative to the config file directory so preflight's
  // filesystem check and the engine's loader agree regardless of the user's
  // CWD. Rewriting the YamlConfig is intentional: preflight stays pure and the
  // engine receives an already-normalised config.
  namespace fs = std::filesystem;
  const std::string config_dir =
      fs::path(options.config_path).parent_path().lexically_normal().string();
  if (!config.atoms.path.empty()) {
    fs::path p(config.atoms.path);
    if (!p.is_absolute() && !config_dir.empty()) {
      config.atoms.path = (fs::path(config_dir) / p).lexically_normal().string();
    }
  }

  // --- Preflight — semantic validation; accumulates all errors.
  const auto errors = io::preflight(config);
  if (!io::preflight_passes(errors)) {
    err << "preflight failed for '" << options.config_path << "':\n" << describe_preflight(errors);
    return 2;
  }

  // --- Thermo destination: either user-supplied file or stdout.
  std::ofstream thermo_file;
  std::ostream* thermo_out = &out;
  if (!options.thermo_path.empty()) {
    thermo_file.open(options.thermo_path);
    if (!thermo_file.is_open()) {
      err << "failed to open thermo file '" << options.thermo_path << "' for writing\n";
      return 1;
    }
    thermo_out = &thermo_file;
  }

  if (!options.quiet) {
    out << "tdmd run: config=" << options.config_path << " steps=" << config.run.n_steps
        << " dt=" << config.integrator.dt << '\n';
  }

  // --- Drive the engine. Path resolution already happened above, so we hand
  // the engine an empty config_dir to prevent a second (incorrect) rewrite.
  SimulationEngine engine;
  try {
    engine.init(config, /*config_dir=*/"");
    (void) engine.run(config.run.n_steps, thermo_out);
    engine.finalize();
  } catch (const std::exception& e) {
    err << "runtime error: " << e.what() << '\n';
    return 1;
  }

  if (!options.quiet) {
    out << "tdmd run: completed " << config.run.n_steps << " steps\n";
  }
  return 0;
}

}  // namespace tdmd::cli
