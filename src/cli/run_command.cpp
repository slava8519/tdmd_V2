#include "tdmd/cli/run_command.hpp"

#include "tdmd/io/preflight.hpp"
#include "tdmd/io/yaml_config.hpp"
#include "tdmd/runtime/simulation_engine.hpp"
#include "tdmd/telemetry/telemetry.hpp"

#include <cstddef>
#include <cxxopts.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TDMD_HAVE_MPI
#include "tdmd/comm/comm_backend.hpp"
#include "tdmd/comm/comm_config.hpp"
#include "tdmd/comm/mpi_host_staging_backend.hpp"
#include "tdmd/comm/ring_backend.hpp"

#include <mpi.h>
#endif

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
      ("dump", "After the final step, write a LAMMPS-compatible per-atom "
               "dump (id type x y z fx fy fz) to <file>",
          cxxopts::value<std::string>())
      ("quiet", "Suppress non-thermo stdout messages",
          cxxopts::value<bool>()->default_value("false"))
      ("timing", "Print a LAMMPS-format timing breakdown to stderr at "
                 "end-of-run (telemetry/SPEC §4.2)",
          cxxopts::value<bool>()->default_value("false"))
      ("telemetry-jsonl", "Write a one-line JSONL telemetry snapshot to "
                          "<file> at end-of-run",
          cxxopts::value<std::string>())
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
  if (parsed.count("dump") > 0) {
    out_options.dump_path = parsed["dump"].as<std::string>();
  } else {
    out_options.dump_path.clear();
  }
  out_options.quiet = parsed["quiet"].as<bool>();
  out_options.timing = parsed["timing"].as<bool>();
  if (parsed.count("telemetry-jsonl") > 0) {
    out_options.telemetry_jsonl_path = parsed["telemetry-jsonl"].as<std::string>();
  } else {
    out_options.telemetry_jsonl_path.clear();
  }

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

  // T5.8 — construct the CommBackend when built with MPI AND an outer caller
  // has MPI_Init'd (the top-level `tdmd` binary's MpiLifecycle does this; the
  // in-process test_cli harness does not). Single-rank execution (nranks == 1)
  // leaves the backend pointer nullptr so the engine's thermo reduction path
  // short-circuits and the run is bit-for-bit identical to the pre-MPI path.
  // This is how D-M5-12 is preserved: the same binary launched as `tdmd run`
  // (no mpirun) must produce the same thermo stream as before.
#ifdef TDMD_HAVE_MPI
  std::unique_ptr<comm::CommBackend> backend;
  {
    int mpi_inited = 0;
    MPI_Initialized(&mpi_inited);
    if (mpi_inited != 0) {
      if (config.comm.backend == io::CommBackendKind::Ring) {
        backend = std::make_unique<comm::RingBackend>();
      } else {
        backend = std::make_unique<comm::MpiHostStagingBackend>();
      }
      comm::CommConfig cc{};
      backend->initialize(cc);
      if (backend->nranks() > 1) {
        engine.set_comm_backend(backend.get());
      }
    }
  }
#endif

  telemetry::Telemetry telemetry;
  const bool telemetry_enabled = options.timing || !options.telemetry_jsonl_path.empty();
  try {
    // Setup (config parse, .data load, initial force warm-up) is not counted.
    // Telemetry is attached only after init() so the wall-clock window and
    // section accumulators both bracket just the run loop — matches LAMMPS's
    // `run` command timing convention.
    engine.init(config, /*config_dir=*/"");
    if (telemetry_enabled) {
      engine.set_telemetry(&telemetry);
      telemetry.begin_run();
    }
    (void) engine.run(config.run.n_steps, thermo_out);
    if (telemetry_enabled) {
      telemetry.end_run();
    }
    if (!options.dump_path.empty()) {
      std::ofstream dump_file(options.dump_path);
      if (!dump_file.is_open()) {
        err << "failed to open dump file '" << options.dump_path << "' for writing\n";
        return 1;
      }
      engine.write_dump_frame(dump_file);
    }
    engine.finalize();
  } catch (const std::exception& e) {
    err << "runtime error: " << e.what() << '\n';
#ifdef TDMD_HAVE_MPI
    if (backend) {
      backend->shutdown();
    }
#endif
    return 1;
  }

#ifdef TDMD_HAVE_MPI
  if (backend) {
    backend->barrier();
    backend->shutdown();
  }
#endif

  // --- Emit telemetry artifacts (after finalize, before the success banner).
  if (options.timing) {
    telemetry.write_lammps_format(err, config.run.n_steps, config.integrator.dt);
  }
  if (!options.telemetry_jsonl_path.empty()) {
    std::ofstream jsonl_file(options.telemetry_jsonl_path);
    if (!jsonl_file.is_open()) {
      err << "failed to open telemetry-jsonl file '" << options.telemetry_jsonl_path
          << "' for writing\n";
      return 1;
    }
    telemetry.write_jsonl(jsonl_file);
  }

  if (!options.quiet) {
    out << "tdmd run: completed " << config.run.n_steps << " steps\n";
  }
  return 0;
}

}  // namespace tdmd::cli
