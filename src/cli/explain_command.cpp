#include "tdmd/cli/explain_command.hpp"

#include "tdmd/cli/field_docs.hpp"
#include "tdmd/io/yaml_config.hpp"
#include "tdmd/perfmodel/hardware_profile.hpp"
#include "tdmd/perfmodel/perfmodel.hpp"

#include <cstdint>
#include <cxxopts.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace tdmd::cli {

namespace {

cxxopts::Options make_explain_options_spec() {
  cxxopts::Options opts("tdmd explain", "Explain a TDMD config (M2 surface: --perf)");
  // clang-format off
  opts.add_options()
      ("h,help", "Print help and exit")
      ("perf", "Print analytic performance prediction",
          cxxopts::value<bool>()->default_value("false"))
      ("runtime", "Focus on runtime architecture (M3+, rejected on M2)",
          cxxopts::value<bool>()->default_value("false"))
      ("scheduler", "Focus on scheduler plan (M4+, rejected on M2)",
          cxxopts::value<bool>()->default_value("false"))
      ("verbose", "Show all details",
          cxxopts::value<bool>()->default_value("false"))
      ("format", "Output format: human (default), json/markdown deferred to M3+",
          cxxopts::value<std::string>()->default_value("human"))
      ("field", "Print a short description of a config field and exit",
          cxxopts::value<std::string>())
      ("config", "Path to tdmd YAML config",
          cxxopts::value<std::string>());
  // clang-format on
  opts.parse_positional({"config"});
  opts.positional_help("<config.yaml>");
  opts.show_positional_help();
  return opts;
}

// Minimal LAMMPS `.data` header peek. `--perf` needs only `N atoms`; loading
// the full atoms section would defeat the "cheap preflight" intent of the
// explain command. The parser is deliberately tolerant of extra header lines
// (atom types, box bounds, velocities) — it scans until it finds `N atoms` or
// hits the first section keyword (`Masses`, `Atoms`, …) and gives up.
//
// Recognised header shape (LAMMPS `write_data` format):
//   <title-line>                         (line 1 — always skipped)
//   [blank / # comment lines]
//   N atoms                              (the line we're after)
//   M atom types                         (NOT confused with `atoms`)
//   xlo xhi ylo yhi zlo zhi              (box bounds — skipped)
//   ...
//   Masses / Atoms / Velocities          (section headers — non-numeric, stop)
std::uint64_t read_atom_count(std::istream& in) {
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (lineno == 1) {
      // First line is the title comment per LAMMPS spec. Always skip.
      continue;
    }
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') {
      continue;  // blank or comment-only line
    }
    std::istringstream iss(line.substr(first));
    std::uint64_t n = 0;
    if (!(iss >> n)) {
      // First token isn't numeric — we've reached a section header like
      // "Masses" or "Atoms # atomic". The `N atoms` line must come before
      // any section in LAMMPS format, so there's nothing more to find.
      break;
    }
    std::string tok;
    if (!(iss >> tok)) {
      continue;  // bare integer without keyword — skip.
    }
    if (tok == "atoms") {
      // Plain `N atoms` header. Any trailing tokens (# comment, extra
      // whitespace) are ignored — only the first keyword matters.
      return n;
    }
    // Not the atoms line — `atom types`, `bonds`, box bounds, etc. Keep scanning.
  }
  throw std::runtime_error("missing 'N atoms' header line");
}

std::uint64_t read_atom_count_from_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open atoms.path '" + path + "'");
  }
  try {
    return read_atom_count(in);
  } catch (const std::exception& e) {
    throw std::runtime_error("atoms.path '" + path + "': " + e.what());
  }
}

PotentialCost pick_potential_cost(io::PotentialStyle style) {
  switch (style) {
    case io::PotentialStyle::Morse:
      return PotentialCost::morse();
    case io::PotentialStyle::EamAlloy:
      // `eam/fs` reuses the `eam_alloy` cost factory at M2 — the FLOP/pair
      // count is within ~10% of `alloy` for the same neighbor count. A
      // dedicated `PotentialCost::eam_fs()` lands with M5 when the EAM CPU
      // reference path is differential-tested.
      return PotentialCost::eam_alloy();
  }
  return PotentialCost::morse();  // unreachable — enum exhaustive on M2.
}

std::string_view style_name(io::PotentialStyle style) {
  switch (style) {
    case io::PotentialStyle::Morse:
      return "morse";
    case io::PotentialStyle::EamAlloy:
      return "eam_alloy";
  }
  return "<unknown>";
}

// TD is runnable only once the TdScheduler lands; master-spec §14 places that
// at M4 for the Morse reference and M5 for the EAM CPU reference.
std::string_view td_available_milestone(io::PotentialStyle style) {
  return (style == io::PotentialStyle::Morse) ? "M4" : "M5";
}

// Two-prediction table per perfmodel/SPEC §6.1. Formatted to stay well under
// the 40-line budget (~29 lines for Morse, ~32 with the EAM "Note:" footer).
void format_perf_output(std::ostream& out,
                        const std::string& config_path,
                        std::uint64_t n_atoms,
                        io::PotentialStyle style,
                        const HardwareProfile& hw,
                        const PotentialCost& pc,
                        const std::vector<PerfPrediction>& ranked) {
  // Preserve caller's stream format state — `std::fixed` on shared streams
  // would otherwise leak into subsequent CLI output.
  const std::ios_base::fmtflags saved_flags = out.flags();
  const std::streamsize saved_precision = out.precision();

  out << "TDMD Performance Prediction\n"
      << "===========================\n\n"
      << "Config:    " << config_path << '\n'
      << "Atoms:     " << n_atoms << '\n'
      << "Potential: " << style_name(style) << "  ("
      << "C_force = " << pc.flops_per_pair << " flops/pair × " << pc.n_neighbors_per_atom
      << " neighbors)\n\n"
      << "Hardware profile: modern_x86_64\n"
      << "  FLOPS/rank: " << hw.cpu_flops_per_sec << " FP64 ops/s\n"
      << "  Intra BW:   " << hw.intra_bw_bytes_per_sec << " B/s\n"
      << "  Inter BW:   " << hw.inter_bw_bytes_per_sec << " B/s\n"
      << "  Sched/iter: " << hw.scheduler_overhead_sec << " s\n"
      << "  Ranks:      " << hw.n_ranks << "\n\n"
      << "Predictions (ascending T_step):\n";

  out << std::fixed << std::setprecision(3);
  for (const auto& p : ranked) {
    out << "  " << p.pattern_name << ":\n"
        << "    T_step:  " << p.t_step_sec * 1.0e6 << " us\n"
        << "    K:       " << p.recommended_K << '\n'
        << "    Speedup: " << p.speedup_vs_baseline << "x\n";
  }
  out.flags(saved_flags);
  out.precision(saved_precision);

  if (style == io::PotentialStyle::EamAlloy) {
    out << "\nNote: EAM uses PotentialCost::eam_alloy() (mid-range, 115 flops/pair).\n"
        << "      Dedicated eam/fs calibration lands in M5.\n";
  }

  out << "\nCaveats:\n"
      << "  - This is an analytic estimate (perfmodel/SPEC §3). TD is not\n"
      << "    runnable for " << style_name(style) << " until " << td_available_milestone(style)
      << ".\n"
      << "  - Hardware profile is hand-curated (modern_x86_64). Auto-probe\n"
      << "    calibration lands in M4.\n"
      << "  - atom_record_size = 32 B (position + type); M9 revisits for NVT.\n";
}

void print_recognised_fields(std::ostream& out) {
  out << "Recognised fields:\n";
  for (const auto& [k, _v] : config_field_descriptions()) {
    out << "  " << k << '\n';
  }
}

}  // namespace

ExplainParseResult parse_explain_options(const std::vector<std::string>& argv,
                                         ExplainOptions& out_options,
                                         std::ostream& help_out) {
  ExplainParseResult result;
  auto spec = make_explain_options_spec();

  std::vector<std::string> storage;
  storage.reserve(argv.size() + 1);
  storage.emplace_back("tdmd explain");
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

  out_options.perf = parsed["perf"].as<bool>();
  out_options.runtime = parsed["runtime"].as<bool>();
  out_options.scheduler = parsed["scheduler"].as<bool>();
  out_options.verbose = parsed["verbose"].as<bool>();
  out_options.format = parsed["format"].as<std::string>();

  if (parsed.count("field") > 0) {
    out_options.explain_field = parsed["field"].as<std::string>();
    // Config is optional in --field mode — mirror `validate --explain`.
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

int explain_command(const ExplainOptions& options, const ExplainStreams& streams) {
  std::ostream& out = streams.out != nullptr ? *streams.out : std::cout;
  std::ostream& err = streams.err != nullptr ? *streams.err : std::cerr;

  // --- Field-documentation mode: orthogonal to perf prediction. Does not
  // require a config file.
  if (!options.explain_field.empty()) {
    const auto& table = config_field_descriptions();
    auto it = table.find(options.explain_field);
    if (it == table.end()) {
      err << "tdmd explain --field: unknown field '" << options.explain_field << "'\n\n";
      print_recognised_fields(err);
      return 2;
    }
    out << options.explain_field << ":\n  " << it->second << '\n';
    return 0;
  }

  // --- Reject M3+ focuses before the expensive parse stage. `--runtime`
  // would need a zoning plan (M3); `--scheduler` needs a materialised DAG
  // from TdScheduler (M4).
  if (options.runtime || options.scheduler) {
    err << "tdmd explain: --runtime and --scheduler focuses are deferred to M3+.\n"
        << "  On M2, use --perf for the analytic performance prediction.\n";
    return 2;
  }

  if (!options.perf) {
    err << "tdmd explain: specify --perf to print the analytic performance prediction.\n"
        << "  Other focuses (--runtime, --scheduler) land in M3+.\n";
    return 2;
  }

  // --- Format gating. The SPEC §5 canonical list is {human, json, markdown};
  // M2 ships only `human`. Accepting but rejecting the other names gives a
  // clearer error than cxxopts' "unrecognised option" would.
  if (options.format != "human") {
    err << "tdmd explain: --format '" << options.format << "' is deferred to M3+.\n"
        << "  M2 supports --format human only.\n";
    return 2;
  }

  if (options.config_path.empty()) {
    err << "tdmd explain: missing positional argument: <config.yaml>\n";
    return 2;
  }

  // --- Permissive config parse: YAML schema only, no preflight. A config
  // that fails semantic checks (dt <= 0, cutoff out of range, missing .data
  // file) still has enough info — `potential.style` and `atoms.path` — for
  // the analytic model. Schema violations (missing required keys, wrong
  // type) are still rejected because they leave the relevant fields
  // uninitialised.
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

  namespace fs = std::filesystem;
  const std::string config_dir =
      fs::path(options.config_path).parent_path().lexically_normal().string();
  std::string atoms_path = config.atoms.path;
  if (!atoms_path.empty()) {
    fs::path p(atoms_path);
    if (!p.is_absolute() && !config_dir.empty()) {
      atoms_path = (fs::path(config_dir) / p).lexically_normal().string();
    }
  }
  if (atoms_path.empty()) {
    err << "tdmd explain: atoms.path is empty in config '" << options.config_path << "'\n";
    return 2;
  }

  std::uint64_t n_atoms = 0;
  try {
    n_atoms = read_atom_count_from_file(atoms_path);
  } catch (const std::exception& e) {
    err << "tdmd explain: " << e.what() << '\n';
    return 2;
  }
  if (n_atoms == 0) {
    err << "tdmd explain: atoms.path '" << atoms_path << "' reports 0 atoms\n";
    return 2;
  }

  const auto hw = HardwareProfile::modern_x86_64();
  const auto pc = pick_potential_cost(config.potential.style);

  try {
    PerfModel model(hw, pc);
    const auto ranked = model.rank(n_atoms);
    format_perf_output(out, options.config_path, n_atoms, config.potential.style, hw, pc, ranked);
  } catch (const std::exception& e) {
    err << "tdmd explain: internal error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

}  // namespace tdmd::cli
