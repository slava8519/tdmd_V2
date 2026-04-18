// SPEC: docs/specs/cli/SPEC.md §2 (CLI surface)
// Exec pack: docs/development/m1_execution_pack.md T1.9
//
// `tdmd` entry point. M1 surface: only `tdmd run`; `tdmd validate` lands in
// T1.10. Subcommand dispatch is intentionally hand-rolled so cxxopts's
// per-subcommand parser stays scoped to a single file.

#include "tdmd/cli/run_command.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

void print_top_level_usage(std::ostream& out) {
  out << "Usage: tdmd <command> [options]\n"
      << "\n"
      << "Commands:\n"
      << "  run <config.yaml>   Run a simulation from a YAML config\n"
      << "  --help, -h          Print this message\n"
      << "\n"
      << "Run 'tdmd <command> --help' for per-command options.\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_top_level_usage(std::cerr);
    return 1;
  }

  const std::string_view cmd{argv[1]};

  if (cmd == "--help" || cmd == "-h" || cmd == "help") {
    print_top_level_usage(std::cout);
    return 0;
  }

  // Collect remaining args for the subcommand handler. We drop argv[0]
  // (program name) and argv[1] (subcommand); the subcommand parser synthesises
  // its own argv[0].
  std::vector<std::string> rest;
  rest.reserve(static_cast<std::size_t>(argc) - 2);
  for (int i = 2; i < argc; ++i) {
    rest.emplace_back(argv[i]);
  }

  if (cmd == "run") {
    tdmd::cli::RunOptions options;
    auto parse = tdmd::cli::parse_run_options(rest, options, std::cout);
    if (parse.help_requested) {
      return 0;
    }
    if (!parse.error.empty()) {
      std::cerr << "tdmd run: " << parse.error << "\n\n";
      std::cerr << "Run 'tdmd run --help' for usage.\n";
      return 2;
    }
    tdmd::cli::RunStreams streams{&std::cout, &std::cerr};
    try {
      return tdmd::cli::run_command(options, streams);
    } catch (const std::exception& e) {
      std::cerr << "unexpected error: " << e.what() << '\n';
      return 1;
    }
  }

  std::cerr << "tdmd: unknown command '" << cmd << "'\n\n";
  print_top_level_usage(std::cerr);
  return 1;
}
