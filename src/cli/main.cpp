// SPEC: docs/specs/cli/SPEC.md §2 (CLI surface)
// Exec pack: docs/development/m1_execution_pack.md T1.9,
//            docs/development/m2_execution_pack.md T2.11
//
// `tdmd` entry point. Dispatches to per-subcommand handlers. M1 shipped `run`
// and `validate`; M2/T2.11 adds `explain`. Subcommand dispatch is hand-rolled
// so cxxopts's per-subcommand parser stays scoped to a single file.

#include "tdmd/cli/explain_command.hpp"
#include "tdmd/cli/run_command.hpp"
#include "tdmd/cli/validate_command.hpp"

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
      << "  run <config.yaml>        Run a simulation from a YAML config\n"
      << "  validate <config.yaml>   Parse + preflight a config without running\n"
      << "  explain <config.yaml>    Print analytic performance prediction (--perf)\n"
      << "  --help, -h               Print this message\n"
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

  if (cmd == "validate") {
    tdmd::cli::ValidateOptions options;
    auto parse = tdmd::cli::parse_validate_options(rest, options, std::cout);
    if (parse.help_requested) {
      return 0;
    }
    if (!parse.error.empty()) {
      std::cerr << "tdmd validate: " << parse.error << "\n\n";
      std::cerr << "Run 'tdmd validate --help' for usage.\n";
      return 2;
    }
    tdmd::cli::ValidateStreams streams{&std::cout, &std::cerr};
    try {
      return tdmd::cli::validate_command(options, streams);
    } catch (const std::exception& e) {
      std::cerr << "unexpected error: " << e.what() << '\n';
      return 1;
    }
  }

  if (cmd == "explain") {
    tdmd::cli::ExplainOptions options;
    auto parse = tdmd::cli::parse_explain_options(rest, options, std::cout);
    if (parse.help_requested) {
      return 0;
    }
    if (!parse.error.empty()) {
      std::cerr << "tdmd explain: " << parse.error << "\n\n";
      std::cerr << "Run 'tdmd explain --help' for usage.\n";
      return 2;
    }
    tdmd::cli::ExplainStreams streams{&std::cout, &std::cerr};
    try {
      return tdmd::cli::explain_command(options, streams);
    } catch (const std::exception& e) {
      std::cerr << "unexpected error: " << e.what() << '\n';
      return 1;
    }
  }

  std::cerr << "tdmd: unknown command '" << cmd << "'\n\n";
  print_top_level_usage(std::cerr);
  return 1;
}
