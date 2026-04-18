# tdmd::cli — `tdmd` executable + subcommand handlers.
# SPEC: docs/specs/cli/SPEC.md
# Exec pack: docs/development/m1_execution_pack.md T1.9

# Static library holds the subcommand logic so tests can exercise the subcommand end-to-end without
# spawning a child process.
add_library(tdmd_cli STATIC run_command.cpp validate_command.cpp explain_command.cpp field_docs.cpp)
add_library(tdmd::cli ALIAS tdmd_cli)

target_include_directories(tdmd_cli PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                           $<INSTALL_INTERFACE:include>)

target_link_libraries(
  tdmd_cli
  PUBLIC tdmd::io tdmd::runtime tdmd::perfmodel tdmd::telemetry tdmd::zoning
  PRIVATE cxxopts::cxxopts)

target_compile_features(tdmd_cli PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_cli)

# The `tdmd` executable itself is a thin main that dispatches to tdmd::cli.
add_executable(tdmd main.cpp)
target_link_libraries(tdmd PRIVATE tdmd::cli)

tdmd_apply_build_flavor(tdmd)
