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
  PRIVATE cxxopts::cxxopts tdmd::comm)

target_compile_features(tdmd_cli PUBLIC cxx_std_20)

# T5.8 — CLI runs MPI_Init_thread + constructs concrete CommBackend when TDMD_ENABLE_MPI; guard the
# #ifdef surface with a compile-time define so main.cpp + run_command.cpp can branch.
if(TDMD_ENABLE_MPI)
  target_compile_definitions(tdmd_cli PUBLIC TDMD_HAVE_MPI=1)
  target_link_libraries(tdmd_cli PUBLIC MPI::MPI_CXX)
endif()

tdmd_apply_build_flavor(tdmd_cli)

# The `tdmd` executable itself is a thin main that dispatches to tdmd::cli.
add_executable(tdmd main.cpp)
target_link_libraries(tdmd PRIVATE tdmd::cli)

tdmd_apply_build_flavor(tdmd)
