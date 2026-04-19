# tdmd::runtime — lifecycle / orchestration helpers: UnitConverter (T1.2) +
# SimulationEngine (T1.9).
# SPEC: docs/specs/runtime/SPEC.md

add_library(tdmd_runtime STATIC unit_converter.cpp simulation_engine.cpp)
add_library(tdmd::runtime ALIAS tdmd_runtime)

target_include_directories(
  tdmd_runtime PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include>)

# SimulationEngine composes every M1 module. The dep on tdmd::io is PRIVATE because its .cpp is the
# only consumer — the public header forward-declares `io::YamlConfig` to avoid a public include
# cycle (io/ already depends on runtime/ for `tdmd::UnitSystem`). Static-lib link cycles are handled
# by CMake's automatic --start-group/--end-group grouping.
target_link_libraries(
  tdmd_runtime
  PUBLIC tdmd::state tdmd::neighbor tdmd::potentials tdmd::integrator tdmd::telemetry
  PRIVATE tdmd::io tdmd::scheduler tdmd::zoning tdmd::comm)

target_compile_features(tdmd_runtime PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_runtime)
