# tdmd::runtime — lifecycle / orchestration helpers (UnitConverter, SimulationEngine).
# SPEC: docs/specs/runtime/SPEC.md

add_library(tdmd_runtime STATIC unit_converter.cpp)
add_library(tdmd::runtime ALIAS tdmd_runtime)

target_include_directories(
  tdmd_runtime PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:include>)

target_compile_features(tdmd_runtime PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_runtime)
