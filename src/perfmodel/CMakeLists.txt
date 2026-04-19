# tdmd::perfmodel — analytic performance predictor (Pattern 1 / 3 at M2).
# SPEC: docs/specs/perfmodel/SPEC.md
# Exec pack: docs/development/m2_execution_pack.md T2.10

add_library(tdmd_perfmodel STATIC perfmodel.cpp hardware_profile.cpp gpu_cost_tables.cpp)
add_library(tdmd::perfmodel ALIAS tdmd_perfmodel)

target_include_directories(
  tdmd_perfmodel PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>)

target_compile_features(tdmd_perfmodel PUBLIC cxx_std_20)

# T7.13 — calibration fixture loader uses yaml-cpp for the external YAML file. Kept PRIVATE (same as
# src/io/) so callers never accidentally link against it.
target_link_libraries(tdmd_perfmodel PRIVATE yaml-cpp::yaml-cpp)

tdmd_apply_build_flavor(tdmd_perfmodel)
