# tdmd::io — LAMMPS data import (T1.3), tdmd.yaml parser + preflight (T1.4),
# + future dump writers, restart, reproducibility bundles.
# SPEC: docs/specs/io/SPEC.md

add_library(tdmd_io STATIC lammps_data_reader.cpp yaml_config.cpp preflight.cpp)
add_library(tdmd::io ALIAS tdmd_io)

target_include_directories(tdmd_io PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                          $<INSTALL_INTERFACE:include>)

# yaml-cpp is an implementation detail — keep it PRIVATE so callers never accidentally link against
# it.
target_link_libraries(
  tdmd_io
  PUBLIC tdmd::state tdmd::runtime
  PRIVATE yaml-cpp::yaml-cpp)

target_compile_features(tdmd_io PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_io)
