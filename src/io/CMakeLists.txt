# tdmd::io — LAMMPS data import (T1.3) + future YAML config, dump writers,
# restart, reproducibility bundles.
# SPEC: docs/specs/io/SPEC.md

add_library(tdmd_io STATIC lammps_data_reader.cpp)
add_library(tdmd::io ALIAS tdmd_io)

target_include_directories(tdmd_io PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                          $<INSTALL_INTERFACE:include>)

target_link_libraries(tdmd_io PUBLIC tdmd::state tdmd::runtime)

target_compile_features(tdmd_io PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_io)
