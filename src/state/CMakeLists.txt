# tdmd::state — atom storage (AtomSoA, Box, SpeciesRegistry).
# SPEC: docs/specs/state/SPEC.md

add_library(tdmd_state STATIC atom_soa.cpp box.cpp species.cpp)
add_library(tdmd::state ALIAS tdmd_state)

target_include_directories(tdmd_state PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                             $<INSTALL_INTERFACE:include>)

target_compile_features(tdmd_state PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_state)
