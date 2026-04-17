# tdmd::neighbor — cell grid and neighbor list management.
# SPEC: docs/specs/neighbor/SPEC.md

add_library(tdmd_neighbor STATIC cell_grid.cpp)
add_library(tdmd::neighbor ALIAS tdmd_neighbor)

target_include_directories(
  tdmd_neighbor PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                       $<INSTALL_INTERFACE:include>)

target_compile_features(tdmd_neighbor PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_neighbor)
