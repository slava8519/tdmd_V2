# tdmd::neighbor — cell grid and neighbor list management.
# SPEC: docs/specs/neighbor/SPEC.md
# Exec pack: docs/development/m6_execution_pack.md T6.4 (GpuNeighborBuilder adapter)

add_library(tdmd_neighbor STATIC cell_grid.cpp neighbor_list.cpp displacement_tracker.cpp
                                 gpu_neighbor_builder.cpp)
add_library(tdmd::neighbor ALIAS tdmd_neighbor)

target_include_directories(
  tdmd_neighbor PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                       $<INSTALL_INTERFACE:include>)

target_link_libraries(tdmd_neighbor PUBLIC tdmd::state tdmd::gpu)

target_compile_features(tdmd_neighbor PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_neighbor)
