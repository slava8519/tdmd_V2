# tdmd::potentials — interaction force models (Morse, EAM, MEAM, SNAP, ...).
# SPEC: docs/specs/potentials/SPEC.md
# Exec pack: docs/development/m6_execution_pack.md T6.5 (EamAlloyGpuAdapter)

add_library(
  tdmd_potentials STATIC
  morse.cpp
  tabulated.cpp
  eam_file.cpp
  eam_alloy.cpp
  eam_fs.cpp
  eam_alloy_gpu_adapter.cpp
  snap_file.cpp
  snap.cpp
  snap/sna_engine.cpp
  snap_gpu_adapter.cpp)
add_library(tdmd::potentials ALIAS tdmd_potentials)

target_include_directories(
  tdmd_potentials PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

target_link_libraries(tdmd_potentials PUBLIC tdmd::state tdmd::neighbor tdmd::gpu)

target_compile_features(tdmd_potentials PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_potentials)
