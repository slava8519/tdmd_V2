# tdmd::zoning — M3 spatial decomposition planner.
# SPEC: docs/specs/zoning/SPEC.md (v1.0, 2026-04-16)
# Exec pack: docs/development/m3_execution_pack.md T3.2–T3.6
#
# Three scheme implementations (Linear1D / Decomp2D / Hilbert3D) land in
# T3.3–T3.5; this CMakeLists just establishes the target and its sources.

add_library(tdmd_zoning STATIC zoning.cpp linear1d.cpp decomp2d.cpp)
add_library(tdmd::zoning ALIAS tdmd_zoning)

target_include_directories(tdmd_zoning PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                              $<INSTALL_INTERFACE:include>)

target_link_libraries(tdmd_zoning PUBLIC tdmd::state)

target_compile_features(tdmd_zoning PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_zoning)
