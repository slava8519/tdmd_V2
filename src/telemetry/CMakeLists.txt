# tdmd::telemetry — M2 single-thread timing aggregator.
# SPEC: docs/specs/telemetry/SPEC.md §§3, 4.2, 6
# Exec pack: docs/development/m2_execution_pack.md T2.12
#
# Surface is deliberately narrow at M2 (begin/end section, JSONL snapshot,
# LAMMPS breakdown). NVTX, async writer, counters, histograms — all M3+/M6.

add_library(tdmd_telemetry STATIC telemetry.cpp)
add_library(tdmd::telemetry ALIAS tdmd_telemetry)

target_include_directories(
  tdmd_telemetry PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:include>)

target_compile_features(tdmd_telemetry PUBLIC cxx_std_20)

tdmd_apply_build_flavor(tdmd_telemetry)
