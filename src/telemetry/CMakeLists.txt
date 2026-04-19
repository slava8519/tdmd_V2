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

# T6.11 NVTX bridge — include/tdmd/telemetry/nvtx.hpp pulls in CUDA's <nvtx3/nvtx3.hpp> when
# TDMD_BUILD_CUDA is defined, so propagate the CUDA toolkit include path + define publicly. nvtx3 is
# header-only and symbols resolve through libnvToolsExt at runtime via CUDA::nvToolsExt linked by
# consumers that actually use ranges (tdmd_gpu). This keeps tdmd_telemetry itself buildable on
# CPU-only matrix without a CUDA link edge.
if(TDMD_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_include_directories(tdmd_telemetry PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(tdmd_telemetry PUBLIC TDMD_BUILD_CUDA=1)
endif()

tdmd_apply_build_flavor(tdmd_telemetry)
