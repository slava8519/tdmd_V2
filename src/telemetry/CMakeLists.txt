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

  # NVTX3 C++ wrapper (nvtx3/nvtx3.hpp) ships with the NVIDIA-hosted CUDA toolkit but is stripped
  # from Ubuntu's apt `nvidia-cuda-toolkit` package (CUDA 12.0 in ubuntu-24.04). Detect presence at
  # configure time so local dev (CUDA 12.5+/13.x) keeps full Nsight instrumentation while the
  # compile-only CI matrix still builds. When absent, TDMD_NVTX_RANGE expands to a zero-cost no-op
  # per nvtx.hpp — no runtime impact.
  find_file(
    TDMD_NVTX3_HEADER
    NAMES nvtx3/nvtx3.hpp
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    NO_DEFAULT_PATH)
  if(TDMD_NVTX3_HEADER)
    target_compile_definitions(tdmd_telemetry PUBLIC TDMD_HAS_NVTX3=1)
    message(STATUS "tdmd_telemetry: NVTX3 found at ${TDMD_NVTX3_HEADER} — "
                   "TDMD_NVTX_RANGE emits scoped ranges")
  else()
    target_compile_definitions(tdmd_telemetry PUBLIC TDMD_HAS_NVTX3=0)
    message(STATUS "tdmd_telemetry: NVTX3 header nvtx3/nvtx3.hpp not in "
                   "${CUDAToolkit_INCLUDE_DIRS} — TDMD_NVTX_RANGE falls back "
                   "to no-op (install CUDA 12.5+ from nvidia.com for Nsight " "ranges)")
  endif()
endif()

tdmd_apply_build_flavor(tdmd_telemetry)
