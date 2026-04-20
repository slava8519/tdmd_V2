# tdmd::gpu — M6 GPU primitive layer (RAII wrappers + allocator contract + device pool).
# SPEC: docs/specs/gpu/SPEC.md (v1.0, 2026-04-19)
# Exec pack: docs/development/m6_execution_pack.md T6.2–T6.13
#
# T6.2: PIMPL-hidden skeleton (types + allocator interface).
# T6.3: factories (probe_devices / make_stream / make_event) + DevicePool
#       (cudaMallocAsync-backed, D-M6-12) + PinnedHostPool (D-M6-3).
# T6.4+: .cu kernel translation units layered on top.

set(tdmd_gpu_sources
    gpu_types.cpp
    device_allocator.cpp
    factories.cpp
    device_pool.cpp
    neighbor_list_gpu.cu
    eam_alloy_gpu.cu
    eam_alloy_gpu_mixed.cu
    integrator_vv_gpu.cu
    snap_gpu.cu)

# T6.4 / T6.5 / T6.6 / T6.8 / T8.6a: the .cu TUs carry both CUDA kernels (TDMD_BUILD_CUDA=1 branch)
# and a pure-C++ throwing stub (TDMD_BUILD_CUDA=0 branch). Under CPU-only builds we retag their
# language so the host compiler picks them up — the files never contain kernel syntax outside the
# CUDA guard. eam_alloy_gpu_mixed.cu (T6.8) and snap_gpu.cu (T8.6a) are always built so dispatch via
# `TDMD_FLAVOR_MIXED_FAST` / runtime.backend: gpu resolves at adapter-link time regardless of the
# selected flavor.
if(NOT TDMD_BUILD_CUDA)
  set_source_files_properties(neighbor_list_gpu.cu eam_alloy_gpu.cu eam_alloy_gpu_mixed.cu
                              integrator_vv_gpu.cu snap_gpu.cu PROPERTIES LANGUAGE CXX)
endif()

add_library(tdmd_gpu STATIC ${tdmd_gpu_sources})
add_library(tdmd::gpu ALIAS tdmd_gpu)

target_include_directories(tdmd_gpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                           $<INSTALL_INTERFACE:include>)
# Internal header (cuda_handles.hpp) sits in src/gpu/ alongside the .cpp files — PRIVATE so it
# doesn't leak into downstream targets (D-M6-17).
target_include_directories(tdmd_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(tdmd_gpu PUBLIC cxx_std_20)

# Surface TDMD_BUILD_CUDA as a compile definition so the .cpp files can guard CUDA-runtime includes.
# Link cudart when CUDA is enabled.
if(TDMD_BUILD_CUDA)
  target_compile_definitions(tdmd_gpu PUBLIC TDMD_BUILD_CUDA=1)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(tdmd_gpu PUBLIC CUDA::cudart)
else()
  target_compile_definitions(tdmd_gpu PUBLIC TDMD_BUILD_CUDA=0)
endif()

# T6.11 NVTX bridge — telemetry's nvtx.hpp expands to `nvtx3::scoped_range` under TDMD_BUILD_CUDA.
# nvtx3 is a **header-only** C++ API that resolves libnvToolsExt symbols lazily at first range entry
# (dlsym path — see nvtxDetail/nvtxInit.h in CUDA 13.1). No explicit link dependency is required.
# Inheriting tdmd::telemetry propagates the include path and the TDMD_BUILD_CUDA definition used by
# the macro.
target_link_libraries(tdmd_gpu PUBLIC tdmd::telemetry)

tdmd_apply_build_flavor(tdmd_gpu)
