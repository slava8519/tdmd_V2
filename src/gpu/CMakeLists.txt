# tdmd::gpu — M6 GPU primitive layer (RAII wrappers + allocator contract).
# SPEC: docs/specs/gpu/SPEC.md (v1.0, 2026-04-19)
# Exec pack: docs/development/m6_execution_pack.md T6.2–T6.13
#
# T6.2 ships a pure-C++ skeleton — PIMPL-hidden CUDA state (D-M6-17) lets
# this target compile on CPU-only hosts (CI compile-only path per D-M6-6).
# T6.3 replaces gpu_types.cpp with a .cu translation unit holding the
# cudaStream_t / cudaEvent_t Impl definitions.

set(tdmd_gpu_sources gpu_types.cpp device_allocator.cpp)

add_library(tdmd_gpu STATIC ${tdmd_gpu_sources})
add_library(tdmd::gpu ALIAS tdmd_gpu)

target_include_directories(tdmd_gpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                           $<INSTALL_INTERFACE:include>)

target_compile_features(tdmd_gpu PUBLIC cxx_std_20)

# Surface TDMD_BUILD_CUDA as a compile definition so T6.3+ .cu translation units can guard
# CUDA-runtime includes. The skeleton itself is CUDA-free.
if(TDMD_BUILD_CUDA)
  target_compile_definitions(tdmd_gpu PUBLIC TDMD_BUILD_CUDA=1)
else()
  target_compile_definitions(tdmd_gpu PUBLIC TDMD_BUILD_CUDA=0)
endif()

tdmd_apply_build_flavor(tdmd_gpu)
