# TDMD root CMake.
# Master spec: §7 (BuildFlavor × ExecProfile), §D (precision policy),
#              §D.14 (build system), §D.16 (__restrict__).
# See docs/development/build_instructions.md for usage.

cmake_minimum_required(VERSION 3.25)

# ------------------------------------------------------------------------------
# Project.
# ------------------------------------------------------------------------------
project(
  tdmd
  VERSION 0.1.0
  DESCRIPTION "Time-Decomposition Molecular Dynamics"
  LANGUAGES CXX)

# Prevent in-source builds — they spray build artifacts into the source tree and confuse git.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are disallowed. Use: cmake -B build --preset default")
endif()

# ------------------------------------------------------------------------------
# Options (cache variables visible via `cmake -LH`).
# ------------------------------------------------------------------------------
option(TDMD_BUILD_CUDA "Enable CUDA targets (GPU build)" ON)
option(TDMD_BUILD_TESTS "Build unit tests" ON)
option(TDMD_BUILD_BENCHMARKS "Build microbenchmarks" OFF)
option(TDMD_WARNINGS_AS_ERRORS "Treat warnings as errors (use ON in CI)" OFF)
option(TDMD_ENABLE_MPI "Enable MPI-dependent comm backends (M5+)" ON)
option(TDMD_ENABLE_NCCL "Enable NCCL backend (M7+, auto-detect at configure)" ON)

set(TDMD_BUILD_FLAVOR
    "Fp64ReferenceBuild"
    CACHE STRING "Active BuildFlavor (master spec §7.1)")
set_property(
  CACHE TDMD_BUILD_FLAVOR
  PROPERTY STRINGS
           "Fp64ReferenceBuild"
           "Fp64ProductionBuild"
           "MixedFastBuild"
           "MixedFastSnapOnlyBuild"
           "MixedFastAggressiveBuild"
           "Fp32ExperimentalBuild")

# RTX 5080 (Blackwell) = sm_120, requires CUDA 12.8+. Users on CUDA 12.6 must override:
# -DTDMD_CUDA_ARCHS=89
set(TDMD_CUDA_ARCHS
    "120"
    CACHE STRING "CUDA compute capabilities (semicolon list, or 'native')")

# ------------------------------------------------------------------------------
# Language standards.
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for clang-tidy and editor integrations.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# CUDA setup — conditional on TDMD_BUILD_CUDA.
# ------------------------------------------------------------------------------
if(TDMD_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "CUDA compiler not found; disabling CUDA targets. "
                    "To enable, install CUDA 12.8+ and re-configure, or set -DTDMD_BUILD_CUDA=OFF.")
    set(TDMD_BUILD_CUDA OFF)
  else()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    include(cmake/FindCUDAArchHelper.cmake)
    tdmd_resolve_cuda_archs(TDMD_CUDA_ARCHS_RESOLVED "${TDMD_CUDA_ARCHS}")
    set(CMAKE_CUDA_ARCHITECTURES ${TDMD_CUDA_ARCHS_RESOLVED})

    # sm_120 requires CUDA 12.8+. Emit hard error if toolchain is older — nvcc would fail anyway,
    # but our message is clearer.
    if("120" IN_LIST CMAKE_CUDA_ARCHITECTURES)
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.8")
        message(
          FATAL_ERROR
            "sm_120 (RTX 5080 Blackwell) requires CUDA 12.8+, but detected "
            "${CMAKE_CUDA_COMPILER_VERSION}. Options:\n"
            "  1. Upgrade: sudo apt install cuda-toolkit-12-8 (or newer)\n"
            "  2. Override: cmake -DTDMD_CUDA_ARCHS=89 (Ada / RTX 40xx fallback)")
      endif()
    endif()

    message(STATUS "TDMD CUDA: enabled, archs=${CMAKE_CUDA_ARCHITECTURES}, "
                   "nvcc=${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
else()
  message(STATUS "TDMD CUDA: disabled (TDMD_BUILD_CUDA=OFF)")
endif()

# ------------------------------------------------------------------------------
# Shared includes.
# ------------------------------------------------------------------------------
include(cmake/CompilerWarnings.cmake)
include(cmake/BuildFlavors.cmake)
include(cmake/Dependencies.cmake)

message(STATUS "TDMD BuildFlavor: ${TDMD_BUILD_FLAVOR}")

# ------------------------------------------------------------------------------
# Optional MPI — concrete comm backends (M5+ MpiHostStaging, Ring) link to it. TDMD_ENABLE_MPI=OFF
# disables those backends; the abstract CommBackend + types (T5.2) still compile and the rest of the
# build stays green.
# ------------------------------------------------------------------------------
if(TDMD_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if(NOT MPI_CXX_FOUND)
    message(WARNING "TDMD_ENABLE_MPI=ON but MPI not found. Disabling MPI-dependent backends. "
                    "Install an MPI implementation (OpenMPI 4.1+ or MPICH 4+) or configure with "
                    "-DTDMD_ENABLE_MPI=OFF to silence this warning.")
    set(TDMD_ENABLE_MPI OFF)
  else()
    message(STATUS "TDMD MPI: enabled, ${MPI_CXX_COMPILER} (version ${MPI_CXX_VERSION})")
  endif()
else()
  message(STATUS "TDMD MPI: disabled (TDMD_ENABLE_MPI=OFF)")
endif()

# ------------------------------------------------------------------------------
# Optional NCCL — NcclBackend (M7+) links to it. TDMD_ENABLE_NCCL=ON is best-effort: we auto-detect
# via find_path/find_library, and silently disable the backend (constructor throws at runtime,
# engine falls back to MpiHostStaging) if NCCL is absent. This mirrors the CUDA-aware-MPI probe-gate
# pattern from T7.3 and keeps the build green on dev machines without NCCL installed.
# ------------------------------------------------------------------------------
if(TDMD_ENABLE_NCCL
   AND TDMD_BUILD_CUDA
   AND TDMD_ENABLE_MPI)
  find_path(NCCL_INCLUDE_DIR nccl.h HINTS $ENV{NCCL_HOME}/include /usr/include
                                          /usr/local/cuda/include /usr/local/include)
  find_library(
    NCCL_LIBRARY nccl HINTS $ENV{NCCL_HOME}/lib $ENV{NCCL_HOME}/lib64 /usr/lib/x86_64-linux-gnu
                            /usr/local/cuda/lib64 /usr/local/lib)
  if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    message(STATUS "TDMD NCCL: enabled, ${NCCL_LIBRARY} (headers ${NCCL_INCLUDE_DIR})")
  else()
    message(STATUS "TDMD NCCL: requested but not found — NcclBackend will stub-throw at "
                   "construction (fallback to MpiHostStaging is engine's job).")
    set(TDMD_ENABLE_NCCL OFF)
  endif()
else()
  message(STATUS "TDMD NCCL: disabled (TDMD_ENABLE_NCCL=OFF, or CUDA/MPI missing)")
  set(TDMD_ENABLE_NCCL OFF)
endif()

# ------------------------------------------------------------------------------
# Testing.
# ------------------------------------------------------------------------------
if(TDMD_BUILD_TESTS)
  enable_testing()
  include(CTest)
endif()

# ------------------------------------------------------------------------------
# Subdirectories.
# ------------------------------------------------------------------------------
add_subdirectory(src)

if(TDMD_BUILD_TESTS)
  add_subdirectory(tests)
  add_subdirectory(verify)
endif()

if(TDMD_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ------------------------------------------------------------------------------
# Summary.
# ------------------------------------------------------------------------------
message(STATUS "--------------------------------------------------")
message(STATUS "TDMD ${PROJECT_VERSION} configured")
message(STATUS "  Flavor:        ${TDMD_BUILD_FLAVOR}")
message(STATUS "  CUDA:          ${TDMD_BUILD_CUDA}")
if(TDMD_BUILD_CUDA)
  message(STATUS "  CUDA arches:   ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Tests:         ${TDMD_BUILD_TESTS}")
message(STATUS "  Benchmarks:    ${TDMD_BUILD_BENCHMARKS}")
message(STATUS "  Warnings=Err:  ${TDMD_WARNINGS_AS_ERRORS}")
message(STATUS "--------------------------------------------------")
