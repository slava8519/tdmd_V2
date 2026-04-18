# tdmd::comm — M5 transport abstraction (CommBackend + types).
# SPEC: docs/specs/comm/SPEC.md (v1.0, 2026-04-16)
# Exec pack: docs/development/m5_execution_pack.md T5.2–T5.12

add_library(tdmd_comm STATIC comm_backend.cpp)
add_library(tdmd::comm ALIAS tdmd_comm)

target_include_directories(tdmd_comm PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                            $<INSTALL_INTERFACE:include>)

target_compile_features(tdmd_comm PUBLIC cxx_std_20)

# Optional MPI — linked in by concrete backends in T5.4+ (MpiHostStaging, Ring). TDMD_ENABLE_MPI
# defaults ON on Linux. Off disables all MPI-using backends; the skeleton abstract interface (T5.2)
# still compiles cleanly.
if(TDMD_ENABLE_MPI)
  target_compile_definitions(tdmd_comm PUBLIC TDMD_ENABLE_MPI=1)
  if(TARGET MPI::MPI_CXX)
    target_link_libraries(tdmd_comm PUBLIC MPI::MPI_CXX)
  endif()
else()
  target_compile_definitions(tdmd_comm PUBLIC TDMD_ENABLE_MPI=0)
endif()

tdmd_apply_build_flavor(tdmd_comm)
