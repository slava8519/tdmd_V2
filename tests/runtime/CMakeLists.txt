add_executable(
  test_runtime test_unit_converter.cpp test_unit_converter_lj.cpp test_simulation_engine.cpp
               test_td_mode_smoke.cpp test_k1_regression_smoke.cpp)

target_link_libraries(test_runtime PRIVATE tdmd::runtime tdmd::io Catch2::Catch2WithMain)

# test_simulation_engine.cpp reuses the T1.3/T1.4 io fixtures (32-atom Al FCC .data + valid
# tdmd.yaml) instead of shipping a runtime-specific copy.
target_compile_definitions(test_runtime
                           PRIVATE TDMD_IO_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/io/fixtures")

add_test(NAME test_runtime COMMAND test_runtime)

# ---------------------------------------------------------------------------
# T5.8 — 2-rank multi-rank TD smoke. K=1 P=2 thermo bit-exact to K=1 P=1 via deterministic Kahan
# ring reduction. Requires TDMD_ENABLE_MPI + mpirun.
# ---------------------------------------------------------------------------
if(TDMD_ENABLE_MPI)
  find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
  if(NOT MPIEXEC_EXECUTABLE)
    message(WARNING "TDMD_ENABLE_MPI=ON but mpirun/mpiexec not found; "
                    "skipping test_multirank_td_smoke_2rank.")
  else()
    add_executable(test_multirank_td_smoke_2rank main_mpi.cpp test_multirank_td_smoke_2rank.cpp)

    # Catch2::Catch2 (library-only) — main_mpi.cpp wraps MPI_Init/Finalize.
    target_link_libraries(test_multirank_td_smoke_2rank PRIVATE tdmd::runtime tdmd::io tdmd::comm
                                                                Catch2::Catch2 MPI::MPI_CXX)

    target_compile_definitions(test_multirank_td_smoke_2rank
                               PRIVATE TDMD_IO_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/io/fixtures")

    if(MPIEXEC_EXECUTABLE MATCHES "mpirun$")
      set(TDMD_MPI_EXTRA_FLAGS "--oversubscribe")
    else()
      set(TDMD_MPI_EXTRA_FLAGS "")
    endif()

    add_test(NAME test_multirank_td_smoke_2rank
             COMMAND ${MPIEXEC_EXECUTABLE} ${TDMD_MPI_EXTRA_FLAGS} -np 2
                     $<TARGET_FILE:test_multirank_td_smoke_2rank>)
  endif()
endif()
