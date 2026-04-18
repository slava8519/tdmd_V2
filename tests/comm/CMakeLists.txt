# tdmd::comm tests — M5 T5.2+T5.3+T5.4.
# Exec pack: docs/development/m5_execution_pack.md
#
# test_comm: non-MPI types + serializer unit tests (always built).
# test_mpi_host_staging_2rank: MPI-only, needs TDMD_ENABLE_MPI + mpirun.

add_executable(test_comm test_comm_types.cpp test_packet_serializer.cpp)

target_link_libraries(test_comm PRIVATE tdmd::comm Catch2::Catch2WithMain)

add_test(NAME test_comm COMMAND test_comm)

# ---------------------------------------------------------------------------
# MPI-gated test binaries. T5.4 ships the 2-rank ping-pong + reduction smoke. T5.5 (RingBackend)
# will extend the list with test_ring_backend_4rank.
# ---------------------------------------------------------------------------
if(TDMD_ENABLE_MPI)
  find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
  if(NOT MPIEXEC_EXECUTABLE)
    message(WARNING "TDMD_ENABLE_MPI=ON but mpirun/mpiexec not found; "
                    "skipping test_mpi_host_staging_2rank.")
  else()
    add_executable(test_mpi_host_staging_2rank main_mpi.cpp test_mpi_host_staging_2rank.cpp)

    # Catch2::Catch2 (library-only, no WithMain) — we ship our own main in main_mpi.cpp wrapping
    # MPI_Init/Finalize.
    target_link_libraries(test_mpi_host_staging_2rank PRIVATE tdmd::comm Catch2::Catch2
                                                              MPI::MPI_CXX)

    # --oversubscribe allows OpenMPI to run 2 ranks even on single-core CI containers where the
    # default is 1 slot. MPICH has no equivalent flag, detected via the binary name.
    set(TDMD_MPIEXEC_NP 2)
    if(MPIEXEC_EXECUTABLE MATCHES "mpirun$")
      set(TDMD_MPI_EXTRA_FLAGS "--oversubscribe")
    else()
      set(TDMD_MPI_EXTRA_FLAGS "")
    endif()

    add_test(NAME test_mpi_host_staging_2rank
             COMMAND ${MPIEXEC_EXECUTABLE} ${TDMD_MPI_EXTRA_FLAGS} -np ${TDMD_MPIEXEC_NP}
                     $<TARGET_FILE:test_mpi_host_staging_2rank>)

    # --- T5.5 4-rank ring test -------------------------------------------
    add_executable(test_ring_4rank main_mpi.cpp test_ring_4rank.cpp)
    target_link_libraries(test_ring_4rank PRIVATE tdmd::comm Catch2::Catch2 MPI::MPI_CXX)
    add_test(NAME test_ring_4rank COMMAND ${MPIEXEC_EXECUTABLE} ${TDMD_MPI_EXTRA_FLAGS} -np 4
                                          $<TARGET_FILE:test_ring_4rank>)
  endif()
endif()
