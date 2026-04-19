add_executable(
  test_scheduler
  test_scheduler_types.cpp
  test_safety_certificate.cpp
  test_zone_state_machine.cpp
  test_scheduler_core.cpp
  test_select_ready_tasks.cpp
  test_commit_protocol.cpp
  test_retry_canonical.cpp
  test_deadlock_watchdog.cpp
  test_determinism_same_seed.cpp
  test_queue_ordering_determinism.cpp
  test_k_batching_pipeline.cpp)

target_link_libraries(test_scheduler PRIVATE tdmd::scheduler tdmd::zoning tdmd::runtime tdmd::io
                                             Catch2::Catch2WithMain)

# test_determinism_same_seed.cpp reuses the same io fixtures as tests/runtime/ (shrunken Morse on
# 32-atom Al FCC).
target_compile_definitions(test_scheduler
                           PRIVATE TDMD_IO_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/io/fixtures")

add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(fuzz_cert_monotonicity fuzz_cert_monotonicity.cpp)
target_link_libraries(fuzz_cert_monotonicity PRIVATE tdmd::scheduler Catch2::Catch2WithMain)

add_test(NAME fuzz_cert_monotonicity COMMAND fuzz_cert_monotonicity)

add_executable(fuzz_zone_state_invariants fuzz_zone_state_invariants.cpp)
target_link_libraries(fuzz_zone_state_invariants PRIVATE tdmd::scheduler Catch2::Catch2WithMain)

add_test(NAME fuzz_zone_state_invariants COMMAND fuzz_zone_state_invariants)

add_executable(fuzz_frontier_invariant fuzz_frontier_invariant.cpp)
target_link_libraries(fuzz_frontier_invariant PRIVATE tdmd::scheduler tdmd::zoning
                                                      Catch2::Catch2WithMain)

add_test(NAME fuzz_frontier_invariant COMMAND fuzz_frontier_invariant)

# T5.6: K-batching I6 fuzzer pinned to D-M5-1 legal set {1, 2, 4, 8}.
add_executable(fuzz_k_batching_invariants fuzz_k_batching_invariants.cpp)
target_link_libraries(fuzz_k_batching_invariants PRIVATE tdmd::scheduler tdmd::zoning
                                                         Catch2::Catch2WithMain)

add_test(NAME fuzz_k_batching_invariants COMMAND fuzz_k_batching_invariants)

# ---------------------------------------------------------------------------
# T5.7 — 2-rank peer dispatch integration test. Requires TDMD_ENABLE_MPI + mpirun. Drives Completed
# → Packed → InFlight → Committed on the sender and verifies on_zone_data_arrived fires on the
# receiver's peer zone.
# ---------------------------------------------------------------------------
if(TDMD_ENABLE_MPI)
  find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
  if(NOT MPIEXEC_EXECUTABLE)
    message(WARNING "TDMD_ENABLE_MPI=ON but mpirun/mpiexec not found; "
                    "skipping test_peer_dispatch_2rank.")
  else()
    add_executable(test_peer_dispatch_2rank main_mpi.cpp test_peer_dispatch_2rank.cpp)

    # Catch2::Catch2 (library-only, no WithMain) — main_mpi.cpp wraps MPI_Init/Finalize.
    target_link_libraries(test_peer_dispatch_2rank PRIVATE tdmd::scheduler tdmd::zoning tdmd::comm
                                                           Catch2::Catch2 MPI::MPI_CXX)

    if(MPIEXEC_EXECUTABLE MATCHES "mpirun$")
      set(TDMD_MPI_EXTRA_FLAGS "--oversubscribe")
    else()
      set(TDMD_MPI_EXTRA_FLAGS "")
    endif()

    add_test(NAME test_peer_dispatch_2rank COMMAND ${MPIEXEC_EXECUTABLE} ${TDMD_MPI_EXTRA_FLAGS}
                                                   -np 2 $<TARGET_FILE:test_peer_dispatch_2rank>)
  endif()
endif()
