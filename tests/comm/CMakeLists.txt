# tdmd::comm tests — M5 T5.2 skeleton + types smoke.
# Exec pack: docs/development/m5_execution_pack.md T5.2
#
# T5.2 ships the abstract CommBackend + value types only. Concrete backend
# tests (2-rank ping, 4-rank ring) arrive with T5.4 / T5.5 and will live in
# this same directory (test_mpi_host_staging.cpp, test_ring_backend.cpp).

add_executable(test_comm test_comm_types.cpp)

target_link_libraries(test_comm PRIVATE tdmd::comm Catch2::Catch2WithMain)

add_test(NAME test_comm COMMAND test_comm)
