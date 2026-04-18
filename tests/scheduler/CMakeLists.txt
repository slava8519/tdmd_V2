add_executable(test_scheduler test_scheduler_types.cpp test_safety_certificate.cpp
                              test_zone_state_machine.cpp test_scheduler_core.cpp)

target_link_libraries(test_scheduler PRIVATE tdmd::scheduler tdmd::zoning Catch2::Catch2WithMain)

add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(fuzz_cert_monotonicity fuzz_cert_monotonicity.cpp)
target_link_libraries(fuzz_cert_monotonicity PRIVATE tdmd::scheduler Catch2::Catch2WithMain)

add_test(NAME fuzz_cert_monotonicity COMMAND fuzz_cert_monotonicity)

add_executable(fuzz_zone_state_invariants fuzz_zone_state_invariants.cpp)
target_link_libraries(fuzz_zone_state_invariants PRIVATE tdmd::scheduler Catch2::Catch2WithMain)

add_test(NAME fuzz_zone_state_invariants COMMAND fuzz_zone_state_invariants)
