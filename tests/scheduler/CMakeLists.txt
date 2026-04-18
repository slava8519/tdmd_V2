add_executable(test_scheduler test_scheduler_types.cpp test_safety_certificate.cpp)

target_link_libraries(test_scheduler PRIVATE tdmd::scheduler Catch2::Catch2WithMain)

add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(fuzz_cert_monotonicity fuzz_cert_monotonicity.cpp)
target_link_libraries(fuzz_cert_monotonicity PRIVATE tdmd::scheduler Catch2::Catch2WithMain)

add_test(NAME fuzz_cert_monotonicity COMMAND fuzz_cert_monotonicity)
