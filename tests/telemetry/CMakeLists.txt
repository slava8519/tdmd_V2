add_executable(test_telemetry test_telemetry.cpp)

target_link_libraries(test_telemetry PRIVATE tdmd::telemetry Catch2::Catch2WithMain)

add_test(NAME test_telemetry COMMAND test_telemetry)
