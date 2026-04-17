add_executable(test_runtime test_unit_converter.cpp)

target_link_libraries(test_runtime PRIVATE tdmd::runtime Catch2::Catch2WithMain)

add_test(NAME test_runtime COMMAND test_runtime)
