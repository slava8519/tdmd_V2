add_executable(test_zoning test_zoning_types.cpp)

target_link_libraries(test_zoning PRIVATE tdmd::zoning Catch2::Catch2WithMain)

add_test(NAME test_zoning COMMAND test_zoning)
