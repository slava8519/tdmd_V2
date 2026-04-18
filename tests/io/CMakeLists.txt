add_executable(test_io test_lammps_data_reader.cpp test_yaml_config.cpp test_preflight.cpp)

target_link_libraries(test_io PRIVATE tdmd::io tdmd::state tdmd::runtime Catch2::Catch2WithMain)

target_compile_definitions(test_io
                           PRIVATE TDMD_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

add_test(NAME test_io COMMAND test_io)
