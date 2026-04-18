add_executable(test_cli test_run_command.cpp test_validate_command.cpp)

target_link_libraries(test_cli PRIVATE tdmd::cli Catch2::Catch2WithMain)

# Reuse the io fixtures (.data + .yaml) rather than shipping a cli-specific copy. Keeps the test
# surface focused on dispatch / exit codes.
target_compile_definitions(test_cli
                           PRIVATE TDMD_IO_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/io/fixtures")

add_test(NAME test_cli COMMAND test_cli)
