add_executable(test_runtime test_unit_converter.cpp test_unit_converter_lj.cpp
                            test_simulation_engine.cpp)

target_link_libraries(test_runtime PRIVATE tdmd::runtime tdmd::io Catch2::Catch2WithMain)

# test_simulation_engine.cpp reuses the T1.3/T1.4 io fixtures (32-atom Al FCC .data + valid
# tdmd.yaml) instead of shipping a runtime-specific copy.
target_compile_definitions(test_runtime
                           PRIVATE TDMD_IO_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/io/fixtures")

add_test(NAME test_runtime COMMAND test_runtime)
