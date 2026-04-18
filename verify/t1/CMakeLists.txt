# T1 differential test — shells out to verify/t1/run_differential.py (Python).
# Runs under ctest alongside the unit tests; SKIPs automatically when the
# LAMMPS oracle submodule has not been built (see docs/development/
# m0_execution_pack.md "Option A").

add_executable(test_t1_differential test_t1_differential.cpp)
target_link_libraries(test_t1_differential PRIVATE Catch2::Catch2WithMain)

# Absolute repo root is known at configure time — bake it in as a compile define. The tdmd binary
# path is a generator expression (resolved at build time), so it goes via an environment variable on
# the test's command line.
target_compile_definitions(test_t1_differential PRIVATE TDMD_REPO_ROOT="${CMAKE_SOURCE_DIR}")

# The test binary depends on the `tdmd` CLI being built — ctest does not infer this from the
# environment-variable linkage below.
add_dependencies(test_t1_differential tdmd)

add_test(NAME test_t1_differential COMMAND test_t1_differential)
set_tests_properties(test_t1_differential PROPERTIES ENVIRONMENT "TDMD_CLI_BIN=$<TARGET_FILE:tdmd>"
                                                     SKIP_RETURN_CODE 77)
