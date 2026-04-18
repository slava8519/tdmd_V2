# T4 differential test — M2 acceptance gate (Ni-Al EAM/alloy vs LAMMPS).
# Shells out to verify/t4/run_differential.py; SKIPs when the LAMMPS oracle
# submodule has not been built (Option A, see docs/development/
# m0_execution_pack.md).
#
# Structurally identical to verify/t1/CMakeLists.txt — each differential
# benchmark keeps its own tiny CMakeLists so adding / removing one is a
# single add_subdirectory() toggle in the parent.

add_executable(test_t4_differential test_t4_differential.cpp)
target_link_libraries(test_t4_differential PRIVATE Catch2::Catch2WithMain)

target_compile_definitions(test_t4_differential PRIVATE TDMD_REPO_ROOT="${CMAKE_SOURCE_DIR}")

add_dependencies(test_t4_differential tdmd)

add_test(NAME test_t4_differential COMMAND test_t4_differential)
set_tests_properties(test_t4_differential PROPERTIES ENVIRONMENT "TDMD_CLI_BIN=$<TARGET_FILE:tdmd>"
                                                     SKIP_RETURN_CODE 77)
