# T0 morse-analytic — pure-C++ test (no LAMMPS dependency, no Python harness).
#
# Unlike T1 the benchmark's ground truth is a closed-form formula, so the test
# loads the YAML configs directly via tdmd::io + tdmd::runtime and does its
# own comparison inline. See docs/specs/verify/SPEC.md §4.1 (T0 overview) and
# verify/benchmarks/t0_morse_analytic/README.md for the physics.

add_executable(test_t0_analytic test_t0_analytic.cpp)
target_link_libraries(test_t0_analytic PRIVATE tdmd::io tdmd::runtime tdmd::state
                                               Catch2::Catch2WithMain)

target_compile_definitions(test_t0_analytic PRIVATE TDMD_REPO_ROOT="${CMAKE_SOURCE_DIR}")

add_test(NAME test_t0_analytic COMMAND test_t0_analytic)
