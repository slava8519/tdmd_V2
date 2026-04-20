add_executable(test_potentials test_morse.cpp)

target_link_libraries(test_potentials PRIVATE tdmd::potentials tdmd::neighbor tdmd::state
                                              Catch2::Catch2WithMain)

add_test(NAME test_potentials COMMAND test_potentials)

add_executable(test_tabulated test_tabulated.cpp)

target_link_libraries(test_tabulated PRIVATE tdmd::potentials Catch2::Catch2WithMain)

add_test(NAME test_tabulated COMMAND test_tabulated)

add_executable(test_eam_file test_eam_file.cpp)

target_link_libraries(test_eam_file PRIVATE tdmd::potentials Catch2::Catch2WithMain)

target_compile_definitions(test_eam_file
                           PRIVATE TDMD_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

add_test(NAME test_eam_file COMMAND test_eam_file)

add_executable(test_eam_alloy test_eam_alloy.cpp)

target_link_libraries(test_eam_alloy PRIVATE tdmd::potentials tdmd::neighbor tdmd::state
                                             Catch2::Catch2WithMain)

add_test(NAME test_eam_alloy COMMAND test_eam_alloy)

add_executable(test_eam_fs test_eam_fs.cpp)

target_link_libraries(test_eam_fs PRIVATE tdmd::potentials tdmd::neighbor tdmd::state
                                          Catch2::Catch2WithMain)

add_test(NAME test_eam_fs COMMAND test_eam_fs)

# T8.2: LAMMPS oracle ML-SNAP subset — canonical W_2940_2017_2 fixture path gate. Self-skips with
# exit 77 when the LAMMPS submodule isn't initialized (public CI per Option A D-M6-6). Pure-C++ path
# check, no CUDA / no LAMMPS runtime.
add_executable(test_lammps_oracle_snap_fixture test_lammps_oracle_snap_fixture.cpp)
target_link_libraries(test_lammps_oracle_snap_fixture PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(test_lammps_oracle_snap_fixture
                           PRIVATE TDMD_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME test_lammps_oracle_snap_fixture COMMAND test_lammps_oracle_snap_fixture)
set_tests_properties(test_lammps_oracle_snap_fixture PROPERTIES SKIP_RETURN_CODE 77)

# T8.4a: SnapData parser + SnapPotential skeleton unit tests. Canonical fixture is the LAMMPS
# submodule W_2940_2017_2.snap* — tests auto-skip (exit 77) when the submodule isn't initialized,
# matching the oracle-fixture path gate above.
add_executable(test_snap_file test_snap_file.cpp)
target_link_libraries(test_snap_file PRIVATE tdmd::potentials tdmd::state tdmd::neighbor
                                             Catch2::Catch2WithMain)
target_compile_definitions(test_snap_file
                           PRIVATE TDMD_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME test_snap_file COMMAND test_snap_file)
set_tests_properties(test_snap_file PROPERTIES SKIP_RETURN_CODE 77)
