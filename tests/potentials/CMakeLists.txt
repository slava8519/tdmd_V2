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
