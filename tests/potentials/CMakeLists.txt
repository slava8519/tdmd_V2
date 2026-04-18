add_executable(test_potentials test_morse.cpp)

target_link_libraries(test_potentials PRIVATE tdmd::potentials tdmd::neighbor tdmd::state
                                              Catch2::Catch2WithMain)

add_test(NAME test_potentials COMMAND test_potentials)

add_executable(test_tabulated test_tabulated.cpp)

target_link_libraries(test_tabulated PRIVATE tdmd::potentials Catch2::Catch2WithMain)

add_test(NAME test_tabulated COMMAND test_tabulated)
