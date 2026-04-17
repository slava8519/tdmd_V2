add_executable(test_state test_atom_soa.cpp)

target_link_libraries(test_state PRIVATE tdmd::state Catch2::Catch2WithMain)

add_test(NAME test_state COMMAND test_state)
