add_executable(test_neighbor test_cell_grid.cpp)

target_link_libraries(test_neighbor PRIVATE tdmd::neighbor Catch2::Catch2WithMain)

add_test(NAME test_neighbor COMMAND test_neighbor)
