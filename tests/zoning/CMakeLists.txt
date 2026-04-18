add_executable(test_zoning test_zoning_types.cpp test_linear1d.cpp test_decomp2d.cpp
                           test_hilbert.cpp test_hilbert3d.cpp test_default_planner.cpp)

target_link_libraries(test_zoning PRIVATE tdmd::zoning Catch2::Catch2WithMain)

add_test(NAME test_zoning COMMAND test_zoning)
