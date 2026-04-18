add_executable(test_perfmodel test_perfmodel.cpp)

target_link_libraries(test_perfmodel PRIVATE tdmd::perfmodel Catch2::Catch2WithMain)

add_test(NAME test_perfmodel COMMAND test_perfmodel)
