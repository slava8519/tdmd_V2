add_executable(test_scheduler test_scheduler_types.cpp)

target_link_libraries(test_scheduler PRIVATE tdmd::scheduler Catch2::Catch2WithMain)

add_test(NAME test_scheduler COMMAND test_scheduler)
