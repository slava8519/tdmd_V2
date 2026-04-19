add_executable(test_perfmodel test_perfmodel.cpp)

target_link_libraries(test_perfmodel PRIVATE tdmd::perfmodel Catch2::Catch2WithMain)

add_test(NAME test_perfmodel COMMAND test_perfmodel)

# T6.11: GPU cost-table structural + wiring tests. The ±20% accuracy gate vs Nsight-measured data is
# scoped as T6.11b (needs profiling run on target GPU).
add_executable(test_gpu_cost_tables test_gpu_cost_tables.cpp)
target_link_libraries(test_gpu_cost_tables PRIVATE tdmd::perfmodel Catch2::Catch2WithMain)
add_test(NAME test_gpu_cost_tables COMMAND test_gpu_cost_tables)

# T7.10: Pattern 2 (two-level hybrid TD × SD) cost prediction — geometry helper, hybrid step
# formula, halo linear scaling with face-neighbor count, recommendation decision logic.
add_executable(test_perfmodel_pattern2 test_perfmodel_pattern2.cpp)
target_link_libraries(test_perfmodel_pattern2 PRIVATE tdmd::perfmodel Catch2::Catch2WithMain)
add_test(NAME test_perfmodel_pattern2 COMMAND test_perfmodel_pattern2)
