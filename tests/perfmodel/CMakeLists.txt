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

# T7.13: Pattern 1 ±20% calibration gate — loads verify/measurements/gpu_cost_calibration.yaml and
# validates |predict_step_gpu_sec - measured| < 20% per D-M6-8 / T6.11b. Graceful SKIP when fixture
# missing (Option A CI policy, D-M6-6). Pattern 2 ±25% gate (D-M7-9) is T7.13b.
add_executable(test_gpu_cost_calibration test_gpu_cost_calibration.cpp)
target_link_libraries(test_gpu_cost_calibration PRIVATE tdmd::perfmodel Catch2::Catch2WithMain)
target_compile_definitions(test_gpu_cost_calibration PRIVATE TDMD_REPO_ROOT="${CMAKE_SOURCE_DIR}")
add_test(NAME test_gpu_cost_calibration COMMAND test_gpu_cost_calibration)
