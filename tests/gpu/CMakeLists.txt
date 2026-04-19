# tdmd::gpu tests — M6 T6.2.
# Exec pack: docs/development/m6_execution_pack.md
#
# test_gpu_types: skeleton type + config tests (pure C++, no CUDA needed).
# T6.3+ will add .cu-based kernel unit tests gated on TDMD_BUILD_CUDA.

add_executable(test_gpu_types test_gpu_types.cpp)

target_link_libraries(test_gpu_types PRIVATE tdmd::gpu Catch2::Catch2WithMain)

add_test(NAME test_gpu_types COMMAND test_gpu_types)
