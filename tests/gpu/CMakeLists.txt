# tdmd::gpu tests — M6 T6.2 + T6.3.
# Exec pack: docs/development/m6_execution_pack.md
#
# test_gpu_types: skeleton type + config tests (pure C++, no CUDA needed).
# test_device_pool: runtime allocator tests; each case skips if CUDA is
#   unavailable so the suite stays green on CPU-only builds + GPU-less CI.

add_executable(test_gpu_types test_gpu_types.cpp)
target_link_libraries(test_gpu_types PRIVATE tdmd::gpu Catch2::Catch2WithMain)
add_test(NAME test_gpu_types COMMAND test_gpu_types)

add_executable(test_device_pool test_device_pool.cpp)
target_link_libraries(test_device_pool PRIVATE tdmd::gpu Catch2::Catch2WithMain)
add_test(NAME test_device_pool COMMAND test_device_pool)

# T6.4: Neighbor-list GPU kernel — bit-exact vs CPU reference.
add_executable(test_neighbor_list_gpu test_neighbor_list_gpu.cpp)
target_link_libraries(test_neighbor_list_gpu PRIVATE tdmd::gpu tdmd::neighbor tdmd::state
                                                     Catch2::Catch2WithMain)
add_test(NAME test_neighbor_list_gpu COMMAND test_neighbor_list_gpu)
