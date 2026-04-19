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

# T6.5: EAM/alloy GPU kernel — ≤1e-12 rel vs CPU reference (gpu/SPEC §7.2).
add_executable(test_eam_alloy_gpu test_eam_alloy_gpu.cpp)
target_link_libraries(test_eam_alloy_gpu PRIVATE tdmd::gpu tdmd::potentials tdmd::neighbor
                                                 tdmd::state Catch2::Catch2WithMain)
target_compile_definitions(
  test_eam_alloy_gpu
  PRIVATE TDMD_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/potentials/fixtures"
          TDMD_VERIFY_POTENTIALS_DIR="${CMAKE_SOURCE_DIR}/verify/third_party/potentials")
add_test(NAME test_eam_alloy_gpu COMMAND test_eam_alloy_gpu)

# T6.6: VV NVE integrator GPU kernel — bit-exact vs CPU reference (gpu/SPEC §7.3).
add_executable(test_integrator_vv_gpu test_integrator_vv_gpu.cpp)
target_link_libraries(test_integrator_vv_gpu PRIVATE tdmd::gpu tdmd::integrator tdmd::state
                                                     Catch2::Catch2WithMain)
add_test(NAME test_integrator_vv_gpu COMMAND test_integrator_vv_gpu)
