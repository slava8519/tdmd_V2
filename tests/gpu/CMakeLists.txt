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

# T8.6a: SnapGpu + SnapGpuAdapter plumbing. Asserts the class skeleton compiles and links on all
# flavors, rejects M8-scope flags, and throws the T8.6b sentinel from compute(). Full kernel-body
# bit-exactness gate (D-M8-13, ≤1e-12 rel vs CPU) lands at T8.7 after T8.6b. Self-skips with exit 77
# if the LAMMPS submodule (W_2940_2017_2.snap* fixture) is not initialized — same Option A
# convention as test_snap_compute.
add_executable(test_snap_gpu_plumbing test_snap_gpu_plumbing.cpp)
target_link_libraries(test_snap_gpu_plumbing PRIVATE tdmd::gpu tdmd::potentials tdmd::state
                                                     Catch2::Catch2WithMain)
target_compile_definitions(
  test_snap_gpu_plumbing
  PRIVATE TDMD_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/potentials/fixtures")
add_test(NAME test_snap_gpu_plumbing COMMAND test_snap_gpu_plumbing)
set_tests_properties(test_snap_gpu_plumbing PROPERTIES SKIP_RETURN_CODE 77)

# T6.6: VV NVE integrator GPU kernel — bit-exact vs CPU reference (gpu/SPEC §7.3).
add_executable(test_integrator_vv_gpu test_integrator_vv_gpu.cpp)
target_link_libraries(test_integrator_vv_gpu PRIVATE tdmd::gpu tdmd::integrator tdmd::state
                                                     Catch2::Catch2WithMain)
add_test(NAME test_integrator_vv_gpu COMMAND test_integrator_vv_gpu)

# T6.8: MixedFast EAM/alloy GPU kernel — ≤ 1e-6 rel force / ≤ 1e-8 rel PE vs Fp64 reference GPU
# (D-M6-8). Compares `EamAlloyGpu` vs `EamAlloyGpuMixed` directly, bypassing the flavor-dispatched
# adapter so the threshold applies regardless of the selected BuildFlavor. T6.11: NVTX audit —
# asserts every kernel launch in src/gpu/*.cu sits inside a scope carrying a TDMD_NVTX_RANGE macro
# (D-M6-14). Pure structural check, runs on CPU-only + CUDA builds.
add_executable(test_nvtx_audit test_nvtx_audit.cpp)
target_link_libraries(test_nvtx_audit PRIVATE tdmd::gpu Catch2::Catch2WithMain)
target_compile_definitions(test_nvtx_audit PRIVATE TDMD_GPU_SRC_DIR="${CMAKE_SOURCE_DIR}/src/gpu")
add_test(NAME test_nvtx_audit COMMAND test_nvtx_audit)

add_executable(test_eam_mixed_fast_within_threshold test_eam_mixed_fast_within_threshold.cpp)
target_link_libraries(
  test_eam_mixed_fast_within_threshold PRIVATE tdmd::gpu tdmd::potentials tdmd::neighbor
                                               tdmd::state Catch2::Catch2WithMain)
target_compile_definitions(
  test_eam_mixed_fast_within_threshold
  PRIVATE TDMD_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/potentials/fixtures"
          TDMD_VERIFY_POTENTIALS_DIR="${CMAKE_SOURCE_DIR}/verify/third_party/potentials")
add_test(NAME test_eam_mixed_fast_within_threshold COMMAND test_eam_mixed_fast_within_threshold)

# T7.0: 100-step NVE drift on Ni-Al EAM/alloy with runtime.backend=gpu + MixedFastBuild. Asserts
# |E(100) - E(0)| / |E(0)| <= 1e-6 (per-100-step D-M6-8 dense-cutoff budget with 10x margin under
# the 1000-step 1e-5 cap). Self-skips on no-CUDA, Reference-only, or CPU-only builds (all of those
# paths are covered by other gates).
add_executable(test_t4_nve_drift test_t4_nve_drift.cpp)
target_link_libraries(test_t4_nve_drift PRIVATE tdmd::gpu tdmd::runtime tdmd::io
                                                Catch2::Catch2WithMain)
target_compile_definitions(test_t4_nve_drift PRIVATE TDMD_REPO_ROOT="${CMAKE_SOURCE_DIR}")
add_test(NAME test_t4_nve_drift COMMAND test_t4_nve_drift)
set_tests_properties(test_t4_nve_drift PROPERTIES SKIP_RETURN_CODE 4)

# T7.8: K-deep 2-stream compute/mem overlap gate. Runs K=4 sync vs pipelined EAM on 10976-atom Al
# FCC; asserts overlap ratio (t_serial - t_pipelined) / t_pipelined >= 0.30 (exec pack §5 T7.8
# acceptance). Self-skips on no-CUDA hosts; reference bit-exactness of slot 0 vs serial oracle is
# checked inline at 1e-12 rel (gpu/SPEC §7.2).
add_executable(test_overlap_budget test_overlap_budget.cpp)
target_link_libraries(test_overlap_budget PRIVATE tdmd::gpu tdmd::scheduler tdmd::potentials
                                                  tdmd::neighbor tdmd::state Catch2::Catch2WithMain)
target_compile_definitions(
  test_overlap_budget
  PRIVATE TDMD_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/potentials/fixtures")
add_test(NAME test_overlap_budget COMMAND test_overlap_budget)

# ---------------------------------------------------------------------------
# T8.0 (T7.8b carry-forward) — 2-rank K=4 overlap gate. K=4 sync vs pipelined EAM + synthetic halo
# MPI_Sendrecv per slot; asserts overlap ratio >= 0.30 (gpu/SPEC §3.2c). Self-skips with exit code 4
# when < 2 CUDA devices are visible (dev hosts with 1 GPU cannot measure 2-rank overlap
# meaningfully; the real 30% measurement is cloud-burst-gated and ties into T8.11). Bit-exact slot 0
# vs serial oracle checked inline at 1e-12 rel (D-M6-7).
# ---------------------------------------------------------------------------
if(TDMD_ENABLE_MPI)
  find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
  if(NOT MPIEXEC_EXECUTABLE)
    message(WARNING "TDMD_ENABLE_MPI=ON but mpirun/mpiexec not found; "
                    "skipping test_overlap_budget_2rank.")
  else()
    add_executable(test_overlap_budget_2rank main_mpi.cpp test_overlap_budget_2rank.cpp)

    target_link_libraries(
      test_overlap_budget_2rank
      PRIVATE tdmd::gpu
              tdmd::scheduler
              tdmd::potentials
              tdmd::neighbor
              tdmd::state
              Catch2::Catch2
              MPI::MPI_CXX)

    target_compile_definitions(
      test_overlap_budget_2rank
      PRIVATE TDMD_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/potentials/fixtures")

    if(MPIEXEC_EXECUTABLE MATCHES "mpirun$")
      set(TDMD_MPI_EXTRA_FLAGS "--oversubscribe")
    else()
      set(TDMD_MPI_EXTRA_FLAGS "")
    endif()

    add_test(NAME test_overlap_budget_2rank COMMAND ${MPIEXEC_EXECUTABLE} ${TDMD_MPI_EXTRA_FLAGS}
                                                    -np 2 $<TARGET_FILE:test_overlap_budget_2rank>)
    set_tests_properties(test_overlap_budget_2rank PROPERTIES SKIP_RETURN_CODE 4)
  endif()
endif()
