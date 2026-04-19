# T6.5 micro-bench: CPU vs GPU EAM/alloy force at 10⁴ / 10⁵ atoms.
# Exec pack: docs/development/m6_execution_pack.md T6.5 (micro-bench)
# Runs only under CUDA-enabled builds — CPU-only has nothing to compare.

if(NOT TDMD_BUILD_CUDA)
  return()
endif()

add_executable(bench_eam_gpu_vs_cpu bench.cpp)
target_link_libraries(bench_eam_gpu_vs_cpu PRIVATE tdmd::potentials tdmd::gpu tdmd::neighbor
                                                   tdmd::state)
target_compile_definitions(
  bench_eam_gpu_vs_cpu
  PRIVATE TDMD_BENCH_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/potentials/fixtures")
target_compile_features(bench_eam_gpu_vs_cpu PRIVATE cxx_std_20)
tdmd_apply_build_flavor(bench_eam_gpu_vs_cpu)
