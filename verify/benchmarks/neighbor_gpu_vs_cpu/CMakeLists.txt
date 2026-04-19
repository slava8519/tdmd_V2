# T6.4 micro-bench: CPU vs GPU neighbor-list build at 10⁴ / 10⁵ atoms.
# Exec pack: docs/development/m6_execution_pack.md T6.4 (micro-bench)
# Runs only under CUDA-enabled builds — CPU-only has nothing to compare.

if(NOT TDMD_BUILD_CUDA)
  return()
endif()

add_executable(bench_neighbor_gpu_vs_cpu bench.cpp)
target_link_libraries(bench_neighbor_gpu_vs_cpu PRIVATE tdmd::gpu tdmd::neighbor tdmd::state)
target_compile_features(bench_neighbor_gpu_vs_cpu PRIVATE cxx_std_20)
tdmd_apply_build_flavor(bench_neighbor_gpu_vs_cpu)
