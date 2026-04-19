# T6.6 micro-bench: CPU vs GPU velocity-Verlet half-kick + drift at 10⁴ / 10⁵ atoms.
# Exec pack: docs/development/m6_execution_pack.md T6.6 (micro-bench)
# Runs only under CUDA-enabled builds — CPU-only has nothing to compare.

if(NOT TDMD_BUILD_CUDA)
  return()
endif()

add_executable(bench_integrator_gpu_vs_cpu bench.cpp)
target_link_libraries(bench_integrator_gpu_vs_cpu PRIVATE tdmd::integrator tdmd::gpu tdmd::state)
target_compile_features(bench_integrator_gpu_vs_cpu PRIVATE cxx_std_20)
tdmd_apply_build_flavor(bench_integrator_gpu_vs_cpu)
