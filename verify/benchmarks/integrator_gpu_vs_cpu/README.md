# VV NVE integrator — CPU vs GPU micro-bench (T6.6)

Standalone bench (not a CTest). Times a single CPU / GPU
velocity-Verlet step (`pre_force` + `post_force`) for 10⁴ and 10⁵ atoms.

## Build

Requires `TDMD_BUILD_CUDA=ON`. The target is included via
`verify/CMakeLists.txt`, so a standard CUDA build of `tdmd` picks it up
automatically when `TDMD_BUILD_TESTS=ON`.

```sh
cmake -S . -B build -DTDMD_BUILD_CUDA=ON
cmake --build build --target bench_integrator_gpu_vs_cpu
./build/verify/benchmarks/integrator_gpu_vs_cpu/bench_integrator_gpu_vs_cpu
```

## What it measures

- `pre_force_step + post_force_step` pair, in microseconds (median of 5,
  1 warmup). Not end-to-end NVE with a real potential — isolates the
  integrator kernel.
- GPU path includes per-call H2D + kernel + D2H (T6.6 adapter shape;
  resident-on-GPU arrives in T6.7).

## Output

- stdout + `bench_results.txt`: columns `size CPU_us GPU_us speedup`.

## Caveats

- Very fast kernel — per-call overhead (pool allocate + cudaMemcpyAsync
  launches) dominates at 10⁴. Speedup at that size is expected to be
  modest; real integrator payoff arrives with the resident-on-GPU
  pattern (T6.7).
