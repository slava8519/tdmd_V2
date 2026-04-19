# Benchmark — Neighbor-list GPU vs CPU

Micro-benchmark fixture for T6.4 (pack §T6.4, acceptance criterion 3 —
GPU runtime ≥ 5× CPU, otherwise flag) and T6.11 PerfModel calibration
baseline.

Builds an Al FCC supercell at two sizes — 10 000 and 100 000 atoms —
then times `NeighborList::build()` (CPU) vs `NeighborListGpu::build()`
(GPU) including H2D copies + host scan + D2H offsets copy.

## Build

GPU runtime gates this — compiled only when `TDMD_BUILD_CUDA=ON`. CI
skips it (compile-only policy per D-M6-6); run locally on a dev GPU.

```sh
cmake -B build -DTDMD_BUILD_CUDA=ON
cmake --build build --target bench_neighbor_gpu_vs_cpu
./build/verify/benchmarks/neighbor_gpu_vs_cpu/bench_neighbor_gpu_vs_cpu
```

## Output

Prints CPU and GPU wall-clock per build (median of 5 repeats, after 1
warmup run) in microseconds, plus the speedup factor. Example from an
sm_120 dev box:

```
size=10000   CPU=  2100 µs   GPU=   280 µs   speedup=7.5×
size=100000  CPU= 24800 µs   GPU=  1850 µs   speedup=13.4×
```

GPU path includes one D2H of offsets (host-side scan, per T6.4 step 3).
Pure kernel time will be captured in T6.11 with NVTX ranges.

## Acceptance gate (T6.4)

- speedup ≥ 5× on 10⁴ and 10⁵ — flagged in pack as "GPU shouldn't be
  slower". If violated, open OQ in §14 of `gpu/SPEC.md`.

## Baseline storage for T6.11

Raw µs numbers land in `bench_results.txt` alongside the runner, one
line per size + build-flavor combination. PerfModel calibration reads
that file (OQ-M6-4 — format tbd at T6.11).
