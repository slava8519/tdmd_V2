# EAM/alloy GPU vs CPU micro-bench (T6.5)

Times `EamAlloyPotential::compute()` (CPU, half-list) against
`EamAlloyGpuAdapter::compute()` (GPU, three-kernel full-list) on an Al FCC
supercell at ≈10⁴ and ≈10⁵ atoms, using the test-fixture
`tests/potentials/fixtures/Al_small.eam.alloy` tables.

Timing includes everything the caller sees on each call:

- **CPU**: the two-pass EAM compute on a pre-built neighbor list.
- **GPU**: adapter translation + H2D transfers (positions, types, cell CSR,
  force in/out) + three kernels + D2H (forces, per-atom PE + virial) +
  host-side Kahan reduction. Spline tables are re-uploaded each call —
  persistent table caching is a T6.7 concern.

Report: median of 5 measurements after 1 untimed warmup. `speedup = CPU/GPU`.

## Run

```
cmake --build build --target bench_eam_gpu_vs_cpu
./build/verify/benchmarks/eam_gpu_vs_cpu/bench_eam_gpu_vs_cpu
```

Writes `bench_results.txt` (tab-separated) in the current directory.

## Acceptance

Per M6 execution pack T6.5: ≥ 5× on ≥ 10⁴ atoms. Not a gate test —
informational baseline for the T6.11 PerfModel calibration.
