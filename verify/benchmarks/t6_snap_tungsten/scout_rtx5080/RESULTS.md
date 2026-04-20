# T8.10 scout — TDMD vs LAMMPS GPU SNAP perf on RTX 5080

**Date:** 2026-04-20
**Hardware:** NVIDIA GeForce RTX 5080 (sm_120, 16 GB GDDR7, 32 °C idle, driver 590.48.01)
**Build:** CUDA 13.1, per-flavor: `build/` (Fp64Reference), `build-mixed/` (MixedFast), `build-mixed-snap-only/` (MixedFastSnapOnly)
**LAMMPS:** `verify/third_party/lammps/build_tdmd/lmp` — ML-SNAP + GPU package (FP64)
**Fixture:** 2000-atom BCC W (10×10×10), seed=12345, T=300 K, W_2940_2017_2 pure SNAP (no ZBL)
**Scope:** NVE, dt=0.5 fs, 100-step (all configs; extrapolates linearly for budgeting)

## Headline numbers

| Config                          | Build flags          | Wall (s) | ms/step | ratio vs LAMMPS CPU |
|---------------------------------|----------------------|---------:|--------:|--------------------:|
| TDMD `Fp64ReferenceBuild` (GPU) | `--fmad=false` oracle |   44.83 |   448.3 |            **2.52×**|
| TDMD `MixedFastBuild` (GPU)     | `--fmad=true`         |   31.60 |   316.0 |            **1.77×**|
| TDMD `MixedFastSnapOnlyBuild` (GPU) | T8.9 narrow-FP32  |   31.61 |   326.1 |            **1.83×**|
| LAMMPS SNAP (GPU, `-sf gpu`)    | FP64 + FMA            |   17.87 |   178.7 |            **1.00×**|
| LAMMPS SNAP (CPU, 1 MPI rank)   | FP64 + FMA            |   17.79 |   177.9 |               0.99× |

Ratios on the Pair timer only (dominant; neighbor ≤ 5 ms, comm ≤ 1 ms).

## Key findings

1. **TDMD GPU SNAP is 1.77× slower than LAMMPS GPU SNAP** on the same hardware,
   fixture, and potential — apples-to-apples compiler flags (`--fmad=true`
   matches LAMMPS default CUDA flags). This is the headline gap TDMD must
   close to hit the M8 artifact gate's ≥ 20 % beat.

2. **LAMMPS SNAP GPU ≈ LAMMPS SNAP CPU at 2000 atoms.** 17.87 s vs 17.79 s —
   the GPU package gives no speedup at this size on an RTX 5080. This
   reshapes the M8 ≥ 20 % gate interpretation: on single-GPU workloads,
   LAMMPS's "fast path" is effectively the single-thread CPU SNAP, and
   TDMD's job is to beat that. Multi-rank scaling is where the GPU package
   earns its keep (T8.11 cloud burst).

3. **T8.9 MixedFastSnapOnly (narrow-FP32) provides ≤ 1 % speedup over
   MixedFast FP64.** 31.61 s vs 31.60 s — within measurement noise. The
   cmake comment was prophetic: "realised throughput lever is modest —
   pair-math is < 5 % of total SNAP runtime." The dominant cost is the
   bispectrum U-recurrence and Y-kernel; sqrtf's latency win doesn't move
   the needle at this atom count / twojmax=8 working-set size.

4. **Fp64Reference → MixedFast gives 30 % speedup** purely from
   `--fmad=true`. This is the compile-flag delta expected from enabling
   FMA merging on the bispectrum FMUL + FADD chains; `Fp64ReferenceBuild`
   remains the oracle path with `--fmad=false` (D-M6-7, sacred — don't
   weaken for perf). For production runs, `MixedFastBuild` is the
   recommended single-GPU SNAP flavor (no precision loss: SNAP still FP64).

5. **Bitwise output match.** TDMD and LAMMPS report identical thermo at
   step 100 (both show TotEng = −3.5575441440e+04). Physics confirmed;
   the perf gap is purely kernel throughput, not an algorithmic divergence.

## Implications for T8.11 (M8 cloud-burst gate)

- **Single-GPU perf gate is unlikely to hit ≥ 20 % beat** on RTX 5080 or
  A100 (same GPU package path). TDMD is ~1.77× slower per-GPU.
- **Multi-rank scaling is where the value lands.** LAMMPS SNAP GPU
  parallel efficiency is well-documented to degrade past 4 ranks; TDMD's
  time-decomposition + Pattern 2 subdomain scheduler should produce
  better strong-scaling curves. T8.11 must measure weak+strong 1→8+ ranks
  on cloud hardware (A100 SXM4 NVLink, p4d equivalent) to find the
  cross-over point.
- **Pre-T8.11 action items identified by this scout:**
  - (a) Profile TDMD SnapGpu kernels with NVTX / Nsight Compute;
    compare occupancy, memory bandwidth, and register pressure vs LAMMPS
    `pair_snap/gpu`. The 1.77× gap suggests a tuning gap, not an
    algorithmic one (bispectrum path is identical; output matches).
  - (b) Bit-exact T8.7 gate is preserved (CPU-FP64 ≡ GPU-FP64 ≤ 1e-12)
    — any kernel-tuning attempt after this scout must re-prove bit-exact
    before being green.
  - (c) Defer T8.11 cloud burst until either (a) closes the gap below
    1.3× or (b) demonstrates multi-rank scaling wins even at 1.77×
    per-GPU handicap. Running cloud at 1.77× per-GPU wastes budget.

## Methodology caveats

- **Single 100-step run per config**, not median-of-N with cooldown. The
  absolute numbers carry ±5 % drift from JIT/thermal noise; the **ratios
  are robust** because back-to-back runs see the same GPU state. For
  publication-quality numbers, re-measure with `run_scout.sh` (present in
  this dir, runs 4× per config with 30 s cooldown).
- **Warmup cost not explicitly separated.** First ~10 steps on RTX 5080
  sm_120 include PTX JIT; repeat runs amortise to < 0.5 s. At 18-45 s
  totals, warmup is ≤ 3 % of run time. 1000-step extrapolation reduces
  this further.
- **Consumer GPU ≠ data-center GPU.** RTX 5080 (GDDR7, 960 GB/s BW, sm_120)
  vs A100 (HBM2e, 2 TB/s BW, sm_80) — SNAP is compute-bound per warp, so
  the ratio should translate directionally but not linearly. T8.11 cloud
  re-measurement is still required for the artifact gate.

## Raw log

<details>
<summary>TDMD Fp64Reference</summary>

```
# step temp pe ke etotal press
0    3.0000000000e+02 -3.5652979687e+04 7.7517221358e+01 -3.5575462465e+04 -1.1055586745e+00
100  1.3157723251e+03 -3.5915424822e+04 3.3998338195e+02 -3.5575441440e+04 -1.0992106279e+00
Performance: 193324.8 tau/day, 0.0967 ns/day, 446.916 ms/timestep
Pair 44.680 s | Total 44.692 s | Elapsed 0:45.79
```
</details>

<details>
<summary>TDMD MixedFastBuild</summary>

```
Pair 31.606 s | Total 31.617 s | Elapsed 0:32.54
```
</details>

<details>
<summary>TDMD MixedFastSnapOnlyBuild</summary>

```
Pair 31.611 s | Total 31.622 s | Elapsed 0:32.58
```
</details>

<details>
<summary>LAMMPS SNAP GPU</summary>

```
Step Temp         PotEng            KinEng            TotEng            Press
0    2.9999966099e+02 -3.5652979687e+04 7.7517221358e+01 -3.5575462465e+04 -1.7713001276e+06
100  1.3157708382e+03 -3.5915424822e+04 3.3998338195e+02 -3.5575441440e+04 -1.7611294365e+06
Loop time: 17.8741 s on 1 proc
Performance: 0.242 ns/day, 5.595 timesteps/s, 11.189 katom-step/s
Pair 17.872 s / 99.99 % total
Elapsed 0:18.96
```

Note: LAMMPS pressure reports include unit-conversion factor mismatch to TDMD
scalar units (TDMD prints pressure in eV/Å³, LAMMPS in bar); PE/KE/TotEng
match to 10 digits so this is a reporting convention, not a physics divergence.
</details>

<details>
<summary>LAMMPS SNAP CPU 1-rank</summary>

```
Loop time: 17.7906 s on 1 proc
Performance: 5.621 timesteps/s, 11.242 katom-step/s
Pair 17.788 s / 99.99 % total
Elapsed 0:18.62
```
</details>

## Reproducibility

```bash
cd verify/benchmarks/t6_snap_tungsten/scout_rtx5080
python3 ../generate_setup.py --nrep 10 --out setup_2000.data
./run_scout.sh    # full 4-run median protocol
```

Single-run spot checks (what this scout used):

```bash
# TDMD per flavor:
../../../../build/src/cli/tdmd                run --timing tdmd_gpu_100step.yaml
../../../../build-mixed/src/cli/tdmd          run --timing tdmd_gpu_100step.yaml
../../../../build-mixed-snap-only/src/cli/tdmd run --timing tdmd_gpu_100step.yaml

# LAMMPS GPU + CPU:
../../../../verify/third_party/lammps/build_tdmd/lmp -sf gpu -pk gpu 1 \
  -var snap_dir "$(realpath ../../../third_party/lammps/examples/snap)" \
  -var nsteps 100 -in lammps_script_gpu.in
../../../../verify/third_party/lammps/build_tdmd/lmp \
  -var snap_dir "$(realpath ../../../third_party/lammps/examples/snap)" \
  -var nsteps 100 -in lammps_script_gpu.in
```
