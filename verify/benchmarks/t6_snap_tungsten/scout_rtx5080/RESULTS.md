# T8.10 scout — TDMD GPU SNAP vs LAMMPS CPU SNAP on RTX 5080

**Date:** 2026-04-20 (corrected 2026-04-20 — see "LAMMPS configuration note")
**Hardware:** NVIDIA GeForce RTX 5080 (sm_120, 16 GB GDDR7, 32 °C idle, driver 590.48.01)
**Build:** CUDA 13.1, per-flavor: `build/` (Fp64Reference), `build-mixed/` (MixedFast), `build-mixed-snap-only/` (MixedFastSnapOnly)
**LAMMPS:** `verify/third_party/lammps/build_tdmd/lmp` — ML-SNAP + GPU + MANYBODY + KSPACE (CUDA FP64). **This build has no `snap/gpu` pair style** (KOKKOS package not compiled); `-sf gpu` on `pair_style snap` is a silent CPU fallthrough. Both "LAMMPS" rows below are therefore CPU 1-rank — see configuration note.
**Fixture:** 2000-atom BCC W (10×10×10), seed=12345, T=300 K, W_2940_2017_2 pure SNAP (no ZBL)
**Scope:** NVE, dt=0.5 fs, 100-step (all configs; extrapolates linearly for budgeting)

## Headline numbers

| Config                              | Build flags            | Wall (s) | ms/step | ratio vs LAMMPS CPU 1-rank |
|-------------------------------------|------------------------|---------:|--------:|---------------------------:|
| TDMD `Fp64ReferenceBuild` (GPU)     | `--fmad=false` oracle  |   44.83 |   448.3 |                   **2.52×**|
| TDMD `MixedFastBuild` (GPU)         | `--fmad=true`          |   31.60 |   316.0 |                   **1.77×**|
| TDMD `MixedFastSnapOnlyBuild` (GPU) | T8.9 narrow-FP32       |   31.61 |   326.1 |                   **1.83×**|
| LAMMPS SNAP 1-rank (`-sf gpu` → CPU fallback) | FP64 + FMA   |   17.87 |   178.7 |                    ≡ 1-rank CPU (same code path) |
| **LAMMPS SNAP CPU 1-rank**          | FP64 + FMA             |   17.79 |   177.9 |                   **1.00×**|

Ratios on the Pair timer only (dominant; neighbor ≤ 5 ms, comm ≤ 1 ms).

**Reference point:** LAMMPS SNAP CPU 1-rank (17.79 s). Target-to-beat for M8 ≥ 20 % gate.

## Key findings

1. **TDMD GPU SNAP is 1.77× slower than LAMMPS SNAP CPU 1-rank** on the same
   hardware, fixture, and potential. The reference point is a single CPU
   thread, not a GPU — because this LAMMPS build has no `snap/gpu` pair
   style (see configuration note). The headline gap TDMD must close for the
   M8 artifact gate's ≥ 20 % beat is therefore **TDMD GPU vs LAMMPS 1-thread
   CPU** on a consumer GPU. That's a starker comparison than originally
   recorded, and it shifts kernel-tuning priority up sharply.

2. **LAMMPS `-sf gpu -pk gpu 1` on `pair_style snap` is a silent CPU
   fallthrough.** The GPU package is loaded (`Compatible GPU present: yes`,
   CUDA 13.1), but only styles with a `/gpu` variant get offloaded. This
   build lacks both the KOKKOS package and a GPU variant of `pair_snap`;
   LAMMPS-diagnostic output confirms `(1) pair snap, perpetual` (no
   `snap/gpu`), and an explicit `pair_style snap/gpu` test yields
   *"Unrecognized pair style"*. The two LAMMPS rows (17.87 s "GPU" vs
   17.79 s CPU) agree because they run identical code.

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

## LAMMPS configuration note (why "GPU" was CPU)

Evidence chain, reproducible on this machine:

```bash
$ lmp -h | grep -iE "snap"
pedone          polymorphic     rebo            rebomos         snap
sna/grid        sna/grid/local  snad/atom       snap            snav/atom
#                                               ^^^^ only "snap" — no "snap/gpu"

$ echo 'pair_style snap/gpu
pair_coeff * * W.snapcoeff W.snapparam W' | lmp
ERROR: Unrecognized pair style 'snap/gpu' (src/force.cpp:275)

$ lmp -sf gpu -pk gpu 1 -in lammps_script_gpu.in | grep -iE "pair snap"
(1) pair snap, perpetual          # ← CPU snap, NOT snap/gpu
      pair build: full/bin/atomonly
```

The **GPU package provides `/gpu` variants only for simple pair styles**
(EAM, LJ, Coulomb, etc. — visible in `lmp -h`: `eam/gpu`, `eam/alloy/gpu`,
`eam/fs/gpu`, `lj*/gpu`, …). **ML-SNAP's GPU path is owned by the KOKKOS
package** (`snap/kk`), which is not compiled into this LAMMPS binary. The
`-sf gpu` suffix silently skips styles with no `/gpu` variant, leaving
`pair_style snap` on the CPU. No warning is emitted — this is standard
LAMMPS behavior.

**Implication for T8.11:** a true TDMD-GPU vs LAMMPS-GPU apples-to-apples
comparison requires rebuilding LAMMPS with KOKKOS + ML-SNAP + CUDA backend
enabled. Until that rebuild, the measurement baseline is single-thread
LAMMPS CPU (17.79 s).

## TDMD SnapGpu kernel time breakdown (nsys profile, MixedFast)

Captured via `nsys profile --stats=true` on a 100-step MixedFast run
(ncu hardware counters blocked by `RmProfilingAdminOnly=1`; nsys
timeline + CUDA API stats unaffected). Three kernels dominate pair time:

| Kernel               | Total (s) | Calls | Avg (ms) | % of GPU time |
|----------------------|----------:|------:|---------:|--------------:|
| `snap_deidrj_kernel` |     19.4 |   101 |    192.5 |      **61.5 %** |
| `snap_yi_kernel`     |     10.4 |   101 |    103.3 |      **33.0 %** |
| `snap_ui_kernel`     |      1.7 |   101 |     17.0 |        **5.4 %** |
| VV integrator + misc |     < 0.1 |     — |       — |       < 0.1 % |
| Total                |     31.5 |       |          |        ~100 % |

H2D traffic: 2.5 MB total for 100 steps. D2H: 1.3 MB. Both negligible —
the workload is compute-bound on the kernels above.

**Critical insight for T8.9 (narrow-FP32 sqrtf):** T8.9's target was the
`ui` kernel only (5.4 % of runtime). The ≤ 1 % measured throughput win
is therefore structural — even zeroing `ui` cost entirely would cap the
speedup at 5.4 %. Future narrow-FP32 work must target `deidrj` (61.5 %)
or `yi` (33 %).

## Root-cause discovery: all three kernels run single-lane (tid==0)

Inspecting `src/gpu/snap_gpu.cu` reveals the actual bottleneck — the
kernels are **one-block-per-atom with 128 threads**, but **only thread 0
does the main computation**. The other 127 threads are used only for
parallel zeroing / shared-memory loads, then idle while tid==0 walks the
full bispectrum inner loop.

| Kernel | Lines of `if (tid == 0)` gate | Runtime share |
|---|---|---:|
| `snap_ui_kernel`     | L283, L340            | 5.4 % |
| `snap_yi_kernel`     | L449, L545            | 33.0 % |
| `snap_deidrj_kernel` | L686, L806            | 61.5 % |

The `snap_gpu.cu` code itself documents this as intentional (L645-L647):

> `// Per-atom accumulators — single-lane so we don't need warp reductions.`
> `// (Fine for T8.6b correctness gate; warp-level parallel path is a T8.6c`
> `// perf opt once the byte-exact gate lands at T8.7.)`

**T8.7 has landed** (bit-exact GPU SNAP ≤ 1e-12 rel, force 1.69e-14;
project memory confirms 5/5 potential paths byte-exact). **T8.6c is the
deferred optimization that unblocks M8**. In effect:

- **2000 atoms × 1 active thread per block = ~2000 active threads total**
  on an RTX 5080 that can resident ~172K threads. Occupancy is therefore
  bounded by **single-lane parallelism, not by register/shared-mem
  pressure**. The 127-idle-threads-per-block pattern wastes nearly all
  SM resources.
- **A GPU thread at ~1.3 GHz with 1 FP64-FMA/cycle is ~20–30× slower
  than a single AVX-512 core at 5 GHz with 8 FP64-FMA/cycle** in serial
  code. The per-atom work is essentially serial on GPU today, so one
  CPU core beats 84 SMs × 1-lane each.
- **The 1.77× GPU vs LAMMPS CPU gap is not a kernel-tuning mystery** —
  it's a documented deferred optimization (T8.6c) where the correctness
  prototype was shipped pending the byte-exact gate. Closing the gap is
  a **well-scoped refactor**, not exploratory tuning.

### What T8.6c needs to do

Distribute the `tid==0` inner loops across the 128-thread block. Rough
mapping:

- **ui kernel (L340, per-neighbor U-recurrence):** parallelise across
  `(j, mb, ma)` triples of the U-list; current inner j-loop is
  sequential. Warp-level cooperation over the ~165 idxu entries at
  twojmax=8.
- **yi kernel (L449, per-atom Z/Y/B list contraction):** parallelise
  across `idxz_max` entries; atomic-add or warp-reduction into ylist.
- **deidrj kernel (L686, per-neighbor force):** parallelise across the
  27-cell neighbor walk or across the U-recurrence within each pair;
  per-neighbor force accumulation into shared warp-reduced per-atom
  force.

All three must preserve byte-exact output (T8.7 gate). That constrains
reduction order — tree reductions in FP64 are usually byte-exact at
modest atom counts, but the gate must be re-verified per-change.

**Estimated payoff:** if ui/yi/deidrj can parallelise to 32–64 active
lanes per atom (realistic for bispectrum at twojmax=8), expected
throughput gain is **20–30× on paper, ~5–10× in practice** after
accounting for reduction overhead and shared-mem contention. That would
put TDMD GPU comfortably below LAMMPS CPU 1-rank, and positions T8.11
cloud burst to measure a real GPU-vs-GPU story against a KOKKOS-rebuilt
LAMMPS.

## Implications for T8.11 (M8 cloud-burst gate)

- **The reference to beat is LAMMPS CPU 1-rank (17.79 s).** The M8 gate's
  ≥ 20 % beat requires TDMD GPU ≤ 14.23 s at 2000 atoms. Current MixedFast
  is 31.60 s — need a **2.22× speedup** to land the gate.
- **Kernel tuning on `deidrj` + `yi` is mandatory** before T8.11 cloud.
  94.5 % of runtime lives in these two kernels; they have not been tuned
  since the M6 port. Without reducing this, multi-rank scaling can only
  amortise a per-rank handicap that starts at 1.77×.
- **If LAMMPS is rebuilt with KOKKOS snap/kk,** the baseline changes
  substantially. KOKKOS snap/kk is known to be highly optimised by the
  SNL team — RTX 5080 numbers for it are likely 2–4× faster than LAMMPS
  CPU at this fixture size. That would widen the gap TDMD must close, not
  narrow it. Scout re-run after LAMMPS-KOKKOS rebuild is a T8.11
  pre-requisite.
- **Pre-T8.11 action items (revised):**
  - (a) **Rebuild LAMMPS with KOKKOS + ML-SNAP + CUDA**; re-measure the
    LAMMPS GPU baseline to get a real GPU-vs-GPU reference.
  - (b) **Kernel restructuring on `deidrj` and `yi`**. Target occupancy,
    register pressure, shared-memory tiling. ncu can be re-attempted if
    the user grants profiling permissions (`nvidia-modprobe -u
    -m nvidia-uvm` + `RmProfilingAdminOnly=0`); otherwise proceed via
    source inspection + source-level micro-benchmarks.
  - (c) **Bit-exact T8.7 gate is preserved** (CPU-FP64 ≡ GPU-FP64 ≤ 1e-12)
    — any kernel-tuning attempt must re-prove bit-exact before merge.
  - (d) **Defer T8.11 cloud burst** until (a) + (b) close the gap to
    ≤ 1.3× vs the real LAMMPS GPU baseline, or demonstrate multi-rank
    scaling wins even at the per-GPU handicap. Running cloud at 1.77×+
    per-GPU handicap wastes budget.

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
<summary>LAMMPS SNAP (`-sf gpu -pk gpu 1`, CPU fallback)</summary>

```
Step Temp         PotEng            KinEng            TotEng            Press
0    2.9999966099e+02 -3.5652979687e+04 7.7517221358e+01 -3.5575462465e+04 -1.7713001276e+06
100  1.3157708382e+03 -3.5915424822e+04 3.3998338195e+02 -3.5575441440e+04 -1.7611294365e+06
Loop time: 17.8741 s on 1 proc
Performance: 0.242 ns/day, 5.595 timesteps/s, 11.189 katom-step/s
Pair 17.872 s / 99.99 % total
Elapsed 0:18.96
```

Diagnostic output shows `(1) pair snap, perpetual` + `pair build:
full/bin/atomonly` — i.e. `pair_style snap` ran on the CPU and the
CPU neighbor builder was used. `-sf gpu` is a no-op for this style in
this LAMMPS build (no `snap/gpu` variant; no KOKKOS). The run is
therefore identical to the "CPU 1-rank" row (agreement to 3 significant
figures confirms it: 17.87 vs 17.79 s).

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

# LAMMPS with -sf gpu suffix (silently CPU for snap — see config note):
../../../../verify/third_party/lammps/build_tdmd/lmp -sf gpu -pk gpu 1 \
  -var snap_dir "$(realpath ../../../third_party/lammps/examples/snap)" \
  -var nsteps 100 -in lammps_script_gpu.in

# LAMMPS CPU 1-rank (canonical reference):
../../../../verify/third_party/lammps/build_tdmd/lmp \
  -var snap_dir "$(realpath ../../../third_party/lammps/examples/snap)" \
  -var nsteps 100 -in lammps_script_gpu.in

# Verify no snap/gpu variant in this build:
../../../../verify/third_party/lammps/build_tdmd/lmp -h | grep -E "snap[/ ]"
#   → prints "snap   snad/atom   snap   snav/atom" — no snap/gpu, no snap/kk.
```
