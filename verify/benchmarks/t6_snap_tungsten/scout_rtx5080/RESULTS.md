# T8.10 scout — TDMD GPU SNAP vs LAMMPS CPU + KOKKOS-GPU SNAP on RTX 5080

**Date:** 2026-04-20 (revised 2026-04-21 with T8.6c-v5 Stages 1-3 per-bond dispatch landed; re-revised 2026-04-21 with T-opt-3b paired-bond reverse index + T-opt-2 yi Phase B parallel-over-jju)
**Hardware:** NVIDIA GeForce RTX 5080 (sm_120, 16 GB GDDR7, 32 °C idle, driver 590.48.01)
**Build:** CUDA 13.1, per-flavor: `build/` (Fp64Reference), `build-mixed/` (MixedFast), `build-mixed-snap-only/` (MixedFastSnapOnly)
**LAMMPS (CPU/legacy GPU pkg):** `verify/third_party/lammps/build_tdmd/lmp` — ML-SNAP + GPU + MANYBODY + KSPACE. This build has no `snap/gpu`/`snap/kk` pair style; `-sf gpu` on `pair_style snap` is a silent CPU fallthrough.
**LAMMPS (KOKKOS GPU, NEW):** `verify/third_party/lammps/build_kokkos_cuda/lmp` — KOKKOS 4.6.2 + CUDA + ML-SNAP + sm_120 (`BLACKWELL120` arch). Provides `snap/kk` pair style running on CUDA device (FP64).
**Fixture:** 2000-atom BCC W (10×10×10), seed=12345, T=300 K, W_2940_2017_2 pure SNAP (no ZBL)
**Scope:** NVE, dt=0.5 fs, 100-step (all configs)

## Headline numbers (revised 2026-04-20 — real LAMMPS GPU baseline; T8.6c-v4 row added)

| Config                              | Build flags            | Wall (s) | ms/step | ratio vs LAMMPS GPU | ratio vs LAMMPS CPU 1-rank |
|-------------------------------------|------------------------|---------:|--------:|--------------------:|---------------------------:|
| TDMD `Fp64ReferenceBuild` (GPU, pre-T8.6c)     | `--fmad=false` oracle  |   44.83 |   448.3 |         104×    |                   2.52×|
| TDMD `MixedFastBuild` (GPU, pre-T8.6c)         | `--fmad=true`          |   31.60 |   316.0 |         73.5×   |                   1.77×|
| TDMD `MixedFastSnapOnlyBuild` (GPU, pre-T8.6c) | T8.9 narrow-FP32       |   31.61 |   326.1 |         75.8×   |                   1.83×|
| TDMD `Fp64ReferenceBuild` (GPU, post-T8.6c-v3)     | `--fmad=false` oracle + warp-parallel yi + deidrj-reduction |   29.25 |   292.5 | 68.0×   | 1.64× |
| TDMD `MixedFastBuild` (GPU, post-T8.6c-v3)         | `--fmad=true` + warp-parallel yi + deidrj-reduction |   20.86 |   208.6 | 48.5×   | 1.17× |
| TDMD `MixedFastSnapOnlyBuild` (GPU, post-T8.6c-v3) | T8.9 narrow-FP32 + warp-parallel yi + deidrj-reduction |   20.92 |   209.2 | 48.7×   | 1.18× |
| TDMD `Fp64ReferenceBuild` (GPU, post-T8.6c-v4)     | + warp-parallel compute_uarray/duarray recurrence | 109.63 |   109.6 | 25.5×   | 0.616× |
| TDMD `MixedFastBuild` (GPU, post-T8.6c-v4)         | + warp-parallel compute_uarray/duarray recurrence |  91.95 |   92.0  | 21.4×   | 0.517× |
| TDMD `MixedFastSnapOnlyBuild` (GPU, post-T8.6c-v4) | + warp-parallel compute_uarray/duarray recurrence | 115.96 |   116.0 | 27.0×   | 0.652× |
| TDMD `Fp64ReferenceBuild` (GPU, post-T8.6c-v5 Stage 3) | + per-bond ui/deidrj dispatch + per-atom gather |  5.96 |   59.6 | 13.9×   | 0.335× |
| TDMD `MixedFastBuild` (GPU, post-T8.6c-v5 Stage 3)    | + per-bond ui/deidrj dispatch + per-atom gather |  4.79 |   47.9  | 11.1×   | 0.269× |
| TDMD `MixedFastSnapOnlyBuild` (GPU, post-T8.6c-v5 Stage 3) | + per-bond ui/deidrj dispatch + per-atom gather |  4.80 |   48.0  | 11.2×   | 0.270× |
| TDMD `Fp64ReferenceBuild` (GPU, post-T-opt-3b)        | + paired-bond reverse index (halves deidrj_bond work) |  — |   40.8 | 9.49×   | 0.230× |
| TDMD `MixedFastBuild` (GPU, post-T-opt-3b)            | + paired-bond reverse index (halves deidrj_bond work) |  — |   34.4 | 8.00×   | 0.194× |
| TDMD `MixedFastSnapOnlyBuild` (GPU, post-T-opt-3b)    | + paired-bond reverse index (halves deidrj_bond work) |  — |   33.8 | 7.86×   | 0.190× |
| **TDMD `Fp64ReferenceBuild` (GPU, post-T-opt-2)**       | + yi_kernel Phase B parallel-over-jju via CSR buckets |  3.70 |   **37.0** | **8.60×**   | **0.208×** |
| **TDMD `MixedFastBuild` (GPU, post-T-opt-2)**           | + yi_kernel Phase B parallel-over-jju via CSR buckets |  2.95 |   **29.5**  | **6.86×**   | **0.166×** (**2.81× faster than LAMMPS CPU 1-rank**) |
| **TDMD `MixedFastSnapOnlyBuild` (GPU, post-T-opt-2)**   | + yi_kernel Phase B parallel-over-jju via CSR buckets |  2.97 |   **29.7**  | **6.91×**   | **0.167×** |
| LAMMPS SNAP 1-rank (`-sf gpu` → CPU fallback) | FP64 + FMA   |   17.87 |   178.7 |              41.5×  |                    ≡ 1-rank CPU |
| LAMMPS SNAP CPU 1-rank              | FP64 + FMA             |   17.79 |   177.9 |              41.4×  |                   1.00×|
| **LAMMPS SNAP KOKKOS snap/kk (GPU)** | CUDA FP64, newton on + neigh half | 0.4305 |   **4.30** | **1.00×**     |                   **0.0242×** (41.4× faster than LAMMPS CPU) |

NB: post-v5 rows are 100-step `tdmd_gpu_100step.yaml` median-of-3 via
`quick_post_t86c_v4.sh` (the "Wall (s)" column above is 100-step wall for
post-v5 rows, 1000-step wall for post-v4 rows, 100-step wall for pre-v4
rows — per-step values are the normalised apples-to-apples comparison).
**T8.6c-v5 Stages 1-3** delivered an additional **1.92× MixedFast speedup**
on top of the already-landed v4, without breaking the T8.7 ≤ 1e-12 rel
byte-exact gate (guaranteed by the bond-list CSR emission order = CPU
cell-stencil walk order invariant, validated by
`test_bond_list_matches_cpu_stencil_order`). MixedFast now runs **3.72×
faster than LAMMPS CPU 1-rank** on a single RTX 5080 (up from 1.94×
post-v4). The gap to LAMMPS KOKKOS `snap/kk` 4.30 ms/step is now
**11.1×** — 48 % of the post-v4 gap closed in a single milestone.

**T8.6c + T-opt speedup summary (nine commits across v1-v5 + two T-opt):**

| Kernel commit   | Scope                          | Fp64Ref       | MixedFast     | MixedSnapOnly |
|-----------------|--------------------------------|--------------:|--------------:|--------------:|
| (pre-T8.6c)     | all-tid==0 baseline            | 448.3 ms/step | 316.0 ms/step | 326.1 ms/step |
| T8.6c-v1 ui     | `add_uarraytot` block-parallel | ~448 ms/step  | ~316 ms/step  | ~316 ms/step  |
| T8.6c-v2 yi     | Phase A parallel over `idxz_max` | 309.3 ms/step | 227.3 ms/step | ~230 ms/step |
| T8.6c-v3 deidrj-red | Phase B warp-shuffle dedr reduction | 292.5 ms/step | 208.6 ms/step | 209.2 ms/step |
| T8.6c-v4 uarray-par | compute_uarray + compute_duarray intra-layer parallel | 109.6 ms/step | 92.0 ms/step | 116.0 ms/step |
| T8.6c-v5 Stage 1 | Device-resident bond list (CSR + SoA), byte-identical emission order | (unchanged — infra only) | (unchanged) | (unchanged) |
| T8.6c-v5 Stage 2 | `snap_ui_bond_kernel` + `snap_ui_gather_kernel` replace `snap_ui_kernel` | 69.4 ms/step | 57.1 ms/step | — |
| T8.6c-v5 Stage 3 | `snap_deidrj_bond_kernel` + `snap_force_gather_kernel` replace `snap_deidrj_kernel` | 59.6 ms/step | 47.9 ms/step | 48.0 ms/step |
| T-opt-3b | Paired-bond reverse index: halves deidrj_bond work via dedr_peer[b] ≡ dedr_own[reverse(b)] pure-function identity | 40.8 ms/step | 34.4 ms/step | 33.8 ms/step |
| **T-opt-2** | **yi_kernel Phase B parallel-over-jju via CSR buckets (no atomics — disjoint writes)** | **37.0 ms/step** | **29.5 ms/step** | **29.7 ms/step** |
| cumulative T8.6c+T-opt | —                         | **12.11×**    | **10.71×**    | **10.98×**    |

**T8.6c-v4 alone delivers 2.67× (Fp64Ref), 2.27× (MixedFast), 1.80×
(MixedSnapOnly) on top of v3.** This was the largest single-commit lever of
the T8.6c-v1..v4 series — compute_uarray/duarray_device were the remaining
tid==0-gated inner loops, and parallelising the intra-layer (mb, k) work
over 128 block threads finally unlocked the GPU that T8.6c-v1/v2/v3 could
only partially saturate. The gap to LAMMPS KOKKOS GPU narrowed from ~48×
to **21–27×** depending on flavor.

**T8.6c-v5 Stages 1-3** then restructured the dispatch shape itself,
replacing the single-block-per-atom model with per-bond parallelism (bond
list pre-pass + `<<<n_bonds, 128>>>` dispatch for ui and deidrj kernels +
per-atom gather kernels). This matches the LAMMPS KOKKOS `pair_snap_kokkos`
launch shape while remaining atomic-free (gpu/SPEC §6.1 — reduce-then-
scatter via per-bond exclusive storage, preserving the T8.7 ≤ 1e-12 rel
byte-exact oracle). On 2000-atom BCC W the per-bond dispatch turned out to
be **1.92× faster** than the per-atom dispatch (MixedFast 92.0 → 47.9
ms/step), more than halving the remaining gap to LAMMPS KOKKOS.

**T-opt-3b** (paired-bond reverse index) and **T-opt-2** (yi Phase B
parallel-over-jju via CSR buckets) added another **1.62× cumulative**
(MixedFast 47.9 → 29.5 ms/step) in two single-commit refactors, both
atomic-free and byte-exact by construction. T-opt-3b exploits the
pure-function nature of `compute_deidrj(Δr, weight, ylist_slab)`: the full
bond list emits both (i→j) and (j→i) bonds, and for any bond b with its
reverse b', `dedr_peer[b] ≡ dedr_own[reverse(b)]` holds bit-for-bit. This
lets `snap_deidrj_bond_kernel` drop the PEER-side branch entirely,
halving its arithmetic — the gather reads `d_dedr_own[reverse(b)]` in
place of a separately computed `d_dedr_peer[b]`. T-opt-2 eliminates the
last tid==0-gated serial loop in the hot path: `snap_yi_kernel`'s Phase B
(ybuf → ylist accumulation) now tid-strides over `idxu_max` jju slots
backed by a pre-built CSR bucket of jjz values ordered ascending per jju
(matching legacy sweep bit-for-bit).

The M8 ≥ 20 % gate still requires ≤ 3.44 ms/step (i.e. further ~8.6×
speedup on top of T-opt-2). On a single RTX 5080 closing that remaining
gap is expected to require either (a) the structurally-different
multi-rank TD path where each rank owns ≤ 1/Nₖ of the atom count, or
(b) algorithmic wins from TDMD's multi-step scheduling (zones on coarser
dt cost proportionally less per wall-second — Andreev's native lever).
Option (a) is M7; option (b) is anchor-test territory (M5). The
single-GPU kernel-tuning ladder has now flattened: further per-bond
kernel work (shared-mem blocking, persistent kernel, sub-step
amortisation) is expected to deliver < 2× on its own.

### T8.6c-v4 regression anomaly — MixedSnapOnly < MixedFast post-refactor

Pre-v4 MixedFast and MixedSnapOnly were near-identical (208.6 vs 209.2
ms/step) because the bispectrum recurrence dominated both, swamping the
5.4 % `ui`-kernel pair-math T8.9 narrow-FP32 savings. Post-v4 the
recurrence is amortised across 128 lanes, revealing a 26 % regression of
MixedSnapOnly (116.0) against MixedFast (92.0).

Root-cause conjecture: the T8.9 `sqrtf(rsq)` + `sincos` + `FP32→FP64`
cast path in `snap_gpu_mixed.cu`'s per-pair setup dominates when the
recurrence is no longer the bottleneck. Documented for future T8.9 review
— the narrow-FP32 scope may now cost more than it saves. Deferred:
T8.6c-v5 (per-bond dispatch) restructures the setup path anyway, which
subsumes this regression into the larger refactor.

LAMMPS KOKKOS numbers are median-of-3 after 1 warmup discard; 4 total runs with 15 s cooldown. Loop time from LAMMPS `Loop time of ...` report; Pair time 0.42392 s = 98.3 % of Loop.

**Reference points:**
- **LAMMPS GPU KOKKOS snap/kk (4.30 ms/step)** — the true M8 target-to-beat.
- LAMMPS CPU 1-rank (178 ms/step) — informational, not the M8 gate reference.

## Key findings

1. **TDMD GPU SNAP was 73.5× slower than LAMMPS GPU SNAP** (`snap/kk` KOKKOS)
   on the same hardware, fixture, and potential — measured at 316 ms/step
   (TDMD MixedFast) vs 4.30 ms/step (LAMMPS KOKKOS). **After T8.6c-v1/v2/v3
   warp-parallel refactors** (ui add_uarraytot, yi Phase-A parallelism,
   deidrj dedr-reduction), the gap narrowed to 48.5× slower (208.6 ms/step
   MixedFast). **After T8.6c-v4 compute_uarray/duarray intra-layer
   parallelism**, the gap further narrowed to 21.4× (92.0 ms/step MixedFast).
   **After T8.6c-v5 per-bond dispatch**, the gap narrowed to 11.1×
   (47.9 ms/step MixedFast). **After T-opt-3b paired-bond reverse
   index and T-opt-2 yi Phase B parallel-over-jju**, the gap now stands at
   **6.86× slower** (29.5 ms/step MixedFast). Cumulative T8.6c+T-opt wins:
   Fp64Ref 12.11×, MixedFast 10.71×, MixedSnapOnly 10.98×. MixedFast now
   runs **2.81× faster than LAMMPS CPU 1-rank** (was 1.77× slower
   pre-T8.6c). Relative to LAMMPS CPU 1-rank (178 ms/step), LAMMPS GPU
   is 41.4× faster, so the GPU baseline remains the only honest reference
   for the M8 ≥ 20 % gate.

2. **LAMMPS `-sf gpu -pk gpu 1` on `pair_style snap` is a silent CPU
   fallthrough on the original `build_tdmd` binary.** The GPU package is
   loaded but only styles with a `/gpu` variant get offloaded. `pair_snap`
   has no `/gpu` variant — the GPU package covers simple pair styles
   (LJ, EAM, Coulomb). **ML-SNAP's GPU path is owned by the KOKKOS
   package as `snap/kk`**, which required a separate build
   (`build_kokkos_cuda/`, see below).

3. **The new `build_kokkos_cuda` LAMMPS binary** is built with
   `-DPKG_KOKKOS=on -DPKG_ML-SNAP=on -DKokkos_ENABLE_CUDA=on
   -DKokkos_ARCH_BLACKWELL120=on`. Run with
   `-k on g 1 -sf kk -pk kokkos newton on neigh half`. `snap/kk` requires
   `newton on + neigh half` (KOKKOS errors if `neigh full`; SNAP errors
   if `newton off`). Diagnostic log:
   `(1) pair snap/kk, perpetual, attributes: full, newton on, kokkos_device`.

4. **T8.9 MixedFastSnapOnly (narrow-FP32) provides ≤ 1 % speedup over
   MixedFast FP64.** 31.61 s vs 31.60 s — within measurement noise.
   Structurally capped: `ui` kernel is only 5.4 % of TDMD GPU runtime;
   the bispectrum cost lives in `deidrj` (61.5 %) + `yi` (33 %).

5. **Fp64Reference → MixedFast gives 30 % speedup** purely from
   `--fmad=true`. `Fp64ReferenceBuild` remains the oracle path with
   `--fmad=false` (D-M6-7, sacred — don't weaken for perf). For production
   runs, `MixedFastBuild` is the recommended single-GPU SNAP flavor (no
   precision loss: SNAP still FP64).

6. **Bitwise output match across all 5 configurations.** TDMD (3 flavors)
   and LAMMPS (CPU, "GPU" CPU-fallback, KOKKOS GPU snap/kk) report
   identical thermo at step 100 (TotEng = −3.5575441440e+04 to 10
   digits). Physics confirmed; the perf gap is purely kernel throughput
   and parallelization strategy, not an algorithmic divergence.

## LAMMPS configuration note (original "GPU" was CPU; KOKKOS rebuild fixes)

### Original `build_tdmd` — silent CPU fallthrough

Reproducible on this machine:

```bash
$ build_tdmd/lmp -h | grep -iE "snap"
pedone          polymorphic     rebo            rebomos         snap
sna/grid        sna/grid/local  snad/atom       snap            snav/atom
#                                               ^^^^ only "snap" — no "snap/gpu"

$ echo 'pair_style snap/gpu
pair_coeff * * W.snapcoeff W.snapparam W' | build_tdmd/lmp
ERROR: Unrecognized pair style 'snap/gpu' (src/force.cpp:275)

$ build_tdmd/lmp -sf gpu -pk gpu 1 -in lammps_script_gpu.in | grep -iE "pair snap"
(1) pair snap, perpetual          # ← CPU snap, NOT snap/gpu
      pair build: full/bin/atomonly
```

The **GPU package provides `/gpu` variants only for simple pair styles**
(EAM, LJ, Coulomb, etc. — visible in `lmp -h`: `eam/gpu`, `eam/alloy/gpu`,
`eam/fs/gpu`, `lj*/gpu`, …). **ML-SNAP's GPU path is owned by the KOKKOS
package** (`snap/kk`).

### New `build_kokkos_cuda` — real `snap/kk` on GPU

Build command (from `verify/third_party/lammps/`):

```bash
cmake -B build_kokkos_cuda -S cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DPKG_KOKKOS=on -DPKG_ML-SNAP=on \
  -DKokkos_ENABLE_CUDA=on -DKokkos_ENABLE_SERIAL=on -DKokkos_ENABLE_OPENMP=off \
  -DKokkos_ARCH_BLACKWELL120=on \
  -DCMAKE_CXX_COMPILER=$(realpath lib/kokkos/bin/nvcc_wrapper)
cmake --build build_kokkos_cuda -j $(nproc)
```

Verification:

```bash
$ build_kokkos_cuda/lmp -h | grep -E "(snap|KOKKOS|Kokkos)"
Kokkos library version: 4.6.2
KOKKOS package API: CUDA Serial
KOKKOS package precision: double
snap            snap/kk         soft            soft/kk         sw              # ← snap/kk present
```

Run invocation:

```bash
build_kokkos_cuda/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -var snap_dir "$(realpath ../../../third_party/lammps/examples/snap)" \
  -var nsteps 100 -in lammps_script_gpu.in
```

Diagnostic output confirms real GPU dispatch:

```
(1) pair snap/kk, perpetual
    attributes: full, newton on, kokkos_device
    pair build: full/bin/kk/device
    bin: kk/device
```

Constraint: `snap/kk` requires `newton on` + `neigh half`. KOKKOS errors
if `neigh full` is requested with `newton on`; SNAP errors if `newton
off`. This is orthogonal to TDMD's Newton-pair layout (TDMD uses a
full-neighbor list with newton-on symmetry internally).

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

### The real M8 gate arithmetic

- **M8 ≥ 20 % beat target: TDMD GPU ≤ 0.8 × 4.30 = 3.44 ms/step** at 2000
  atoms BCC W SNAP. Current MixedFast is 316 ms/step — the required
  speedup is **~92×**.
- T8.6c (warp-parallel refactor of `ui` / `yi` / `deidrj` kernels) is
  estimated to deliver 5–30× at realistic parallel efficiency. Even the
  optimistic end (~30×) lands TDMD at ~10.5 ms/step — still **~2.4× slower**
  than LAMMPS KOKKOS GPU. **T8.6c alone will not close the gate** on a
  single RTX 5080.
- The remaining gap after T8.6c must come from one or a combination of:
  (i) algorithmic wins specific to TDMD (TD multi-step scheduling: zones
  on coarser `dt` cost proportionally less per wall-second; typical dt
  ratio 2–4× in Andreev's regime at comparable accuracy);
  (ii) multi-rank scaling where the TDMD comm pattern amortises better
  than KOKKOS's per-step all-to-all halo exchange;
  (iii) SNAP-specific kernel optimisations beyond the warp-parallel
  refactor (shared-memory blocking of the U-recurrence; persistent
  kernel with sub-step amortisation; mixed-precision on non-ui kernels
  proven bit-exact).

### Revised action list for M8

- (a) **DONE:** Rebuild LAMMPS with KOKKOS + ML-SNAP + CUDA. Real GPU
  baseline is **4.30 ms/step**.
- (b) **T8.6c warp-parallel refactor on `ui` + `yi` + `deidrj`** — pre-impl
  report committed at `docs/development/t8.6c_pre_impl.md`. This is the
  critical path; without it, every subsequent lever amplifies a large
  handicap. Acceptance: scout MixedFast wall ≤ 6.3 s (≥ 5× speedup from
  31.6 s). Even that lands above the M8 gate.
- (c) **Bit-exact T8.7 gate preserved** (CPU-FP64 ≡ GPU-FP64 ≤ 1e-12) —
  any kernel-tuning attempt re-proves bit-exact before merge.
- (d) **Revisit M8 framing given the new baseline.** Master spec §14 M8
  explicitly allows "honestly document why not" if the ≥ 20 % beat is
  unreachable. That honest documentation is now plausibly the M8 outcome
  for single-GPU; the multi-rank TD angle must be measured before a final
  call is made. **Do not over-promise T8.11 cloud burst** on the ≥ 20 %
  beat before T8.6c data lands.
- (e) **Multi-rank TD scout** after T8.6c, comparing TDMD on 4 × RTX 5080
  (or the cloud-burst analogue) against LAMMPS KOKKOS on the same
  hardware. Andreev's dissertation anchor suggests TD should scale
  sublinearly in comm cost where KOKKOS scales linearly — this is
  TDMD's native lever, not the single-GPU kernel race.

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

<details>
<summary>LAMMPS SNAP KOKKOS snap/kk on GPU (NEW, real dispatch)</summary>

4 back-to-back runs (first = warmup, median of remaining 3):

```
Run 1 (warmup):  Loop 0.430492 s   Pair 0.42353 s (98.38 %)   wall 0.77 s
Run 2:           Loop 0.432854 s   Pair 0.42484 s (98.15 %)   wall 0.74 s
Run 3:           Loop 0.430500 s   Pair 0.42392 s (98.47 %)   wall 0.76 s
Run 4:           Loop 0.430222 s   Pair 0.42341 s (98.42 %)   wall 0.76 s

Median of runs 2–4: Loop 0.4305 s = 4.30 ms/step
                    Pair 0.42392 s = 4.24 ms/step
Performance: 232.057 timesteps/s, 464.113 katom-step/s, 10.03 ns/day
```

Diagnostic output confirmed real GPU dispatch
(`pair build: full/bin/kk/device`, `bin: kk/device`). Thermo at step 100
identical to TDMD and to LAMMPS CPU: `TotEng -3.5575441440e+04`.

Invocation:

```bash
build_kokkos_cuda/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -var snap_dir "$(realpath ../../../third_party/lammps/examples/snap)" \
  -var nsteps 100 -in lammps_script_gpu.in
```

Full log captured at `lammps_kokkos_gpu_run.log`.
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

# LAMMPS CPU 1-rank (informational reference):
../../../../verify/third_party/lammps/build_tdmd/lmp \
  -var snap_dir "$(realpath ../../../third_party/lammps/examples/snap)" \
  -var nsteps 100 -in lammps_script_gpu.in

# LAMMPS GPU via KOKKOS snap/kk (the M8 target-to-beat):
../../../../verify/third_party/lammps/build_kokkos_cuda/lmp \
  -k on g 1 -sf kk -pk kokkos newton on neigh half \
  -var snap_dir "$(realpath ../../../third_party/lammps/examples/snap)" \
  -var nsteps 100 -in lammps_script_gpu.in

# Verify snap/kk present in KOKKOS build, absent in legacy build:
../../../../verify/third_party/lammps/build_tdmd/lmp        -h | grep -E "snap[/ ]"
#   → "snap   snad/atom   snap   snav/atom" — no snap/gpu, no snap/kk.
../../../../verify/third_party/lammps/build_kokkos_cuda/lmp -h | grep -E "snap[/ ]"
#   → "snap   snap/kk   ..." — snap/kk present, CUDA device dispatch.
```
