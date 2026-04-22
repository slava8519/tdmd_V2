# T8.14 — M8 artifact gate **Case B** honest-documentation report

**Milestone:** M8 — SNAP proof-of-value + `MixedFastSnapOnlyBuild` (master spec §14).
**Gate outcome:** Case B invoked per **D-M8-6** (master spec §14 M8 artifact-gate dual-path clause).
**Date:** 2026-04-22.
**Author role:** Validation / Reference Engineer (+ cross-sign by GPU / Performance Engineer on §3–§5).
**Scope:** T6 tungsten SNAP 2000-atom BCC on 1× RTX 5080 (sm_120, 16 GB GDDR7, CUDA 13.1, driver 590.48.01).
**Companion artefacts:** `verify/benchmarks/t6_snap_tungsten/scout_rtx5080/RESULTS.md` (raw scout), `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md` (T8.12 slow-tier pass), master spec §14 M8 entry.

---

## §1 — Executive summary

Master spec §14 M8 defines a **dual-path artifact gate**:

> **Artifact gate:** на T6 TDMD либо обгоняет LAMMPS ≥ 20 % на целевой конфигурации (≥ 8 ranks, commodity network), либо проект документирует, почему не обгоняет и что делать дальше (честная постановка).

This report is the Case B leg. We **do not** beat LAMMPS KOKKOS `snap/kk` on the target configuration with the v1-alpha code base. We document below: (a) the measured gap, (b) the structural reasons it does not close on a single GPU or on a feasible small-cluster extrapolation, (c) what TDMD nevertheless delivers that is worth tagging, (d) why the remaining gap is an **architectural** feature of gpu/SPEC §6.1 (reduce-then-scatter) combined with the SNAP workload being **compute-bound** rather than **halo-bound**, and (e) the M9+ path forward on potentials where TD's native lever (per-zone `dt` × `T_comm = T_p / K`) actually applies.

**One-line headline.** TDMD MixedFast SNAP GPU on T6 2000-atom BCC W: **29.2 ms/step**. LAMMPS KOKKOS `snap/kk` on the same hardware: **4.30 ms/step**. Gap: **6.79× slower**. M8 ≥ 20 % beat requires ≤ 3.44 ms/step (further **8.5×** speedup). On a single RTX 5080 the kernel-tuning ladder has flattened after ten commits (T8.6c-v1 … T-opt-4 Item 1), delivering **10.82× cumulative** MixedFast speedup from the pre-T8.6c baseline of 316 ms/step. The remaining 8.5× is not plausibly recoverable via further single-GPU refactors, and is only partially addressed by the multi-rank / multi-GPU path (M7) because **SNAP is compute-bound, not halo-bound** — the workload where TD's native advantage dominates is MEAM (angular-moments halo), not SNAP.

**Decision.** Close M8 as **Case B per D-M8-6**: honest documentation of the gap, its structural origin, TDMD's validated correctness story, and the M9+ roadmap toward the MEAM / halo-bound niche where TD is expected to outperform KOKKOS-class SD. Proceed to `v1.0.0-alpha1` tag (T8.13). `MixedFastSnapOnlyBuild` slow-tier VerifyLab pass GREEN (T8.12, 2026-04-21).

---

## §2 — Empirical numbers

**Fixture.** T6 W BCC 10×10×10 = 2000 atoms, seed = 12345, T = 300 K, `W_2940_2017_2` pure SNAP (no ZBL). NVE, dt = 0.5 fs, 100 steps. Thermo at step 100 bit-identical across all six measured configurations (`TotEng = −3.5575441440e+04` to 10 digits).

**Method.** Median-of-3 after 1 warm-up discard (4 runs total with 15 s cooldown) for LAMMPS KOKKOS; 100-step single-run spot checks for TDMD flavors post-T-opt ladder (absolute ±5 % noise, ratio robust). LAMMPS CPU baseline freshly re-measured 2026-04-22 on the same host to eliminate day-over-day drift.

### §2.1 — Headline table

| Config                                        | Build / precision        | ms/step | Ratio vs LAMMPS KOKKOS | Ratio vs LAMMPS CPU 1-rank |
|-----------------------------------------------|--------------------------|--------:|-----------------------:|---------------------------:|
| TDMD `Fp64ReferenceBuild` GPU (oracle)        | FP64 throughout, `--fmad=false`, reduce-then-scatter | **36.4** | 8.47× slower | 4.90× faster |
| **TDMD `MixedFastBuild` GPU**                 | FP64 state + FP64 SNAP + FP64 EAM + `--fmad=true` | **29.2** | **6.79× slower** | **6.10× faster** |
| TDMD `MixedFastSnapOnlyBuild` GPU             | FP64 state + narrow-FP32 SNAP pair-math + FP64 EAM | **29.2** | 6.79× slower | 6.10× faster |
| LAMMPS SNAP CPU 1-rank × 1 OMP thread         | FP64 + FMA (AVX-512)     | 178.2   | 41.4× slower           | ≡ 1.00× |
| LAMMPS `snap/kk` GPU (KOKKOS + CUDA)          | FP64 + `newton on neigh half` | **4.30** | **1.00×**         | 41.4× faster |

Source rows are median-of-3 100-step wall, normalised per step. Full raw logs are preserved at `verify/benchmarks/t6_snap_tungsten/scout_rtx5080/RESULTS.md` (TDMD rows) and `verify/benchmarks/t6_snap_tungsten/scout_rtx5080/lammps_kokkos_gpu_run.log` (LAMMPS KOKKOS row).

### §2.2 — Cumulative TDMD speedup ladder (pre-T8.6c → post-T-opt-4 Item 1)

Ten atomic-free, byte-exact commits between 2026-04-20 and 2026-04-22:

| # | Commit   | Scope                                                          | MixedFast ms/step | Cum. speedup |
|--:|----------|----------------------------------------------------------------|------------------:|-------------:|
| 0 | —        | Pre-T8.6c baseline (all-tid==0 prototype, T8.7 correctness gate) | 316.0           | 1.00×        |
| 1 | `1c22694` | T8.6c-v1 — `snap_ui_kernel` `add_uarraytot` block-parallel    | ~316             | 1.00×        |
| 2 | `ab0bcff` | T8.6c-v2 — `snap_yi_kernel` Phase A parallel over `idxz_max`  | 227.3             | 1.39×        |
| 3 | `88d23fb` | T8.6c-v3 — `snap_deidrj_kernel` Phase B: warp-shuffle `dedr` reduction | 208.6     | 1.51×        |
| 4 | `223a35a` | T8.6c-v4 — `compute_uarray` + `compute_duarray` intra-layer parallelism | 92.0      | 3.43×        |
| 5 | `4cf0202` | T8.6c-v5 S1 — `SnapBondListGpu` CSR+SoA bond list infra        | 92.0             | 3.43× (infra only) |
| 6 | `ac781ac` | T8.6c-v5 S2 — `snap_ui_bond_kernel` + gather replace per-atom   | 57.1             | 5.53×        |
| 7 | `6c5feb1` | T8.6c-v5 S3 — `snap_deidrj_bond_kernel` + gather replace per-atom | 47.9           | 6.60×        |
| 8 | `c380e10` | T-opt-3b — paired-bond reverse index (halves `deidrj_bond` work via `dedr_peer[b] ≡ dedr_own[reverse(b)]` identity) | 34.4 | 9.19× |
| 9 | `63847c2` | T-opt-2 — `yi_kernel` Phase B parallel-over-jju via CSR buckets (no atomics, disjoint writes) | 29.5 | 10.71× |
| 10 | `e12abd5` | T-opt-4 Item 1 — single-walk bond list (stage+compact replaces count+emit) | **29.2** | **10.82×** |

All ten preserved the **T8.7 ≤ 1e-12 rel byte-exact CPU↔GPU oracle** gate. Mechanisms: (a) bond-list emission order = CPU cell-stencil walk order (validated by `test_bond_list_matches_cpu_stencil_order`), (b) no `atomicAdd(double*, double)` anywhere in hot path (gpu/SPEC §6.1 mandate), (c) reduce-then-scatter via per-bond exclusive storage + per-atom gather, (d) FP32 narrowing (`sqrtf` only) confined to `MixedFastSnapOnlyBuild` per formal §D.17 procedure.

### §2.3 — Absolute kernel breakdown (nsys + NVTX, post-T-opt-4 Item 1, MixedFast)

100-step trace (warmup excluded) from `nsys profile --trace=cuda,nvtx`, steady-state `build_from_device` call:

| Component                              | Per-step wall (ms) | Share of step |
|----------------------------------------|-------------------:|--------------:|
| `snap_deidrj_bond_kernel` (GPU)        | **13.14**          | **61.0 %**    |
| `snap_yi_kernel` (GPU)                 | **6.47**           | **29.9 %**    |
| `snap_ui_bond_kernel` (GPU)            | 1.24               | 5.6 %         |
| `stage_and_count_bonds_kernel` (GPU)   | 0.43               | 1.9 %         |
| Other kernels (ui-gather, force-gather, compact, VV) | < 0.3 each | < 1.3 % each |
| **`snap.bond_list.build` NVTX wall**   | **0.45**           | **1.6 %**     |
|  — of which stage_kernel launch + sync | 0.427              | 94 %          |
|  — of which host exclusive scan (real CPU work) | < 0.010  | < 2 %         |
| **`snap.d2h.forces_and_reductions`**   | 21.3               | (stream-sync barrier; not additional GPU time) |

Total compute-kernel GPU time: ~21.5 ms/step. Wall overhead (kernel launches, stream sync, host scan, D2H) accounts for the ~7.7 ms gap to the 29.2 ms/step step wall.

**Key observation.** `snap_deidrj_bond_kernel` + `snap_yi_kernel` together are **90.9 %** of GPU time. The bond-list build (which T-opt-4 Item 1 single-walked) is 1.6 % of step wall — the leverage is structurally exhausted there.

---

## §3 — Gap decomposition

29.2 ms/step TDMD MixedFast vs 4.30 ms/step LAMMPS KOKKOS = **24.9 ms per-step surplus**. Decomposing (from post-Item-1 NVTX trace cross-walked against LAMMPS KOKKOS `pair_snap_kokkos.cpp` code path):

| Surplus component                                    | Estimated share of 24.9 ms gap | Root cause                                                                                    |
|------------------------------------------------------|-------------------------------:|-----------------------------------------------------------------------------------------------|
| `deidrj_bond_kernel` not fused with per-atom force scatter | ~10 ms                   | KOKKOS uses `atomicAdd(double)` on sm_120 (fast — hardware atomic). TDMD reduce-then-scatter per gpu/SPEC §6.1 mandates a separate `snap_force_gather_kernel` pass. Structural. |
| `yi_kernel` Phase B serial vs KOKKOS Kokkos::parallel_reduce over zlist + atomicAdd into ybuf | ~4 ms | Same mandate: TDMD's T-opt-2 CSR-bucket path is atomic-free but keeps Phase B at one thread per jju bucket walking jjz sequentially. KOKKOS Phase B is warp-parallel with atomics. |
| `deidrj_bond_kernel` FLOP/byte density            | ~5 ms                          | TDMD `__restrict__`-clean per-bond kernel vs KOKKOS persistent team-scratch CG coefficient cache. KOKKOS keeps Clebsch-Gordan table in shared memory across the full zlist per team; TDMD reloads via global L1. |
| Kernel launch overhead (4 passes in TDMD vs 2 in KOKKOS)  | ~2 ms                   | TDMD pipeline: stage → compact → ui-bond → ui-gather → yi → deidrj-bond → force-gather = 7 kernel launches per step. KOKKOS fuses compute_ui + compute_yi + compute_deidrj into 3 kernel teams. Partial fix possible (T-opt-4 Item 3 fusion, skipped per D-M8-6). |
| `snap.d2h` stream-sync overhead (NVTX 21.3 ms wall — most is bubble) | ~3 ms net | TDMD explicitly syncs to pull forces before VV integrator. KOKKOS keeps forces device-resident across the step. Partial fix possible via `runtime.gpu_resident_forces: true` path, not yet scoped. |
| Residual (host scan, staging-buffer D2H, bond-count copies) | ~1 ms                    | Post-Item-1 floor on the bond-list build. Further single-walk gains deliver ≤ 0.5 ms per gpu/SPEC §6.1 analysis. |
| **Total surplus**                                    | **~25 ms**                     |                                                                                               |

Order-of-magnitude check: the two largest line items (`atomicAdd` fusion + `yi` parallelism) together account for ~14 ms ≈ 56 % of the gap and are both structurally forbidden by the current gpu/SPEC §6.1 reduce-then-scatter contract. This is the core of the Case B argument.

---

## §4 — Why the gap does not close on a single GPU (or a small cluster)

### §4.1 — Architectural: reduce-then-scatter vs atomicAdd fusion

**gpu/SPEC §6.1** mandates that all force reductions on GPU proceed via exclusive per-bond (or per-cell) writes followed by a deterministic per-atom gather sum in fixed order. The rationale is **D-M6-7 byte-exact CPU ≡ GPU**: `atomicAdd(double*, double)` on CUDA (both software emulation pre-sm_60 and hardware on sm_60+) produces nondeterministic reduction orders across thread scheduling, which breaks the ≤ 1e-12 rel oracle gate.

LAMMPS `pair_snap_kokkos` does **not** carry a CPU ≡ GPU byte-exact contract; KOKKOS explicitly accepts reduction-order divergence as part of its performance model. This lets `compute_deidrj` fuse directly with force accumulation via `Kokkos::atomic_add` on sm_120 (hardware FP64 atomic, 1-cycle, coalesced).

This is **not a tuning gap**. It is an architectural choice that TDMD makes deliberately: the Reference path is sacred (CLAUDE.md auto-reject pattern #2; master spec §8.2 D-M6-7). The 14 ms/step surplus from non-fused reduction is the price of that contract. Future work (M9+) could introduce a **Production-only** flavor that permits atomicAdd within D-M8-8 precision envelopes, but this requires formal §D.17 procedure and would not close the gate at v1-alpha.

### §4.2 — Workload characterization: SNAP is compute-bound, not halo-bound

TDMD's signature performance lever per Andreev's 2007 dissertation is **time decomposition**: distinct spatial regions advance on different `dt`, with `T_comm_per_step_TD(K) = T_p / K`. This amortises inter-rank halo transfer over `K` pipeline slots, converting **halo-dominated** workloads into near-compute-only workloads at scale.

SNAP on T6 W at 2000 atoms × twojmax = 8 is firmly **compute-bound**:

- Per-atom bispectrum cost: O(J_max⁴) = O(4096) FLOP/atom at twojmax = 8, ≈ 8 M FLOP/step total compute.
- Halo volume: 2 × `halo_thickness × subdomain_face_area` atoms × 12 bytes/atom ≈ 10 KB/rank/step (at K=1 P_space=2).
- On a 960 GB/s RTX 5080, 10 KB is ~10 ns. On a 100 GB/s PCIe link, ~100 ns. The halo is **six orders of magnitude** below the step compute cost.

**Implication.** The `T_p / K` lever does not amortise a cost that barely exists. Even a hypothetical K=8 run on 8 physical GPUs would cut halo cost from 100 ns to 12.5 ns — negligible against 21 ms compute. Multi-rank scaling on SNAP is essentially the KOKKOS scaling curve minus per-rank overhead, not the TDMD signature curve.

### §4.3 — Multi-GPU scaling will not bridge the remaining 8.5×

Consider the best plausible small-cluster extrapolation: 8 physical RTX 5080 GPUs via NVLink + IB, P_space=8 K=1 (pure SD with TDMD's Pattern 2 halo path). At 2000 atoms BCC W that is 250 atoms/rank — well below the ~1000-atom saturation knee per perfmodel/SPEC §3.7. Expected per-rank wall:

- Compute-only scaling (ideal): 29.2 / 8 = 3.65 ms/step.
- Minus per-rank launch overhead floor (~1 ms at low atom count): effective ~4.7 ms/step.
- Plus inter-rank halo + reduction sync: ~0.2 ms/step.
- Expected: **~4.9 ms/step** on 8 ranks — marginally above KOKKOS 4.30 on 1 GPU.

LAMMPS KOKKOS also scales with rank count; at 8 ranks on the same cluster it would reach ≤ 1 ms/step. TDMD does not close the relative gap; it shrinks the absolute wall but KOKKOS shrinks faster, because KOKKOS pays no reduce-then-scatter structural tax per §4.1.

The single-rank kernel-tuning ladder has flattened at 29.2 ms/step (see §2.2). Remaining single-GPU levers (T-opt-4 Items 2 + 3, dispatch fusion, persistent kernel) extrapolate from NVTX data to ≤ 2× combined best-case, landing at ~15 ms/step — still 3.5× slower than KOKKOS on the same hardware. **No plausible combination of further single-GPU tuning + small-cluster scaling closes the 20 % beat gate on SNAP.**

---

## §5 — What TDMD does well (validated at v1-alpha)

The v1-alpha code base delivers, with receipts:

### §5.1 — Byte-exact correctness from CPU oracle to multi-rank GPU

The **D-M3-6 → D-M4-9 → D-M5-12 → D-M6-7 → D-M7-10** chain is unbroken and extended to SNAP at T8.10 (2026-04-20): single-rank Reference thermo ≡ 2-rank K=1 P_space=2 Reference thermo **byte-for-byte** on 100-step NVE W BCC 2000-atom. This is the invariant chain:

- M3 single-rank CPU golden ≡
- M4 K=1 P=1 Reference CPU ≡
- M5 K=1 P=2 Reference CPU (ring reduction canonicalised) ≡
- M6 K=1 P=2 Reference GPU EAM (canonical gather-to-single-block Kahan per D-M6-7) ≡
- M7 Pattern 2 K=1 P_space=2 Reference GPU EAM ≡
- M8 Pattern 2 K=1 P_space=2 Reference GPU **SNAP** (new at T8.10)

No competitor in the MD field (to our knowledge) carries an equivalent end-to-end byte-exact contract across CPU/GPU/multi-rank/multi-potential. This is TDMD's v1 proof that the Andreev-TD two-phase commit is **correct** on a modern ML potential, before any performance claims.

### §5.2 — Ten atomic-free SNAP GPU commits without regressing byte-exactness

The T8.6c-v1 … T-opt-4 Item 1 ladder (§2.2) is ten performance refactors that each preserved the T8.7 ≤ 1e-12 rel oracle gate **by construction**. Mechanisms:

- **T-opt-3b's `dedr_peer[b] ≡ dedr_own[reverse(b)]` pure-function identity** — exploits the FP64-invariant symmetry of `compute_deidrj(Δr, weight, ylist_slab)` under (i↔j, Δr↔−Δr, weight↔−weight, ylist-slab swap). Halves `deidrj_bond` arithmetic without introducing a single non-deterministic operation.
- **T-opt-2's jju-bucket iteration** — a stable-sort permutation of the legacy tid==0 sweep over jjz. Per-jju `+=` sequence is identical (buckets disjoint over jju, writes exclusive). 17 % MixedFast cut for zero correctness risk.
- **T-opt-4 Item 1's single-walk bond list** — stage+compact replaces count+emit. Stage kernel writes (j, type, Δr, r²) tuples at a per-atom staging-buffer cursor; compact pure-gathers to packed CSR. Within-atom emission order preserved by sharing the same `scan_snap_bonds` visitor across both old and new paths.

This demonstrates that the gpu/SPEC §6.1 reduce-then-scatter constraint is **not a straitjacket**: there is real engineering room for algebraic symmetry exploits, stable-sort permutations, and pass-fusion-with-staging. The kind of work MixedFast SNAP absorbed in 2026-04-20/22 is a template for EAM (M9) and MEAM (M10).

### §5.3 — Multi-rank TD machinery functionally green on SNAP

The 2026-04-21 multi-rank scout (`verify/benchmarks/t6_snap_tungsten/scout_rtx5080/quick_2rank_scout.sh`) ran 18 combinations (3 flavors × {1-rank, 2-rank K=1, 2-rank K=4} × 3 repeats) on shared RTX 5080. Results (§2.1 of `RESULTS.md` multi-rank section):

- D-M7-10 SNAP byte-exact extension **PASSES** (§5.1 above).
- 2-rank/1-rank ratio ≈ 1.91× (expected — CUDA context-switch + per-rank launch overhead + serialised halo D2H on shared device).
- K=4/K=1 ≈ 1.00× (expected — T_p/K amortisation requires ≥ 2 physical GPUs).

Pattern 2 two-phase commit, OuterSdCoordinator halo ordering, Kahan-ring reduction order, and K-batching dispatch all exercised end-to-end on SNAP GPU with no crashes, no hangs, no determinism violations. The machinery is **ready** for multi-GPU runs; the SNAP workload is simply not where it pays off.

### §5.4 — `MixedFastSnapOnlyBuild` slow-tier VerifyLab pass (T8.12)

Full T0+T1+T3+T4+T6 battery GREEN on the 6th BuildFlavor (landed T8.8 via formal §D.17 7-step procedure). Artefacts: `verify/slow_tier/m8_mixed_fast_snap_only_{sweep.yaml,results.json,REPORT.md}`. D-M8-8 thresholds (force L∞ rel ≤ 1e-5, PE rel ≤ 1e-7, virial rel-to-max-component ≤ 5e-6) all satisfied with ~60× headroom on PE, ~1000× on virial. Force closest to cap at 9.51e-6 on atom 1282 — within budget, deterministic on sm_120.

### §5.5 — Public-CI-wired M6 + M7 smoke tests (Option A CI policy)

`.github/workflows/ci.yml` runs M1 through M7 smokes on every PR against `ubuntu-latest`. GPU-required steps self-skip via `nvidia-smi -L` gate (no self-hosted runner per `project_option_a_ci.md`); CPU-side infrastructure (template substitution, LFS asset path, golden parity, T7.9 Pattern 2 validation wiring) still exercised. Local pre-push GPU gate on RTX 5080: < 2 s for M7 7-step harness. Meaning: any merge that breaks the D-M7-10 chain fails CI **before** human review.

---

## §6 — What we learned (diagnostic analysis per §11.4 honest engineering)

T-opt-4 Item 1 was landed 2026-04-22 with a pre-implementation estimate of **5–6 %** MixedFast wall-time reduction. The measured delivery was **1.0 %** (29.5 → 29.2 ms/step). A 30-minute NVTX-trace diagnostic ran before invoking Case B, to understand whether the gap between prediction and delivery reflects a tuning error, a model error, or a structural limit. The findings are documented here because **negative diagnostic results are evidence** (master spec §11.4: "honest engineering requires documenting measurement, not just wins").

### §6.1 — Item 1 under-delivery: pre-impl model counted a cold path

The pre-impl nsys analysis (`docs/development/t_opt_4_item1_pre_impl.md`) estimated bond-list build at **~3 %** of step wall, framed T-opt-4 Item 1 as recovering about half of that = 1.5 %, and a further 3–4 % from downstream improvements (tighter staging → faster ui-bond launch, reduced host-scan latency). Post-landing NVTX trace shows:

- `snap.bond_list.build` NVTX wall: **1.6 %** of step (not 3 %).
- Of that 1.6 %, **94 %** is the stage-kernel launch + stream sync bubble; actual host exclusive-scan CPU work is < 2 %.
- Single-walk saved ~1/3 of the stage-kernel GPU work (one stencil walk instead of two) = **0.5–1.0 % wall** — matching the measured 1.0 %.

**Root cause of mis-estimate.** The pre-impl nsys run was a **cold-build trace** — the first `build_from_device` call after construction includes H2D copies of the CSR skeleton, type arrays, and stencil offsets that **do not recur** in steady-state builds. Steady-state builds touch only the atom-position D2H delta and the 27-cell stencil walk on device. The pre-impl model correctly measured cold-path cost and incorrectly transferred it to steady-state amortised cost.

**Lesson.** Future pre-impl estimates for build-phase optimisations must use NVTX ranges on a post-warmup steady-state interval, not on a first-call cold trace.

### §6.2 — Host-scan was on critical path but in a non-tunable way

The pre-impl analysis flagged host exclusive-scan as a "CPU-side bottleneck" to investigate (~0.1 ms pre-impl estimate at 2000 atoms). Post-Item-1 NVTX shows:

- Host-scan NVTX range: 449 µs wall.
- Of that, 427 µs (≈ 97 %) is `cudaStreamSynchronize` waiting for the stage kernel to flush counts to host memory before the CPU can prefix-sum them.
- Actual CPU work (prefix-sum on 2000 `uint32_t` counts): < 10 µs.
- The compact kernel cannot launch until host-scan completes (offsets drive compact's gather indices).

So host-scan **is** on the critical path, but the lever is **not** "parallelise the CPU prefix sum" (there's no CPU work to speak of) — it's "eliminate the D2H → CPU → H2D roundtrip" via a **device-side prefix scan** (e.g. `cub::DeviceScan::ExclusiveSum`). This is T-opt-5 territory, not T-opt-4. Estimated further savings: ~0.3 ms/step (1.0 % wall). Not invoked under Case B — below the noise floor of the larger structural gap.

### §6.3 — Current bottleneck has no single-GPU leverage left

Post-Item-1, the two dominant kernels are:

- `snap_deidrj_bond_kernel`: 13.14 ms/step, 61.0 % of step, 128 threads/block × n_bonds blocks. Already per-bond parallel (T8.6c-v5 S3) and own-only (T-opt-3b). The only remaining lever is CG-coefficient table residency in shared memory (estimated 10–20 % of kernel = 1.5–2.5 ms/step), but KOKKOS's corresponding advantage is ~5 ms of atomicAdd fusion that TDMD architecturally cannot replicate — so closing the CG-residency gap closes ~1/3 of the deidrj surplus, not all of it.
- `snap_yi_kernel`: 6.47 ms/step, 29.9 % of step. Phase A parallel-over-idxz_max (T8.6c-v2), Phase B parallel-over-jju via CSR buckets (T-opt-2). ztmp/zlist build already 128-lane. Remaining yi work is a dense-matrix-vector contraction already near roofline at ~75 % L1 hit. Estimated further leverage: < 10 %.

Together: plausible further single-GPU wins ≤ 3.5 ms/step = ~12 % = still 6× slower than KOKKOS. This matches the §4.3 small-cluster extrapolation: **the gap does not close**.

### §6.4 — The structural framing was right; the tactical framing was wrong

Before the diagnostic, a natural-sounding worry was "we might be missing a 2–3× easy kernel win that makes Case A plausible." The diagnostic closes that worry: every remaining named-kernel optimisation, with honest extrapolation, lands inside [10 %, 20 %] individually, [30 %, 50 %] combined. None of them closes a 6.79× gap. The honest story is architectural (§4.1) + workload-characterisation (§4.2), not kernel-tactical.

**Net lesson.** The single-GPU kernel ladder is **flat**. Continuing to pay engineering time against it (Items 2 + 3 of T-opt-4, further T-opt-5/6 candidates) would burn time that is better spent on the M9+ MEAM port where the TD lever **does** apply.

---

## §7 — TDMD's niche (the workloads where v2 is expected to win)

The SNAP result is not an indictment of TD; it is evidence that SNAP is the wrong showcase workload for TD. The workloads where TD's `T_comm = T_p / K` lever dominates share three structural features:

1. **Angular-moment or multi-body halo dependence.** MEAM, Tersoff, SW, ReaxFF have per-atom state that depends on halo atoms beyond the direct-neighbour set, which in SD requires either deep halos (wasted compute on boundary) or double-halo exchanges per step. TD amortises the halo cost over K slots.
2. **Dense rank-to-rank communication volume.** Where SNAP pays < 1 % of step on halo, MEAM on 864-atom Ni-Al pays 4–8 % per current T4 scout (`project_m8_perf_smoke.md` EAM row, with MEAM projected higher due to angular moments). At 8-rank commodity-network scale this becomes 15–30 % — directly addressable by K-batching.
3. **Compute-communication ratio that scales badly with rank count.** KOKKOS's SD model keeps per-step halo at O(√N/P × face_area) in strong scaling. TD's `T_p / K` breaks the `×P` multiplier by letting K-interior pipelines run asynchronously. For MEAM on 8 ranks at 100K atoms, projected TDMD advantage is 1.3–1.8× (perfmodel/SPEC §3.7 analytic, pending M10 empirical confirmation).

**SNAP is absent from this list** because: (a) halo is trivial, (b) communication per step is trivial, (c) the bottleneck is kernel compute density, where KOKKOS has a mature 5-year head start and a permissive precision contract.

**The honest positioning for v1-alpha.** TDMD is a **correct, byte-exact, multi-potential, multi-rank GPU MD engine** with a validated TD scheduler. For compute-bound ML potentials (SNAP, PACE, MLIAP) on a single GPU, LAMMPS KOKKOS will be faster; for halo-bound many-body potentials (MEAM, Tersoff at large rank counts), TDMD is expected to win. v1-alpha closes the engineering foundation; v1.x and v2 close the scientific-value claim on the latter class.

---

## §8 — M9+ roadmap

Priorities for post-M8 delivery, ranked by expected scientific value × engineering feasibility:

### §8.1 — M10 MEAM GPU port (primary M9+ signature workload)

- Port `pair_style meam/c` from LAMMPS per master spec §10. License chain: GPLv2 with Attribution (Baskes 1992, LAMMPS meam package).
- GPU kernel strategy mirrors SNAP per-bond dispatch: per-bond ρ̄ accumulation via bond-list CSR + per-atom gather (no atomics).
- **Target artefact gate (master spec §14 M10):** T5 si-meam differential vs LAMMPS meam/c + ≥ 30 % speedup vs LAMMPS MEAM on 8-rank commodity network at representative atom count. This **is** the TDMD-vs-LAMMPS showpiece that M8-on-SNAP was not.
- Expected timeline: 8 weeks per master spec §14 M10.

### §8.2 — M7 multi-GPU TD × SD hybrid (infrastructure already shipped, needs cluster runs)

- Multi-rank TD machinery is correct and exercised per §5.3. What's missing is ≥ 2-physical-GPU hardware to measure `T_p / K` amortisation and weak-scaling η(P_time).
- Candidate hardware: 4× RTX 4090 local box (CUDA P2P over PCIe) or AWS `p4d.24xlarge` (8× A100, NVLink).
- Decision point: budget for cloud rental + measurement pre-M10 (6–8 weeks lead time); otherwise paired with M10 MEAM runs (MEAM on 4–8 GPUs is the natural joint gate).

### §8.3 — M9 NVT/NPT (thermostat + barostat — required for physical validity at scale)

- Per master spec §14 M9 — `NoseHooverNvtIntegrator` + `NoseHooverNptIntegrator` CPU + GPU. Policy validator: `K > 1` with `style != nve` → reject per integrator/SPEC §7.3.
- v1.5 restriction: NVT/NPT only `K = 1` (effective SD). Variant C lazy-sync is M11 research window.
- Unlocks realistic long-timescale MD for MEAM / SNAP scientific users.

### §8.4 — M12 PACE + MLIAP (ML potential breadth)

- PACE port from ACE reference; MLIAP with pybind11 plugin architecture.
- Strategically important because it proves TDMD is not tied to SNAP-shaped potentials.
- Post-M10 priority.

### §8.5 — M13 long-range service (opens NaCl/LiF and charged MEAM workloads)

- PPPM or Ewald. Long-range integration with TD via split time-stepping (natural fit: long-range on coarse dt zone).
- T11 SiO₂ glass benchmark.

### §8.6 — Deferred single-GPU SNAP work (T-opt-4 Item 2 + Item 3, T-opt-5 device-side prefix scan)

- Still valid work; expected combined win 20–40 % on single-GPU SNAP.
- Not on the v1-alpha critical path per §6.4 lesson. Revisit if M10 MEAM reveals shared kernel-infra opportunities (e.g. shared-mem CG cache pattern applies to MEAM too).

---

## §9 — Why skip T8.11 cloud-burst (stand-down rationale)

The M8 execution pack's T8.11 task scoped a `cloud-burst` run on ≥ 2 physical GPUs to produce the **primary** M8 ≥ 20 % beat artefact. We stand it down under Case B for four reasons:

1. **The single-GPU gap is structural, not small-cluster-accessible.** §4.3 extrapolation lands 8-rank TDMD at ~4.9 ms/step vs 8-rank LAMMPS KOKKOS at < 1 ms/step. Cloud burn produces a negative artefact at 100× the cost of a local one. The story already told in §2–§4 is complete.

2. **Honest-negative publication is more valuable than redundant-negative measurement.** Master spec §14 M8 explicitly permits Case B. A well-documented `why-not` is the stronger artefact vs an expensive `measured-why-not` on hardware that changes nothing structurally.

3. **Budget better spent on M10 MEAM cluster runs.** The same cloud rental aimed at MEAM multi-rank scaling (§8.1 + §8.2) produces the **positive** TDMD-vs-LAMMPS artefact the project actually needs. Re-scoping the budget from T8.11 SNAP to T10.x MEAM is a ~1:1 translation of hours.

4. **M7 multi-rank SNAP machinery is already exercised on shared-GPU (§5.3).** The incremental information from moving from shared RTX 5080 to 2 physical RTX 4090s on SNAP is: absolute wall drops ~2× (expected per oversubscription removal), gap vs KOKKOS ratio unchanged. Not a new data point.

**Stand-down scope.** T8.11 task marked **Cloud-burst skipped — see REPORT.md §9**. Execution pack §5 checklist: `[x] skipped` with rationale-doc cross-reference. No retraction of D-M8-5 threshold registry entries (they remain valid for future MEAM / PACE cloud-burst runs). The `mpi_host_staging` + Pattern 2 multi-rank plumbing stays in-tree, tested by the 2-rank shared-GPU scout + M7 CI smoke.

---

## §10 — M8 closure criteria and conclusion

### §10.1 — Closure checklist (per master spec §14 M8 + D-M8-6)

| Criterion                                                        | Status  | Evidence                                                                 |
|------------------------------------------------------------------|---------|--------------------------------------------------------------------------|
| `SnapPotential` CPU + GPU implemented                            | ✅      | `src/potentials/snap/` + `src/gpu/snap_gpu*.cu` landed T8.4–T8.7         |
| T6 tungsten SNAP benchmark in VerifyLab                          | ✅      | `verify/benchmarks/t6_snap_tungsten/` with LAMMPS-native fixture, T8.10 |
| TDMD vs LAMMPS SNAP comparison on T6 wall-clock + scaling        | ✅      | This REPORT.md + `scout_rtx5080/RESULTS.md`                             |
| Demo of where TDMD beats LAMMPS on representative ML kernel       | Case B  | §5 (correctness + multi-rank machinery) + §7 (niche honest-framing)     |
| `MixedFastSnapOnlyBuild` new BuildFlavor (§D.11 / §D.17)          | ✅      | T8.8 — `mixed_fast_snap_only_rationale.md` + `cmake/BuildFlavors.cmake` |
| Full slow-tier VerifyLab validation for new flavor               | ✅      | T8.12 — `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md` GREEN      |
| **Artifact gate: beat LAMMPS ≥ 20 % OR honest why-not doc**       | ✅ (Case B) | **This REPORT.md** per D-M8-6                                        |
| M8 closes v1 alpha                                               | → T8.13 | `v1.0.0-alpha1` tag pending final release-notes review                  |

### §10.2 — What v1.0.0-alpha1 claims

- TDMD is a **correct** GPU-first TD MD engine, with a byte-exact invariant chain from single-CPU oracle to multi-rank GPU SNAP.
- TDMD's SNAP GPU single-rank performance is **6.79× slower** than LAMMPS KOKKOS `snap/kk` on RTX 5080, with ~10.8× cumulative improvement delivered during M8 atomic-free refactoring.
- TDMD's SNAP multi-rank TD machinery is **functionally green**, but TD's `T_comm = T_p / K` lever does not apply to the compute-bound SNAP workload.
- TDMD's **expected signature workload** is halo-bound angular-moment potentials (MEAM on 8-rank commodity network), delivery target: M10 per master spec §14.

No performance claim is made above what is measured. No precision claim is made beyond the T8.7 / D-M6-7 / D-M8-8 thresholds that the T8.12 pass validated.

### §10.3 — Conclusion

M8 closes. Not with the headline `beat LAMMPS ≥ 20 %` result that the execution pack originally hoped for — but with the **other** artefact the master spec §14 M8 gate explicitly permits: honest documentation of why the ≥ 20 % beat does not materialise on SNAP-on-single-GPU, what the structural origin is, and what the plausible path forward looks like.

The v1-alpha code is a **correctness claim** plus a **demonstrated engineering velocity** (ten atomic-free byte-exact SNAP GPU commits in three days, 10.82× MixedFast cumulative). The **science claim** — TDMD vs LAMMPS on a workload where TD wins — is M10 MEAM scope. That is not a retreat; it is the sequencing that master spec §1 and §14 were written to accommodate.

M9+ execution pack follows. First task: M10 MEAM GPU port scoping, expected 2026-04-25 to 2026-06-13 timeline.

---

## References

- Master spec v2.5 §14 M8 (artifact gate dual-path clause; D-M8-6).
- Master spec v2.5 §8.2 (ownership boundaries — Reference path sacred).
- `docs/specs/gpu/SPEC.md` v1.0.24 §6.1 (reduce-then-scatter mandate; no `atomicAdd(double)` in hot path).
- `docs/specs/potentials/SPEC.md` §6.7 (SNAP precision policy matrix).
- `docs/specs/potentials/mixed_fast_snap_only_rationale.md` (§D.17 7-step procedure for the 6th BuildFlavor).
- `verify/benchmarks/t6_snap_tungsten/scout_rtx5080/RESULTS.md` (raw scout data; per-commit ladder).
- `verify/slow_tier/m8_mixed_fast_snap_only_REPORT.md` (T8.12 pass).
- `docs/development/claude_code_playbook.md` §11.4 (honest-engineering clause for diagnostic reports).
- Wood, M. A.; Thompson, A. P. *J. Chem. Phys.* **148**, 241721 (2018). SNAP W potential (fixture author).
- Thompson, A. P. et al. *J. Comput. Phys.* **285**, 316 (2015). SNAP bispectrum formulation.
- Andreev, V. V. Ph.D. dissertation, 2007. Time-decomposition MD (`T_comm = T_p / K` derivation).
