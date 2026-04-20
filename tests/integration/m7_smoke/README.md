# M7 integration smoke — Pattern 2 K=1 P_space=2, byte-exact to M6

<!-- markdownlint-disable MD013 -->

This smoke is the acceptance gate for the M7 milestone (master spec §14
M7). It exercises the M7 user surface end-to-end — `zoning.subdomains:
[2,1,1]` Pattern 2 two-level TD×SD scheduling on top of the M4/M5/M6
deterministic TD scheduler + MpiHostStaging transport + GPU runtime —
in a 2-rank host-staged configuration (each rank owns one subdomain).

It is **not** a physics oracle. Numerical correctness of the EAM path
was validated at M2 by the T4 differential against LAMMPS, and the GPU
reduction order was validated at M6. This harness is a **determinism**
check: in Pattern 2 K=1 P_space=2 with `runtime.backend: gpu`, does the
Reference-flavor run produce thermo byte-identical to the M6 (= M5 =
M4 = M3) golden?

## The D-M7-10 contract

Execution pack §T7.14 and scheduler/SPEC Pattern 2 integration section
define the byte-exact acceptance gate:

> With `runtime.backend: gpu`, `scheduler.td_mode: true`,
> `pipeline_depth_cap: 1`, `comm.backend: mpi_host_staging`,
> `zoning.subdomains: [2, 1, 1]`, K=1 on 2 ranks, a
> `Fp64ReferenceBuild` binary MUST produce thermo byte-identical to the
> M6 golden (D-M6-7 chain), which in turn equals the M5 golden = M4
> golden = M3 golden.

The invariant chain:

- D-M5-9 (Kahan-ring reduction-order determinism for thermo tallies).
- D-M6-7 (GPU Reference force reduction canonicalised to match CPU
  summation order — gather-to-single-block + sorted Kahan-add).
- D-M7-10 (Pattern 2 K=1 degenerates to Pattern 1 spatial decomposition;
  the two-level scheduler with `pipeline_depth_cap: 1` produces the
  same DAG topology and the same halo arrival ordering as Pattern 1,
  extended by the `OuterSdCoordinator` peer-halo canonicalisation —
  R-M7-5, per-peer `(peer_subdomain_id, time_level)` sort before
  release).

Drift here is a bug in exactly one of those layers.

**The M7 `thermo_golden.txt` is the M6 `thermo_golden.txt`**, copied
verbatim (which is the M5 = M4 = M3 golden). The harness checks that
the two golden files match bit-for-bit before running the binary — this
defends against someone editing one without syncing the others.

## What the harness does

7 steps, short-circuiting on the first failure.

| # | Check                                                            | Exit |
|---|------------------------------------------------------------------|------|
| 1 | Pre-flight: M7 golden == M6 golden (D-M7-10 chain)               | 2    |
| 2 | Local-only gate: `nvidia-smi` reports ≥1 GPU, else SKIP          | 0    |
| 3 | Single-rank `tdmd validate` — Pattern 2 preflight (T7.9)          | 2    |
| 4 | `mpirun --np 2 tdmd validate <config>` — 2-rank Pattern 2 config  | 2    |
| 5 | `mpirun --np 2 tdmd run --telemetry-jsonl` exits 0 < 60 s         | 2/3  |
| 6 | Thermo byte-matches M7 golden (= M6 = M5 = M4 = M3 golden)        | 1    |
| 7 | Telemetry invariants (run_end, wall-time, boundary_stalls_total=0)| 1    |

Step 3 is the **Pattern 2 preflight** — `SimulationEngine::preflight`
(T7.9) validates that `zoning.subdomains` tiles the box under the
resolved rank grid. It runs in single-rank mode first (fast-fail before
paying MPI setup cost) and then is re-validated under the real 2-rank
launch.

Step 2 is the **Option A gate**: the smoke SKIPs on any host that does
not expose a CUDA device. This is the explicit escape hatch for CI
(D-M6-6: no self-hosted GPU runner) — the harness still runs
end-to-end so infrastructure rot (template typo, missing LFS asset,
broken golden parity) fails loudly even when the GPU path cannot
execute.

## Why this is local-only

D-M6-6 explicitly forbids a public-repo self-hosted GPU runner
(reused in Option A for M7). The `build-gpu` CI matrix (T6.12) covers
compile + link only. The smoke lives in-tree so developers can invoke
it locally before pushing:

```bash
tests/integration/m7_smoke/run_m7_smoke.sh \
  --tdmd build/src/cli/tdmd
```

and it is also the **mandatory pre-merge local gate** for any M7+
change touching `src/scheduler/`, `src/zoning/` (subdomain support),
`src/comm/hybrid_backend*`, `src/comm/gpu_aware_mpi*`,
`src/comm/nccl_backend*`, `src/runtime/pattern2_wire*`, or
`src/perfmodel/pattern2_cost*`.

Companion local gates that must also stay green on M7+ PRs (D-M7-17
regression preservation):

- M1..M6 integration smokes (this directory's siblings).
- T1 / T4 differentials vs LAMMPS (`tests/potentials/`).
- T3-gpu EAM-substitute efficiency curve
  (`tests/integration/m5_anchor_test/` + T7.12 extension).

## Flags / env

- `--tdmd <path>` — path to the `tdmd` binary (required; must be built
  with `-DTDMD_ENABLE_MPI=ON -DTDMD_BUILD_CUDA=ON`).
- `--mpirun <path>` — alternate MPI launcher; defaults to `mpirun`.
- `--keep-workdir` — preserve the tmp workdir on success.
- `TDMD_UPDATE_GOLDENS=1` — overwrite the local golden instead of
  comparing. Commit only after Validation Engineer review. **Updating
  the M7 golden without also updating M6's (and M5/M4/M3) breaks
  D-M7-10.**
- `TDMD_SMOKE_BUDGET_SEC=N` — override the default 60 s wall-time
  budget (M7 is comparable to M6 because K=1 means Pattern 2
  degenerates to Pattern 1 spatial decomposition at runtime; the
  Pattern 2 scheduler overhead is a constant-factor startup cost).

## Build-flavor coverage

The harness runs exactly one tdmd binary per invocation. Reference
coverage (D-M7-10 byte-exact) is the default. `MixedFastBuild`
coverage is deliberately **not** layered here: D-M6-8 / D-M7 mixed
thresholds are owned by `tests/potentials/test_t4_differential_mixed.cpp`
and T7.11's scaling benchmark.

## When this smoke fails

- **exit 1 from step 6** — thermo divergence against M7 golden. The
  Pattern 2 path is perturbing bits. Check (in order): scheduler
  `OuterSdCoordinator::release_boundary()` still sorts peer halos by
  `(peer_subdomain_id, time_level)` before commit (R-M7-5 / D-M7-10);
  `SubdomainBoundaryDependency::emit_boundary()` still uses the
  canonical Kahan-ring reduction; GPU reduction order in
  `src/gpu/eam_alloy_gpu.cu` unchanged (D-M6-7 chain); no stray atomic
  add in force accumulation at subdomain boundary; BuildFlavor is
  Fp64Reference (the harness auto-detects MixedFast via telemetry and
  exits 2 with a clear diagnostic).
- **exit 1 from step 7** — telemetry invariant broken. Most likely:
  unbalanced `ScopedSection` from a Pattern 2 path (RAII hole around
  an MPI call that threw), OR non-zero `boundary_stalls_total`
  indicating `OuterSdCoordinator` watchdog fired (D-M7-14) — inspect
  the JSONL line.
- **exit 2** — infra. Missing `mpirun`, `tdmd` built without CUDA or
  MPI, LFS assets not pulled, or M7/M6 golden parity broken (someone
  edited one without syncing).
- **exit 3** — wall-time blowout. 60 s budget; 2-rank 2-subdomain
  10-step Pattern 2 K=1 run on 864 atoms should stay under ~5 s on
  commodity hardware. A spike almost always means a debug build, a
  hang in `OuterSdCoordinator::wait_for_boundary()` (watchdog not
  tripping fast enough), or a kernel that silently fell into a D2H +
  CPU loop.

## Scope boundaries

- **Not covered here**: K>1 Pattern 2 (time-dimension pipelining).
  Pattern 2 at K>1 is M9+ territory (master spec §14 M9); this smoke
  locks in only the K=1 spatial-decomposition degeneracy.
- **Not covered here**: Multi-node Pattern 2 (≥2 nodes). Honorable
  best-effort probe lives in T7.11's scaling benchmark, not here
  (D-M7-8).
- **Not covered here**: HybridBackend / NCCL paths (T7.4 / T7.5).
  Those have their own byte-exact vs MpiHostStaging gates in unit
  tests; this smoke locks the Reference+MpiHostStaging path only for
  the D-M7-10 chain.
- **Not covered here**: PerfModel Pattern 2 ±25% gate (D-M7-9 /
  T7.13b). Orthogonal local gate; T7.13 shipped Pattern 1 ±20%.
- **Not covered here**: 30% compute/mem overlap gate — owned by
  `tests/gpu/test_overlap_budget*.cpp` (T7.8 single-rank shipped;
  2-rank extension tracked as T7.8 carry-forward in the exec pack).
- **Not covered here**: SNAP / MEAM / PACE potential coverage. M8+
  physics work.
