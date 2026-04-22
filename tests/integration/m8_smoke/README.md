# M8 integration smoke — Pattern 2 K=1 P_space=2 SNAP, byte-exact to 1-rank

<!-- markdownlint-disable MD013 -->

This smoke is the acceptance gate for the M8 milestone (master spec §14
M8). It is the SNAP analog of `m7_smoke`, extending the D-M7-10
byte-exact chain from EAM/alloy to SNAP — the D-M8-13 acceptance gate.

It exercises the M8 user surface end-to-end — `potential.style: snap`
driven through the M7 Pattern 2 two-level TD × SD scheduler +
MpiHostStaging transport + GPU runtime — in a 2-rank host-staged
configuration on the T6 canonical fixture (1024-atom W BCC, `W_2940_2017_2`
Wood 2017 coefficients).

It is **not** a physics oracle. Numerical correctness of the SNAP path
was validated at M8 T8.5 by the T6 differential against LAMMPS (D-M8-7
byte-exact, 250-atom fixture), and the GPU reduce-then-scatter order
was validated at T8.7. This harness is a **determinism** check: does
the 2-rank Pattern 2 K=1 P_space=2 run produce thermo byte-identical to
a 1-rank K=1 P=1 run of the same config?

## The D-M8-13 contract

Execution pack §T8.13 and scheduler/SPEC Pattern 2 integration define
the byte-exact acceptance gate:

> With `runtime.backend: gpu`, `scheduler.td_mode: true`,
> `pipeline_depth_cap: 1`, `comm.backend: mpi_host_staging`,
> `zoning.subdomains: [2, 1, 1]`, K=1 on 2 ranks, a
> `Fp64ReferenceBuild` binary with `potential.style: snap` MUST produce
> thermo byte-identical to the same binary running the same config with
> the `zoning` section removed (1-rank K=1 P=1). This is the D-M7-10
> byte-exact chain extension from EAM to SNAP.

The invariant chain for this smoke:

- D-M5-9 (Kahan-ring reduction-order determinism for thermo tallies).
- D-M8-7 (CPU Fp64 byte-exact SNAP oracle against LAMMPS — T6
  differential at 250 atoms).
- D-M7-10 (Pattern 2 K=1 degenerates to Pattern 1 spatial decomposition;
  OuterSdCoordinator peer-halo canonicalisation + Kahan-ring reduction).
- D-M8-13 (Pattern 2 K=1 on SNAP reproduces 1-rank thermo bit-for-bit —
  extends D-M7-10 from linear-reduction EAM to angular-moment SNAP
  bispectrum force, both with reduce-then-scatter GPU force order per
  gpu/SPEC §6.1).

Drift here is a bug in exactly one of those layers.

## Golden file

Unlike `m7_smoke` — whose `thermo_golden.txt` is a verbatim copy of
the M6 (= M5 = M4 = M3) golden — SNAP has no prior milestone-chain
golden to inherit. The M8 golden is generated **by the harness itself**
at bring-up: under `TDMD_UPDATE_GOLDENS=1` the harness strips the
`zoning` section from the instantiated config, runs a 1-rank
Fp64Reference pass, and writes that thermo to `thermo_golden.txt`.

Subsequent harness invocations compare the 2-rank Pattern 2 K=1
thermo against that 1-rank golden. The D-M8-13 byte-exact check is
therefore self-consistent: regenerating the golden on a broken build
would be caught on the same run by the 2-rank pass that follows.

## What the harness does

7 steps, short-circuiting on the first failure.

| # | Check                                                                     | Exit  |
|---|---------------------------------------------------------------------------|-------|
| 1 | LAMMPS submodule probe — `W_2940_2017_2.snap*` fixture present            | 77    |
| 2 | LFS probe + local-only gate: `nvidia-smi` reports ≥1 GPU, else SKIP       | 0 / 2 |
| 3 | Single-rank `tdmd validate` — Pattern 2 preflight (T7.9)                  | 2     |
| 4 | `mpirun -np 2 tdmd validate <config>` — 2-rank Pattern 2 config           | 2     |
| 5 | `mpirun -np 2 tdmd run --telemetry-jsonl` exits 0 < 120 s                 | 2 / 3 |
| 6 | Thermo byte-matches M8 golden (D-M8-13 1-rank oracle)                     | 1     |
| 7 | Telemetry invariants (run_end, wall-time, boundary_stalls_total=0, …)     | 1     |

Step 1 is the **submodule gate**: SNAP coefficient files ship via the
LAMMPS git submodule (D-M8-3; no TDMD-side binary tracked). Absent
submodule → `exit 77` (Catch2 `SKIP_RETURN_CODE`), matching
`test_lammps_oracle_snap_fixture`.

Step 3 is the **Pattern 2 preflight** — `SimulationEngine::preflight`
(T7.9) validates that `zoning.subdomains` tiles the box under the
resolved rank grid. It runs in single-rank mode first (fast-fail before
paying MPI setup cost) and then is re-validated under the real 2-rank
launch.

Step 2 is the **Option A gate**: the smoke SKIPs on any host that does
not expose a CUDA device. This is the explicit escape hatch for CI
(D-M6-6: no self-hosted GPU runner) — the harness still runs
end-to-end so infrastructure rot (template typo, missing LFS asset,
submodule drift) fails loudly even when the GPU path cannot execute.

## Why this is local-only

D-M6-6 explicitly forbids a public-repo self-hosted GPU runner
(reused in Option A for M7 / M8). The `build-gpu` CI matrix (T6.12)
covers compile + link only. The smoke lives in-tree so developers can
invoke it locally before pushing:

```bash
tests/integration/m8_smoke/run_m8_smoke.sh \
  --tdmd build/src/cli/tdmd
```

and it is also the **mandatory pre-merge local gate** for any M8+
change touching `src/potentials/snap*`, `src/gpu/snap*.cu`,
`src/scheduler/` (Pattern 2 integration), `src/comm/hybrid_backend*`,
or `src/runtime/pattern2_wire*`.

Companion local gates that must also stay green on M8+ PRs (D-M8-14
regression preservation):

- M1..M7 integration smokes (this directory's siblings).
- `tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh` (10-step NVE
  drift gate on the same T6 1024-atom fixture — physics-side
  check, single-rank).
- T1 / T4 / T6 differentials vs LAMMPS (`tests/potentials/`).

## Flags / env

- `--tdmd <path>` — path to the `tdmd` binary (required; must be built
  with `-DTDMD_ENABLE_MPI=ON -DTDMD_BUILD_CUDA=ON`; BuildFlavor must be
  `Fp64ReferenceBuild` — the harness assumes the oracle flavor).
- `--mpirun <path>` — alternate MPI launcher; defaults to `mpirun`.
- `--keep-workdir` — preserve the tmp workdir on success.
- `TDMD_UPDATE_GOLDENS=1` — regenerate the golden. Runs a 1-rank
  Fp64Reference pass (config with `zoning` section stripped), copies
  its thermo to `thermo_golden.txt`, and then proceeds with the
  normal 2-rank verification pass. The diff step then doubles as a
  D-M8-13 check on the freshly-generated golden, so update-goldens
  mode is itself a byte-exact assertion.
- `TDMD_M8_SMOKE_BUDGET_SEC=N` — override the default 120 s wall-time
  budget (Fp64Reference SNAP on 1024 atoms at 10 steps fits well
  within that envelope on commodity hardware).

## Build-flavor coverage

The harness runs exactly one tdmd binary per invocation. Reference
coverage (D-M8-13 byte-exact) is the default. `Fp64ProductionBuild`
coverage is orthogonal (shipped at T8.10 / T8.12); `MixedFast` SNAP
coverage is owned by `tests/potentials/test_t6_differential_mixed.cpp`
(D-M8-8 NVE-drift envelope, √-scaled 1e-6 gate) and
`tests/integration/m8_smoke_t6/` (NVE conservation gate on the same
T6 fixture). M8 Case B closure (verify/benchmarks/t6_snap_scaling/REPORT.md)
documents the single-GPU MixedFast performance profile vs
KOKKOS `snap/kk`.

## When this smoke fails

- **exit 1 from step 6** — thermo divergence against 1-rank golden.
  Pattern 2 is perturbing SNAP bits. Check (in order): scheduler
  `OuterSdCoordinator::release_boundary()` still sorts peer halos by
  `(peer_subdomain_id, time_level)` before commit (R-M7-5 / D-M7-10);
  `SubdomainBoundaryDependency::emit_boundary()` still uses the
  canonical Kahan-ring reduction; GPU SNAP reduce-then-scatter order
  in `src/gpu/snap_*.cu` unchanged (gpu/SPEC §6.1 — no `atomicAdd(double)`
  on force accumulation); no stray atomic add at subdomain boundary;
  BuildFlavor is Fp64Reference (other flavors are covered by sibling
  harnesses, not this one).
- **exit 1 from step 7** — telemetry invariant broken. Most likely:
  unbalanced `ScopedSection` from a Pattern 2 path (RAII hole around
  an MPI call that threw), OR non-zero `boundary_stalls_total`
  indicating `OuterSdCoordinator` watchdog fired (D-M7-14) — inspect
  the JSONL line.
- **exit 2** — infra. Missing `mpirun`, `tdmd` built without CUDA or
  MPI, LFS assets not pulled, golden is the unpopulated placeholder
  stub (run with `TDMD_UPDATE_GOLDENS=1` once to generate), or
  submodule fixture absent (→ exit 77 instead).
- **exit 3** — wall-time blowout. 120 s budget; 2-rank 2-subdomain
  10-step Pattern 2 K=1 SNAP run on 1024 atoms should stay under
  ~30 s on commodity hardware (Fp64Reference is the slow flavor —
  MixedFast and Production are order-of-magnitude faster). A spike
  almost always means a debug build, a hang in
  `OuterSdCoordinator::wait_for_boundary()`, or a GPU kernel that
  silently fell into a D2H + CPU loop.
- **exit 77** — LAMMPS submodule not initialized. Run
  `git submodule update --init --recursive` then retry.

## Scope boundaries

- **Not covered here**: K>1 Pattern 2 (time-dimension pipelining).
  Pattern 2 at K>1 for SNAP is M9+ territory (master spec §14 M9);
  this smoke locks in only the K=1 spatial-decomposition degeneracy.
- **Not covered here**: Multi-node Pattern 2 (≥2 nodes). T8.11
  cloud-burst strong-scaling baseline (v1.5).
- **Not covered here**: NVE drift gate — owned by sibling
  `tests/integration/m8_smoke_t6/` (single-rank D-M8-8 envelope).
- **Not covered here**: SNAP / LAMMPS differential oracle — owned by
  `tests/potentials/test_t6_differential.cpp` (D-M8-7 at T8.5) and
  `tests/potentials/test_t6_differential_mixed.cpp` (D-M8-8 at T8.7).
- **Not covered here**: MixedFast SNAP performance vs KOKKOS — owned
  by `verify/benchmarks/t6_snap_scaling/REPORT.md` (T8.14 Case B
  closure artifact).
