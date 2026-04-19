# M6 integration smoke — GPU force path + 2-rank MpiHostStaging, byte-exact to M5

<!-- markdownlint-disable MD013 -->

This smoke is the acceptance gate for the M6 milestone (master spec §14
M6). It exercises the M6 user surface end-to-end — `runtime.backend: gpu`
on top of the M4/M5 deterministic TD scheduler + MpiHostStaging transport
— in a 2-rank host-staged configuration.

It is **not** a physics oracle. Numerical correctness of the EAM path was
validated at M2 by the T4 differential against LAMMPS. This harness is a
**determinism** check: in K=1 P=2 with `runtime.backend: gpu`, does the
Reference-flavor GPU run produce thermo byte-identical to the M5 (= M4 =
M3) CPU golden?

## The D-M6-7 contract

Execution pack §T6.13 and runtime/SPEC §2.3 define the byte-exact
acceptance gate:

> With `runtime.backend: gpu`, `scheduler.td_mode: true`,
> `pipeline_depth_cap: 1`, `comm.backend: mpi_host_staging`, K=1 on
> 2 ranks, a `Fp64ReferenceBuild` binary MUST produce thermo
> byte-identical to the same config run on CPU (D-M5-12 chain), which
> in turn equals the M5 golden = M4 golden = M3 golden.

The invariant chain:

- D-M5-9 (Kahan-ring reduction-order determinism for thermo tallies).
- D-M6-7 (GPU Reference force reduction canonicalised to match CPU
  summation order — gather-to-single-block + sorted Kahan-add).
- Everything downstream of the `PotentialModel::compute()` call is
  reused from the CPU path: thermo tallies, neighbor-list reorder
  stability, scheduler commit protocol.

Drift here is a bug in exactly one of those layers.

**The M6 `thermo_golden.txt` is the M5 `thermo_golden.txt`**, copied
verbatim. The harness checks that the two golden files match bit-for-bit
before running the binary — this defends against someone editing one
without syncing the other.

## What the harness does

6 steps, short-circuiting on the first failure.

| # | Check                                                        | Exit |
|---|--------------------------------------------------------------|------|
| 1 | Pre-flight: M6 golden == M5 golden (D-M6-7 chain)            | 2    |
| 2 | Local-only gate: `nvidia-smi` reports ≥1 GPU, else SKIP      | 0    |
| 3 | `mpirun --np 2 tdmd validate <config>` — 2-rank GPU config OK | 2    |
| 4 | `mpirun --np 2 tdmd run --telemetry-jsonl` exits 0 < 60 s    | 2/3  |
| 5 | Thermo byte-matches M6 golden (= M5 golden = … = M3 golden)  | 1    |
| 6 | Telemetry invariants (run_end event, wall-time budget)       | 1    |

Step 2 is the **Option A gate**: the smoke SKIPs on any host that does
not expose a CUDA device. This is the explicit escape hatch for CI
(D-M6-6: no self-hosted GPU runner) — the harness still runs end-to-end
so infrastructure rot (template typo, missing LFS asset, broken
substitution) fails loudly even when the GPU path cannot execute.

## Why this is local-only

D-M6-6 explicitly forbids a public-repo self-hosted GPU runner. The
`build-gpu` CI matrix (T6.12) covers compile + link only. The smoke
lives in-tree so developers can invoke it locally before pushing:

```bash
tests/integration/m6_smoke/run_m6_smoke.sh \
  --tdmd build/src/cli/tdmd
```

and it is also the **mandatory pre-merge local gate** for any M6+
change touching `src/gpu/`, `src/potentials/eam_alloy*`,
`src/integrator/*_gpu*`, `src/comm/mpi_host_staging*`, or
`src/runtime/gpu_context*`.

Note: the companion **T3-gpu anchor test** (see
`verify/benchmarks/t3_al_fcc_large_anchor_gpu/` and
`tests/integration/m5_anchor_test/`) is a separate, longer
(~2-5 min) local gate that checks the dissertation 10% efficiency
window on the GPU path. Run it before any merge that touches GPU
kernel math.

## Flags / env

- `--tdmd <path>` — path to the `tdmd` binary (required; must be built
  with `-DTDMD_ENABLE_MPI=ON -DTDMD_BUILD_CUDA=ON`).
- `--mpirun <path>` — alternate MPI launcher; defaults to `mpirun`.
- `--keep-workdir` — preserve the tmp workdir on success.
- `TDMD_UPDATE_GOLDENS=1` — overwrite the local golden instead of
  comparing. Commit only after Validation Engineer review. **Updating
  the M6 golden without also updating M5's breaks D-M6-7.**
- `TDMD_SMOKE_BUDGET_SEC=N` — override the default 60 s wall-time
  budget (M6 is slower than M5 because of H2D/D2H + kernel launches
  on a tiny 864-atom fixture).

## Build-flavor coverage

The harness runs exactly one tdmd binary per invocation. Reference
coverage (D-M6-7 byte-exact) is the default. For `MixedFastBuild`
coverage (D-M6-8 thresholds, not byte-exact) invoke twice with a
second binary from a MixedFast build tree:

```bash
# Reference coverage — byte-exact gate
tests/integration/m6_smoke/run_m6_smoke.sh --tdmd build/src/cli/tdmd

# MixedFast coverage — invoked with the thresholded differential, not
# the byte-exact gate (covered by T6.8a test_t4_differential_mixed).
```

The MixedFast variant is deliberately NOT layered into this smoke:
D-M6-7 and D-M6-8 are different acceptance modes, and bundling them
would dilute the byte-exact failure signal. T6.8a's threshold diff
already owns the MixedFast gate.

## When this smoke fails

- **exit 1 from step 5** — thermo divergence against M6 golden. The GPU
  Reference path is perturbing bits. Check (in order): reduction order
  in `src/gpu/eam_alloy_gpu.cu` still matches the CPU gather-Kahan
  sequence (D-M6-7); `comm::deterministic_sum_double` still wraps the
  thermo reductions on the host side (D-M5-9); no stray atomic add in
  force accumulation; BuildFlavor is Fp64Reference (the harness
  auto-detects MixedFast via telemetry and exits 2 with a clear
  diagnostic).
- **exit 1 from step 6** — telemetry invariant broken. Usually an
  unbalanced `ScopedSection` from a GPU path (RAII hole around a CUDA
  call that threw). Inspect `ignored_end_calls` and the JSONL line.
- **exit 2** — infra. Missing `mpirun`, `tdmd` built without CUDA or MPI,
  or LFS assets not pulled.
- **exit 3** — wall-time blowout. 60 s budget; 2-rank GPU 10-step on
  864 atoms should stay under ~5 s on commodity hardware including
  H2D/D2H. A spike almost always means a debug build or a kernel that
  silently fell into a D2H + CPU loop.

## Scope boundaries

- **Not covered here**: MixedFast byte-exact (impossible by design).
  D-M6-8 thresholds for MixedFast are owned by
  `tests/potentials/test_t4_differential_mixed.cpp`.
- **Not covered here**: T3-gpu anchor (dissertation efficiency curve).
  Separate long-running local gate.
- **Not covered here**: 2-stream overlap ≥ 30% (T6.9). That's a
  performance gate measured by NVTX-bracketed wall-time, owned by
  `tests/gpu/test_overlap_budget.cpp`.
- **Not covered here**: Multi-GPU per rank (D-M6-3 explicitly punts to
  M7+).
