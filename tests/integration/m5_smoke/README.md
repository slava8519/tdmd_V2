# M5 integration smoke — TD scheduler + 2-rank MpiHostStaging, byte-exact to M4

<!-- markdownlint-disable MD013 -->

This smoke is the acceptance gate for the M5 milestone (master spec §14
M5). It exercises end-to-end the M5 user surface — the opt-in
`scheduler.td_mode: true` flag from M4 combined with M5's concrete
`comm.backend: mpi_host_staging` transport — in a 2-rank mesh-topology
configuration.

It is **not** a physics oracle. Numerical correctness of the EAM path
was validated at M2 by the T4 differential against LAMMPS. This
harness is a **determinism** check: in K=1 P=2, does the TD-mode run
through MpiHostStaging produce thermo byte-identical to the M4 golden
(= M3 golden, via D-M4-9)?

## The D-M5-12 contract

Execution pack §T5.12 and scheduler/SPEC §12.3 define the byte-exact
acceptance gate:

> With `scheduler.td_mode: true`, `pipeline_depth_cap: 1`,
> `comm.backend: mpi_host_staging`, K=1 on 2 ranks, the 10-step thermo
> stream MUST equal the same config's K=1 P=1 output bit-for-bit
> (which in turn equals the M4 golden, which equals the M3 golden).

The reduction-order invariant (D-M5-9) guarantees this: the thermo
tallies go through `deterministic_sum_double` (Kahan-summed ring-sum)
instead of raw `MPI_Allreduce`, so rank count does not perturb bits.
Any drift here is a bug in one of:

1. The scheduler touched atom state it shouldn't have (cross-rank
   migration ordering).
2. Reduction-order determinism slipped (e.g. a stray `std::accumulate`
   on a non-canonical rank-list ordering).
3. Neighbor-list reorder stability broke across ranks.

**The M5 `thermo_golden.txt` is the M4 `thermo_golden.txt`**, copied
verbatim. The smoke checks that the two golden files match bit-for-bit
before running the binary (just like M4 does against M3) — this
defends against someone editing one without syncing the other.

## What the harness does

5 steps, short-circuiting on the first failure.

| # | Check                                                         | Exit |
|---|---------------------------------------------------------------|------|
| 1 | Pre-flight: M5 golden == M4 golden (D-M5-12 chain)            | 2    |
| 2 | `mpirun --np 2 tdmd validate <config>` — accepts 2-rank comm  | 2    |
| 3 | `mpirun --np 2 tdmd run --telemetry-jsonl` exits 0 < 30 s     | 2/3  |
| 4 | Thermo byte-matches M5 golden (= M4 golden = M3 golden)       | 1    |
| 5 | Telemetry invariants (run_end event, wall-time budget, no RAII leaks) | 1 |

## Flags / env

- `--tdmd <path>` — path to the tdmd binary (required; must be built
  with `-DTDMD_ENABLE_MPI=ON`).
- `--mpirun <path>` — alternate MPI launcher; defaults to `mpirun`
  from `$PATH`. `mpiexec` works too on MPICH systems.
- `--keep-workdir` — preserve the tmp workdir on success.
- `TDMD_UPDATE_GOLDENS=1` — overwrite goldens instead of comparing.
  Commit only after Validation Engineer review. NOTE: updating the M5
  golden without also updating M4's breaks D-M5-12.
- `TDMD_SMOKE_BUDGET_SEC=N` — override the default 30 s wall-time
  budget.

## Why MPI is conditional in CI

Public `ubuntu-latest` runners don't ship OpenMPI, so the `build-cpu`
job installs `openmpi-bin libopenmpi-dev` alongside the compiler apt
line. The M5 smoke step is gated on `mpirun` being on `$PATH`; if the
install step is skipped (matrix-disabled variant, third-party fork),
the step safely exits 0 with an explicit "M5 smoke SKIPPED — no MPI"
line rather than failing the job.

## When this smoke fails

- **exit 1 from step 4** — thermo divergence. K=1 P=2 is perturbing
  bits. Check (in order): `deterministic_sum_double` still called for
  thermo reductions; `CommBackend::reduce_doubles` path canonical;
  rank-list ordering in the allreduce shim matches P=1 single-tree.
- **exit 1 from step 5** — telemetry invariant broken. Usually an
  unbalanced `ScopedSection` (RAII hole) — inspect `ignored_end_calls`.
- **exit 2** — infra. Missing `mpirun`, `tdmd` built without MPI
  (check `cmake --log-level=VERBOSE` output for `TDMD_ENABLE_MPI:
  enabled`), or LFS assets not pulled.
- **exit 3** — wall-time blowout. 2-rank 10-step should stay under
  10 s on commodity hardware; any spike above 30 s means either a
  debug build or a comm backend deadlock.

## Scope boundaries

- **Not covered here**: anchor-test (T3, full dissertation reproduction).
  That is a slow-tier local gate — see
  `tests/integration/m5_anchor_test/` and its `README.md`.
- **Not covered here**: K > 1 pipelining. K=1 is the only scheduler
  depth this smoke exercises. K=4 is validated by
  `tests/scheduler/test_k_batching.cpp` (unit level).
- **Not covered here**: RingBackend. The MpiHostStaging backend is
  the M5 smoke target; RingBackend gets its own unit test
  (`tests/comm/test_ring_backend.cpp`) and participates in the
  anchor-test run (T3 config uses `backend: ring`).
