# M2 integration smoke — Ni-Al EAM/alloy + telemetry + explain

This smoke is the acceptance gate for the M2 milestone (master spec §14):
it exercises end-to-end the user surface that M2 delivered — config ingest,
EAM/alloy force calculation, velocity-Verlet integration, thermo output,
`tdmd explain --perf`, and the new telemetry (LAMMPS breakdown + JSONL).

It is *not* a physics oracle. Numerical correctness of the EAM path is
validated by the per-atom differential T4 (`verify/benchmarks/t4_nial_alloy`)
against LAMMPS. This harness is a **pipeline** check: do all M2 components
produce consistent bytes on a canonical workload, and does a 10-step run
stay byte-identical to a committed golden.

## What the harness does

5 steps, short-circuiting on the first failure. See the shell script for
the exact commands and exit-code convention.

| # | Check                                       | Exit code on fail |
|---|---------------------------------------------|-------------------|
| 1 | `tdmd validate <config>` ⇒ zero             | 2 (infrastructure) |
| 2 | `tdmd explain --perf` ⇒ Pattern 1 + 3 + M5  | 2 (infrastructure) |
| 3 | `tdmd run --timing --telemetry-jsonl` ⇒ ok  | 2 (infrastructure) |
| 4 | Thermo log byte-matches the golden          | 1 (physics) |
| 5 | Telemetry JSONL schema + LAMMPS breakdown   | 2 (infrastructure) |

Plus a wall-time budget check that exits 3 (performance) if the whole run
exceeds `TDMD_SMOKE_BUDGET_SEC` (default 10s).

Exit codes are differentiated because they route to different humans: a
physics regression (1) needs Validation Engineer eyes; an infrastructure
failure (2) is usually a missing LFS blob or a build-env issue; a
performance regression (3) is a Perf Engineer ping.

## Why reuse T4 assets

The T4 benchmark (`verify/benchmarks/t4_nial_alloy/`) already committed:

- `setup.data` — 864-atom Ni-Al B2/L1₂ supercell (LFS-tracked, 83 KiB).
- `NiAl_Mishin_2004.eam.alloy` — Mishin *et al.* 2004 potential (raw,
  1.9 MiB, committed directly; see
  `verify/third_party/potentials/README.md` for the licensing trail).

Reusing them for the M2 smoke is a deliberate policy:

1. **The smoke has no oracle** — we only need a consistent, non-trivial
   EAM workload. T4 already picked one and it has per-atom differential
   coverage vs LAMMPS, so we know the input is physically meaningful.
2. **One set of assets, two gates.** T4 validates numerics; M2 smoke
   validates the pipeline. If the T4 potential ever moves / is
   regenerated, one coordinated update flows both ways.
3. **Cheap.** No new LFS objects, no new third-party files to audit.
4. **M5/M10 pattern.** Future milestone smokes (anchor-test M5, NPT M10)
   should follow the same convention — reuse the nearest committed
   benchmark asset rather than growing a parallel fixture tree. A new
   milestone smoke only earns its own fixture if no existing canonical
   benchmark fits.

## Regenerating the golden

The committed `nial_eam_10steps_thermo_golden.txt` is the exact thermo
stream from a clean `main`-branch build on the author's dev box. Any
intentional change to the integrator, neighbor-list rebuild logic, or
EAM path will cause step-4 to fail until the golden is refreshed.

```bash
TDMD_UPDATE_GOLDENS=1 \
  tests/integration/m2_smoke/run_nial_eam_smoke.sh \
    --tdmd build/src/cli/tdmd
```

This overwrites the golden in place. **Commit the new golden only after
a Validation Engineer has reviewed the per-step delta** — an unreviewed
regeneration would silently launder a physics regression past CI.

## Running manually

```bash
cmake --build build
tests/integration/m2_smoke/run_nial_eam_smoke.sh \
  --tdmd build/src/cli/tdmd
```

Pass `--keep-workdir` to inspect the materialised config, thermo log,
and telemetry artifacts after a run.

## Source-of-truth pointers

- Exec pack: `docs/development/m2_execution_pack.md` §T2.13.
- Milestone gate: master spec §14 M2.
- Telemetry format: `docs/specs/telemetry/SPEC.md` §4.2 (LAMMPS
  breakdown), §6 (overhead budget).
- Harness philosophy (oracle-free smokes): `docs/specs/verify/SPEC.md` §7.
- Explain CLI: `docs/specs/cli/SPEC.md`; covered by `src/cli/explain_command.cpp`.
