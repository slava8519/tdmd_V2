# M3 integration smoke — zoning plan + mid-run neighbor rebuild

This smoke is the acceptance gate for the M3 milestone (master spec §14):
it exercises end-to-end the user surface that M3 delivered — the zoning
planner's scheme selection (T3.3–T3.6), the hardened neighbor displacement
tracker (T3.8), and their integration through the CLI (`tdmd explain
--zoning`) and the run path (`tdmd run` with a thin skin that forces a
mid-run rebuild).

It is *not* a physics oracle. Numerical correctness of the EAM path is
validated by the per-atom differential T4
(`verify/benchmarks/t4_nial_alloy`) against LAMMPS. This harness is a
**pipeline** check: does the M3 scaffolding produce the expected plan and
stay byte-identical to committed goldens.

## What the harness does

5 steps, short-circuiting on the first failure. See the shell script for
the exact commands and exit-code convention.

| # | Check                                       | Exit code on fail |
|---|---------------------------------------------|-------------------|
| 1 | `tdmd validate <config>` ⇒ zero             | 2 (infrastructure) |
| 2 | `tdmd explain --zoning` matches zoning golden | 1 (plan regression) |
| 3 | `tdmd run --timing` ⇒ ok                    | 2 (infrastructure) |
| 4 | Thermo log byte-matches the thermo golden   | 1 (physics) |
| 5 | Neigh time > 5 % of total (rebuild fired)   | 1 (neighbor) |

Plus a wall-time budget check that exits 3 (performance) if the whole run
exceeds `TDMD_SMOKE_BUDGET_SEC` (default 10s).

Exit codes are differentiated because they route to different humans: a
plan regression (1) needs Neighbor/Zoning Engineer eyes; a physics
regression (1) needs Validation Engineer eyes; an infrastructure failure
(2) is usually a missing LFS blob or a build-env issue; a performance
regression (3) is a Perf Engineer ping.

## Why reuse T4 assets

The T4 benchmark (`verify/benchmarks/t4_nial_alloy/`) already committed
the 864-atom Ni-Al supercell and Mishin 2004 potential. D-M3-6 in the M3
execution pack formalised reuse: the smoke needs a non-trivial EAM
workload and T4 already picked one. Same rationale as the M2 smoke,
same assets, different gate.

## Rebuild forcing via skin

The M3 smoke uses `neighbor.skin = 0.05 Å` (vs `0.3 Å` for the M2 smoke)
to push the displacement tracker's threshold to `0.025 Å`. At 300 K on
the T4 configuration with `dt = 0.001 ps`, typical atom displacements
reach the threshold within ≈ 3–5 steps. The `Neigh` fraction of `%total`
in `tdmd run --timing` is ≈ 33 % — well above the 5 % sanity floor.
If a future change to the rebuild API silently skips the tracker,
step 5 exits 1 before CI merges.

## Regenerating the goldens

Both `thermo_golden.txt` and `zoning_rationale_golden.txt` are the exact
output from a clean `main`-branch build on the author's dev box.

```bash
TDMD_UPDATE_GOLDENS=1 \
  tests/integration/m3_smoke/run_m3_zoning_smoke.sh \
    --tdmd build/src/cli/tdmd
```

**Commit regenerated goldens only after a reviewer (Validation Engineer
for `thermo_golden.txt`; Zoning / Neighbor Engineer for
`zoning_rationale_golden.txt`) has checked the delta.** An unreviewed
regeneration would silently launder a regression past CI.

The zoning golden intentionally omits the numeric box / cutoff / skin
lines and the static caveat preamble — only the §3.4 decision output
(scheme, zone counts, N_min, n_opt, canonical order length, advisories)
is pinned. That way cosmetic changes to the explain preamble don't
break the smoke while the plan itself is still verified.

## Running manually

```bash
cmake --build build_cpu
tests/integration/m3_smoke/run_m3_zoning_smoke.sh \
  --tdmd build_cpu/src/cli/tdmd
```

Pass `--keep-workdir` to inspect the materialised config, thermo log,
and explain output after a run.

## Source-of-truth pointers

- Exec pack: `docs/development/m3_execution_pack.md` §T3.9.
- Milestone gate: master spec §14 M3.
- Zoning scheme selection: `docs/specs/zoning/SPEC.md` §3.4.
- Neighbor rebuild hygiene: `docs/specs/neighbor/SPEC.md` §5, §6.
- Harness philosophy (oracle-free smokes): `docs/specs/verify/SPEC.md` §7.
- Explain CLI: `docs/specs/cli/SPEC.md`; covered by
  `src/cli/explain_command.cpp::explain_command`.
