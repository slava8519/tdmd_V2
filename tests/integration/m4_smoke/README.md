# M4 integration smoke — TD scheduler, K=1, byte-exact to M3

This smoke is the acceptance gate for the M4 milestone (master spec §14):
it exercises end-to-end the user surface that M4 delivered — the opt-in
`scheduler.td_mode: true` YAML flag (D-M4-11) that wires the
`CausalWavefrontScheduler` (T4.5–T4.8) into the run loop (T4.9) behind
an unchanged legacy sequential physics path.

It is **not** a physics oracle. Numerical correctness of the EAM path is
validated by the per-atom differential T4
(`verify/benchmarks/t4_nial_alloy`) against LAMMPS. This harness is a
**determinism** check: in K=1 single-rank, does the TD-mode run produce
thermo byte-identical to the M3 smoke's legacy output?

## The D-M4-9 contract

Master spec §14 (M4) and execution pack §T4.9 define the byte-exact
acceptance gate:

> With `scheduler.td_mode: true` and K=1 on a single rank, the 10-step
> thermo stream MUST equal the same config's legacy-mode output bit-for-bit.

In K=1 the scheduler wraps the canonical per-step in its zone lifecycle
(refresh → select → mark\_computing → mark\_completed → commit → release →
on\_zone\_data\_arrived) but does not dispatch physics — forces and the
integrator are invoked identically to the legacy path. Neighbor-list
reduction order is preserved, so no floating-point drift can sneak in.

**The M4 `thermo_golden.txt` is the M3 `thermo_golden.txt`**, copied
verbatim. The smoke checks both: (a) the two golden files match bit-for-bit
before running the binary, and (b) the live 10-step output matches them.
If (a) fails, someone broke the D-M4-9 contract at commit time; if (b)
fails, the scheduler is touching state it shouldn't.

## What the harness does

4 steps, short-circuiting on the first failure.

| # | Check                                            | Exit code on fail |
|---|--------------------------------------------------|-------------------|
| 1 | `tdmd validate <config>` with `td_mode: true`    | 2 (infrastructure) |
| 2 | `tdmd run --timing` under td\_mode exits zero    | 2 (infrastructure) |
| 3 | Thermo byte-matches the M3 golden (D-M4-9)       | 1 (determinism)    |
| 4 | Neigh time > 5 % of total — rebuild fired        | 1 (neighbor)       |

Plus a wall-time budget check that exits 3 (performance) if the whole
run exceeds `TDMD_SMOKE_BUDGET_SEC` (default 10s). The harness also
pre-checks (before invoking the binary) that the M4 golden is
byte-identical to the M3 golden — a defensive guard against someone
editing one without syncing the other.

## Why reuse the M3 config and M3 golden

Master spec §14 M4 defines the gate as "TD scheduler produces identical
output to the legacy path." The cleanest formulation is: **use the M3
config, flip `td_mode: true`, compare to the M3 golden**. Anything else
would add variables beyond the one the gate is about.

Concretely: this smoke's `smoke_config.yaml.template` is the M3 template
byte-for-byte except for the added `scheduler.td_mode: true` block.
D-M4-9 collapses to `diff m3/thermo m4/thermo == 0` after a clean run.

## Why not a cert-failure / deadlock check

T4.11's exec-pack spec suggests probing `certificate_failures_total == 0`
and `deadlock_warnings_total == 0`. In M4 K=1 single-rank these are
structurally zero: certificates are stubbed always-safe (no live-state
adapter yet — M7+), and the scheduler never stalls because every zone
transitions Ready → Committed in one step. Cross-run introspection of
these counters lives in the C++ tests (T4.8 `test_deadlock_watchdog`,
T4.10 `test_determinism_same_seed` on `event_log()`) where assertion
depth matches what can actually vary in K=1.

## Regenerating the goldens

`thermo_golden.txt` is the M3 golden. If you need to regenerate it:

```
# regenerate BOTH or you break D-M4-9
TDMD_UPDATE_GOLDENS=1 tests/integration/m3_smoke/run_m3_zoning_smoke.sh --tdmd <path>
TDMD_UPDATE_GOLDENS=1 tests/integration/m4_smoke/run_m4_td_smoke.sh   --tdmd <path>
# verify
diff tests/integration/m3_smoke/thermo_golden.txt \
     tests/integration/m4_smoke/thermo_golden.txt
```

If the two files differ after regeneration, stop — that's exactly the
bug the D-M4-9 gate exists to catch.

## When this smoke fails

- **exit 1 from step 3** — `td_mode: true` changed the thermo. The K=1
  scheduler is mutating state it shouldn't, or reduction order slipped.
  Start by bisecting scheduler changes since the last green run; run
  `tests/scheduler/test_determinism_same_seed.cpp` locally for tighter
  signal (compares the full atom SoA + event log, not just thermo).
- **exit 1 from step 4** — neighbor rebuild didn't fire. Under TD mode
  the `recompute_forces()` path goes through the same `rebuild_neighbors()`
  guard as legacy. If this fails only under TD, the scheduler is
  short-circuiting the rebuild trigger — probably in `td_step()`.
- **exit 2** — infra. Usually missing LFS assets (`git lfs pull`) or
  a stale `tdmd` binary build.
- **exit 3** — wall-time budget. Try `TDMD_SMOKE_BUDGET_SEC=30` locally
  to separate "genuinely slow" from "CI runner variance."
