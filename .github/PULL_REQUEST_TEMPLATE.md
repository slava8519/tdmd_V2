# Pull request

<!--
Per playbook §1.3 (pre-implementation report) and §4 (session report).
Delete sections that genuinely do not apply; do not leave them blank.
-->

## Summary

<!-- 1-3 sentences describing what changed and why. -->

## Role (session)

<!-- Pick exactly one from playbook §2. If the work spans roles, split the PR. -->

- [ ] Architect / Spec Steward
- [ ] Core Runtime Engineer
- [ ] Scheduler / Determinism Engineer
- [ ] Neighbor / Migration Engineer
- [ ] Physics Engineer
- [ ] GPU / Performance Engineer
- [ ] Validation / Reference Engineer
- [ ] Scientist UX Engineer

## Pre-implementation report

<!-- Paste the report (playbook §1.3) or link to the commit / issue where it lives. -->

## Session report

<!-- Paste the session report (playbook §4) or link. Include: completed items,
     files changed, tests added, acceptance criteria met, risks, SPEC deltas. -->

## SPEC deltas

<!-- List every spec change this PR makes. "None" if the PR only touches code. -->

- [ ] None
- [ ] Master spec `TDMD_Engineering_Spec.md`
- [ ] Module SPEC: <module> (link)
- [ ] Template: SPEC / TESTPLAN / MODULE_README

Rationale:

## Tests added

| Layer | Count | Location |
|---|---|---|
| Unit | | |
| Property (≥10⁵ cases) | | |
| Differential (vs LAMMPS) | | |
| Determinism | | |
| Performance baseline | | |

## Pipelines impacted

- [ ] A — Lint + build
- [ ] B — Unit
- [ ] C — Property fuzz
- [ ] D — Differential vs LAMMPS
- [ ] E — Performance (regression gate)
- [ ] F — Reproducibility / determinism

## Acceptance criteria met

<!-- Reference the task prompt or execution pack entry. Tick each criterion. -->

- [ ] ...

## Auto-reject self-check (playbook §5.1)

- [ ] No hidden "second engine" or architectural fork for Fast mode.
- [ ] `Fp64ReferenceBuild + Reference ExecProfile` path not weakened.
- [ ] No cross-module state mutation (ownership boundaries intact).
- [ ] Scheduler does not call MPI directly (routes through `comm/`).
- [ ] Property tests ≥ 10⁵ cases per invariant.
- [ ] Differential thresholds have SI units.
- [ ] Stable sort used where Reference profile requires it.
- [ ] Hot kernels qualify pointer params with `__restrict__` (or NOLINT rationale).
- [ ] Mixed-precision changes go through `NumericConfig` with CI revalidation.
- [ ] No hardcoded units (everything via `UnitConverter`).

## Risks / follow-ups

<!-- Anything reviewer should pay extra attention to; what's deferred to next PR. -->

## Checklist

- [ ] CI green (all jobs)
- [ ] `pre-commit run --all-files` clean
- [ ] Documentation updated where affected
- [ ] Cross-linked from master spec §14 milestone list (if this closes a gate)
