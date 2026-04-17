# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

**Documentation-only.** No source code, no build system, no tests, no CI exist yet. The project is at "specification complete, M0 implementation not yet started." There are no build / lint / test commands to run.

The deliverables here are Markdown specs (mostly Russian, some English-mixed). Edits to this repo are spec edits — treat them with the same rigor as code: changes to interfaces or invariants require a SPEC delta, not a silent rewrite.

## What TDMD is (one paragraph)

TDMD is a planned GPU-first standalone molecular-dynamics engine whose primary parallelism principle is **time decomposition (TD)** from V.V. Andreev's 2007 dissertation: distinct spatial regions advance on **different time steps** simultaneously, exploiting locality of short-range potentials. Target niche: expensive local many-body / ML potentials (EAM, MEAM, SNAP, MLIAP, PACE) on commodity HPC / cloud clusters where inter-rank bandwidth is the bottleneck. TDMD is **not** rRESPA, **not** Parareal, **not** a LAMMPS clone — see master spec §1–§3.

## Document hierarchy (order of truth)

When two documents disagree, the higher-numbered source is wrong:

1. `TDMD_Engineering_Spec.md` — master spec v2.5, ~2966 lines, the only authoritative source.
2. `docs/specs/<module>/SPEC.md` — 13 module contracts (scheduler, zoning, perfmodel, neighbor, potentials, comm, state, integrator, runtime, io, telemetry, cli, verify).
3. Execution pack of the current milestone (will be created at M0).
4. Code (does not exist yet).

`docs/_sources/` is read-only context (Andreev dissertation, deep research report, prior spec drafts). Never edit these.

`INDEX.md` is the topical jump-table; use it to find the right section quickly instead of grepping the master spec blind.

## Required reading before touching anything

This is enforced by the playbook (`docs/development/claude_code_playbook.md` §1.1 — seven mandatory rules). Before *any* edit:

1. Find and read the relevant master-spec sections.
2. Read the module `SPEC.md` of the module being changed.
3. Read recent edits in the affected area for context.
4. Produce a **pre-implementation report** (playbook §1.3 template) covering: understanding, relevant spec sections, invariants to preserve, explicit assumptions, files to change, tests planned, risks.

**No code (or spec edit) is written until the pre-implementation report is reviewed.** This is absolute.

End every working session with a **structured report** (playbook §4): completed items, files changed, tests added, acceptance criteria met, risks, SPEC deltas proposed.

## Roles (one per session, never two simultaneously)

The playbook defines 8 canonical roles. Pick exactly one per task and stay in it; if the task crosses scopes, propose a handoff (playbook §7.1) rather than expanding scope:

- **Architect / Spec Steward** — design decisions, inter-module contracts, SPEC deltas, reconciliation.
- **Core Runtime Engineer** — `SimulationEngine`, lifecycle, config, restart/resume.
- **Scheduler / Determinism Engineer** — `TdScheduler`, `SafetyCertificate`, DAG, commit protocol, determinism.
- **Neighbor / Migration Engineer** — `CellGrid`, `NeighborList`, skin, rebuild triggers, stable reorder.
- **Physics Engineer** — Morse, EAM, MEAM, SNAP, PACE, MLIAP, integrators (NVE, NVT, NPT).
- **GPU / Performance Engineer** — CUDA kernels, streams, overlap, NVTX, perf tuning.
- **Validation / Reference Engineer** — differential vs LAMMPS, regression baselines, anchor-test.
- **Scientist UX Engineer** — CLI, YAML, preflight, explain, recipes.

Each role has a system prompt in playbook §2; use it verbatim when starting a session.

## Spec edit procedure

Interface or architectural change (playbook §9.1):

1. Architect role writes a SPEC delta in branch `spec-delta-<topic>`.
2. PR contains **only** `.md` changes — no code.
3. Human review + merge.
4. Separate PR(s) implement the new spec.

Module-internal changes that don't affect contracts: only the module SPEC needs updating.

When updating the master spec, append to **Приложение C (change log)** with explicit rationale — do not silently rewrite.

## Auto-reject patterns (playbook §5)

These get rejected without review:

- Hidden second engine (a "Fast" path that forks force calculation instead of being a policy on the shared core).
- Reference path degraded for performance — `Fp64Reference + Reference profile` is canonical oracle.
- Mutating state owned by another module (e.g. `potentials/` writing atom positions).
- Scheduler calling MPI directly (must go through `comm/`).
- Property fuzzer with < 10⁵ cases for new invariants.
- Differential threshold without SI units (`< 0.01` is meaningless; `< 1e-10 Å` is OK).
- Unstable sort in Reference profile where stable is required.
- Hot kernel without `__restrict__` on pointer params lacking explicit `NOLINT(tdmd-missing-restrict)` rationale (master spec §D.16).
- Mixed-precision switch without explicit `NumericConfig` change and CI gate revalidation.
- Hardcoded units (everything via `UnitConverter`).

## Architectural invariants to preserve in any edit

- **One core, many policies.** All optimizations are policy layers over a shared core. No architectural forks "for Fast mode."
- **Reference path sacred.** `Fp64ReferenceBuild + Reference ExecProfile` is the bitwise oracle; never weakened to ease optimizations.
- **Ownership boundaries (master spec §8.2).** State owns atoms; scheduler owns time; neighbor owns locality; comm owns transport. Cross-boundary mutation is a bug.
- **Two-phase commit** in scheduler — never single-phase.
- **Anchor-test (M5)** numerically reproduces Andreev's Al FCC 10⁶ experiment within 10%. This is foundational; its failure is project-level crisis. If anchor-test fails after a merge, stop performance work and bisect.
- **BuildFlavor × ExecProfile** is compile-time numeric semantics × runtime policy. There is no free runtime type-switching.

## Critical milestone gates

From master spec §14:

- **M4** — first working deterministic TD scheduler (single-rank, K=1, Reference).
- **M5** — multi-rank TD + anchor-test (must reproduce dissertation result).
- **M7** — two-level Pattern 2 (TD inside subdomain × SD between subdomains), production multi-node target.
- **M8** — SNAP proof-of-value: either beat LAMMPS SNAP by ≥20% on ≥8 ranks or honestly document why not.

M0–M3 are skeleton + CPU reference + zoning; M6 is GPU path. v1 total ≈ 14 months for 3–5 engineers.

## Things specific to this codebase

- Documents are predominantly **Russian**. Module SPECs and playbook mix Russian prose with English code identifiers and tables. Preserve the language of the surrounding text when editing.
- The `verify/` module (`docs/specs/verify/SPEC.md`, ~1133 lines) is the cross-module scientific validation layer — threshold registry, canonical benchmarks T0–T7, LAMMPS as git submodule, anchor-test framework. It is the single source of truth for all numerical tolerances; threshold changes require Validation Engineer + Architect review.
- Five `BuildFlavor`s exist: `Fp64ReferenceBuild`, `Fp64ProductionBuild`, `MixedFastBuild` (Philosophy B, default mixed), `MixedFastAggressiveBuild` (Philosophy A, opt-in), `Fp32ExperimentalBuild`. Adding a sixth requires the formal procedure in master spec §D.17.
- `K` (pipeline depth) is the *only* one-dimensional balance parameter in Pattern 2. Auto-K policy lives in master spec §6.5a; do not introduce alternative knobs.
- NVT/NPT in TD: only `K=1` is allowed in v1.5 (master spec §14 M9, integrator/SPEC §7.3). Per-zone thermostat (Variant B) is explicitly rejected; lazy-sync (Variant C) is M11 research.
- Open questions are tracked in master spec Приложение B.2 and per-module SPECs — do not invent answers; if blocking a task, ask.
