# Contributing to TDMD

TDMD is developed primarily through AI-assisted engineering following a strict spec-driven workflow. Before making any change — spec edit, code, tests, or configuration — every contributor (human or AI agent) must read:

1. **[`TDMD_Engineering_Spec.md`](./TDMD_Engineering_Spec.md)** — master engineering spec (v2.5). The single source of truth.
2. **[`docs/development/claude_code_playbook.md`](./docs/development/claude_code_playbook.md)** — procedural playbook for AI agents. Defines 8 canonical roles, task template, auto-reject conditions, merge gates.
3. **[`CLAUDE.md`](./CLAUDE.md)** — condensed rules tailored for Claude Code sessions in this repository.

## Workflow (playbook §1)

Every task follows the same sequence:

1. **Spec first.** Read relevant master-spec sections and the module `SPEC.md` of any module being touched.
2. **Pre-implementation report** (playbook §1.3 template) — understanding, invariants, assumptions, files to change, tests planned.
3. **Human review** of the plan before any code or spec edit.
4. **Implementation** strictly within approved scope (no scope creep per playbook §1.1.7).
5. **Structured session report** (playbook §4) on completion — files changed, tests added, acceptance criteria met, risks, SPEC deltas proposed.

## Spec changes

Changes to interfaces or architectural contracts require a **SPEC delta** (playbook §9.1):

1. Architect role drafts the delta in a branch named `spec-delta-<topic>`.
2. The PR contains **only** Markdown changes — no code.
3. Human approval and merge.
4. Separate PR(s) implement the new spec.

Module-internal changes that do not affect public contracts: only the module `SPEC.md` needs updating (still via a focused PR).

When updating the master spec, append to **Приложение C (change log)** with explicit rationale — never silently rewrite.

## Pull requests

Every PR must:

- Attach a pre-implementation report and a session report (see `.github/PULL_REQUEST_TEMPLATE.md` once T0.6 lands).
- Pass all applicable CI pipelines (A–F per master spec §11).
- Receive at least one human review approval.
- Declare SPEC deltas (or explicitly "none").

## Auto-reject patterns (playbook §5)

These get rejected without review — do not submit PRs that contain:

- A hidden second engine (any "Fast" path forking force calculation).
- A degraded Reference path for performance.
- A module mutating state it does not own.
- Scheduler code calling MPI directly (must route through `comm/`).
- Property fuzzer with fewer than 10⁵ cases for new invariants.
- Differential threshold without SI units.
- Unstable sort in the Reference profile where stable is required.
- Hot kernels without `__restrict__` on pointer parameters lacking an explicit `NOLINT(tdmd-missing-restrict)` rationale.
- Mixed-precision switch without an explicit `NumericConfig` change and CI gate revalidation.
- Hardcoded units bypassing `UnitConverter`.

## Roles

Work is assigned to exactly one of eight canonical roles per session (playbook §2). A role is a hat, not an identity — you wear one at a time. If a task crosses scopes, propose a handoff (playbook §7.1) instead of expanding scope.

## Code of conduct

Be constructive, precise, and respectful. Disagreements resolve through SPEC deltas or Architect arbitration (playbook §7.2). If Architect cannot resolve — escalate to the human maintainer.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/` (added in task T0.6). Until CI lands, report issues as GitHub issues with a clear reproduction and a reference to the affected spec section.
