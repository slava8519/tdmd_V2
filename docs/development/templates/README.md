# Templates

Canonical templates used when creating new modules or updating existing module documentation. Each template encodes a structural contract — sections that must appear, in this order, in every document of that type.

## Available templates

| Template | When to use | Target location |
|---|---|---|
| [`SPEC_template.md`](SPEC_template.md) | Bootstrapping a new module's SPEC, or rewriting an existing one to match the canonical structure. | `docs/specs/<module>/SPEC.md` |
| [`TESTPLAN_template.md`](TESTPLAN_template.md) | Once a module's first test layer (unit) lands, write the TESTPLAN to map all layers explicitly. | `docs/specs/<module>/TESTPLAN.md` |
| [`MODULE_README_template.md`](MODULE_README_template.md) | Each module gets a README in `src/<module>/README.md` — short intro for newcomers; full contract stays in SPEC.md. | `src/<module>/README.md` |

## Workflow

The three documents play distinct roles and are written in this order:

1. **SPEC.md first** — the contract. Architect role drafts in a `spec-delta-<module>` branch. PR contains only Markdown (playbook §9.1). Human approval required before any code.
2. **MODULE_README.md alongside skeleton** — when the module's `src/<module>/CMakeLists.txt` lands (e.g. M0/T0.5), the README lands with it as the public-facing intro.
3. **TESTPLAN.md as tests appear** — initially a stub describing the *intent*; fleshed out as each test layer (unit → property → differential → determinism → performance → anchor) lands.

## Why templates

Without a structural template, module SPECs drift apart in shape and become harder to compare, audit, or onboard new agents to. Existing module SPECs in `docs/specs/*/SPEC.md` were written before this template existed; they will be normalized to this structure as part of regular maintenance, not as a forced migration.

## Updating templates

Changes to a template are SPEC-delta-equivalent in spirit: they shift the contract for *all future* documents of that type. Submit template edits in a focused PR, Architect-reviewed, with rationale.

## See also

- [`../claude_code_playbook.md`](../claude_code_playbook.md) — full playbook, especially §3 (task template) and §6.1 (new module session structure)
- [`../m0_execution_pack.md`](../m0_execution_pack.md) — M0 task decomposition
- [`../../specs/`](../../specs/) — existing module SPECs for reference

---

*M0/T0.2 deliverable.*
