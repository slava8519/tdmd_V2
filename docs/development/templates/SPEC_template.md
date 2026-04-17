# <Module Name> — SPEC

<!--
TEMPLATE USAGE:
  1. Copy this file to docs/specs/<module>/SPEC.md
  2. Replace all <PLACEHOLDER> markers
  3. Remove or fill all `<!-- TODO: ... -->` blocks
  4. Cross-link from master spec §8.1 (module list)
  5. Submit as SPEC PR (no code) per playbook §9.1

TONE: this document is the *contract* of the module. Optimistic prose belongs in
README.md; hard guarantees belong here. Future agents read SPEC.md as binding.
-->

**Module:** `<module>`
**Status:** draft / accepted / superseded
**Parent:** `TDMD_Engineering_Spec.md` v2.5 §<X.Y>
**Owners (role):** <Architect / Core Runtime / Scheduler / ... — see playbook §2>
**Last updated:** YYYY-MM-DD

---

## 1. Purpose, scope, boundaries

### 1.1. Purpose

<!-- TODO: 1-2 sentences. What does this module exist for? What problem does it solve
that no other module owns? -->

### 1.2. Scope (in)

<!-- TODO: bullet list of responsibilities this module owns. -->

### 1.3. Out of scope

<!-- TODO: bullet list of things this module DOES NOT own. Reference the module
that does own them. Critical for ownership boundaries (master spec §8.2). -->

### 1.4. Glossary (module-specific)

<!-- TODO: terms used in this SPEC that are not defined in master spec §1-§4. -->

---

## 2. Public interface

### 2.1. C++ API

```cpp
// TODO: header-level declarations exposed to other modules.
// One class / function per subsection. Include doc comments with units (eV, Å, ps).

namespace tdmd {

class <PrimaryClass> {
public:
    // TODO
};

}  // namespace tdmd
```

Note: TDMD uses a flat `namespace tdmd` (master spec convention). Module identity lives in the include path (`tdmd/<module>/<header>.hpp`), not in a nested namespace.

### 2.2. CUDA API (if applicable)

<!-- TODO: kernel signatures, launch policies, restrict qualifiers per §D.16.
Mark hot kernels with [[tdmd::hot_kernel]]. -->

### 2.3. Versioning rules

<!-- TODO: how does this module's state version (per state/SPEC) interact with
its public API? What invalidates downstream consumers? -->

---

## 3. Algorithms and formulas

<!-- TODO: mathematical core of the module. Cite master spec sections and dissertation
appendix A entries where applicable. Document all units (eV, Å, ps).

For numerical code, specify precision (Fp64 / Fp32 / mixed) per §7 and Приложение D. -->

### 3.1. <Algorithm name>

<!-- TODO: pseudocode + formula + complexity. -->

---

## 4. Policy definitions

<!-- TODO: enumerate all policies this module exposes via PolicyBundle.
Each policy: name, type, default value, valid range, when to use.

Examples from existing modules:
  - scheduler: PriorityPolicy, RetryPolicy
  - neighbor: SkinPolicy, RebuildTriggerPolicy
  - integrator: TimestepPolicy, ThermostatCouplingPolicy
-->

---

## 5. Tests

### 5.1. Test layers required

| Layer | Required for this module | Notes |
|---|---|---|
| Unit | yes / no | <!-- TODO --> |
| Property fuzz (≥10⁵ cases) | yes / no | <!-- TODO if yes: which invariants --> |
| Differential vs LAMMPS | yes / no | <!-- TODO if yes: which T-benchmarks --> |
| Determinism (bitwise) | yes / no | <!-- TODO if yes: in which BuildFlavor --> |
| Performance (baseline) | yes / no | <!-- TODO if yes: which benchmarks --> |

### 5.2. Mandatory invariants (verifiable)

<!-- TODO: list invariants that MUST hold. Each invariant is testable. Reference
master spec §13.4 invariants I1-I7 if applicable to scheduler-related modules. -->

- **I-mod-1:** ...

### 5.3. Threshold registry entries

<!-- TODO: numerical tolerances that go into verify/thresholds.yaml. Each must
have SI units (per playbook §5.2). Format:
  threshold_name: value [unit]   # rationale
-->

---

## 6. Telemetry hooks

<!-- TODO: metrics this module emits. Per master spec §12 and telemetry/SPEC.

Each hook: name, type (counter/gauge/histogram), unit, NVTX range name (if applicable),
collection cost. -->

---

## 7. Roadmap alignment

<!-- TODO: which milestones this module participates in. Reference master spec §14. -->

| Milestone | This module's deliverable |
|---|---|
| M<N> | <!-- TODO --> |

---

## 8. Open questions

<!-- TODO: known unknowns. Each question should reference a section that documents
the alternative being considered. Track in master spec Приложение B.2 if cross-cutting. -->

- **OQ-mod-1:** ...

---

## 9. Change log (this module)

### v0.1 (YYYY-MM-DD, draft)

- Initial draft.

---

*Template version: 1.0 (M0/T0.2). Source: `docs/development/templates/SPEC_template.md`.*
