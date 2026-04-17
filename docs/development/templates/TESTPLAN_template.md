# <Module Name> — TESTPLAN

<!--
TEMPLATE USAGE:
  1. Copy this file to docs/specs/<module>/TESTPLAN.md
  2. Fill each `<!-- TODO: ... -->` based on module SPEC.md §5
  3. Add to module CMakeLists.txt as tests source dir
  4. Update CI matrix in .github/workflows/ci.yml if module needs new pipeline

CRITICAL: every test claim here MUST be backed by an actual test file. Aspirational
tests are technical debt, not documentation.
-->

**Module:** `<module>`
**Status:** draft / active
**Parent SPEC:** [`docs/specs/<module>/SPEC.md`](./SPEC.md)
**Master spec test pyramid:** §13
**VerifyLab integration:** [`docs/specs/verify/SPEC.md`](../verify/SPEC.md)
**Last updated:** YYYY-MM-DD

---

## 1. Test pyramid layers (master spec §13)

| Layer | Applies to this module | CI pipeline | Where |
|---|---|---|---|
| L1 — Unit | yes / no | B (Unit) | `tests/<module>/unit/` |
| L2 — Property fuzz (≥10⁵ cases) | yes / no | C (Property) | `tests/<module>/property/` |
| L3 — Differential vs LAMMPS | yes / no | D (Differential) | `tests/<module>/differential/` |
| L4 — Determinism (bitwise) | yes / no | F (Reproducibility) | `tests/<module>/determinism/` |
| L5 — Performance baseline | yes / no | E (Performance) | `benchmarks/<module>/` |
| L6 — Anchor / Tier-3 (slow) | yes / no | nightly / release | `verify/canonical/` |

---

## 2. Unit tests (L1)

**Framework:** Catch2 v3.

<!-- TODO: list each unit test with one-line purpose. Group by file. Example:

`test_atom_soa.cpp`:
  - "AtomSoA default construction" — verify zero-sized container
  - "AtomSoA push_back preserves identity" — id field stable across resize
  - "AtomSoA memory layout is SoA, not AoS" — pointer arithmetic check
-->

---

## 3. Property tests (L2)

**Framework:** Catch2 generators + custom shrinker; minimum **10⁵ cases per invariant** (playbook §5.2).

<!-- TODO: list each property and the invariant it asserts. Example:

`property_certificate.cpp`:
  - Property "monotone safe time": for any zone state, safe_until time
    is non-decreasing as displacement grows.
    Cases: 10⁶. Generator: random ZoneState fixtures.
    Shrinker: minimize zone size on failure.
-->

---

## 4. Differential tests (L3, vs LAMMPS oracle)

**LAMMPS:** `verify/third_party/lammps/install_tdmd/bin/lmp` (per T0.7).
**Threshold registry:** [`docs/specs/verify/thresholds.yaml`](../verify/thresholds.yaml) (when created in M1+).

<!-- TODO: list each differential test with:
  - LAMMPS input script path (verify/canonical/<test>/in.lammps)
  - TDMD config path (verify/canonical/<test>/tdmd.yaml)
  - Compared quantities (forces, energy, virial) with explicit SI thresholds
  - Expected wall-clock budget

Example:
  T1-morse-fcc:
    LAMMPS: verify/canonical/T1-morse-fcc/in.lammps
    TDMD:   verify/canonical/T1-morse-fcc/tdmd.yaml
    Compare:
      max |F_tdmd - F_lammps| / |F_lammps|  <  1e-10  (Reference profile)
      max |F_tdmd - F_lammps| / |F_lammps|  <  1e-5   (MixedFast profile)
      |E_total_tdmd - E_total_lammps|       <  1e-12 eV/atom
-->

---

## 5. Determinism tests (L4)

**Required only in `Fp64ReferenceBuild + Reference ExecProfile`** (master spec §7).

<!-- TODO: list determinism guarantees this module must satisfy. Examples:
  - Same seed → bitwise identical output across 10 repeated runs
  - Same input + different rank count → bitwise identical (layout-invariant)
  - Restart equivalence: continue from checkpoint at step N → identical to fresh run
-->

---

## 6. Performance baselines (L5)

**Baseline storage:** `benchmarks/<module>/baselines/` (versioned in git, JSON format).
**Regression gate:** wall-clock degradation > 5% blocks merge (playbook §8.1 Pipeline E).

<!-- TODO: list each baseline benchmark with:
  - Workload (atom count, potential, integrator, steps)
  - Baseline hardware (e.g. RTX 5080, Intel Xeon 8358)
  - Stored metric (median wall-clock per step, p95 per step)
  - Tolerance for regression
-->

---

## 7. Anchor / canonical tests (L6, slow tier)

<!-- TODO: only fill if this module participates in T3 anchor-test (master spec §13.3).
Most modules do NOT — only scheduler, integrator, and potentials participate.

If applicable, reference verify/SPEC §4.4 (T3 benchmark) and document this module's
role in the anchor-test contract. -->

---

## 8. Threshold registry entries

<!-- TODO: list every numerical tolerance this module's tests rely on. Each MUST:
  - Have SI units explicit (no bare numbers)
  - Have a rationale (why this value)
  - Be referenced in the verify/thresholds.yaml registry once that file exists

Format:
  - `<module>.<threshold_name>`: <value> [unit]
    Rationale: <why>
    Used in: <which test files>
-->

---

## 9. Fixtures

<!-- TODO: list test data files this module depends on. Use Git LFS for files >500KB
(.gitattributes already handles common patterns).

Format:
  - `tests/<module>/fixtures/<file>`: <purpose, generator, regen procedure>
-->

---

## 10. CI pipeline mapping

| Test layer | Pipeline | Trigger |
|---|---|---|
| Unit | B | every PR |
| Property | C | every PR |
| Differential | D | PR touching this module |
| Determinism | F | PR touching this module |
| Performance | E | PR touching this module + nightly |
| Anchor | slow tier | nightly + release |

<!-- TODO: confirm which pipelines apply, override if module needs special treatment. -->

---

## 11. Known gaps and TODOs

<!-- TODO: list test layers NOT YET implemented but planned. Reference milestone. -->

---

*Template version: 1.0 (M0/T0.2). Source: `docs/development/templates/TESTPLAN_template.md`.*
