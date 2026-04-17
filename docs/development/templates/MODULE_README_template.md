# <Module Name>

<!--
TEMPLATE USAGE:
  1. Copy this file to src/<module>/README.md
  2. Fill all `<!-- TODO: ... -->` blocks
  3. Keep brief — this is the *intro* to the module, not the contract.
     The contract lives in docs/specs/<module>/SPEC.md.
  4. Update when public API changes (semantic, not every internal refactor).
-->

**Status:** skeleton / under development / stable / deprecated
**SPEC:** [`docs/specs/<module>/SPEC.md`](../../docs/specs/<module>/SPEC.md)
**TESTPLAN:** [`docs/specs/<module>/TESTPLAN.md`](../../docs/specs/<module>/TESTPLAN.md)
**Owner role:** <one of 8 canonical roles — see playbook §2>

## What this module does

<!-- TODO: 1-3 sentences. Plain English. No jargon a Physics PhD wouldn't immediately
recognize. -->

## Scope boundaries

<!-- TODO: 2-3 bullets explaining what this module owns vs. what neighboring modules own.
This is the README-level version of SPEC §1.3. -->

- This module **owns**: ...
- This module **does not own**: ...

## Public API at a glance

<!-- TODO: very brief overview of the main types and entry points. Don't replicate
SPEC §2 in full — just the headline. Link to full API doc. -->

```cpp
namespace tdmd {
    class <PrimaryClass>;     // <one-liner>
}
```

Note: TDMD uses a flat `namespace tdmd` (master spec convention); module identity is conveyed via include path (`tdmd/<module>/<header>.hpp`) and file layout, not namespace nesting.

Full API: [SPEC §2](../../docs/specs/<module>/SPEC.md#2-public-interface).

## Dependencies

<!-- TODO: list other tdmd modules and external libs this depends on. Include
direction (uses / used by). -->

- **Uses:** ...
- **Used by:** ...
- **External:** ...

## Build

This module is built as part of the root TDMD build:

```bash
cmake --preset default
cmake --build build --target tdmd_<module>
```

See [`docs/development/build_instructions.md`](../../docs/development/build_instructions.md) for full setup.

## Tests

```bash
ctest --test-dir build --tests-regex <module>
```

See [TESTPLAN](../../docs/specs/<module>/TESTPLAN.md) for layer breakdown.

## Examples

<!-- TODO: link to examples/<module>/ if any exist. Otherwise: "See tests/<module>/
for usage examples until proper examples land." -->

## Known limitations

<!-- TODO: things this module intentionally does not support yet, with milestone
target. Be honest. Avoid listing every TODO in the code — list things a *user* of
the module would want to know. -->

- ...

## Telemetry

<!-- TODO: 1-2 sentences pointing at SPEC §6 telemetry section. Don't enumerate. -->

See [SPEC §6](../../docs/specs/<module>/SPEC.md#6-telemetry-hooks) for the full list of metrics emitted.

## See also

<!-- TODO: cross-links to related modules, relevant master spec sections. -->

---

*Template version: 1.0 (M0/T0.2). Source: `docs/development/templates/MODULE_README_template.md`.*
