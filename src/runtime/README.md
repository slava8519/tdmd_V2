# runtime

**Status:** under development (M1: `UnitConverter` only)
**SPEC:** [`docs/specs/runtime/SPEC.md`](../../docs/specs/runtime/SPEC.md)
**TESTPLAN:** not yet written
**Owner role:** Core Runtime Engineer / Physics Engineer (for UnitConverter)

## What this module does

Houses lifecycle orchestration (`SimulationEngine`, arriving in T1.9) and the
single unit conversion point of TDMD (`UnitConverter`). Per master spec §5.3
the internal representation is always `metal`; this module is the only place
where unit-system arithmetic lives.

## Scope boundaries

- This module **owns**: `UnitConverter`, `SimulationEngine` (post-T1.9),
  lifecycle state machine, runtime policy bundle.
- This module **does not own**: atoms (see `state/`), neighbor lists (see
  `neighbor/`), force calculations (see `potentials/`), time scheduling
  (see `scheduler/`).

## Public API at a glance

```cpp
namespace tdmd {
    enum class UnitSystem;
    struct LengthQ, EnergyQ, /* ... 8 strong typedefs */;
    class UnitConverter;   // metal native in M1, lj stub (throws) until M2
}
```

Full API: [SPEC](../../docs/specs/runtime/SPEC.md) +
[`unit_converter.hpp`](include/tdmd/runtime/unit_converter.hpp).

## Dependencies

- **Uses:** standard library only in M1 (`<stdexcept>`, `<string>`).
- **Used by:** `io/` (T1.3), `cli/` preflight (T1.4+), `state/` indirectly
  (via caller conversions at import time).
- **External:** none.

## Build

```bash
cmake --preset cpu-only
cmake --build build_cpu --target tdmd_runtime
```

## Tests

```bash
ctest --test-dir build_cpu --tests-regex runtime
```

## Known limitations

- M1: `UnitConverter::*_from_lj` / `*_to_lj` throw `NotImplementedInM1Error`.
  API signatures are stable and will not change in M2 when real conversions
  land (exec pack D-M1-6).
- M1: `SimulationEngine` not present yet; arrives in T1.9.

## See also

- Master spec §5.3 — unit system support policy
- [`docs/development/m1_execution_pack.md`](../../docs/development/m1_execution_pack.md) T1.2

---

*Template: `docs/development/templates/MODULE_README_template.md` v1.0.*
