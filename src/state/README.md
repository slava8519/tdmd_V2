# state

**Status:** skeleton / under development
**SPEC:** [`docs/specs/state/SPEC.md`](../../docs/specs/state/SPEC.md)
**TESTPLAN:** not yet written (M1)
**Owner role:** Core Runtime Engineer (see playbook §2)

## What this module does

Owns atom state on each MPI rank: positions, velocities, forces, species, and the associated Structure-of-Arrays (SoA) storage. Provides stable identity across timesteps and is the sole allowed mutator of atom coordinates.

## Scope boundaries

- This module **owns**: `AtomSoA`, `StateManager`, migration bookkeeping, SoA memory layout.
- This module **does not own**: neighbor lists (see `neighbor/`), force kernels (see `potentials/`), integration semantics (see `integrator/`).

## Public API at a glance

```cpp
namespace tdmd {
    struct AtomSoA;          // per-rank atom SoA container
    class StateManager;      // lifecycle, add/remove, migration (M1)
}
```

Full API: [SPEC §2](../../docs/specs/state/SPEC.md#2-public-interface).

## Dependencies

- **Uses:** none in M0.
- **Used by:** `neighbor/`, `integrator/`, `potentials/` (all M1+).
- **External:** none in M0.

## Build

```bash
cmake --preset default
cmake --build build --target tdmd_state
```

See [`docs/development/build_instructions.md`](../../docs/development/build_instructions.md).

## Tests

```bash
ctest --test-dir build --tests-regex state
```

## Examples

See `tests/state/` for usage examples until proper examples land.

## Known limitations

- M0: only a skeleton (empty `AtomSoA` struct). Real fields and invariants arrive in M1.

## Telemetry

See [SPEC §6](../../docs/specs/state/SPEC.md#6-telemetry-hooks) (TBD).

## See also

- [`../neighbor/`](../neighbor/README.md)
- [`../integrator/`](../integrator/README.md)

---

*Template: `docs/development/templates/MODULE_README_template.md` v1.0.*
