# neighbor

**Status:** skeleton / under development
**SPEC:** [`docs/specs/neighbor/SPEC.md`](../../docs/specs/neighbor/SPEC.md)
**TESTPLAN:** not yet written (M1)
**Owner role:** Neighbor / Migration Engineer (see playbook §2)

## What this module does

Builds and maintains the cell grid and neighbor list used by short-range force kernels. Owns rebuild-trigger policy (skin), displacement tracking, and stable reordering required by the Reference BuildFlavor.

## Scope boundaries

- This module **owns**: `CellGrid`, `NeighborList`, `DisplacementTracker`, skin policy, stable reorder.
- This module **does not own**: atom data (see `state/`), MPI halo exchange (see `comm/`), potential evaluation (see `potentials/`).

## Public API at a glance

```cpp
namespace tdmd {
    struct CellGrid;           // cell partitioning of the simulation box
    struct NeighborList;       // pairwise neighbor data
    class NeighborManager;     // orchestration (M1+)
}
```

Full API: [SPEC §2](../../docs/specs/neighbor/SPEC.md#2-public-interface).

## Dependencies

- **Uses:** `state/` (read-only access to atom coordinates).
- **Used by:** `potentials/`, `scheduler/` (M1+).
- **External:** none in M0.

## Build

```bash
cmake --preset default
cmake --build build --target tdmd_neighbor
```

## Tests

```bash
ctest --test-dir build --tests-regex neighbor
```

## Examples

See `tests/neighbor/` for usage examples until proper examples land.

## Known limitations

- M0: only a skeleton (empty `CellGrid` struct). Real partitioning algorithm arrives in M1.

## Telemetry

See [SPEC §6](../../docs/specs/neighbor/SPEC.md#6-telemetry-hooks) (TBD).

## See also

- [`../state/`](../state/README.md)
- [`../integrator/`](../integrator/README.md)

---

*Template: `docs/development/templates/MODULE_README_template.md` v1.0.*
