# integrator

**Status:** skeleton / under development
**SPEC:** [`docs/specs/integrator/SPEC.md`](../../docs/specs/integrator/SPEC.md)
**TESTPLAN:** not yet written (M1)
**Owner role:** Physics Engineer (see playbook §2)

## What this module does

Implements time integration schemes: Velocity-Verlet (NVE), Nose-Hoover (NVT/NPT), Langevin. Defines the per-zone time step interface that the scheduler consumes.

## Scope boundaries

- This module **owns**: `Integrator` interface, concrete integrator implementations, timestep-related policies.
- This module **does not own**: scheduling/safety certificates (see `scheduler/`), atom data mutation (see `state/` — integrator asks state to apply updates), force evaluation (see `potentials/`).

## Public API at a glance

```cpp
namespace tdmd {
    class Integrator;                    // abstract interface
    class VelocityVerletIntegrator;      // NVE (M1)
    class NoseHooverNvtIntegrator;       // NVT (M1+)
    class NoseHooverNptIntegrator;       // NPT (M1+)
    class LangevinIntegrator;            // stochastic NVT (M1+)
}
```

Full API: [SPEC §2](../../docs/specs/integrator/SPEC.md#2-public-interface).

## Dependencies

- **Uses:** `state/` (atom data), `potentials/` (force evaluation) — both M1+.
- **Used by:** `scheduler/` (calls `advance(dt)` per zone) — M4+.
- **External:** none in M0.

## Build

```bash
cmake --preset default
cmake --build build --target tdmd_integrator
```

## Tests

```bash
ctest --test-dir build --tests-regex integrator
```

## Examples

See `tests/integrator/` for usage examples until proper examples land.

## Known limitations

- M0: only the abstract `Integrator` interface skeleton. Concrete implementations arrive in M1. NVT/NPT in TD-mode is restricted to `K=1` in v1.5 (master spec §14 M9).

## Telemetry

See [SPEC §6](../../docs/specs/integrator/SPEC.md#6-telemetry-hooks) (TBD).

## See also

- [`../state/`](../state/README.md)
- [`../neighbor/`](../neighbor/README.md)
- Master spec §14 M9 (NVT/NPT restriction), integrator/SPEC §7.3.

---

*Template: `docs/development/templates/MODULE_README_template.md` v1.0.*
