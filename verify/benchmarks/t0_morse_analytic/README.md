# Benchmark T0 — Morse analytic dimer (metal + lj cross-check)

<!-- markdownlint-disable MD013 -->

**Tier:** fast (`verify/SPEC.md` §8.2).
**Purpose:** the first gate of the verification pyramid — a unit-sanity test
that pins the force / PE pipeline to a closed-form answer **before** any
LAMMPS-based comparison (T1+) is trusted. If T0 is red, no differential
result is meaningful.

T0 is also the M2 cross-check for `UnitConverter`: the same physical system
is expressed in both `units: metal` and `units: lj` (with identity reference),
and post-ingest state must be bitwise-identical.

## System

| Quantity          | Value                            |
|-------------------|----------------------------------|
| Atoms             | 2, single species                |
| Separation `r`    | 3.5 Å  (= `r0` + 0.5)            |
| Positions         | (8.25, 10, 10), (11.75, 10, 10)  |
| Box               | 20 × 20 × 20 Å (periodic)        |
| Potential         | Morse                            |
| `D`               | 1.0 eV                           |
| `α`               | 1.0 Å⁻¹                          |
| `r0`              | 3.0 Å                            |
| cutoff            | 6.0 Å                            |
| cutoff treatment  | `hard_cutoff`                    |
| Initial velocity  | 0 (system at rest)               |
| Integrator        | velocity_verlet (unused)         |
| Timestep          | 0.001 ps (unused at 0 steps)     |
| Steps             | 0                                |

The 20 Å box satisfies the M1 neighbor-list stencil constraint
`L ≥ 3·(cutoff + skin)` = 18.9 Å (see `neighbor/SPEC.md`), so the same cell
grid works with and without periodic-image neighbour pairs — at r = 3.5 Å
along x there is exactly one pair inside the cutoff (the direct bond).

## Why `hard_cutoff`

The closed-form Morse force

```text
|F(r)| = 2·D·α·(1 − e) · e,   where e = exp(−α·(r − r0))
```

has no cutoff correction. TDMD's production default
(`shifted_force`) subtracts a linear ramp so `F(r_c) = 0` exactly, which
diverges from the analytic expression by an O(exp(−α·(r_c − r0))) constant.
For r = 3.5 Å, α = 1, r_c = 6 Å that correction is ~5·10⁻² · 0.049 ≈ 2.5·10⁻³
on |F| — orders of magnitude larger than our 1·10⁻¹² threshold. We therefore
pin `cutoff_strategy: hard_cutoff` on both configs.

## Analytic reference values

TDMD (and LAMMPS) use the `U(r0) = -D` convention — i.e. the depth of the
potential well equals `D`, and `U(∞) = 0` under `hard_cutoff`:

```text
δ   = r − r0 = 0.5
e   = exp(−0.5) ≈ 0.6065306597126334
PE  = D · (1 − e)² − D          = D · (e² − 2·e)    ≈ −0.845170800202508...  eV
|F| = −dU/dr = 2·D·α·(1 − e)·e                       ≈ +0.477302937960458...  eV/Å
```

The `U(r0) = 0` textbook form (`PE = D·(1-e)²`) differs by the additive
constant `D`; forces are identical. The test file keeps the two conventions
in one place so the next reader does not rediscover this the hard way.

- `F_x` on atom 0 (lower-x) = +|F|  (pulled toward atom 1)
- `F_x` on atom 1 (higher-x) = −|F| (pulled toward atom 0)
- `F_y`, `F_z` = 0 on both atoms (colinear along x)

## Metal / lj identity-reference cross-check

With `reference: {sigma: 1, epsilon: 1, mass: 1}` the lj → metal conversion
is:

- length:  `x_metal = x_lj · σ = x_lj` (identity)
- mass:    `m_metal = m_lj · m_ref = m_lj` (identity)
- energy:  `U_metal = U_lj · ε = U_lj` (identity)
- velocity / time: scale by `sqrt(mvv2e)` factor which does **not** collapse
  to 1 under identity σ/ε/m. For T0 this is irrelevant (system at rest and
  0 steps — no time integration).

Under these conditions the post-ingest `AtomSoA` (positions, masses, types,
forces) from the lj config must match the metal config bitwise. The test
asserts this directly with `REQUIRE` rather than a tolerance.

## Files

| File                    | Role                                                |
|-------------------------|-----------------------------------------------------|
| `README.md`             | This document                                       |
| `config_metal.yaml`     | TDMD config, `units: metal`                         |
| `config_lj.yaml`        | TDMD config, `units: lj` + identity reference       |
| `setup_metal.data`      | Atom data file (Å, g/mol, Å/ps)                     |
| `setup_lj.data`         | Atom data file in lj units (bit-identical with identity σ=ε=m=1) |
| `checks.yaml`           | Threshold-registry path declarations                |

## How to run

```bash
# Runs under ctest (pure C++, no LAMMPS):
ctest --test-dir build_cpu -R test_t0_analytic -V
```

The test is pure C++ and has no external dependencies — no LAMMPS oracle,
no Python harness. The driver loads each config via `tdmd::io::parse_yaml_config`,
inits `tdmd::SimulationEngine` in-process, reads forces from `engine.atoms()`
(which populates on init — see `runtime/SPEC.md` §2.2), and compares against
the closed-form reference values above.

## Comparison scope

| Check                                   | Tolerance              |
|-----------------------------------------|------------------------|
| `F_x` on each atom vs analytic          | 1·10⁻¹² relative       |
| `F_y`, `F_z` = 0 exactly                | `== 0.0` bitwise        |
| Potential energy vs analytic            | 1·10⁻¹² relative       |
| Metal state vs lj-identity state         | bitwise (`==`) on positions, masses, forces |

Velocities are not compared (system at rest, both sides are exactly 0).
