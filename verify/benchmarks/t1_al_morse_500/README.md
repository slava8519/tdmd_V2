# Benchmark T1 — Al FCC Morse NVE (500 atoms, 100 steps)

<!-- markdownlint-disable MD013 -->

**Tier:** fast (`verify/SPEC.md` §8.2).
**Purpose:** first numerical validation gate of the project. Establishes the
differential-vs-LAMMPS harness (`run_differential.py`) and exercises the
end-to-end pipeline: YAML config → LAMMPS-data ingest → build → force / thermo
loop → comparison against the LAMMPS oracle.

## System

| Quantity             | Value                                    |
|----------------------|------------------------------------------|
| Element              | Al                                       |
| Lattice              | FCC, `a = 4.05 Å`                        |
| Size                 | 5 × 5 × 5 unit cells (500 atoms)         |
| Box                  | 20.25 × 20.25 × 20.25 Å (periodic)       |
| Potential            | Morse, Girifalco–Weizer parameters       |
| `D`                  | 0.2703 eV                                |
| `α`                  | 1.1646 Å⁻¹                               |
| `r₀`                 | 3.253 Å                                  |
| cutoff               | 6.0 Å (see §"cutoff choice" below)       |
| cutoff treatment     | hard truncation (`hard_cutoff`)          |
| Initial temperature  | 300 K, seed 12345                        |
| Integrator           | velocity-Verlet (NVE)                    |
| Timestep             | 0.001 ps                                 |
| Steps                | 100                                      |
| Thermo output period | every 10 steps                           |

## Cutoff choice

The Girifalco–Weizer cutoff is commonly quoted at ~8 Å. For the 500-atom
5×5×5 box (L = 20.25 Å) that value violates the CellGrid invariant
`L ≥ 3·(r_c + skin)` (M1 only supports single-cell stencil, see
`neighbor/SPEC.md`). T1 therefore uses `r_c = 6.0 Å`, which still captures the
first four neighbour shells of FCC Al (2.86, 4.05, 4.96, 5.73 Å) and keeps
the differential run self-consistent on both sides. Larger-cutoff variants
are exercised from T2 onwards on larger boxes.

## Cutoff strategy parity

LAMMPS' `pair_style morse` applies a hard truncation at `r_c` (no shift on
either energy or force). TDMD's `cutoff_strategy: hard_cutoff` matches this
behaviour bit-for-bit by design. TDMD's production default
(`shifted_force`) is intentionally NOT used here — that variant differs from
LAMMPS by a finite (non-roundoff-sized) ramp, and the T1 harness is about
measuring FP64 reduction-order residuals, not comparing two distinct physical
cutoff treatments. Strategy-B parity (LAMMPS `pair_modify shift yes`) is
deferred to a later milestone.

## Files in this directory

| File                        | Role                                                |
|-----------------------------|-----------------------------------------------------|
| `README.md`                 | This document                                       |
| `config_metal.yaml`         | TDMD config, `units: metal` (points at `setup.data` produced by LAMMPS) |
| `config_lj.yaml`            | TDMD config, `units: lj` + identity reference — paired with `config_metal.yaml` for the D-M1-6 cross-check (see `verify/SPEC.md` §4.5.1) |
| `lammps_script_metal.in`    | LAMMPS input: build lattice, assign velocities, emit `setup.data`, run, emit thermo + `final.data` |
| `checks.yaml`               | Which thresholds (from `verify/thresholds/thresholds.yaml`) this benchmark exercises |

## How to run locally

```bash
# From the repo root, after a CPU build (see tools/lammps_smoke_test.sh for
# the oracle side) and building tdmd:
python3 verify/t1/run_differential.py \
    --benchmark verify/benchmarks/t1_al_morse_500 \
    --tdmd build_cpu/src/cli/tdmd \
    --lammps verify/third_party/lammps/install_tdmd/bin/lmp \
    --lammps-libdir verify/third_party/lammps/install_tdmd/lib \
    --thresholds verify/thresholds/thresholds.yaml \
    --variant both          # metal + lj cross-check (default: metal only)
```

The harness exits 0 on pass, non-zero on any threshold violation. When the
LAMMPS binary is missing the harness prints a clear SKIP message and exits
with the dedicated skip code (77) — consumed by the Catch2 wrapper to
translate into a CI-skip (not a failure).

## lj cross-check (M2, T2.4)

`--variant both` additionally runs TDMD with `config_lj.yaml` (identity
reference σ=ε=m_ref=1). The harness pre-scales the Velocities block of the
LAMMPS-produced `setup.data` by `sqrt(mvv2e_metal) ≈ 0.01018` so the lj→metal
conversion inside `UnitConverter::velocity_from_lj` recovers the original
Å/ps magnitudes — `dt_lj = 0.001 / sqrt(mvv2e)` does the same for time.
Length/mass/energy columns pass through untouched (identity multiply-by-1).

Since both TDMD runs emit metal-unit thermo regardless of input, the
cross-diff is a direct column-by-column comparison at
`benchmarks.t1_al_morse_500.cross_unit_relative = 1.0e-10`. The measured
residual is `0.0` on every column at every step — the D-M1-6 transparency
invariant is closed bit-exactly. See `verify/SPEC.md` §4.5.1 for the full
rationale.

## Comparison scope (M1)

T1 at M1 compares thermo row-wise: `step`, `temp`, `pe`, `ke`, `etotal`, `press`.
Positions / velocities dumped via LAMMPS `write_data` / TDMD full-state dump
are reserved for M2 once the TDMD dump writer (see `io/SPEC.md` §5.1) lands.
The threshold registry already carries `kinematics.*` entries so the M2
extension is additive.
