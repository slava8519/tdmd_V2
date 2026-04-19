# Benchmark T3 — Al FCC 10⁶ LJ NVE (anchor-test)

<!-- markdownlint-disable MD013 MD033 -->

**Tier:** slow (`verify/SPEC.md` §8.2).
**Purpose:** **existence proof** of the TDMD project — reproduce Andreev's 2007
dissertation figures 29–30 (performance + efficiency vs. `N_processors`) on the
Ring backend, 1D-linear zoning along Z, within 10 % after hardware
normalisation. The M5 milestone **cannot merge** without the harness (T5.11)
exercising this fixture and reporting PASS.

This directory is the **fixture** half of T3: config, reference data, LAMMPS
parity script, acceptance criteria, and the setup-regen script. The driver
(`anchor_test_runner.py`) lives in `verify/harness/anchor_test_runner/` and
lands at T5.11.

## Experiment

| Quantity              | Value                                           |
|-----------------------|-------------------------------------------------|
| Element               | Al                                              |
| Lattice               | FCC, `a = 4.05 Å`                               |
| Size                  | 63 × 63 × 63 unit cells → 1 000 188 ≈ 10⁶ atoms |
| Box                   | 255.15 × 255.15 × 255.15 Å (periodic xyz)       |
| Potential             | Lennard-Jones (Andreev §3.5 parameters)         |
| `ε`                   | 0.3930 eV                                       |
| `σ`                   | 2.620 Å                                         |
| cutoff                | 8.0 Å (hard truncation — matches dissertation)  |
| Initial temperature   | 300 K, seed 12345                               |
| Integrator            | velocity-Verlet (NVE)                           |
| Timestep              | 0.001 ps (1 fs)                                 |
| Steps                 | 1000                                            |
| Thermo output period  | every 100 steps                                 |
| Zoning scheme         | `linear_1d` (§3.1 — Andreev §2.2 baseline)      |
| Comm backend          | `ring` (§7.2 Kahan ring reduction)              |
| Pipeline depth `K`    | 1 (default — bit-exact with M4 regression)      |
| Ranks probed          | 4, 8, 16, 32 (dissertation figures 29–30)       |

## Provenance of reference data

`dissertation_reference_data.csv` holds the per-`N_procs` performance and
parallel-efficiency points extracted from Andreev's figures 29 and 30. Andreev
reports his numbers relative to a 2007-era Intel Xeon Harpertown node (~9
GFLOP/core, peak); `hardware_normalization.py` produces a scalar
`ghz_flops_ratio` from a synthetic LJ kernel micro-benchmark run on the current
machine so the dissertation datapoints can be compared against today's silicon
on an equal-work basis.

**STATUS:** the initial CSV shipped with T5.10 is a **preliminary placeholder**
derived from the asymptotic Andreev-§2.2 Linear1D model (`efficiency =
1 - α·(N-1)/N` with `α ≈ 0.05` — the typical halo-overhead coefficient for
short-range LJ at `N_min = 2`). These values must be replaced by a human
extracting points from the scanned figures via WebPlotDigitizer (or
equivalent) before T5.11 runs as a gating CI check. The placeholder is shaped
correctly (monotone-decreasing efficiency, monotone-increasing absolute
performance, plausible slope) so T5.11 harness code can be developed and
unit-tested against it, but the 10 %-within-dissertation invariant is
meaningless until the replacement lands. See `R-M5-8` in the M5 execution
pack.

## Files in this directory

| File                                | Role                                        |
|-------------------------------------|---------------------------------------------|
| `README.md`                         | This document                               |
| `config.yaml`                       | TDMD config (LJ, linear_1d, ring, NVE)      |
| `lammps_script.in`                  | LAMMPS parity script (physics cross-check)  |
| `dissertation_reference_data.csv`   | Extracted figures 29–30 points              |
| `hardware_normalization.py`         | Current-hw ↔ 2007-Harpertown scalar         |
| `checks.yaml`                       | Acceptance threshold declarations           |
| `acceptance_criteria.md`            | Pass/fail rules + escalation path           |

Companion data (LAMMPS-generated lattice):

| File                                          | Role                              |
|-----------------------------------------------|-----------------------------------|
| `../../data/t3_al_fcc_large_anchor/regen_setup.sh` | One-shot LAMMPS regeneration |
| `../../data/t3_al_fcc_large_anchor/setup.data`     | 10⁶-atom lattice (not checked in — see below) |

**`setup.data` is intentionally not committed.** The file is ~95 MiB
uncompressed (≈24 MiB under `xz -9`) and trivially regeneratable from
`regen_setup.sh` in ~30 seconds of LAMMPS CPU time. Following the T1 precedent
(`t1_al_morse_500` also regenerates its `setup.data` at harness run-time
rather than committing a binary blob), T5.11's harness will invoke
`regen_setup.sh` before the first TDMD run in a fresh workspace. This keeps
the repository git-LFS-free and the fixture self-contained; the cost is a
one-time LAMMPS warm-up per CI run.

## How to regenerate `setup.data` manually

```bash
cd verify/data/t3_al_fcc_large_anchor
./regen_setup.sh \
    --lammps <path-to-lmp>    # e.g. verify/third_party/lammps/install_tdmd/bin/lmp
```

The script is idempotent — a second invocation recognises an existing
`setup.data` and skips. Pass `--force` to overwrite.

## Comparison scope (M5)

T3 at M5 compares against the dissertation, **not** against LAMMPS — the 10 %
tolerance explicitly accepts reference-extraction uncertainty (±2 % per
R-M5-8) plus hardware normalisation noise (±5 %) plus TDMD's own run-to-run
variance (±1 %). The LAMMPS parity script (`lammps_script.in`) runs the same
LJ physics on the same lattice and is consumed by a separate sanity check
(bit-exact thermo at step 0, within `thermo_energy_relative` at step 1000) so
T3 provides two independent gates: dissertation fidelity (primary) and
LAMMPS LJ-physics parity (secondary).
