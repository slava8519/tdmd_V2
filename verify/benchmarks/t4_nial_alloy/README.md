# T4 — Ni-Al EAM/alloy differential benchmark

**M2 ACCEPTANCE GATE.** Failing this benchmark blocks closure of milestone M2
(master spec §14 M2, docs/specs/verify/SPEC.md §4.7).

## Purpose

First end-to-end TDMD-vs-LAMMPS differential against an **EAM/alloy**
potential — the canonical many-body pair model for metal alloys. Exercises
every layer a production physics run depends on:

| Layer | Exercised here |
|-------|---------------|
| setfl parser (`potentials::parse_eam_alloy`) | Mishin 2004 Ni-Al setfl file verbatim |
| `EamAlloyPotential` two-pass force kernel  | 864 atoms, ~60k pair interactions per step |
| `TabulatedFunction` Hermite spline          | Same table on both engines → bit-match achievable |
| `SimulationEngine` multi-species dispatch   | First non-trivial multi-species run |
| T2.8 forces-diff layer                      | Per-atom forces at step 100 |

Everything before T4 exercises at most one of these; T4 is the first test
where a regression anywhere in the stack surfaces as a differential failure.

## System

- **Lattice**: FCC, 6 × 6 × 6 conventional cells, a₀ = 3.52 Å (Ni lattice).
- **Atoms**: 864 total. 50 : 50 Ni : Al (432 each), random shuffle with
  Python's `random.Random(12345)`. LAMMPS type 1 = Ni (mass 58.71),
  type 2 = Al (mass **26.982** — verbatim from the setfl header, not
  26.9815385; see "Known non-issues" below). Ordering pinned to the Mishin
  2004 setfl file's `species_names = ["Ni", "Al"]`.
- **Initial velocities**: Maxwell-Boltzmann at 300 K, same PRNG seed; COM
  momentum subtracted; kinetic energy rescaled so reported temperature is
  exactly 300 K.
- **Potential**: Mishin 2004 EAM/alloy, `pair_style eam/alloy`. See
  `verify/third_party/potentials/README.md` for provenance.
- **Integrator**: Velocity-Verlet NVE, dt = 0.001 ps.
- **Run length**: 100 steps (same profile as T1, so thermo comparability is
  apples-to-apples).
- **Boundary**: Fully periodic (pp pp pp).

## Initial state

`setup.data` is **committed** (not LAMMPS-generated). Both TDMD and LAMMPS
consume it verbatim, so the two engines start from bit-identical atoms. The
file is emitted by `generate_setup.py` in this directory; re-running the
script with the unchanged seed reproduces `setup.data` byte-for-byte.

To regenerate:

```bash
python3 verify/benchmarks/t4_nial_alloy/generate_setup.py
```

Do **not** edit `setup.data` by hand — the harness's differential residual
depends on the two engines seeing the exact same bytes.

## Parameter file

`verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy` — committed
verbatim from the NIST Interatomic Potentials Repository. See
[third_party README](../../third_party/potentials/README.md) for provenance
and license. The LAMMPS submodule (`verify/third_party/lammps/`) ships
**without** this file — it lives in the NIST repository, not in the LAMMPS
examples tree, so we committed it alongside the benchmark.

> **Deviation from exec pack T2.9 spec**: the exec pack originally assumed
> the Mishin file would be fetched at harness setup time (~50 KB, not
> committed). The actual NIST file is 1.9 MB of cubic-spline tables and is
> licensed public domain — we commit it verbatim in `third_party/potentials/`
> for reproducibility and to avoid a runtime dependency on nist.gov.

## Acceptance criteria

Thresholds live in `verify/thresholds/thresholds.yaml` under
`benchmarks.t4_nial_alloy.*`. In brief:

| Observable           | Threshold (relative) | Rationale |
|----------------------|----------------------|-----------|
| Potential energy     | 1e-10                | FP64 reduction-order only |
| Kinetic energy       | 1e-10                | FP64 reduction-order only |
| Total energy         | 1e-10                | Sum of the above |
| Temperature          | 2e-6                 | k_B discrepancy (same as T1) |
| Pressure             | 1e-8                 | nktv2p matched to LAMMPS |
| **Forces (per-atom per-component)** | **1e-10** | **M2 gate clause** |

Measured residuals at T2.9 landing (x86_64 + gcc-13):

| Observable | max rel | headroom |
|------------|---------|----------|
| potential energy | 0.0 (bit-identical) | ∞ |
| kinetic energy   | 0.0 (bit-identical) | ∞ |
| total energy     | 0.0 (bit-identical) | ∞ |
| temperature      | 1.13e-6             | 1.77× under |
| pressure         | 3.65e-11            | 270× under  |
| **forces (per-atom per-component)** | **3.25e-11** | **3.08× under** |

Forces at 3e-11 against a 1e-10 budget leaves headroom for FP64 reduction-
order drift on other toolchains without raising the threshold.

## Running

Locally (after `tools/build_lammps.sh`):

```bash
cmake --build build_cpu --target test_t4_differential tdmd
ctest --test-dir build_cpu -R test_t4_differential --output-on-failure
```

Or invoke the Python harness directly:

```bash
python3 verify/t4/run_differential.py \
  --benchmark   verify/benchmarks/t4_nial_alloy \
  --tdmd        build_cpu/src/cli/tdmd \
  --lammps      verify/third_party/lammps/install_tdmd/bin/lmp \
  --lammps-libdir verify/third_party/lammps/install_tdmd/lib \
  --thresholds  verify/thresholds/thresholds.yaml
```

On public CI the LAMMPS oracle is absent, so the test SKIPs cleanly (Option A,
see `docs/development/m0_execution_pack.md`).

## Files

| File | Purpose |
|------|---------|
| `README.md`               | this file |
| `generate_setup.py`       | one-shot Python generator for `setup.data` |
| `setup.data`              | committed initial state (864 atoms) |
| `config_metal.yaml`       | TDMD input — metal units |
| `lammps_script_metal.in`  | LAMMPS oracle script — metal units |
| `checks.yaml`             | thermo + forces comparison declarations |

No LJ variant — EAM setfl tables are dimensional by convention, and the
`SimulationEngine::init` dispatch rejects `units=lj` for `style=eam/alloy`
with an explicit error.

## Known non-issues

- **Temperature thermometer drift** (~1.13e-6 rel): comes from LAMMPS using
  the older truncated `boltz = 8.617343e-5` vs TDMD's CODATA 2018
  `8.617333262e-5`. Same residual as T1; widening the threshold would
  require changing a physical constant, which we refuse.
- **Forces ordering**: LAMMPS freely reorders atoms during migration.
  TDMD's T2.8 dump layer sorts rows by atom id (`write_dump_frame`) and the
  LAMMPS script uses `dump_modify sort id`, so both on-disk files are in
  canonical order and the forces-diff aligns by id without extra work.
- **Al mass is 26.982, not 26.9815385**: LAMMPS `pair_eam_alloy::coeff()`
  silently overrides atom masses with values from the setfl header. The
  Mishin 2004 file declares Al = 26.982, so `setup.data` must match or the
  two engines disagree on KE at step 0 by ~1.7e-5 rel (which propagates
  into 3 % forces drift at step 100 via integrator divergence — discovered
  while bringing T4 up). Both the generator and the committed data file
  use 26.982 for this reason.
