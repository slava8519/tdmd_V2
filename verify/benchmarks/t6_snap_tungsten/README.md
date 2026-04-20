# T6 — SNAP tungsten differential (`t6_snap_tungsten`)

**Status:** **D-M8-7 byte-exact gate GREEN** — landed M8 **T8.5 (2026-04-20)**.
CPU FP64 TDMD SnapPotential ≡ LAMMPS FP64 `pair_style snap` to per-atom-force
max_rel ≈ 8.8e-13 under 1e-12 budget on 250-atom BCC W, 100 NVE steps. Thermo
PE/KE/Etotal match to FP64 bytes, Temperature/Pressure within T1/T4 precedent
budgets. Pure SNAP path only (no ZBL — deferred M9+); the canonical production
fixture `lammps_script.in` retains `hybrid/overlay zbl+snap` for physics
realism, but the byte-exact gate (`lammps_script_metal.in` + `config_metal.yaml`)
scopes to the SNAP bispectrum path standalone. Scaling to 1024/8192-atom
variants deferred to T8.10. See §14 M8 acceptance gate in `TDMD_Engineering_Spec.md`
and `docs/development/m8_execution_pack.md` T8.5 / T8.10.

## Purpose

T6 is TDMD's **SNAP proof-of-value** differential benchmark — the M8 primary gate. It exercises the full SNAP code path (bispectrum → energy → force) on the canonical Wood & Thompson 2017 tungsten fixture against LAMMPS ML-SNAP as oracle, and feeds the M8 acceptance gate (≥ 20 % speedup vs LAMMPS on ≥ 8 ranks, OR honest documented why-not — master spec §14 M8).

Scientific justification for this fixture choice: see `docs/specs/verify/SPEC.md` §4.7 (canonical fixture) and `docs/specs/potentials/SPEC.md` §6 (SNAP module contract).

## Canonical fixture (D-M8-3)

| Property | Value |
|---|---|
| Coefficient file | `W_2940_2017_2.snap` (include wrapper), + `.snapcoeff` (56 linear coefficients) + `.snapparam` |
| Source | Wood & Thompson, arXiv:1702.07042 (2017), 2940 DFT training configurations |
| Species | Pure W (single species) |
| Crystal | BCC, lattice parameter `a = 3.1803` Å |
| Pair style | `hybrid/overlay zbl 4.0 4.8 snap` |
| `twojmax` | 8 (→ `k_max = 55` = 56 coefficients − 1) |
| `rcutfac` | 4.73442 |
| Max cutoff | `rcutfac · (R_W + R_W) = 4.73442 · (0.5 + 0.5) = 4.73442` Å (effective) |

**Path resolution (M8 D-M6-6 Option A pattern):**

```
verify/third_party/lammps/examples/snap/W_2940_2017_2.snap
verify/third_party/lammps/examples/snap/W_2940_2017_2.snapcoeff
verify/third_party/lammps/examples/snap/W_2940_2017_2.snapparam
```

The TDMD repo does **not** track these files (D-M8-3 repo-size preservation); they arrive with the M1-landed LAMMPS submodule `verify/third_party/lammps/` (pin `stable_22Jul2025_update4`). Benchmark driver resolves them via `LAMMPS_EXAMPLES_DIR` env var (defaults to the submodule path) so the same script works on cloud-burst machines where the submodule may sit at a different path.

## Test configurations

T6 ships **three size variants** per m8 exec pack §4 T8.10:

| Variant | Box | N | Run length | Purpose |
|---|---|---|---|---|
| `small` (landed T8.5) | **5×5×5** BCC W — see note below | **250** | 100 steps | Smoke, CI, single-rank Fp64Reference byte-exact oracle |
| `medium` | 8×8×8 BCC W | 1024 | 500 steps | Scaling probe input (T8.10) |
| `large` | 16×16×16 BCC W | 8192 | 1000 steps | Scaling probe (cloud-burst T8.11) |

**Why 5×5×5 not 4×4×4 for the `small` variant:** TDMD's CellGrid halo stencil
requires `L_axis ≥ 3·(cutoff + skin)`. For SNAP W (cutoff 4.73442 Å + skin 0.3 Å)
that is 15.10 Å; the advertised 4×4×4 box is only 12.72 Å and fails box-
too-small at ingest. 5×5×5 (15.90 Å) is the smallest BCC variant the SNAP path
can run on; 8×8×8 and 16×16×16 clear the constraint with margin. This is a
TDMD-side constraint, not a LAMMPS one — documented in `generate_setup.py`.

All variants use `dt = 5e-4 ps`, `T₀ = 300 K`, `seed = 12345` (landed T8.5
generator; upstream LAMMPS Wood 2017 fixture used 4928459 but that value only
matters when `velocity create` is the source of initial velocities — this
fixture pre-computes velocities in Python to avoid LAMMPS/TDMD RNG divergence),
NVE integration, thermo every 10 steps.

## Acceptance thresholds

Threshold entries in `verify/thresholds/thresholds.yaml`:

- **CPU FP64 byte-exact (D-M8-7):** `benchmarks.t6_snap_tungsten.cpu_fp64_vs_lammps.*` — per-atom force ≤ 1e-12 rel, total PE ≤ 1e-12 rel. Exercised by T8.5.
- **GPU FP64 byte-exact (D-M6-7 extension):** `benchmarks.t6_snap_tungsten.gpu_fp64_vs_cpu_fp64.*` — per-atom force ≤ 1e-12 rel. Exercised by T8.7.
- **GPU MixedFast dense-cutoff (D-M8-8):** `benchmarks.gpu_mixed_fast_snap_only.snap.*` — SNAP force ≤ 1e-5 rel, SNAP energy ≤ 1e-7 rel, NVE drift ≤ 1e-5 per 1000 steps. Exercised by T8.9 + T8.12.

See `checks.yaml` for the authoritative list of registered checks.

## Oracle subset gate (landed T8.2)

Before T6 measurement can run, the LAMMPS oracle must be operational:

```bash
# Submodule initialized and built (tools/build_lammps.sh with PKG_ML-SNAP=on):
./verify/third_party/lammps/install_tdmd/bin/lmp -h | grep -iE 'ML-SNAP'
#   → non-empty match confirms ML-SNAP package is built in

# Upstream example runs cleanly:
cd verify/third_party/lammps/examples/snap/
LD_LIBRARY_PATH=../../install_tdmd/lib ../../install_tdmd/bin/lmp -in in.snap.W.2940
#   → 1.2 s on dev hardware; thermo matches log.15Jun20.snap.W.2940.g++.1 to 5-decimal
```

Path-existence Catch2 gate (T8.2): `tests/potentials/test_lammps_oracle_snap_fixture` self-skips (exit 77) on uninitialized submodule, fails if fixture files are missing from a correctly initialized submodule.

## Running the benchmark (T8.5 onwards)

**End-to-end differential harness (landed T8.5):**

```bash
# Fp64ReferenceBuild CPU binary (byte-exact oracle):
python3 verify/t6/run_differential.py \
    --benchmark verify/benchmarks/t6_snap_tungsten \
    --tdmd build-cpu-strict/src/cli/tdmd \
    --lammps verify/third_party/lammps/install_tdmd/bin/lmp \
    --lammps-libdir verify/third_party/lammps/install_tdmd/lib \
    --thresholds verify/thresholds/thresholds.yaml
# → PASS at max_rel ≈ 8.8e-13 forces (under 1e-12 budget).
```

Catch2 wrapper exposing this as a ctest target:

```bash
ctest -R test_t6_differential --output-on-failure
# → Pass at ~17s (LAMMPS+TDMD end-to-end).
```

Canonical production fixture (ZBL+SNAP, for physics-realism reference, NOT
byte-exact gate):

```bash
./verify/third_party/lammps/install_tdmd/bin/lmp \
    -var workdir /tmp/t6_out \
    -var nrep 4 \
    -var nsteps 100 \
    -var snap_dir $PWD/verify/third_party/lammps/examples/snap \
    -in verify/benchmarks/t6_snap_tungsten/lammps_script.in
```

## Status checklist (M8 T8.5 / T8.10)

- [x] **T8.10a** — Scaffold landed 2026-04-20: `README.md`, `checks.yaml`, `lammps_script.in`, threshold-registry entries `benchmarks.t6_snap_tungsten.*` in `verify/thresholds/thresholds.yaml`.
- [x] **T8.5** — D-M8-7 byte-exact gate landed 2026-04-20: `generate_setup.py` (5×5×5 BCC W, 250 atoms), `setup.data` (committed), `lammps_script_metal.in` (pure SNAP, no ZBL), `config_metal.yaml` (TDMD style:snap), `verify/t6/run_differential.py` driver, `verify/t6/test_t6_differential.cpp` Catch2 wrapper. Forces max_rel ≈ 8.8e-13 under 1e-12 budget. ctest 39/39 green.
- [ ] **T8.10** — Medium/large variants (1024, 8192 atoms) + extended run lengths.
- [ ] **T8.11** — Scaling probe driver (8-rank baseline + cloud-burst scaling). Feeds `verify/benchmarks/t6_snap_scaling/results_<date>.json` artefact for M8 acceptance gate close.

## Cross-references

- `docs/specs/verify/SPEC.md` §4.7 — canonical fixture choice rationale.
- `docs/specs/potentials/SPEC.md` §6 — SNAP module interface and byte-exact mandate.
- `docs/development/m8_execution_pack.md` §4 T8.10 — full task template.
- `verify/third_party/lammps_README.md` — SNAP fixture section (artefact table + sanity-run snippet, landed T8.2).
- `verify/thresholds/thresholds.yaml` `benchmarks.t6_snap_tungsten` + `benchmarks.gpu_mixed_fast_snap_only` — threshold registry.
