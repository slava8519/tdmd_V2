# M8 T6 integration smoke

End-to-end regression gate closing M8 T8.10 on top of T8.5 / T8.7 on the
canonical T6 tungsten SNAP fixture (`W_2940_2017_2.snap`, Wood & Thompson
2017; 1024-atom BCC W; single-rank Fp64Reference NVE; 10 steps).

**Gate:** `|E_total(step=10) − E_total(step=0)| / |E_total(step=0)| ≤ 1e-7`
(D-M8-8 NVE-drift envelope — see *Gate derivation* below.)

**Status:** shipped M8 T8.10 (2026-04-21).

## What this smoke covers

- Fixture resolution: `W_2940_2017_2.snapcoeff` / `.snapparam` arrive via
  the M1-landed LAMMPS submodule (`verify/third_party/lammps/`);
  `setup_1024.data` ships via Git LFS. Both absences are handled cleanly
  (submodule absent → exit 77; LFS pointer unresolved → exit 2 with fix
  hint).
- SNAP path end-to-end: bispectrum → energy → force on GPU Fp64Reference
  for the canonical T6 1024-atom W BCC fixture (8×8×8 conventional BCC
  unit cells × 2 atoms/cell).
- Integrator determinism: 10-step NVE energy drift at VV's O((ω·dt)²)
  truncation-error floor — headroom check under the D-M8-8 NVE-drift
  envelope (1e-7 relative drift over 10 steps; see *Gate derivation*).

## What this smoke does NOT cover

Orthogonal gates exercised by sibling infrastructure, not re-run here:

- Byte-exact TDMD ≡ LAMMPS differential — exercised by T8.5
  `verify/t6/test_t6_differential.cpp` on the 250-atom `config_metal.yaml`
  fixture (CPU FP64 oracle gate, D-M8-7).
- GPU FP64 ≡ CPU FP64 byte-exact — exercised by T8.7
  `tests/gpu/test_snap_gpu_bit_exact.cpp` on a 2000-atom rattled fixture
  (D-M6-7 extension to SNAP).
- MixedFast / MixedFastSnapOnly threshold gate (D-M8-8) — exercised by
  T8.9 + T8.12 slow-tier sweep.
- Multi-rank Pattern 2 K=1 byte-exact on SNAP — exercised by the
  `verify/benchmarks/t6_snap_tungsten/scout_rtx5080/` scout (D-M7-10 chain
  extension, 2026-04-21).

## Running

This is a local-only pre-push gate (Option A — no public-runner GPU CI).
Invoked directly from the command line:

```bash
# Default — tdmd on $PATH, auto-skip if no GPU:
./tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh

# Explicit tdmd binary path:
./tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh --tdmd build/src/cli/tdmd

# Keep the workdir on success (for debugging thermo / stderr):
./tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh --keep-workdir

# Bump the budget on slow hardware:
TDMD_M8_SMOKE_BUDGET_SEC=120 ./tests/integration/m8_smoke_t6/run_m8_smoke_t6.sh
```

**Binary requirements:** Fp64ReferenceBuild (`build/`) with
`-DTDMD_BUILD_CUDA=ON`. The smoke uses `runtime.backend: gpu` so a CUDA
device MUST be visible (skipped with exit 0 otherwise).

**Fixture requirements:**
- LAMMPS submodule initialized
  (`git submodule update --init --recursive`).
- `setup_1024.data` resolved via Git LFS (`git lfs pull`). If it is
  missing, regenerate deterministically:
  ```bash
  cd verify/benchmarks/t6_snap_tungsten
  python3 generate_setup.py --nrep 8 --out setup_1024.data
  ```

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | Green — gate passed. Also: SKIPPED (no GPU visible). |
| 1    | Physics regression — energy drift exceeds 1e-12. Bisect before relaxing. |
| 2    | Infra — missing binary, LFS pointer unresolved, thermo malformed. |
| 3    | Perf — smoke exceeded wall-time budget (default 60 s). |
| 77   | SKIP — LAMMPS submodule not initialized (fixture absent). Matches the Catch2 `SKIP_RETURN_CODE` convention used by `test_lammps_oracle_snap_fixture`. |

## Gate derivation

The M8 exec pack §T8.10 originally specified `≤ 1e-12` but that envelope
conflates two different phenomena:

1. **FP64 accumulation floor** on a single *static* force evaluation —
   dominated by `k_max · ε_FP64 ~ 1e-14` for SNAP bispectrum. Relevant
   to D-M8-7 byte-exact oracle gate (exercised by T8.5 on 250 atoms).
2. **NVE energy drift over integrated steps** — dominated by
   velocity-Verlet's O((ω·dt)²) *local truncation* error. For W at
   `dt = 0.5 fs`, `ω_D ≈ 50 rad/ps`, `(ω·dt)² ~ 6e-4` per step; after
   phasing the relative drift lands in the `1e-9` … `1e-5` range, well
   above any FP64 floor.

This smoke measures (2), so the gate is anchored to the D-M8-8 NVE-drift
threshold registered in `verify/thresholds/thresholds.yaml`
(`gpu_mixed_fast_snap_only.snap.nve_drift_per_1000_steps = 1e-5`).
Linearly scaled to a 10-step run: **1e-7 relative drift**. On bring-up
(2026-04-21, RTX 5080, Fp64Reference) the measured drift was `2.5e-9`
— ≈40× headroom, enough to catch regressions without flapping.

An exec-pack SPEC delta at T8.10 session-report time amends line 1081
to state the 1e-7 envelope explicitly.

## If the gate trips

A drift exceeding 1e-7 on the T6 1024-atom 10-step NVE run points to:

- Changed reduction layout in `src/potentials/snap/*` (per-atom vs
  per-bond vs per-(atom,jju) grouping — all must reduce deterministically).
- Integrator regression in `src/integrator/velocity_verlet_gpu.cu` (e.g.,
  reordered half-kick).
- Neighbor list ordering regression in `src/neighbor/cell_grid.cu`
  (stable-sort requirement for Fp64Reference).

Bisect procedure: `git bisect` between the last green smoke and HEAD, with
this script as the inverted-exit predicate. Do NOT silently relax the
gate — master spec §D.15 red-flag protocol applies.

## Cross-references

- Master spec §14 M8 — milestone gate.
- `docs/specs/verify/SPEC.md` §4.7 — T6 canonical fixture choice.
- `docs/development/m8_execution_pack.md` §T8.10 — task specification.
- `verify/benchmarks/t6_snap_tungsten/README.md` — benchmark description.
- `verify/benchmarks/t6_snap_tungsten/config.yaml.template` — the config
  this smoke instantiates.
- `verify/thresholds/thresholds.yaml` `benchmarks.t6_snap_tungsten.*` —
  threshold registry.
