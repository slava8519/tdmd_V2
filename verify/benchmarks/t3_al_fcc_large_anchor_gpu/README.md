# Benchmark T3-gpu — Ni-Al EAM/alloy GPU anchor (CPU ↔ GPU Reference bit-exact gate)

<!-- markdownlint-disable MD013 MD033 -->

**Tier:** slow (`verify/SPEC.md` §8.2 + `gpu/SPEC.md` §11.4).
**Purpose:** **structural existence proof** of TDMD's M6 GPU path — demonstrate
that the `Fp64ReferenceBuild + runtime.backend: gpu` path produces thermo
output bit-exact to the CPU Reference path over 100 steps, and that the
`MixedFastBuild` GPU path stays within D-M6-8 thresholds of GPU Reference,
on the same Ni-Al EAM/alloy fixture T4 uses. M6 milestone gate (`T6.13`)
consumes this harness output.

This directory is the **fixture** half of T3-gpu: config, checks, GPU
hardware probe stub, acceptance criteria. The driver (`anchor_test_runner`)
is shared with the CPU T3 anchor — `runner.py` dispatches on the `backend`
key in `checks.yaml`.

## Scope realism — why not Morse/LJ 10⁶?

The CPU T3 fixture (`t3_al_fcc_large_anchor/`) reproduces Andreev §3.5 on
**Al FCC 10⁶ atoms + Morse** at 8 Å cutoff — physics-equivalent to the
dissertation LJ at the scaling-profile level. Porting that fixture verbatim
to GPU is blocked on the **Morse GPU kernel** (gpu/SPEC.md §1.2 defers all
non-EAM pair styles to M9+).

**T6.10a (shipped v1.0.8) → T7.12 (shipped v1.0.14) progression.** T6.10a
anchored the harness on what M6 actually delivers: single-rank GPU execution
of the already-landed `{NL, EAM, VV}` stack on a smaller Ni-Al EAM/alloy
fixture (the same one T4 and T6.7 gate against). T7.12 reopened gate (3)
as the **EAM-substitute Pattern 2 strong-scaling probe** — the second
historic blocker (Pattern 2 GPU scheduler dispatch — was T6.9b) closed via
T7.5 HybridBackend + T7.7 SubdomainBoundaryDependency. The remaining
Morse-vs-dissertation comparison stays deferred to M9+.

| Gate                                  | Status      | Shipped in |
|---------------------------------------|-------------|------------|
| (1) CPU ≡ GPU Reference byte-exact    | **active**  | T6.10a     |
| (2) MixedFast within D-M6-8 thresholds| **active**  | T6.10a     |
| (3) Pattern 2 EAM-substitute efficiency probe | **active** | T7.12      |
| (3') Morse-vs-dissertation comparison | deferred    | M9+ (Morse GPU kernel) |

See `acceptance_criteria.md` §"Gate (3) — efficiency curve (T7.12)" for
the gate-3 contract and what the M9+ Morse arm would add when its kernel
lands.

## Experiment (T6.10a + T7.12 scope)

| Quantity              | Value                                                     |
|-----------------------|-----------------------------------------------------------|
| System                | Ni-Al B2 (50:50) — same as T4 `t4_nial_alloy`             |
| Lattice               | FCC, 6×6×6 unit cells                                     |
| Size                  | 864 atoms                                                 |
| Potential             | EAM/alloy (Mishin 2004 NiAl)                              |
| cutoff                | from `.eam.alloy` file (6.287 Å)                          |
| Initial velocities    | 300 K, seed 12345 (T4 golden state)                       |
| Integrator            | velocity-Verlet (NVE)                                     |
| Timestep              | 0.001 ps (1 fs)                                           |
| Steps                 | 100 (matches T6.7 gate — amortises JIT + warm-up)         |
| Thermo output period  | every step (byte-compare the full trace)                  |
| Ranks probed (gate 1+2)| 1 — CPU ≡ GPU Reference byte-exact thermo gate           |
| Ranks probed (gate 3) | [1, 2] — Pattern 2 EAM-substitute efficiency probe (T7.12) |
| Backend               | `runtime.backend: gpu` (Fp64Reference + MixedFast variants) |

`setup.data` is reused from `../t4_nial_alloy/setup.data` — symlinked at
fixture level so the harness does not need its own regeneration script.
The Mishin 2004 EAM file lives at `../../third_party/potentials/NiAl_Mishin_2004.eam.alloy`
(committed, ~200 KB).

## Files in this directory

| File                                | Role                                                       |
|-------------------------------------|------------------------------------------------------------|
| `README.md`                         | This document                                              |
| `config.yaml`                       | TDMD config (EAM/alloy Ni-Al + `runtime.backend: gpu`)     |
| `checks.yaml`                       | Acceptance threshold declarations + backend dispatch key   |
| `hardware_normalization_gpu.py`     | GPU probe (stub; replaced when M9+ Morse arm reopens)      |
| `acceptance_criteria.md`            | Pass/fail rules + gate (3) T7.12 contract                  |

`dissertation_reference_data.csv` is **deliberately absent**. T7.12 grades
gate (3) against an absolute `efficiency_floor_pct: 80.0` (D-M7-8 / T7.11
parity), not against the Andreev Morse curve — see "scope realism" above.
The CSV would be added back when the M9+ Morse GPU kernel lands and
unlocks the literal Morse-vs-dissertation arm; layout would mirror the
CPU T3 schema. `acceptance_criteria.md` documents the M9+ fixture path.

## Harness invocation

```bash
# Local pre-push (requires CUDA device visible):
python -m verify.harness.anchor_test_runner \
    --benchmark-dir verify/benchmarks/t3_al_fcc_large_anchor_gpu \
    --tdmd-bin build/tdmd \
    --ranks 1

# Force both backend variants sequentially (Reference + MixedFast):
python -m verify.harness.anchor_test_runner \
    --benchmark-dir verify/benchmarks/t3_al_fcc_large_anchor_gpu \
    --gpu-flavors both
```

`checks.yaml::backend: gpu` tells the runner to inject `runtime.backend: gpu`
into the config when launching TDMD. On CPU-only machines (no CUDA device)
the runner emits `STATUS_RED` with `failure_mode = NO_CUDA_DEVICE` —
identical handling pattern to the CPU T3's `HARDWARE_MISMATCH`.

## Scope reminder

T6.10a (v1.0.8) delivered the infrastructure + gates (1) and (2). T7.12
(v1.0.14) reopened gate (3) as the Pattern 2 EAM-substitute efficiency
probe. The literal Morse-vs-dissertation comparison stays deferred to
M9+ (Morse GPU kernel). Effective gate count: **three active** (T7.12
onward), not two.
