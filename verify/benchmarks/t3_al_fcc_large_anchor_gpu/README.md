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
to GPU is blocked on **two unshipped milestones**:

1. **Morse GPU kernel** — `gpu/SPEC.md` §1.2 defers all non-EAM pair styles
   (LJ, Morse, MEAM, SNAP, PACE, MLIAP) to M9+. M6 GPU scope is `{NL, EAM,
   VV}`, exactly what shipped in T6.4 / T6.5 / T6.6.
2. **Pattern 2 GPU scheduler dispatch** — multi-rank efficiency curves
   require the scheduler to drive GPU compute across subdomains. That
   lands with T6.9b / Pattern 2 GPU in M7 (see `gpu/SPEC.md` §9.5).

Rather than ship a placeholder fixture that would stay red until M9, T6.10a
anchors the harness on **what M6 actually delivers**: single-rank GPU
execution of the already-landed `{NL, EAM, VV}` stack on a smaller Ni-Al
EAM/alloy fixture (the same one T4 and T6.7 gate against). Gates (1) and
(2) are reached — those are the mandatory invariants. Gate (3) (efficiency
curve vs dissertation) is deferred to T6.10b behind the two dependencies
above.

See `acceptance_criteria.md` §"Deferred gates" for the exact deferred scope
and the dependency graph.

## Experiment (T6.10a scope)

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
| Ranks probed          | 1 (single-rank; multi-rank gate deferred to T6.10b)       |
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
| `hardware_normalization_gpu.py`     | GPU probe (M6 stub; full CUDA EAM micro-kernel — T6.10b)   |
| `acceptance_criteria.md`            | Pass/fail rules + deferred gate documentation              |

`dissertation_reference_data.csv` is **deliberately absent** in T6.10a. The
efficiency-curve gate (gate 3) is deferred to T6.10b; without the efficiency
comparison, no dissertation-reference points are consumed. `acceptance_criteria.md`
tracks the planned CSV layout (same schema as T3 CPU) for T6.10b's
reintroduction.

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

T6.10a delivers the infrastructure + gates (1) and (2). T6.10b reintroduces
the efficiency-curve gate after Morse GPU + Pattern 2 GPU land (M7 + M9).
The M6 merge gate is two-level, not three.
