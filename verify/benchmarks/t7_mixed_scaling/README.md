# Benchmark T7 — Pattern 2 mixed-scaling (Ni-Al EAM)

<!-- markdownlint-disable MD013 MD033 -->

**Tier:** slow (`verify/SPEC.md` §8.2 + master spec §14 M7).
**Purpose:** **strong-scaling probe** of TDMD's M7 Pattern 2 stack —
demonstrate that splitting a Ni-Al EAM ~10⁵-atom system across `N`
GPU subdomains (`zoning.subdomains: [N,1,1]`) achieves the
dissertation-grade efficiency targets:

- **single-node, ≥80%** through `N ∈ {2, 4, 8}`,
- **2-node, ≥70%** at 2 nodes × 8 GPU = 16 ranks
  (honorable best-effort per D-M7-8 — no permanent dev infra).

This is the per-PR regression guard for Pattern 2 scaling once it
ships in M7. Multi-node is **opportunistic** — local pre-push runs
the 1-node curve; cloud-burst sessions probe the 2-node tier.

## Why EAM and not Morse / dissertation LJ?

The dissertation Al-FCC 10⁶ Morse fixture (`t3_al_fcc_large_anchor`)
remains the reproducibility anchor for the **CPU** path. M7 cannot use
it on GPU because:

1. **Morse GPU kernel is M9+** (`gpu/SPEC.md` §1.2). The M6/M7 GPU
   compute path covers `{NL, EAM, VV}` only.
2. The same Mishin 2004 Ni-Al EAM/alloy fixture used by T4, T6.7, and
   T6.13 is the path of least surprise for the M7 milestone — no new
   physics surface area, just a larger atom count.

T7 measures **scaling, not absolute fidelity**. The dissertation's
absolute MD-step/sec numbers are hardware-locked (2007 Harpertown
cluster); the T3 anchor handles that comparison on CPU. T7 is a
pure efficiency probe: how well does `t_step(N) × N` track `t_step(1)`?

## Experiment

| Quantity              | Value                                                     |
|-----------------------|-----------------------------------------------------------|
| System                | Ni-Al B2 (50:50) — same potential as T4                   |
| Lattice               | FCC, 32×32×32 unit cells                                  |
| Size                  | 131,072 atoms (~1.3×10⁵)                                  |
| Potential             | EAM/alloy (Mishin 2004 NiAl)                              |
| cutoff                | from `.eam.alloy` file (6.287 Å)                          |
| Initial velocities    | 300 K, seed 12345 (T4 generator chain)                    |
| Integrator            | velocity-Verlet (NVE)                                     |
| Timestep              | 0.001 ps (1 fs)                                           |
| Steps                 | 100 (amortises kernel JIT + warm-up)                      |
| Thermo period         | every step (full trace for telemetry capture)             |
| Ranks probed          | 1, 2, 4, 8 single-node; 16 if 2-node burst available      |
| Backend               | `runtime.backend: gpu`, `comm.backend: hybrid` (T7.5)     |
| Subdomain layout      | `[N, 1, 1]` — harness injects per rank-count              |

`setup.data` is **not committed** — `generate_setup.py` produces it
deterministically from the T4 generator (same seed, same algorithm,
just larger nx/ny/nz). At ~7.5 MB for 131k atoms, lazy regeneration
keeps the repo git-LFS-free.

## Files in this directory

| File                                | Role                                                       |
|-------------------------------------|------------------------------------------------------------|
| `README.md`                         | This document                                              |
| `config.yaml`                       | TDMD config (EAM/alloy Ni-Al + Pattern 2 base)             |
| `checks.yaml`                       | Per-rank-count efficiency gates                            |
| `generate_setup.py`                 | Lazy regen — delegates to T4 generator with nx=ny=nz=32    |
| `hardware_normalization.py`         | PerfModel-based normalisation stub (T7.13 calibration)     |

## Harness invocation

```bash
# Local pre-push (single-node strong scaling, dev GPU):
python -m verify.harness.scaling_runner \
    --benchmark-dir verify/benchmarks/t7_mixed_scaling \
    --tdmd-bin build/tdmd \
    --ranks 1,2,4

# 2-node opportunistic (cloud burst):
python -m verify.harness.scaling_runner \
    --benchmark-dir verify/benchmarks/t7_mixed_scaling \
    --tdmd-bin build/tdmd \
    --ranks 1,2,4,8,16 \
    --node-counts 1,2
```

The harness writes augmented configs (with `zoning.subdomains: [N,1,1]`
and absolute paths) into a workdir and launches `mpirun -np N tdmd run`
per probe point. Telemetry JSONL is parsed for `total_wall_sec`; rate
is `n_steps / total_wall_sec`; efficiency is `rate(N) / (rate(1) × N)`.

## Pattern 1 regression guard

`subdomains: [1, 1, 1]` (the N=1 anchor case) routes through the
single-subdomain path in `SimulationEngine` (T7.9 wire) and produces
**byte-exact** thermo equal to the M6 T6.13 baseline on the same
131k-atom config. The harness asserts this when `1 ∈ ranks_to_probe`
and a baseline thermo file is provided via `--baseline-thermo`. This
keeps the T7 probe a pure-additional path — Pattern 1 byte-exactness
is preserved across every N=1 invocation.

## Scope reminder

- **Inter-node NCCL** is M8+; T7's 2-node tier uses `HybridBackend`
  (T7.5) which composes intra-node NCCL with inter-node GpuAwareMPI.
- **Morse fidelity** is M9+. T7 ships an EAM substitute per D-M7-16.
- **CI automation** of multi-node is excluded (Option A — see
  memory `project_option_a_ci.md`). Local pre-push only.
