# T3-gpu anchor-test — Acceptance criteria

<!-- markdownlint-disable MD013 -->

T3-gpu is consumed by the M6 milestone gate (`T6.13`). It extends the M5
T3 anchor-test into the GPU era: the shared `anchor_test_runner` harness
dispatches on `checks.yaml::backend`, with the GPU path injecting
`runtime.backend: gpu` into a tmp copy of the fixture config before launch.

The T6.10a milestone ships **gates (1) + (2) + harness dispatch + mocked
pytest coverage**. Gate (3) (efficiency curve vs dissertation) is deferred
to T6.10b behind the two documented dependencies.

## Pass / fail pseudocode (T6.10a scope)

```
# Gate (1) — CPU ≡ GPU Reference byte-exact thermo
cpu_thermo = run_tdmd(config.yaml, backend=cpu)          # 100 steps
gpu_thermo = run_tdmd(config.yaml, backend=gpu)          # 100 steps
fail_if cpu_thermo_bytes != gpu_thermo_bytes

# Gate (2) — GPU MixedFast within D-M6-8 thresholds of GPU Reference
# Delegated to T6.8a differential harness. The anchor runner asserts
# on exit code, not the raw numbers, to avoid duplicating FP-compare
# logic at the Python layer.
exit_code = invoke_t6_8a_differential()
fail_if exit_code != 0
```

Both gates run on the same 864-atom Ni-Al EAM/alloy fixture as T4 and T6.7.
Wall-clock budget: < 10 minutes on a dev GPU (single-rank 100 steps is
sub-minute; budget pads for cold-start + probe overhead).

### Status tri-state (same as CPU T3)

| Level   | Meaning                                                                     |
|---------|-----------------------------------------------------------------------------|
| `GREEN` | All `fail_if` clauses clean, no warnings.                                   |
| `YELLOW`| Gate (1) + (2) clean; advisory warning (e.g. stub GPU probe note).          |
| `RED`   | ≥ 1 `fail_if` hit. M6 merge blocked. Triage per §"Failure modes".           |

### CPU-only machine handling

If the runner is invoked with `backend: gpu` on a machine without a visible
CUDA device, the harness emits `STATUS_RED` with
`failure_mode = NO_CUDA_DEVICE`. This parallels the CPU T3's
`HARDWARE_MISMATCH` hard-fail pattern. Local pre-push is the designated
run surface (D-M6-6 — CI has no GPU runner).

## Deferred gates (T6.10b scope)

Gate (3) — **GPU efficiency curve vs Andreev dissertation** — is not in
T6.10a. Two hard dependencies:

1. **Morse GPU kernel.** `gpu/SPEC.md` §1.2 defers all non-EAM pair styles
   to M9+. The dissertation fixture uses Morse (physics-equivalent to
   Andreev's LJ at the scaling-profile level); swapping to EAM would
   produce a different strong-scaling curve with no published baseline
   to compare against.
2. **Pattern 2 GPU scheduler dispatch.** The efficiency curve is a
   multi-rank strong-scaling measurement. Single-rank GPU execution
   cannot probe the TD-overlap efficiency profile the dissertation
   reports. Pattern 2 GPU lands in M7 (T6.9b depends on it — see
   `gpu/SPEC.md` §9.5).

T6.10b unblocks when both land. The fixture layout already provisions the
future additions:

| File (T6.10b)                            | Role                                         |
|------------------------------------------|----------------------------------------------|
| `dissertation_reference_data.csv`        | Reintroduced — same schema as CPU T3         |
| `hardware_normalization_gpu.py`          | Replaced — real CUDA EAM density micro-kernel |
| `config.yaml` (modified)                 | Swap to Morse once GPU kernel lands          |
| `checks.yaml::efficiency_curve.status`   | Flip `deferred` → activated                  |

See `checks.yaml` for the planned (currently inactive) schema.

## Failure modes + escalation path

Classifications reported in the JSON report when `overall_status == RED`:

1. **`NO_CUDA_DEVICE`** — runner invoked with `backend: gpu` but no visible
   CUDA device. Fix: run on a GPU-equipped machine, or skip via
   `backend: cpu` override. Not a regression in the code path itself.

2. **`CPU_GPU_REFERENCE_DIVERGE`** — gate (1) trips. CPU and GPU Reference
   thermo streams differ at any byte. Fix: bisect the `src/gpu/` diff
   against the last known-green commit. The D-M6-7 invariant is foundational
   — any drift is a real bug. Cross-reference T6.5 EAM ≤1e-12 gate and
   T6.6 VV byte-equal gate: if either of those is green but the engine
   thermo diverges, the bug is in the engine wiring (T6.7 code path) or
   in the stream ordering (T6.9a dual-stream exposure).

3. **`MIXED_FAST_OVER_BUDGET`** — gate (2) trips. MixedFast exceeds the
   T6.8a achieved thresholds (force 1e-5 rel, PE 1e-7 rel). Fix: bisect
   against the last known-green MixedFast commit. If the threshold was
   intentionally tightened in checks.yaml, double-check against the T6.8a
   closed-form value; D-M6-8 target 1e-6 is still a stretch goal (T6.8b).

4. **`RUNTIME_BUDGET_BLOWOUT`** — 10-minute budget exceeded. Fix: profile
   the slow path under NVTX (T6.11 lands this); usually indicates GPU
   cold-start or memory pool warm-up. If the slow phase is MPI init,
   the fixture is still running the 1-rank comm backend and the blowout
   is infra, not physics.

5. **`STUB_PROBE_WARNING`** (YELLOW, not RED) — the GPU probe stub always
   emits a reminder note that the efficiency-curve gate is deferred. This
   surfaces in the harness log but does not flip overall_passed.

## Ship criteria for T6.10a

This document is fixture-only — T6.10a passes when:

- [x] All five fixture files (README, config.yaml, checks.yaml,
      hardware_normalization_gpu.py, acceptance_criteria.md) exist and
      are well-formed.
- [x] `hardware_normalization_gpu.py` runs to completion and emits a
      numeric scalar on stdout (exits 0, valid JSON with `--json`).
- [x] `checks.yaml::backend: gpu` parses in the runner (extension wired
      in `runner.py`).
- [x] Mocked pytest coverage in
      `verify/harness/anchor_test_runner/test_anchor_runner.py` exercises
      the GPU dispatch path.
- [x] `gpu/SPEC.md` §11.4 cross-references this directory.

The real CPU ↔ GPU byte-exact gate runs locally pre-push as the M6
acceptance check. No self-hosted CI runner (D-M6-6 — Option A policy
per `project_option_a_ci.md`).
