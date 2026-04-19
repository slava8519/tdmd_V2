# T3-gpu anchor-test — Acceptance criteria

<!-- markdownlint-disable MD013 -->

T3-gpu is consumed by the M6 milestone gate (`T6.13`). It extends the M5
T3 anchor-test into the GPU era: the shared `anchor_test_runner` harness
dispatches on `checks.yaml::backend`, with the GPU path injecting
`runtime.backend: gpu` into a tmp copy of the fixture config before launch.

The T6.10a milestone shipped **gates (1) + (2) + harness dispatch + mocked
pytest coverage**. Gate (3) was deferred to T6.10b behind two hard
dependencies.

**T7.12 update (2026-04-19)** — gate (3) reopened as
`status: active_eam_substitute` per D-M7-16 (M7 execution pack EAM-substitute
scope authorisation). The Morse-vs-dissertation comparison itself remains
deferred to M9+ (no GPU Morse kernel — `gpu/SPEC.md` §1.2); T7.12 measures
Pattern 2 GPU strong-scaling on the same Ni-Al EAM/alloy fixture used by
gates (1)+(2). Pattern 2 dispatch landed in T7.5 (HybridBackend) and
the SubdomainBoundaryDependency wiring in T7.7 — both prerequisites
satisfied. See §"Gate (3) — efficiency curve (T7.12)" below.

## Pass / fail pseudocode

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

# Gate (3) — Pattern 2 GPU strong-scaling efficiency probe (T7.12)
# Activated when checks.yaml::efficiency_curve.status == "active_eam_substitute".
# The runner injects zoning.subdomains: [N, 1, 1] per probe rank.
sps_anchor = run_tdmd(config.yaml, backend=gpu, subdomains=[1,1,1]).steps_per_sec
for n in efficiency_curve.ranks_to_probe[1:]:
    sps_n = run_tdmd(config.yaml, backend=gpu, subdomains=[n,1,1]).steps_per_sec
    eff_pct = 100.0 * sps_n * 1 / (sps_anchor * n)
    fail_if eff_pct < efficiency_curve.efficiency_floor_pct  # default 80% per D-M7-8
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

## Gate (3) — efficiency curve (T7.12)

**Status:** `active_eam_substitute` (T7.12 reopened from `deferred`).

**What it measures.** Pattern 2 GPU strong-scaling on the Ni-Al EAM/alloy
864-atom fixture, with `zoning.subdomains: [N, 1, 1]` injected per probe
rank. Each rank-count produces one `GpuGateResult` with
`gate_name = "efficiency_curve_N{NN}"` carrying:

| Field                       | Meaning                                                              |
|-----------------------------|----------------------------------------------------------------------|
| `n_procs`                   | Rank count                                                           |
| `measured_steps_per_sec`    | Derived from telemetry `total_wall_sec` and `run.n_steps`            |
| `measured_efficiency_pct`   | `100 * sps(N) * anchor_n / (sps_anchor * N)`; anchor ≡ 100%          |
| `floor_pct`                 | Threshold from `checks.yaml::efficiency_curve.efficiency_floor_pct`  |
| `passed`                    | True iff `measured_efficiency_pct >= floor_pct`                      |
| `detail`                    | Human-readable summary including the EAM-substitute provenance tag   |

**What it does NOT measure.** Morse-vs-dissertation comparison. Andreev
fig 29/30 used Morse, and we use Ni-Al EAM/alloy here — the absolute
numbers are not expected to track Andreev's published data and are not
graded against it. The substitute exercises the same Pattern 2 scheduler
/ comm / zoning code path that Morse would exercise, with the upside of
M6-blessed kernels (T6.5 EAM, T6.6 VV) and known thresholds.

**Floor rationale.** Default `efficiency_floor_pct: 80.0` matches D-M7-8
single-node target and `verify/benchmarks/t7_mixed_scaling/checks.yaml`
single-node gate (T7.11 parity).

**When the Morse fidelity comparison reopens.** The two original
dependencies still apply:

1. **Morse GPU kernel.** `gpu/SPEC.md` §1.2 defers all non-EAM pair styles
   to M9+. Until then the dissertation Morse curve is unmeasurable on GPU.
2. **Pattern 2 GPU scheduler dispatch.** Landed in T7.5 (HybridBackend) +
   T7.7 (SubdomainBoundaryDependency). This dependency is now satisfied;
   the EAM substitute exercises it.

When the M9+ Morse GPU kernel lands, the fixture path is:

| File (M9+)                               | Role                                            |
|------------------------------------------|-------------------------------------------------|
| `dissertation_reference_data.csv`        | Re-add with Morse Andreev fig 29/30 extraction  |
| `hardware_normalization_gpu.py`          | Replace stub with real CUDA EAM density probe   |
| `config.yaml` (parallel, not replacement)| Add `config_morse.yaml` for the dissertation arm|
| `checks.yaml`                            | Add `morse_dissertation_comparison:` block      |

The EAM substitute does not need to retire — both arms can coexist.

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
   emits a reminder note that the gpu_flops_ratio is unity (no real
   FLOPs measurement until the M9+ Morse GPU kernel lands). Surfaces in
   the harness log but does not flip overall_passed.

6. **`EFFICIENCY_BELOW_FLOOR`** (T7.12) — gate (3) trips. One or more
   probe ranks measured efficiency below `efficiency_floor_pct`. Fix
   pattern matches `verify/benchmarks/t7_mixed_scaling/`: bisect against
   the last known-green commit; if the regression is in the scheduler /
   zoning / comm code paths the same probe in t7_mixed_scaling will
   typically fail too. If T3-gpu fails but t7_mixed_scaling passes, the
   regression is fixture-specific (864 atoms × Ni-Al EAM at K=1).

## Ship criteria

### T6.10a (shipped)

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

### T7.12 (shipped — gate (3) EAM substitute activation)

- [x] `checks.yaml::efficiency_curve.status` flipped from `deferred` to
      `active_eam_substitute`; `morse_fidelity_blocker` provenance string
      retained; `efficiency_floor_pct: 80.0` matches D-M7-8 / T7.11 parity.
- [x] `_run_gpu_two_level()` extended with `_run_gpu_efficiency_probe()`
      branch; `_write_augmented_config()` accepts `subdomains_xyz=[N,1,1]`
      kwarg; `_launch_tdmd_with_backend()` forwards it.
- [x] `GpuGateResult` extended with `n_procs / measured_steps_per_sec /
      measured_efficiency_pct / floor_pct` (all default-None for gates 1/2).
- [x] Mocked pytest coverage: 8 new `GpuEfficiencyProbeTest` cases + 3
      `WriteAugmentedConfigSubdomainsTest` cases (29/29 unittests green).
- [x] `gpu/SPEC.md` §11.4 + §16 change log v1.0.14 entry.

The real CPU ↔ GPU byte-exact gate + Pattern 2 efficiency probe both
run locally pre-push as the M6/M7 acceptance check. No self-hosted CI
runner (D-M6-6 — Option A policy per `project_option_a_ci.md`).
