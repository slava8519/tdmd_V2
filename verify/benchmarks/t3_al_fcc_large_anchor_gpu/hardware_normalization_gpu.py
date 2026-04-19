#!/usr/bin/env python3
"""T3-gpu anchor-test — GPU hardware normalisation helper (M6 stub).

T6.10a scope: emit a deterministic JSON payload so the harness has a
stable schema + callable entry point to consume. The real GPU FLOPs probe
(CUDA EAM density micro-kernel) is **T6.10b work** — blocked on the same
dependency graph as the efficiency-curve gate (Morse GPU M9+ and Pattern 2
GPU M7). Until those land, the efficiency-curve gate is deferred and the
`gpu_flops_ratio` scalar is unused downstream.

See `checks.yaml::efficiency_curve.status: deferred` for the broader
context.

Why ship a stub now? Two reasons:
  1. The runner's `hardware_probe.py` needs a callable script path to
     round-trip through the T3 CPU code path without special-casing
     the GPU branch. A stub that emits valid JSON lets the shared code
     path stay symmetric.
  2. `nvidia-smi` reporting of the active GPU model is reportable in the
     M6 smoke report (T6.13) — useful provenance even when the flops
     scalar is unused.

Usage:
    python3 hardware_normalization_gpu.py           # prints scalar to stdout
    python3 hardware_normalization_gpu.py --json    # JSON object on stdout

Exit codes:
    0 — success (always, even on probe failures — stub never hard-fails)
    1 — argument parse error
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any


# M6 placeholder. T6.10b replaces this with a real CUDA EAM density-kernel
# probe run, divided by a measured sm_XX baseline (likely sm_80 Ampere
# reference — dissertation Alpha cluster had no GPU, so any baseline is
# hypothetical). Until then, the scalar stays unused by the harness.
STUB_GPU_FLOPS_RATIO = 1.0
STUB_BASELINE_NOTE = (
    "T6.10a stub — efficiency-curve gate is deferred (T6.10b); "
    "scalar unused until Morse GPU + Pattern 2 GPU dispatch land"
)


def probe_gpu_model() -> str | None:
    """Return the first visible CUDA device's name via `nvidia-smi`, or None.

    Uses `nvidia-smi --query-gpu=name --format=csv,noheader` — the standard
    non-interactive probe. Returns None (not throws) if nvidia-smi is
    absent, fails, or emits empty output — the stub does not hard-fail.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.strip().splitlines()
    return first_line[0].strip() if first_line else None


def gpu_flops_ratio() -> float:
    """T6.10a stub — returns the fixed placeholder ratio."""
    return STUB_GPU_FLOPS_RATIO


def emit_payload(local: float, baseline_note: str) -> dict[str, Any]:
    return {
        # Match the CPU hardware_normalization.py schema so the harness
        # can consume both through a single code path (just swap keys
        # when `backend: gpu`).
        "ghz_flops_ratio": local,  # alias — harness still reads this key
        "gpu_flops_ratio": local,  # explicit name for GPU consumers
        "baseline_gflops": 0.0,  # not applicable to GPU stub
        "local_gflops": 0.0,  # would be populated by T6.10b CUDA kernel
        "gpu_model": probe_gpu_model(),
        "note": baseline_note,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the GPU ↔ dissertation hypothetical baseline ratio "
            "(T6.10a stub — real probe ships in T6.10b)."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON object instead of a plain scalar",
    )
    # Accept --baseline-gflops for CLI compatibility with CPU probe, even
    # though the GPU stub ignores it.
    parser.add_argument(
        "--baseline-gflops",
        type=float,
        default=0.0,
        help="accepted for symmetry with CPU probe; ignored by GPU stub",
    )
    args = parser.parse_args(argv)

    ratio = gpu_flops_ratio()
    payload = emit_payload(local=ratio, baseline_note=STUB_BASELINE_NOTE)
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"{ratio:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
