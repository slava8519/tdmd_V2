"""scaling_runner — strong-scaling probe driver for the T7 benchmark.

Public API:
    * :class:`RunnerConfig`       — paths + knobs (CLI-constructed)
    * :class:`ScalingProbePoint`  — per-N measurement record
    * :class:`ScalingReport`      — aggregate report (green / yellow / red)
    * :class:`ScalingRunner`      — orchestrator (single entry: ``run``)

The harness pattern mirrors ``anchor_test_runner`` so the two stay code-
review-symmetric: load checks.yaml, enumerate ranks_to_probe, write
augmented configs into a workdir, launch ``mpirun -np N tdmd run``,
parse telemetry, compute strong-scaling efficiency, gate.

T7 scope: GPU Pattern 2, single-node strong-scaling 1→8 GPU mandatory,
2-node opportunistic. CPU-only / non-GPU paths are out of scope (the CPU
T3 anchor handles that).
"""

from .runner import (
    RunnerConfig,
    ScalingProbePoint,
    ScalingReport,
    ScalingRunner,
    STATUS_GREEN,
    STATUS_RED,
    STATUS_YELLOW,
)

__all__ = [
    "RunnerConfig",
    "ScalingProbePoint",
    "ScalingReport",
    "ScalingRunner",
    "STATUS_GREEN",
    "STATUS_RED",
    "STATUS_YELLOW",
]
