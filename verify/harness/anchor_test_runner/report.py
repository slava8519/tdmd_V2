"""Dataclasses + JSON emitter for the T3 anchor-test report.

The structure is deliberately flat + explicit so the M5 retrospective
(acceptance_criteria.md §"Failure modes") can cross-reference every
``status`` / ``failure_mode`` field without re-running the harness.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any


# ---------------------------------------------------------------------------
# Status tri-state (matches acceptance_criteria.md table)
# ---------------------------------------------------------------------------

STATUS_GREEN = "GREEN"
STATUS_YELLOW = "YELLOW"
STATUS_RED = "RED"


@dataclasses.dataclass
class HardwareProbeResult:
    """One-shot current-hardware FLOPs measurement (cached 24h).

    ``ghz_flops_ratio`` is the dimensionless scalar the runner multiplies
    into each dissertation reference point before the absolute-perf diff.
    Must be ≥ 1.0; below 1.0 means the current machine is slower than the
    2007 Harpertown baseline and the comparison is undefined — runner
    hard-fails with ``HARDWARE_MISMATCH``.
    """

    local_gflops: float
    baseline_gflops: float
    ghz_flops_ratio: float
    probe_timestamp_utc: str
    cached: bool  # True iff loaded from ``~/.cache/tdmd/hardware_flops.json``

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class GpuGateResult:
    """One gate outcome in the T3-gpu two-level harness.

    Originally T6.10a shipped two gates: ``cpu_gpu_reference_bit_exact`` and
    ``mixed_fast_vs_reference``. T7.12 added the EAM-substitute Pattern 2
    efficiency probe — emits one ``GpuGateResult`` per probed rank with
    ``gate_name = f"efficiency_curve_N{n:02d}"`` and the measurement fields
    populated. Gates 1/2 leave the measurement fields as ``None``.
    """

    gate_name: str  # stable identifier — matches checks.yaml key
    passed: bool
    status: str  # STATUS_GREEN / STATUS_YELLOW / STATUS_RED
    detail: str  # human-readable summary ("thermo 4210 bytes ≡ 4210 bytes")
    cpu_thermo_path: str | None = None  # optional — populated for gate 1
    gpu_thermo_path: str | None = None  # optional — populated for gate 1
    # T7.12 — populated only for efficiency_curve_N* gates.
    n_procs: int | None = None
    measured_steps_per_sec: float | None = None
    measured_efficiency_pct: float | None = None
    floor_pct: float | None = None

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class AnchorTestPoint:
    """One row of the report — one rank count, compared against one CSV row.

    ``efficiency_relative_error`` uses the dissertation value as the
    denominator (per checks.yaml ``dissertation_comparison.efficiency_relative``).
    ``absolute_performance_relative_error`` uses ``reference * hw_ratio``
    as the denominator — the hardware-normalised reference.
    """

    n_procs: int

    # Measured
    measured_performance_mdps: float  # mega-steps-per-day — TDMD steps * 86400/s
    measured_efficiency_pct: float
    measured_wall_seconds: float
    measured_n_steps: int

    # Dissertation reference (raw CSV value, pre-normalisation)
    reference_performance_mdps: float
    reference_efficiency_pct: float
    reference_source_figure: str
    reference_note: str

    # Comparison
    normalised_reference_performance_mdps: float  # reference * ghz_flops_ratio
    efficiency_relative_error: float
    absolute_performance_relative_error: float
    efficiency_tolerance: float
    absolute_performance_tolerance: float

    # Derived status
    efficiency_passed: bool  # primary gate — contributes to overall_passed
    absolute_performance_warned: bool  # secondary — YELLOW only
    status: str  # STATUS_GREEN / STATUS_YELLOW / STATUS_RED

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class AnchorTestReport:
    """Top-level structured report emitted by ``AnchorTestRunner.run()``.

    ``overall_passed == True`` iff every point's ``efficiency_passed`` is
    true AND no hardware hard-fail was tripped. YELLOW / absolute-perf
    warnings do not flip this bit; see ``any_warning``.
    """

    # Per-point detail
    points: list[AnchorTestPoint]

    # Aggregate status
    overall_passed: bool
    overall_status: str  # STATUS_GREEN / STATUS_YELLOW / STATUS_RED
    any_warning: bool

    # Provenance
    dissertation_reference_commit: str  # git SHA of the CSV revision
    tdmd_commit: str
    benchmark_directory: str
    checks_yaml_path: str
    hardware: HardwareProbeResult

    # Lifecycle
    report_timestamp_utc: str
    wall_clock_minutes: float

    # Human-readable log lines (one per normalisation or comparison step).
    normalization_log: list[str]
    # Optional failure-mode classification if ``overall_status == STATUS_RED``.
    # One of the seven CPU identifiers from acceptance_criteria.md
    # ("REF_DATA_STALE", "HARDWARE_NORMALIZATION_OFF", ..., "HARDWARE_MISMATCH")
    # or one of the GPU identifiers from t3_al_fcc_large_anchor_gpu
    # ("NO_CUDA_DEVICE", "CPU_GPU_REFERENCE_DIVERGE", "MIXED_FAST_OVER_BUDGET",
    # "RUNTIME_BUDGET_BLOWOUT", "EFFICIENCY_BELOW_FLOOR" — added T7.12).
    failure_mode: str | None = None
    # T6.10a — populated only for the T3-gpu two-level anchor path. CPU T3
    # runs leave this as ``None``. ``backend`` records which dispatch branch
    # produced the report ("cpu" or "gpu") so consumers can disambiguate
    # without introspecting the file system layout.
    backend: str = "cpu"
    gpu_gates: list[GpuGateResult] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "points": [p.to_json() for p in self.points],
            "overall_passed": self.overall_passed,
            "overall_status": self.overall_status,
            "any_warning": self.any_warning,
            "dissertation_reference_commit": self.dissertation_reference_commit,
            "tdmd_commit": self.tdmd_commit,
            "benchmark_directory": self.benchmark_directory,
            "checks_yaml_path": self.checks_yaml_path,
            "hardware": self.hardware.to_json(),
            "report_timestamp_utc": self.report_timestamp_utc,
            "wall_clock_minutes": self.wall_clock_minutes,
            "normalization_log": list(self.normalization_log),
            "failure_mode": self.failure_mode,
            "backend": self.backend,
            "gpu_gates": (
                [g.to_json() for g in self.gpu_gates] if self.gpu_gates else None
            ),
        }

    def write_json(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=False))

    # -----------------------------------------------------------------
    # Human summary
    # -----------------------------------------------------------------

    def format_console_summary(self) -> str:
        """Terse console summary — one line per point/gate + footer verdict."""
        lines: list[str] = []
        if self.backend == "gpu":
            gate_count = len(self.gpu_gates) if self.gpu_gates else 0
            lines.append(
                f"T3-gpu anchor — {gate_count} gate(s) (backend=gpu, T6.10a two-level)"
            )
            for g in self.gpu_gates or []:
                mark = {
                    STATUS_GREEN: "ok  ",
                    STATUS_YELLOW: "warn",
                    STATUS_RED: "FAIL",
                }.get(g.status, "??? ")
                lines.append(f"  [{mark}] {g.gate_name}: {g.detail}")
        else:
            lines.append(
                f"T3 anchor-test — {len(self.points)} point(s), "
                f"hw_ratio={self.hardware.ghz_flops_ratio:.3f}"
            )
            for p in self.points:
                mark = {
                    STATUS_GREEN: "ok  ",
                    STATUS_YELLOW: "warn",
                    STATUS_RED: "FAIL",
                }.get(p.status, "??? ")
                lines.append(
                    f"  [{mark}] N={p.n_procs:>3}  "
                    f"eff_measured={p.measured_efficiency_pct:6.2f}%  "
                    f"eff_ref={p.reference_efficiency_pct:6.2f}%  "
                    f"eff_rel_err={p.efficiency_relative_error * 100:5.2f}%  "
                    f"(tol={p.efficiency_tolerance * 100:.0f}%)  "
                    f"abs_rel_err={p.absolute_performance_relative_error * 100:5.2f}%"
                )
        footer_cpu = {
            STATUS_GREEN: "GREEN — all points within tolerance",
            STATUS_YELLOW: "YELLOW — efficiency passed; absolute-perf warning(s)",
            STATUS_RED: "RED — efficiency fail_if tripped",
        }
        footer_gpu = {
            STATUS_GREEN: "GREEN — CPU≡GPU bit-exact + MixedFast delegated green",
            STATUS_YELLOW: "YELLOW — gates passed; advisory warning(s) (e.g. delegated gate)",
            STATUS_RED: "RED — gate tripped; see gpu_gates detail",
        }
        footer_map = footer_gpu if self.backend == "gpu" else footer_cpu
        footer = footer_map.get(self.overall_status, "?")
        lines.append(f"  overall: {footer}")
        if self.failure_mode:
            lines.append(f"  failure_mode: {self.failure_mode}")
        return "\n".join(lines)
