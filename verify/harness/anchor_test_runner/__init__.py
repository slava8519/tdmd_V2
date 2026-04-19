"""T3 Anchor-test runner (T5.11) — dissertation reproduction harness.

See ``docs/specs/verify/SPEC.md`` §7.4 and ``m5_execution_pack.md`` T5.11.

The :class:`AnchorTestRunner` drives TDMD across a declared set of rank
counts, collects wall-clock telemetry, normalises against a one-shot
hardware FLOPs probe (cached 24h in ``~/.cache/tdmd/``), and compares
point-by-point against the Andreev §3.5 dissertation reference CSV.

Exit contract: ``runner.run()`` returns an :class:`AnchorTestReport`
whose :attr:`overall_passed` is ``True`` iff every probed rank-count
satisfied the efficiency tolerance (primary fail-gate); the
absolute-performance tolerance is reported separately as a ``YELLOW``
warning.
"""

from .report import AnchorTestPoint, AnchorTestReport, HardwareProbeResult
from .runner import AnchorTestRunner, RunnerConfig

__all__ = [
    "AnchorTestPoint",
    "AnchorTestReport",
    "AnchorTestRunner",
    "HardwareProbeResult",
    "RunnerConfig",
]
