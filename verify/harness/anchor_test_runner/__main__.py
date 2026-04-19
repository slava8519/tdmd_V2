"""CLI front-end: ``python -m verify.harness.anchor_test_runner``.

Resolves defaults against the repo layout (benchmark at
``verify/benchmarks/t3_al_fcc_large_anchor``, tdmd binary at
``build/tdmd``). All knobs are overridable via flags.

Exit codes:
    0 — report.overall_passed and overall_status == GREEN
    1 — report.overall_status == YELLOW (efficiency clean, abs-perf warn)
    2 — report.overall_status == RED   (efficiency fail_if tripped)
    3 — runtime / setup error (missing binary, setup.data regen failed)
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback

from .report import STATUS_GREEN, STATUS_RED, STATUS_YELLOW
from .runner import AnchorTestRunner, RunnerConfig


def _parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    default_benchmark = repo_root / "verify" / "benchmarks" / "t3_al_fcc_large_anchor"
    default_tdmd = repo_root / "build" / "tdmd"
    default_report = repo_root / "build" / "t3_anchor_report.json"
    default_workdir = repo_root / "build" / "t3_anchor_workdir"

    parser = argparse.ArgumentParser(
        prog="anchor_test_runner",
        description="T3 anchor-test driver — reproduces Andreev §3.5 on local MPI.",
    )
    parser.add_argument("--benchmark-dir", type=pathlib.Path, default=default_benchmark)
    parser.add_argument("--tdmd-bin", type=pathlib.Path, default=default_tdmd)
    parser.add_argument(
        "--mpirun-bin", type=pathlib.Path, default=pathlib.Path("mpirun")
    )
    parser.add_argument("--workdir", type=pathlib.Path, default=default_workdir)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=default_report,
        help="structured JSON report path",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="override checks.yaml::ranks_to_probe",
    )
    parser.add_argument(
        "--force-probe", action="store_true", help="bypass 24h hardware probe cache"
    )
    parser.add_argument(
        "--lammps-bin",
        type=pathlib.Path,
        default=None,
        help="LAMMPS binary for setup.data regen (if missing)",
    )
    parser.add_argument("--per-run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--backend",
        choices=["cpu", "gpu"],
        default=None,
        help=(
            "override checks.yaml::backend. Without this flag the runner "
            "reads the fixture's declared backend (CPU T3 → cpu, T3-gpu → gpu)."
        ),
    )
    return parser.parse_args(argv)


def _exit_code_for(status: str) -> int:
    if status == STATUS_GREEN:
        return 0
    if status == STATUS_YELLOW:
        return 1
    if status == STATUS_RED:
        return 2
    return 3


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        config = RunnerConfig(
            benchmark_dir=args.benchmark_dir.resolve(),
            tdmd_bin=args.tdmd_bin.resolve(),
            mpirun_bin=args.mpirun_bin,
            output_report_path=args.output.resolve(),
            workdir=args.workdir.resolve(),
            ranks_override=args.ranks,
            force_probe=args.force_probe,
            per_run_timeout_seconds=args.per_run_timeout_seconds,
            lammps_bin=args.lammps_bin.resolve() if args.lammps_bin else None,
            backend_override=args.backend,
        )
        runner = AnchorTestRunner(config)
        report = runner.run()
    except Exception as exc:  # noqa: BLE001 — top-level CLI handler
        print(f"anchor-test runner failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3

    report.write_json(config.output_report_path)
    print(report.format_console_summary())
    print(f"report: {config.output_report_path}")
    return _exit_code_for(report.overall_status)


if __name__ == "__main__":
    sys.exit(main())
