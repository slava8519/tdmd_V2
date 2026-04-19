"""CLI front-end: ``python -m verify.harness.scaling_runner``.

Exit codes:
    0 — overall_status == GREEN (all gates pass)
    2 — overall_status == RED   (gate fail or Pattern 1 byte-exact break)
    3 — runtime / setup error (missing binary, setup.data regen failed)

YELLOW is not currently used by ScalingRunner; the gate model is binary
(green/red). Reserved for T7.13 calibration drift warnings.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

from .runner import (
    RunnerConfig,
    ScalingRunner,
    STATUS_GREEN,
    STATUS_RED,
    STATUS_YELLOW,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    default_benchmark = repo_root / "verify" / "benchmarks" / "t7_mixed_scaling"
    default_tdmd = repo_root / "build" / "tdmd"
    default_report = repo_root / "build" / "t7_scaling_report.json"
    default_workdir = repo_root / "build" / "t7_scaling_workdir"

    parser = argparse.ArgumentParser(
        prog="scaling_runner",
        description="T7 mixed-scaling driver — Pattern 2 strong-scaling probe.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=pathlib.Path,
        default=default_benchmark,
    )
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
        type=str,
        default=None,
        help=(
            "comma-separated rank counts (override checks.yaml::ranks_to_probe). "
            "Must include 1 as efficiency anchor. Example: --ranks 1,2,4"
        ),
    )
    parser.add_argument("--per-run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--baseline-thermo",
        type=pathlib.Path,
        default=None,
        help=(
            "Pattern 1 baseline thermo file. When supplied AND "
            "checks.yaml::pattern1_baseline_byte_exact is true, the runner "
            "byte-compares the N=1 thermo trace against this file."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["gpu", "cpu"],
        default=None,
        help="override checks.yaml::backend (default 'gpu' for T7)",
    )
    parser.add_argument(
        "--skip-setup-regen",
        action="store_true",
        help="never invoke generate_setup.py — fail if setup.data missing",
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
    ranks_override: list[int] | None = None
    if args.ranks:
        try:
            ranks_override = [
                int(x.strip()) for x in args.ranks.split(",") if x.strip()
            ]
        except ValueError as exc:
            print(f"--ranks must be comma-separated integers: {exc}", file=sys.stderr)
            return 3

    try:
        config = RunnerConfig(
            benchmark_dir=args.benchmark_dir.resolve(),
            tdmd_bin=args.tdmd_bin.resolve(),
            mpirun_bin=args.mpirun_bin,
            output_report_path=args.output.resolve(),
            workdir=args.workdir.resolve(),
            ranks_override=ranks_override,
            per_run_timeout_seconds=args.per_run_timeout_seconds,
            skip_setup_regen=args.skip_setup_regen,
            baseline_thermo_path=(
                args.baseline_thermo.resolve() if args.baseline_thermo else None
            ),
            backend_override=args.backend,
        )
        runner = ScalingRunner(config)
        report = runner.run()
    except Exception as exc:  # noqa: BLE001 — top-level CLI handler
        print(f"scaling_runner failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3

    config.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_report_path.open("w") as fh:
        json.dump(report.to_dict(), fh, indent=2, sort_keys=False)

    # Console summary
    print(f"T7 mixed-scaling — overall: {report.overall_status}")
    for p in report.points:
        gate_str = (
            f"≥{p.gate_pct:.1f}% [{p.gate_name}]"
            if p.gate_pct is not None
            else "(no gate — anchor)"
        )
        print(
            f"  N={p.n_procs:2d}  "
            f"steps/s={p.measured_steps_per_sec:8.3f}  "
            f"E={p.measured_efficiency_pct:6.2f}%  "
            f"gate={gate_str}  "
            f"{p.status}"
        )
    if report.pattern1_baseline_byte_exact is not None:
        print(
            f"  Pattern 1 byte-exact: "
            f"{'PASS' if report.pattern1_baseline_byte_exact else 'FAIL'} "
            f"(diff_byte={report.pattern1_baseline_diff_byte})"
        )
    if report.failure_mode:
        print(f"  failure_mode: {report.failure_mode}")
    print(f"report: {config.output_report_path}")
    return _exit_code_for(report.overall_status)


if __name__ == "__main__":
    sys.exit(main())
