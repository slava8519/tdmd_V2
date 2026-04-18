#!/usr/bin/env python3
"""T1 differential harness — runs LAMMPS + TDMD and diffs thermo streams.

Usage (from repo root)::

    python3 verify/t1/run_differential.py \\
        --benchmark   verify/benchmarks/t1_al_morse_500 \\
        --tdmd        build_cpu/src/cli/tdmd \\
        --lammps      verify/third_party/lammps/install_tdmd/bin/lmp \\
        --lammps-libdir verify/third_party/lammps/install_tdmd/lib \\
        --thresholds  verify/thresholds/thresholds.yaml

Exit codes
----------
  0  all thresholds satisfied (PASS)
  1  one or more thresholds violated (FAIL)
  2  harness setup error (invalid args, tdmd binary missing, LAMMPS crashed)
 77  LAMMPS binary unavailable — test should be SKIPPED (mirrors autotools
     convention; consumed by the Catch2 wrapper).

SPEC: ``docs/specs/verify/SPEC.md`` §5 (LAMMPS integration), §7.1
(differential runner). Exec pack: ``docs/development/m1_execution_pack.md``
T1.11.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Make `import verify.compare` work when invoked from the repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verify import compare  # noqa: E402  (import after sys.path tweak)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2
EXIT_SKIP = 77


def _die(msg: str, code: int = EXIT_ERROR) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _skip(msg: str) -> None:
    print(f"SKIP: {msg}", file=sys.stderr)
    sys.exit(EXIT_SKIP)


def run_lammps(
    lammps_bin: pathlib.Path,
    lammps_libdir: pathlib.Path | None,
    script: pathlib.Path,
    workdir: pathlib.Path,
) -> pathlib.Path:
    """Run LAMMPS with the given script. Returns the emitted log path."""
    log_path = workdir / "lammps.log"
    env = os.environ.copy()
    if lammps_libdir is not None:
        existing = env.get("LD_LIBRARY_PATH", "")
        libdir_abs = str(lammps_libdir.resolve())
        env["LD_LIBRARY_PATH"] = f"{libdir_abs}:{existing}" if existing else libdir_abs
    # Resolve to absolute before passing — subprocess.run(cwd=workdir) would
    # otherwise break any relative path the caller passed in.
    cmd = [
        str(lammps_bin.resolve()),
        "-in",
        str(script.resolve()),
        "-var",
        "workdir",
        str(workdir.resolve()),
        "-log",
        str(log_path.resolve()),
        "-screen",
        "none",
    ]
    print(f"[T1] running LAMMPS: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=workdir, check=False)
    if result.returncode != 0:
        _die(f"LAMMPS exited with code {result.returncode}; see {log_path}")
    return log_path


def run_tdmd(
    tdmd_bin: pathlib.Path,
    benchmark_config: pathlib.Path,
    setup_data: pathlib.Path,
    workdir: pathlib.Path,
) -> pathlib.Path:
    """Write a workdir-local copy of the config (absolute atoms.path) and run tdmd.

    Returns the path to the TDMD thermo file.
    """
    import yaml  # local import keeps the file PyYAML-free until needed.

    with benchmark_config.open() as fh:
        cfg = yaml.safe_load(fh)
    cfg["atoms"]["path"] = str(setup_data.resolve())

    resolved_cfg = workdir / "tdmd_config.yaml"
    with resolved_cfg.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    thermo_path = workdir / "tdmd_thermo.log"
    cmd = [
        str(tdmd_bin),
        "run",
        "--quiet",
        "--thermo",
        str(thermo_path),
        str(resolved_cfg),
    ]
    print(f"[T1] running TDMD:   {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        _die(f"TDMD exited with code {result.returncode}")
    return thermo_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="T1 differential harness (TDMD vs LAMMPS)."
    )
    parser.add_argument(
        "--benchmark",
        type=pathlib.Path,
        required=True,
        help="Path to the benchmark directory "
        "(must contain config.yaml, lammps_script.in, checks.yaml).",
    )
    parser.add_argument(
        "--tdmd", type=pathlib.Path, required=True, help="Path to the tdmd binary."
    )
    parser.add_argument(
        "--lammps", type=pathlib.Path, required=True, help="Path to the lmp binary."
    )
    parser.add_argument(
        "--lammps-libdir",
        type=pathlib.Path,
        default=None,
        help="Directory containing liblammps.so.* (prepended to LD_LIBRARY_PATH).",
    )
    parser.add_argument(
        "--thresholds",
        type=pathlib.Path,
        required=True,
        help="Path to thresholds.yaml (master threshold registry).",
    )
    parser.add_argument(
        "--workdir",
        type=pathlib.Path,
        default=None,
        help="Working directory (default: a tempdir that is removed on success).",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Do not delete the tempdir at the end (useful for debugging).",
    )
    parser.add_argument(
        "--expect-fail",
        action="store_true",
        help="Invert the exit code (for self-check that planted bugs do fail).",
    )
    args = parser.parse_args(argv)

    if not args.tdmd.exists() or not os.access(args.tdmd, os.X_OK):
        _die(f"tdmd binary not found / not executable: {args.tdmd}")
    if not args.lammps.exists() or not os.access(args.lammps, os.X_OK):
        _skip(
            f"LAMMPS binary not available at {args.lammps}; run tools/build_lammps.sh "
            "locally before invoking this harness."
        )

    benchmark_dir = args.benchmark.resolve()
    config_yaml = benchmark_dir / "config.yaml"
    lammps_script = benchmark_dir / "lammps_script.in"
    checks_yaml = benchmark_dir / "checks.yaml"
    for p in (config_yaml, lammps_script, checks_yaml):
        if not p.exists():
            _die(f"benchmark asset missing: {p}")

    owned_tmp = None
    if args.workdir is None:
        owned_tmp = tempfile.mkdtemp(prefix="tdmd_t1_")
        workdir = pathlib.Path(owned_tmp)
    else:
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        lammps_log = run_lammps(args.lammps, args.lammps_libdir, lammps_script, workdir)
        setup_data = workdir / "setup.data"
        if not setup_data.exists():
            _die(f"LAMMPS did not emit setup.data at {setup_data}")

        tdmd_thermo = run_tdmd(args.tdmd, config_yaml, setup_data, workdir)

        tolerances = compare.load_thresholds(args.thresholds)
        column_map, checks, name = compare.load_checks(checks_yaml, tolerances)

        tdmd_cols, tdmd_rows = compare.parse_thermo_file(tdmd_thermo)
        lmp_cols, lmp_rows = compare.extract_lammps_thermo(lammps_log)

        report = compare.compare_columns(
            tdmd_cols=tdmd_cols,
            tdmd_rows=tdmd_rows,
            lmp_cols=lmp_cols,
            lmp_rows=lmp_rows,
            column_map=column_map,
            checks=checks,
            name=name,
        )
        print(report.summary())

        actual_pass = report.passed
        if args.expect_fail:
            actual_pass = not actual_pass
            print(
                f"[T1] --expect-fail: inverted verdict, "
                f"harness-level result = {'PASS' if actual_pass else 'FAIL'}"
            )
        return EXIT_PASS if actual_pass else EXIT_FAIL
    finally:
        if owned_tmp is not None and not args.keep_workdir:
            shutil.rmtree(owned_tmp, ignore_errors=True)
        elif args.keep_workdir:
            print(f"[T1] workdir retained at {workdir}")


if __name__ == "__main__":
    sys.exit(main())
