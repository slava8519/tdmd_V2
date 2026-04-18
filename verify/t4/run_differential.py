#!/usr/bin/env python3
"""T4 differential harness — TDMD vs LAMMPS on the Ni-Al EAM/alloy benchmark.

This is the M2 **acceptance gate** driver. A failure here blocks M2 closure
per master spec §14 M2 and docs/specs/verify/SPEC.md §4.7.

Usage (from repo root)::

    python3 verify/t4/run_differential.py \\
        --benchmark   verify/benchmarks/t4_nial_alloy \\
        --tdmd        build_cpu/src/cli/tdmd \\
        --lammps      verify/third_party/lammps/install_tdmd/bin/lmp \\
        --lammps-libdir verify/third_party/lammps/install_tdmd/lib \\
        --thresholds  verify/thresholds/thresholds.yaml

Exit codes (mirrors T1 convention):
    0   all thresholds satisfied (PASS)
    1   one or more thresholds violated (FAIL)
    2   harness setup error (missing file, TDMD crash, LAMMPS crash, ...)
   77   LAMMPS binary unavailable — SKIP (autotools-style; consumed by the
        Catch2 wrapper).

Scope vs. T1:
    * No lj variant — EAM setfl tables are dimensional by convention.
    * Forces diff (T2.8 layer) is mandatory at 1e-10 relative; thermo diff
      reuses the T1 threshold envelope (same residual sources).
    * ``setup.data`` is committed (not LAMMPS-generated), so the T4 LAMMPS
      script passes it via a ``-var setup_data <abs_path>`` variable.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verify import compare  # noqa: E402  (import after sys.path tweak)
from verify.harness import differential_runner as runner  # noqa: E402

EXIT_PASS = runner.EXIT_PASS
EXIT_FAIL = runner.EXIT_FAIL
EXIT_ERROR = runner.EXIT_ERROR
EXIT_SKIP = runner.EXIT_SKIP


def _die(msg: str, code: int = EXIT_ERROR) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _skip(msg: str) -> None:
    print(f"SKIP: {msg}", file=sys.stderr)
    sys.exit(EXIT_SKIP)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="T4 differential harness (TDMD vs LAMMPS, Ni-Al EAM)."
    )
    parser.add_argument(
        "--benchmark",
        type=pathlib.Path,
        required=True,
        help="Path to the benchmark directory (must contain config_metal.yaml, "
        "lammps_script_metal.in, checks.yaml, setup.data).",
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
        help="Working directory (default: tempdir that is removed on success).",
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
    lammps_script = benchmark_dir / "lammps_script_metal.in"
    checks_yaml = benchmark_dir / "checks.yaml"
    config_metal = benchmark_dir / "config_metal.yaml"
    setup_data = benchmark_dir / "setup.data"
    eam_file = (
        REPO_ROOT / "verify/third_party/potentials/NiAl_Mishin_2004.eam.alloy"
    ).resolve()

    for p in (lammps_script, checks_yaml, config_metal, setup_data, eam_file):
        if not p.exists():
            _die(f"benchmark asset missing: {p}")

    owned_tmp = None
    if args.workdir is None:
        owned_tmp = tempfile.mkdtemp(prefix="tdmd_t4_")
        workdir = pathlib.Path(owned_tmp)
    else:
        workdir = args.workdir
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        # --- LAMMPS stage: read the committed setup.data, run NVE 100 steps,
        # dump forces at step 100.
        try:
            lammps_log = runner.run_lammps(
                args.lammps,
                args.lammps_libdir,
                lammps_script,
                workdir,
                extra_vars={
                    "setup_data": str(setup_data.resolve()),
                    "eam_file": str(eam_file),
                },
            )
        except runner.HarnessError as exc:
            _die(str(exc))
        lammps_dump = workdir / "lammps.dump"
        if not lammps_dump.exists():
            _die(f"LAMMPS did not emit lammps.dump at {lammps_dump}")

        # --- TDMD stage: same initial state, same EAM setfl.  The harness
        # rewrites the yaml config into the workdir with absolute paths so
        # the relative ``../../third_party/potentials/…`` resolution does
        # not break once the config moves out of the benchmark directory.
        try:
            tdmd_out = runner.run_tdmd(
                args.tdmd,
                config_metal,
                setup_data,
                workdir,
                "metal",
                emit_dump=True,
                extra_absolute_paths={
                    ("potential", "params", "file"): eam_file,
                },
            )
        except runner.HarnessError as exc:
            _die(str(exc))
        if tdmd_out.dump_path is None or not tdmd_out.dump_path.exists():
            _die("TDMD did not emit a forces dump (emit_dump=True expected)")

        tolerances = compare.load_thresholds(args.thresholds)

        # --- Diff #1: thermo columns (same pipeline as T1).
        thermo_report = runner.diff_thermo_vs_lammps(
            tdmd_thermo=tdmd_out.thermo_path,
            lammps_log=lammps_log,
            checks_yaml=checks_yaml,
            tolerances=tolerances,
            label="metal vs lammps",
        )
        print(thermo_report.summary())

        # --- Diff #2: per-atom forces at step 100 (the T4 acceptance clause).
        forces_threshold = compare.resolve_threshold(
            tolerances, "benchmarks.t4_nial_alloy.forces_relative"
        )
        forces_diff, forces_summary = runner.diff_forces_vs_lammps(
            tdmd_dump=tdmd_out.dump_path,
            lammps_dump=lammps_dump,
            threshold_rel=forces_threshold,
            label="metal",
        )
        print(forces_summary)

        actual_pass = thermo_report.passed and forces_diff.passed
        if args.expect_fail:
            actual_pass = not actual_pass
            print(
                f"[T4] --expect-fail: inverted verdict, "
                f"harness-level result = {'PASS' if actual_pass else 'FAIL'}"
            )
        return EXIT_PASS if actual_pass else EXIT_FAIL
    finally:
        if owned_tmp is not None and not args.keep_workdir:
            shutil.rmtree(owned_tmp, ignore_errors=True)
        elif args.keep_workdir:
            print(f"[T4] workdir retained at {workdir}")


if __name__ == "__main__":
    sys.exit(main())
