#!/usr/bin/env python3
"""T6 differential harness — TDMD vs LAMMPS on the SNAP tungsten benchmark.

This is the **D-M8-7 CPU FP64 byte-exact gate** driver — the canonical SNAP
oracle lock per master spec §14 M8 and docs/specs/verify/SPEC.md §4.7. A
failure here blocks M8 closure and indicates a bug in the SnapPotential
force body port (T8.4b). Threshold relaxation requires Validation Engineer
+ Architect review (master spec §D.15 bisect protocol).

Usage (from repo root)::

    python3 verify/t6/run_differential.py \\
        --benchmark   verify/benchmarks/t6_snap_tungsten \\
        --tdmd        build_cpu/src/cli/tdmd \\
        --lammps      verify/third_party/lammps/install_tdmd/bin/lmp \\
        --lammps-libdir verify/third_party/lammps/install_tdmd/lib \\
        --thresholds  verify/thresholds/thresholds.yaml

Exit codes (mirrors T1/T4 convention):
    0   all thresholds satisfied (PASS)
    1   one or more thresholds violated (FAIL)
    2   harness setup error (missing file, TDMD crash, LAMMPS crash, ...)
   77   LAMMPS binary unavailable — SKIP

Scope vs. T4:
    * Pure SNAP, no ZBL — TDMD's ZBL pair lands in M9+; the canonical Wood
      2017 production fixture (lammps_script.in) uses hybrid/overlay zbl+snap
      for physics realism, but the byte-exact gate compares the bispectrum
      path standalone (lammps_script_metal.in).
    * Coefficient files come from the LAMMPS submodule (D-M8-3 repo-size
      preservation) at verify/third_party/lammps/examples/snap/. The
      benchmark driver passes snap_dir as a -var so cloud-burst hosts can
      relocate the submodule.
    * Forces threshold is 1e-12 rel (one decade tighter than T4's EAM 1e-10)
      because the Wood 2017 coefficients are the canonical published set
      (no reparametrization noise) and SNAP bispectrum has shallower
      summation depth than EAM's two-pass embedding.
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
        description="T6 differential harness (TDMD vs LAMMPS, SNAP tungsten)."
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
        "--snap-dir",
        type=pathlib.Path,
        default=None,
        help="Directory holding W_2940_2017_2.snapcoeff/.snapparam (defaults to "
        "verify/third_party/lammps/examples/snap relative to repo root).",
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
    snap_dir = (
        args.snap_dir.resolve()
        if args.snap_dir is not None
        else (REPO_ROOT / "verify/third_party/lammps/examples/snap").resolve()
    )
    coeff_file = snap_dir / "W_2940_2017_2.snapcoeff"
    param_file = snap_dir / "W_2940_2017_2.snapparam"

    for p in (
        lammps_script,
        checks_yaml,
        config_metal,
        setup_data,
        coeff_file,
        param_file,
    ):
        if not p.exists():
            _die(f"benchmark asset missing: {p}")

    owned_tmp = None
    if args.workdir is None:
        owned_tmp = tempfile.mkdtemp(prefix="tdmd_t6_")
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
                    "snap_dir": str(snap_dir),
                },
            )
        except runner.HarnessError as exc:
            _die(str(exc))
        lammps_dump = workdir / "lammps.dump"
        if not lammps_dump.exists():
            _die(f"LAMMPS did not emit lammps.dump at {lammps_dump}")

        # --- TDMD stage: same initial state, same SNAP coefficients. The harness
        # rewrites the yaml config into the workdir с absolute paths so the
        # relative `../../third_party/lammps/examples/snap/...` resolution does
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
                    ("potential", "params", "coeff_file"): coeff_file,
                    ("potential", "params", "param_file"): param_file,
                },
            )
        except runner.HarnessError as exc:
            _die(str(exc))
        if tdmd_out.dump_path is None or not tdmd_out.dump_path.exists():
            _die("TDMD did not emit a forces dump (emit_dump=True expected)")

        tolerances = compare.load_thresholds(args.thresholds)

        # --- Diff #1: thermo columns (same pipeline as T1/T4).
        thermo_report = runner.diff_thermo_vs_lammps(
            tdmd_thermo=tdmd_out.thermo_path,
            lammps_log=lammps_log,
            checks_yaml=checks_yaml,
            tolerances=tolerances,
            label="metal vs lammps",
        )
        print(thermo_report.summary())

        # --- Diff #2: per-atom forces at step 100 (the D-M8-7 acceptance clause).
        forces_threshold = compare.resolve_threshold(
            tolerances, "benchmarks.t6_snap_tungsten.cpu_fp64_vs_lammps.forces_relative"
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
                f"[T6] --expect-fail: inverted verdict, "
                f"harness-level result = {'PASS' if actual_pass else 'FAIL'}"
            )
        return EXIT_PASS if actual_pass else EXIT_FAIL
    finally:
        if owned_tmp is not None and not args.keep_workdir:
            shutil.rmtree(owned_tmp, ignore_errors=True)
        elif args.keep_workdir:
            print(f"[T6] workdir retained at {workdir}")


if __name__ == "__main__":
    sys.exit(main())
