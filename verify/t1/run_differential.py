#!/usr/bin/env python3
"""T1 differential harness — runs LAMMPS + TDMD and diffs thermo streams.

Usage (from repo root)::

    python3 verify/t1/run_differential.py \\
        --benchmark   verify/benchmarks/t1_al_morse_500 \\
        --tdmd        build_cpu/src/cli/tdmd \\
        --lammps      verify/third_party/lammps/install_tdmd/bin/lmp \\
        --lammps-libdir verify/third_party/lammps/install_tdmd/lib \\
        --thresholds  verify/thresholds/thresholds.yaml \\
        --variant     both

Variants
--------
  metal  TDMD-metal (config_metal.yaml) vs LAMMPS-metal. Default.
  lj     TDMD-lj    (config_lj.yaml, identity reference) vs LAMMPS-metal.
         Setup velocities are pre-scaled by sqrt(mvv2e_metal) so the lj→metal
         conversion inside TDMD's UnitConverter recovers the original Å/ps
         values. Length/mass/energy are σ=ε=m_ref=1 identity.
  both   runs metal + lj, then cross-checks TDMD-metal ≡ TDMD-lj at
         ``benchmarks.t1_al_morse_500.cross_unit_relative`` (1e-10 rel).
         This is the D-M1-6 invariant — UnitConverter is a numerical no-op
         on the downstream force/integrator pipeline.

Exit codes
----------
  0  all thresholds satisfied (PASS)
  1  one or more thresholds violated (FAIL)
  2  harness setup error (invalid args, tdmd binary missing, LAMMPS crashed)
 77  LAMMPS binary unavailable — test should be SKIPPED (mirrors autotools
     convention; consumed by the Catch2 wrapper).

SPEC: ``docs/specs/verify/SPEC.md`` §4.5 (T1 lj variant), §5 (LAMMPS
integration), §7.1 (differential runner). Exec pack: T1.11 (M1) + T2.4 (M2).
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import shutil
import sys
import tempfile

# Make `import verify.compare` work when invoked from the repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verify import compare  # noqa: E402  (import after sys.path tweak)
from verify.harness import differential_runner as runner  # noqa: E402

EXIT_PASS = runner.EXIT_PASS
EXIT_FAIL = runner.EXIT_FAIL
EXIT_ERROR = runner.EXIT_ERROR
EXIT_SKIP = runner.EXIT_SKIP

# LAMMPS `units metal` mvv2e — the single constant that couples KE=½mv² in
# (g/mol)·(Å/ps)² to energy in eV. Mirrored in src/runtime/unit_converter.cpp
# (kMetalMvv2e in src/runtime/include/tdmd/runtime/physical_constants.hpp);
# keep these two numbers identical or the cross-check residual will explode.
METAL_MVV2E = 1.0364269e-4

# v_lj = v_metal · sqrt(mvv2e).  With identity σ=ε=m_ref=1 this is the full
# lj velocity unit — anything extra would be sqrt(ε/m)·σ · sqrt(mvv2e) but
# σ=ε=m=1 collapses the prefactor to 1.
LJ_VELOCITY_SCALE_METAL_TO_LJ = math.sqrt(METAL_MVV2E)


def _die(msg: str, code: int = EXIT_ERROR) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _skip(msg: str) -> None:
    print(f"SKIP: {msg}", file=sys.stderr)
    sys.exit(EXIT_SKIP)


# Process runners live in ``verify/harness/differential_runner.py`` (T2.8).
# The T1 driver retains the lj-variant-specific bits below; LAMMPS + TDMD
# invocation is delegated to the shared runner so T4 and future benchmarks
# exercise the same code path.


def scale_velocities_for_lj(
    src_data: pathlib.Path, dst_data: pathlib.Path, scale: float
) -> None:
    """Rewrite a LAMMPS data file, multiplying each velocity by ``scale``.

    Input is LAMMPS-metal setup.data (velocities in Å/ps). With identity
    σ=ε=m_ref=1, the non-velocity sections (box, Masses, Atoms) are valid lj
    numerics bit-for-bit — only the Velocities block needs the sqrt(mvv2e)
    pre-scale so that TDMD's ``UnitConverter::velocity_from_lj`` recovers the
    original Å/ps magnitudes on ingest.

    LAMMPS ``write_data nocoeff`` format (the only shape T1 emits):
        <header comment / counts>
        ...
        Velocities
        <blank>
        <id> <vx> <vy> <vz>
        <id> <vx> <vy> <vz>
        ...
        <blank>
        <next section or EOF>

    We walk the file in a two-state machine: outside the Velocities block the
    lines pass through untouched; inside, we parse ``id vx vy vz`` and emit
    scaled values. Any line that doesn't look like a 4-token numeric row
    terminates the block (robust against trailing sections).
    """
    with src_data.open() as fh:
        lines = fh.readlines()

    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "Velocities":
            out.append(line)
            i += 1
            # Consume optional blank line after section header.
            if i < len(lines) and lines[i].strip() == "":
                out.append(lines[i])
                i += 1
            # Scale body until blank or non-numeric row.
            while i < len(lines):
                stripped = lines[i].strip()
                if not stripped:
                    out.append(lines[i])
                    i += 1
                    break
                tokens = stripped.split()
                if len(tokens) != 4:
                    break  # unrelated content — don't consume.
                try:
                    atom_id = int(tokens[0])
                    vx = float(tokens[1]) * scale
                    vy = float(tokens[2]) * scale
                    vz = float(tokens[3]) * scale
                except ValueError:
                    break
                # %.17g round-trips double precision; repr() would work too but
                # '17g' is what compare.py-style parsers already expect.
                out.append(f"{atom_id} {vx:.17g} {vy:.17g} {vz:.17g}\n")
                i += 1
            continue
        out.append(line)
        i += 1

    with dst_data.open("w") as fh:
        fh.writelines(out)


def cross_unit_compare(
    a_thermo: pathlib.Path,
    b_thermo: pathlib.Path,
    columns: list[str],
    threshold_rel: float,
    name: str,
) -> compare.DiffReport:
    """Diff two TDMD thermo files at a uniform relative threshold.

    Both files have identical column layout (TDMD emits metal-unit thermo
    regardless of input unit system), so no column map or pressure conversion
    is needed — unlike ``compare.compare_columns`` which is LAMMPS-aware.
    """
    a_cols, a_rows = compare.parse_thermo_file(a_thermo)
    b_cols, b_rows = compare.parse_thermo_file(b_thermo)

    a_idx = {c: i for i, c in enumerate(a_cols)}
    b_idx = {c: i for i, c in enumerate(b_cols)}
    if "step" not in a_idx or "step" not in b_idx:
        raise ValueError("both thermo tables must carry a 'step' column")

    a_by_step = {int(r[a_idx["step"]]): r for r in a_rows}
    b_by_step = {int(r[b_idx["step"]]): r for r in b_rows}
    common = sorted(set(a_by_step) & set(b_by_step))
    if not common:
        return compare.DiffReport(
            name=name,
            passed=False,
            column_diffs=[],
            note="no overlapping thermo steps between metal and lj TDMD logs",
        )

    diffs: list[compare.ColumnDiff] = []
    all_passed = True
    for col in columns:
        if col not in a_idx or col not in b_idx:
            diffs.append(
                compare.ColumnDiff(
                    column=col,
                    max_abs_diff=float("inf"),
                    max_rel_diff=float("inf"),
                    at_row=-1,
                    at_step=-1,
                    threshold=threshold_rel,
                    passed=False,
                )
            )
            all_passed = False
            continue
        max_abs = 0.0
        max_rel = 0.0
        at_row = 0
        at_step = common[0]
        for row_idx, step in enumerate(common):
            a = a_by_step[step][a_idx[col]]
            b = b_by_step[step][b_idx[col]]
            abs_d = abs(a - b)
            denom = max(abs(a), abs(b))
            rel_d = abs_d / denom if denom else 0.0
            if rel_d > max_rel:
                max_rel = rel_d
                max_abs = abs_d
                at_row = row_idx
                at_step = step
        passed = max_rel <= threshold_rel
        all_passed = all_passed and passed
        diffs.append(
            compare.ColumnDiff(
                column=col,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                at_row=at_row,
                at_step=at_step,
                threshold=threshold_rel,
                passed=passed,
            )
        )
    return compare.DiffReport(name=name, passed=all_passed, column_diffs=diffs)


def _diff_against_lammps(
    tdmd_thermo: pathlib.Path,
    lammps_log: pathlib.Path,
    checks_yaml: pathlib.Path,
    tolerances,
    label: str,
) -> compare.DiffReport:
    """Shim around the shared harness so the T1 call sites stay unchanged."""
    return runner.diff_thermo_vs_lammps(
        tdmd_thermo=tdmd_thermo,
        lammps_log=lammps_log,
        checks_yaml=checks_yaml,
        tolerances=tolerances,
        label=f"{label} vs lammps",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="T1 differential harness (TDMD vs LAMMPS)."
    )
    parser.add_argument(
        "--benchmark",
        type=pathlib.Path,
        required=True,
        help="Path to the benchmark directory (must contain config_<variant>.yaml, "
        "lammps_script_metal.in, checks.yaml).",
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
        "--variant",
        choices=["metal", "lj", "both"],
        default="metal",
        help="Which TDMD unit variant to run (default: metal). `both` adds a "
        "TDMD-metal ≡ TDMD-lj cross-check at benchmarks.<bench>.cross_unit_relative.",
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
    lammps_script = benchmark_dir / "lammps_script_metal.in"
    checks_yaml = benchmark_dir / "checks.yaml"
    config_metal = benchmark_dir / "config_metal.yaml"
    config_lj = benchmark_dir / "config_lj.yaml"

    need_metal = args.variant in ("metal", "both")
    need_lj = args.variant in ("lj", "both")

    required_assets = [lammps_script, checks_yaml]
    if need_metal:
        required_assets.append(config_metal)
    if need_lj:
        required_assets.append(config_lj)
    for p in required_assets:
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
        try:
            lammps_log = runner.run_lammps(
                args.lammps, args.lammps_libdir, lammps_script, workdir
            )
        except runner.HarnessError as exc:
            _die(str(exc))
        setup_data = workdir / "setup.data"
        if not setup_data.exists():
            _die(f"LAMMPS did not emit setup.data at {setup_data}")

        tolerances = compare.load_thresholds(args.thresholds)

        reports: list[compare.DiffReport] = []
        thermo_metal: pathlib.Path | None = None
        thermo_lj: pathlib.Path | None = None

        if need_metal:
            try:
                tdmd_out = runner.run_tdmd(
                    args.tdmd,
                    config_metal,
                    setup_data,
                    workdir,
                    "metal",
                    emit_dump=False,
                )
            except runner.HarnessError as exc:
                _die(str(exc))
            thermo_metal = tdmd_out.thermo_path
            reports.append(
                _diff_against_lammps(
                    thermo_metal, lammps_log, checks_yaml, tolerances, "metal"
                )
            )

        if need_lj:
            setup_data_lj = workdir / "setup_lj.data"
            scale_velocities_for_lj(
                setup_data, setup_data_lj, LJ_VELOCITY_SCALE_METAL_TO_LJ
            )
            try:
                tdmd_out = runner.run_tdmd(
                    args.tdmd,
                    config_lj,
                    setup_data_lj,
                    workdir,
                    "lj",
                    emit_dump=False,
                )
            except runner.HarnessError as exc:
                _die(str(exc))
            thermo_lj = tdmd_out.thermo_path
            reports.append(
                _diff_against_lammps(
                    thermo_lj, lammps_log, checks_yaml, tolerances, "lj"
                )
            )

        if args.variant == "both":
            assert thermo_metal is not None and thermo_lj is not None
            _, checks, bench_name = compare.load_checks(checks_yaml, tolerances)
            cross_threshold = compare.resolve_threshold(
                tolerances, f"benchmarks.{bench_name}.cross_unit_relative"
            )
            # Use the same column set the LAMMPS-diff exercises, so cross-check
            # coverage stays in lock-step with what the benchmark promises.
            cross_cols = [c.column for c in checks]
            reports.append(
                cross_unit_compare(
                    thermo_metal,
                    thermo_lj,
                    cross_cols,
                    cross_threshold,
                    name=f"{bench_name} [metal ≡ lj cross-unit]",
                )
            )

        for report in reports:
            print(report.summary())

        actual_pass = all(r.passed for r in reports)
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
