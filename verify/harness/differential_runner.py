"""Generic TDMD-vs-LAMMPS differential runner (T2.8 MVP).

The per-benchmark drivers (``verify/t1/run_differential.py`` at M1,
``verify/t4/run_differential.py`` coming in T2.9) share the same outer
loop:

    1. Run LAMMPS with the oracle script → produces ``lammps.log`` and
       optionally ``lammps.dump`` + ``setup.data`` as side effects.
    2. Run TDMD with the benchmark config, pointing atoms at the LAMMPS-
       emitted ``setup.data``. Emit ``tdmd_thermo.log`` and optionally
       ``tdmd.dump`` via the CLI ``--dump`` flag.
    3. Diff thermo column-wise against LAMMPS's thermo block.
    4. (T2.8+) Diff per-atom forces by aligning dump frames on atom id.
    5. Emit a structured report; exit code encodes pass/fail.

This module exposes the building blocks (``run_lammps``, ``run_tdmd``,
``diff_thermo_vs_lammps``, ``diff_forces_vs_lammps``) as free functions.
The T1 driver wraps them with its own CLI + lj-variant policy; T4 (Ni-Al
EAM) will reuse them unchanged.

SPEC: ``docs/specs/verify/SPEC.md`` §7.1 (differential runner). Exec pack:
``docs/development/m2_execution_pack.md`` T2.8.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import sys
from typing import Mapping

# Make `import verify.compare` work both when invoked via `python3 -m
# verify.harness.differential_runner` and when re-exported by per-benchmark
# drivers that manipulate sys.path themselves.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verify import compare  # noqa: E402

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2
EXIT_SKIP = 77


class HarnessError(RuntimeError):
    """Setup / infrastructure failure — raised by the runner primitives.

    Distinct from a comparison failure (which produces a DiffReport with
    ``passed=False``) so per-benchmark drivers can map them to different
    exit codes.
    """


# ---------------------------------------------------------------------------
# Process runners
# ---------------------------------------------------------------------------


def run_lammps(
    lammps_bin: pathlib.Path,
    lammps_libdir: pathlib.Path | None,
    script: pathlib.Path,
    workdir: pathlib.Path,
    log_name: str = "lammps.log",
) -> pathlib.Path:
    """Invoke ``lmp -in <script>`` from ``workdir``. Returns the log path.

    ``lammps_libdir`` is prepended to ``LD_LIBRARY_PATH`` so locally-built
    LAMMPS binaries that link against ``liblammps.so`` in a sibling dir
    load cleanly without requiring a system-wide install.
    """
    log_path = workdir / log_name
    env = os.environ.copy()
    if lammps_libdir is not None:
        existing = env.get("LD_LIBRARY_PATH", "")
        libdir_abs = str(lammps_libdir.resolve())
        env["LD_LIBRARY_PATH"] = f"{libdir_abs}:{existing}" if existing else libdir_abs
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
    print(f"[harness] LAMMPS: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=workdir, check=False)
    if result.returncode != 0:
        raise HarnessError(
            f"LAMMPS exited with code {result.returncode}; see {log_path}"
        )
    return log_path


@dataclasses.dataclass
class TdmdRun:
    """Paths to the artifacts produced by a single ``run_tdmd`` invocation."""

    thermo_path: pathlib.Path
    dump_path: pathlib.Path | None  # None when ``emit_dump=False``


def run_tdmd(
    tdmd_bin: pathlib.Path,
    benchmark_config: pathlib.Path,
    setup_data: pathlib.Path,
    workdir: pathlib.Path,
    label: str,
    emit_dump: bool = False,
) -> TdmdRun:
    """Run TDMD with a workdir-local copy of the config (absolute atoms.path).

    ``label`` disambiguates per-variant artifacts so the metal and lj runs
    coexist in the same workdir without clobbering each other (T1 lj
    cross-check). ``emit_dump=True`` adds ``--dump`` for T2.8+ forces diffs.
    """
    import yaml

    with benchmark_config.open() as fh:
        cfg = yaml.safe_load(fh)
    cfg["atoms"]["path"] = str(setup_data.resolve())

    resolved_cfg = workdir / f"tdmd_config_{label}.yaml"
    with resolved_cfg.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    thermo_path = workdir / f"tdmd_thermo_{label}.log"
    dump_path = workdir / f"tdmd_dump_{label}.lmp" if emit_dump else None
    cmd = [
        str(tdmd_bin),
        "run",
        "--quiet",
        "--thermo",
        str(thermo_path),
    ]
    if dump_path is not None:
        cmd += ["--dump", str(dump_path)]
    cmd.append(str(resolved_cfg))
    print(f"[harness] TDMD ({label}): {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise HarnessError(f"TDMD ({label}) exited with code {result.returncode}")
    return TdmdRun(thermo_path=thermo_path, dump_path=dump_path)


# ---------------------------------------------------------------------------
# Diff primitives
# ---------------------------------------------------------------------------


def diff_thermo_vs_lammps(
    tdmd_thermo: pathlib.Path,
    lammps_log: pathlib.Path,
    checks_yaml: pathlib.Path,
    tolerances: Mapping,
    label: str,
) -> compare.DiffReport:
    """Column-diff the TDMD thermo against LAMMPS's thermo block.

    Wraps ``compare.load_checks`` + ``compare.compare_columns`` so the
    per-benchmark driver doesn't have to repeat the parse/adapter dance.
    """
    column_map, checks, name = compare.load_checks(checks_yaml, tolerances)
    tdmd_cols, tdmd_rows = compare.parse_thermo_file(tdmd_thermo)
    lmp_cols, lmp_rows = compare.extract_lammps_thermo(lammps_log)
    return compare.compare_columns(
        tdmd_cols=tdmd_cols,
        tdmd_rows=tdmd_rows,
        lmp_cols=lmp_cols,
        lmp_rows=lmp_rows,
        column_map=column_map,
        checks=checks,
        name=f"{name} [{label} thermo]",
    )


def diff_forces_vs_lammps(
    tdmd_dump: pathlib.Path,
    lammps_dump: pathlib.Path,
    threshold_rel: float,
    label: str,
) -> tuple[compare.ForcesDiff, str]:
    """Load two dump frames and diff per-atom forces.

    Returns the raw ``ForcesDiff`` plus a pre-formatted summary line so
    drivers can print reports uniformly. Forces rely on a single scalar
    threshold (``benchmarks.<bench>.forces_relative``); per-component
    thresholds are not currently supported and would be a SPEC delta.
    """
    tdmd_frame = compare.parse_dump_file(tdmd_dump)
    lmp_frame = compare.parse_dump_file(lammps_dump)
    diff = compare.compare_forces(tdmd_frame, lmp_frame, threshold_rel)
    mark = "ok " if diff.passed else "FAIL"
    summary = (
        f"[{mark}] forces  "
        f"max_abs={diff.max_abs_diff:.3e}  "
        f"max_rel={diff.max_rel_diff:.3e}  "
        f"(threshold_rel={diff.threshold:.1e}) "
        f"at atom id={diff.at_atom_id} component={diff.at_component} "
        f"[{label}, N={diff.n_atoms}]"
    )
    return diff, summary
