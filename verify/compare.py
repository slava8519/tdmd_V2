"""Generic numerical comparison helpers for TDMD VerifyLab.

Used by per-benchmark harnesses (``verify/t1/run_differential.py`` being the
first) to diff parallel thermo logs against a tolerance registry.

Design choices
--------------
* Standard library only (no numpy / pandas) so the harness runs in a stock
  CI environment without a virtualenv.
* File format: whitespace-separated numeric columns with a leading header
  line that starts with ``#`` and names the columns. This matches both the
  TDMD thermo output and the LAMMPS thermo block produced by
  ``thermo_style custom``.
* All comparisons are **relative**. Tolerances carry SI / native-unit
  annotations in ``verify/thresholds/thresholds.yaml``; see SPEC
  ``docs/specs/verify/SPEC.md`` §3.

SPEC: ``docs/specs/verify/SPEC.md`` §3 (threshold registry), §7.3
(observables comparator).
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Iterable, Mapping

import yaml

# LAMMPS ``units metal`` reports pressure in bar. TDMD reports in eV/Å³.
#
# LAMMPS's internal conversion uses `nktv2p = 1.6021765e+6` (see
# `verify/third_party/lammps/src/update.cpp`). We mirror that truncated
# constant here — using modern CODATA 2018 (1.602176634e+6) would insert a
# systematic ~8e-8 residual that the harness would incorrectly attribute to
# physics. This is the single definitional difference between the two
# stacks; using LAMMPS's constant removes it.
EV_PER_A3_IN_BAR = 1.6021765e6


@dataclasses.dataclass(frozen=True)
class ColumnCheck:
    """One row from a benchmark's ``checks.yaml`` (column_relative variant)."""

    name: str
    column: str
    threshold_rel: float
    required: bool = True


@dataclasses.dataclass
class ColumnDiff:
    """Outcome of comparing a single column across two thermo tables."""

    column: str
    max_abs_diff: float
    max_rel_diff: float
    at_row: int  # 0-based index into the step column
    at_step: int
    threshold: float
    passed: bool


@dataclasses.dataclass
class DiffReport:
    name: str
    passed: bool
    column_diffs: list[ColumnDiff]
    note: str = ""

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.name}",
            f"  status: {'PASS' if self.passed else 'FAIL'}",
        ]
        for cd in self.column_diffs:
            mark = "ok " if cd.passed else "FAIL"
            lines.append(
                f"  [{mark}] {cd.column:<8} "
                f"max_abs={cd.max_abs_diff:.3e} "
                f"max_rel={cd.max_rel_diff:.3e} "
                f"(threshold_rel={cd.threshold:.1e}) "
                f"at row {cd.at_row} (step={cd.at_step})"
            )
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Threshold registry access
# ---------------------------------------------------------------------------


def load_thresholds(path: pathlib.Path) -> Mapping[str, Any]:
    with path.open() as fh:
        doc = yaml.safe_load(fh)
    return doc.get("tolerances", {})


def resolve_threshold(tolerances: Mapping[str, Any], dotted_path: str) -> float:
    """Resolve a dotted key path (e.g. ``benchmarks.t1.x``) into a numeric value."""
    node: Any = tolerances
    for segment in dotted_path.split("."):
        if not isinstance(node, Mapping) or segment not in node:
            raise KeyError(
                f"threshold path '{dotted_path}' not found (segment '{segment}')"
            )
        node = node[segment]
    if not isinstance(node, (int, float)):
        raise TypeError(
            f"threshold path '{dotted_path}' resolved to non-numeric value: {node!r}"
        )
    return float(node)


# ---------------------------------------------------------------------------
# Thermo file parsing
# ---------------------------------------------------------------------------


def parse_thermo_file(path: pathlib.Path) -> tuple[list[str], list[list[float]]]:
    """Parse a whitespace-separated thermo log.

    Returns ``(columns, rows)`` where ``columns`` is the list of column names
    (taken from the first ``#``-prefixed line) and ``rows`` is a list of
    float lists, one per non-comment non-empty line.
    """
    columns: list[str] | None = None
    rows: list[list[float]] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if columns is None:
                    tokens = line.lstrip("#").split()
                    columns = tokens
                continue
            tokens = line.split()
            try:
                rows.append([float(t) for t in tokens])
            except ValueError:
                # Non-numeric content after the header — treat as EOF of the
                # thermo block (LAMMPS emits trailing "Loop time..." lines).
                break
    if columns is None:
        raise ValueError(f"{path}: no header line (expected a line starting with '#')")
    return columns, rows


def extract_lammps_thermo(
    log_path: pathlib.Path,
) -> tuple[list[str], list[list[float]]]:
    """Extract the thermo table from a LAMMPS log file.

    LAMMPS emits the table between ``Per MPI rank memory allocation`` and
    ``Loop time of`` lines. First line after the memory header is the column
    header (unprefixed); subsequent lines are numeric rows.
    """
    with log_path.open() as fh:
        lines = fh.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.startswith("Per MPI rank memory allocation"):
            start = i + 1
            break
    if start is None:
        raise ValueError(
            f"{log_path}: no 'Per MPI rank memory allocation' marker found"
        )

    # LAMMPS emits CamelCase column names (`Step Temp PotEng KinEng TotEng
    # Press`). Normalise to lowercase so checks.yaml can address them with
    # stable identifiers.
    header_tokens = [tok.lower() for tok in lines[start].split()]
    rows: list[list[float]] = []
    for line in lines[start + 1 :]:
        if line.startswith("Loop time of"):
            break
        tokens = line.split()
        if not tokens:
            continue
        try:
            rows.append([float(t) for t in tokens])
        except ValueError:
            break
    return header_tokens, rows


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b))
    if denom == 0.0:
        return 0.0
    return abs(a - b) / denom


def compare_columns(
    tdmd_cols: list[str],
    tdmd_rows: list[list[float]],
    lmp_cols: list[str],
    lmp_rows: list[list[float]],
    column_map: Mapping[str, str],
    checks: Iterable[ColumnCheck],
    name: str,
) -> DiffReport:
    """Compare each requested column between TDMD and LAMMPS thermo tables.

    * ``column_map[tdmd_name] == lammps_name`` for columns whose names differ
      between the two stacks.
    * The ``step`` column is used as a row key — only rows whose step values
      match bit-for-bit are compared (this catches thermo-frequency drift).
    * The LAMMPS ``press`` column is converted from bar to eV/Å³ before
      comparison so the numeric ranges on both sides line up.
    """
    tdmd_idx = {name: i for i, name in enumerate(tdmd_cols)}
    lmp_idx = {name: i for i, name in enumerate(lmp_cols)}
    if "step" not in tdmd_idx or "step" not in lmp_idx:
        raise ValueError("both thermo tables must carry a 'step' column")
    t_step = tdmd_idx["step"]
    l_step = lmp_idx["step"]

    # Intersect steps (both sides emit thermo at every 10th step; integer equality).
    tdmd_by_step = {int(row[t_step]): row for row in tdmd_rows}
    lmp_by_step = {int(row[l_step]): row for row in lmp_rows}
    common_steps = sorted(set(tdmd_by_step) & set(lmp_by_step))
    if not common_steps:
        return DiffReport(
            name=name,
            passed=False,
            column_diffs=[],
            note="no overlapping thermo steps between TDMD and LAMMPS logs",
        )

    diffs: list[ColumnDiff] = []
    all_passed = True
    for check in checks:
        t_name = check.column
        l_name = column_map.get(t_name, t_name)
        if t_name not in tdmd_idx:
            diffs.append(
                ColumnDiff(
                    column=t_name,
                    max_abs_diff=float("inf"),
                    max_rel_diff=float("inf"),
                    at_row=-1,
                    at_step=-1,
                    threshold=check.threshold_rel,
                    passed=False,
                )
            )
            all_passed = False
            continue
        if l_name not in lmp_idx:
            diffs.append(
                ColumnDiff(
                    column=t_name,
                    max_abs_diff=float("inf"),
                    max_rel_diff=float("inf"),
                    at_row=-1,
                    at_step=-1,
                    threshold=check.threshold_rel,
                    passed=False,
                )
            )
            all_passed = False
            continue

        max_abs = 0.0
        max_rel = 0.0
        at_row = 0
        at_step = common_steps[0]
        for row_idx, step in enumerate(common_steps):
            a = tdmd_by_step[step][tdmd_idx[t_name]]
            b = lmp_by_step[step][lmp_idx[l_name]]
            if t_name == "press":
                b = b / EV_PER_A3_IN_BAR
            abs_diff = abs(a - b)
            rel_diff = _rel_diff(a, b)
            if rel_diff > max_rel:
                max_rel = rel_diff
                max_abs = abs_diff
                at_row = row_idx
                at_step = step
        passed = max_rel <= check.threshold_rel
        all_passed = all_passed and (passed or not check.required)
        diffs.append(
            ColumnDiff(
                column=t_name,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                at_row=at_row,
                at_step=at_step,
                threshold=check.threshold_rel,
                passed=passed,
            )
        )

    return DiffReport(name=name, passed=all_passed, column_diffs=diffs)


def load_checks(
    checks_path: pathlib.Path, tolerances: Mapping[str, Any]
) -> tuple[dict[str, str], list[ColumnCheck], str]:
    """Parse a benchmark's ``checks.yaml`` into a column map + check list."""
    with checks_path.open() as fh:
        doc = yaml.safe_load(fh)
    name = doc.get("benchmark", checks_path.parent.name)
    column_map = dict(doc.get("thermo_columns", {}))
    checks: list[ColumnCheck] = []
    for entry in doc.get("checks", []):
        if entry.get("type") != "column_relative":
            continue
        checks.append(
            ColumnCheck(
                name=entry["name"],
                column=entry["column"],
                threshold_rel=resolve_threshold(tolerances, entry["threshold_path"]),
                required=bool(entry.get("required", True)),
            )
        )
    return column_map, checks, name


# ---------------------------------------------------------------------------
# Dump file parsing + per-atom forces comparison (T2.8)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DumpFrame:
    """A single LAMMPS-style atom dump frame.

    ``rows`` is keyed by atom id so the consumer can align two frames without
    assuming the file order is identical — LAMMPS reorders atoms during
    migration, and TDMD (after T2.8) sorts by id at emit time, but the
    compare path accepts either ordering.
    """

    timestep: int
    columns: list[str]
    rows: dict[int, list[float]]


def parse_dump_file(path: pathlib.Path) -> DumpFrame:
    """Parse a LAMMPS ``dump custom`` text file (single frame).

    Format (matches ``dump_custom.cpp::header_item``)::

        ITEM: TIMESTEP
        <int>
        ITEM: NUMBER OF ATOMS
        <int>
        ITEM: BOX BOUNDS pp pp pp
        <xlo> <xhi>
        <ylo> <yhi>
        <zlo> <zhi>
        ITEM: ATOMS id type x y z fx fy fz
        <id> <type> <x> <y> <z> <fx> <fy> <fz>
        ...

    Only one frame per file is supported (M2 final-step dump). Multi-frame
    trajectory parsing is deferred to M3. Extra atom-block columns beyond the
    baseline eight are preserved in ``columns`` but the ``id`` column is
    mandatory — without it rows cannot be keyed for alignment.
    """
    with path.open() as fh:
        lines = [line.rstrip("\n") for line in fh]

    i = 0
    timestep = -1
    n_declared = -1
    columns: list[str] = []
    rows: dict[int, list[float]] = {}

    while i < len(lines):
        line = lines[i]
        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
        elif line == "ITEM: NUMBER OF ATOMS":
            n_declared = int(lines[i + 1].strip())
            i += 2
        elif line.startswith("ITEM: BOX BOUNDS"):
            # Three rows follow; the comparator does not use them (forces +
            # positions travel with the atom rows), so skip over them.
            i += 4
        elif line.startswith("ITEM: ATOMS"):
            columns = line[len("ITEM: ATOMS ") :].split()
            if "id" not in columns:
                raise ValueError(f"{path}: dump missing mandatory 'id' column")
            id_col = columns.index("id")
            i += 1
            while i < len(lines):
                body = lines[i].strip()
                if not body:
                    i += 1
                    continue
                if body.startswith("ITEM:"):
                    break
                tokens = body.split()
                if len(tokens) != len(columns):
                    raise ValueError(
                        f"{path}: malformed atom row at line ~{i + 1}: "
                        f"got {len(tokens)} tokens, expected {len(columns)}"
                    )
                atom_id = int(float(tokens[id_col]))
                rows[atom_id] = [float(t) for t in tokens]
                i += 1
        else:
            i += 1

    if timestep < 0:
        raise ValueError(f"{path}: no 'ITEM: TIMESTEP' header found")
    if n_declared < 0:
        raise ValueError(f"{path}: no 'ITEM: NUMBER OF ATOMS' header found")
    if not columns:
        raise ValueError(f"{path}: no 'ITEM: ATOMS' block found")
    if len(rows) != n_declared:
        raise ValueError(
            f"{path}: declared {n_declared} atoms but parsed {len(rows)} rows"
        )
    return DumpFrame(timestep=timestep, columns=columns, rows=rows)


@dataclasses.dataclass
class ForcesDiff:
    """Summary of a forces-vs-forces comparison between two dump frames."""

    n_atoms: int
    max_abs_diff: float
    max_rel_diff: float
    at_atom_id: int
    at_component: str  # 'fx' | 'fy' | 'fz'
    threshold: float
    passed: bool


def compare_forces(tdmd: DumpFrame, lmp: DumpFrame, threshold_rel: float) -> ForcesDiff:
    """Align two dump frames by atom id and diff per-component forces.

    * Atoms present in one frame but not the other are a hard error — we
      never silently drop atoms. (LAMMPS migration can shuffle the file
      order but never drops atoms in the M2 NVE setting.)
    * ``threshold_rel`` is evaluated per component; the worst component-
      atom pair drives the verdict. Absolute residuals are reported alongside
      so the caller can see how close small-force components came to the
      relative floor (where ``|F|→0`` inflates the rel ratio).
    * Residuals are reported at whichever atom id carried the worst
      ratio so a bisect is immediately actionable.
    """
    t_cols = {name: idx for idx, name in enumerate(tdmd.columns)}
    l_cols = {name: idx for idx, name in enumerate(lmp.columns)}
    for name in ("fx", "fy", "fz"):
        if name not in t_cols:
            raise ValueError(f"TDMD dump missing force column '{name}'")
        if name not in l_cols:
            raise ValueError(f"LAMMPS dump missing force column '{name}'")

    t_ids = set(tdmd.rows.keys())
    l_ids = set(lmp.rows.keys())
    if t_ids != l_ids:
        missing = t_ids.symmetric_difference(l_ids)
        raise ValueError(
            f"atom-id sets disagree between dumps; symmetric diff size={len(missing)}"
        )

    max_abs = 0.0
    max_rel = 0.0
    at_id = 0
    at_comp = "fx"
    for atom_id in sorted(t_ids):
        t_row = tdmd.rows[atom_id]
        l_row = lmp.rows[atom_id]
        for comp in ("fx", "fy", "fz"):
            a = t_row[t_cols[comp]]
            b = l_row[l_cols[comp]]
            abs_d = abs(a - b)
            denom = max(abs(a), abs(b))
            rel_d = abs_d / denom if denom > 0.0 else 0.0
            if rel_d > max_rel:
                max_rel = rel_d
                max_abs = abs_d
                at_id = atom_id
                at_comp = comp
    passed = max_rel <= threshold_rel
    return ForcesDiff(
        n_atoms=len(t_ids),
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        at_atom_id=at_id,
        at_component=at_comp,
        threshold=threshold_rel,
        passed=passed,
    )
