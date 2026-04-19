"""Strong-scaling probe runner — T7 benchmark driver.

Entry: :meth:`ScalingRunner.run`. Loads checks.yaml, iterates
``ranks_to_probe``, launches ``mpirun -np N tdmd run`` per N with
``zoning.subdomains: [N, 1, 1]`` injected into a workdir-local config
copy, parses telemetry JSON, computes strong-scaling efficiency,
and emits a :class:`ScalingReport`.

The launcher is swappable (``RunnerConfig.launch_fn``) so the pytest
suite mocks it without spawning a process. Telemetry parsing is
shared with ``anchor_test_runner`` (single-line JSONL today, same
``total_wall_sec`` field).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import subprocess
import sys
from typing import Callable, Mapping, Sequence

import yaml


STATUS_GREEN = "GREEN"
STATUS_YELLOW = "YELLOW"
STATUS_RED = "RED"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunnerConfig:
    """Paths + knobs handed to :class:`ScalingRunner`.

    Path-like fields are absolute by the time the dataclass reaches
    the runner; the CLI is responsible for resolving / defaulting.
    """

    benchmark_dir: pathlib.Path  # e.g. verify/benchmarks/t7_mixed_scaling
    tdmd_bin: pathlib.Path
    mpirun_bin: pathlib.Path
    output_report_path: pathlib.Path
    workdir: pathlib.Path  # scratch dir; auto-created if missing

    # Override checks.yaml::ranks_to_probe. None ⇒ use checks.yaml.
    ranks_override: list[int] | None = None
    # Per-mpirun timeout. Default 30 minutes per probe — Pattern 2 131k
    # atom 100-step on a dev GPU should finish in well under 5 minutes.
    per_run_timeout_seconds: float = 1800.0
    # When True, runner does not regenerate setup.data even if missing —
    # fails instead. Tests flip this so they don't shell out.
    skip_setup_regen: bool = False
    # Optional baseline thermo file for the N=1 byte-exact regression
    # gate. None ⇒ skip the gate (still emit the comparison row).
    baseline_thermo_path: pathlib.Path | None = None
    # Swappable launcher — None ⇒ real mpirun. Smoke tests substitute a
    # mock that returns synthesised telemetry without spawning a process.
    launch_fn: "LaunchFn | None" = None
    # Override checks.yaml::backend (default "gpu" if absent).
    backend_override: str | None = None


@dataclasses.dataclass
class ScalingProbePoint:
    """One (n_procs, measured) data point in the strong-scaling curve."""

    n_procs: int
    measured_wall_seconds: float
    measured_n_steps: int
    measured_steps_per_sec: float
    # Strong-scaling efficiency (0..100+). 1.0 at the anchor by construction.
    measured_efficiency_pct: float
    # The active gate row from checks.yaml (`single_node` / `two_node` / None).
    gate_name: str | None
    gate_pct: float | None
    gate_passed: bool
    status: str  # GREEN / YELLOW / RED
    # Anchor info — useful for postmortem when the curve is shaped wrong.
    anchor_n: int
    anchor_steps_per_sec: float


@dataclasses.dataclass
class ScalingReport:
    """Aggregate output of one harness invocation."""

    points: list[ScalingProbePoint]
    overall_status: str  # GREEN / YELLOW / RED
    overall_passed: bool
    failure_mode: str | None
    benchmark_directory: str
    checks_yaml_path: str
    backend: str
    pattern1_baseline_byte_exact: bool | None  # None = gate not configured
    pattern1_baseline_diff_byte: int | None  # -1 = identical; None = not run
    report_timestamp_utc: str
    wall_clock_minutes: float
    normalization_log: list[str]

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status,
            "overall_passed": self.overall_passed,
            "failure_mode": self.failure_mode,
            "benchmark_directory": self.benchmark_directory,
            "checks_yaml_path": self.checks_yaml_path,
            "backend": self.backend,
            "pattern1_baseline_byte_exact": self.pattern1_baseline_byte_exact,
            "pattern1_baseline_diff_byte": self.pattern1_baseline_diff_byte,
            "report_timestamp_utc": self.report_timestamp_utc,
            "wall_clock_minutes": round(self.wall_clock_minutes, 4),
            "normalization_log": list(self.normalization_log),
            "points": [dataclasses.asdict(p) for p in self.points],
        }


# Launcher signature — (runner, n_procs, workdir, thermo_path) → telemetry dict.
LaunchFn = Callable[["ScalingRunner", int, pathlib.Path, pathlib.Path], dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_checks_yaml(path: pathlib.Path) -> Mapping[str, object]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _read_config_n_steps(config_path: pathlib.Path) -> int:
    with config_path.open() as fh:
        data = yaml.safe_load(fh)
    try:
        return int(data["run"]["n_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"config {config_path} missing run.n_steps") from exc


def _parse_telemetry_jsonl(path: pathlib.Path) -> dict:
    text = path.read_text().strip()
    if not text:
        raise RuntimeError(f"telemetry file {path} was empty")
    return json.loads(text.splitlines()[0])


def _write_augmented_config(
    source_config_path: pathlib.Path,
    dest_config_path: pathlib.Path,
    n_subdomains_x: int,
) -> None:
    """Write a config copy with ``zoning.subdomains: [N, 1, 1]`` and absolute
    paths so the augmented file is launchable from anywhere.

    Mirrors ``anchor_test_runner._write_augmented_config`` — keep behaviour
    identical so cross-harness diffs surface real differences, not config-
    rewriting drift.
    """
    with source_config_path.open() as fh:
        data = yaml.safe_load(fh) or {}
    source_dir = source_config_path.parent

    atoms = data.setdefault("atoms", {})
    atoms_path = atoms.get("path")
    if atoms_path and not pathlib.Path(atoms_path).is_absolute():
        atoms["path"] = str((source_dir / atoms_path).resolve())

    potential = data.get("potential", {})
    params = potential.get("params", {}) if isinstance(potential, dict) else {}
    eam_file = params.get("file") if isinstance(params, dict) else None
    if eam_file and not pathlib.Path(eam_file).is_absolute():
        params["file"] = str((source_dir / eam_file).resolve())

    zoning = data.setdefault("zoning", {})
    zoning["subdomains"] = [int(n_subdomains_x), 1, 1]

    dest_config_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_config_path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _resolve_gate(
    n: int, gates: Mapping[str, Mapping[str, object]]
) -> tuple[str | None, float | None]:
    """Find the (name, pct) pair from ``checks.yaml::efficiency_gates`` whose
    ``[min_rank_inclusive, max_rank_inclusive]`` window contains ``n``. Returns
    (None, None) when no gate is configured for ``n`` (e.g. N=1 anchor).
    """
    for name, row in gates.items():
        lo = int(row.get("min_rank_inclusive", 0))
        hi = int(row.get("max_rank_inclusive", 10**9))
        if lo <= n <= hi:
            return name, float(row.get("gate_pct", 0.0))
    return None, None


def _first_byte_diff(a: bytes, b: bytes) -> int:
    """Same semantics as ``anchor_test_runner._first_byte_diff``."""
    if a == b:
        return -1
    shorter = min(len(a), len(b))
    for i in range(shorter):
        if a[i] != b[i]:
            return i
    return shorter


# ---------------------------------------------------------------------------
# Real mpirun launcher
# ---------------------------------------------------------------------------


def _launch_tdmd_mpirun(
    runner: "ScalingRunner",
    n_procs: int,
    per_rank_workdir: pathlib.Path,
    thermo_path: pathlib.Path,
) -> dict:
    """Spawn ``mpirun -np N tdmd run`` and return parsed telemetry JSON.

    Writes augmented config (with subdomains: [N,1,1]) under
    ``per_rank_workdir/config.yaml``, captures thermo + telemetry, returns
    parsed telemetry. Raises RuntimeError on any subprocess / IO failure.
    """
    cfg = runner.config
    per_rank_workdir.mkdir(parents=True, exist_ok=True)
    telemetry_path = per_rank_workdir / "telemetry.jsonl"
    augmented_config = per_rank_workdir / "config.yaml"

    source_config = cfg.benchmark_dir / "config.yaml"
    if not source_config.is_file():
        raise FileNotFoundError(f"missing {source_config}")
    _write_augmented_config(source_config, augmented_config, n_procs)

    cmd = [
        str(cfg.mpirun_bin),
        "-np",
        str(n_procs),
        str(cfg.tdmd_bin),
        "run",
        "--quiet",
        "--thermo",
        str(thermo_path),
        "--telemetry-jsonl",
        str(telemetry_path),
        str(augmented_config),
    ]
    runner.log(f"launching (N={n_procs}): {' '.join(cmd)}")
    completed = subprocess.run(
        cmd,
        cwd=per_rank_workdir,
        capture_output=True,
        text=True,
        timeout=cfg.per_run_timeout_seconds,
        check=False,
    )
    (per_rank_workdir / "tdmd.stdout").write_text(completed.stdout)
    (per_rank_workdir / "tdmd.stderr").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"tdmd run (N={n_procs}) exited with code {completed.returncode}; "
            f"see {per_rank_workdir}/tdmd.stderr"
        )
    if not telemetry_path.is_file():
        raise RuntimeError(
            f"tdmd run (N={n_procs}) produced no telemetry at {telemetry_path}"
        )
    if not thermo_path.is_file():
        raise RuntimeError(
            f"tdmd run (N={n_procs}) produced no thermo at {thermo_path}"
        )
    return _parse_telemetry_jsonl(telemetry_path)


# ---------------------------------------------------------------------------
# Setup.data lazy regen
# ---------------------------------------------------------------------------


def _regen_setup_data(runner: "ScalingRunner") -> None:
    """Invoke ``generate_setup.py`` if ``setup.data`` is missing.

    Resolves the data path from the config's ``atoms.path`` field so the
    runner does not hard-code the ``verify/data/t7_mixed_scaling/`` location.
    """
    cfg = runner.config
    config_path = cfg.benchmark_dir / "config.yaml"
    if not config_path.is_file():
        return
    with config_path.open() as fh:
        data = yaml.safe_load(fh) or {}
    atoms_path = (data.get("atoms") or {}).get("path")
    if not atoms_path:
        return
    abs_data = pathlib.Path(atoms_path)
    if not abs_data.is_absolute():
        abs_data = (cfg.benchmark_dir / atoms_path).resolve()
    if abs_data.is_file():
        return
    if cfg.skip_setup_regen:
        raise RuntimeError(
            f"setup.data missing at {abs_data} and runner was configured "
            f"with skip_setup_regen=True"
        )
    regen = cfg.benchmark_dir / "generate_setup.py"
    if not regen.is_file():
        raise RuntimeError(
            f"setup.data missing at {abs_data} and no generate_setup.py "
            f"alongside the fixture"
        )
    runner.log(f"setup.data missing — regenerating via {regen}")
    completed = subprocess.run(
        [sys.executable, str(regen), "--out", str(abs_data)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"generate_setup.py failed (exit {completed.returncode}):\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ScalingRunner:
    """Drives the T7 strong-scaling sweep and emits :class:`ScalingReport`."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self._log_lines: list[str] = []

    def log(self, line: str) -> None:
        self._log_lines.append(line)

    def run(self) -> ScalingReport:
        cfg = self.config
        start = datetime.datetime.now(tz=datetime.timezone.utc)

        # 1. Lazy regen — only if needed.
        _regen_setup_data(self)

        # 2. Load fixture artefacts.
        checks = _load_checks_yaml(cfg.benchmark_dir / "checks.yaml")
        declared_backend = str(checks.get("backend", "gpu")).lower()
        backend = (cfg.backend_override or declared_backend).lower()
        if backend != "gpu":
            self.log(
                f"warning: T7 benchmark expects backend=gpu; got '{backend}' "
                f"(checks.yaml='{declared_backend}', "
                f"override='{cfg.backend_override}')"
            )

        gates_block = checks.get("efficiency_gates") or {}
        if not isinstance(gates_block, Mapping):
            raise RuntimeError("checks.yaml::efficiency_gates must be a mapping")
        ranks: Sequence[int] = list(
            cfg.ranks_override or checks.get("ranks_to_probe", [])
        )
        if not ranks:
            raise RuntimeError(
                "no ranks_to_probe (checks.yaml or --ranks override required)"
            )
        ranks = sorted({int(x) for x in ranks})
        if 1 not in ranks:
            raise RuntimeError(
                "ranks_to_probe must include 1 — needed as efficiency anchor"
            )

        # 3. Launch sweep.
        launch: LaunchFn = cfg.launch_fn or _launch_tdmd_mpirun
        anchor_steps_per_sec: float | None = None
        anchor_n: int | None = None
        anchor_thermo_path: pathlib.Path | None = None

        points: list[ScalingProbePoint] = []
        for n in ranks:
            per_rank_workdir = cfg.workdir / f"N{n:03d}"
            thermo_path = per_rank_workdir / "thermo.dat"
            try:
                telemetry = launch(self, n, per_rank_workdir, thermo_path)
            except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                self.log(f"launch failed at N={n}: {exc}")
                return self._emit_hard_fail(
                    failure_mode="LAUNCH_FAILURE",
                    points=points,
                    backend=backend,
                    start=start,
                )
            wall = float(telemetry["total_wall_sec"])
            n_steps = _read_config_n_steps(cfg.benchmark_dir / "config.yaml")
            steps_per_sec = (n_steps / wall) if wall > 0 else 0.0

            if anchor_steps_per_sec is None:
                anchor_steps_per_sec = steps_per_sec
                anchor_n = n
                anchor_thermo_path = thermo_path
                self.log(
                    f"efficiency anchor: N={n} → " f"steps_per_sec={steps_per_sec:.4f}"
                )

            # Strong-scaling efficiency. Anchor is N=1 by spec
            # (ranks ordering enforces it above).
            efficiency_pct = (
                100.0 * steps_per_sec * anchor_n / (anchor_steps_per_sec * n)
                if anchor_steps_per_sec > 0
                else 0.0
            )

            gate_name, gate_pct = _resolve_gate(n, gates_block)
            gate_passed: bool
            status: str
            if gate_pct is None:
                # No gate configured — anchor or out-of-range probe. Mark green
                # by convention; the YAML author chose not to gate this point.
                gate_passed = True
                status = STATUS_GREEN
            else:
                gate_passed = efficiency_pct >= gate_pct
                status = STATUS_GREEN if gate_passed else STATUS_RED

            points.append(
                ScalingProbePoint(
                    n_procs=n,
                    measured_wall_seconds=wall,
                    measured_n_steps=n_steps,
                    measured_steps_per_sec=steps_per_sec,
                    measured_efficiency_pct=efficiency_pct,
                    gate_name=gate_name,
                    gate_pct=gate_pct,
                    gate_passed=gate_passed,
                    status=status,
                    anchor_n=anchor_n or 0,
                    anchor_steps_per_sec=anchor_steps_per_sec or 0.0,
                )
            )

        # 4. Pattern 1 byte-exact regression — N=1 thermo vs supplied baseline.
        pattern1_gate_configured = bool(checks.get("pattern1_baseline_byte_exact"))
        pattern1_diff_byte: int | None = None
        pattern1_passed: bool | None = None
        if pattern1_gate_configured and cfg.baseline_thermo_path:
            if anchor_thermo_path is None or not anchor_thermo_path.is_file():
                self.log("Pattern 1 baseline gate requested but N=1 thermo missing")
                pattern1_passed = False
            else:
                anchor_bytes = anchor_thermo_path.read_bytes()
                baseline_bytes = cfg.baseline_thermo_path.read_bytes()
                pattern1_diff_byte = _first_byte_diff(anchor_bytes, baseline_bytes)
                pattern1_passed = pattern1_diff_byte == -1
                self.log(
                    f"Pattern 1 byte-exact: "
                    f"{'PASS' if pattern1_passed else f'FAIL @ byte {pattern1_diff_byte}'}"
                )
        elif pattern1_gate_configured:
            self.log(
                "Pattern 1 baseline gate configured but no --baseline-thermo "
                "supplied — skipping byte-exact comparison"
            )

        # 5. Aggregate verdict.
        any_red = any(p.status == STATUS_RED for p in points) or (
            pattern1_passed is False
        )
        if any_red:
            overall_status = STATUS_RED
            overall_passed = False
            failure_mode = (
                "PATTERN1_BYTE_EXACT_BREAK"
                if pattern1_passed is False
                else "EFFICIENCY_GATE_FAIL"
            )
        else:
            overall_status = STATUS_GREEN
            overall_passed = True
            failure_mode = None

        end = datetime.datetime.now(tz=datetime.timezone.utc)
        return ScalingReport(
            points=points,
            overall_status=overall_status,
            overall_passed=overall_passed,
            failure_mode=failure_mode,
            benchmark_directory=str(cfg.benchmark_dir),
            checks_yaml_path=str(cfg.benchmark_dir / "checks.yaml"),
            backend=backend,
            pattern1_baseline_byte_exact=pattern1_passed,
            pattern1_baseline_diff_byte=pattern1_diff_byte,
            report_timestamp_utc=start.isoformat(timespec="seconds"),
            wall_clock_minutes=(end - start).total_seconds() / 60.0,
            normalization_log=list(self._log_lines),
        )

    # -----------------------------------------------------------------
    def _emit_hard_fail(
        self,
        *,
        failure_mode: str,
        points: list[ScalingProbePoint],
        backend: str,
        start: datetime.datetime,
    ) -> ScalingReport:
        cfg = self.config
        end = datetime.datetime.now(tz=datetime.timezone.utc)
        return ScalingReport(
            points=points,
            overall_status=STATUS_RED,
            overall_passed=False,
            failure_mode=failure_mode,
            benchmark_directory=str(cfg.benchmark_dir),
            checks_yaml_path=str(cfg.benchmark_dir / "checks.yaml"),
            backend=backend,
            pattern1_baseline_byte_exact=None,
            pattern1_baseline_diff_byte=None,
            report_timestamp_utc=start.isoformat(timespec="seconds"),
            wall_clock_minutes=(end - start).total_seconds() / 60.0,
            normalization_log=list(self._log_lines),
        )
