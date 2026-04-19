"""AnchorTestRunner — top-level driver for the T3 benchmark.

Single entry: :meth:`AnchorTestRunner.run`. Each invocation enumerates
``checks.yaml::ranks_to_probe``, calls ``mpirun -np N ./tdmd run`` per
point, parses telemetry, normalises against the hardware probe, and
diffs against ``dissertation_reference_data.csv``.

The driver is structurally split into:
    * :class:`RunnerConfig`   — paths + knobs (CLI-constructed)
    * ``_load_reference_csv`` — CSV ingest
    * ``_load_checks_yaml``   — thresholds + ranks list
    * ``_launch_tdmd_once``   — one ``mpirun`` invocation (swappable for
                                the smoke-test mocked-TDMD path)
    * :meth:`AnchorTestRunner.run` — orchestrator

Only the orchestrator is public.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import pathlib
import shutil
import subprocess
from typing import Callable, Mapping

import yaml

from .hardware_probe import probe
from .report import (
    STATUS_GREEN,
    STATUS_RED,
    STATUS_YELLOW,
    AnchorTestPoint,
    AnchorTestReport,
    HardwareProbeResult,
)

LaunchFn = Callable[["AnchorTestRunner", int, pathlib.Path], dict]


@dataclasses.dataclass
class RunnerConfig:
    """Paths + knobs handed to :class:`AnchorTestRunner`.

    Everything path-like is absolute by the time the dataclass reaches
    the runner; the CLI is responsible for resolving/defaulting.
    """

    # Paths
    benchmark_dir: pathlib.Path  # e.g. verify/benchmarks/t3_al_fcc_large_anchor
    tdmd_bin: pathlib.Path
    mpirun_bin: pathlib.Path
    output_report_path: pathlib.Path
    workdir: pathlib.Path  # scratch dir; auto-created if missing

    # Knobs
    ranks_override: list[int] | None = None  # None → use checks.yaml
    force_probe: bool = False
    # Seconds per mpirun invocation ceiling. Defaults to 30 minutes, same
    # order-of-magnitude as the wall_clock_budget_minutes in checks.yaml.
    per_run_timeout_seconds: float = 1800.0
    # When True, the runner does not regenerate setup.data even if it
    # is missing — fails instead. Tests flip this on so they don't
    # invoke LAMMPS transparently.
    skip_setup_regen: bool = False
    lammps_bin: pathlib.Path | None = None  # required iff regen needed
    # Swappable TDMD launcher — None ⇒ real mpirun. Smoke tests substitute
    # a mocked launcher that returns a synthesised telemetry payload
    # without spawning a process.
    launch_fn: LaunchFn | None = None


# ---------------------------------------------------------------------------
# CSV + YAML ingest helpers
# ---------------------------------------------------------------------------


def _load_reference_csv(path: pathlib.Path) -> dict[int, dict[str, object]]:
    """Return ``{n_procs: row}`` indexed by processor count.

    The CSV is expected to have columns ``n_procs,performance_mdps,
    efficiency_pct,source_figure,note``; extra columns are tolerated and
    round-tripped into the output report so future extensions (e.g.
    confidence intervals) do not need a schema bump.
    """
    out: dict[int, dict[str, object]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(row for row in fh if not row.lstrip().startswith("#"))
        for row in reader:
            try:
                n = int(row["n_procs"])
            except (KeyError, ValueError) as exc:
                raise RuntimeError(
                    f"reference CSV {path} — missing/invalid n_procs column"
                ) from exc
            out[n] = row
    if not out:
        raise RuntimeError(f"reference CSV {path} contained no data rows")
    return out


def _load_checks_yaml(path: pathlib.Path) -> Mapping[str, object]:
    with path.open() as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# TDMD launch + telemetry parsing
# ---------------------------------------------------------------------------


def _parse_telemetry_jsonl(path: pathlib.Path) -> dict:
    """Parse the single-line JSONL that `tdmd run --telemetry-jsonl` emits."""
    text = path.read_text().strip()
    if not text:
        raise RuntimeError(f"telemetry file {path} was empty")
    # Take first line only — Telemetry writes exactly one JSON object today.
    first = text.splitlines()[0]
    return json.loads(first)


def _launch_tdmd_mpirun(
    runner: "AnchorTestRunner", n_procs: int, per_rank_workdir: pathlib.Path
) -> dict:
    """Real MPI launch path. Returns the parsed telemetry JSON.

    The per-rank workdir isolates scratch artefacts (telemetry file, any
    thermo file TDMD may choose to write) so concurrent `ranks_to_probe`
    entries never collide.
    """
    cfg = runner.config
    per_rank_workdir.mkdir(parents=True, exist_ok=True)
    telemetry_path = per_rank_workdir / "telemetry.jsonl"

    # Resolve the config.yaml relative to the benchmark directory; do not
    # mutate the on-disk copy — TDMD's path-resolution rewrites
    # atoms.path based on the config's parent directory, and we want the
    # committed fixture untouched.
    config_path = cfg.benchmark_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")

    cmd = [
        str(cfg.mpirun_bin),
        "-np",
        str(n_procs),
        str(cfg.tdmd_bin),
        "run",
        "--quiet",
        "--telemetry-jsonl",
        str(telemetry_path),
        str(config_path),
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
            f"tdmd run (N={n_procs}) completed but did not write telemetry to {telemetry_path}"
        )
    return _parse_telemetry_jsonl(telemetry_path)


# ---------------------------------------------------------------------------
# Setup.data regeneration — invokes ../../data/.../regen_setup.sh when
# the file is missing (T1 precedent; see T5.10 README).
# ---------------------------------------------------------------------------


def _regen_setup_data(runner: "AnchorTestRunner") -> None:
    cfg = runner.config
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    data_dir = repo_root / "verify" / "data" / "t3_al_fcc_large_anchor"
    setup_data = data_dir / "setup.data"
    if setup_data.is_file():
        return
    if cfg.skip_setup_regen:
        raise RuntimeError(
            f"setup.data missing at {setup_data} and runner was configured "
            f"with skip_setup_regen=True"
        )
    if cfg.lammps_bin is None:
        raise RuntimeError(
            f"setup.data missing at {setup_data}; pass --lammps <path> or set LAMMPS_BIN"
        )
    regen = data_dir / "regen_setup.sh"
    runner.log(f"setup.data missing — regenerating via {regen}")
    completed = subprocess.run(
        [str(regen), "--lammps", str(cfg.lammps_bin)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"regen_setup.sh failed (exit {completed.returncode}):\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AnchorTestRunner:
    """Drives the T3 benchmark and emits :class:`AnchorTestReport`.

    Never depends on LAMMPS for comparison (T3 is a dissertation match,
    not an oracle diff) — LAMMPS is only invoked transparently if
    ``setup.data`` needs regeneration on a fresh workspace.
    """

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self._log_lines: list[str] = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(self) -> AnchorTestReport:
        cfg = self.config
        start = datetime.datetime.now(tz=datetime.timezone.utc)

        # 1. Setup.data regeneration — only when needed, per T1 precedent.
        _regen_setup_data(self)

        # 2. Load fixture artefacts.
        checks = _load_checks_yaml(cfg.benchmark_dir / "checks.yaml")
        reference = _load_reference_csv(
            cfg.benchmark_dir / "dissertation_reference_data.csv"
        )
        ranks = [int(x) for x in cfg.ranks_override or checks.get("ranks_to_probe", [])]
        if not ranks:
            raise RuntimeError("no ranks_to_probe provided (checks.yaml or override)")
        eff_tol = float(checks["dissertation_comparison"]["efficiency_relative"])
        abs_tol = float(
            checks["dissertation_comparison"]["absolute_performance_relative"]
        )

        # 3. Hardware probe (cached unless --force-probe).
        hw = probe(force_probe=cfg.force_probe)
        self.log(
            f"hardware probe: local_gflops={hw.local_gflops:.4f}  "
            f"baseline_gflops={hw.baseline_gflops:.4f}  ratio={hw.ghz_flops_ratio:.4f}  "
            f"cached={hw.cached}"
        )
        if hw.ghz_flops_ratio < 1.0:
            return self._emit_hard_fail(
                failure_mode="HARDWARE_MISMATCH",
                points=[],
                hw=hw,
                ranks=ranks,
                reason=(
                    f"hw_ratio={hw.ghz_flops_ratio:.3f} < 1.0 — current machine is "
                    f"slower than 2007 Harpertown baseline; comparison undefined"
                ),
                start=start,
                checks_path=cfg.benchmark_dir / "checks.yaml",
            )

        # 4. Sequential point-by-point launches.
        launch: LaunchFn = cfg.launch_fn or _launch_tdmd_mpirun
        measured_serial_mdps: float | None = None
        points: list[AnchorTestPoint] = []
        for n in ranks:
            if n not in reference:
                raise RuntimeError(
                    f"reference CSV has no row for n_procs={n}; cannot compare. "
                    f"Available: {sorted(reference)}"
                )
            per_rank_workdir = cfg.workdir / f"N{n:03d}"
            telemetry = launch(self, n, per_rank_workdir)
            total_wall_sec = float(telemetry["total_wall_sec"])
            n_steps = _read_config_n_steps(cfg.benchmark_dir / "config.yaml")
            steps_per_sec = (n_steps / total_wall_sec) if total_wall_sec > 0 else 0.0
            # Convert to "mdps" = million-steps-per-day (matches the CSV column
            # naming; 1e6 divisor cancels against the *1e6 in the CSV).
            measured_mdps = steps_per_sec * 86400.0 / 1.0e6

            # Efficiency anchored on the single-rank run if ranks_to_probe
            # starts with 1; otherwise anchor on the smallest N actually
            # launched and document it in the log.
            #
            # Strong-scaling efficiency: E(N) = rate(N) * anchor_n /
            # (rate(anchor_n) * N). Perfect scaling ⇒ E=1 (100%). At the
            # anchor point E≡1 by construction.
            if measured_serial_mdps is None:
                measured_serial_mdps = measured_mdps
                anchor_n = n
                self.log(
                    f"efficiency anchor: N={anchor_n} → measured_mdps={measured_mdps:.4f}"
                )
            measured_eff = (
                100.0 * measured_mdps * anchor_n / (measured_serial_mdps * n)
                if measured_mdps > 0 and measured_serial_mdps is not None
                else 0.0
            )

            ref_row = reference[n]
            ref_mdps = float(ref_row["performance_mdps"])
            ref_eff = float(ref_row["efficiency_pct"])
            normalised_ref_mdps = ref_mdps * hw.ghz_flops_ratio

            eff_rel_err = abs(measured_eff - ref_eff) / ref_eff if ref_eff > 0 else 0.0
            abs_rel_err = (
                abs(measured_mdps - normalised_ref_mdps) / normalised_ref_mdps
                if normalised_ref_mdps > 0
                else 0.0
            )

            eff_passed = eff_rel_err <= eff_tol
            abs_warned = abs_rel_err > abs_tol

            if not eff_passed:
                status = STATUS_RED
            elif abs_warned:
                status = STATUS_YELLOW
            else:
                status = STATUS_GREEN

            points.append(
                AnchorTestPoint(
                    n_procs=n,
                    measured_performance_mdps=measured_mdps,
                    measured_efficiency_pct=measured_eff,
                    measured_wall_seconds=total_wall_sec,
                    measured_n_steps=n_steps,
                    reference_performance_mdps=ref_mdps,
                    reference_efficiency_pct=ref_eff,
                    reference_source_figure=str(ref_row.get("source_figure", "")),
                    reference_note=str(ref_row.get("note", "")),
                    normalised_reference_performance_mdps=normalised_ref_mdps,
                    efficiency_relative_error=eff_rel_err,
                    absolute_performance_relative_error=abs_rel_err,
                    efficiency_tolerance=eff_tol,
                    absolute_performance_tolerance=abs_tol,
                    efficiency_passed=eff_passed,
                    absolute_performance_warned=abs_warned,
                    status=status,
                )
            )

        overall_passed = all(p.efficiency_passed for p in points)
        any_warning = any(p.absolute_performance_warned for p in points)
        if not overall_passed:
            overall_status = STATUS_RED
            failure_mode = (
                "REF_DATA_STALE"
                if _csv_is_placeholder(
                    cfg.benchmark_dir / "dissertation_reference_data.csv"
                )
                else "DETERMINISM_BREAK"
            )
        elif any_warning:
            overall_status = STATUS_YELLOW
            failure_mode = None
        else:
            overall_status = STATUS_GREEN
            failure_mode = None

        end = datetime.datetime.now(tz=datetime.timezone.utc)
        report = AnchorTestReport(
            points=points,
            overall_passed=overall_passed,
            overall_status=overall_status,
            any_warning=any_warning,
            dissertation_reference_commit=_git_revision_of(
                cfg.benchmark_dir / "dissertation_reference_data.csv"
            ),
            tdmd_commit=_git_revision_of(cfg.tdmd_bin),
            benchmark_directory=str(cfg.benchmark_dir),
            checks_yaml_path=str(cfg.benchmark_dir / "checks.yaml"),
            hardware=hw,
            report_timestamp_utc=start.isoformat(timespec="seconds"),
            wall_clock_minutes=(end - start).total_seconds() / 60.0,
            normalization_log=list(self._log_lines),
            failure_mode=failure_mode,
        )
        return report

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def log(self, line: str) -> None:
        self._log_lines.append(line)

    def _emit_hard_fail(
        self,
        *,
        failure_mode: str,
        points: list[AnchorTestPoint],
        hw: HardwareProbeResult,
        ranks: list[int],
        reason: str,
        start: datetime.datetime,
        checks_path: pathlib.Path,
    ) -> AnchorTestReport:
        self.log(f"HARD FAIL ({failure_mode}): {reason}")
        end = datetime.datetime.now(tz=datetime.timezone.utc)
        cfg = self.config
        return AnchorTestReport(
            points=points,
            overall_passed=False,
            overall_status=STATUS_RED,
            any_warning=False,
            dissertation_reference_commit=_git_revision_of(
                cfg.benchmark_dir / "dissertation_reference_data.csv"
            ),
            tdmd_commit=_git_revision_of(cfg.tdmd_bin),
            benchmark_directory=str(cfg.benchmark_dir),
            checks_yaml_path=str(checks_path),
            hardware=hw,
            report_timestamp_utc=start.isoformat(timespec="seconds"),
            wall_clock_minutes=(end - start).total_seconds() / 60.0,
            normalization_log=list(self._log_lines),
            failure_mode=failure_mode,
        )


# ---------------------------------------------------------------------------
# Small module-level helpers kept free-standing so tests can stub them
# ---------------------------------------------------------------------------


def _read_config_n_steps(config_path: pathlib.Path) -> int:
    """Extract ``run.n_steps`` from the YAML config without loading TDMD."""
    with config_path.open() as fh:
        data = yaml.safe_load(fh)
    try:
        return int(data["run"]["n_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"config {config_path} missing run.n_steps") from exc


def _git_revision_of(path: pathlib.Path) -> str:
    """Return the git SHA of the current HEAD affecting ``path``.

    On detached HEADs / missing git metadata the string 'unknown' is
    returned rather than raising — the anchor-test report stays emit-able
    even on a downloaded tarball.
    """
    if not shutil.which("git"):
        return "unknown"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path.parent if path.exists() else pathlib.Path.cwd()),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _csv_is_placeholder(path: pathlib.Path) -> bool:
    """Heuristic: if the CSV comments include the string 'placeholder' or
    'R-M5-8', its source values have not been extracted yet. See the
    CSV header in T5.10.
    """
    try:
        with path.open() as fh:
            head = fh.read(2048)
    except OSError:
        return False
    return "placeholder" in head.lower() or "r-m5-8" in head.lower()
