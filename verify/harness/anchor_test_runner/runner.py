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
    GpuGateResult,
    HardwareProbeResult,
)

LaunchFn = Callable[["AnchorTestRunner", int, pathlib.Path], dict]
# GPU launcher signature — takes (runner, n_procs, workdir, backend,
# thermo_path) and returns parsed telemetry JSON. ``backend`` is "cpu" or
# "gpu"; ``thermo_path`` is where TDMD's ``--thermo`` flag writes the
# thermo table so the runner can byte-compare CPU vs GPU streams.
#
# T7.12 widened to ``Callable[..., dict]`` so the gate-3 efficiency probe
# can pass the optional ``subdomains_xyz`` kwarg without breaking gates 1/2
# call sites that still use the strict 5-positional form. Mock launchers
# in tests accept ``**kwargs`` to absorb unknown args.
GpuLaunchFn = Callable[..., dict]


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
    # T6.10a — separate swappable launcher for the GPU two-level anchor path.
    # The GPU path needs a backend-aware launcher because it runs the same
    # fixture twice (CPU backend then GPU backend) and byte-compares the
    # thermo streams. The CPU ``launch_fn`` signature does not carry a
    # backend argument, so we keep a second slot for the GPU smoke tests
    # to mock without touching the CPU contract.
    gpu_launch_fn: "GpuLaunchFn | None" = None
    # T6.10a — override ``checks.yaml::backend`` from the CLI. None ⇒ use
    # whatever the checks.yaml declares (cpu if absent). Accepted values:
    # "cpu" or "gpu".
    backend_override: str | None = None


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
        reader = csv.DictReader(
            row for row in fh if row.strip() and not row.lstrip().startswith("#")
        )
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


def _write_augmented_config(
    source_config_path: pathlib.Path,
    dest_config_path: pathlib.Path,
    backend: str,
    *,
    subdomains_xyz: list[int] | None = None,
) -> None:
    """Write a copy of ``source_config_path`` into ``dest_config_path`` with
    ``runtime.backend`` set to ``backend``.

    The committed fixture config does not carry ``runtime.backend`` so the
    same file is usable by both CPU and GPU launches. The runner mutates
    a tmp copy (never the fixture itself) just before invoking TDMD.

    Relative paths in ``atoms.path`` and ``potential.params.file`` resolve
    against the **config file's directory** per SimulationEngine's
    ``resolve_atoms_path`` contract. The augmented config lives in a
    different directory (the runner workdir) so we rewrite relative
    inputs to absolute paths here — otherwise TDMD would look for the
    inputs next to the augmented file.

    When ``subdomains_xyz`` is supplied, ``zoning.subdomains: [Nx, Ny, Nz]``
    is injected so the harness can probe Pattern 2 strong-scaling without
    committing N-specific fixtures (T7.12 EAM-substitute efficiency probe).
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

    data.setdefault("runtime", {})["backend"] = backend
    if subdomains_xyz is not None:
        if len(subdomains_xyz) != 3:
            raise RuntimeError(
                f"subdomains_xyz must be a 3-element [Nx,Ny,Nz] list; "
                f"got {subdomains_xyz!r}"
            )
        data.setdefault("zoning", {})["subdomains"] = [int(x) for x in subdomains_xyz]
    dest_config_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_config_path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _launch_tdmd_with_backend(
    runner: "AnchorTestRunner",
    n_procs: int,
    workdir: pathlib.Path,
    backend: str,
    thermo_path: pathlib.Path,
    *,
    subdomains_xyz: list[int] | None = None,
) -> dict:
    """GPU anchor launcher. Real MPI path with ``--thermo`` capture.

    Writes an augmented config into ``workdir/config.yaml`` (with
    ``runtime.backend`` injected + relative paths resolved to absolute)
    then invokes tdmd via mpirun. Returns parsed telemetry JSON; the
    thermo trace is captured to ``thermo_path`` by TDMD's ``--thermo``
    flag and is byte-compared by the runner separately.

    ``subdomains_xyz`` (T7.12) — when supplied, injects
    ``zoning.subdomains: [Nx, Ny, Nz]`` into the augmented config so the
    Pattern 2 GPU efficiency probe can vary subdomain count without
    committing N-specific fixtures.
    """
    cfg = runner.config
    workdir.mkdir(parents=True, exist_ok=True)
    telemetry_path = workdir / "telemetry.jsonl"
    augmented_config = workdir / "config.yaml"

    source_config = cfg.benchmark_dir / "config.yaml"
    if not source_config.is_file():
        raise FileNotFoundError(f"missing {source_config}")
    _write_augmented_config(
        source_config, augmented_config, backend, subdomains_xyz=subdomains_xyz
    )

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
    runner.log(f"launching (backend={backend}, N={n_procs}): {' '.join(cmd)}")
    completed = subprocess.run(
        cmd,
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=cfg.per_run_timeout_seconds,
        check=False,
    )
    (workdir / "tdmd.stdout").write_text(completed.stdout)
    (workdir / "tdmd.stderr").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"tdmd run (backend={backend}, N={n_procs}) exited with code "
            f"{completed.returncode}; see {workdir}/tdmd.stderr"
        )
    if not telemetry_path.is_file():
        raise RuntimeError(
            f"tdmd run (backend={backend}, N={n_procs}) produced no telemetry at {telemetry_path}"
        )
    if not thermo_path.is_file():
        raise RuntimeError(
            f"tdmd run (backend={backend}, N={n_procs}) produced no thermo at {thermo_path}"
        )
    return _parse_telemetry_jsonl(telemetry_path)


def _probe_gpu_model(hardware_normalization_script: pathlib.Path) -> dict:
    """Invoke the GPU hardware probe script and return its JSON payload.

    The T6.10a stub never hard-fails — it returns ``gpu_model: None`` when
    nvidia-smi is not visible. The caller (``_run_gpu_two_level``) decides
    whether to proceed based on what the downstream TDMD launch reports.
    """
    if not hardware_normalization_script.is_file():
        raise FileNotFoundError(
            f"missing GPU hardware normalisation script at "
            f"{hardware_normalization_script}"
        )
    cmd = [
        "python3",
        str(hardware_normalization_script),
        "--json",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"GPU hardware probe exited with code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"GPU hardware probe did not emit JSON: '{completed.stdout.strip()}'"
        ) from exc


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

        # 0. Dispatch — T6.10a. ``checks.yaml::backend`` selects the CPU T3
        # code path (default) or the T3-gpu two-level path (``backend: gpu``).
        # ``cfg.backend_override`` lets the CLI force a backend without
        # editing the fixture. The CPU path stays untouched when the key
        # is absent (backward compat with T3 CPU fixture).
        checks_preview = _load_checks_yaml(cfg.benchmark_dir / "checks.yaml")
        declared_backend = str(checks_preview.get("backend", "cpu")).lower()
        effective_backend = (cfg.backend_override or declared_backend).lower()
        if effective_backend not in ("cpu", "gpu"):
            raise RuntimeError(
                f"backend must be one of {{cpu, gpu}}; got '{effective_backend}' "
                f"(checks.yaml declared '{declared_backend}', "
                f"override '{cfg.backend_override}')"
            )
        if effective_backend == "gpu":
            return self._run_gpu_two_level(start, checks_preview)

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
    # GPU two-level path (T6.10a)
    # -----------------------------------------------------------------

    def _run_gpu_two_level(
        self, start: datetime.datetime, checks: Mapping[str, object]
    ) -> AnchorTestReport:
        """T3-gpu two-level anchor — CPU≡GPU Reference + MixedFast-within-budget.

        The T6.10a scope ships gates (1) and (2); gate (3) (efficiency curve
        vs dissertation) is deferred to T6.10b. See
        ``verify/benchmarks/t3_al_fcc_large_anchor_gpu/acceptance_criteria.md``
        for the scope reasoning.
        """
        cfg = self.config
        checks_path = cfg.benchmark_dir / "checks.yaml"

        # The GPU anchor does not consume the dissertation CSV (gate 3
        # deferred), so we build a placeholder HardwareProbeResult to keep
        # the AnchorTestReport schema intact. The GPU probe output goes to
        # the normalization_log so the gpu_model / provenance is recoverable.
        probe_script = cfg.benchmark_dir / "hardware_normalization_gpu.py"
        gpu_probe_payload: dict | None = None
        try:
            gpu_probe_payload = _probe_gpu_model(probe_script)
            self.log(
                f"GPU probe: model={gpu_probe_payload.get('gpu_model')!r} "
                f"ratio={gpu_probe_payload.get('gpu_flops_ratio', 'n/a')!r} "
                f"(note: {gpu_probe_payload.get('note', '')})"
            )
        except (FileNotFoundError, RuntimeError) as exc:
            self.log(f"GPU probe failed: {exc}")
        gpu_model_visible = bool(
            gpu_probe_payload and gpu_probe_payload.get("gpu_model")
        )

        hw_placeholder = HardwareProbeResult(
            local_gflops=0.0,
            baseline_gflops=0.0,
            ghz_flops_ratio=1.0,  # GPU stub ratio; unused until T6.10b
            probe_timestamp_utc=start.isoformat(timespec="seconds"),
            cached=False,
        )

        ranks_to_probe = [int(x) for x in checks.get("ranks_to_probe", [1])]
        if ranks_to_probe != [1]:
            self.log(
                f"GPU anchor T6.10a scope is single-rank; checks.yaml declares "
                f"ranks_to_probe={ranks_to_probe} — running the first entry "
                f"only (multi-rank GPU scaling waits on T6.9b / T6.10b)"
            )
        n_procs = ranks_to_probe[0] if ranks_to_probe else 1

        # If the probe couldn't find a GPU, attempt the launch anyway —
        # some environments hide nvidia-smi but still expose CUDA. The real
        # dispositive signal is whether the tdmd launch completes. But if
        # both probe and launch fail, we must report NO_CUDA_DEVICE cleanly.
        launcher: GpuLaunchFn = cfg.gpu_launch_fn or _launch_tdmd_with_backend

        cpu_workdir = cfg.workdir / "gpu_anchor_cpu"
        gpu_workdir = cfg.workdir / "gpu_anchor_gpu"
        cpu_thermo = cpu_workdir / "thermo.dat"
        gpu_thermo = gpu_workdir / "thermo.dat"

        gates: list[GpuGateResult] = []
        overall_failure_mode: str | None = None

        # Gate 1 — CPU ≡ GPU Reference byte-exact thermo.
        gate1_cfg = checks.get("cpu_gpu_reference_bit_exact") or {}
        run_gate1 = bool(gate1_cfg.get("thermo_byte_equal", True))
        if run_gate1:
            try:
                launcher(self, n_procs, cpu_workdir, "cpu", cpu_thermo)
                launcher(self, n_procs, gpu_workdir, "gpu", gpu_thermo)
            except RuntimeError as exc:
                gates.append(
                    GpuGateResult(
                        gate_name="cpu_gpu_reference_bit_exact",
                        passed=False,
                        status=STATUS_RED,
                        detail=(
                            f"launch failed: {exc}"
                            + (
                                ""
                                if gpu_model_visible
                                else " (GPU probe also reported no visible model)"
                            )
                        ),
                    )
                )
                overall_failure_mode = (
                    "NO_CUDA_DEVICE"
                    if not gpu_model_visible
                    else "CPU_GPU_REFERENCE_DIVERGE"
                )
            else:
                cpu_bytes = cpu_thermo.read_bytes()
                gpu_bytes = gpu_thermo.read_bytes()
                passed = cpu_bytes == gpu_bytes
                detail = (
                    f"thermo {len(cpu_bytes)} bytes ≡ {len(gpu_bytes)} bytes"
                    if passed
                    else (
                        f"thermo diverged — CPU {len(cpu_bytes)} bytes vs "
                        f"GPU {len(gpu_bytes)} bytes (first diff at byte "
                        f"{_first_byte_diff(cpu_bytes, gpu_bytes)})"
                    )
                )
                gates.append(
                    GpuGateResult(
                        gate_name="cpu_gpu_reference_bit_exact",
                        passed=passed,
                        status=STATUS_GREEN if passed else STATUS_RED,
                        detail=detail,
                        cpu_thermo_path=str(cpu_thermo),
                        gpu_thermo_path=str(gpu_thermo),
                    )
                )
                if not passed and overall_failure_mode is None:
                    overall_failure_mode = "CPU_GPU_REFERENCE_DIVERGE"

        # Gate 2 — MixedFast within D-M6-8 thresholds, delegated to T6.8a.
        # T6.10a does not re-invoke the C++ differential test — it is part
        # of the pre-push test suite the user runs before calling this
        # harness. We record the delegation in the report so the T6.13
        # smoke consumer has explicit provenance.
        gate2_cfg = checks.get("mixed_fast_vs_reference") or {}
        if gate2_cfg:
            gates.append(
                GpuGateResult(
                    gate_name="mixed_fast_vs_reference",
                    passed=True,
                    status=STATUS_YELLOW,  # advisory — delegated
                    detail=(
                        f"delegated to T6.8a differential "
                        f"(test_eam_mixed_fast_within_threshold). Thresholds: "
                        f"force_linf≤{gate2_cfg.get('force_relative_linf_threshold')}, "
                        f"PE_rel≤{gate2_cfg.get('energy_relative_threshold')}, "
                        f"virial_rel_normalized≤{gate2_cfg.get('virial_relative_normalized_threshold')}. "
                        f"Source: {gate2_cfg.get('source', 'n/a')}"
                    ),
                )
            )

        # Gate 3 — efficiency curve.
        #
        # T6.10a shipped this as `status: deferred` (no GPU Morse + no
        # Pattern 2 GPU dispatch). T7.12 reopens it as
        # `status: active_eam_substitute`: Pattern 2 GPU strong-scaling on
        # the same Ni-Al EAM/alloy fixture used by gates (1)+(2), per
        # D-M7-16 EAM substitute scope. Morse-vs-dissertation is still
        # deferred to M9+ (no GPU Morse kernel — gpu/SPEC §1.2).
        gate3_cfg = checks.get("efficiency_curve") or {}
        gate3_status = str(gate3_cfg.get("status", "")).lower()
        gate3_dissertation_ref = "n/a (gate 3 deferred — no GPU Morse kernel)"
        if gate3_status == "deferred":
            self.log(
                f"gate 3 (efficiency curve) deferred to "
                f"{gate3_cfg.get('deferred_to', 'T7.12')}; blockers: "
                f"{gate3_cfg.get('blockers', [])}"
            )
        elif gate3_status == "active_eam_substitute":
            efficiency_gates = self._run_gpu_efficiency_probe(
                gate3_cfg, launcher, n_procs_anchor=n_procs
            )
            gates.extend(efficiency_gates)
            for g in efficiency_gates:
                if not g.passed and overall_failure_mode is None:
                    overall_failure_mode = "EFFICIENCY_BELOW_FLOOR"
            gate3_dissertation_ref = (
                "n/a (T7.12 EAM substitute — Morse-vs-dissertation "
                "deferred to M9+ per D-M7-16)"
            )
        elif gate3_status:
            self.log(
                f"gate 3 (efficiency curve) — unknown status "
                f"{gate3_status!r}; skipping (treat as deferred)"
            )

        overall_passed = all(g.passed for g in gates)
        any_warning = any(g.status == STATUS_YELLOW for g in gates)
        if not overall_passed:
            overall_status = STATUS_RED
        elif any_warning:
            overall_status = STATUS_YELLOW
        else:
            overall_status = STATUS_GREEN

        end = datetime.datetime.now(tz=datetime.timezone.utc)
        return AnchorTestReport(
            points=[],  # GPU anchor uses gpu_gates instead of points
            overall_passed=overall_passed,
            overall_status=overall_status,
            any_warning=any_warning,
            dissertation_reference_commit=gate3_dissertation_ref,
            tdmd_commit=_git_revision_of(cfg.tdmd_bin),
            benchmark_directory=str(cfg.benchmark_dir),
            checks_yaml_path=str(checks_path),
            hardware=hw_placeholder,
            report_timestamp_utc=start.isoformat(timespec="seconds"),
            wall_clock_minutes=(end - start).total_seconds() / 60.0,
            normalization_log=list(self._log_lines),
            failure_mode=overall_failure_mode,
            backend="gpu",
            gpu_gates=gates,
        )

    # -----------------------------------------------------------------
    # T7.12 — Pattern 2 GPU efficiency probe (EAM substitute)
    # -----------------------------------------------------------------

    def _run_gpu_efficiency_probe(
        self,
        gate3_cfg: Mapping[str, object],
        launcher: GpuLaunchFn,
        n_procs_anchor: int,
    ) -> "list[GpuGateResult]":
        """Run the Pattern 2 GPU strong-scaling probe (T7.12).

        For each ``n`` in ``efficiency_curve.ranks_to_probe`` (must include
        the anchor — typically 1), launch tdmd with
        ``zoning.subdomains: [n, 1, 1]`` injected. The first probe point
        sets the steps/sec baseline; subsequent points report
        ``efficiency = 100 * sps(n) * anchor_n / (sps_anchor * n)``.

        Each probe emits a ``GpuGateResult``. The anchor point is always
        green (efficiency ≡ 100% by construction). Subsequent points pass
        iff ``measured_efficiency_pct >= efficiency_floor_pct``.

        The provenance string in every detail line records the
        EAM-substitute scope so report consumers cannot mistake this for
        a literal Morse-vs-dissertation comparison.
        """
        cfg = self.config
        ranks = [int(x) for x in gate3_cfg.get("ranks_to_probe", [1])]
        if not ranks:
            raise RuntimeError(
                "efficiency_curve.ranks_to_probe is empty; cannot probe efficiency"
            )
        if 1 not in ranks:
            raise RuntimeError(
                f"efficiency_curve.ranks_to_probe must include 1 as the "
                f"strong-scaling anchor; got {ranks}"
            )
        floor_pct = float(gate3_cfg.get("efficiency_floor_pct", 80.0))
        provenance_tag = (
            f"EAM substitute per D-M7-16 (Morse fidelity blocker: "
            f"{gate3_cfg.get('morse_fidelity_blocker', 'M9+ Morse GPU kernel')})"
        )
        n_steps = _read_config_n_steps(cfg.benchmark_dir / "config.yaml")

        out: list[GpuGateResult] = []
        anchor_steps_per_sec: float | None = None
        anchor_n: int | None = None
        for n in ranks:
            workdir = cfg.workdir / f"gpu_eff_N{n:02d}"
            thermo_path = workdir / "thermo.dat"
            try:
                telemetry = launcher(
                    self,
                    n,
                    workdir,
                    "gpu",
                    thermo_path,
                    subdomains_xyz=[n, 1, 1],
                )
            except RuntimeError as exc:
                out.append(
                    GpuGateResult(
                        gate_name=f"efficiency_curve_N{n:02d}",
                        passed=False,
                        status=STATUS_RED,
                        detail=(
                            f"Pattern 2 GPU launch failed for N={n}: {exc}. "
                            f"{provenance_tag}"
                        ),
                        n_procs=n,
                    )
                )
                continue

            total_wall = float(telemetry.get("total_wall_sec", 0.0))
            steps_per_sec = (n_steps / total_wall) if total_wall > 0 else 0.0

            if anchor_steps_per_sec is None:
                anchor_steps_per_sec = steps_per_sec
                anchor_n = n
                self.log(
                    f"efficiency probe anchor: N={n} → steps/s={steps_per_sec:.4f}"
                )
                # Anchor probe is informational; passes by definition.
                out.append(
                    GpuGateResult(
                        gate_name=f"efficiency_curve_N{n:02d}",
                        passed=True,
                        status=STATUS_GREEN,
                        detail=(
                            f"anchor: steps/s={steps_per_sec:.3f} "
                            f"(efficiency ≡ 100% by construction). "
                            f"{provenance_tag}"
                        ),
                        n_procs=n,
                        measured_steps_per_sec=steps_per_sec,
                        measured_efficiency_pct=100.0,
                        floor_pct=floor_pct,
                    )
                )
                continue

            if anchor_steps_per_sec <= 0 or anchor_n is None:
                # Anchor produced zero throughput — cannot derive efficiency.
                out.append(
                    GpuGateResult(
                        gate_name=f"efficiency_curve_N{n:02d}",
                        passed=False,
                        status=STATUS_RED,
                        detail=(
                            f"cannot compute efficiency for N={n}: "
                            f"anchor steps/s={anchor_steps_per_sec}. "
                            f"{provenance_tag}"
                        ),
                        n_procs=n,
                        measured_steps_per_sec=steps_per_sec,
                        floor_pct=floor_pct,
                    )
                )
                continue

            efficiency_pct = (
                100.0 * steps_per_sec * anchor_n / (anchor_steps_per_sec * n)
            )
            passed = efficiency_pct >= floor_pct
            out.append(
                GpuGateResult(
                    gate_name=f"efficiency_curve_N{n:02d}",
                    passed=passed,
                    status=STATUS_GREEN if passed else STATUS_RED,
                    detail=(
                        f"N={n}: steps/s={steps_per_sec:.3f} → "
                        f"efficiency={efficiency_pct:.2f}% vs anchor N={anchor_n} "
                        f"(floor={floor_pct:.1f}%). {provenance_tag}"
                    ),
                    n_procs=n,
                    measured_steps_per_sec=steps_per_sec,
                    measured_efficiency_pct=efficiency_pct,
                    floor_pct=floor_pct,
                )
            )
        return out

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


def _first_byte_diff(a: bytes, b: bytes) -> int:
    """Return the byte-offset of the first difference between ``a`` and ``b``.

    Returns -1 iff ``a == b``. If the common prefix is equal and one is a
    strict prefix of the other, returns the length of the shorter string
    (the position where the next byte would have been).
    """
    if a == b:
        return -1
    shorter = min(len(a), len(b))
    for i in range(shorter):
        if a[i] != b[i]:
            return i
    return shorter


def _csv_is_placeholder(path: pathlib.Path) -> bool:
    """Heuristic: the T5.10 placeholder CSV explicitly declared itself as
    ``# STATUS: preliminary placeholder``. T6.0 replaced the values with
    real extracted points from Andreev fig 29/30 and the header now
    begins ``# STATUS: extracted``. Only the literal placeholder sentinel
    counts — casual mentions of the word in commentary do not.
    """
    try:
        with path.open() as fh:
            head = fh.read(2048)
    except OSError:
        return False
    return "# STATUS: preliminary placeholder" in head
