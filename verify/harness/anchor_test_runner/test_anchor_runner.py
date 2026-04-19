"""T5.11 smoke tests — driver-logic-only, no real MPI / no real TDMD.

The real anchor-test run is slow tier (gated behind ``TDMD_SLOW_TIER=1``)
and only exercised locally pre-push. These tests verify:

    1. The runner's orchestrator wires telemetry → efficiency → diff
       correctly given a synthesised telemetry payload.
    2. The hardware probe caches correctly and honours ``force_probe``.
    3. The JSON report round-trips through write_json() unchanged.
    4. A CSV row mismatched against ``ranks_to_probe`` raises cleanly.

Run via ``python -m unittest verify.harness.anchor_test_runner.test_anchor_runner``.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import textwrap
import unittest

from .hardware_probe import probe
from .report import STATUS_GREEN, STATUS_RED, STATUS_YELLOW
from .runner import AnchorTestRunner, RunnerConfig, _first_byte_diff


# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------


def _write_checks_yaml(path: pathlib.Path, ranks: list[int], eff_tol: float = 0.10):
    path.write_text(
        textwrap.dedent(f"""
        benchmark: t3_al_fcc_large_anchor
        ranks_to_probe: {ranks}
        dissertation_comparison:
          efficiency_relative: {eff_tol}
          absolute_performance_relative: 0.25
        lammps_parity:
          step_0_force_relative: 1.0e-10
          step_1000_etotal_relative: 1.0e-6
        nve_drift:
          max_relative_drift: 1.0e-6
          window_steps: 1000
        runtime:
          wall_clock_budget_minutes: 60
        """).strip()
    )


def _write_reference_csv(
    path: pathlib.Path, rows: list[tuple[int, float, float]], placeholder: bool = True
):
    hdr = ""
    if placeholder:
        # Canonical sentinel the runner's _csv_is_placeholder detector grep is for.
        hdr = "# STATUS: preliminary placeholder — replace before production.\n"
    body = "n_procs,performance_mdps,efficiency_pct,source_figure,note\n"
    body += "\n".join(f"{n},{perf},{eff},fig29,ph" for (n, perf, eff) in rows)
    path.write_text(hdr + body + "\n")


def _write_config_yaml(path: pathlib.Path, n_steps: int = 1000):
    # Minimal; only ``run.n_steps`` is actually read by the runner.
    path.write_text(
        textwrap.dedent(f"""
        simulation: {{units: metal, seed: 1}}
        atoms: {{source: lammps_data, path: setup.data}}
        potential: {{style: morse, params: {{D: 0.1, alpha: 1.0, r0: 3.0, cutoff: 8.0}}}}
        integrator: {{style: velocity_verlet, dt: 0.001}}
        neighbor: {{skin: 2.0}}
        thermo: {{every: 100}}
        run: {{n_steps: {n_steps}}}
        comm: {{backend: mpi_host_staging, topology: mesh}}
        """).strip()
    )


def _write_placeholder_script(path: pathlib.Path):
    """Write a stub hardware_normalization.py that emits a deterministic JSON."""
    path.write_text(
        textwrap.dedent("""
        #!/usr/bin/env python3
        import json
        import sys
        payload = {
            "ghz_flops_ratio": 2.5,
            "baseline_gflops": 9.0,
            "local_gflops": 22.5,
        }
        print(json.dumps(payload))
        sys.exit(0)
        """).strip()
    )
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# Mock launcher — returns telemetry payloads deterministically
# ---------------------------------------------------------------------------


def make_mock_launcher(wall_sec_by_n: dict[int, float]):
    """Return a ``launch_fn`` that produces synthetic telemetry JSON."""

    def _fn(runner, n_procs: int, per_rank_workdir: pathlib.Path) -> dict:
        per_rank_workdir.mkdir(parents=True, exist_ok=True)
        return {
            "event": "run_end",
            "total_wall_sec": wall_sec_by_n[n_procs],
            "sections": {"Pair": wall_sec_by_n[n_procs] * 0.6},
            "ignored_end_calls": 0,
        }

    return _fn


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class HardwareProbeCacheTest(unittest.TestCase):
    def test_fresh_probe_writes_cache_and_cached_read_returns_same_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            script = tmpdir / "hardware_normalization.py"
            _write_placeholder_script(script)
            cache = tmpdir / "hw_cache.json"

            first = probe(
                force_probe=True,
                hardware_normalization_script=script,
                cache_path=cache,
            )
            self.assertFalse(first.cached)
            self.assertAlmostEqual(first.ghz_flops_ratio, 2.5, places=6)
            self.assertTrue(cache.is_file())

            second = probe(
                force_probe=False,
                hardware_normalization_script=script,
                cache_path=cache,
            )
            self.assertTrue(second.cached)
            self.assertAlmostEqual(second.ghz_flops_ratio, 2.5, places=6)

    def test_force_probe_bypasses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            script = tmpdir / "hardware_normalization.py"
            _write_placeholder_script(script)
            cache = tmpdir / "hw_cache.json"

            probe(
                force_probe=True, hardware_normalization_script=script, cache_path=cache
            )
            second = probe(
                force_probe=True,
                hardware_normalization_script=script,
                cache_path=cache,
            )
            self.assertFalse(second.cached)

    def test_corrupt_cache_falls_back_to_fresh_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            script = tmpdir / "hardware_normalization.py"
            _write_placeholder_script(script)
            cache = tmpdir / "hw_cache.json"
            cache.write_text("{not json")
            result = probe(
                force_probe=False,
                hardware_normalization_script=script,
                cache_path=cache,
            )
            self.assertFalse(result.cached)


class AnchorRunnerMockedTest(unittest.TestCase):
    def _make_runner(
        self,
        tmpdir: pathlib.Path,
        ranks: list[int],
        wall_by_n: dict[int, float],
        reference_rows: list[tuple[int, float, float]],
        eff_tol: float = 0.10,
        placeholder_ref: bool = True,
    ) -> AnchorTestRunner:
        bench = tmpdir / "benchmarks" / "t3"
        bench.mkdir(parents=True)
        _write_checks_yaml(bench / "checks.yaml", ranks, eff_tol=eff_tol)
        _write_reference_csv(
            bench / "dissertation_reference_data.csv",
            reference_rows,
            placeholder=placeholder_ref,
        )
        _write_config_yaml(bench / "config.yaml", n_steps=1000)

        hw_script = tmpdir / "hardware_normalization.py"
        _write_placeholder_script(hw_script)

        cfg = RunnerConfig(
            benchmark_dir=bench,
            tdmd_bin=tmpdir / "tdmd",
            mpirun_bin=pathlib.Path("mpirun"),
            output_report_path=tmpdir / "report.json",
            workdir=tmpdir / "workdir",
            ranks_override=None,
            force_probe=True,
            skip_setup_regen=True,
            launch_fn=make_mock_launcher(wall_by_n),
        )
        runner = AnchorTestRunner(cfg)

        # Monkey-patch the hardware probe to use our stub script without
        # touching ~/.cache. The runner's probe() call reads from
        # :mod:`hardware_probe` at module-scope; we redirect through a
        # shim.
        from . import runner as _runner_mod

        original_probe = _runner_mod.probe
        # Use a per-test temp cache to avoid touching ~/.cache.
        cache_path = tmpdir / "hw_cache.json"

        def shimmed_probe(force_probe: bool = False):
            return original_probe(
                force_probe=True,
                hardware_normalization_script=hw_script,
                cache_path=cache_path,
            )

        _runner_mod.probe = shimmed_probe
        self.addCleanup(lambda: setattr(_runner_mod, "probe", original_probe))
        # setup.data regen must be skipped — create a marker file to bypass
        # the existence check's error branch.
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        data_dir = repo_root / "verify" / "data" / "t3_al_fcc_large_anchor"
        setup = data_dir / "setup.data"
        created_fake_setup = False
        if not setup.is_file():
            data_dir.mkdir(parents=True, exist_ok=True)
            setup.write_text("# fake setup.data for T5.11 unit test — safe to delete\n")
            created_fake_setup = True

            def _cleanup():
                try:
                    setup.unlink()
                except OSError:
                    pass

            self.addCleanup(_cleanup)
        _ = created_fake_setup
        return runner

    def test_all_green_exit(self):
        """Measured efficiency matches reference within 10% → GREEN."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            # Serial = 1 sec/1000 steps = 86.4 mdps.
            # N=2 linear scaling → 0.5s total; efficiency = 100%.
            # Reference in the CSV also has 100% for N=2 to match.
            ranks = [1, 2]
            wall = {1: 1.0, 2: 0.5}
            ref = [(1, 86.4 / 2.5, 100.0), (2, 172.8 / 2.5, 100.0)]
            # Divide by hw_ratio=2.5 so normalised ref == measured.
            runner = self._make_runner(tmpdir, ranks, wall, ref)
            report = runner.run()
            self.assertEqual(report.overall_status, STATUS_GREEN)
            self.assertTrue(report.overall_passed)
            self.assertFalse(report.any_warning)
            self.assertEqual(len(report.points), 2)

    def test_efficiency_fail_trips_red(self):
        """Measured efficiency 50%, reference 100% → eff_rel_err=0.5 > 0.10."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            ranks = [1, 2]
            # Serial = 1 mdps-worth; N=2 runs *slower* per-rank → eff=50%.
            wall = {1: 1.0, 2: 1.0}
            ref = [(1, 86.4 / 2.5, 100.0), (2, 172.8 / 2.5, 100.0)]
            runner = self._make_runner(tmpdir, ranks, wall, ref)
            report = runner.run()
            self.assertEqual(report.overall_status, STATUS_RED)
            self.assertFalse(report.overall_passed)
            self.assertEqual(report.failure_mode, "REF_DATA_STALE")

    def test_absolute_perf_only_warning_is_yellow(self):
        """Efficiency good; absolute perf 30% off → YELLOW."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            ranks = [1, 2]
            wall = {1: 1.0, 2: 0.5}  # both 100% eff
            # Reference perf ×0.7 of measured → abs_rel_err = 3/7 ≈ 0.43 > 0.25
            measured_mdps_1 = 86.4
            measured_mdps_2 = 172.8
            ref = [
                (1, 0.7 * measured_mdps_1 / 2.5, 100.0),
                (2, 0.7 * measured_mdps_2 / 2.5, 100.0),
            ]
            runner = self._make_runner(tmpdir, ranks, wall, ref)
            report = runner.run()
            self.assertEqual(report.overall_status, STATUS_YELLOW)
            self.assertTrue(report.overall_passed)
            self.assertTrue(report.any_warning)

    def test_report_round_trips_to_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            ranks = [1, 2]
            wall = {1: 1.0, 2: 0.5}
            ref = [(1, 86.4 / 2.5, 100.0), (2, 172.8 / 2.5, 100.0)]
            runner = self._make_runner(tmpdir, ranks, wall, ref)
            report = runner.run()
            out = tmpdir / "rep.json"
            report.write_json(out)
            data = json.loads(out.read_text())
            self.assertEqual(data["overall_status"], STATUS_GREEN)
            self.assertEqual(len(data["points"]), 2)
            self.assertIn("hardware", data)
            self.assertIn("ghz_flops_ratio", data["hardware"])

    def test_missing_reference_row_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            ranks = [1, 2, 4]
            wall = {1: 1.0, 2: 0.5, 4: 0.25}
            ref = [(1, 86.4 / 2.5, 100.0), (2, 172.8 / 2.5, 100.0)]  # no N=4
            runner = self._make_runner(tmpdir, ranks, wall, ref)
            with self.assertRaises(RuntimeError) as cm:
                runner.run()
            self.assertIn("n_procs=4", str(cm.exception))


# ---------------------------------------------------------------------------
# T6.10a — T3-gpu two-level anchor mocked tests
# ---------------------------------------------------------------------------


def _write_gpu_checks_yaml(
    path: pathlib.Path,
    bit_exact: bool = True,
    gate2_present: bool = True,
):
    body = [
        "benchmark: t3_al_fcc_large_anchor_gpu",
        "backend: gpu",
        "ranks_to_probe: [1]",
    ]
    if bit_exact:
        body.append(
            "cpu_gpu_reference_bit_exact: {thermo_byte_equal: true, steps: 100}"
        )
    if gate2_present:
        body += [
            "mixed_fast_vs_reference:",
            "  force_relative_linf_threshold: 1.0e-5",
            "  energy_relative_threshold: 1.0e-7",
            "  virial_relative_normalized_threshold: 5.0e-6",
            "  source: T6.8a achieved",
        ]
    body += [
        "efficiency_curve:",
        "  status: deferred",
        "  deferred_to: T6.10b",
        "  blockers: [T6.9b, M9 Morse GPU]",
        "runtime: {wall_clock_budget_minutes: 10}",
    ]
    path.write_text("\n".join(body) + "\n")


def _write_gpu_config_yaml(path: pathlib.Path):
    # Minimal config — no runtime.backend (the runner injects it).
    path.write_text(
        textwrap.dedent("""
        simulation: {units: metal, seed: 1}
        atoms: {source: lammps_data, path: setup.data}
        potential:
          style: eam/alloy
          params: {file: Ni-Al.eam.alloy}
        integrator: {style: velocity_verlet, dt: 0.001}
        neighbor: {skin: 0.3}
        thermo: {every: 1}
        run: {n_steps: 100}
        comm: {backend: mpi_host_staging, topology: mesh}
        """).strip()
    )


def _write_stub_gpu_probe(path: pathlib.Path, gpu_model: str | None = "Stub GPU"):
    """Write a python3 stub that mimics hardware_normalization_gpu.py."""
    if gpu_model is None:
        model_literal = "None"
    else:
        model_literal = repr(gpu_model)
    path.write_text(
        textwrap.dedent(f"""
        #!/usr/bin/env python3
        import json
        import sys
        payload = {{
            "ghz_flops_ratio": 1.0,
            "gpu_flops_ratio": 1.0,
            "baseline_gflops": 0.0,
            "local_gflops": 0.0,
            "gpu_model": {model_literal},
            "note": "T6.10a unit-test stub",
        }}
        print(json.dumps(payload))
        sys.exit(0)
        """).strip()
    )
    path.chmod(0o755)


def make_gpu_mock_launcher(
    thermo_bytes_by_backend: dict[str, bytes],
    fail_backend: str | None = None,
):
    """GPU launcher mock — writes pre-canned thermo bytes for each backend.

    ``fail_backend='gpu'`` triggers a RuntimeError on the GPU launch so the
    runner can exercise the NO_CUDA_DEVICE fallback path.
    """

    def _fn(runner, n_procs: int, workdir: pathlib.Path, backend: str, thermo_path):
        if fail_backend == backend:
            raise RuntimeError(f"mock launcher injected failure for backend={backend}")
        workdir.mkdir(parents=True, exist_ok=True)
        thermo_path.parent.mkdir(parents=True, exist_ok=True)
        thermo_path.write_bytes(thermo_bytes_by_backend[backend])
        return {
            "event": "run_end",
            "total_wall_sec": 1.0,
            "sections": {"Pair": 0.6},
            "ignored_end_calls": 0,
        }

    return _fn


class GpuAnchorRunnerMockedTest(unittest.TestCase):
    """T3-gpu two-level mocked dispatch (T6.10a)."""

    def _make_gpu_runner(
        self,
        tmpdir: pathlib.Path,
        launch_fn,
        gpu_model: str | None = "Stub GPU",
        gate2_present: bool = True,
    ) -> AnchorTestRunner:
        bench = tmpdir / "benchmarks" / "t3_gpu"
        bench.mkdir(parents=True)
        _write_gpu_checks_yaml(bench / "checks.yaml", gate2_present=gate2_present)
        _write_gpu_config_yaml(bench / "config.yaml")
        _write_stub_gpu_probe(
            bench / "hardware_normalization_gpu.py", gpu_model=gpu_model
        )
        cfg = RunnerConfig(
            benchmark_dir=bench,
            tdmd_bin=tmpdir / "tdmd",
            mpirun_bin=pathlib.Path("mpirun"),
            output_report_path=tmpdir / "report.json",
            workdir=tmpdir / "workdir",
            skip_setup_regen=True,
            gpu_launch_fn=launch_fn,
        )
        return AnchorTestRunner(cfg)

    def test_gpu_anchor_byte_exact_green(self):
        """CPU + GPU thermo match → gate1 GREEN; gate2 advisory YELLOW → overall YELLOW."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            same = b"Step KE PE TE\n0 1.0 -2.0 -1.0\n1 1.01 -2.02 -1.01\n"
            runner = self._make_gpu_runner(
                tmpdir,
                make_gpu_mock_launcher({"cpu": same, "gpu": same}),
            )
            report = runner.run()
            self.assertEqual(report.backend, "gpu")
            self.assertIsNotNone(report.gpu_gates)
            # Gate 1 passed (byte-equal).
            g1 = next(
                g
                for g in report.gpu_gates
                if g.gate_name == "cpu_gpu_reference_bit_exact"
            )
            self.assertTrue(g1.passed)
            self.assertEqual(g1.status, STATUS_GREEN)
            # Gate 2 present → advisory YELLOW.
            g2 = next(
                g for g in report.gpu_gates if g.gate_name == "mixed_fast_vs_reference"
            )
            self.assertTrue(g2.passed)
            self.assertEqual(g2.status, STATUS_YELLOW)
            # Overall YELLOW because delegated gate 2 is advisory.
            self.assertTrue(report.overall_passed)
            self.assertEqual(report.overall_status, STATUS_YELLOW)
            self.assertTrue(report.any_warning)
            self.assertIsNone(report.failure_mode)

    def test_gpu_anchor_byte_exact_green_no_gate2(self):
        """Without gate2, a matching gate1 yields GREEN overall."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            same = b"Step KE PE TE\n0 1.0 -2.0 -1.0\n"
            runner = self._make_gpu_runner(
                tmpdir,
                make_gpu_mock_launcher({"cpu": same, "gpu": same}),
                gate2_present=False,
            )
            report = runner.run()
            self.assertEqual(report.overall_status, STATUS_GREEN)
            self.assertTrue(report.overall_passed)
            self.assertFalse(report.any_warning)
            self.assertEqual(len(report.gpu_gates), 1)

    def test_gpu_anchor_byte_exact_diverge_red(self):
        """CPU + GPU thermo differ → gate1 RED + CPU_GPU_REFERENCE_DIVERGE."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            cpu = b"Step KE PE TE\n0 1.0 -2.0 -1.0\n"
            gpu = b"Step KE PE TE\n0 1.0 -2.0 -1.001\n"  # last byte differs
            runner = self._make_gpu_runner(
                tmpdir,
                make_gpu_mock_launcher({"cpu": cpu, "gpu": gpu}),
            )
            report = runner.run()
            self.assertEqual(report.overall_status, STATUS_RED)
            self.assertFalse(report.overall_passed)
            self.assertEqual(report.failure_mode, "CPU_GPU_REFERENCE_DIVERGE")
            g1 = next(
                g
                for g in report.gpu_gates
                if g.gate_name == "cpu_gpu_reference_bit_exact"
            )
            self.assertFalse(g1.passed)
            self.assertIn("diverged", g1.detail)

    def test_gpu_anchor_no_cuda_device_red(self):
        """GPU probe reports no model + GPU launch fails → NO_CUDA_DEVICE."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            runner = self._make_gpu_runner(
                tmpdir,
                make_gpu_mock_launcher(
                    {"cpu": b"placeholder\n", "gpu": b"placeholder\n"},
                    fail_backend="gpu",
                ),
                gpu_model=None,  # probe returns None → cannot see a GPU
            )
            report = runner.run()
            self.assertEqual(report.overall_status, STATUS_RED)
            self.assertEqual(report.failure_mode, "NO_CUDA_DEVICE")

    def test_gpu_anchor_reports_roundtrip_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            same = b"Step KE PE TE\n0 1.0 -2.0 -1.0\n"
            runner = self._make_gpu_runner(
                tmpdir,
                make_gpu_mock_launcher({"cpu": same, "gpu": same}),
            )
            report = runner.run()
            out = tmpdir / "gpu_rep.json"
            report.write_json(out)
            data = json.loads(out.read_text())
            self.assertEqual(data["backend"], "gpu")
            self.assertIsNotNone(data["gpu_gates"])
            self.assertEqual(len(data["gpu_gates"]), 2)
            # points list empty for GPU anchor (no CPU-style perf points in T6.10a).
            self.assertEqual(data["points"], [])

    def test_backend_override_forces_gpu_on_cpu_fixture(self):
        """CLI --backend gpu overrides checks.yaml::backend even when absent."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = pathlib.Path(tmp)
            bench = tmpdir / "benchmarks" / "t3_cpu_fixture"
            bench.mkdir(parents=True)
            # CPU-style fixture (no backend key) — but CLI asks for gpu.
            (bench / "checks.yaml").write_text(
                "benchmark: cpu_fixture\n"
                "ranks_to_probe: [1]\n"
                "cpu_gpu_reference_bit_exact: {thermo_byte_equal: true}\n"
                "runtime: {wall_clock_budget_minutes: 10}\n"
            )
            _write_gpu_config_yaml(bench / "config.yaml")
            _write_stub_gpu_probe(bench / "hardware_normalization_gpu.py")
            same = b"Step\n0 1.0\n"
            cfg = RunnerConfig(
                benchmark_dir=bench,
                tdmd_bin=tmpdir / "tdmd",
                mpirun_bin=pathlib.Path("mpirun"),
                output_report_path=tmpdir / "rep.json",
                workdir=tmpdir / "workdir",
                skip_setup_regen=True,
                gpu_launch_fn=make_gpu_mock_launcher({"cpu": same, "gpu": same}),
                backend_override="gpu",
            )
            runner = AnchorTestRunner(cfg)
            report = runner.run()
            self.assertEqual(report.backend, "gpu")
            self.assertTrue(report.overall_passed)


class FirstByteDiffTest(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(_first_byte_diff(b"abc", b"abc"), -1)

    def test_first_byte_differs(self):
        self.assertEqual(_first_byte_diff(b"xbc", b"abc"), 0)

    def test_last_byte_differs(self):
        self.assertEqual(_first_byte_diff(b"abx", b"abc"), 2)

    def test_prefix(self):
        self.assertEqual(_first_byte_diff(b"ab", b"abc"), 2)


if __name__ == "__main__":
    unittest.main()
