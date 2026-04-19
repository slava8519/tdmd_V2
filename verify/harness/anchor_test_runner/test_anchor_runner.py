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
from .runner import AnchorTestRunner, RunnerConfig


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


if __name__ == "__main__":
    unittest.main()
