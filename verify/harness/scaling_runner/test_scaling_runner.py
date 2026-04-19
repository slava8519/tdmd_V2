"""T7.11 smoke tests — driver-logic-only, no real MPI / no real TDMD.

The real T7 strong-scaling sweep is slow tier (local pre-push only,
needs visible CUDA devices). These pytests verify the orchestrator
correctness given synthesised telemetry payloads:

    1. Efficiency formula: E(N) = rate(N) / (rate(1) × N) × 100.
    2. Gate dispatch: per-N rank → matching efficiency_gates entry.
    3. Augmented config injection: ``zoning.subdomains: [N, 1, 1]`` with
       absolute paths in atoms.path / potential.params.file.
    4. Pattern 1 byte-exact gate logic (PASS / FAIL / skip-when-no-baseline).
    5. Hard-fail propagation: launch failure → STATUS_RED + LAUNCH_FAILURE.
    6. ranks_to_probe must include 1 (anchor invariant).

Run via ``python -m pytest verify/harness/scaling_runner/test_scaling_runner.py``.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import textwrap
import unittest

import yaml

from .runner import (
    RunnerConfig,
    ScalingRunner,
    STATUS_GREEN,
    STATUS_RED,
    _first_byte_diff,
    _resolve_gate,
    _write_augmented_config,
)


def _write_checks_yaml(
    path: pathlib.Path,
    ranks: list[int],
    *,
    pattern1_baseline: bool = True,
):
    path.write_text(
        textwrap.dedent(f"""
        benchmark: t7_mixed_scaling
        backend: gpu
        ranks_to_probe: {ranks}
        efficiency_gates:
          single_node:
            min_rank_inclusive: 2
            max_rank_inclusive: 8
            gate_pct: 80.0
            rationale: D-M7-8 single-node target
          two_node:
            min_rank_inclusive: 9
            max_rank_inclusive: 16
            gate_pct: 70.0
            rationale: D-M7-8 2-node target
        pattern1_baseline_byte_exact: {str(pattern1_baseline).lower()}
        runtime:
          wall_clock_budget_minutes: 60
        """).strip()
    )


def _write_config_yaml(path: pathlib.Path, n_steps: int = 100):
    path.write_text(
        textwrap.dedent(f"""
        simulation: {{units: metal, seed: 12345}}
        atoms: {{source: lammps_data, path: setup.data}}
        potential:
          style: eam/alloy
          params:
            file: NiAl_Mishin_2004.eam.alloy
        integrator: {{style: velocity_verlet, dt: 0.001}}
        neighbor: {{skin: 0.3}}
        thermo: {{every: 1}}
        run: {{n_steps: {n_steps}}}
        zoning: {{scheme: linear_1d}}
        scheduler: {{td_mode: true, pipeline_depth_cap: 1}}
        comm: {{backend: hybrid, topology: mesh}}
        """).strip()
    )


def _make_fake_launcher(rates_steps_per_sec: dict[int, float]):
    """Build a launcher that returns synthesised telemetry per N.

    ``rates_steps_per_sec`` maps N → desired rate. The launcher computes
    total_wall_sec = n_steps / rate_for_N, writes a fake thermo file (for
    the byte-exact comparison test), and returns the telemetry dict.
    """

    def _launch(runner, n_procs, workdir, thermo_path):
        rate = rates_steps_per_sec.get(n_procs)
        if rate is None:
            raise RuntimeError(f"test launcher has no rate for N={n_procs}")
        # Write a deterministic thermo file so the byte-exact comparison
        # has actual bytes to read. Content depends on N so the
        # baseline-mismatch test can flip it.
        workdir.mkdir(parents=True, exist_ok=True)
        thermo_path.parent.mkdir(parents=True, exist_ok=True)
        thermo_path.write_bytes(f"thermo for N={n_procs}\n".encode())
        n_steps = 100  # must match _write_config_yaml above
        wall = n_steps / rate
        return {"total_wall_sec": wall}

    return _launch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEfficiencyFormula(unittest.TestCase):
    def test_perfect_scaling_at_anchor(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = pathlib.Path(td)
            bench = tdp / "bench"
            bench.mkdir()
            _write_config_yaml(bench / "config.yaml")
            _write_checks_yaml(bench / "checks.yaml", ranks=[1, 2, 4])

            launcher = _make_fake_launcher(
                {1: 100.0, 2: 200.0, 4: 400.0}  # perfect scaling
            )
            cfg = RunnerConfig(
                benchmark_dir=bench,
                tdmd_bin=tdp / "tdmd",
                mpirun_bin=pathlib.Path("mpirun"),
                output_report_path=tdp / "report.json",
                workdir=tdp / "workdir",
                skip_setup_regen=True,  # no setup.data needed; mock launcher
                launch_fn=launcher,
            )
            # Skip lazy regen by also setting atoms.path to something
            # that doesn't exist + skip_setup_regen=False would error.
            # Instead we enable skip_setup_regen and just make sure
            # the config doesn't trip _regen_setup_data: mark file as if it
            # exists by writing one.
            (bench / "setup.data").write_text("")
            report = ScalingRunner(cfg).run()

            self.assertEqual(report.overall_status, STATUS_GREEN)
            self.assertEqual(len(report.points), 3)
            anchor = report.points[0]
            self.assertEqual(anchor.n_procs, 1)
            self.assertAlmostEqual(anchor.measured_efficiency_pct, 100.0)
            self.assertAlmostEqual(report.points[1].measured_efficiency_pct, 100.0)
            self.assertAlmostEqual(report.points[2].measured_efficiency_pct, 100.0)

    def test_below_gate_marks_red(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = pathlib.Path(td)
            bench = tdp / "bench"
            bench.mkdir()
            _write_config_yaml(bench / "config.yaml")
            _write_checks_yaml(bench / "checks.yaml", ranks=[1, 2])
            (bench / "setup.data").write_text("")

            # N=2 only delivers 1.5x → 75% efficiency, below 80% gate.
            launcher = _make_fake_launcher({1: 100.0, 2: 150.0})
            cfg = RunnerConfig(
                benchmark_dir=bench,
                tdmd_bin=tdp / "tdmd",
                mpirun_bin=pathlib.Path("mpirun"),
                output_report_path=tdp / "report.json",
                workdir=tdp / "workdir",
                skip_setup_regen=True,
                launch_fn=launcher,
            )
            report = ScalingRunner(cfg).run()

            self.assertEqual(report.overall_status, STATUS_RED)
            self.assertEqual(report.failure_mode, "EFFICIENCY_GATE_FAIL")
            n2 = next(p for p in report.points if p.n_procs == 2)
            self.assertAlmostEqual(n2.measured_efficiency_pct, 75.0)
            self.assertEqual(n2.gate_name, "single_node")
            self.assertEqual(n2.gate_pct, 80.0)
            self.assertFalse(n2.gate_passed)


class TestRanksMustIncludeAnchor(unittest.TestCase):
    def test_missing_anchor_raises(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = pathlib.Path(td)
            bench = tdp / "bench"
            bench.mkdir()
            _write_config_yaml(bench / "config.yaml")
            _write_checks_yaml(bench / "checks.yaml", ranks=[2, 4])
            (bench / "setup.data").write_text("")

            cfg = RunnerConfig(
                benchmark_dir=bench,
                tdmd_bin=tdp / "tdmd",
                mpirun_bin=pathlib.Path("mpirun"),
                output_report_path=tdp / "report.json",
                workdir=tdp / "workdir",
                skip_setup_regen=True,
                launch_fn=_make_fake_launcher({2: 200.0, 4: 400.0}),
            )
            with self.assertRaisesRegex(RuntimeError, "must include 1"):
                ScalingRunner(cfg).run()


class TestAugmentedConfig(unittest.TestCase):
    def test_subdomains_injected(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = pathlib.Path(td)
            src = tdp / "config.yaml"
            dst = tdp / "augmented.yaml"
            _write_config_yaml(src)
            _write_augmented_config(src, dst, n_subdomains_x=4)

            with dst.open() as fh:
                aug = yaml.safe_load(fh)
            self.assertEqual(aug["zoning"]["subdomains"], [4, 1, 1])
            # Relative atoms.path resolved to absolute beside the source.
            atoms_path = pathlib.Path(aug["atoms"]["path"])
            self.assertTrue(atoms_path.is_absolute())
            self.assertEqual(atoms_path.name, "setup.data")
            # Potential file likewise resolved.
            pot_path = pathlib.Path(aug["potential"]["params"]["file"])
            self.assertTrue(pot_path.is_absolute())
            self.assertEqual(pot_path.name, "NiAl_Mishin_2004.eam.alloy")


class TestGateDispatch(unittest.TestCase):
    def test_resolve_gate_single_node_window(self):
        gates = {
            "single_node": {
                "min_rank_inclusive": 2,
                "max_rank_inclusive": 8,
                "gate_pct": 80.0,
            },
            "two_node": {
                "min_rank_inclusive": 9,
                "max_rank_inclusive": 16,
                "gate_pct": 70.0,
            },
        }
        # 1: no gate (anchor)
        self.assertEqual(_resolve_gate(1, gates), (None, None))
        # 2..8: single_node @ 80
        for n in (2, 4, 8):
            name, pct = _resolve_gate(n, gates)
            self.assertEqual(name, "single_node")
            self.assertEqual(pct, 80.0)
        # 9..16: two_node @ 70
        for n in (9, 12, 16):
            name, pct = _resolve_gate(n, gates)
            self.assertEqual(name, "two_node")
            self.assertEqual(pct, 70.0)
        # 17+: no gate configured
        self.assertEqual(_resolve_gate(17, gates), (None, None))


class TestPattern1ByteExact(unittest.TestCase):
    def test_baseline_match_passes(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = pathlib.Path(td)
            bench = tdp / "bench"
            bench.mkdir()
            _write_config_yaml(bench / "config.yaml")
            _write_checks_yaml(bench / "checks.yaml", ranks=[1, 2])
            (bench / "setup.data").write_text("")

            # Both N=1 anchor and the baseline produce the identical bytes
            # ("thermo for N=1\n") because the fake launcher writes the
            # same content at N=1, and we set the baseline to exactly that.
            baseline = tdp / "baseline_thermo.dat"
            baseline.write_bytes(b"thermo for N=1\n")

            cfg = RunnerConfig(
                benchmark_dir=bench,
                tdmd_bin=tdp / "tdmd",
                mpirun_bin=pathlib.Path("mpirun"),
                output_report_path=tdp / "report.json",
                workdir=tdp / "workdir",
                skip_setup_regen=True,
                launch_fn=_make_fake_launcher({1: 100.0, 2: 200.0}),
                baseline_thermo_path=baseline,
            )
            report = ScalingRunner(cfg).run()
            self.assertTrue(report.pattern1_baseline_byte_exact)
            self.assertEqual(report.pattern1_baseline_diff_byte, -1)
            self.assertEqual(report.overall_status, STATUS_GREEN)

    def test_baseline_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = pathlib.Path(td)
            bench = tdp / "bench"
            bench.mkdir()
            _write_config_yaml(bench / "config.yaml")
            _write_checks_yaml(bench / "checks.yaml", ranks=[1, 2])
            (bench / "setup.data").write_text("")

            baseline = tdp / "baseline_thermo.dat"
            baseline.write_bytes(b"DIFFERENT thermo\n")

            cfg = RunnerConfig(
                benchmark_dir=bench,
                tdmd_bin=tdp / "tdmd",
                mpirun_bin=pathlib.Path("mpirun"),
                output_report_path=tdp / "report.json",
                workdir=tdp / "workdir",
                skip_setup_regen=True,
                launch_fn=_make_fake_launcher({1: 100.0, 2: 200.0}),
                baseline_thermo_path=baseline,
            )
            report = ScalingRunner(cfg).run()
            self.assertFalse(report.pattern1_baseline_byte_exact)
            self.assertGreaterEqual(report.pattern1_baseline_diff_byte, 0)
            self.assertEqual(report.overall_status, STATUS_RED)
            self.assertEqual(report.failure_mode, "PATTERN1_BYTE_EXACT_BREAK")

    def test_no_baseline_skips_gate_silently(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = pathlib.Path(td)
            bench = tdp / "bench"
            bench.mkdir()
            _write_config_yaml(bench / "config.yaml")
            _write_checks_yaml(bench / "checks.yaml", ranks=[1, 2])
            (bench / "setup.data").write_text("")

            cfg = RunnerConfig(
                benchmark_dir=bench,
                tdmd_bin=tdp / "tdmd",
                mpirun_bin=pathlib.Path("mpirun"),
                output_report_path=tdp / "report.json",
                workdir=tdp / "workdir",
                skip_setup_regen=True,
                launch_fn=_make_fake_launcher({1: 100.0, 2: 200.0}),
                baseline_thermo_path=None,  # gate cannot run
            )
            report = ScalingRunner(cfg).run()
            self.assertIsNone(report.pattern1_baseline_byte_exact)
            self.assertIsNone(report.pattern1_baseline_diff_byte)
            self.assertEqual(report.overall_status, STATUS_GREEN)


class TestLaunchFailure(unittest.TestCase):
    def test_launch_exception_propagates_as_red(self):
        def _bad_launcher(runner, n, wd, thermo):
            raise RuntimeError("simulated tdmd crash")

        with tempfile.TemporaryDirectory() as td:
            tdp = pathlib.Path(td)
            bench = tdp / "bench"
            bench.mkdir()
            _write_config_yaml(bench / "config.yaml")
            _write_checks_yaml(bench / "checks.yaml", ranks=[1, 2])
            (bench / "setup.data").write_text("")

            cfg = RunnerConfig(
                benchmark_dir=bench,
                tdmd_bin=tdp / "tdmd",
                mpirun_bin=pathlib.Path("mpirun"),
                output_report_path=tdp / "report.json",
                workdir=tdp / "workdir",
                skip_setup_regen=True,
                launch_fn=_bad_launcher,
            )
            report = ScalingRunner(cfg).run()
            self.assertEqual(report.overall_status, STATUS_RED)
            self.assertEqual(report.failure_mode, "LAUNCH_FAILURE")


class TestReportSerialisation(unittest.TestCase):
    def test_to_dict_round_trips_through_json(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = pathlib.Path(td)
            bench = tdp / "bench"
            bench.mkdir()
            _write_config_yaml(bench / "config.yaml")
            _write_checks_yaml(bench / "checks.yaml", ranks=[1, 2])
            (bench / "setup.data").write_text("")

            cfg = RunnerConfig(
                benchmark_dir=bench,
                tdmd_bin=tdp / "tdmd",
                mpirun_bin=pathlib.Path("mpirun"),
                output_report_path=tdp / "report.json",
                workdir=tdp / "workdir",
                skip_setup_regen=True,
                launch_fn=_make_fake_launcher({1: 100.0, 2: 200.0}),
            )
            report = ScalingRunner(cfg).run()
            blob = json.dumps(report.to_dict(), sort_keys=False)
            parsed = json.loads(blob)
            self.assertEqual(parsed["overall_status"], "GREEN")
            self.assertEqual(parsed["points"][0]["n_procs"], 1)
            self.assertEqual(parsed["points"][1]["n_procs"], 2)


class TestFirstByteDiff(unittest.TestCase):
    def test_identical_returns_minus_one(self):
        self.assertEqual(_first_byte_diff(b"abc", b"abc"), -1)

    def test_first_diff_index(self):
        self.assertEqual(_first_byte_diff(b"abc", b"abd"), 2)

    def test_prefix_returns_shorter_length(self):
        self.assertEqual(_first_byte_diff(b"abc", b"abcd"), 3)


if __name__ == "__main__":
    unittest.main()
