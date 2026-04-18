"""Unit tests for the dump-file parsing + forces-compare layer (T2.8).

Exercises ``verify.compare.parse_dump_file`` and ``compare_forces`` against
hand-crafted fixtures — no LAMMPS / TDMD binary required, so these tests
run on any stock Python environment (the CI Pipeline A `python3` image).

Run with: ``python3 -m unittest discover verify/tests``
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verify import compare  # noqa: E402


def _write_dump(path: pathlib.Path, body: str) -> None:
    path.write_text(body)


# Minimal well-formed LAMMPS-style dump body used by parser tests. Four
# atoms, ids shuffled on purpose so the parser cannot accidentally rely on
# file order.
DUMP_WELLFORMED = """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z fx fy fz
3 1 3.0 0.0 0.0  0.30  0.00  0.00
1 1 1.0 0.0 0.0  0.10  0.00  0.00
4 1 4.0 0.0 0.0  0.40  0.00  0.00
2 1 2.0 0.0 0.0  0.20  0.00  0.00
"""


class ParseDumpTests(unittest.TestCase):
    def test_parses_timestep_and_atom_count(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "dump.lmp"
            _write_dump(p, DUMP_WELLFORMED)
            frame = compare.parse_dump_file(p)
        self.assertEqual(frame.timestep, 100)
        self.assertEqual(len(frame.rows), 4)
        self.assertEqual(frame.columns, ["id", "type", "x", "y", "z", "fx", "fy", "fz"])

    def test_rows_keyed_by_id(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "dump.lmp"
            _write_dump(p, DUMP_WELLFORMED)
            frame = compare.parse_dump_file(p)
        # id column holds 1..4; fx column holds 0.1*id. The parser's
        # contract is "rows keyed by the integer id column", so querying
        # by id should return the correct per-atom row regardless of the
        # file-order shuffle in the fixture.
        for atom_id in (1, 2, 3, 4):
            row = frame.rows[atom_id]
            self.assertAlmostEqual(row[5], 0.1 * atom_id, places=12)  # fx

    def test_rejects_mismatched_row_count(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "dump.lmp"
            _write_dump(
                p,
                """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 1
0 1
0 1
ITEM: ATOMS id type x y z fx fy fz
1 1 0 0 0 0 0 0
2 1 0 0 0 0 0 0
""",
            )
            with self.assertRaises(ValueError):
                compare.parse_dump_file(p)

    def test_rejects_missing_id_column(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "dump.lmp"
            _write_dump(
                p,
                """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 1
0 1
0 1
ITEM: ATOMS type x y z fx fy fz
1 0 0 0 0 0 0
""",
            )
            with self.assertRaises(ValueError):
                compare.parse_dump_file(p)


class CompareForcesTests(unittest.TestCase):
    def _make_frame(
        self,
        n: int,
        scale: float,
        perturb: dict[int, tuple[float, float, float]] | None = None,
    ) -> compare.DumpFrame:
        """Synthesise a dump frame with analytic forces f_i = scale*(i, i, i).

        ``perturb`` injects a delta at a specific atom id so tests can plant a
        known residual and check the compare layer pins the correct id.
        """
        rows: dict[int, list[float]] = {}
        for i in range(1, n + 1):
            fx = scale * i
            fy = scale * i
            fz = scale * i
            if perturb and i in perturb:
                dx, dy, dz = perturb[i]
                fx += dx
                fy += dy
                fz += dz
            rows[i] = [
                float(i),  # id
                1.0,  # type
                float(i),  # x
                0.0,
                0.0,  # y, z
                fx,
                fy,
                fz,
            ]
        return compare.DumpFrame(
            timestep=0,
            columns=["id", "type", "x", "y", "z", "fx", "fy", "fz"],
            rows=rows,
        )

    def test_identical_frames_pass(self):
        a = self._make_frame(10, scale=1.0)
        b = self._make_frame(10, scale=1.0)
        diff = compare.compare_forces(a, b, threshold_rel=1.0e-12)
        self.assertTrue(diff.passed)
        self.assertEqual(diff.max_abs_diff, 0.0)
        self.assertEqual(diff.max_rel_diff, 0.0)
        self.assertEqual(diff.n_atoms, 10)

    def test_detects_injected_component_delta(self):
        a = self._make_frame(10, scale=1.0)
        # Inject a fy delta on atom 7 (7.00 vs 7.07). `compare_forces` uses
        # `max(|a|,|b|)` for the denominator (mirrors the thermo comparator
        # and `docs/specs/verify/SPEC.md §3.4`), so rel = 0.07 / 7.07, not
        # 0.07 / 7.00.
        b = self._make_frame(10, scale=1.0, perturb={7: (0.0, 0.07, 0.0)})
        diff = compare.compare_forces(a, b, threshold_rel=1.0e-3)
        self.assertFalse(diff.passed)
        self.assertEqual(diff.at_atom_id, 7)
        self.assertEqual(diff.at_component, "fy")
        self.assertAlmostEqual(diff.max_rel_diff, 0.07 / 7.07, places=10)
        self.assertAlmostEqual(diff.max_abs_diff, 0.07, places=10)
        # A looser threshold flips the verdict.
        diff_loose = compare.compare_forces(a, b, threshold_rel=0.1)
        self.assertTrue(diff_loose.passed)

    def test_zero_forces_handled_without_division_error(self):
        zero_frame = compare.DumpFrame(
            timestep=0,
            columns=["id", "type", "x", "y", "z", "fx", "fy", "fz"],
            rows={
                1: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                2: [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        )
        diff = compare.compare_forces(zero_frame, zero_frame, threshold_rel=1.0e-12)
        self.assertTrue(diff.passed)
        self.assertEqual(diff.max_abs_diff, 0.0)
        self.assertEqual(diff.max_rel_diff, 0.0)

    def test_mismatched_id_sets_rejected(self):
        a = self._make_frame(5, scale=1.0)
        b = self._make_frame(6, scale=1.0)
        with self.assertRaises(ValueError):
            compare.compare_forces(a, b, threshold_rel=1.0e-10)

    def test_alignment_across_shuffled_files(self):
        # End-to-end: a file with ids written out-of-order should parse and
        # compare identically to one written in-order. This is the headline
        # invariant of T2.8 — LAMMPS shuffles during migration.
        in_order = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 1
0 1
0 1
ITEM: ATOMS id type x y z fx fy fz
1 1 0 0 0 0.1 0.2 0.3
2 1 0 0 0 0.4 0.5 0.6
3 1 0 0 0 0.7 0.8 0.9
"""
        shuffled = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 1
0 1
0 1
ITEM: ATOMS id type x y z fx fy fz
3 1 0 0 0 0.7 0.8 0.9
1 1 0 0 0 0.1 0.2 0.3
2 1 0 0 0 0.4 0.5 0.6
"""
        with tempfile.TemporaryDirectory() as d:
            p_a = pathlib.Path(d) / "a.lmp"
            p_b = pathlib.Path(d) / "b.lmp"
            _write_dump(p_a, in_order)
            _write_dump(p_b, shuffled)
            fa = compare.parse_dump_file(p_a)
            fb = compare.parse_dump_file(p_b)
        diff = compare.compare_forces(fa, fb, threshold_rel=1.0e-12)
        self.assertTrue(diff.passed)
        self.assertEqual(diff.max_rel_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
