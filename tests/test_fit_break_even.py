"""Tests for fit_break_even.py.

Run with:
    python3 -m unittest discover tests -v

This module produces the branch's published number and previously had no tests at
all, which is the wrong way round: an arithmetic error here does not crash, it
prints a confident break-even that is simply wrong.

Two groups matter most. The crossing tests feed the solver coefficient sets whose
answers are known in advance -- including the degenerate ones the earlier
``max(r1, r2)`` handled by raising or by reporting a crossing in the wrong
direction. The load tests pin the checks that stop a mislabelled measurement
directory from silently becoming a data point.
"""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import fit_break_even as fb  # noqa: E402


def _summary(arm, n, cost, **overrides):
    """Build a summary.json payload shaped like ab_compare.summarise's output."""
    row = {
        "arm": arm,
        "iterations": 1,
        "stop_reason": "done",
        "turns": 10,
        "context_tokens": 100_000,
        "cost_usd": cost,
        "cost_is_complete": True,
        "correct": n,
        "attempted": n,
        "wrong": 0,
        "missing": 0,
        "wrong_names": [],
        "total_files": n,
    }
    row.update(overrides)
    return [row]


class Solver(unittest.TestCase):
    def test_solves_a_known_system(self):
        # x + y = 3, x - y = 1  ->  x = 2, y = 1
        x, y = fb.solve([[1.0, 1.0, 3.0], [1.0, -1.0, 1.0]])
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 1.0)

    def test_a_singular_system_raises_rather_than_returning_nonsense(self):
        with self.assertRaises(ValueError):
            fb.solve([[1.0, 1.0, 2.0], [2.0, 2.0, 4.0]])


class Crossing(unittest.TestCase):
    """The degenerate cases the previous max(r1, r2) got wrong."""

    def test_a_normal_superlinear_single_arm_crosses_once_in_the_positive_range(self):
        # single = 1 + 0.05N + 0.001N^2 vs chain = 0.19N
        got = fb.crossing((1.0, 0.05, 0.001), (0.0, 0.19))
        self.assertTrue(got.describes_a_threshold)
        assert got.n is not None
        self.assertGreater(got.n, 100)

    def test_a_sublinear_single_arm_reports_no_crossing_rather_than_a_reversed_one(self):
        """Negative quadratic term: the single arm is never overtaken.

        max(r1, r2) would return a positive root here and present it as a
        break-even, pointing the conclusion the wrong way.
        """
        self.assertEqual(fb.crossing((1.0, 0.05, -0.001), (0.0, 0.19)).kind, "single_always")

    def test_parallel_lines_do_not_divide_by_zero(self):
        """Same slope, single arm dearer by a constant: the chain simply wins throughout."""
        got = fb.crossing((1.0, 0.19, 0.0), (0.0, 0.19))
        self.assertEqual(got.kind, "chain_always")
        self.assertIsNone(got.n)

    def test_a_linear_single_arm_cheaper_above_the_root_is_not_called_a_break_even(self):
        """The bug the sweep caught, using this module's own published coefficients.

        Zeroing only the quadratic term, the old code answered 6.28 -- while the
        single arm is in fact cheaper ABOVE that point (5.94 vs 18.86 at N=100).
        A root is only a threshold if the CHAIN wins above it.
        """
        got = fb.crossing((0.8658, 0.050726, 0.0), (0.0, 0.188645))
        self.assertEqual(got.kind, "single_always")
        self.assertIsNone(got.n)

    def test_a_linear_single_arm_that_grows_faster_does_cross(self):
        """Single starts cheaper (no fixed cost) but climbs faster; chain wins above 50."""
        got = fb.crossing((0.0, 0.29, 0.0), (5.0, 0.19))
        self.assertEqual(got.kind, "crossing")
        assert got.n is not None
        self.assertAlmostEqual(got.n, 50.0)

    def test_a_negative_discriminant_says_the_chain_always_wins_not_never(self):
        """The second reversed case: chain cheaper everywhere was printed as never."""
        got = fb.crossing((5.0, 0.5, 0.001), (0.0, 0.1))
        self.assertEqual(got.kind, "chain_always")

    def test_only_positive_crossings_are_reported(self):
        self.assertIsNone(fb.crossing((0.0, 0.5, 0.001), (0.0, 0.1)).n)


class Fitting(unittest.TestCase):
    def _points(self, arm, pairs):
        return [fb.Point(arm=arm, n=n, cost_usd=c, context_tokens=1000 * n, turns=n) for n, c in pairs]

    def test_a_quadratic_is_recovered_exactly_from_points_on_it(self):
        pts = self._points("single", [(n, 2.0 + 0.5 * n + 0.01 * n * n) for n in (10, 20, 30, 40)])
        a, b, c = fb.fit_quadratic(pts)
        self.assertAlmostEqual(a, 2.0, places=4)
        self.assertAlmostEqual(b, 0.5, places=5)
        self.assertAlmostEqual(c, 0.01, places=6)

    def test_through_origin_pins_the_intercept_to_zero(self):
        pts = self._points("single", [(n, 0.5 * n + 0.01 * n * n) for n in (10, 20, 30, 40)])
        a, b, c = fb.fit_quadratic(pts, through_origin=True)
        self.assertEqual(a, 0.0)
        self.assertAlmostEqual(b, 0.5, places=5)

    def test_a_line_through_the_origin_recovers_its_slope(self):
        pts = self._points("chain", [(n, 0.2 * n) for n in (24, 48)])
        intercept, slope = fb.fit_line(pts, through_origin=True)
        self.assertEqual(intercept, 0.0)
        self.assertAlmostEqual(slope, 0.2, places=6)

    def test_a_free_line_recovers_both_coefficients(self):
        pts = self._points("chain", [(n, 1.0 + 0.2 * n) for n in (24, 48)])
        intercept, slope = fb.fit_line(pts, through_origin=False)
        self.assertAlmostEqual(intercept, 1.0, places=4)
        self.assertAlmostEqual(slope, 0.2, places=6)

    def test_the_three_fits_disagree_which_is_the_whole_point_of_a_band(self):
        single = self._points("single", [(n, 0.9 + 0.05 * n + 0.0014 * n * n) for n in (12, 24, 48, 72)])
        chain = self._points("chain", [(n, 0.19 * n) for n in (24, 48)])
        fits = fb.build_fits(single, chain)
        self.assertEqual(len(fits), 3)
        crossings = [f.crossing for f in fits if f.crossing is not None]
        self.assertGreater(max(crossings) - min(crossings), 1.0)

    def test_the_report_states_that_the_band_is_beyond_the_measured_range(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fb.main([])
        out = buf.getvalue()
        self.assertIn("EXTRAPOLATION", out)
        self.assertIn("largest task size actually measured", out)


class LoadingChecksTheDataItReads(unittest.TestCase):
    """The directory name is not evidence; every one of these was unchecked before."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, arm, n, payload):
        d = self.root / f"read_{arm}_{n}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(payload))

    def test_a_well_formed_measurement_loads(self):
        self._write("single", 24, _summary("single", 24, 2.9297))
        point = fb.load_point(self.root, "single", 24)
        self.assertEqual(point.n, 24)
        self.assertAlmostEqual(point.cost_usd, 2.9297)

    def test_a_directory_whose_summary_names_a_different_arm_is_rejected(self):
        self._write("single", 24, _summary("chain", 24, 2.9))
        with self.assertRaises(ValueError):
            fb.load_point(self.root, "single", 24)

    def test_a_total_files_disagreeing_with_the_directory_name_is_rejected(self):
        self._write("single", 24, _summary("single", 48, 2.9))
        with self.assertRaises(ValueError) as ctx:
            fb.load_point(self.root, "single", 24)
        self.assertIn("total_files", str(ctx.exception))

    def test_an_arm_that_did_not_score_100_percent_is_rejected(self):
        self._write("single", 24, _summary("single", 24, 2.9, correct=23))
        with self.assertRaises(ValueError):
            fb.load_point(self.root, "single", 24)

    def test_an_incomplete_cost_is_rejected(self):
        self._write("single", 24, _summary("single", 24, 2.9, cost_is_complete=False))
        with self.assertRaises(ValueError):
            fb.load_point(self.root, "single", 24)

    def test_a_two_arm_summary_is_rejected_rather_than_indexed_blindly(self):
        self._write("single", 24, _summary("single", 24, 2.9) + _summary("single", 24, 9.9))
        with self.assertRaises(ValueError):
            fb.load_point(self.root, "single", 24)

    def test_a_missing_file_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            fb.load_point(self.root, "single", 99)


class CommandLine(unittest.TestCase):
    def test_running_with_no_arguments_does_not_raise(self):
        """The old version read sys.argv[1] at module scope and raised IndexError."""
        args = fb._build_parser().parse_args([])
        self.assertTrue(str(args.measurements).endswith("measurements"))

    def test_missing_measurements_exit_nonzero_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(fb.main([tmp]), 1)

    def test_the_committed_measurements_still_produce_a_band(self):
        """Guards the published result itself against a silent change in the data."""
        code = fb.main(["--json"])
        self.assertEqual(code, 0)

    def test_the_human_report_prints_a_band_and_never_a_bare_point(self):
        """The correction this module carries: no single number in the output."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = fb.main(["--check-alternative"])
        out = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("BREAK-EVEN BAND", out)
        self.assertIn("Quote the band, never a single number", out)
        self.assertIn("EXTRAPOLATION: the largest task size actually measured is N = 72", out)
        self.assertNotIn("BREAK-EVEN: N =", out)

    def test_the_report_names_all_three_fits_so_the_asymmetry_stays_visible(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fb.main([])
        out = buf.getvalue()
        for name in ("published", "symmetric A", "symmetric B"):
            self.assertIn(name, out)

    def test_residuals_are_printed_at_more_precision_than_the_claim_they_broke(self):
        """The commit said "within 2.2%" off a one-decimal display; it was -2.2370%."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fb.main([])
        self.assertIn("-2.2370%", buf.getvalue())

    def test_the_alternative_model_is_reported_when_asked_for(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fb.main(["--check-alternative"])
        self.assertIn("constant price per", buf.getvalue())

    def test_a_summary_that_is_not_a_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "read_single_24").mkdir()
            (root / "read_single_24" / "summary.json").write_text(json.dumps({"arm": "single"}))
            with self.assertRaises(ValueError):
                fb.load_point(root, "single", 24)


class AlternativeModel(unittest.TestCase):
    def test_the_constant_price_check_returns_one_row_per_point(self):
        pts = [
            fb.Point(arm="single", n=n, cost_usd=c, context_tokens=t, turns=k)
            for n, c, t, k in [(12, 1.63, 652085, 19), (24, 2.93, 1782826, 38)]
        ]
        rows = fb.constant_price_residuals(pts)
        self.assertEqual(len(rows), 2)
        for _point, fitted, err in rows:
            self.assertIsInstance(fitted, float)
            self.assertIsInstance(err, float)


if __name__ == "__main__":
    unittest.main()
