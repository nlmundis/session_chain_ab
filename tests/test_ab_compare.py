"""Tests for ab_compare.py.

Run with:
    python3 -m unittest discover tests -v

No arm is ever launched here; every test works on synthetic modules in a
tempdir. The grader is the part worth testing, because it is what makes the
comparison a measurement rather than an impression: if grading is wrong, both
arms get scored wrong and the conclusion is confidently false.

The specific defect these guard against is an arm looking cheap by doing less.
``missing`` and ``attempted`` are asserted separately from ``correct`` for that
reason.
"""

import contextlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import ab_compare as ab  # noqa: E402

_MODULE_WITH_DOCSTRING = '''"""A docstring."""


def alpha():
    pass


async def beta():
    pass


class Gamma:
    def method_should_not_count(self):
        def nested_should_not_count():
            pass
        return nested_should_not_count
'''

_MODULE_WITHOUT_DOCSTRING = """import os


def only_one():
    return os
"""


class GroundTruth(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name)
        (self.repo / "a_with_doc.py").write_text(_MODULE_WITH_DOCSTRING)
        (self.repo / "b_no_doc.py").write_text(_MODULE_WITHOUT_DOCSTRING)
        self.addCleanup(self._tmp.cleanup)

    def test_counts_only_module_level_defs(self):
        """Methods and nested functions must not count, or the task grades its own ambiguity."""
        truth = ab.ground_truth(ab.select_files(self.repo, 10))
        self.assertEqual(truth["a_with_doc.py"].top_level_defs, 2)
        self.assertTrue(truth["a_with_doc.py"].has_docstring)
        self.assertEqual(truth["b_no_doc.py"].top_level_defs, 1)
        self.assertFalse(truth["b_no_doc.py"].has_docstring)

    def test_unparseable_module_is_omitted_rather_than_scored(self):
        (self.repo / "c_broken.py").write_text("def (((:\n")
        truth = ab.ground_truth(ab.select_files(self.repo, 10))
        self.assertNotIn("c_broken.py", truth)
        self.assertEqual(len(truth), 2)

    def test_selection_is_deterministic_and_bounded(self):
        first = ab.select_files(self.repo, 1)
        second = ab.select_files(self.repo, 1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)


class Grading(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.repo = self.dir / "repo"
        self.repo.mkdir()
        (self.repo / "a_with_doc.py").write_text(_MODULE_WITH_DOCSTRING)
        (self.repo / "b_no_doc.py").write_text(_MODULE_WITHOUT_DOCSTRING)
        self.truth = ab.ground_truth(ab.select_files(self.repo, 10))
        self.report = self.dir / "report.json"
        self.addCleanup(self._tmp.cleanup)

    def _write(self, files):
        self.report.write_text(json.dumps({"files": files}))

    def test_a_fully_correct_report_scores_clean(self):
        self._write(
            {
                "a_with_doc.py": {"top_level_defs": 2, "has_docstring": True},
                "b_no_doc.py": {"top_level_defs": 1, "has_docstring": False},
            }
        )
        score = ab.grade(self.report, self.truth)
        self.assertEqual((score["correct"], score["wrong"], score["missing"]), (2, 0, 0))

    def test_an_arm_that_skipped_files_is_marked_missing_not_ignored(self):
        """The defect that would make a lazy arm look cheap."""
        self._write({"a_with_doc.py": {"top_level_defs": 2, "has_docstring": True}})
        score = ab.grade(self.report, self.truth)
        self.assertEqual(score["correct"], 1)
        self.assertEqual(score["missing"], 1)
        self.assertEqual(score["attempted"], 1)
        self.assertEqual(score["total_files"], 2)

    def test_either_wrong_fact_fails_the_file(self):
        self._write(
            {
                "a_with_doc.py": {"top_level_defs": 3, "has_docstring": True},
                "b_no_doc.py": {"top_level_defs": 1, "has_docstring": True},
            }
        )
        score = ab.grade(self.report, self.truth)
        self.assertEqual(score["correct"], 0)
        self.assertEqual(sorted(score["wrong_names"]), ["a_with_doc.py", "b_no_doc.py"])

    def test_a_missing_or_unparseable_report_scores_zero_rather_than_raising(self):
        self.assertEqual(ab.grade(self.dir / "nope.json", self.truth)["correct"], 0)
        self.report.write_text("{not json")
        self.assertEqual(ab.grade(self.report, self.truth)["correct"], 0)

    def test_bare_mapping_without_the_files_wrapper_is_still_graded(self):
        """Tolerated on purpose: shape drift must not silently score an arm at zero."""
        self.report.write_text(
            json.dumps({"a_with_doc.py": {"top_level_defs": 2, "has_docstring": True}})
        )
        self.assertEqual(ab.grade(self.report, self.truth)["correct"], 1)


class PromptFairness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name)
        (self.repo / "a.py").write_text(_MODULE_WITHOUT_DOCSTRING)
        self.files = ab.select_files(self.repo, 10)
        self.addCleanup(self._tmp.cleanup)

    def test_arms_differ_only_in_the_batch_clause(self):
        """Diffs the two prompts line by line rather than sampling substrings.

        The substring version could not detect the confound its own docstring
        named: an extra instruction added to ONE arm passed, because every
        sampled substring was still present. Comparing the full texts makes the
        assertion match the claim.
        """
        single = ab._task_prompt(self.repo / "r.json", self.files, None)
        chain = ab._task_prompt(self.repo / "r.json", self.files, 4)
        only_single = [ln for ln in single.splitlines() if ln not in chain.splitlines()]
        only_chain = [ln for ln in chain.splitlines() if ln not in single.splitlines()]
        self.assertEqual(len(only_single), 1, f"unexpected single-only text: {only_single}")
        self.assertEqual(len(only_chain), 1, f"unexpected chain-only text: {only_chain}")
        self.assertIn("one session", only_single[0])
        self.assertIn("at most 4 files", only_chain[0])

    def test_both_prompts_name_every_file_and_the_report_path(self):
        for batch in (None, 4):
            prompt = ab._task_prompt(self.repo / "report.json", self.files, batch)
            self.assertIn("a.py", prompt)
            self.assertIn("report.json", prompt)


class Summary(unittest.TestCase):
    def _chain(self, cost, tokens, turns, iterations=1, stop="done"):
        import session_loop as sl

        results = tuple(
            sl.IterationResult(
                verdict=sl.IterationVerdict.OK,
                session_id=f"s{i}",
                cost_usd=cost / iterations,
                cost_is_known=True,
                context_tokens=tokens // iterations,
                num_turns=turns // iterations,
                wall_seconds=1.0,
                denials=(),
                done=True,
                structured_output={"done": True},
                text="",
            )
            for i in range(iterations)
        )
        return sl.ChainResult(iterations=results, stop_reason=stop)

    def test_cost_is_normalised_by_correct_files_not_attempted(self):
        chain = self._chain(cost=2.0, tokens=100_000, turns=10)
        score = {
            "correct": 4,
            "attempted": 8,
            "total_files": 8,
            "wrong": 4,
            "missing": 0,
            "wrong_names": [],
        }
        summary = ab.summarise("single", chain, score)
        self.assertAlmostEqual(summary["cost_per_correct"], 0.5)
        self.assertEqual(summary["tokens_per_correct"], 25_000)

    def test_zero_correct_reports_none_rather_than_dividing_by_zero(self):
        summary = ab.summarise(
            "chain",
            self._chain(cost=1.0, tokens=10, turns=1),
            {
                "correct": 0,
                "attempted": 0,
                "total_files": 3,
                "wrong": 0,
                "missing": 3,
                "wrong_names": [],
            },
        )
        self.assertIsNone(summary["cost_per_correct"])
        self.assertIsNone(summary["tokens_per_correct"])


class RecursiveSelection(unittest.TestCase):
    """Four of the six published points required the recursive path; it was untested."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name)
        (self.repo / "b_root.py").write_text(_MODULE_WITHOUT_DOCSTRING)
        for sub in ("lib", "tests", "venv"):
            (self.repo / sub).mkdir()
            (self.repo / sub / f"a_{sub}.py").write_text(_MODULE_WITH_DOCSTRING)
        self.addCleanup(self._tmp.cleanup)

    def test_recursive_reaches_subdirectories_and_skips_venv(self):
        found = [p.name for p in ab.select_files(self.repo, 99, recursive=True)]
        self.assertIn("a_lib.py", found)
        self.assertIn("a_tests.py", found)
        self.assertNotIn("a_venv.py", found)

    def test_non_recursive_sees_only_the_root(self):
        self.assertEqual([p.name for p in ab.select_files(self.repo, 99)], ["b_root.py"])

    def test_prefix_property_holds_within_one_mode(self):
        small = ab.select_files(self.repo, 2, recursive=True)
        large = ab.select_files(self.repo, 99, recursive=True)
        self.assertEqual(small, large[:2])

    def test_prefix_property_does_not_span_the_recursive_boundary(self):
        """Pins the documented limit, so nobody re-derives it by ruining a curve."""
        flat = ab.select_files(self.repo, 1)
        deep = ab.select_files(self.repo, 1, recursive=True)
        self.assertNotEqual(flat, deep)

    def test_truth_is_keyed_by_relative_path_not_basename(self):
        files = ab.select_files(self.repo, 99, recursive=True)
        keys = set(ab.ground_truth(files, repo=self.repo))
        self.assertIn("lib/a_lib.py", keys)
        self.assertIn("tests/a_tests.py", keys)

    def test_two_modules_sharing_a_basename_stay_distinct(self):
        (self.repo / "lib" / "same.py").write_text(_MODULE_WITH_DOCSTRING)
        (self.repo / "tests" / "same.py").write_text(_MODULE_WITHOUT_DOCSTRING)
        files = ab.select_files(self.repo, 99, recursive=True)
        truth = ab.ground_truth(files, repo=self.repo)
        self.assertIn("lib/same.py", truth)
        self.assertIn("tests/same.py", truth)
        self.assertNotEqual(truth["lib/same.py"], truth["tests/same.py"])

    def test_prompt_lists_paths_the_arm_can_actually_open(self):
        files = ab.select_files(self.repo, 99, recursive=True)
        prompt = ab._task_prompt(self.repo / "r.json", files, None, repo=self.repo)
        self.assertIn("lib/a_lib.py", prompt)
        self.assertIn("tests/a_tests.py", prompt)


class StrictGrading(unittest.TestCase):
    """A malformed report must fail, never coincide with the right answer."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.repo = self.dir / "repo"
        self.repo.mkdir()
        (self.repo / "b_no_doc.py").write_text(_MODULE_WITHOUT_DOCSTRING)
        self.truth = ab.ground_truth(ab.select_files(self.repo, 9), repo=self.repo)
        self.report = self.dir / "report.json"
        self.addCleanup(self._tmp.cleanup)

    def _score(self, entry):
        self.report.write_text(json.dumps({"files": {"b_no_doc.py": entry}}))
        return ab.grade(self.report, self.truth)

    def test_boolean_true_does_not_satisfy_a_count_of_one(self):
        self.assertEqual(self._score({"top_level_defs": True, "has_docstring": False})["correct"], 0)

    def test_a_truthy_string_does_not_satisfy_a_false_docstring(self):
        self.assertEqual(self._score({"top_level_defs": 1, "has_docstring": "no"})["correct"], 0)

    def test_an_integer_does_not_satisfy_a_boolean_docstring_answer(self):
        """1 == True in Python, so a report answering 1 for a docstring would coincide."""
        (self.repo / "a_with_doc.py").write_text(_MODULE_WITH_DOCSTRING)
        truth = ab.ground_truth(ab.select_files(self.repo, 9), repo=self.repo)
        self.report.write_text(json.dumps({"files": {"a_with_doc.py": {"top_level_defs": 2, "has_docstring": 1}}}))
        self.assertEqual(ab.grade(self.report, truth)["correct"], 0)

    def test_a_numeric_string_does_not_satisfy_a_count(self):
        self.assertEqual(self._score({"top_level_defs": "1", "has_docstring": False})["correct"], 0)

    def test_the_correct_types_still_pass(self):
        self.assertEqual(self._score({"top_level_defs": 1, "has_docstring": False})["correct"], 1)

    def test_wrong_names_are_not_truncated(self):
        for i in range(15):
            (self.repo / f"m{i:02d}.py").write_text(_MODULE_WITHOUT_DOCSTRING)
        truth = ab.ground_truth(ab.select_files(self.repo, 99), repo=self.repo)
        self.report.write_text(
            json.dumps({"files": {n: {"top_level_defs": 99, "has_docstring": True} for n in truth}})
        )
        score = ab.grade(self.report, truth)
        self.assertEqual(len(score["wrong_names"]), score["wrong"])
        self.assertGreater(score["wrong"], 10)

    def test_a_non_dict_entries_blob_scores_zero_rather_than_raising(self):
        self.report.write_text(json.dumps({"files": ["not", "a", "mapping"]}))
        self.assertEqual(ab.grade(self.report, self.truth)["correct"], 0)


class SummaryKeepsErrorEvidence(unittest.TestCase):
    def test_wrong_and_missing_survive_into_the_durable_summary(self):
        import session_loop as sl

        chain = sl.ChainResult(iterations=(), stop_reason="done")
        score = {
            "correct": 3,
            "attempted": 4,
            "total_files": 5,
            "wrong": 1,
            "missing": 1,
            "wrong_names": ["a.py"],
        }
        summary = ab.summarise("single", chain, score)
        self.assertEqual(summary["wrong"], 1)
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["wrong_names"], ["a.py"])


class ProvenanceRecord(unittest.TestCase):
    """Without this record no committed measurement can be reproduced."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name)
        (self.repo / "a.py").write_text(_MODULE_WITH_DOCSTRING)
        (self.repo / "bad.py").write_text("def (((:\n")
        self.addCleanup(self._tmp.cleanup)

    def test_it_names_the_flag_the_count_and_the_file_list(self):
        args = ab._build_parser().parse_args(
            ["--out", str(self.repo), "--files", "9", "--recursive", "--batch", "4"]
        )
        args.repo = self.repo
        files = ab.select_files(self.repo, 9, recursive=True)
        truth = ab.ground_truth(files, repo=self.repo)
        rec = ab.provenance(args, files, truth)
        self.assertTrue(rec["recursive"])
        self.assertEqual(rec["batch"], 4)
        self.assertEqual(rec["files_selected"], len(files))
        self.assertIn("a.py", rec["file_list"])

    def test_it_names_the_repository_without_its_absolute_path(self):
        """Committed provenance would otherwise publish the runner's home directory."""
        args = ab._build_parser().parse_args(["--out", str(self.repo), "--files", "9"])
        args.repo = self.repo
        files = ab.select_files(self.repo, 9)
        rec = ab.provenance(args, files, ab.ground_truth(files, repo=self.repo))
        self.assertEqual(rec["repo"], self.repo.resolve().name)
        self.assertNotIn(str(self.repo.resolve().parent), json.dumps(rec))

    def test_unparseable_files_are_counted_not_absorbed(self):
        args = ab._build_parser().parse_args(["--out", str(self.repo), "--files", "9"])
        args.repo = self.repo
        files = ab.select_files(self.repo, 9)
        rec = ab.provenance(args, files, ab.ground_truth(files, repo=self.repo))
        self.assertEqual(rec["files_dropped_unparseable"], 1)

    def test_repo_commit_degrades_to_a_marker_outside_git(self):
        self.assertIsInstance(ab.repo_commit(self.repo), str)


class ArmExecution(unittest.TestCase):
    """run_arm sets every experimental control and was executed by no test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name)
        (self.repo / "a.py").write_text(_MODULE_WITH_DOCSTRING)
        self.addCleanup(self._tmp.cleanup)

    def _run(self, **kw):
        import session_loop as sl

        captured = {}

        def fake_chain(prompt, **kwargs):
            captured["prompt"] = prompt
            captured.update(kwargs)
            return sl.ChainResult(iterations=(), stop_reason="done")

        real = sl.run_chain
        sl.run_chain = fake_chain
        try:
            chain, score = ab.run_arm(
                "single",
                repo=self.repo,
                files=ab.select_files(self.repo, 9),
                report_path=self.repo / "rep.json",
                ledger=self.repo / "l.jsonl",
                batch=None,
                max_iterations=1,
                budget_usd=3.0,
                **kw,
            )
        finally:
            sl.run_chain = real
        return captured, score

    def test_the_tool_allowlist_reaches_the_chain(self):
        """--allowedTools only pre-approves; --tools is what removes Bash, Grep and Glob."""
        captured, _ = self._run()
        self.assertEqual(captured["extra_args"], ["--tools", "Read,Write", "--allowedTools", ab._ALLOWED_TOOLS])
        self.assertEqual(ab._TOOLS.split(","), ab._ALLOWED_TOOLS.split())
        self.assertNotIn("Bash", ab._ALLOWED_TOOLS)
        self.assertNotIn("Grep", ab._ALLOWED_TOOLS)

    def test_permission_mode_and_schema_reach_the_chain(self):
        captured, _ = self._run()
        self.assertEqual(captured["permission_mode"], "acceptEdits")
        self.assertIn("done", json.loads(captured["schema"])["required"])

    def test_a_stale_report_is_removed_before_the_arm_starts(self):
        (self.repo / "rep.json").write_text(json.dumps({"files": {"a.py": {}}}))
        self._run()
        self.assertFalse((self.repo / "rep.json").exists())

    def test_truth_is_snapshotted_before_the_arm_can_edit_the_repo(self):
        """The arm holds Write; grading must not use a repo it could have changed."""
        import session_loop as sl

        def editing_chain(prompt, **kwargs):
            (self.repo / "a.py").write_text("def added_by_the_arm():\n    pass\n")
            return sl.ChainResult(iterations=(), stop_reason="done")

        real = sl.run_chain
        sl.run_chain = editing_chain
        try:
            (self.repo / "rep.json").write_text("")
            _, score = ab.run_arm(
                "single",
                repo=self.repo,
                files=ab.select_files(self.repo, 9),
                report_path=self.repo / "rep.json",
                ledger=self.repo / "l.jsonl",
                batch=None,
                max_iterations=1,
                budget_usd=1.0,
            )
        finally:
            sl.run_chain = real
        # Pre-edit truth: 2 top-level defs and a docstring. Post-edit it would be 1
        # and no docstring, so a truth taken afterwards would grade a different task.
        self.assertEqual(score["total_files"], 1)
        self.assertEqual(score["missing"], 1)

    def test_a_report_matching_the_edited_module_is_graded_wrong(self):
        """Distinguishes a pre-edit snapshot from a post-edit one, which the test above cannot."""
        import session_loop as sl

        def editing_chain(prompt, **kwargs):
            (self.repo / "a.py").write_text("def added_by_the_arm():\n    pass\n")
            (self.repo / "rep.json").write_text(json.dumps(
                {"files": {"a.py": {"top_level_defs": 1, "has_docstring": False}}}))
            return sl.ChainResult(iterations=(), stop_reason="done")

        real = sl.run_chain
        sl.run_chain = editing_chain
        try:
            _, score = ab.run_arm("single", repo=self.repo, files=ab.select_files(self.repo, 9),
                                  report_path=self.repo / "rep.json", ledger=self.repo / "l.jsonl",
                                  batch=None, max_iterations=1, budget_usd=1.0)
        finally:
            sl.run_chain = real
        self.assertEqual((score["correct"], score["wrong"], score["missing"]), (0, 1, 0))


class ExitCode(unittest.TestCase):
    def test_an_arm_that_stopped_early_does_not_exit_zero(self):
        import session_loop as sl

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = pathlib.Path(tmp.name)
        (repo / "a.py").write_text(_MODULE_WITH_DOCSTRING)

        def dead_chain(prompt, **kwargs):
            return sl.ChainResult(iterations=(), stop_reason="degraded_by_denial")

        real = sl.run_chain
        sl.run_chain = dead_chain
        try:
            code = ab.main(
                ["--repo", str(repo), "--files", "1", "--arm", "single", "--out", str(repo / "out")]
            )
        finally:
            sl.run_chain = real
        self.assertEqual(code, 1)

    def test_no_modules_found_is_an_error_not_a_result(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        empty = pathlib.Path(tmp.name)
        self.assertEqual(
            ab.main(["--repo", str(empty), "--files", "4", "--out", str(empty / "o")]), 1
        )


class SingleFilePool(unittest.TestCase):
    """A one-file pool must key on the filename, not on the common-path artifact."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name)
        (self.repo / "only.py").write_text(_MODULE_WITHOUT_DOCSTRING)
        self.addCleanup(self._tmp.cleanup)

    def test_truth_key_is_the_filename_not_a_dot(self):
        files = ab.select_files(self.repo, 1)
        truth = ab.ground_truth(files)
        self.assertEqual(list(truth), ["only.py"])
        self.assertNotIn(".", truth)

    def test_the_prompt_names_the_file_not_a_dot(self):
        files = ab.select_files(self.repo, 1)
        prompt = ab._task_prompt(self.repo / "r.json", files, None)
        self.assertIn("only.py", prompt)

    def test_a_report_using_that_key_scores_correct(self):
        files = ab.select_files(self.repo, 1)
        truth = ab.ground_truth(files)
        report = self.repo / "r.json"
        report.write_text(
            json.dumps({"files": {"only.py": {"top_level_defs": 1, "has_docstring": False}}})
        )
        self.assertEqual(ab.grade(report, truth)["correct"], 1)


class PathsOutsideTheRoot(unittest.TestCase):
    """A file that is not under the common root must degrade, not raise."""

    def test_truth_and_prompt_fall_back_to_the_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            other = pathlib.Path(tmp) / "other"
            repo.mkdir()
            other.mkdir()
            (repo / "a.py").write_text(_MODULE_WITHOUT_DOCSTRING)
            (other / "b.py").write_text(_MODULE_WITHOUT_DOCSTRING)
            files = [repo / "a.py", other / "b.py"]
            truth = ab.ground_truth(files, repo=repo)
            self.assertIn("a.py", truth)
            self.assertIn(str(other / "b.py"), truth)
            prompt = ab._task_prompt(repo / "r.json", files, None, repo=repo)
            self.assertIn(str(other / "b.py"), prompt)


class SummaryCarriesForwardAcrossSittings(unittest.TestCase):
    """Running the arms separately must not delete the earlier arm's result."""

    def test_a_second_arm_preserves_the_first(self):
        import session_loop as sl

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = pathlib.Path(tmp.name)
        (repo / "a.py").write_text(_MODULE_WITH_DOCSTRING)
        out = repo / "out"
        out.mkdir()
        args = ab._build_parser().parse_args(["--repo", str(repo), "--files", "1", "--out", str(out)])
        args.repo = repo.resolve()
        key = ab.provenance_key(args, ab.ground_truth(ab.select_files(args.repo, 1), repo=args.repo))
        (out / "summary.json").write_text(json.dumps([{"arm": "single", "correct": 1, "provenance_key": key}]))

        def fake_chain(prompt, **kwargs):
            return sl.ChainResult(iterations=(), stop_reason="done")

        real = sl.run_chain
        sl.run_chain = fake_chain
        try:
            ab.main(
                ["--repo", str(repo), "--files", "1", "--arm", "chain", "--out", str(out)]
            )
        finally:
            sl.run_chain = real
        arms = {s["arm"] for s in json.loads((out / "summary.json").read_text())}
        self.assertEqual(arms, {"single", "chain"})

    def test_an_arm_from_a_different_task_is_refused_before_anything_runs(self):
        import session_loop as sl

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = pathlib.Path(tmp.name)
        (repo / "a.py").write_text(_MODULE_WITH_DOCSTRING)
        out = repo / "out"
        out.mkdir()
        calls = []

        def fake_chain(prompt, **kwargs):
            calls.append(kwargs)
            return sl.ChainResult(iterations=(), stop_reason="done")

        real = sl.run_chain
        sl.run_chain = fake_chain
        try:
            for existing in ([{"arm": "single", "correct": 1}], [{"arm": "single", "provenance_key": "other"}]):
                (out / "summary.json").write_text(json.dumps(existing))
                with self.subTest(existing=existing), contextlib.redirect_stderr(io.StringIO()):
                    code = ab.main(["--repo", str(repo), "--files", "1", "--arm", "chain", "--out", str(out)])
                    self.assertEqual(code, 1)
        finally:
            sl.run_chain = real
        self.assertEqual(calls, [], "nothing may run, and so nothing may be spent")

    def test_a_corrupt_existing_summary_is_ignored_rather_than_fatal(self):
        import session_loop as sl

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = pathlib.Path(tmp.name)
        (repo / "a.py").write_text(_MODULE_WITH_DOCSTRING)
        out = repo / "out"
        out.mkdir()
        (out / "summary.json").write_text("{not json")

        def fake_chain(prompt, **kwargs):
            return sl.ChainResult(iterations=(), stop_reason="done")

        real = sl.run_chain
        sl.run_chain = fake_chain
        try:
            ab.main(["--repo", str(repo), "--files", "1", "--arm", "single", "--out", str(out)])
        finally:
            sl.run_chain = real
        self.assertTrue((out / "provenance.json").exists())

    def test_repo_commit_returns_a_real_sha_inside_a_git_repo(self):
        sha = ab.repo_commit(pathlib.Path(__file__).resolve().parent.parent)
        self.assertRegex(sha, r"^[0-9a-f]{40}$|^unknown$")



class MainControls(unittest.TestCase):
    """The controls main() applies before and after the arms run."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = pathlib.Path(self._tmp.name) / "repo"
        self.repo.mkdir()
        for i in range(5):
            (self.repo / f"m{i}.py").write_text(_MODULE_WITH_DOCSTRING)

    def _main(self, argv, chain=None):
        import session_loop as sl

        calls = []

        def default_chain(prompt, **kwargs):
            return sl.ChainResult(iterations=(), stop_reason="done")

        def recording(prompt, **kwargs):
            calls.append({"prompt": prompt, **kwargs})
            return (chain or default_chain)(prompt, **kwargs)

        real = sl.run_chain
        sl.run_chain = recording
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = ab.main(argv)
        finally:
            sl.run_chain = real
        return code, calls

    def test_a_relative_out_is_resolved_before_the_prompt_names_it(self):
        cwd = os.getcwd()
        os.chdir(self._tmp.name)
        try:
            _, calls = self._main(["--repo", "repo", "--files", "5", "--out", "out_rel"])
        finally:
            os.chdir(cwd)
        expected = str((pathlib.Path(self._tmp.name) / "out_rel" / "report_single.json").resolve())
        self.assertIn(expected, calls[0]["prompt"])
        self.assertTrue(pathlib.Path(calls[0]["cwd"]).is_absolute())

    def test_the_single_arm_gets_one_session_and_the_chain_enough_to_finish(self):
        _, calls = self._main(["--repo", str(self.repo), "--files", "5", "--batch", "2", "--out", str(self.repo.parent / "o")])
        single, chain = calls
        self.assertEqual(single["max_iterations"], 1)
        self.assertEqual(chain["max_iterations"], 3 + 2)

    def test_a_pool_smaller_than_requested_is_refused_unless_allowed(self):
        out = str(self.repo.parent / "o")
        code, calls = self._main(["--repo", str(self.repo), "--files", "24", "--out", out])
        self.assertEqual((code, calls), (1, []))
        _, calls = self._main(["--repo", str(self.repo), "--files", "24", "--allow-short-pool", "--out", out])
        self.assertEqual(len(calls), 2)

    def test_every_arm_is_graded_against_the_snapshot_taken_before_any_arm_ran(self):
        import session_loop as sl

        out = self.repo.parent / "o"

        def editing(prompt, **kwargs):
            # Write a report that is correct for the modules AS THEY WERE, then
            # edit one. Graded against the single pre-run snapshot, both arms
            # score 5; re-snapshotting per arm would mark m0 wrong for the second.
            report = re.search(r"Maintain a JSON report at (\S+) ", prompt).group(1)
            entry = {"top_level_defs": 2, "has_docstring": True}
            pathlib.Path(report).write_text(json.dumps({"files": {f"m{i}.py": entry for i in range(5)}}))
            (self.repo / "m0.py").write_text("def x():\n    pass\n")
            return sl.ChainResult(iterations=(), stop_reason="done")

        code, _ = self._main(["--repo", str(self.repo), "--files", "5", "--batch", "5", "--out", str(out)], chain=editing)
        truth = json.loads((out / "ground_truth.json").read_text())
        self.assertEqual(truth["m0.py"]["top_level_defs"], 2)
        summaries = json.loads((out / "summary.json").read_text())
        self.assertTrue(all(s.get("repo_modified_during_run") for s in summaries))
        self.assertEqual({s["arm"]: s["correct"] for s in summaries}, {"single": 5, "chain": 5})
        self.assertEqual(code, 1)


class PoolExclusions(unittest.TestCase):
    def test_hidden_and_environment_directories_are_never_drawn_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            for rel in (".venv/lib/site-packages/pkg/__init__.py", "env/x.py", "node_modules/y.py",
                        ".tox/z.py", "src/app.py", "tests/.hidden/w.py"):
                path = repo / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x = 1\n")
            found = [str(p.relative_to(repo)) for p in ab.select_files(repo, 99, recursive=True)]
        self.assertEqual(found, ["src/app.py"])


class NonIntegralCounts(unittest.TestCase):
    def test_a_fractional_count_is_wrong(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            (repo / "a.py").write_text(_MODULE_WITH_DOCSTRING)
            truth = ab.ground_truth(ab.select_files(repo, 9), repo=repo)
            report = repo / "r.json"
            report.write_text(json.dumps({"files": {"a.py": {"top_level_defs": 2.5, "has_docstring": True}}}))
            self.assertEqual(ab.grade(report, truth)["correct"], 0)
            report.write_text(json.dumps({"files": {"a.py": {"top_level_defs": 2.0, "has_docstring": True}}}))
            self.assertEqual(ab.grade(report, truth)["correct"], 1)


if __name__ == "__main__":
    unittest.main()
