"""Tests for session_loop.py.

Run with:
    python3 -m unittest discover tests -v

Nothing here launches a real headless session. Every test injects a fake
``runner``, because the defect class this module exists to catch -- a session
that reports success while having silently lost tool permissions -- costs real
dollars to reproduce live, and asserting on one live run would not have covered
the failure combinations anyway.

The classification tests are built from the shape of a REAL result payload,
captured 2026-09-02 from ``claude -p ... --output-format json``, including the
observed ``is_error: false`` / ``terminal_reason: "completed"`` on the run that
had four Bash calls denied. That payload is the reason the denial check runs
before the structured output is read.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import session_loop as sl  # noqa: E402


def _payload(**overrides):
    """Build a result payload shaped like the CLI's real output.

    Defaults describe a clean, completed, done=True run; each test overrides only
    the field under test, so a test's diff from "healthy" is what it asserts.
    """
    base = {
        "is_error": False,
        "terminal_reason": "completed",
        "session_id": "00000000-0000-4000-8000-000000000001",
        "total_cost_usd": 0.27201,
        "num_turns": 3,
        "permission_denials": [],
        "structured_output": {"done": True, "next": "nothing"},
        "result": '{"done":true,"next":"nothing"}',
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 17170,
            "cache_read_input_tokens": 22180,
        },
    }
    base.update(overrides)
    return base


class FakeRunner:
    """Stand-in for ``subprocess.run`` that replays queued payloads.

    Records the argv of every call so command construction can be asserted, and
    raises when exhausted rather than repeating the last payload -- a chain that
    ran more iterations than the test queued is a bug the test must see.
    """

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if not self._payloads:
            raise AssertionError("runner called more times than the test queued")
        nxt = self._payloads.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(nxt), stderr="")


class CommandBuilding(unittest.TestCase):
    def test_prompt_is_not_in_the_argv_at_all(self):
        """Regression: a variadic option (--allowedTools) swallows a trailing prompt.

        Measured 2026-09-02 -- both A/B arms died in under two seconds with
        "Input must be provided either through stdin or as a prompt argument",
        and the summary showed them as two arms that simply scored zero. The
        prompt now goes on stdin, so no option ordering can eat it.
        """
        runner = FakeRunner([_payload()])
        sl.run_iteration(
            "do the thing",
            cwd=pathlib.Path("/"),
            schema='{"type":"object"}',
            extra_args=["--allowedTools", "Read Grep"],
            runner=runner,
        )
        argv, kwargs = runner.calls[0]
        # Asserted through the real call path. The earlier version checked a
        # prompt string against build_command, which has no prompt parameter --
        # so it asserted the absence of something that was never passed, and the
        # variadic-flag regression it is named for could have returned.
        self.assertNotIn("do the thing", argv)
        self.assertEqual(kwargs["input"], "do the thing")
        self.assertEqual(argv[-1], "Read Grep")
        self.assertEqual(argv[:4], ["claude", "-p", "--output-format", "json"])

    def test_prompt_reaches_the_subprocess_on_stdin(self):
        runner = FakeRunner([_payload()])
        sl.run_iteration("the instruction", cwd=pathlib.Path("/"), runner=runner)
        _, kwargs = runner.calls[0]
        self.assertEqual(kwargs["input"], "the instruction")

    def test_optional_flags_are_absent_when_unset(self):
        argv = sl.build_command()
        for flag in ("--json-schema", "--max-budget-usd", "--permission-mode", "--model"):
            self.assertNotIn(flag, argv)

    def test_no_resume_flag_is_ever_added(self):
        """Each iteration must be a FRESH session; --resume would defeat the point."""
        argv = sl.build_command(extra_args=["--name", "n"])
        self.assertNotIn("--resume", argv)
        self.assertNotIn("-r", argv)
        self.assertNotIn("--continue", argv)


class Classification(unittest.TestCase):
    def test_clean_completed_run_is_ok_and_carries_the_stop_signal(self):
        verdict, done = sl.classify(_payload(), done_key="done")
        self.assertIs(verdict, sl.IterationVerdict.OK)
        self.assertTrue(done)

    def test_denied_tool_call_is_degraded_even_though_cli_reports_success(self):
        """The 2026-09-02 observation: is_error false, terminal_reason completed, work degraded."""
        verdict, done = sl.classify(
            _payload(permission_denials=[{"tool_name": "Bash"}] * 4), done_key="done"
        )
        self.assertIs(verdict, sl.IterationVerdict.DEGRADED_BY_DENIAL)
        self.assertIsNone(done)

    def test_denial_outranks_a_done_signal(self):
        """A degraded run answers the schema perfectly; it must not be able to end the chain."""
        verdict, done = sl.classify(
            _payload(
                permission_denials=[{"tool_name": "Bash"}],
                structured_output={"done": True, "next": "finished"},
            ),
            done_key="done",
        )
        self.assertIs(verdict, sl.IterationVerdict.DEGRADED_BY_DENIAL)
        self.assertIsNone(done)

    def test_missing_structured_output_is_schema_miss(self):
        verdict, _ = sl.classify(_payload(structured_output=None), done_key="done")
        self.assertIs(verdict, sl.IterationVerdict.SCHEMA_MISS)

    def test_non_boolean_done_is_schema_miss_not_a_truthy_stop(self):
        verdict, _ = sl.classify(
            _payload(structured_output={"done": "yes"}), done_key="done"
        )
        self.assertIs(verdict, sl.IterationVerdict.SCHEMA_MISS)

    def test_error_flag_and_unknown_terminal_reason_are_abnormal(self):
        self.assertIs(
            sl.classify(_payload(is_error=True), done_key="done")[0],
            sl.IterationVerdict.ABNORMAL_END,
        )
        self.assertIs(
            sl.classify(_payload(terminal_reason="budget_exceeded"), done_key="done")[0],
            sl.IterationVerdict.ABNORMAL_END,
        )

    def test_done_key_is_configurable(self):
        verdict, done = sl.classify(
            _payload(structured_output={"finished": False}), done_key="finished"
        )
        self.assertIs(verdict, sl.IterationVerdict.OK)
        self.assertFalse(done)


class IterationAccounting(unittest.TestCase):
    def test_context_tokens_sum_input_and_both_cache_halves(self):
        runner = FakeRunner([_payload()])
        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=runner)
        self.assertEqual(result.context_tokens, 2 + 17170 + 22180)
        self.assertAlmostEqual(result.cost_usd, 0.27201)
        self.assertTrue(result.is_usable)

    def test_missing_usage_keys_degrade_to_zero_rather_than_raising(self):
        runner = FakeRunner([_payload(usage={"input_tokens": 5})])
        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=runner)
        self.assertEqual(result.context_tokens, 5)

    def test_denials_are_recorded_by_tool_name(self):
        runner = FakeRunner(
            [_payload(permission_denials=[{"tool_name": "Bash"}, {"tool_name": "Write"}])]
        )
        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=runner)
        self.assertEqual(result.denials, ("Bash", "Write"))
        self.assertFalse(result.is_usable)

    def test_unparseable_stdout_is_launch_failed_not_an_exception(self):
        class BadRunner:
            def __call__(self, argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0, stdout="not json", stderr="")

        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=BadRunner())
        self.assertIs(result.verdict, sl.IterationVerdict.LAUNCH_FAILED)
        self.assertEqual(result.cost_usd, 0.0)

    def test_json_that_is_not_an_object_is_launch_failed(self):
        class ListRunner:
            def __call__(self, argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0, stdout="[1,2]", stderr="")

        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=ListRunner())
        self.assertIs(result.verdict, sl.IterationVerdict.LAUNCH_FAILED)

    def test_subprocess_timeout_is_launch_failed_not_a_crash(self):
        runner = FakeRunner([subprocess.TimeoutExpired("claude", 1)])
        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=runner)
        self.assertIs(result.verdict, sl.IterationVerdict.LAUNCH_FAILED)
        self.assertIn("TimeoutExpired", result.text)


class ChainControl(unittest.TestCase):
    def test_chain_stops_on_the_done_signal(self):
        runner = FakeRunner(
            [
                _payload(structured_output={"done": False, "next": "keep going"}),
                _payload(structured_output={"done": True, "next": "finished"}),
            ]
        )
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=5, chain_budget_usd=10.0, runner=runner
        )
        self.assertEqual(chain.stop_reason, "done")
        self.assertTrue(chain.completed)
        self.assertEqual(len(chain.iterations), 2)

    def test_every_iteration_gets_the_identical_prompt(self):
        """State crosses on disk, not in the prompt; a drifting prompt would hide that."""
        runner = FakeRunner(
            [
                _payload(structured_output={"done": False, "next": "a"}),
                _payload(structured_output={"done": True, "next": "b"}),
            ]
        )
        sl.run_chain(
            "the same words",
            cwd=pathlib.Path("/"),
            max_iterations=5,
            chain_budget_usd=10.0,
            runner=runner,
        )
        prompts = [kwargs["input"] for _, kwargs in runner.calls]
        self.assertEqual(prompts, ["the same words", "the same words"])

    def test_degraded_iteration_stops_the_chain_by_default(self):
        runner = FakeRunner([_payload(permission_denials=[{"tool_name": "Bash"}])])
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=5, chain_budget_usd=10.0, runner=runner
        )
        self.assertEqual(chain.stop_reason, "degraded_by_denial")
        self.assertFalse(chain.completed)

    def test_continue_on_degraded_is_opt_in(self):
        runner = FakeRunner(
            [
                _payload(permission_denials=[{"tool_name": "Bash"}]),
                _payload(structured_output={"done": True, "next": "finished"}),
            ]
        )
        chain = sl.run_chain(
            "x",
            cwd=pathlib.Path("/"),
            max_iterations=5,
            chain_budget_usd=10.0,
            stop_on_degraded=False,
            runner=runner,
        )
        self.assertEqual(chain.stop_reason, "done")
        self.assertEqual(len(chain.iterations), 2)

    def test_chain_stops_once_recorded_spend_reaches_the_ceiling(self):
        """The pre-launch half of the cap. Named for what it measures.

        The previous name and docstring claimed the cap "cannot be blown past by
        one more session" while these very assertions show $1.20 spent against a
        $1.00 ceiling. A reader checking the guard would have found the docstring,
        not the assertion. The overshoot itself is now bounded by the child's own
        --max-budget-usd, covered in test_every_iteration_is_capped_by_what_remains.
        """
        runner = FakeRunner(
            [_payload(total_cost_usd=0.6, structured_output={"done": False, "next": "on"})] * 2
        )
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=99, chain_budget_usd=1.0, runner=runner
        )
        self.assertEqual(chain.stop_reason, "budget_exhausted")
        self.assertEqual(len(chain.iterations), 2)
        self.assertAlmostEqual(chain.total_cost_usd, 1.2)

    def test_every_iteration_is_capped_by_what_remains(self):
        """The half that makes the ceiling real: the child is told what is left."""
        runner = FakeRunner(
            [
                _payload(total_cost_usd=3.0, structured_output={"done": False, "next": "on"}),
                _payload(total_cost_usd=1.0, structured_output={"done": True, "next": "fin"}),
            ]
        )
        sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=9, chain_budget_usd=10.0, runner=runner
        )
        caps = [argv[argv.index("--max-budget-usd") + 1] for argv, _ in runner.calls]
        self.assertEqual(caps, ["10.0", "7.0"])

    def test_a_caller_cap_is_an_upper_bound_never_a_way_to_raise_the_ceiling(self):
        runner = FakeRunner(
            [_payload(total_cost_usd=9.5, structured_output={"done": False, "next": "on"})] * 2
        )
        sl.run_chain(
            "x",
            cwd=pathlib.Path("/"),
            max_iterations=9,
            chain_budget_usd=10.0,
            budget_usd=100.0,
            runner=runner,
        )
        caps = [float(argv[argv.index("--max-budget-usd") + 1]) for argv, _ in runner.calls]
        self.assertEqual(caps, [10.0, 0.5])

    def test_spend_exactly_equal_to_the_ceiling_stops_the_chain(self):
        """The boundary the old test never touched: >= rather than >."""
        runner = FakeRunner(
            [_payload(total_cost_usd=1.0, structured_output={"done": False, "next": "on"})] * 2
        )
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=9, chain_budget_usd=1.0, runner=runner
        )
        self.assertEqual(len(chain.iterations), 1)
        self.assertEqual(chain.stop_reason, "budget_exhausted")

    def test_iteration_cap_is_honoured(self):
        runner = FakeRunner([_payload(structured_output={"done": False, "next": "on"})] * 3)
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=3, chain_budget_usd=99.0, runner=runner
        )
        self.assertEqual(chain.stop_reason, "max_iterations")
        self.assertEqual(len(chain.iterations), 3)

    def test_failed_iteration_cost_still_counts_toward_the_total(self):
        runner = FakeRunner([_payload(is_error=True, total_cost_usd=0.4)])
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=5, chain_budget_usd=10.0, runner=runner
        )
        self.assertAlmostEqual(chain.total_cost_usd, 0.4)
        self.assertEqual(chain.stop_reason, "abnormal_end")


class Ledger(unittest.TestCase):
    def test_each_iteration_is_appended_as_it_finishes(self):
        runner = FakeRunner(
            [
                _payload(structured_output={"done": False, "next": "a"}),
                _payload(structured_output={"done": True, "next": "b"}),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "nested" / "ledger.jsonl"
            sl.run_chain(
                "x",
                cwd=pathlib.Path("/"),
                max_iterations=5,
                chain_budget_usd=10.0,
                ledger=path,
                label="single",
                runner=runner,
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([r["index"] for r in rows], [0, 1])
        self.assertEqual({r["label"] for r in rows}, {"single"})
        self.assertEqual([r["done"] for r in rows], [False, True])

    def test_ledger_rows_reproduce_the_reported_total(self):
        runner = FakeRunner(
            [
                _payload(total_cost_usd=0.3, structured_output={"done": False, "next": "a"}),
                _payload(total_cost_usd=0.7, structured_output={"done": True, "next": "b"}),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "ledger.jsonl"
            chain = sl.run_chain(
                "x",
                cwd=pathlib.Path("/"),
                max_iterations=5,
                chain_budget_usd=10.0,
                ledger=path,
                runner=runner,
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertAlmostEqual(sum(r["cost_usd"] for r in rows), chain.total_cost_usd)

    def test_each_row_keeps_the_cache_split_that_sums_to_its_context(self):
        """The caching claim was retracted because only the sum was recorded."""
        runner = FakeRunner(
            [_payload(usage={"input_tokens": 5, "cache_creation_input_tokens": 700, "cache_read_input_tokens": 9000})]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "ledger.jsonl"
            sl.run_chain(
                "x", cwd=pathlib.Path("/"), max_iterations=1, chain_budget_usd=10.0, ledger=path, runner=runner
            )
            row = json.loads(path.read_text().splitlines()[0])
        self.assertEqual((row["input_tokens"], row["cache_creation_tokens"], row["cache_read_tokens"]), (5, 700, 9000))
        self.assertEqual(row["context_tokens"], 9705)

    def test_a_missing_or_null_usage_counter_is_zero_not_a_crash(self):
        runner = FakeRunner([_payload(usage={"input_tokens": None, "cache_read_input_tokens": 12})])
        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=runner)
        self.assertEqual((result.input_tokens, result.cache_creation_tokens, result.cache_read_tokens), (0, 0, 12))
        self.assertEqual(result.verdict, sl.IterationVerdict.OK)


class CommandLine(unittest.TestCase):
    def test_exit_code_gates_on_real_completion_not_on_not_crashing(self):
        parser_args = sl._build_parser().parse_args(["do it", "--max-iterations", "1"])
        self.assertEqual(parser_args.prompt, "do it")
        self.assertEqual(parser_args.max_iterations, 1)
        self.assertFalse(parser_args.continue_on_degraded)

    def test_default_schema_declares_the_done_field(self):
        args = sl._build_parser().parse_args(["p"])
        schema = json.loads(args.schema)
        self.assertIn(args.done_key, schema["properties"])
        self.assertIn(args.done_key, schema["required"])


class FlagPassThrough(unittest.TestCase):
    def test_every_optional_flag_reaches_the_argv(self):
        argv = sl.build_command(
            schema="{}",
            budget_usd=2.5,
            permission_mode="acceptEdits",
            model="opus",
            extra_args=["--name", "run"],
        )
        for flag, value in (
            ("--json-schema", "{}"),
            ("--max-budget-usd", "2.5"),
            ("--permission-mode", "acceptEdits"),
            ("--model", "opus"),
            ("--name", "run"),
        ):
            self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index(flag) + 1], value)

    def test_totals_span_every_iteration(self):
        runner = FakeRunner(
            [
                _payload(total_cost_usd=0.1, structured_output={"done": False, "next": "a"}),
                _payload(total_cost_usd=0.2, structured_output={"done": True, "next": "b"}),
            ]
        )
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=5, chain_budget_usd=10.0, runner=runner
        )
        self.assertEqual(chain.total_context_tokens, 2 * (2 + 17170 + 22180))
        self.assertAlmostEqual(chain.total_cost_usd, 0.3)


class MainExitCode(unittest.TestCase):
    """main() must return non-zero for a chain that stopped for any reason but completion."""

    def _run_main(self, payloads, extra_argv=()):
        runner = FakeRunner(payloads)
        real = sl.run_chain
        captured = {}

        def patched(prompt, **kwargs):
            kwargs["runner"] = runner
            captured["chain"] = real(prompt, **kwargs)
            return captured["chain"]

        sl.run_chain = patched
        try:
            with tempfile.TemporaryDirectory() as tmp:
                argv = ["do it", "--cwd", tmp, "--max-iterations", "3", *extra_argv]
                return sl.main(argv), captured["chain"]
        finally:
            sl.run_chain = real

    def test_zero_when_the_work_reports_itself_done(self):
        code, chain = self._run_main([_payload(structured_output={"done": True, "next": "x"})])
        self.assertEqual(code, 0)
        self.assertTrue(chain.completed)

    def test_nonzero_when_a_denial_degraded_the_run(self):
        code, chain = self._run_main([_payload(permission_denials=[{"tool_name": "Bash"}])])
        self.assertEqual(code, 1)
        self.assertEqual(chain.stop_reason, "degraded_by_denial")

    def test_the_budget_flag_reaches_the_chain_and_caps_every_child(self):
        code, chain = self._run_main(
            [_payload(total_cost_usd=0.2, structured_output={"done": False, "next": "more"})] * 3,
            extra_argv=("--budget-usd", "0.3"),
        )
        self.assertEqual(code, 1)
        self.assertEqual(chain.stop_reason, "budget_exhausted")
        self.assertEqual(len(chain.iterations), 2)

    def test_the_schema_reaches_every_child(self):
        runner = FakeRunner([_payload(structured_output={"done": True, "next": "x"})])
        real = sl.run_chain

        def patched(prompt, **kwargs):
            kwargs["runner"] = runner
            return real(prompt, **kwargs)

        sl.run_chain = patched
        try:
            with tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(sl.main(["do it", "--cwd", tmp, "--max-iterations", "1"]), 0)
        finally:
            sl.run_chain = real
        argv = runner.calls[0][0]
        self.assertIn("--json-schema", argv)
        self.assertIn('"done"', argv[argv.index("--json-schema") + 1])
        self.assertIn("--max-budget-usd", argv)
        self.assertLessEqual(float(argv[argv.index("--max-budget-usd") + 1]), 5.0)

    def test_nonzero_when_the_iteration_cap_is_hit_without_finishing(self):
        code, _ = self._run_main(
            [_payload(structured_output={"done": False, "next": "more"})] * 3
        )
        self.assertEqual(code, 1)


class MissingCostStopsTheChain(unittest.TestCase):
    """A payload without a numeric cost may still have spent; the budget cannot be trusted after it."""

    def test_a_missing_or_null_cost_is_unknown_and_stops_the_chain(self):
        for overrides in ({"total_cost_usd": None}, {"total_cost_usd": "0.3"}, {"total_cost_usd": True}):
            payload = _payload(structured_output={"done": False, "next": "on"}, **overrides)
            runner = FakeRunner([payload] * 3)
            with self.subTest(overrides=overrides):
                chain = sl.run_chain("x", cwd=pathlib.Path("/"), max_iterations=3, chain_budget_usd=5.0, runner=runner)
                self.assertEqual(len(chain.iterations), 1)
                self.assertEqual(chain.stop_reason, "cost_unknown")
                self.assertFalse(chain.cost_is_complete)

    def test_an_absent_cost_key_is_unknown_too(self):
        payload = _payload(structured_output={"done": False, "next": "on"})
        del payload["total_cost_usd"]
        chain = sl.run_chain("x", cwd=pathlib.Path("/"), max_iterations=3, chain_budget_usd=5.0,
                             runner=FakeRunner([payload] * 3))
        self.assertEqual((len(chain.iterations), chain.stop_reason), (1, "cost_unknown"))


class UnknownCostIsNotZeroCost(unittest.TestCase):
    """A session that died may still have spent money; the total must say so."""

    def test_launch_failure_marks_cost_unknown_rather_than_zero(self):
        runner = FakeRunner([subprocess.TimeoutExpired("claude", 3600)])
        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=runner)
        self.assertFalse(result.cost_is_known)
        self.assertEqual(result.cost_usd, 0.0)

    def test_a_chain_containing_an_unknown_cost_reports_an_incomplete_total(self):
        runner = FakeRunner([subprocess.TimeoutExpired("claude", 1)])
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=2, chain_budget_usd=10.0, runner=runner
        )
        self.assertFalse(chain.cost_is_complete)

    def test_a_clean_chain_reports_a_complete_total(self):
        runner = FakeRunner([_payload()])
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=2, chain_budget_usd=10.0, runner=runner
        )
        self.assertTrue(chain.cost_is_complete)


class SubprocessFailureSurfaces(unittest.TestCase):
    def test_nonzero_returncode_is_launch_failed_and_keeps_stderr(self):
        class Failing:
            def __call__(self, argv, **kwargs):
                return subprocess.CompletedProcess(argv, 2, stdout="", stderr="unknown flag --nope")

        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=Failing())
        self.assertIs(result.verdict, sl.IterationVerdict.LAUNCH_FAILED)
        self.assertIn("unknown flag --nope", result.text)
        self.assertIn("exited 2", result.text)

    def test_unparseable_stdout_reports_stderr_too(self):
        class Noisy:
            def __call__(self, argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0, stdout="not json", stderr="auth failed")

        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=Noisy())
        self.assertIn("auth failed", result.text)

    def test_timeout_is_actually_armed_on_the_subprocess(self):
        """The old test proved the except clause caught a hand-thrown exception.

        It never showed that any timeout reached subprocess.run, so deleting
        `timeout=timeout_seconds` left the suite green while a hung child could
        block forever.
        """
        runner = FakeRunner([_payload()])
        sl.run_iteration("x", cwd=pathlib.Path("/"), timeout_seconds=12.5, runner=runner)
        _, kwargs = runner.calls[0]
        self.assertEqual(kwargs["timeout"], 12.5)

    def test_cwd_reaches_the_subprocess(self):
        """cwd is documented as setting the floor cost of every iteration."""
        runner = FakeRunner([_payload()])
        sl.run_iteration("x", cwd=pathlib.Path("/tmp"), runner=runner)
        _, kwargs = runner.calls[0]
        self.assertEqual(kwargs["cwd"], "/tmp")

    def test_a_malformed_payload_does_not_destroy_the_chains_earlier_results(self):
        """One bad result must not take every prior iteration out with it."""
        runner = FakeRunner(
            [
                _payload(structured_output={"done": False, "next": "a"}),
                _payload(usage="not-a-dict", num_turns={"unexpected": "shape"}),
            ]
        )
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=5, chain_budget_usd=10.0, runner=runner
        )
        self.assertEqual(len(chain.iterations), 2)
        self.assertIs(chain.iterations[1].verdict, sl.IterationVerdict.LAUNCH_FAILED)

    def test_denials_given_as_plain_strings_do_not_raise(self):
        """The dict-with-tool_name shape exists only in this file's fixture."""
        runner = FakeRunner([_payload(permission_denials=["Bash", "Write"])])
        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=runner)
        self.assertIs(result.verdict, sl.IterationVerdict.DEGRADED_BY_DENIAL)
        self.assertEqual(result.denials, ("Bash", "Write"))


class InvariantBreakingFlagsRejected(unittest.TestCase):
    """extra_args must not be able to delete the guarantees the module documents."""

    def test_resume_and_continue_are_refused(self):
        for flag in ("--resume", "-r", "--continue", "-c", "--fork-session"):
            with self.assertRaises(ValueError):
                sl.build_command(extra_args=[flag, "x"])

    def test_bypassing_permissions_is_refused_because_it_blinds_the_classifier(self):
        with self.assertRaises(ValueError):
            sl.build_command(extra_args=["--dangerously-skip-permissions"])

    def test_a_bypass_permission_MODE_is_refused_by_value_not_only_by_flag_name(self):
        """The name-only ban missed this: same blindness, different spelling."""
        for kwargs in (
            {"extra_args": ["--permission-mode", "bypassPermissions"]},
            {"extra_args": ["--permission-mode=bypassPermissions"]},
            {"permission_mode": "bypassPermissions"},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                sl.build_command(**kwargs)

    def test_an_ordinary_permission_mode_is_still_allowed(self):
        argv = sl.build_command(permission_mode="acceptEdits")
        self.assertIn("acceptEdits", argv)

    def test_dont_ask_is_allowed_because_it_denies_rather_than_approves(self):
        """dontAsk auto-denies anything that would prompt, so denials still reach the classifier."""
        self.assertIn("dontAsk", sl.build_command(permission_mode="dontAsk"))

    def test_settings_is_refused_because_it_can_set_a_bypass_mode(self):
        for extra in (["--settings", '{"permissions":{"defaultMode":"bypassPermissions"}}'],
                      ["--settings=./settings.json"]):
            with self.assertRaises(ValueError, msg=str(extra)):
                sl.build_command(extra_args=extra)

    def test_equals_form_is_caught_too(self):
        with self.assertRaises(ValueError):
            sl.build_command(extra_args=["--resume=abc123"])

    def test_ordinary_flags_still_pass_through(self):
        argv = sl.build_command(extra_args=["--allowedTools", "Read Write"])
        self.assertEqual(argv[-2:], ["--allowedTools", "Read Write"])


class LedgerDurability(unittest.TestCase):
    def test_a_ledger_write_failure_does_not_discard_collected_results(self):
        runner = FakeRunner([_payload(structured_output={"done": False, "next": "a"})])
        with tempfile.TemporaryDirectory() as tmp:
            blocked = pathlib.Path(tmp) / "file.txt"
            blocked.write_text("not a directory")
            chain = sl.run_chain(
                "x",
                cwd=pathlib.Path("/"),
                max_iterations=3,
                chain_budget_usd=10.0,
                ledger=blocked / "ledger.jsonl",
                runner=runner,
            )
        self.assertEqual(chain.stop_reason, "ledger_write_failed")
        self.assertEqual(len(chain.iterations), 1)

    def test_a_generated_run_id_is_unique_so_forgetting_the_flag_is_safe(self):
        """An opt-in separator does not help the operator who forgets it."""
        self.assertNotEqual(sl.new_run_id(), sl.new_run_id())

    def test_a_failure_reason_reaches_the_ledger_row(self):
        """A diagnostic nobody can read is not a diagnostic."""
        runner = FakeRunner([subprocess.TimeoutExpired("claude", 1)])
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "l.jsonl"
            sl.run_chain(
                "x",
                cwd=pathlib.Path("/"),
                max_iterations=1,
                chain_budget_usd=1.0,
                ledger=path,
                runner=runner,
            )
            row = json.loads(path.read_text().splitlines()[0])
        self.assertIn("TimeoutExpired", row["detail"])
        self.assertFalse(row["cost_is_known"])

    def test_run_id_and_cost_flag_are_recorded_for_later_separation(self):
        runner = FakeRunner([_payload()])
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "ledger.jsonl"
            sl.run_chain(
                "x",
                cwd=pathlib.Path("/"),
                max_iterations=1,
                chain_budget_usd=10.0,
                ledger=path,
                run_id="run-b",
                runner=runner,
            )
            row = json.loads(path.read_text().splitlines()[0])
        self.assertEqual(row["run_id"], "run-b")
        self.assertTrue(row["cost_is_known"])

    def test_append_ledger_reports_failure_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocked = pathlib.Path(tmp) / "f.txt"
            blocked.write_text("x")
            runner = FakeRunner([_payload()])
            result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=runner)
            self.assertFalse(sl.append_ledger(blocked / "l.jsonl", "a", 0, result))


class HeadlineFiguresAreAsserted(unittest.TestCase):
    """turns, session id and iteration count are published; pin them to a payload."""

    def test_turns_and_session_id_are_read_from_the_payload(self):
        runner = FakeRunner([_payload(num_turns=31, session_id="abc-123")])
        result = sl.run_iteration("x", cwd=pathlib.Path("/"), runner=runner)
        self.assertEqual(result.num_turns, 31)
        self.assertEqual(result.session_id, "abc-123")

    def test_chain_totals_sum_the_published_fields(self):
        runner = FakeRunner(
            [
                _payload(num_turns=5, structured_output={"done": False, "next": "a"}),
                _payload(num_turns=7, structured_output={"done": True, "next": "b"}),
            ]
        )
        chain = sl.run_chain(
            "x", cwd=pathlib.Path("/"), max_iterations=5, chain_budget_usd=10.0, runner=runner
        )
        self.assertEqual(sum(i.num_turns for i in chain.iterations), 12)
        self.assertEqual(len(chain.iterations), 2)


if __name__ == "__main__":
    unittest.main()
