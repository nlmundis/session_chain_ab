#!/usr/bin/env python3
"""Run a long task as a chain of short Claude Code sessions instead of one long one.

WHAT THIS IS, AND WHAT IT IS NOT
    The loop itself is not new. Geoffrey Huntley's "Ralph" technique
    (``while :; do cat PROMPT.md | claude-code ; done``, ghuntley.com/ralph),
    continuous-claude, and Anthropic's own ``anthropics/cwc-long-running-agents``
    all run a fresh headless session per pass with the state carried in files.
    What this module adds is the part an EXPERIMENT needs: a per-iteration cost
    ledger, a typed stop signal, a verdict that refuses to build on silently
    degraded work, and hard budget caps, so that ``ab_compare.py`` can measure
    whether chaining is actually cheaper than one long session on the same task.

WHY THE QUESTION IS WORTH MEASURING
    Every tool call in a session re-reads the whole accumulated context, so a
    long session's per-turn cost grows with its length. Clearing context resets
    that growth, but every fresh session re-pays its startup context (system
    prompt, tool schemas, CLAUDE.md, memory) and re-reads whatever state it needs.
    Which effect wins depends on task length, caching, and pricing, and is not
    settled by argument in either direction. Anthropic's cost guidance, advising
    ``/clear`` when switching to unrelated work, says stale context "wastes tokens
    on every subsequent message"; the same page notes that history is re-read at
    the cached token rate. Whether clearing also pays WITHIN one long task is the
    open question this harness measures.

    A session cannot clear its own context, but a process boundary can: every
    ``claude -p`` invocation is a fresh session unless ``--resume`` is passed,
    so the process exit is the clear.

THE SHAPE
    A driver loop, outside the model. Each iteration launches a fresh session
    with the SAME prompt; the only state that crosses the process boundary is a
    file on disk that the prompt tells each session to read and update (a
    progress note, a task list, a report). A child starts empty and reads what it needs, rather than
    inheriting a parent's accumulated mess.

    The continue-or-stop decision is read from ``--json-schema`` structured
    output as a boolean. Be precise about what that buys: the TRANSPORT is a
    rule -- a typed field the driver parses, not prose it interprets -- but the
    VALUE is still the model's own claim about work it just did. A chain cannot
    tell a finished job from one the model believes is finished. Whatever
    actually grades the work has to live outside this module; ``ab_compare.py``
    grades against ``ast`` for exactly that reason.

WHY THE VERDICT IS NOT ``is_error``
    Measured 2026-09-02, and this is the reason ``IterationVerdict`` exists at
    all. A probe run of ``claude -p`` on a slash command that needed Bash had four Bash calls
    denied by the permission layer, silently rewrote its approach to a grep
    sweep, produced a visibly degraded answer -- and reported
    ``is_error: false`` with ``terminal_reason: "completed"``. Headless mode
    cannot ask for an approval, so a missing permission does not stop a run; it
    quietly changes what the run did.

    A loop that trusted ``is_error`` would therefore chain iterations on top of
    degraded work and report success at the end. So a non-empty
    ``permission_denials`` is classified as a FAILED iteration here, and the
    default is to stop the chain rather than to build on it.

MEASURED FLOOR PER ITERATION (2026-09-02, Opus 5, the author's machine)
    A one-word headless reply that used no tools cost $0.272 and read 48,258
    context tokens from a home directory with a large CLAUDE.md and memory index
    ($0.183 / 39,352 tokens from a bare directory, which loads less). That is the
    price of the startup context -- system prompt, tool schemas, CLAUDE.md,
    memory -- and it is paid again by every iteration. The probe output was not
    kept, so treat these two figures as context, not as reproducible evidence;
    your own floor depends on what your sessions load.

    This is what decides whether the sawtooth is worth running: chaining wins
    only when the context a single long session would drag costs more per turn
    than reloading the startup context from scratch. ``ab_compare`` exists to
    measure that break-even rather than assume it.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import pathlib
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from typing import Any

# Keys summed from a result's ``usage`` block to get one iteration's total
# context read: input plus both cache halves, since a cache read is a context
# read that was merely cheaper. The halves are ALSO recorded separately (see
# IterationResult), because a single sum cannot say whether caching did the work.
_CONTEXT_TOKEN_KEYS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

def new_run_id() -> str:
    """Mint an identifier separating this run from earlier ones in a shared ledger.

    Generated rather than left blank by default. An opt-in separator does not
    solve the problem it exists for: the operator who would forget ``--run-id`` is
    precisely the one whose two runs become indistinguishable rows.
    """
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


# Flags a caller must not smuggle in through ``extra_args``. Each one silently
# falsifies something the module promises rather than causing a visible error:
# a resumed or continued session is not the fresh context the whole design rests
# on, and a permission mode that auto-approves everything means no tool call is
# ever denied, so DEGRADED_BY_DENIAL can never fire and the classifier reports
# clean runs forever. ``--settings`` is refused too: it can set
# ``permissions.defaultMode`` to bypassPermissions from inline JSON or a file,
# and this module does not parse either to find out. Rejected loudly, because
# the failure is otherwise invisible.
_INVARIANT_BREAKING_FLAGS = frozenset(
    {"--resume", "-r", "--continue", "-c", "--fork-session", "--dangerously-skip-permissions",
     "--settings"}
)

# Permission modes that auto-approve every tool call. Banning only the flag NAME
# was not enough: `--permission-mode bypassPermissions` reaches exactly the state
# `--dangerously-skip-permissions` does -- no call is ever denied, so
# permission_denials stays empty and DEGRADED_BY_DENIAL can never fire -- and it
# spells it differently, so a name-only check waved it through.
#
# `dontAsk` is NOT here, and an earlier version wrongly refused it as
# auto-approving. It is the opposite: it auto-DENIES every call that would
# otherwise prompt (code.claude.com/docs/en/permission-modes), so those denials
# still reach permission_denials and the classifier still sees them.
_BLIND_PERMISSION_MODES = frozenset({"bypassPermissions"})

# ``terminal_reason`` value the CLI reports for a run that ended on its own
# terms. Anything else is treated as an abnormal end, because the full
# vocabulary is not documented and guessing at it would fail open.
_TERMINAL_REASON_OK = "completed"


class IterationVerdict(enum.Enum):
    """How one headless session ended, judged by rules rather than by its own claim.

    Deliberately not a boolean, and deliberately not read off ``is_error``: a run
    that silently lost tool permissions reports itself successful (see the module
    docstring).

    The members are kept distinct because they mean different things to a HUMAN
    reading a ledger, not because the driver branches five ways -- ``run_chain``
    continues only on OK and stops on everything else, with one flag-gated
    exception for DEGRADED. A ledger that recorded a single "failed" would lose
    the difference between a run that produced unusable work and one that
    produced none, which is the difference that decides whether to re-run.
    """

    OK = "ok"
    """Completed, no denied tool call, and ``done_key`` present in the structured
    output as an actual bool -- a truthy string does not qualify."""

    DEGRADED_BY_DENIAL = "degraded_by_denial"
    """Completed, but at least one tool call was denied, so the work routed around a gap."""

    SCHEMA_MISS = "schema_miss"
    """Completed, but returned no usable structured output, so continue-or-stop is unreadable."""

    ABNORMAL_END = "abnormal_end"
    """The CLI reported an error or a ``terminal_reason`` other than ``completed``."""

    LAUNCH_FAILED = "launch_failed"
    """The subprocess did not produce a parseable result at all; no work happened."""


@dataclasses.dataclass(frozen=True)
class IterationResult:
    """One headless session's outcome, cost, and the stop signal it returned.

    Holds what the driver needs to decide whether to run again and what the A/B
    needs to compare, so callers never re-parse the CLI's JSON themselves.
    """

    verdict: IterationVerdict
    session_id: str | None
    cost_usd: float
    context_tokens: int
    num_turns: int
    wall_seconds: float
    cost_is_known: bool
    """Whether ``cost_usd`` was read from the CLI or is a placeholder.

    False on LAUNCH_FAILED, where the child may have run for an hour and spent
    real money before the timeout fired. The number is zero there because nothing
    reported one, NOT because nothing was spent, and a total that silently folds
    in such a zero understates the run. Callers summing cost must say so.
    """
    denials: tuple[str, ...]
    """``tool_name`` of each denied call, in order. Empty when nothing was denied."""
    done: bool | None
    """The chain's stop signal, or None when it could not be read."""
    structured_output: dict[str, Any] | None
    text: str
    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    """The three halves of ``context_tokens``, kept apart in the ledger.

    The first published analysis of this harness attributed the single session's
    advantage to prompt caching and had to retract it: only the sum reached the
    ledger, so no committed file could tell a cache read from a cache write. A
    chained arm whose sessions start from a different git status (Claude Code
    shares a cached prefix across sequential sessions only when that snapshot
    matches) would show up here as cache creation where the single arm shows
    cache reads.
    """

    @property
    def is_usable(self) -> bool:
        """Whether later iterations may safely build on this one's work.

        Only OK qualifies. A DEGRADED run's output is real text that reads as an
        answer, which is exactly why it needs an explicit gate.
        """
        return self.verdict is IterationVerdict.OK


@dataclasses.dataclass(frozen=True)
class ChainResult:
    """A whole chain of sessions: every iteration, plus why the chain stopped.

    Totals are summed here rather than recomputed by callers so that a reported
    cost always matches the ledger rows it came from.
    """

    iterations: tuple[IterationResult, ...]
    stop_reason: str

    @property
    def total_cost_usd(self) -> float:
        """Dollars the CLI reported, summed across every iteration.

        A LOWER BOUND, not the true spend: an iteration whose cost is unknown
        (see ``IterationResult.cost_is_known``) contributes zero. Check
        ``cost_is_complete`` before quoting this as what a run cost.
        """
        return sum(i.cost_usd for i in self.iterations)

    @property
    def cost_is_complete(self) -> bool:
        """Whether every iteration reported its own cost, making the total exact."""
        return all(i.cost_is_known for i in self.iterations)

    @property
    def total_context_tokens(self) -> int:
        """Context tokens the CLI reported, summed across every iteration.

        A LOWER BOUND for the same reason ``total_cost_usd`` is: an iteration
        that never reported a usage block contributes zero, which is not the same
        as having read nothing. ``cost_is_complete`` covers this field too.
        """
        return sum(i.context_tokens for i in self.iterations)

    @property
    def completed(self) -> bool:
        """Whether the chain ended because the work reported itself done."""
        return self.stop_reason == "done"


def build_command(
    *,
    schema: str | None = None,
    budget_usd: float | None = None,
    permission_mode: str | None = None,
    model: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Assemble the ``claude -p`` argv for one iteration.

    The prompt is NOT in the argv: it goes on stdin, and this is a correctness
    fix rather than a style choice. Several CLI options are variadic
    (``--allowedTools <tools...>``, ``--add-dir``, ``--betas``), and a variadic
    option swallows every argument after it -- including a trailing prompt. That
    failure is silent in the worst way: measured 2026-09-02, both arms of the
    first A/B run died in under two seconds with "Input must be provided either
    through stdin or as a prompt argument", having spent nothing and having
    looked, in the summary, exactly like two arms that scored zero. Passing the
    prompt on stdin makes the argv order irrelevant.

    Split out from ``run_iteration`` so the argv can be asserted in tests without
    launching a session, which costs real money on every run.

    Args:
        schema: JSON Schema string for structured output. When given, the
            driver reads the stop signal from it.
        budget_usd: Hard per-iteration spend cap passed to ``--max-budget-usd``.
        permission_mode: Value for ``--permission-mode``. Worth setting
            explicitly: headless mode cannot prompt, so an unallowed tool is
            denied rather than asked about.
        model: Model alias or full name.
        extra_args: Appended verbatim, for flags this wrapper does not model.

    Returns:
        The argv list. The prompt is not part of it; pass it on stdin.
    """
    argv = ["claude", "-p", "--output-format", "json"]
    if schema is not None:
        argv += ["--json-schema", schema]
    if budget_usd is not None:
        argv += ["--max-budget-usd", str(budget_usd)]
    if permission_mode is not None:
        argv += ["--permission-mode", permission_mode]
    if model is not None:
        argv += ["--model", model]
    if permission_mode in _BLIND_PERMISSION_MODES:
        raise ValueError(
            f"permission_mode={permission_mode!r} auto-approves every tool call, so no call "
            "is ever denied and DEGRADED_BY_DENIAL can never fire. The classifier would "
            "report clean runs forever."
        )

    extra = list(extra_args)
    names = {a.split("=", 1)[0] for a in extra}
    banned = sorted(names & _INVARIANT_BREAKING_FLAGS)
    if banned:
        raise ValueError(
            f"extra_args may not contain {banned}: these flags remove guarantees this "
            "module is built on -- a resumed session is not a fresh one, and a bypassed "
            "permission mode makes the denial classifier structurally blind."
        )
    # Values matter as much as names: --permission-mode is a legitimate flag whose
    # value can do what the banned flags do.
    for i, arg in enumerate(extra):
        value = None
        if arg.startswith("--permission-mode="):
            value = arg.split("=", 1)[1]
        elif arg == "--permission-mode" and i + 1 < len(extra):
            value = extra[i + 1]
        if value in _BLIND_PERMISSION_MODES:
            raise ValueError(
                f"extra_args sets --permission-mode {value!r}, which auto-approves every "
                "tool call and makes the denial classifier structurally blind."
            )
    argv += extra
    return argv


def _usage_int(usage: dict[str, Any], key: str) -> int:
    """One ``usage`` counter as an int, zero when missing or null."""
    return int(usage.get(key, 0) or 0)


def _context_tokens(usage: dict[str, Any]) -> int:
    """Sum one result's context read from its ``usage`` block.

    Missing keys count as zero rather than raising, because a future CLI version
    dropping a key should degrade the measurement, not kill a running chain.
    """
    return sum(_usage_int(usage, key) for key in _CONTEXT_TOKEN_KEYS)


def classify(payload: dict[str, Any], *, done_key: str) -> tuple[IterationVerdict, bool | None]:
    """Judge a parsed CLI result and extract the chain's stop signal.

    The ordering of the checks is the point. A denial is judged BEFORE the
    structured output is read, because a degraded run answers the schema
    perfectly well -- it just answers it about work that routed around a missing
    permission. Reading ``done`` first would let a degraded run end the chain
    claiming success.

    Args:
        payload: The object parsed from ``--output-format json``.
        done_key: Field in ``structured_output`` holding the stop boolean.

    Returns:
        The verdict, and the stop signal when one could be read.
    """
    if payload.get("is_error") or payload.get("terminal_reason") != _TERMINAL_REASON_OK:
        return IterationVerdict.ABNORMAL_END, None

    if payload.get("permission_denials"):
        return IterationVerdict.DEGRADED_BY_DENIAL, None

    structured = payload.get("structured_output")
    if not isinstance(structured, dict) or not isinstance(structured.get(done_key), bool):
        return IterationVerdict.SCHEMA_MISS, None

    return IterationVerdict.OK, structured[done_key]


def run_iteration(
    prompt: str,
    *,
    cwd: pathlib.Path,
    schema: str | None = None,
    done_key: str = "done",
    budget_usd: float | None = None,
    permission_mode: str | None = None,
    model: str | None = None,
    extra_args: Sequence[str] = (),
    timeout_seconds: float = 3600.0,
    runner: Any = subprocess.run,
) -> IterationResult:
    """Launch one fresh headless session and judge how it ended.

    Args:
        prompt: Instruction for this session.
        cwd: Working directory. It materially changes the startup context, and
            therefore the floor cost of the iteration.
        schema: JSON Schema string for structured output.
        done_key: Field in the structured output holding the stop boolean.
        budget_usd: Per-iteration spend cap.
        permission_mode: Value for ``--permission-mode``.
        model: Model alias or full name.
        extra_args: Extra CLI flags, e.g. a ``--allowedTools`` allowlist. Pinning
            the toolset matters for comparisons: an arm that lost a tool mid-run
            is not comparable to one that kept it.
        timeout_seconds: Wall-clock cap on the subprocess.
        runner: Injection point for ``subprocess.run``, so tests never spend money.

    Returns:
        The iteration's verdict, cost, and stop signal. Never raises for a failed
        session: a chain must be able to record a failure and stop cleanly.
    """
    argv = build_command(
        schema=schema,
        budget_usd=budget_usd,
        permission_mode=permission_mode,
        model=model,
        extra_args=extra_args,
    )
    started = time.monotonic()

    def failed(reason: str) -> IterationResult:
        """Build the LAUNCH_FAILED result, with cost marked UNKNOWN rather than zero."""
        return IterationResult(
            verdict=IterationVerdict.LAUNCH_FAILED,
            session_id=None,
            cost_usd=0.0,
            cost_is_known=False,
            context_tokens=0,
            num_turns=0,
            wall_seconds=time.monotonic() - started,
            denials=(),
            done=None,
            structured_output=None,
            text=reason,
        )

    try:
        completed = runner(
            argv,
            cwd=str(cwd),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return failed(f"{type(exc).__name__}: {exc}")

    # stderr is where the CLI puts the reason it refused to start -- a bad flag,
    # a missing binary, an auth failure. Discarding it turns every such case into
    # an indistinguishable "unparseable output", which is what made the variadic
    # --allowedTools bug read as two arms that merely scored zero.
    stderr = (getattr(completed, "stderr", "") or "").strip()
    returncode = getattr(completed, "returncode", 0)
    if returncode:
        return failed(f"claude exited {returncode}: {stderr or '(no stderr)'}")

    try:
        payload = json.loads(completed.stdout)
    except ValueError as exc:
        return failed(f"unparseable stdout ({exc}); stderr: {stderr or '(none)'}")

    if not isinstance(payload, dict):
        return failed("CLI returned JSON that was not an object")

    # Everything below reads caller-opaque fields out of another program's output.
    # _context_tokens already degrades rather than raising on a missing key; this
    # extends the same rule to its siblings, because run_chain has no try around
    # the loop body and one malformed payload would otherwise discard every
    # earlier iteration's result along with it.
    try:
        verdict, done = classify(payload, done_key=done_key)
        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        structured = payload.get("structured_output")
        denials = tuple(
            str(d.get("tool_name", "?")) if isinstance(d, dict) else str(d)
            for d in payload.get("permission_denials") or []
        )
        raw_cost = payload.get("total_cost_usd")
        cost_is_known = isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool)
        return IterationResult(
            verdict=verdict,
            session_id=payload.get("session_id"),
            cost_usd=float(raw_cost) if cost_is_known else 0.0,
            cost_is_known=cost_is_known,
            context_tokens=_context_tokens(usage),
            input_tokens=_usage_int(usage, "input_tokens"),
            cache_creation_tokens=_usage_int(usage, "cache_creation_input_tokens"),
            cache_read_tokens=_usage_int(usage, "cache_read_input_tokens"),
            num_turns=int(payload.get("num_turns", 0) or 0),
            wall_seconds=time.monotonic() - started,
            denials=denials,
            done=done,
            structured_output=structured if isinstance(structured, dict) else None,
            text=str(payload.get("result", "")),
        )
    except (TypeError, ValueError, AttributeError) as exc:
        return failed(f"result JSON had an unexpected shape: {type(exc).__name__}: {exc}")


def append_ledger(
    path: pathlib.Path,
    label: str,
    index: int,
    result: IterationResult,
    *,
    run_id: str = "",
) -> bool:
    """Append one iteration to a JSONL ledger, creating the file if needed.

    Written per iteration rather than at the end so a chain that is killed
    mid-run still leaves the cost of what it already spent on disk.

    A write failure is REPORTED, never raised. Raising would propagate out of
    ``run_chain``'s loop and discard every IterationResult collected so far --
    losing the whole record to protect part of it, which is the opposite of what
    a durability mechanism is for.

    Args:
        path: JSONL file to append to.
        label: Distinguishes chains sharing one file.
        index: Position within this chain, restarting at 0 per chain.
        result: The iteration to record.
        run_id: Identifier separating two runs of the SAME label into the same
            file. Without it, re-running a chain into an existing ledger produces
            duplicate (label, index) rows that cannot afterwards be told apart.

    Returns:
        True if the row was written, False if it could not be.
    """
    row = {
        "run_id": run_id,
        "label": label,
        "index": index,
        "verdict": result.verdict.value,
        "session_id": result.session_id,
        "cost_usd": result.cost_usd,
        "cost_is_known": result.cost_is_known,
        "context_tokens": result.context_tokens,
        "input_tokens": result.input_tokens,
        "cache_creation_tokens": result.cache_creation_tokens,
        "cache_read_tokens": result.cache_read_tokens,
        "num_turns": result.num_turns,
        "wall_seconds": round(result.wall_seconds, 2),
        "denials": list(result.denials),
        "done": result.done,
        "detail": result.text if not result.cost_is_known else "",
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return True
    except OSError:
        return False


def run_chain(
    prompt: str,
    *,
    cwd: pathlib.Path,
    max_iterations: int,
    chain_budget_usd: float,
    ledger: pathlib.Path | None = None,
    label: str = "chain",
    run_id: str = "",
    stop_on_degraded: bool = True,
    **iteration_kwargs: Any,
) -> ChainResult:
    """Run fresh sessions back to back until the work reports itself done.

    Each iteration gets the identical prompt and a context of its own; the file
    the prompt names is what carries the work forward. The loop stops on the
    first of: the stop signal, a failed iteration, the iteration cap, or the
    cumulative budget.

    THE BUDGET IS A REAL CEILING, not merely a pre-launch check. Two mechanisms,
    and the second is the one that matters: the loop refuses to START an
    iteration once recorded spend reaches the ceiling, AND every iteration is
    launched with ``--max-budget-usd`` set to the budget still remaining, so a
    single runaway session cannot carry the total past the cap. Without the
    second, the ceiling bounded only what a chain began -- one $50 iteration
    against a $5 ceiling was permitted, and the chain then stopped, having
    already overspent tenfold.

    An explicit per-iteration ``budget_usd`` still applies; the remaining budget
    is used as an additional upper bound on it, never as a way to raise it.

    Args:
        prompt: Instruction handed to every iteration, unchanged.
        cwd: Working directory for every iteration.
        max_iterations: Hard cap on sessions launched.
        chain_budget_usd: Cumulative spend ceiling across the chain.
        ledger: JSONL path to append each iteration to.
        label: Ledger label distinguishing this chain from others in one file.
        run_id: Passed to the ledger so two runs of the same label in one file
            stay separable.
        stop_on_degraded: Whether a denied tool call ends the chain. Default
            True: continuing would chain later work onto work that silently
            routed around a missing permission.
        **iteration_kwargs: Passed through to ``run_iteration``.

    Returns:
        Every iteration run, and the reason the chain stopped.
    """
    results: list[IterationResult] = []
    spent = 0.0
    stop_reason = "max_iterations"
    caller_cap = iteration_kwargs.pop("budget_usd", None)

    for index in range(max_iterations):
        remaining = chain_budget_usd - spent
        if remaining <= 0:
            stop_reason = "budget_exhausted"
            break

        # The child's own cap is the tighter of what the caller asked for and
        # what the chain can still afford, so the ceiling holds even if one
        # iteration would otherwise run away.
        cap = remaining if caller_cap is None else min(caller_cap, remaining)
        result = run_iteration(prompt, cwd=cwd, budget_usd=cap, **iteration_kwargs)
        results.append(result)
        spent += result.cost_usd
        if ledger is not None and not append_ledger(ledger, label, index, result, run_id=run_id):
            # Losing the durable record is a real degradation of the run, so it
            # is surfaced rather than swallowed -- but the results already in
            # memory are returned intact.
            stop_reason = "ledger_write_failed"
            break
        if not result.cost_is_known:
            # A run that reported no cost may still have spent. The budget can
            # no longer be enforced from what was recorded, so the chain stops
            # rather than launching another child with a cap it cannot trust.
            stop_reason = result.verdict.value if not result.is_usable else "cost_unknown"
            break

        if result.is_usable:
            if result.done:
                stop_reason = "done"
                break
        elif result.verdict is IterationVerdict.DEGRADED_BY_DENIAL and not stop_on_degraded:
            continue
        else:
            stop_reason = result.verdict.value
            break

    return ChainResult(iterations=tuple(results), stop_reason=stop_reason)


def _build_parser() -> argparse.ArgumentParser:
    """Assemble the CLI parser, kept separate so tests can exercise it directly."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt", help="Instruction handed to every iteration, unchanged.")
    parser.add_argument("--cwd", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--budget-usd", type=float, default=5.0, help="Cumulative chain ceiling.")
    parser.add_argument("--iteration-budget-usd", type=float, default=None)
    parser.add_argument("--ledger", type=pathlib.Path, default=None)
    parser.add_argument("--label", default="chain")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Separates two runs of the same label appended to one ledger file. "
        "Auto-generated when omitted, since an operator who forgets it is exactly "
        "the case that produces indistinguishable duplicate rows.",
    )
    parser.add_argument("--permission-mode", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--schema",
        default='{"type":"object","properties":{"done":{"type":"boolean"},'
        '"next":{"type":"string"}},"required":["done","next"]}',
        help="JSON Schema for structured output; must define the done field.",
    )
    parser.add_argument("--done-key", default="done")
    parser.add_argument(
        "--continue-on-degraded",
        action="store_true",
        help="Keep going after a denied tool call. Off by default, and rarely right.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a chain from the command line and print a one-line-per-iteration summary.

    Returns:
        0 when the chain finished because the work reported itself done, 1 otherwise,
        so a shell caller can gate on real completion rather than on exit-without-crash.
    """
    args = _build_parser().parse_args(argv)
    chain = run_chain(
        args.prompt,
        cwd=args.cwd,
        max_iterations=args.max_iterations,
        chain_budget_usd=args.budget_usd,
        ledger=args.ledger,
        label=args.label,
        run_id=args.run_id if args.run_id is not None else new_run_id(),
        stop_on_degraded=not args.continue_on_degraded,
        schema=args.schema,
        done_key=args.done_key,
        budget_usd=args.iteration_budget_usd,
        permission_mode=args.permission_mode,
        model=args.model,
    )
    for index, result in enumerate(chain.iterations):
        denied = f" denied={','.join(result.denials)}" if result.denials else ""
        print(
            f"[{index}] {result.verdict.value} turns={result.num_turns} "
            f"ctx={result.context_tokens} ${result.cost_usd:.4f}{denied}"
        )
    qualifier = "" if chain.cost_is_complete else " (LOWER BOUND: an iteration reported no cost)"
    print(
        f"stop={chain.stop_reason} iterations={len(chain.iterations)} "
        f"total_ctx={chain.total_context_tokens} total=${chain.total_cost_usd:.4f}{qualifier}"
    )
    return 0 if chain.completed else 1


if __name__ == "__main__":
    sys.exit(main())
