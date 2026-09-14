#!/usr/bin/env python3
"""Measure whether chaining short sessions beats one long session, on one identical task.

WHY THIS EXISTS RATHER THAN AN ESTIMATE
    ``session_loop.py`` runs a task as a chain of fresh sessions, but whether
    that actually saves anything is an empirical question with a real
    break-even, not a principle. Every fresh session re-pays the startup context
    (see the measured floor in ``session_loop``), so chaining wins only when a
    long session's per-turn re-read cost exceeds that reload. Below some task
    length, chaining is strictly worse.

    This finds that break-even by running the SAME task both ways and comparing
    dollars, context tokens, and correctness.

THE TASK, AND WHY THIS ONE
    Report, for each of N Python modules, the number of top-level ``def``
    statements and whether the module has a docstring.

    Two properties make it a fair instrument. It is TOOL-HEAVY: the cost being
    measured is context re-read per tool call, so the task must force many calls
    rather than one long think. And it is MACHINE-GRADABLE against ``ast``, so
    neither arm is scored by a model -- a rule judges both, which is the whole
    point of not having a model grade a model.

    IT IS NOT READ-ONLY, and an earlier version of this docstring claimed it was.
    Each arm holds Write, runs under ``permission_mode="acceptEdits"``, and
    ``run_arm`` deletes any stale report before starting. The graded modules are
    read-only BY INSTRUCTION rather than by permission, so a misbehaving arm
    could in principle edit the files it is graded on. ``run_arm`` therefore
    snapshots the ground truth BEFORE the arm runs, so no arm can be graded
    against its own edit.

    Correctness is reported alongside cost because a cheap arm that quietly
    answered fewer files is not cheaper. The comparison is dollars PER CORRECT
    FILE, not dollars.

WHAT THIS DOES NOT SETTLE
    One task, one repo, one model. It measures the break-even for a wide shallow
    sweep, which is the shape that most punishes a long session. A deep task that
    genuinely needs its accumulated context would land differently, and this
    harness would need a different task to say so.

    Two variables it does not control. Prompt-cache lifetime: Claude Code uses a
    one-hour cache on a subscription and five minutes on an API key, which moves
    both arms' costs. And git status: sequential sessions share a cached prefix
    only when the git status snapshot taken at startup matches, so a chained arm
    whose passes change the working tree (by writing its report inside the
    repository, say) may re-write its cache every pass. Keep ``--out`` outside
    the measured repository, and read the ledger's cache split to see which
    happened.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

import session_loop as sl

# Tools both arms may use. Identical across arms, so the permission posture is
# never itself a difference between them.
#
# ENFORCED WITH --tools, NOT ONLY --allowedTools. The pilot passed only
# ``--allowedTools``, which PRE-APPROVES the named tools and restricts nothing:
# per the CLI reference, "To restrict which tools are available, use `--tools`
# instead", and a built-in set of read-only Bash commands (grep, wc, cat, ...)
# runs without a prompt in every mode. So in the committed pilot the restriction
# below held BY INSTRUCTION ONLY -- the prompt told the arms Bash was
# unavailable -- and nothing recorded whether they complied.
#
# Read and Write ONLY, and the exclusions are the instrument rather than a
# restriction. The cost this harness measures is context re-read per tool call,
# so the task must make tool calls scale with N. Two tools break that:
#
#   Bash  -- one `python3 -c "import ast"` answers every file at once.
#   Grep  -- aggregates ACROSS files in a single call, which is the same
#            shortcut wearing different clothes.
#
# Grep was allowed in the first curve and quietly destroyed the measurement:
# cost rose only 1.8x while N rose 4x, a sublinear curve that looked like a
# finding and was an artifact of the sweep collapsing into a few cross-file Grep
# calls. Read is the only tool here that is inherently one-file-at-a-time.
#
# PROVENANCE, stated because this comment used to cite per-tool call counts that
# nothing in this repo could confirm: that breakdown came from a session
# transcript under ~/.claude/projects/, outside the repo and not retained. The
# ledger schema records no tool-call counts, so treat any such figure as
# unreproducible here. What is committed and checkable is the discarded curve
# itself, under measurements/grep_era_RETRACTED/.
_ALLOWED_TOOLS = "Read Write"
# The same set in the comma-separated form ``--tools`` takes.
_TOOLS = "Read,Write"

_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "done": {"type": "boolean"},
            "next": {"type": "string"},
            "files_completed": {"type": "integer"},
        },
        "required": ["done", "next"],
    }
)


def _common_root(paths: Sequence[pathlib.Path]) -> pathlib.Path:
    """Return the directory every given file sits under.

    Derived from the PARENTS, never from the paths themselves. ``os.path.commonpath``
    of a single file returns that file, so keying off it directly would make the
    relative path of a one-file pool ``"."`` -- a key no report could ever match,
    turning a small run into a silent zero score.

    Args:
        paths: One or more file paths.

    Returns:
        Their common parent directory.
    """
    return pathlib.Path(os.path.commonpath([str(p.parent) for p in paths]))


@dataclasses.dataclass(frozen=True)
class FileTruth:
    """The gradable facts about one module, derived from ``ast`` rather than asked for."""

    top_level_defs: int
    has_docstring: bool


def ground_truth(
    paths: Sequence[pathlib.Path], *, repo: pathlib.Path | None = None
) -> dict[str, FileTruth]:
    """Compute the correct answer for every file, so grading needs no judgment.

    Keyed by the SAME relative-path string ``_task_prompt`` hands the arm, so a
    report key and a truth key cannot fail to line up. Keying by bare basename
    was wrong in principle for a recursive pool: two modules sharing a name in
    different directories collapse into one entry, silently shrinking the graded
    set below the N the prompt asked for. (Measured 2026-09-02: zero collisions
    at every N up to the full 154-module pool, so no committed measurement was
    affected -- the key change closes the hole rather than fixing a wrong number.)

    Args:
        paths: Modules to measure.
        repo: Root the keys are relative to; defaults to the common ancestor.

    Returns:
        Mapping of relative path to its truth. Files that fail to parse are
        OMITTED, which shrinks the denominator -- ``main`` records the count of
        dropped files in the provenance so the gap is visible rather than
        absorbed into a smaller total_files.
    """
    root = repo if repo is not None else _common_root(paths)
    truth: dict[str, FileTruth] = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        defs = sum(
            1
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        try:
            key = str(path.relative_to(root))
        except ValueError:
            key = str(path)
        truth[key] = FileTruth(
            top_level_defs=defs, has_docstring=ast.get_docstring(tree) is not None
        )
    return truth


# Directories never drawn from: environments and caches hold installed or
# generated modules, not the repository's own code.
_EXCLUDED_DIRS = frozenset({"venv", "env", "site-packages", "node_modules", "__pycache__", "build", "dist"})


def _is_excluded(relative: pathlib.Path) -> bool:
    """Whether a module path, relative to the repo, sits under a hidden, environment or cache directory."""
    return any(part.startswith(".") or part in _EXCLUDED_DIRS for part in relative.parts[:-1])


def select_files(repo: pathlib.Path, limit: int, *, recursive: bool = False) -> list[pathlib.Path]:
    """Pick the task's modules deterministically, so both arms face the same work.

    Sorted and truncated, rather than sampled, because a comparison whose two arms
    saw different files measures nothing. Truncating a fixed ordering also means
    the N=24 pool is a strict PREFIX of the N=96 pool, which is what lets several
    task sizes sit on one cost curve instead of being unrelated points.

    THE PREFIX PROPERTY HOLDS ONLY WITHIN ONE VALUE OF ``recursive``. The two
    modes glob different populations, so a non-recursive pool of 48 is NOT the
    first 48 of a recursive pool of 96 -- switching the flag between points of a
    single curve silently breaks the comparison. It also holds only for a FIXED
    repository: the pool is a live glob, so adding or removing a module between
    runs changes which files a later point covers. Both are why ``main`` writes a
    provenance record naming the flag, the count, and the exact file list.

    Args:
        repo: Directory to search.
        limit: How many modules the task covers.
        recursive: Search subdirectories too, needed because the break-even sits
            above the number of modules in the repo root. NOTE: the prefix
            property does NOT survive a change of this flag -- see above. (An
            earlier version of this line claimed it did, and also quoted a fixed
            root module count, which drifts every time a module is added.)

    Returns:
        Up to ``limit`` module paths, in a stable order.
    """
    pattern = "**/*.py" if recursive else "*.py"
    found = [p for p in repo.glob(pattern) if not _is_excluded(p.relative_to(repo))]
    return sorted(found, key=lambda p: (p.name, str(p)))[:limit]


def grade(report_path: pathlib.Path, truth: dict[str, FileTruth]) -> dict[str, Any]:
    """Score one arm's written report against the ground truth.

    A missing file counts as not-correct rather than being skipped: an arm that
    stopped early must not score as well as one that finished, or the cost
    comparison rewards giving up.

    Args:
        report_path: JSON file the arm was told to write.
        truth: Output of ``ground_truth``.

    Returns:
        Counts of attempted, correct, wrong and missing files, plus the FULL
        sorted list of wrong names. Not truncated: the earlier ten-name cap
        dropped the rest with no marker, which is exactly the silent loss the
        harness is meant to make visible.
    """
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}

    entries = raw.get("files", raw) if isinstance(raw, dict) else {}
    if not isinstance(entries, dict):
        entries = {}

    correct, wrong_names, attempted = 0, [], 0
    for name, expected in truth.items():
        got = entries.get(name)
        if not isinstance(got, dict):
            continue
        attempted += 1
        # Types are checked, not just values. Python's True == 1, so a report
        # saying top_level_defs=True would otherwise score as 1, and any truthy
        # value -- including the STRING "false" -- would satisfy a bool() cast.
        # A malformed report must fail, never coincide.
        defs, doc = got.get("top_level_defs"), got.get("has_docstring")
        # An integral float is accepted because the prompt asks for "a JSON
        # number" and json.loads turns 3.0 into a float; rejecting it would
        # score a compliant report wrong. A bool is still refused, since
        # True == 1 in Python and would otherwise coincide with a count of one.
        ok = False
        if isinstance(defs, (int, float)) and not isinstance(defs, bool):
            ok = (
                float(defs).is_integer()
                and int(defs) == expected.top_level_defs
                and isinstance(doc, bool)
                and doc == expected.has_docstring
            )
        if ok:
            correct += 1
        else:
            wrong_names.append(name)

    return {
        "total_files": len(truth),
        "files_graded": len(truth),
        "attempted": attempted,
        "correct": correct,
        "wrong": len(wrong_names),
        "missing": len(truth) - attempted,
        "wrong_names": sorted(wrong_names),
    }


def _task_prompt(
    report_path: pathlib.Path,
    files: Sequence[pathlib.Path],
    batch: int | None,
    *,
    repo: pathlib.Path | None = None,
) -> str:
    """Write the instruction both arms receive, differing only in the batch clause.

    Kept in one function so the two arms cannot drift apart in wording, which
    would confound the measurement with prompt differences.

    Args:
        report_path: Where the arm must maintain its JSON report.
        files: The modules to cover.
        batch: Files per chained session; None for the single-session arm.
        repo: Root that listed paths are relative to. Defaults to the common
            ancestor of ``files``.

    Returns:
        The instruction text.
    """
    root = repo if repo is not None else _common_root(files)
    # Paths RELATIVE TO THE ROOT, never bare basenames. With --recursive the pool
    # spans the repo root, lib/ and tests/, and an earlier version named only
    # files[0].parent while listing every module by basename -- so for any file
    # outside that one directory the arm was handed a path that does not exist,
    # and with Grep and Glob withdrawn it had no way to find the real one except
    # by guessing. That charged some arms turns that others never paid.
    rel = []
    for f in files:
        try:
            rel.append(str(f.relative_to(root)))
        except ValueError:
            rel.append(str(f))
    names = ", ".join(rel)
    batch_clause = (
        f"Do at most {batch} files that are not already in the report, then stop and "
        "report done=false. Report done=true only when every file is present."
        if batch is not None
        else "Do every file in one session, then report done=true."
    )
    return (
        "For each Python module listed below, determine two facts: the number of "
        "TOP-LEVEL function definitions and whether the module has a module-level "
        "docstring.\n\n"
        "TOP-LEVEL means a def or async def that is a DIRECT child of the module "
        "body. Do not count methods inside classes, nested functions, or a def "
        "inside an if/try/with block even when it sits at column 0. This spells "
        "out the rule so the instruction and the grader cannot disagree.\n\n"
        f"Modules, as paths relative to {root}: {names}\n\n"
        f"Maintain a JSON report at {report_path} with the exact shape:\n"
        '{"files": {"<path as listed above>": {"top_level_defs": <int>, "has_docstring": <bool>}}}\n'
        "Use exactly the path strings listed above as the keys. top_level_defs must be a "
        "JSON number and has_docstring a JSON true/false; a string will be scored wrong.\n"
        "Read the existing report first if it exists and preserve entries already in it; "
        "add your new entries to it.\n\n"
        "Only Read and Write are enabled in this session. Open each file with Read and "
        "count; do not try to answer several files with one call.\n\n"
        f"{batch_clause}\n"
        "Return structured output with done, next, and files_completed."
    )


def run_arm(
    label: str,
    *,
    repo: pathlib.Path,
    files: Sequence[pathlib.Path],
    report_path: pathlib.Path,
    ledger: pathlib.Path,
    batch: int | None,
    max_iterations: int,
    budget_usd: float,
    run_id: str = "",
    model: str | None = None,
    truth: dict[str, FileTruth] | None = None,
) -> tuple[sl.ChainResult, dict[str, Any]]:
    """Run one arm end to end and grade it against a truth snapshot taken first.

    The single-session arm is just a chain with ``max_iterations=1``, so both arms
    go through identical measurement code and any accounting bug hits both.

    THE ORDER MATTERS. Ground truth is computed BEFORE the chain starts. The arms
    hold Write under ``acceptEdits``, so a truth derived afterwards would be
    derived from a repository the arm being graded could have edited -- grading
    it against its own output. Snapshotting first makes that impossible rather
    than merely unlikely.

    Args:
        label: Arm name, used in the ledger and the summary.
        repo: Working directory and the root paths are relative to.
        files: Modules the arm must cover.
        report_path: Where the arm maintains its report.
        ledger: JSONL appended to per iteration.
        batch: Files per session for the chained arm; None for the single arm.
        max_iterations: Session cap for this arm.
        budget_usd: Ceiling for this arm.
        run_id: Separates repeated runs of one arm within a shared ledger.
        model: Model for every session in this arm; None uses the CLI default.
        truth: The ground truth to grade against. ``main`` passes the ONE
            snapshot it took and wrote to ground_truth.json before any arm ran,
            so a later arm is never graded against a repository an earlier arm
            edited. When None, a snapshot is taken here, before this arm starts.

    Returns:
        The chain result and its score.
    """
    if truth is None:
        truth = ground_truth(files, repo=repo)
    report_path.unlink(missing_ok=True)
    chain = sl.run_chain(
        _task_prompt(report_path, files, batch, repo=repo),
        cwd=repo,
        max_iterations=max_iterations,
        chain_budget_usd=budget_usd,
        ledger=ledger,
        label=label,
        run_id=run_id,
        model=model,
        schema=_SCHEMA,
        permission_mode="acceptEdits",
        budget_usd=budget_usd,
        extra_args=["--tools", _TOOLS, "--allowedTools", _ALLOWED_TOOLS],
    )
    return chain, grade(report_path, truth)


def summarise(label: str, chain: sl.ChainResult, score: dict[str, Any]) -> dict[str, Any]:
    """Reduce one arm to the comparable numbers, including cost per CORRECT file.

    Dollars alone would let an arm look cheap by answering less, which is why the
    normalised figure is the headline and raw cost is kept beside it.

    The error evidence travels WITH the summary. ``grade`` computes wrong,
    missing and the wrong names "for inspection", and an earlier version of this
    function dropped all three -- so the only durable artifact recorded that an
    arm scored 46/48 without recording which two files it got wrong, and nothing
    on disk could contradict the correctness half of a published figure.

    Args:
        label: Arm name.
        chain: The chain that ran.
        score: Output of ``grade``.

    Returns:
        The arm's comparable numbers plus its error evidence.
    """
    correct = score["correct"]
    return {
        "arm": label,
        "iterations": len(chain.iterations),
        "stop_reason": chain.stop_reason,
        "turns": sum(i.num_turns for i in chain.iterations),
        "context_tokens": chain.total_context_tokens,
        "cost_usd": round(chain.total_cost_usd, 4),
        "cost_is_complete": chain.cost_is_complete,
        "correct": correct,
        "attempted": score["attempted"],
        "files_graded": score.get("files_graded", score["total_files"]),
        "wrong": score["wrong"],
        "missing": score["missing"],
        "wrong_names": score["wrong_names"],
        "total_files": score["total_files"],
        "cost_per_correct": round(chain.total_cost_usd / correct, 4) if correct else None,
        "tokens_per_correct": round(chain.total_context_tokens / correct) if correct else None,
        "denials": sorted({d for i in chain.iterations for d in i.denials}),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Assemble the CLI parser, kept separate so tests can exercise it without running an arm."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--files", type=int, default=24, help="How many modules the task covers.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Draw modules from subdirectories too, for task sizes larger than the repo root holds.",
    )
    parser.add_argument("--batch", type=int, default=4, help="Files per chained session.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Session cap for the chained arm. Default: enough to finish, ceil(files / batch) + 2.",
    )
    parser.add_argument(
        "--allow-short-pool",
        action="store_true",
        help="Run even when the repository holds fewer modules than --files asks for.",
    )
    parser.add_argument("--budget-usd", type=float, default=10.0, help="Ceiling PER ARM.")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="Directory for artifacts.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Separates repeated runs sharing one --out directory. Auto-generated when omitted.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model for every session. Recorded in the provenance either way: an "
        "unrecorded model makes two runs under different CLI defaults indistinguishable.",
    )
    parser.add_argument(
        "--arm",
        choices=("single", "chain", "both"),
        default="both",
        help="Run one arm only, to spread spend across sittings.",
    )
    return parser


def repo_commit(repo: pathlib.Path) -> str:
    """Return the repo's current commit, or a marker when it cannot be determined.

    Recorded in the provenance because the file pool is a live glob: the same
    ``--files 48`` covers a different 48 modules after a commit adds or removes
    one, so a measurement without a commit id cannot be re-derived. Failure
    returns a marker rather than raising, since an un-versioned directory is a
    reason to label the data, not to refuse to measure.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def provenance(args: argparse.Namespace, files: Sequence[pathlib.Path], truth: dict[str, Any]) -> dict[str, Any]:
    """Record everything needed to re-derive this run's task from the repo.

    Written because the first four-point curve committed only summaries: no N, no
    ``--recursive``, no batch size, no commit, and no file list. A reader could
    not tell which modules any point covered, and in fact could not tell that all
    the points shared one pool -- so the published curve could be neither
    reproduced nor falsified from what was kept.

    Args:
        args: Parsed CLI arguments.
        files: The selected modules.
        truth: Ground truth, whose size reveals any unparseable files dropped.

    Returns:
        A JSON-serialisable provenance record.
    """
    return {
        "repo": args.repo.resolve().name,
        "repo_commit": repo_commit(args.repo),
        "files_requested": args.files,
        "files_selected": len(files),
        "files_graded": len(truth),
        "files_dropped_unparseable": len(files) - len(truth),
        "recursive": bool(args.recursive),
        "batch": args.batch,
        "max_iterations": args.max_iterations,
        "budget_usd": args.budget_usd,
        "arm": args.arm,
        "run_id": args.run_id,
        "model": args.model or "cli default (unrecorded by the CLI itself)",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "allowed_tools": _ALLOWED_TOOLS,
        "file_list": sorted(truth),
    }


def provenance_key(args: argparse.Namespace, truth: dict[str, Any]) -> str:
    """Fingerprint the task an arm measures: its file pool, batch size, model and tool set.

    Two arms may share one summary only when this matches, so running them in
    separate sittings cannot put different experiments side by side.
    """
    task = {"files": sorted(truth), "batch": args.batch, "model": args.model, "tools": _TOOLS,
            "recursive": bool(args.recursive)}
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:16]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison, recording enough to reproduce it, and report per arm.

    Returns:
        0 only if every requested arm ran cleanly to completion AND graded every
        file correctly-or-wrongly without stopping early; 1 otherwise. The old
        condition was merely "at least one correct file per arm", which let an
        arm that died on iteration one with a single lucky file exit 0 -- the
        exact "a run that measured nothing looks like a result" case the exit
        code exists to prevent.
    """
    args = _build_parser().parse_args(argv)
    if args.run_id is None:
        args.run_id = sl.new_run_id()
    # Absolute before anything uses them: the prompt hands --out to a session
    # whose cwd is --repo, so a relative --out would be written inside the
    # measured repository and graded from a different directory.
    args.repo = args.repo.resolve()
    args.out = args.out.resolve()
    files = select_files(args.repo, args.files, recursive=args.recursive)
    if not files:
        print(f"no modules found under {args.repo}", file=sys.stderr)
        return 1
    if len(files) < args.files and not args.allow_short_pool:
        print(
            f"--files {args.files} asked for more modules than {args.repo} holds ({len(files)}"
            f"{'' if args.recursive else ', without --recursive'}); pass --allow-short-pool to run anyway",
            file=sys.stderr,
        )
        return 1
    if args.max_iterations is None:
        args.max_iterations = -(-len(files) // args.batch) + 2
    args.out.mkdir(parents=True, exist_ok=True)

    truth = ground_truth(files, repo=args.repo)
    (args.out / "ground_truth.json").write_text(
        json.dumps({k: dataclasses.asdict(v) for k, v in truth.items()}, indent=1),
        encoding="utf-8",
    )
    (args.out / "provenance.json").write_text(
        json.dumps(provenance(args, files, truth), indent=1), encoding="utf-8"
    )

    ledger = args.out / "ledger.jsonl"
    summary_path = args.out / "summary.json"
    # Carry forward arms already recorded here, so running the arms in separate
    # sittings leaves one complete summary rather than the later run silently
    # deleting the earlier arm's result while its spend stays in the ledger.
    # Only an arm measured on the SAME task may share the summary: a different
    # pool, batch, model or tool set would silently put two experiments side by
    # side, so a mismatch is refused before anything is spent.
    key = provenance_key(args, truth)
    summaries: list[dict[str, Any]] = []
    try:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            summaries = [s for s in existing if isinstance(s, dict)]
    except (OSError, ValueError):
        summaries = []
    arms = ("single", "chain") if args.arm == "both" else (args.arm,)
    foreign = [s.get("arm") for s in summaries if s.get("arm") not in arms and s.get("provenance_key") != key]
    if foreign:
        print(
            f"{summary_path} already holds arm(s) {foreign} measured on a different task "
            "(pool, batch, model or tools); use a fresh --out",
            file=sys.stderr,
        )
        return 1

    for label in arms:
        chain, score = run_arm(
            label,
            repo=args.repo,
            files=files,
            report_path=args.out / f"report_{label}.json",
            ledger=ledger,
            batch=None if label == "single" else args.batch,
            max_iterations=1 if label == "single" else args.max_iterations,
            budget_usd=args.budget_usd,
            run_id=args.run_id,
            model=args.model,
            truth=truth,
        )
        summary = summarise(label, chain, score)
        summary["provenance_key"] = key
        summaries = [s for s in summaries if s.get("arm") != label] + [summary]
        print(json.dumps(summary, indent=1))
        # Written after EVERY arm, not once at the end, so a failure in the
        # second arm cannot discard the first arm's result when its spend is
        # already on disk.
        summary_path.write_text(json.dumps(summaries, indent=1), encoding="utf-8")

    if ground_truth(files, repo=args.repo) != truth:
        # Grading used the snapshot taken before any arm ran, so the scores stand,
        # but an arm edited the modules it was told to only read: the run is not
        # a clean measurement, and says so in its artifacts and its exit code.
        for s in summaries:
            if s.get("arm") in arms:
                s["repo_modified_during_run"] = True
        summary_path.write_text(json.dumps(summaries, indent=1), encoding="utf-8")
        print("the measured modules changed during the run; see repo_modified_during_run", file=sys.stderr)
        return 1

    ran = [s for s in summaries if s["arm"] in arms]
    return 0 if ran and all(
        s["stop_reason"] == "done" and s["missing"] == 0 and s["correct"] > 0 for s in ran
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
