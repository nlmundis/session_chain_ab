# session_chain_ab

Is it cheaper to run a long Claude Code task as a chain of short fresh
sessions, or as one long session? This is a harness for measuring that on one
identical task, with a deterministic grader, plus one small **pilot**
measurement. The pilot is not a benchmark.

```bash
python3 fit_break_even.py --check-alternative   # re-derive the pilot result, free
python3 ab_compare.py --repo ~/code/some-python-repo --recursive --files 24 --out /tmp/ab
                                                # run your own; spends real money
```

Standard library only, Python 3.9 or newer. Running a measurement needs the
`claude` CLI and spends money on your account; re-deriving the committed pilot
does not.

---

## The loop is not new

Running a task as a loop of fresh headless sessions, with the state carried in
files, is a known technique. Credit where it is due:

- Geoffrey Huntley's **Ralph** (`while :; do cat PROMPT.md | claude-code ; done`),
  [ghuntley.com/ralph](https://ghuntley.com/ralph/)
- Anthropic's own [cwc-long-running-agents](https://github.com/anthropics/cwc-long-running-agents),
  whose loop calls `claude -p` with "Each pass is a fresh context"
- [continuous-claude](https://github.com/AnandChowdhary/continuous-claude)

`session_loop.py` is that loop, built for an experiment rather than for
production. It keeps a per-iteration cost ledger, reads a typed stop signal,
enforces hard budget caps, and refuses to build on silently degraded work.
That last guard matters: a headless run whose tool calls were denied still
reports `is_error: false` with `terminal_reason: "completed"`.

What this repo adds is the **measurement**: the same task run both ways, graded
by a rule rather than a model, with the break-even fitted as a band.

## The question

Every tool call re-reads the session's whole accumulated context, so a long
session's per-turn cost grows as it goes. A fresh session resets that growth,
but it re-pays its startup context (system prompt, tool schemas, CLAUDE.md,
memory) and has to re-read whatever state it needs.

Anthropic's cost guidance advises `/clear` when switching to unrelated work,
because "Stale context wastes tokens on every subsequent message", and notes that
history is re-read at the cached token rate
([docs](https://code.claude.com/docs/en/costs)). Whether clearing also pays
**within** one long task is what the harness measures.

## The task

For each of N Python modules, report the number of top-level `def` statements
and whether the module has a docstring. Two properties make it a usable
instrument:

- **Tool calls scale with N.** Both arms get `Read` and `Write` only, enforced
  with `--tools Read,Write`. `Bash` would answer every file in one
  `grep -c`, and `Grep` aggregates across files in one call. Allowing Grep is
  how the first curve went wrong; see [What was retracted](#what-was-retracted).
- **A rule grades it.** Answers are checked against Python's `ast` module, so no
  model scores either arm. Cost is reported per **correct** file.

The single arm does all N files in one session. The chained arm does a batch of
files per fresh session (`--batch`, default 4), carrying progress in a report
file.

## Pilot result

Six cells, one run each, measured 2026-09-02 on the author's machine. Every arm
graded every file correct.

| Arm | N | Sessions | Turns | Context tokens | Cost | $ per correct file |
|---|---|---|---|---|---|---|
| single | 12 | 1 | 19 | 652,085 | $1.6348 | 0.136 |
| single | 24 | 1 | 38 | 1,782,826 | $2.9297 | 0.122 |
| single | 48 | 1 | 68 | 5,722,576 | $6.3807 | 0.133 |
| single | 72 | 1 | 92 | 11,584,126 | $11.5596 | 0.161 |
| chain | 24 | 7 | 72 | 2,073,643 | $4.5548 | 0.190 |
| chain | 48 | 13 | 152 | 4,273,149 | $9.0413 | 0.188 |

What the pilot shows, each figure recomputable from `measurements/`:

- **Chaining cost more at both sizes measured**: 1.555x the single session at
  N = 24 and 1.417x at N = 48.
- **In the single arm, context per turn grew 3.669x from N = 12 to N = 72**
  (34,320 to 125,914 tokens), **while cost per turn grew 1.460x** ($0.0860 to
  $0.1256).
- **The fitted break-even band is N = 94 to 107 files**. Above that, chaining
  would be cheaper.

## Read this before quoting any number

- **In the pilot, the tool restriction held by instruction only.** The pilot
  passed `--allowedTools Read Write`, which pre-approves those tools and
  restricts nothing: "To restrict which tools are available, use `--tools`
  instead" ([CLI reference](https://code.claude.com/docs/en/cli-reference)),
  and read-only Bash commands such as `grep` and `wc` run without a prompt in
  every mode ([permissions](https://code.claude.com/docs/en/permissions)). The
  arms were told Bash was unavailable, and no tool calls were recorded, so
  whether any arm took a cross-file shortcut is unknown. Turns grew with N
  (19, 38, 68 and 92 turns for 12, 24, 48 and 72 files), which fits reading
  files one at a time but does not prove it. The harness now passes `--tools`.
- **One run per cell.** Six data points, no replication. Run-to-run noise is
  unmeasured, so none of the scatter is known.
- **The band is an extrapolation.** The largest N measured is 72 and the band
  starts at 94. It is a projection of fitted curves, not an observation.
- **The band is a modelling choice.** Two symmetric fits give 94.2 and 107.3,
  and the retracted asymmetric fit gives 95.0. They are not equally good:
  symmetric B, the upper edge, misses the N = 12 cell by 20.2%, and every
  chain line has only two cells to fit. The script prints each fit's
  residuals and quotes the span, never a point.
- **One task, one model, one machine.** A wide, shallow sweep is the shape that
  most punishes a long session. A deep task that needs its history would land
  differently. The pilot recorded neither its model nor its batch size, and
  startup context depends on what your sessions load.
- **The grading cannot be re-derived.** These cells predate the harness
  recording `ground_truth.json` and `provenance.json`, and the modules were in a
  private repository. You can check the arithmetic, not the grading.
- **Why the single arm was cheaper is not established.** Prompt caching is the
  obvious explanation, but a model with one constant price per context token
  and no caching term reproduces five of the six cells within 6%
  (`--check-alternative`). The pilot ledgers stored only the sum of cached and
  uncached tokens. `session_loop` now records the split for new runs.
- **Two variables were not controlled.** Claude Code's prompt-cache lifetime is
  one hour on a subscription and five minutes on an API key
  ([docs](https://code.claude.com/docs/en/prompt-caching)). Sequential sessions
  share a cached prefix only when the git status snapshot taken at startup
  matches, so a chained arm whose passes change the working tree may re-write
  its cache every pass. Whether that happened in the pilot is unknown.

## What was retracted

Both corrections are kept in the repo, next to what they correct:

- **A curve.** The first measurement allowed `Grep`, which let one call answer
  many files. Cost rose 1.8x while N rose 4x: a sublinear curve that looked like
  a finding and was an artifact. The N = 24 cells are kept in
  `measurements/grep_era_RETRACTED/` and are never fitted; the larger cell behind
  the 4x was not kept.
- **A point and a mechanism.** The first analysis published "BREAK-EVEN: N = 95"
  from an asymmetric fit, and credited prompt caching without evidence that could
  distinguish it. `fit_break_even.py` carries both retractions in its docstring
  and still prints the old fit, labelled `published`, so the asymmetry stays
  visible.

## Run your own

```bash
# one session vs a chain, 24 modules from a repo of your own
python3 ab_compare.py --repo ~/code/some-python-repo --recursive \
    --files 24 --batch 4 --budget-usd 10 --out /tmp/ab-24 --model claude-sonnet-5

# one arm at a time, to spread the spend
python3 ab_compare.py --repo ~/code/some-python-repo --files 48 --arm single --out /tmp/ab-48
```

- **It spends money.** `--budget-usd` is a ceiling **per arm**, passed to the
  CLI's `--max-budget-usd`. The pilot's largest cell cost $11.56.
- **Keep `--out` outside the measured repository.** Writing the report inside it
  changes git status between chained passes, which is one of the uncontrolled
  variables above.
- **The arms can write.** Both run under `acceptEdits` with `Read` and `Write`.
  The modules are read-only by instruction, not by permission. The ground truth
  is captured once, before either arm starts, and every arm is graded against
  it. If the modules change during the run, each summary is marked
  `repo_modified_during_run` and the exit code is 1. Point it at a clean
  checkout.
- **It refuses a task it cannot finish.** `--max-iterations` defaults to enough
  sessions for the chained arm to cover every file (`ceil(files / batch) + 2`),
  and a repository holding fewer modules than `--files` is refused unless you
  pass `--allow-short-pool`. Hidden, environment and cache directories are never
  drawn from.
- **Arms run in separate sittings must measure the same task.** A summary keeps
  arms from earlier runs only when their pool, batch, model and tools match;
  otherwise the run is refused before anything is spent.
- **Pass `--model`.** Provenance records it; without it, the harness cannot
  tell which default model the CLI used.

Each run writes `ground_truth.json`, `provenance.json`, `ledger.jsonl` (with
`input_tokens`, `cache_creation_tokens` and `cache_read_tokens` per session) and
`summary.json` to `--out`. To fit your own cells, lay them out like
`measurements/` and edit `SINGLE_POINTS` and `CHAIN_POINTS` in
`fit_break_even.py`. They are explicit on purpose, so a directory appearing
never silently changes a fit.

`session_loop.py` also runs on its own:

```bash
python3 session_loop.py "Read PROGRESS.md, do the next unchecked item, update PROGRESS.md." \
    --max-iterations 5 --budget-usd 5 --ledger chain.jsonl \
    --schema '{"type":"object","properties":{"done":{"type":"boolean"},"next":{"type":"string"}},"required":["done","next"]}'
```

It refuses a `--permission-mode` of `bypassPermissions`, and, when used as a
library, extra CLI arguments `--resume`, `--continue`, `--fork-session`,
`--dangerously-skip-permissions` and `--settings`. Each one breaks either the
fresh-session premise or the denial check. `dontAsk` is allowed: it denies what
would prompt, so the denials still reach the classifier. It also stops a chain
whose session reported no cost, since the budget can no longer be enforced.

## Limits

- **The CLI's JSON output is not a documented contract.** `session_loop` reads
  `terminal_reason`, `permission_denials`, `structured_output`,
  `total_cost_usd` and the `usage` counters from `claude -p --output-format json`.
  A CLI change can break it. Missing counters degrade to zero rather than
  stopping a chain.
- **The stop signal is the model's own claim.** The `done` field is parsed as a
  typed value, but the model decides it. Only the `ast` grader in `ab_compare`
  judges the work.
- **Measured ledgers hash session ids.** The committed ledgers replace each
  Claude Code session id with a SHA-256 prefix. Runs you make keep the real ids.
- **Tested on Linux and macOS.** CI runs ruff, mypy, the suite and the mutation gate on
  Ubuntu under Python 3.9 through 3.14 and on macOS under 3.12. The tests fake
  the `claude` CLI and never spend anything; they also pass locally with no
  `claude` on PATH. Windows has not been run.

## Develop

```bash
pip install -r requirements-dev.txt   # ruff, mypy and mutt_check, pinned
make test       # unit suite, no network, no claude
make check      # ruff, mypy (strict, against 3.9), unit suite, mutation gate
```

The scripts themselves need nothing beyond the standard library; the pinned
tools are for development and CI only. Test methods are exempt from the
docstring and annotation requirements, since each is named as a sentence
stating what it pins; everything else is held to both.

The mutation gate uses [mutt_check](https://github.com/nlmundis/mutt_check).
Each entry in `mutt_check.toml` reverts one design decision and names the test
that must fail.

MIT licensed.
