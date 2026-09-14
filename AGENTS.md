# Working on session_chain_ab

A harness that measures whether a long Claude Code task is cheaper as a chain
of fresh `claude -p` sessions or as one long session, plus one committed pilot
measurement. `session_loop.py` is the chain driver, `ab_compare.py` the A/B
experiment with an `ast` grader, `fit_break_even.py` the break-even fit over
`measurements/`. Python 3.9 or newer, standard library only.

## Commands

```bash
make check     # the gate: ruff, mypy, unit suite, then the curated mutation gate
make lint      # ruff only
make typecheck # mypy only, strict, against Python 3.9
make test      # unit suite only, a second or two
```

Neither target calls `claude` or the network; the suite fakes the CLI. Never
run `ab_compare.py` or `session_loop.py` for real without the owner's say-so,
because both spend money on their account. `make check` needs the pinned tools
from `requirements-dev.txt` (ruff, mypy, [mutt_check](https://github.com/nlmundis/mutt_check)).

## Changes and releases

`main` is protected by the ruleset in `.github/rulesets/main.json`: every change
lands through a pull request, squash or rebase, after the `all checks passed`
job is green. To release, push a tag `vX.Y.Z` on a commit already on `main`;
`.github/workflows/release.yml` reruns the whole gate on that commit and only
then creates the GitHub Release. Release tags cannot be moved or deleted
(`.github/rulesets/release-tags.json`), so a mistake is fixed by a new tag.

## Rules for changing this repository

- Every number in the README is recomputed in `tests/test_repo.py` from
  `measurements/` or from the fit's output. Change the data, the fit and the
  prose together, or the suite fails.
- Never edit a committed measurement's numbers. A new run is a new directory,
  and a withdrawn one is kept and labelled, like `grep_era_RETRACTED/`.
- Quote the break-even as a band, never a point, and keep the caveats beside
  any figure: one run per cell, extrapolated beyond the largest N, one task.
- Do not claim a mechanism the ledger cannot distinguish. Caching is plausible,
  not demonstrated, for the pilot.
- A mutant goes into `mutt_check.toml` only together with the test that kills
  it, naming that test class in `suites`.
- No runtime dependencies; development tools are pinned in `requirements-dev.txt`.
  Source is held to ruff (including Google-convention docstrings) and strict
  mypy; test methods are exempt from docstrings and annotations only.
- Keep the floor at 3.9. No `match`, no runtime `X | Y` unions: annotations are fine
  under `from __future__ import annotations`, but nothing may evaluate them.
- A name and its docstring must let a reader predict what a function does, and
  why a caller would reach for it, without opening the body.
