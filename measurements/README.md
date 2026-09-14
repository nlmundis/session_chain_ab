# measurements/

Output from `ab_compare.py`, committed on purpose: these files are the EVIDENCE
for the numbers in the README and in `fit_break_even.py`. A published figure whose source
data lives only on one machine cannot be checked by anyone, which is the failure
this directory exists to prevent. Re-derive the headline with:

```
python3 fit_break_even.py --check-alternative
```

One change from the raw output, made for publication: each ledger row's
`session_id` is replaced by the first 16 hex digits of its SHA-256
(`sha256:...`). The rows stay distinguishable, but no longer carry the real
Claude Code session identifiers of the account that ran them. Nothing in the
analysis reads the id.

## What each directory holds

`read_<arm>_<N>/` — one measurement cell. `summary.json` is the per-arm result,
`ledger.jsonl` one row per session. Runs made after 2026-09-02 also carry
`provenance.json` (the repo commit, `--recursive`, batch size, and the exact file
list) and `ground_truth.json`.

## Known gaps in the committed cells, stated rather than discovered later

The six `read_*` cells predate the provenance record, so for those the task
definition is NOT recoverable from this directory. What is known and was verified
afterwards: every one of them was run with `--recursive`, and the pool is a name-sorted prefix of
`**/*.py` in the private repository the measurements were taken in, which is not
published. What is NOT recoverable: the
exact file list, because that repository was itself being modified between runs —
`fit_break_even.py` and its tests did not exist when the smaller cells ran, so
re-running `select_files` today reproduces a slightly different pool. The effect
is one or two files out of 12–72, which no replicated run has measured against, but it is a real
limit on reproducibility and it is why `provenance.json` now exists.

Each cell is a SINGLE run. There is no replication anywhere in this directory, so
no scatter here is measured.

## grep_era_RETRACTED/

**Do not use these numbers.** This cell was produced with `Grep` in the tool
allowlist, which let a session answer many files in one cross-file call instead of
reading them one at a time. The measured curve came out sublinear — cost rose 1.8x
while N rose 4x — which looked like a finding and was an artifact of the
instrument. It is kept, rather than deleted, because the correction is only
legible next to what it corrects. It shares no axis with the `read_*` cells and
must never be fitted alongside them.
