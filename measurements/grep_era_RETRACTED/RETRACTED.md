# RETRACTED — do not fit these numbers

The data in this directory was produced with `Grep` in the tool allowlist. That
let one session answer many files in a single cross-file call instead of reading
them one at a time, so tool calls stopped scaling with the task size — which is
the entire quantity the harness measures. The resulting curve was sublinear (cost
rose 1.8x while N rose 4x); that looked like a finding and was an artifact.

**These cells share (arm, N) coordinates with valid cells** — `single` at N=24 and
`chain` at N=24 both appear here and in `read_single_24/` and `read_chain_24/`
with different costs. Anything that globs `measurements/` for summaries will
collide unless it excludes this directory. `fit_break_even.py` reads only the
explicitly named `read_<arm>_<n>/` paths for that reason.

Kept rather than deleted because a correction is only legible beside what it
corrects. This marker exists because the directory name alone travels poorly:
`summary.json` here is a bare JSON array with no in-file indication that it is
withdrawn, so a file read out of its directory context shows nothing wrong.
