"""Repo invariants: the published numbers, and the docs that describe the code.

The README quotes a pilot result. Every figure in it is recomputed here from
the committed measurements and from the fit's own output, so an edit to the
data, the fit, or the prose cannot leave the three disagreeing.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import stat
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import fit_break_even as fb  # noqa: E402
import session_loop as sl  # noqa: E402

CELLS = [("single", 12), ("single", 24), ("single", 48), ("single", 72), ("chain", 24), ("chain", 48)]


def _read(name: str) -> str:
    """File contents, with the handle closed."""
    with open(os.path.join(REPO, name), encoding="utf-8") as fh:
        return fh.read()


def _summary(arm: str, n: int) -> dict:
    """The committed summary row for one cell."""
    rows = json.loads(_read(os.path.join("measurements", f"read_{arm}_{n}", "summary.json")))
    return rows[0]


class PilotNumbersTest(unittest.TestCase):
    """Every number the README states about the pilot comes out of the committed files."""

    def setUp(self) -> None:
        self.table = _read("README.md")
        self.readme = " ".join(self.table.split())

    def test_the_table_matches_every_committed_summary(self) -> None:
        for arm, n in CELLS:
            row = _summary(arm, n)
            line = (f"| {arm} | {n} | {row['iterations']} | {row['turns']} | {row['context_tokens']:,} | "
                    f"${row['cost_usd']:.4f} | {row['cost_usd'] / row['correct']:.3f} |")
            with self.subTest(cell=f"{arm}_{n}"):
                self.assertIn(line, self.table)
                self.assertEqual(row["correct"], row["total_files"], "the README says every file was correct")

    def test_the_ratios_are_recomputed_not_copied(self) -> None:
        small, large = _summary("single", 12), _summary("single", 72)
        ctx = (large["context_tokens"] / large["turns"]) / (small["context_tokens"] / small["turns"])
        cost = (large["cost_usd"] / large["turns"]) / (small["cost_usd"] / small["turns"])
        self.assertIn(f"grew {ctx:.3f}x from N = 12 to N = 72", self.readme)
        self.assertIn(f"grew {cost:.3f}x", self.readme)
        self.assertIn(f"({small['context_tokens'] / small['turns']:,.0f} to {large['context_tokens'] / large['turns']:,.0f} tokens)",
                      self.readme)
        for n in (24, 48):
            ratio = _summary("chain", n)["cost_usd"] / _summary("single", n)["cost_usd"]
            with self.subTest(n=n):
                self.assertIn(f"{ratio:.3f}x", self.readme)

    def test_the_band_and_the_three_fits_are_what_the_script_prints(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(fb.main([]), 0)
        out = buf.getvalue()
        low, high = re.search(r"BREAK-EVEN BAND: N = (\d+) to (\d+) files", out).groups()
        self.assertIn(f"N = {low} to {high} files", self.readme)
        fits = dict(re.findall(r"^(published|symmetric A|symmetric B) .*\n(?:.*\n){2}  verdict: chaining wins above N = ([\d.]+)", out, re.M))
        self.assertEqual(set(fits), {"published", "symmetric A", "symmetric B"})
        self.assertIn(f"Two symmetric fits give {fits['symmetric A']} and {fits['symmetric B']}", self.readme)
        self.assertIn(f"the retracted asymmetric fit gives {fits['published']}", self.readme)
        b_residuals = out.split("residuals of the symmetric B fit", 1)[1]
        worst = re.search(r"single N= 12: .*\(([+-][\d.]+)%\)", b_residuals).group(1)
        self.assertIn(f"misses the N = 12 cell by {abs(float(worst)):.1f}%", self.readme)

    def test_the_constant_price_claim_is_what_the_script_prints(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fb.main(["--check-alternative"])
        residuals = [abs(float(r)) for r in re.findall(r"\(([+-][\d.]+)%\)", buf.getvalue().split("Alternative", 1)[1])]
        self.assertEqual(len(residuals), 6)
        self.assertEqual(sum(r <= 6.0 for r in residuals), 5)
        self.assertIn("five of the six cells within 6%", self.readme)

    def test_the_largest_measured_size_is_the_one_the_readme_names(self) -> None:
        self.assertEqual(max(fb.SINGLE_POINTS + fb.CHAIN_POINTS), 72)
        self.assertIn("The largest N measured is 72", self.readme)


class MeasurementsTest(unittest.TestCase):
    def test_ledgers_carry_no_real_session_id(self) -> None:
        pattern = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
        for root, _, files in os.walk(os.path.join(REPO, "measurements")):
            for name in files:
                path = os.path.relpath(os.path.join(root, name), REPO)
                with self.subTest(file=path):
                    self.assertIsNone(pattern.search(_read(path)))
                    if name == "ledger.jsonl":
                        for line in _read(path).splitlines():
                            self.assertRegex(json.loads(line)["session_id"], r"^sha256:[0-9a-f]{16}$")

    def test_each_chain_ledger_sums_to_its_summary(self) -> None:
        for arm, n in CELLS:
            rows = [json.loads(x) for x in _read(os.path.join("measurements", f"read_{arm}_{n}", "ledger.jsonl")).splitlines()]
            summary = _summary(arm, n)
            with self.subTest(cell=f"{arm}_{n}"):
                self.assertEqual(len(rows), summary["iterations"])
                self.assertAlmostEqual(sum(r["cost_usd"] for r in rows), summary["cost_usd"], places=3)
                self.assertEqual(sum(r["context_tokens"] for r in rows), summary["context_tokens"])


class DocsTest(unittest.TestCase):
    def test_every_cli_flag_the_readme_uses_exists(self) -> None:
        readme = _read("README.md")
        for script in ("ab_compare.py", "session_loop.py", "fit_break_even.py"):
            commands = re.findall(rf"python3 {re.escape(script)}(?:[^\n]*\\\n)*[^\n]*", readme)
            used = set(re.findall(r"(--[a-z][a-z-]+)", " ".join(commands)))
            self.assertTrue(used, f"no {script} flags parsed from the README; the parser is broken")
            source = _read(script)
            for flag in used:
                with self.subTest(script=script, flag=flag):
                    self.assertIn(f'"{flag}"', source)

    def test_the_refused_flags_the_readme_lists_are_the_ones_refused(self) -> None:
        readme = " ".join(_read("README.md").split())
        for flag in ("--resume", "--continue", "--fork-session", "--dangerously-skip-permissions", "--settings"):
            with self.subTest(flag=flag):
                self.assertIn(flag, sl._INVARIANT_BREAKING_FLAGS)
                self.assertIn(f"`{flag}`", readme)
        for mode in sl._BLIND_PERMISSION_MODES:
            with self.subTest(mode=mode):
                self.assertIn(f"`{mode}`", readme)
        self.assertIn("`dontAsk` is allowed", readme)

    def test_every_ledger_field_the_readme_names_is_written(self) -> None:
        readme = _read("README.md")
        source = _read("session_loop.py")
        for field in ("input_tokens", "cache_creation_tokens", "cache_read_tokens"):
            with self.subTest(field=field):
                self.assertIn(f"`{field}`", readme)
                self.assertIn(f'"{field}": result.{field}', source)

    def test_every_make_target_the_docs_name_exists(self) -> None:
        targets = set(re.findall(r"^([\w-]+):", _read("Makefile"), re.M))
        for doc in ("README.md", "AGENTS.md"):
            for target in re.findall(r"^make ([\w-]+)", _read(doc), re.M):
                with self.subTest(doc=doc, target=target):
                    self.assertIn(target, targets)


class PackagingTest(unittest.TestCase):
    def test_the_scripts_are_executable_with_a_python3_shebang(self) -> None:
        for name in ("session_loop.py", "ab_compare.py", "fit_break_even.py"):
            with self.subTest(name=name):
                self.assertTrue(os.stat(os.path.join(REPO, name)).st_mode & stat.S_IXUSR, f"{name} is not executable")
                self.assertTrue(_read(name).startswith("#!/usr/bin/env python3\n"))

    def test_no_source_file_carries_a_personal_path_or_an_email_address(self) -> None:
        """Walks the tree rather than asking git, so it also runs inside a mutation sandbox copy."""
        pattern = re.compile(r"/Users/\w|/home/\w|C:\\\\Users|[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")
        names = []
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "venv"}]
            names += [os.path.relpath(os.path.join(root, f), REPO) for f in files
                      if f.endswith((".py", ".md", ".json", ".jsonl", ".toml", ".yml")) or f == "Makefile"]
        self.assertIn("session_loop.py", names)
        for name in names:
            if name == os.path.join("tests", "test_repo.py"):
                continue
            with self.subTest(file=name):
                self.assertIsNone(pattern.search(_read(name)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
