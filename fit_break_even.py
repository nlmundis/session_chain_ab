#!/usr/bin/env python3
"""Fit both arms' cost curves from committed measurements and report the crossing.

WHAT THIS PRODUCES, AND WHY IT IS A BAND
    ``ab_compare.py`` measures one task size at a time. This turns those points
    into the number people actually want: the task size above which chaining
    fresh sessions becomes cheaper than one long session.

    It reports a RANGE, not a point, and that is the correction this file exists
    to carry. The first version published "BREAK-EVEN: N = 95 files" from a
    single fit in which the single arm was given a free intercept while the chain
    was forced through the origin -- an asymmetry nothing justified. Fitting both
    arms the same way moves the answer to 94.2 or 107.3 depending on which
    symmetric choice you make, so the modelling choice alone spans ~14%, and
    run-to-run noise, which no committed cell measures, comes on top. A point
    estimate implied a precision the data does not have.

    So all three fits are reported, each with its residuals, and the headline is
    their span. The fits are not equally good: symmetric B, which sets the upper
    edge, misses the N = 12 cell by about 20%.

WHAT IT CANNOT TELL YOU
    Every (arm, N) cell is a SINGLE run. There is no replication, so none of the
    scatter here is measured -- it is inferred from residuals. The crossing also
    sits beyond the largest measured task size, which makes it an extrapolation
    from a quadratic that was fitted, not derived. Treat the band as the best
    available reading of six cells, not as a measured threshold.

RETRACTED CLAIM, RECORDED HERE BECAUSE IT WAS PUBLISHED
    An earlier write-up of these measurements (a commit message in the private
    repository this was extracted from) asserted that prompt caching was the
    mechanism behind the single arm's advantage, citing a falling
    cost-per-million-context-tokens series. That claim is NOT supported by the committed data and is
    withdrawn. ``session_loop._context_tokens`` sums input, cache-creation and
    cache-read tokens into one integer before anything reaches the ledger, so no
    committed file distinguishes a cache read from a cache creation. Worse, a
    model with one CONSTANT price per context token plus a per-turn cost -- no
    caching term at all -- reproduces five of the six measured points within 6%.
    ``--check-alternative`` runs that comparison. Caching remains a plausible
    explanation; it is not a demonstrated one, and showing it would require
    recording the cache split, which ``session_loop`` now writes to every ledger
    row for runs made after these cells.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from collections.abc import Sequence

# Task sizes with a committed measurement, per arm. Kept explicit rather than
# globbed so that a directory appearing or disappearing changes the fit only when
# someone edits this line, never silently between two runs of the script.
SINGLE_POINTS = (12, 24, 48, 72)
CHAIN_POINTS = (24, 48)


@dataclasses.dataclass(frozen=True)
class Point:
    """One measured (arm, N) cell, with the fields the fit depends on."""

    arm: str
    n: int
    cost_usd: float
    context_tokens: int
    turns: int


@dataclasses.dataclass(frozen=True)
class Fit:
    """A named cost model for one arm, with the crossing it implies.

    Named because the whole point of this module is that several defensible fits
    disagree, so a coefficient set is meaningless without the choice that produced it.
    """

    name: str
    single: tuple[float, float, float]
    """Quadratic coefficients (a, b, c) for cost = a + b*N + c*N^2."""
    chain: tuple[float, float]
    """Line coefficients (intercept, slope) for the chain arm."""
    verdict: Verdict
    """What the two fitted curves say: a threshold, or which arm always wins."""

    @property
    def crossing(self) -> float | None:
        """The threshold when there is one, else None."""
        return self.verdict.n


def load_point(root: pathlib.Path, arm: str, n: int) -> Point:
    """Read one committed measurement and check it is the one being asked for.

    The directory name is not evidence. An earlier version trusted it for both
    the arm and the size, then divided every per-file figure by an N taken from
    the filename -- so a mislabelled or two-arm directory would have produced a
    confidently wrong break-even with nothing to flag it.

    Args:
        root: Directory holding the ``read_<arm>_<n>`` measurement folders.
        arm: Arm label the summary must declare.
        n: Task size the summary must report.

    Returns:
        The measurement.

    Raises:
        ValueError: If the file is missing, holds no matching arm, disagrees
            about the task size, or records an incomplete grading.
    """
    path = root / f"read_{arm}_{n}" / "summary.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: cannot read ({exc})") from exc
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list of arm summaries")

    matching = [e for e in entries if isinstance(e, dict) and e.get("arm") == arm]
    if len(matching) != 1:
        raise ValueError(f"{path}: expected exactly one '{arm}' entry, found {len(matching)}")
    s = matching[0]

    if s.get("total_files") != n:
        raise ValueError(f"{path}: total_files={s.get('total_files')} but directory says {n}")
    if s.get("correct") != s.get("total_files"):
        raise ValueError(
            f"{path}: graded {s.get('correct')}/{s.get('total_files')}; an arm that did not "
            "score 100% varies cost and correctness at once, which the fit cannot separate"
        )
    if s.get("cost_is_complete") is False:
        raise ValueError(f"{path}: cost is a lower bound (an iteration reported none)")
    return Point(
        arm=arm,
        n=n,
        cost_usd=float(s["cost_usd"]),
        context_tokens=int(s["context_tokens"]),
        turns=int(s["turns"]),
    )


def solve(matrix: list[list[float]]) -> list[float]:
    """Solve a small dense linear system by Gauss-Jordan with partial pivoting.

    Written out rather than pulling in numpy, which this repo does not otherwise
    depend on, for systems that are never larger than 3x3.

    Args:
        matrix: Rows of an augmented coefficient matrix, each of length n+1.

    Returns:
        The solution vector.

    Raises:
        ValueError: If the system is singular.
    """
    size = len(matrix)
    rows = [list(r) for r in matrix]
    for i in range(size):
        pivot = max(range(i, size), key=lambda r: abs(rows[r][i]))
        if abs(rows[pivot][i]) < 1e-15:
            raise ValueError("singular system: the points do not determine a fit")
        rows[i], rows[pivot] = rows[pivot], rows[i]
        for r in range(size):
            if r != i:
                factor = rows[r][i] / rows[i][i]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[i])]
    return [rows[i][size] / rows[i][i] for i in range(size)]


def fit_quadratic(points: Sequence[Point], *, through_origin: bool = False) -> tuple[float, float, float]:
    """Least-squares quadratic in N over the given points.

    Args:
        points: Measurements for one arm.
        through_origin: Drop the constant term, forcing cost(0) = 0. Use when the
            chain arm is also pinned to the origin, so the two arms are treated
            symmetrically.

    Returns:
        Coefficients (a, b, c) for ``cost = a + b*N + c*N**2``; ``a`` is 0.0 when
        ``through_origin`` is set.
    """
    ns = [p.n for p in points]
    ys = [p.cost_usd for p in points]
    powers = (1, 2) if through_origin else (0, 1, 2)
    system = [
        [sum(n ** (i + j) for n in ns) for j in powers] + [sum(n**i * y for n, y in zip(ns, ys))]
        for i in powers
    ]
    coeffs = solve(system)
    return (0.0, coeffs[0], coeffs[1]) if through_origin else (coeffs[0], coeffs[1], coeffs[2])


def fit_line(points: Sequence[Point], *, through_origin: bool = True) -> tuple[float, float]:
    """Least-squares line over the given points.

    Args:
        points: Measurements for one arm.
        through_origin: Pin the intercept to zero.

    Returns:
        Coefficients (intercept, slope).
    """
    ns = [p.n for p in points]
    ys = [p.cost_usd for p in points]
    if through_origin:
        return 0.0, sum(n * y for n, y in zip(ns, ys)) / sum(n * n for n in ns)
    system = [
        [float(len(ns)), sum(ns), sum(ys)],
        [float(sum(ns)), float(sum(n * n for n in ns)), sum(n * y for n, y in zip(ns, ys))],
    ]
    intercept, slope = solve(system)
    return intercept, slope


@dataclasses.dataclass(frozen=True)
class Verdict:
    """What two fitted cost curves say about which arm to use, and where.

    A bare ``float | None`` could not express this, and that is precisely how the
    first version got two cases backwards. "No crossing" conflates two OPPOSITE
    situations -- the chain is cheaper at every size, or it is cheaper at none --
    and the caller printed the second sentence for both.
    """

    kind: str
    """One of: ``crossing`` (chain wins above ``n``), ``chain_always``,
    ``single_always``, or ``undetermined``."""
    n: float | None
    """The crossing, when ``kind`` is ``crossing``; otherwise None."""

    @property
    def describes_a_threshold(self) -> bool:
        """Whether there is a task size above which chaining wins."""
        return self.kind == "crossing"


def _cheaper_arm_at(single: tuple[float, float, float], chain: tuple[float, float], n: float) -> str:
    """Say which arm the fits make cheaper at one task size.

    Used to resolve the no-crossing cases by evaluating the curves rather than by
    reasoning about coefficient signs, which is where the earlier version erred.
    """
    a, b, c = single
    ci, cs = chain
    diff = (a + b * n + c * n * n) - (ci + cs * n)
    return "chain" if diff > 0 else "single"


def crossing(single: tuple[float, float, float], chain: tuple[float, float]) -> Verdict:
    """Decide whether, and where, the chain arm overtakes the single arm.

    Every degenerate case is resolved by EVALUATING the fitted curves at a large
    task size, never by inferring a direction from the coefficients. Two bugs
    came out of the coefficient reasoning, both of which reported the opposite of
    the truth rather than failing visibly:

    * An exactly-linear single arm returned ``-qc/qb`` for any positive root
      without checking the sign of ``qb``. With the module's own published
      coefficients and the quadratic term zeroed, it answered 6.28 while the
      single arm is in fact cheaper above that point, not below.
    * A negative discriminant returned None, and the caller printed "the chain
      never becomes cheaper" -- when a negative discriminant with a positive
      quadratic term means the chain is cheaper at EVERY size.

    Args:
        single: Quadratic coefficients (a, b, c) for the single arm.
        chain: Line coefficients (intercept, slope) for the chain arm.

    Returns:
        A Verdict naming which arm wins where.
    """
    a, b, c = single
    chain_intercept, chain_slope = chain
    qa, qb, qc = c, b - chain_slope, a - chain_intercept

    # A size far past any plausible task, used only to read off the asymptotic
    # winner when the curves do not cross in the positive range.
    far = 10_000.0
    asymptotic = _cheaper_arm_at(single, chain, far)

    roots: list[float] = []
    if abs(qa) < 1e-15:
        if abs(qb) >= 1e-15:
            roots = [-qc / qb]
    else:
        disc = qb * qb - 4 * qa * qc
        if disc >= 0:
            roots = [(-qb - disc**0.5) / (2 * qa), (-qb + disc**0.5) / (2 * qa)]

    # A root only marks a threshold if the chain is the cheaper arm ABOVE it.
    thresholds = [
        r for r in roots if r > 0 and _cheaper_arm_at(single, chain, r + max(1.0, 0.01 * r)) == "chain"
    ]
    if thresholds:
        return Verdict(kind="crossing", n=max(thresholds))
    return Verdict(kind="chain_always" if asymptotic == "chain" else "single_always", n=None)


def build_fits(single: Sequence[Point], chain: Sequence[Point]) -> list[Fit]:
    """Produce the defensible fits whose disagreement defines the reported band.

    Three, deliberately. The published one is kept so the earlier number stays
    reproducible and its asymmetry stays visible; the two symmetric alternatives
    are what turn a point estimate into a range.

    Args:
        single: Single-arm measurements.
        chain: Chain-arm measurements.

    Returns:
        One Fit per modelling choice.
    """
    return [
        _fit(
            "published (single free intercept, chain through origin)",
            fit_quadratic(single),
            fit_line(chain, through_origin=True),
        ),
        _fit("symmetric A (both arms free intercept)", fit_quadratic(single), fit_line(chain, through_origin=False)),
        _fit(
            "symmetric B (both arms through origin)",
            fit_quadratic(single, through_origin=True),
            fit_line(chain, through_origin=True),
        ),
    ]


def _fit(name: str, single: tuple[float, float, float], chain: tuple[float, float]) -> Fit:
    """Pair a named modelling choice with the verdict its coefficients imply."""
    return Fit(name=name, single=single, chain=chain, verdict=crossing(single, chain))


def constant_price_residuals(points: Sequence[Point]) -> list[tuple[Point, float, float]]:
    """Fit one shared price per context token across BOTH arms, and report the misses.

    This is the alternative to the retracted caching claim. If a single constant
    price per context token, plus a per-turn cost, reproduces both arms, then the
    falling cost-per-million-tokens series is not evidence of a changing cache
    ratio -- it falls automatically because output tokens sit in the numerator
    and not the denominator.

    Args:
        points: Every measurement, both arms together.

    Returns:
        One (point, fitted cost, percent error) triple per measurement.
    """
    xs = [(p.context_tokens, p.turns) for p in points]
    ys = [p.cost_usd for p in points]
    s11 = sum(x * x for x, _ in xs)
    s12 = sum(x * t for x, t in xs)
    s22 = sum(t * t for _, t in xs)
    b1 = sum(x * y for (x, _), y in zip(xs, ys))
    b2 = sum(t * y for (_, t), y in zip(xs, ys))
    price, per_turn = solve([[s11, s12, b1], [s12, s22, b2]])
    out = []
    for p, (x, t) in zip(points, xs):
        fitted = price * x + per_turn * t
        out.append((p, fitted, 100 * (fitted - p.cost_usd) / p.cost_usd))
    return out


def _build_parser() -> argparse.ArgumentParser:
    """Assemble the CLI parser, kept separate so tests can exercise it directly."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "measurements",
        nargs="?",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "measurements",
        help="Directory holding the read_<arm>_<n>/ measurement folders.",
    )
    parser.add_argument(
        "--check-alternative",
        action="store_true",
        help="Also fit a constant price per context token, the alternative to the "
        "retracted prompt-caching explanation.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser


def _describe_verdict(fit: Fit) -> str:
    """One line saying where, if anywhere, chaining becomes cheaper on this fit."""
    if fit.verdict.kind == "crossing":
        return f"chaining wins above N = {fit.crossing:.1f}"
    if fit.verdict.kind == "chain_always":
        return "chaining is cheaper at EVERY size on this fit"
    if fit.verdict.kind == "single_always":
        return "one long session is cheaper at every size on this fit"
    return "undetermined"


def _residual_text(fitted: float, measured: float) -> str:
    """A fitted cost beside the measured one, with the signed error at full precision."""
    return f"fit {fitted:8.4f} vs measured {measured:8.4f}  ({100 * (fitted - measured) / measured:+.4f}%)"


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the curves and print the break-even band.

    Returns:
        0 if at least one fit produced a crossing, 1 otherwise -- so a run whose
        fits never cross cannot be mistaken for one that found a threshold.
    """
    args = _build_parser().parse_args(argv)
    try:
        single = [load_point(args.measurements, "single", n) for n in SINGLE_POINTS]
        chain = [load_point(args.measurements, "chain", n) for n in CHAIN_POINTS]
    except ValueError as exc:
        print(f"cannot fit: {exc}", file=sys.stderr)
        return 1

    fits = build_fits(single, chain)
    crossings = [f.crossing for f in fits if f.crossing is not None]
    largest_measured = max(p.n for p in single + chain)

    if args.json:
        print(
            json.dumps(
                {
                    "points": [dataclasses.asdict(p) for p in single + chain],
                    "fits": [
                        {
                            "name": f.name,
                            "single": f.single,
                            "chain": f.chain,
                            "crossing": f.crossing,
                            "verdict": f.verdict.kind,
                        }
                        for f in fits
                    ],
                    "band": [min(crossings), max(crossings)] if crossings else None,
                    "largest_measured_n": largest_measured,
                    "verdicts": [f.verdict.kind for f in fits],
                },
                indent=1,
            )
        )
        return 0 if crossings else 1

    print("arm      N   turns        ctx      cost   $/file")
    for p in single + chain:
        print(
            f"{p.arm:6} {p.n:3d} {p.turns:7d} {p.context_tokens:10d} "
            f"{p.cost_usd:9.4f} {p.cost_usd / p.n:8.4f}"
        )

    print()
    for f in fits:
        a, b, c = f.single
        ci, cs = f.chain
        where = _describe_verdict(f)
        print(f"{f.name}\n  single(N) = {a:.4f} + {b:.6f}N + {c:.8f}N^2")
        print(f"  chain(N)  = {ci:.4f} + {cs:.6f}N")
        print(f"  verdict: {where}")

        # Residuals for EVERY fit, not only the first: the band's edges come from
    # different fits, and one of them fits the data visibly worse.
    for f in fits:
        print(f"\nresiduals of the {f.name.split(' (')[0]} fit (full precision, not rounded for display):")
        a, b, c = f.single
        for p in single:
            fitted = a + b * p.n + c * p.n * p.n
            print(f"  single N={p.n:3d}: {_residual_text(fitted, p.cost_usd)}")
        ci, cs = f.chain
        for p in chain:
            fitted = ci + cs * p.n
            print(f"  chain  N={p.n:3d}: {_residual_text(fitted, p.cost_usd)}")

    if crossings:
        print(
            f"\nBREAK-EVEN BAND: N = {min(crossings):.0f} to {max(crossings):.0f} files, "
            "from the modelling choice alone. Run-to-run noise is additional and "
            "unmeasured (one run per cell). Quote the band, never a single number."
        )
        print(
            f"  EXTRAPOLATION: the largest task size actually measured is N = {largest_measured}. "
            "The band lies beyond it, so it is a projection of the fits, not an observation."
        )
    elif all(f.verdict.kind == "chain_always" for f in fits):
        print("\nNo crossing: on these fits the CHAIN is cheaper at every task size.")
    elif all(f.verdict.kind == "single_always" for f in fits):
        print("\nNo crossing: on these fits one long session is cheaper at every task size.")
    else:
        print("\nThe fits disagree about which arm wins and none produces a crossing; see above.")

    if args.check_alternative:
        print("\nAlternative to the retracted caching claim -- ONE constant price per")
        print("context token plus a per-turn cost, fitted across both arms:")
        worst = 0.0
        for p, fitted, err in constant_price_residuals(single + chain):
            worst = max(worst, abs(err))
            print(f"  {p.arm:6} N={p.n:3d}: fit ${fitted:7.4f} vs ${p.cost_usd:7.4f}  ({err:+.1f}%)")
        print(
            f"  worst residual {worst:.1f}%. A constant-price model with NO caching term "
            "reproduces most points, which is why the caching explanation is not claimed."
        )

    return 0 if crossings else 1


if __name__ == "__main__":
    sys.exit(main())
