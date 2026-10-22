"""Frozen statistics for the Titan campaign. Pure, no numpy, testable.

Primary estimand: median throughput per measurement arm across the blocks.

Interval: for n IID continuous observations with a unique population median, the
probability that [min, max] brackets the median is 1 - 2*(1/2)^n. At n = 6 that is
1 - 2/64 = 31/32 = 96.875%. This is a POINTWISE coverage statement for one arm --
not a simultaneous band across arms, and not a significance ranking.
"""
from __future__ import annotations
from fractions import Fraction


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        raise ValueError("median of empty sample")
    mid = n // 2
    return v[mid] if n % 2 else (v[mid - 1] + v[mid]) / 2.0


def sample_cv_pct(values):
    """Sample CV in percent, exact-rational then rounded once (no float sqrt)."""
    v = [Fraction(str(x)) for x in values]
    n = len(v)
    if n < 2:
        return None
    mean = sum(v) / n
    if mean == 0:
        raise ValueError("CV undefined for zero mean")
    var = sum((x - mean) ** 2 for x in v) / (n - 1)
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    sd = (Decimal(var.numerator) / Decimal(var.denominator)).sqrt()
    return float(100 * sd / (Decimal(mean.numerator) / Decimal(mean.denominator)))


def range_over_median_pct(values):
    m = median(values)
    if m == 0:
        raise ValueError("range/median undefined for zero median")
    return 100.0 * (max(values) - min(values)) / m


def minmax_median_coverage(n: int) -> Fraction:
    """Exact pointwise coverage of [min,max] for the median, n IID continuous."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive int")
    return Fraction(1) - 2 * Fraction(1, 2) ** n


def interval_within_tolerance(values, tol_pct=5.0) -> bool:
    """Are both [min,max] endpoints within tol_pct of the median?"""
    m = median(values)
    if m == 0:
        return False
    return (abs(max(values) - m) / m * 100.0 <= tol_pct
            and abs(m - min(values)) / m * 100.0 <= tol_pct)


def summarize(values, *, blocks_expected=6, tol_pct=5.0) -> dict:
    v = list(values)
    n = len(v)
    cov = minmax_median_coverage(n)
    complete = n == blocks_expected
    return {
        "n": n, "points": v, "median": median(v) if v else None,
        "min": min(v) if v else None, "max": max(v) if v else None,
        "sample_cv_pct": sample_cv_pct(v) if n >= 2 else None,
        "range_over_median_pct": range_over_median_pct(v) if v else None,
        "interval": [min(v), max(v)] if v else None,
        "minmax_median_coverage": f"{cov.numerator}/{cov.denominator}",
        "minmax_median_coverage_pct": float(cov) * 100.0,
        "blocks_expected": blocks_expected, "blocks_complete": complete,
        "coverage_claimable": complete,
        "coverage_note": ("pointwise median coverage for THIS arm only; not a "
                          "simultaneous band and not a significance ranking"),
        "precision_goal_met": (interval_within_tolerance(v, tol_pct)
                               if complete else None),
        "qualification": (None if complete else
                          f"n={n} < {blocks_expected}: no six-block confidence "
                          f"claim; report descriptively"),
    }


def matched_block_ratio(numer_by_block: dict, denom_by_block: dict) -> dict:
    """Per-block ratios over blocks present in BOTH arms, then the same median."""
    common = sorted(set(numer_by_block) & set(denom_by_block))
    if not common:
        return {"n": 0, "ratios": [], "median_ratio": None,
                "note": "no shared blocks; comparison unestablished"}
    ratios = []
    for b in common:
        d = denom_by_block[b]
        if d == 0:
            raise ValueError(f"zero denominator in block {b}")
        ratios.append(numer_by_block[b] / d)
    s = summarize(ratios)
    s["blocks_used"] = common
    s["ratios"] = ratios
    s["median_ratio"] = s["median"]
    return s
