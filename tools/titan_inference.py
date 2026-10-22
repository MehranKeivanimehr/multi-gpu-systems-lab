"""Coverage justification and the two precision criteria, stated correctly.

Correction 1 -- what justifies 31/32
------------------------------------
The previous stage wrote that 31/32 pointwise median coverage follows from
"exchangeability and continuity". That is FALSE. Counterexample: let S be a
single common sign, equiprobable in {-1,+1}, and U_1..U_n iid Uniform(0,2),
independent of S; set X_i = S * U_i. Each marginal is continuous Uniform(-2,2)
with unique median 0, and the joint law is exchangeable -- yet all X_i share the
sign of S, so [min,max] lies entirely on one side of 0 and NEVER covers the
median. Coverage is 0, not 31/32.

The frozen plan (bundle/plan/run_plan.json, frozen_defaults.statistics.interval)
already states the correct condition: "[min,max] over 6 IID continuous
observations = 31/32 pointwise median coverage". IID continuous sampling is
sufficient:

    P(all X_i > m) = P(all X_i < m) = (1/2)^n   for a continuous median m,
    so P(min < m < max) = 1 - 2*(1/2)^n = 31/32 at n = 6.

The weaker sufficient condition that still works is CONDITIONAL independence
given a common median: if the X_i are independent given a latent parameter and
every conditional law is continuous with the SAME median m, the same product
argument applies conditionally and then integrates. The counterexample fails
precisely this: conditioned on S the median is +1 or -1, not a common 0.

Six fresh processes do not by themselves establish IID. Where the assumption is
not justified, report the descriptive range and make no coverage claim.

Correction 2 -- two precision criteria, both published
------------------------------------------------------
The immutable frozen plan's precision_goal is "endpoints within +/-5% of the
median" (ORIGINAL, 77/90). A full-span criterion (max-min)/median <= 5% is a
STRICTER ADDITIONAL analysis (74/90) introduced in correction, not a
replacement. Both are reported with explicit provenance; the original 13 misses
are preserved and the 3 extra misses belong only to the stricter analysis.

CPU-only; no device access.
"""
from __future__ import annotations

import math
from fractions import Fraction

ORIGINAL_CRITERION = "endpoints_within_5pct_of_median"
STRICTER_CRITERION = "full_span_within_5pct_of_median"

CRITERIA = {
    ORIGINAL_CRITERION: {
        "label": "ORIGINAL frozen goal",
        "provenance": ("bundle/plan/run_plan.json -> frozen_defaults.statistics."
                       "precision_goal, verbatim: 'endpoints within +/-5% of the median'"),
        "definition": "max(|max-m|, |m-min|) / m * 100 <= 5",
        "status": "the goal the campaign was run against; authoritative",
    },
    STRICTER_CRITERION: {
        "label": "STRICTER additional analysis",
        "provenance": "introduced during correction; NOT in the frozen plan",
        "definition": "(max-min) / m * 100 <= 5",
        "status": ("an additional, tighter view reported alongside; it does not "
                   "replace the frozen goal and its extra misses are labelled "
                   "as belonging only to this analysis"),
    },
}

COVERAGE_SUFFICIENT_CONDITIONS = (
    "IID continuous observations (what the frozen plan states)",
    "conditionally independent given a latent parameter, with every conditional "
    "law continuous and sharing the SAME median",
)
COVERAGE_INSUFFICIENT_CONDITIONS = (
    "exchangeability plus continuous marginals alone -- see the common-sign "
    "counterexample, under which coverage is 0",
    "'six fresh processes', which does not establish independence",
)


def minmax_median_coverage(n: int) -> Fraction:
    """Pointwise coverage under IID continuous sampling only."""
    if n < 1:
        return Fraction(0)
    return Fraction(1) - 2 * Fraction(1, 2) ** n


def coverage_statement(n: int, *, iid_justified: bool) -> dict:
    cov = minmax_median_coverage(n)
    if not iid_justified:
        return {
            "n": n, "coverage_claimed": None,
            "report": "descriptive range only",
            "reason": ("IID continuous sampling is not justified for this arm, "
                       "and exchangeability alone is insufficient"),
            "sufficient_conditions": list(COVERAGE_SUFFICIENT_CONDITIONS),
            "insufficient_conditions": list(COVERAGE_INSUFFICIENT_CONDITIONS),
        }
    return {
        "n": n,
        "coverage_claimed": f"{cov.numerator}/{cov.denominator}",
        "coverage_pct": float(cov) * 100.0,
        "conditional_on": COVERAGE_SUFFICIENT_CONDITIONS[0],
        "derivation": ("P(all above m) = P(all below m) = (1/2)^n for a "
                       "continuous median, hence 1 - 2(1/2)^n"),
        "not": "not a simultaneous band and not a significance ranking",
    }


def endpoints_within(values, tol_pct: float = 5.0) -> bool | None:
    """ORIGINAL frozen goal: each endpoint within tol_pct of the median."""
    v = [x for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
    if not v:
        return None
    v.sort()
    n = len(v)
    m = (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0)
    if m == 0:
        return False
    return (abs(max(v) - m) / m * 100.0 <= tol_pct
            and abs(m - min(v)) / m * 100.0 <= tol_pct)


def span_within(values, tol_pct: float = 5.0) -> bool | None:
    """STRICTER additional analysis: the whole span within tol_pct of the median."""
    v = [x for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
    if not v:
        return None
    v.sort()
    n = len(v)
    m = (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0)
    if m == 0:
        return False
    return (max(v) - min(v)) / m * 100.0 <= tol_pct


def evaluate_both(values) -> dict:
    return {
        ORIGINAL_CRITERION: {"met": endpoints_within(values),
                             **{k: v for k, v in CRITERIA[ORIGINAL_CRITERION].items()}},
        STRICTER_CRITERION: {"met": span_within(values),
                             **{k: v for k, v in CRITERIA[STRICTER_CRITERION].items()}},
        "note": ("the stricter criterion implies the original; a group failing "
                 "only the stricter one is labelled stricter-analysis-only"),
    }


def paired_ratio_descriptive(num_by_block: dict, den_by_block: dict) -> dict:
    """Paired ratios reported descriptively, with no reliability claim.

    Adding observations to a min/max range can only widen it, so an existing
    crossing of 1.0 cannot be removed by collecting more blocks.
    """
    common = sorted(set(num_by_block) & set(den_by_block))
    rs = [num_by_block[b] / den_by_block[b] for b in common
          if isinstance(num_by_block[b], (int, float))
          and isinstance(den_by_block[b], (int, float)) and den_by_block[b]]
    if not rs:
        return {"n_paired": 0, "ratios": [], "median_ratio": None,
                "crosses_one": None, "claim": "unestablished"}
    rs_sorted = sorted(rs)
    n = len(rs_sorted)
    med = (rs_sorted[n // 2] if n % 2
           else (rs_sorted[n // 2 - 1] + rs_sorted[n // 2]) / 2.0)
    crosses = min(rs) < 1.0 < max(rs)
    return {
        "n_paired": n, "blocks_used": common, "ratios": rs,
        "median_ratio": med, "min_ratio": min(rs), "max_ratio": max(rs),
        "crosses_one": crosses,
        "claim": ("sample median ratio exceeds 1 but the paired range crosses 1: "
                  "no universal or reliable winner is established"
                  if crosses and med > 1.0 else
                  "paired range does not cross 1" if not crosses else
                  "sample median ratio below 1 with a range crossing 1"),
        "monotonicity_note": ("more blocks can only widen [min,max]; an existing "
                              "crossing of 1 cannot be removed by adding blocks"),
    }
