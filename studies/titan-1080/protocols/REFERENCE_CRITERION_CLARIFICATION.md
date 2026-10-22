# Informed protocol clarification — the 64u logit criterion

**Status: an informed clarification, frozen before any fresh logit values were
collected. It is NOT blind preregistration.** The operator already knew that
the two families' final-step logit digests differ; what was unknown, and is
what this clarification makes evaluable, is whether that difference lies inside
the frozen tolerance.

## What the frozen protocol says

`reference.PROTOCOL` states:

- `comparison`: "greedy output token ids compared exactly; final-token logits
  compared with the oracle's dtype-derived tolerance policy"
- `thresholds.logits_rtol`: "64u for the compared dtype (oracle tolerance
  policy)"
- `thresholds.note`: "no tolerance is widened to accommodate a disagreement"

The oracle policy it points at is `update_oracle.default_tolerances`:

```
u    = finfo(dtype).eps / 2
rtol = 64u
atol = 64u * max(1, max|param_before|)
pass iff  max|param_after - expected|  <=  atol + rtol * max(max|expected|, 1e-300)
```

## What is genuinely underspecified

The oracle compares an **observed** tensor against an **expected** one, and its
`atol` and `rtol` scales are drawn from different operands. The reference
control compares two **equally authoritative** arrays from two GPU families;
neither is "expected". Substituting one for the other would make the verdict
depend on an arbitrary choice of which family is the reference, and swapping
the operands could change the result. That is the ambiguity resolved here.

## The frozen clarification

Let `A` and `B` be the two final-step logit arrays, same shape and dtype.

```
u      = finfo(dtype).eps / 2                  # fp32: 5.9604644775390625e-08
M      = max(max|A|, max|B|)                   # symmetric magnitude scale
atol   = 64u * max(1.0, M)
rtol_term = 64u * max(M, 1e-300)
tol    = atol + rtol_term
pass iff  max_over_elements |A - B|  <=  tol
```

Each decision, and why:

- **u from the compared dtype.** The arrays are compared in their stored dtype;
  `u = eps/2` is the oracle's own definition, unchanged.
- **Symmetry.** `M` is the maximum over *both* arrays, so exchanging `A` and
  `B` gives an identical `tol` and an identical verdict. This is the only
  change of substance from the oracle's one-sided form, and it is forced by the
  control having no privileged side.
- **Zero handling.** `max(1.0, M)` floors the absolute term, so an all-zero or
  tiny-magnitude pair is still compared against `64u`, never against zero. The
  `1e-300` floor on the relative term is carried over from the oracle verbatim.
- **Aggregation: maximum, not mean.** The oracle aggregates by maximum absolute
  deviation, and a mean would let a large isolated disagreement hide. The
  comparator additionally reports the failing-element count and the worst
  witness, but the **verdict** is the maximum.
- **Elementwise, no broadcasting.** Shapes and dtypes must match exactly; a
  mismatch is a refusal, not a comparison.
- **Non-finite values.** Any non-finite element in either array is a refusal,
  not a failure: `max|A-B|` is not meaningful there.
- **No `allclose` defaults.** `numpy.allclose`'s defaults (rtol 1e-5, atol
  1e-8) are not used anywhere; they are far looser than 64u at fp32 and would
  not be this protocol.

The threshold is not widened for any observed result. A maximum deviation above
`tol` is a **valid negative outcome** that closes the comparison, not a reason
to adjust the criterion.
