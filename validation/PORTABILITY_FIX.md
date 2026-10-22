# CPU-statistics portability correction — scientific-v2.1 source update

Author: Mehran Keivanimehr. Date: 2026-10-15.

## Defect and responsibility

The Titan/1080 pre-pilot report identified a genuine repository compatibility defect: a derived CSV was required to regenerate byte-for-byte, but its CV calculation used version-dependent standard-library floating-point operations. Python 3.10 is supported, so this was a validation-design bug, not evidence of corrupted raw results or a broken cluster interpreter. [Reproduction (2026/10), [regression test](../tests/test_scientific_metrics.py), [compatibility evidence](compatibility_proof.json)]

CPython 3.10 rounds the sample variance to a float before taking its square root. CPython 3.12 takes a square root of the exact rational variance. Their rounding paths differ. [CPython (3.10), [stdev source](https://raw.githubusercontent.com/python/cpython/3.10/Lib/statistics.py); CPython (3.12.14), [stdev source](https://raw.githubusercontent.com/python/cpython/v3.12.14/Lib/statistics.py)]

For the reported values `4067.8526683780683`, `4071.2049695865194`, `4071.239699282067`, the old paths produce `0.04780117256388044` and `0.047801172563880445`, one adjacent binary64 step apart. The exact CV of those stored binary64 inputs starts `0.047801172563880444760479557799668149...`; the higher value is nearest for this case. This is not a universal correct-rounding guarantee for the old multi-operation Python 3.12 expression. [Exact proof (2026/10), [reported-case and all-group regression checks](../tests/test_scientific_metrics.py)]

## Fix and proof

`dispersion()` now computes the mean, median and sample variance as exact rationals. CV is the square root of `10000 * variance / mean^2`, rounded once to the nearest IEEE-754 binary64 value, ties to even, using exact integer comparisons with adjacent-float midpoints. It calls neither `statistics.stdev()` nor floating-point square root. Its input semantics are stored JSON values parsed as binary64, not unknown acquisition-time real quantities. [Implementation (2026/10), [metric definitions](../tools/scientific_metrics.py)]

Tests cover the actual 1-ULP case, independence from `statistics.stdev`, rounding midpoints, subnormals, identical inputs, extreme scales and invalid inputs. All 28 recorded groups are checked using exact rounding bounds and an independent 100-digit Decimal calculation. Cross-interpreter checks compare SHA-256 hashes of all five correction artifacts. [Proof (2026/10), [tests](../tests/test_scientific_metrics.py), [driver](../tools/check_portability.py), [runtime receipt](compatibility_proof.json)]

This deliberately updates a **derived correction table**, not an immutable acquisition record. Some last-bit changes are expected after removing intermediate rounding. The original source ZIP is preserved separately; all original measurements retain the unchanged 1,475-file manifest. No manifest check or byte-comparison tolerance was weakened. [Integrity (2026/10), [manifest](../provenance/measurements.sha256.json), [local verification](local_verification.json)]

## Runtime boundary and archive hygiene

The compatibility receipt lists the actual locally exercised versions/platforms. It does not claim Linux Python 3.10.22 or remote CI already passed. The operator must rerun its six CPU checks before GPU work. No Torch, vLLM or benchmark-environment upgrade is needed. [Boundary (2026/10), [receipt](compatibility_proof.json), [continuation prompt](TITAN_1080_CONTINUE.md)]

The original `e0de161c9346276ac33ad926ef4190aa01c1cc90c2179ec4080ec06ae4283df4` archive has no `__pycache__` paths or `.pyc` files. Imports during tests can create them after extraction. Request an exact path listing if the operator believes they were shipped. Cache cleanup is not a replacement for imported-module locations and source hashes. [Archive proof (2026/10), [original-source reproduction receipt](original_portability_reproduction.json)]

A clearly labeled pre-pilot archive may contain `PRE_PILOT_STATUS.md`, null plan observations and no GPU `runs/`. It must not be represented as a completed pilot. [Handoff clarification (2026/10), [continuation instructions](TITAN_1080_CONTINUE.md)]

## Sources

[CPython, 3.10] Python Software Foundation. Statistics module, versioned source linked above.

[CPython, 3.12.14] Python Software Foundation. Statistics module, versioned source linked above.

[Reproduction and proof, 2026/10] Mehran Keivanimehr. Exact-arithmetic implementation, tests and compatibility receipts linked above.
