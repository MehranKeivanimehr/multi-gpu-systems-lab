"""Explicit, CPU-testable metric definitions for the corrected protocol.

These functions establish arithmetic and boundary behavior, not sensor accuracy.
"""
from __future__ import annotations

import math
import struct
import sys
from bisect import bisect_left
from fractions import Fraction

PROTOCOL_VERSION = "scientific-v2"


def resnet50_conv_linear_macs() -> int:
    """TorchVision ResNet-50 v1.5, 224x224, 1000 classes; conv/linear only.

    Excludes batch normalization, activations, pooling and residual additions.
    Stride is in the bottleneck's 3x3 convolution, not its first 1x1.
    """
    total = 112 * 112 * 64 * 3 * 7 * 7
    spatial, channels = 56, 64
    for stage, (planes, blocks) in enumerate(zip((64, 128, 256, 512), (3, 4, 6, 3))):
        for block in range(blocks):
            stride = 2 if stage and block == 0 else 1
            output = spatial // stride
            total += spatial**2 * channels * planes
            total += output**2 * planes**2 * 9
            total += output**2 * planes * (4 * planes)
            if block == 0:
                total += output**2 * channels * (4 * planes)
            spatial, channels = output, 4 * planes
    return total + 2048 * 1000


def cnn_compute(images_s: float, precision: str, fp32_peak: float | None) -> dict:
    """Approximate 4.1 GMAC/image convention, FMA = two operations.

    This is analytical model-work throughput, not measured hardware instruction
    throughput. Mixed FP16/FP32 work has no single FP32-peak denominator.
    """
    tflops = float(images_s) * 8.2 / 1000
    return {
        "analytical_tflops_fma2": tflops,
        "fp32_peak_fraction_pct": 100 * tflops / fp32_peak
        if precision == "fp32" and fp32_peak and fp32_peak > 0 else None,
        "operation_definition": "conv+linear approximation; 4.1 GMAC/image; FMA=2",
    }


def integrate_power(samples: list[tuple[float, float | None]], start: float, stop: float) -> dict:
    """Linear-interpolated trapezoids over exactly [start, stop].

    Requires valid samples bracketing BOTH boundaries. Missing readings inside
    the integration support invalidate the result; no zero filling, clamping,
    extrapolation, or silently integrating a shorter window. Duplicate sample
    times are rejected. Sensor averaging is not undone by interpolation.
    """
    result = {"status": "invalid_window", "energy_j": None,
              "duration_s": stop - start, "method": "linear_interpolation_trapezoid",
              "max_support_gap_s": None, "n_samples": len(samples)}
    if not all(math.isfinite(t) for t in (start, stop)) or stop <= start:
        return result
    ordered = sorted(samples, key=lambda point: point[0])
    times = [point[0] for point in ordered]
    if len(times) < 2 or any(not math.isfinite(t) for t in times):
        return {**result, "status": "insufficient_samples"}
    if any(b <= a for a, b in zip(times, times[1:])):
        return {**result, "status": "duplicate_timestamps"}
    if times[0] > start or times[-1] < stop:
        return {**result, "status": "unbracketed_window"}
    lo = max(0, bisect_left(times, start) - 1)
    hi = bisect_left(times, stop)
    support = ordered[lo:hi + 1]
    if any(p is None or not math.isfinite(p) or p < 0 for _, p in support):
        return {**result, "status": "missing_or_invalid_power"}

    def power_at(t):
        i = bisect_left(times, t)
        if times[i] == t:
            return ordered[i][1]
        ta, pa = ordered[i - 1]
        tb, pb = ordered[i]
        return pa + (pb - pa) * (t - ta) / (tb - ta)

    points = [(start, power_at(start))]
    points.extend((t, p) for t, p in support if start < t < stop)
    points.append((stop, power_at(stop)))
    energy = sum((pb + pa) * (tb - ta) / 2
                 for (ta, pa), (tb, pb) in zip(points, points[1:]))
    return {**result, "status": "ok", "energy_j": energy,
            "max_support_gap_s": max(b[0] - a[0] for a, b in zip(support, support[1:])),
            "support_start_s": support[0][0], "support_stop_s": support[-1][0]}


def decode_proxy(output_tokens: int, requests: int, elapsed: float, prefill_proxy: float) -> float | None:
    """Exclude one token per request; separate-forward subtraction remains a proxy."""
    if requests < 1 or output_tokens <= requests or elapsed <= prefill_proxy:
        return None
    return (output_tokens - requests) / (elapsed - prefill_proxy)


def _sqrt_fraction_binary64(value: Fraction) -> float:
    """Round sqrt of an exact nonnegative rational once, ties to even.

    Binary-search IEEE-754 nonnegative finite bit patterns using exact integer
    square comparisons, then compare with the exact midpoint of the adjacent
    candidates. No version-specific statistics or floating square root is used.
    Domain: sqrt(value) <= the largest finite binary64 value.
    """
    if (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.max_exp) != (2, 53, 1024):
        raise RuntimeError("This method requires IEEE-754 binary64 floats")
    if value < 0:
        raise ValueError("nonnegative rational required")
    if value == 0:
        return 0.0

    def from_bits(bits):
        return struct.unpack(">d", struct.pack(">Q", bits))[0]

    low, high = 0, 0x7ff0000000000000  # high is an exclusive +infinity sentinel
    numerator, denominator = value.numerator, value.denominator
    while high - low > 1:
        middle = (low + high) // 2
        n, d = from_bits(middle).as_integer_ratio()
        if n * n * denominator <= numerator * d * d:
            low = middle
        else:
            high = middle
    lower = from_bits(low)
    if high == 0x7ff0000000000000:
        if Fraction(lower) ** 2 == value:
            return lower
        raise OverflowError("rational square root exceeds the method's finite domain")
    upper = from_bits(high)
    midpoint = (Fraction(lower) + Fraction(upper)) / 2
    midpoint_squared = midpoint * midpoint
    if value < midpoint_squared or (value == midpoint_squared and low % 2 == 0):
        return lower
    return upper


def dispersion(values: list[float]) -> dict:
    """Version-independent statistics of the stored binary64 observations.

    Means, medians and variance are exact rationals until final conversion.
    CV is rounded from sqrt(10000 * sample_variance / mean**2), avoiding both
    statistics.stdev version changes and intermediate float rounding/overflow.
    Raw observations are never modified by this derived calculation.
    """
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("positive finite independent-run values required")
    exact = sorted(Fraction(value) for value in values)
    count = len(exact)
    mean = sum(exact, Fraction(0)) / count
    median = exact[count // 2] if count % 2 else (exact[count // 2 - 1] + exact[count // 2]) / 2
    cv = None
    if count > 1:
        variance = sum((value - mean) ** 2 for value in exact) / (count - 1)
        cv = _sqrt_fraction_binary64(10000 * variance / (mean * mean))
    return {"n": count, "median": float(median), "mean": float(mean),
            "min": min(values), "max": max(values),
            "sample_cv_pct": cv,
            "full_range_over_median_pct": float(100 * (exact[-1] - exact[0]) / median)}
