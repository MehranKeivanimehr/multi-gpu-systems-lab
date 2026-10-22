"""
Generate reports/SUMMARY.md from the aggregated CSVs.

Everything here is derived from results/*.csv, so the summary cannot drift from
the measurements. Run `aggregate.py` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from scientific_metrics import cnn_compute

PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results"
OUT = PROJ / "reports" / "SUMMARY.md"

SHORT = {
    "NVIDIA TITAN RTX": "TITAN RTX",
    "NVIDIA GeForce GTX 1080 Ti": "GTX 1080 Ti",
    "NVIDIA TITAN Xp": "TITAN Xp",
}


def md(df: pd.DataFrame) -> str:
    # Keep analysis portable without an optional Markdown-table dependency.
    def cell(value):
        if pd.isna(value):
            return "unavailable"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |"
                 for row in df.itertuples(index=False, name=None))
    return "\n".join(lines)


def read(name: str) -> pd.DataFrame:
    p = RES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def section_dtype(parts: list[str]) -> None:
    d = read("dtype_gemm")
    if d.empty:
        return
    d = d.copy()
    d["gpu"] = d["gpu"].map(lambda x: SHORT.get(x, x))
    t = d[
        [
            "gpu", "capability", "fp32_tflops", "fp16_tflops", "bf16_tflops",
            "fp16_speedup", "bf16_speedup", "is_bf16_supported_flag",
        ]
    ]
    parts += [
        "## 1. Dtype capability (4096^3 GEMM)",
        "",
        "The historical probe covers three representative devices. The default "
        "BF16 flag includes emulation; it is not a native acceleration promise.",
        "",
        md(t),
        "",
        "The recorded GEMM behavior is workload-specific; it is not a universal precision recommendation.",
        "",
    ]


def section_single(parts: list[str]) -> None:
    d = read("llm_runs")
    if d.empty:
        return
    s = d[(d["mode"] == "single") & (d["status"] == "ok")].copy()
    if s.empty:
        return
    best = s.loc[s.groupby(["group", "precision"])["tokens_per_s"].idxmax()]
    best = best[
        [
            "group", "precision", "micro_batch_per_gpu", "tokens_per_s",
            "peak_alloc_mib", "util_mean_pct", "power_mean_w_per_gpu",
        ]
    ].sort_values("tokens_per_s", ascending=False)
    parts += [
        "## 2. Single-GPU training (GPT-2 124M, seq 512)",
        "",
        "Best configuration per GPU and precision:",
        "",
        md(best),
        "",
        "Full batch sweep in `results/llm_runs.csv`.",
        "",
    ]


def section_scaling(parts: list[str]) -> None:
    d = read("scaling_efficiency")
    if d.empty:
        return
    s = d[d["mode"].isin(["single", "ddp"])].copy()
    s = s[(s["micro_batch_per_gpu"] == 4) & (s["precision"] == "fp32")]
    if not s.empty:
        t = s[
            [
                "family", "group", "n_gpus", "tokens_per_s",
                "tokens_per_s_per_gpu", "speedup_vs_1gpu",
                "scaling_efficiency_pct",
            ]
        ].sort_values(["family", "n_gpus"])
        parts += [
            "## 3. DDP weak scaling (bs=4/GPU, FP32)",
            "",
            md(t),
            "",
        ]

    comm = RES / "comm_bandwidth.csv"
    if comm.exists():
        c = pd.read_csv(comm)
        parts += [
            "### Interconnect (NCCL all-reduce at a 497 MiB buffer)",
            "",
            md(c),
            "",
            "The historical size_mb field is MiB: this buffer is 497 MiB, "
            "about 4.7% larger than the recorded GPT-2 parameter-sized FP32 "
            "gradient volume (about 474.70 MiB). Bandwidth is decimal GB/s.",
            "",
        ]


def section_methods(parts: list[str]) -> None:
    d = read("llm_runs")
    if d.empty:
        return
    m = d[(d["status"] == "ok") & (d["micro_batch_per_gpu"] == 4)].copy()
    m = m[m["world_size"] > 1]
    if m.empty:
        return
    m["method"] = m.apply(
        lambda r: r["mode"]
        if r["mode"] != "deepspeed"
        else f"zero-{int(r['zero_stage'])}",
        axis=1,
    )
    rows = []
    for (grp, prec), g in m.groupby(["group", "precision"]):
        base = g[g["method"] == "ddp"]["tokens_per_s"]
        base = float(base.iloc[0]) if len(base) else None
        for _, r in g.sort_values("tokens_per_s", ascending=False).iterrows():
            rows.append(
                {
                    "group": grp,
                    "precision": prec,
                    "n_gpus": int(r["world_size"]),
                    "method": r["method"],
                    "tokens_per_s": r["tokens_per_s"],
                    "pct_of_ddp": round(r["tokens_per_s"] / base * 100, 1)
                    if base
                    else None,
                    "peak_alloc_mib": r["peak_alloc_mib"],
                }
            )
    t = pd.DataFrame(rows).sort_values(
        ["group", "precision", "tokens_per_s"], ascending=[True, True, False]
    )
    parts += [
        "## 4. Parallelism method comparison (bs=4/GPU)",
        "",
        md(t),
        "",
        "These are single-run complete-recipe observations. Identical initial states, "
        "all optimizer updates and a unique communication cause were not established historically.",
        "",
    ]


def section_cnn(parts: list[str]) -> None:
    d = read("cnn_resnet50")
    if d.empty:
        return
    ok = d[d["status"] == "ok"].copy()
    corrected = [cnn_compute(float(row["throughput"]), row["precision"], float(row["theoretical_fp32_tflops"]))
                 for _, row in ok.iterrows()]
    ok["analytical_tflops_fma2"] = [row["analytical_tflops_fma2"] for row in corrected]
    ok["fp32_peak_fraction_pct"] = [row["fp32_peak_fraction_pct"] for row in corrected]
    best = ok.loc[ok.groupby(["gpu", "precision"])["throughput"].idxmax()]
    best = best.copy()
    best["gpu"] = best["gpu"].map(lambda x: SHORT.get(x, x))
    t = best[
        [
            "gpu", "capability", "precision", "batch_size", "throughput", "analytical_tflops_fma2",
            "theoretical_fp32_tflops", "fp32_peak_fraction_pct",
        ]
    ].sort_values("throughput", ascending=False)
    parts += [
        "## 5. ResNet-50 inference -- campaign-local precision comparison",
        "",
        md(t),
        "",
        "Corrected analytical convention: approximate 4.1 GMAC/image = 8.2 GFLOP/image "
        "with FMA=2. AMP has no FP32-peak percentage. Historical energy values integrate "
        "a partial sample window and are excluded here. These are not hardware occupancy measurements.",
        "",
    ]


def section_infer(parts: list[str]) -> None:
    d = read("inference")
    if d.empty:
        return
    ok = d[d["status"] == "ok"].copy()
    ok["prefill_forward_proxy_ms_not_ttft"] = ok["ttft_ms"]
    ok["gpu"] = ok["gpu"].map(lambda x: SHORT.get(x, x))
    t = ok[
        [
            "engine", "gpu", "dtype", "tensor_parallel", "requests",
            "output_tokens_per_s", "prefill_forward_proxy_ms_not_ttft", "peak_alloc_mib",
        ]
    ]
    parts += [
        "## 6. Inference (TinyLlama-1.1B, 256-token prompt, 128 new tokens)",
        "",
        md(t),
        "",
        "vLLM requires compute capability >= 7.0, so it runs only on the two "
        "TITAN RTX; the 8 Pascal GPUs use HF `generate`. These are different "
        "serving stacks and do not automatically have identical input IDs. HF timing is "
        "a standalone forward proxy; no actual request TTFT is supplied. Energy rankings are withdrawn.",
        "",
    ]


def main() -> int:
    parts = [
        "# Stage-1 Benchmark Results -- Heterogeneous 10-GPU Cluster",
        "",
        "Auto-generated from `results/*.csv` by `bench/make_report.py`. "
        "Narrative, methodology and caveats live in `reports/README.md`.",
        "[Scientific correction record (2026/10), [status](../../../docs/SCIENTIFIC_STATUS.md)]. "
        "New GPU validation is pending; tables below are corrected interpretations of historical records.",
        "",
    ]
    section_dtype(parts)
    section_single(parts)
    section_scaling(parts)
    section_methods(parts)
    section_cnn(parts)
    section_infer(parts)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes("\n".join(parts).encode("utf-8"))
    print(f"wrote {OUT.relative_to(PROJ)} ({len(parts)} blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
