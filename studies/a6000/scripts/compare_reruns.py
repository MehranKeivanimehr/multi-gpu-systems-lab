#!/usr/bin/env python
"""Compare the audit re-runs with the original Stage-1 results (CPU only, reads results/raw).

  Re-run 1  results/raw/llama8b_memory_rerun/       vs  results/raw/train_scaling/llama3.1-8b__{fsdp,zero3}__g4__r*
  Re-run 2  results/raw/vllm_tp_concurrency_rerun/  vs  results/raw/vllm_scaling/ (TP at 256, DP references)

Writes results/rerun/{llama8b_memory_comparison.csv, vllm_tp_concurrency_comparison.csv, tables.md}.
"""
from __future__ import annotations

import json
import re
import statistics as S
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
OUT = ROOT / "results/rerun"
FW = {"fsdp": "FSDP2", "zero3": "ZeRO-3"}
FLUSH_RE = re.compile(r"\] (\d+) pytorch allocator cache flushes since last step")


def load(p: Path):
    with open(p) as f:
        return json.load(f)


def ok_results(exp: str):
    for st in sorted((RAW / exp).glob("*/status.json")):
        if load(st)["status"] == "ok" and (st.parent / "result.json").exists():
            yield st.parent, load(st.parent / "result.json")


# ------------------------------------------------------------------------------------------- Llama-8B memory

def llama_rows() -> pd.DataFrame:
    rows = []
    sources = [("train_scaling", "original (mb2, cap 0.92)")] + [("llama8b_memory_rerun", None)]
    for exp, label in sources:
        for d, r in ok_results(exp):
            s = r["spec"]
            if s["model"] != "llama3.1-8b" or s["n_gpus"] != 4 or s["framework"] not in FW:
                continue
            variant = s.get("variant", "")
            if exp == "train_scaling" and variant:  # skip the bf16-reduce ablation
                continue
            log = (d / "run.log").read_text(errors="replace")
            flushes = [int(x) for x in FLUSH_RE.findall(log)]
            name = label or {"cap95": "rerun cap95 (mb2, cap 0.95)", "mb1": "rerun mb1 (mb1, cap 0.92)",
                             "cap95-expseg": "rerun cap95 + expandable_segments"}.get(variant, variant)
            at = r.get("allocator_train", {})
            am = r.get("allocator_measured", {})
            rows.append({
                "setting": name, "framework": FW[s["framework"]], "repeat": s["repeat"],
                "micro_batch": s["micro_batch"], "mem_fraction": r.get("mem_fraction", 0.92),
                "tokens_s": r["throughput_tokens_s"], "step_ms": r["step_time"]["median_ms"],
                "tokens_per_step": r["samples_per_step"] * s["seq_len"], "mfu": r["mfu"],
                "peak_alloc_gib": r["peak_mem_alloc_gib"], "peak_reserved_gib": r["peak_mem_reserved_gib"],
                "nvml_peak_gib": r["gpu"]["nvml_mem_peak_gib"],
                "sm_clock_mhz": r["gpu"]["sm_clock_mean_mhz"], "power_w": r["gpu"]["power_mean_w_per_gpu"],
                "temp_max_c": r["gpu"]["temp_max_c"],
                "ds_flush_warnings": len(flushes), "ds_flushes": sum(flushes),
                "alloc_retries_train": at.get("num_alloc_retries"), "alloc_retries_measured": am.get("num_alloc_retries"),
                "ooms": at.get("num_ooms"), "cudaMalloc_calls_measured": am.get("num_device_alloc"),
                "cudaFree_calls_measured": am.get("num_device_free"), "run": d.name,
            })
    return pd.DataFrame(rows)


def llama_table(df: pd.DataFrame) -> str:
    base = {fw: g["tokens_s"].median() for fw, g in df[df.setting.str.startswith("original")].groupby("framework")}
    lines = ["| Setting | Framework | Runs | Tokens/step/GPU | Tokens/s (median) | vs original | Step (ms) | MFU | "
             "Peak alloc / reserved (GiB) | Allocator retries (timed window) | DeepSpeed flush warnings |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = ["original", "rerun cap95 (", "rerun cap95 +", "rerun mb1"]
    df = df.assign(_o=df.setting.map(lambda x: next(i for i, o in enumerate(order) if x.startswith(o))))
    for (setting, fw), g in df.sort_values(["_o", "framework"]).groupby(["_o", "framework"], sort=False):
        setting = g.setting.iloc[0]
        tok = g.tokens_s.median()
        comparable = g.micro_batch.iloc[0] == 2
        rel = f"{tok / base[fw]:.3f}x" if comparable and fw in base else "n/c (mb1)"
        retries = ("not recorded" if g.alloc_retries_measured.isna().all()
                   else f"{int(g.alloc_retries_measured.max())}")
        flush = f"{int(g.ds_flush_warnings.max())} ({int(g.ds_flushes.max())} flushes)" if fw == "ZeRO-3" else "n/a"
        lines.append(f"| {setting} | {fw} | {len(g)} | {int(g.tokens_per_step.iloc[0] / 4):,} | {tok:,.0f} | {rel} | "
                     f"{g.step_ms.median():,.0f} | {100 * g.mfu.median():.1f}% | "
                     f"{g.peak_alloc_gib.max():.1f} / {g.peak_reserved_gib.max():.1f} | {retries} | {flush} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- vLLM matched concurrency

def kv_concurrency(run_dir: Path) -> float | None:
    vals = []
    for lg in run_dir.glob("worker_*.log"):
        vals += [float(x) for x in re.findall(r"Maximum concurrency for \d+ tokens per request: ([\d.]+)x",
                                              lg.read_text(errors="replace"))]
    return min(vals) if vals else None


def vllm_rows() -> pd.DataFrame:
    rows = []
    for exp in ("vllm_scaling", "vllm_tp_concurrency_rerun"):
        for d, r in ok_results(exp):
            s = r["spec"]
            conc = s.get("max_num_seqs", 256)
            rows.append({
                "source": "original" if exp == "vllm_scaling" else "rerun", "model": s["model"], "tp": s["tp"],
                "dp": s["dp"], "n_gpus": s["n_gpus"], "max_num_seqs_per_engine": conc,
                "total_concurrency_cap": conc * s["dp"], "repeat": s["repeat"],
                "output_tok_s": r["throughput_output_tokens_s"], "total_tok_s": r["throughput_total_tokens_s"],
                "req_s": r["throughput_requests_s"], "window_s": r["window_s"], "output_tokens": r["output_tokens"],
                "kv_max_concurrency": kv_concurrency(d), "gpu_util_pct": r["gpu"].get("gpu_util_mean_pct"),
                "run": d.name,
            })
    return pd.DataFrame(rows)


def vllm_table(df: pd.DataFrame) -> str:
    med = df.groupby(["model", "source", "tp", "dp"]).agg(
        tok=("output_tok_s", "median"), n=("output_tok_s", "size"), kv=("kv_max_concurrency", "min"),
        cap=("total_concurrency_cap", "first")).reset_index()

    def get(model, source, tp, dp):
        r = med[(med.model == model) & (med.source == source) & (med.tp == tp) & (med.dp == dp)]
        return r.iloc[0] if len(r) else None

    lines = ["| Model | GPUs | TP old (cap 256) | TP new (cap 256 x GPUs) | Change | DP (orig, 256 per GPU) | "
             "DP / TP old | DP / TP new | TP new efficiency vs 1 GPU | KV room (new TP, requests) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in ("qwen2.5-1.5b", "llama3.1-8b"):
        one = get(model, "original", 1, 1)
        for n in (2, 4):
            old, new, dp = get(model, "original", n, 1), get(model, "rerun", n, 1), get(model, "original", 1, n)
            if old is None or new is None or dp is None:
                continue
            eff = new.tok / (n * one.tok) if one is not None else float("nan")
            lines.append(f"| {model} | {n} | {old.tok:,.0f} | {new.tok:,.0f} (cap {int(new.cap)}) | "
                         f"{new.tok / old.tok - 1:+.1%} | {dp.tok:,.0f} | {dp.tok / old.tok:.2f}x | "
                         f"{dp.tok / new.tok:.2f}x | {eff:.0%} | {new.kv:,.0f} |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = ["# Re-run comparison tables (generated by scripts/compare_reruns.py)\n"]
    ll = llama_rows()
    if not ll.empty:
        ll.to_csv(OUT / "llama8b_memory_comparison.csv", index=False)
        parts += ["## Re-run 1: Llama-3.1-8B training, 4 GPUs, memory headroom\n", llama_table(ll), ""]
    vl = vllm_rows()
    if not vl.empty and (vl.source == "rerun").any():
        vl.to_csv(OUT / "vllm_tp_concurrency_comparison.csv", index=False)
        parts += ["## Re-run 2: vLLM tensor parallel with matched concurrency (output tokens/s, medians)\n",
                  vllm_table(vl), ""]
    (OUT / "tables.md").write_text("\n".join(parts) + "\n")
    print((OUT / "tables.md").read_text())


if __name__ == "__main__":
    main()
