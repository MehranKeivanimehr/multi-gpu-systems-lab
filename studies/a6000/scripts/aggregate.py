#!/usr/bin/env python
"""Collect results/raw/<experiment>/<run_id>/{status,result}.json into CSV tables.

Outputs (results/):
  runs.csv           one row per run (all experiments, all statuses)
  summary_train.csv  per (model, framework, n_gpus): median over repeats, spread, scaling efficiency
  summary_vllm.csv   per (model, tp, dp): median over repeats, efficiency vs. the 1-GPU engine
  nccl.csv           all-reduce bus bandwidth per GPU set and message size
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
OUT = ROOT / "results"
SCALING_EXPERIMENTS = ("train_scaling",)
VLLM_EXPERIMENTS = ("vllm_scaling",)


def load(p: Path):
    with open(p) as f:
        return json.load(f)


def run_rows() -> list[dict]:
    rows = []
    for st_path in sorted(RAW.glob("*/*/status.json")):
        st = load(st_path)
        spec = st["spec"]
        row = {
            "experiment": spec["experiment"], "run_id": spec["run_id"], "kind": spec["kind"],
            "model": spec["model"], "variant": spec.get("variant", ""), "framework": spec["framework"], "n_gpus": spec["n_gpus"],
            "gpus": "-".join(map(str, spec["gpus"])), "repeat": spec.get("repeat", 0), "status": st["status"],
            "duration_s": st.get("duration_s"), "idle_wait_before_s": st.get("idle_wait_before_s"),
            "state_gib_per_gpu_est": (st.get("estimate") or {}).get("state_gib_per_gpu_est"),
            "message": st.get("message", ""),
        }
        for k in ("micro_batch", "seq_len", "grad_accum", "activation_checkpointing", "tp", "dp",
                  "num_prompts", "input_len", "output_len"):
            if k in spec:
                row[k] = spec[k]
        res_path = st_path.parent / "result.json"
        if st["status"] == "ok" and res_path.exists():
            r = load(res_path)
            g = r.get("gpu", {})
            row.update({
                "gpu_util_mean_pct": g.get("gpu_util_mean_pct"), "gpu_util_min_pct": g.get("gpu_util_min_pct"),
                "nvml_mem_peak_gib": g.get("nvml_mem_peak_gib"), "power_mean_w_per_gpu": g.get("power_mean_w_per_gpu"),
                "temp_max_c": g.get("temp_max_c"), "sm_clock_mean_mhz": g.get("sm_clock_mean_mhz"),
                "window_s": r.get("window_s"),
            })
            if spec["kind"] == "train":
                row.update({
                    "n_params": r["n_params"], "samples_per_step": r["samples_per_step"],
                    "throughput_samples_s": r["throughput_samples_s"],
                    "throughput_tokens_s": r["throughput_tokens_s"],
                    "throughput_per_gpu_samples_s": r["throughput_per_gpu_samples_s"],
                    "step_median_ms": r["step_time"]["median_ms"], "step_p10_ms": r["step_time"]["p10_ms"],
                    "step_p90_ms": r["step_time"]["p90_ms"], "step_std_ms": r["step_time"]["std_ms"],
                    "tflops_per_gpu": r["tflops_per_gpu"], "mfu": r["mfu"],
                    "peak_mem_alloc_gib": r["peak_mem_alloc_gib"], "peak_mem_reserved_gib": r["peak_mem_reserved_gib"],
                    "setup_s": r["setup_s"], "final_loss": r["final_loss"], "loss_finite": r["loss_finite"],
                    "links": json.dumps(r["env"]["links"]),
                })
            elif spec["kind"] == "vllm":
                row.update({
                    "throughput_output_tokens_s": r["throughput_output_tokens_s"],
                    "throughput_total_tokens_s": r["throughput_total_tokens_s"],
                    "throughput_requests_s": r["throughput_requests_s"],
                    "throughput_output_tokens_s_per_gpu": r["throughput_output_tokens_s_per_gpu"],
                    "load_s_max": r["load_s_max"], "output_tokens": r["output_tokens"],
                    "links": json.dumps(r["env"]["links"]),
                })
        rows.append(row)
    return rows


def nccl_rows() -> list[dict]:
    rows = []
    for res_path in sorted(RAW.glob("nccl*/*/result.json")):
        r = load(res_path)
        links = r["env"]["links"]
        gpus = "-".join(map(str, r["spec"]["gpus"]))
        env = r["spec"].get("env") or {}
        setting = ("P2P disabled (SHM)" if str(env.get("NCCL_P2P_DISABLE")) == "1"
                   else "P2P on NVLink only" if env.get("NCCL_P2P_LEVEL") == "NVL" else "default")
        if r["spec"]["experiment"] == "nccl_diag" and setting == "default":
            continue  # debug-logging repeats of the default runs
        for x in r["rows"]:
            rows.append({"gpus": gpus, "setting": setting, "label": f"{gpus} ({setting})",
                         "n_gpus": r["spec"]["n_gpus"], "links": " ".join(sorted(set(links.values()))), **x})
    return rows


def _spread(s: pd.Series) -> float:
    return float(s.std(ddof=1) / s.mean()) if len(s) > 1 and s.mean() else 0.0


def summarize_train(runs: pd.DataFrame) -> pd.DataFrame:
    tr = runs[(runs["kind"] == "train") & runs["experiment"].isin(SCALING_EXPERIMENTS)]
    if tr.empty:
        return pd.DataFrame()
    keys = ["model", "variant", "framework", "n_gpus"]
    out = []
    for key, g in tr.fillna({"variant": ""}).groupby(keys):
        ok = g[g["status"] == "ok"]
        row = dict(zip(keys, key))
        row["status"] = "ok" if len(ok) else g["status"].iloc[0]
        row["repeats_ok"] = len(ok)
        row["state_gib_per_gpu_est"] = g["state_gib_per_gpu_est"].iloc[0]
        if len(ok):
            for c in ("throughput_samples_s", "throughput_tokens_s", "throughput_per_gpu_samples_s", "step_median_ms",
                      "tflops_per_gpu", "mfu", "peak_mem_alloc_gib", "peak_mem_reserved_gib", "nvml_mem_peak_gib",
                      "gpu_util_mean_pct", "power_mean_w_per_gpu", "temp_max_c", "micro_batch", "seq_len",
                      "samples_per_step"):
                if c in ok and ok[c].notna().any():
                    row[c] = float(ok[c].median())
            row["throughput_cv"] = _spread(ok["throughput_samples_s"])
        out.append(row)
    df = pd.DataFrame(out)
    # Scaling efficiency vs the smallest GPU count that ran OK for the same model + framework.
    df["base_n_gpus"] = pd.NA
    df["scaling_efficiency"] = pd.NA
    df["speedup"] = pd.NA
    for (model, variant, fw), g in df[df["status"] == "ok"].groupby(["model", "variant", "framework"]):
        base = g.sort_values("n_gpus").iloc[0]
        if (g["n_gpus"] > base["n_gpus"]).any() or base["n_gpus"] == 1:
            for idx, r in g.iterrows():
                df.loc[idx, "base_n_gpus"] = int(base["n_gpus"])
                df.loc[idx, "speedup"] = r["throughput_samples_s"] / base["throughput_samples_s"]
                df.loc[idx, "scaling_efficiency"] = (r["throughput_samples_s"] / base["throughput_samples_s"]) / (
                    r["n_gpus"] / base["n_gpus"])
    # Throughput relative to DDP on one GPU (cross-framework baseline) where DDP fits.
    df["vs_ddp_1gpu"] = pd.NA
    for (model, variant), g in df.groupby(["model", "variant"]):
        b = g[(g["framework"] == "ddp") & (g["n_gpus"] == 1) & (g["status"] == "ok")]
        if len(b):
            base = b["throughput_samples_s"].iloc[0]
            for idx, r in g[g["status"] == "ok"].iterrows():
                df.loc[idx, "vs_ddp_1gpu"] = r["throughput_samples_s"] / base
    order = {"ddp": 0, "fsdp": 1, "zero1": 2, "zero2": 3, "zero3": 4}
    df["_o"] = df["framework"].map(order)
    return df.sort_values(["model", "variant", "n_gpus", "_o"]).drop(columns="_o")


def summarize_vllm(runs: pd.DataFrame) -> pd.DataFrame:
    v = runs[(runs["kind"] == "vllm") & runs["experiment"].isin(VLLM_EXPERIMENTS)]
    if v.empty:
        return pd.DataFrame()
    keys = ["model", "tp", "dp", "n_gpus"]
    out = []
    for key, g in v.groupby(keys):
        ok = g[g["status"] == "ok"]
        row = dict(zip(keys, key))
        row["status"] = "ok" if len(ok) else g["status"].iloc[0]
        row["repeats_ok"] = len(ok)
        if len(ok):
            for c in ("throughput_output_tokens_s", "throughput_total_tokens_s", "throughput_requests_s",
                      "throughput_output_tokens_s_per_gpu", "gpu_util_mean_pct", "nvml_mem_peak_gib",
                      "power_mean_w_per_gpu", "load_s_max", "window_s"):
                row[c] = float(ok[c].median())
            row["throughput_cv"] = _spread(ok["throughput_output_tokens_s"])
        out.append(row)
    df = pd.DataFrame(out)
    df["scaling_efficiency"] = pd.NA
    df["speedup"] = pd.NA
    for model, g in df[df["status"] == "ok"].groupby("model"):
        b = g[(g["tp"] == 1) & (g["dp"] == 1)]
        if len(b):
            base = b["throughput_output_tokens_s"].iloc[0]
            for idx, r in g.iterrows():
                df.loc[idx, "speedup"] = r["throughput_output_tokens_s"] / base
                df.loc[idx, "scaling_efficiency"] = r["throughput_output_tokens_s"] / base / r["n_gpus"]
    df["layout"] = df.apply(lambda r: f"TP{int(r.tp)}" if r.dp == 1 else f"DP{int(r.dp)}", axis=1)
    return df.sort_values(["model", "n_gpus", "dp"])


def main():
    runs = pd.DataFrame(run_rows())
    if runs.empty:
        sys.exit("no runs found under results/raw")
    runs.to_csv(OUT / "runs.csv", index=False)
    st = summarize_train(runs)
    if not st.empty:
        st.to_csv(OUT / "summary_train.csv", index=False)
    sv = summarize_vllm(runs)
    if not sv.empty:
        sv.to_csv(OUT / "summary_vllm.csv", index=False)
    nc = pd.DataFrame(nccl_rows())
    if not nc.empty:
        nc.to_csv(OUT / "nccl.csv", index=False)
    print(f"runs: {len(runs)} ({runs['status'].value_counts().to_dict()})")
    print(f"wrote {OUT/'runs.csv'}" + (", summary_train.csv" if not st.empty else "")
          + (", summary_vllm.csv" if not sv.empty else "") + (", nccl.csv" if not nc.empty else ""))


if __name__ == "__main__":
    main()
