"""
Collect results/raw/*.json into tidy CSVs and markdown summary tables.

Safe to run at any point; tracks that have not been run yet are simply absent
from the output rather than causing an error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
RAW = PROJ / "results" / "raw"
OUT = PROJ / "results"


def load(pattern: str) -> list[dict]:
    recs = []
    for p in sorted(RAW.glob(pattern)):
        try:
            with open(p) as fh:
                r = json.load(fh)
            r["_file"] = p.name
            recs.append(r)
        except Exception as exc:  # pragma: no cover
            print(f"  warn: could not read {p.name}: {exc}")
    return recs


def _tel_sum(tel: dict | None, key: str) -> float | None:
    """Sum a per-GPU telemetry dict across all GPUs in the run."""
    if not tel or not tel.get(key):
        return None
    return round(sum(tel[key].values()), 2)


def _tel_mean(tel: dict | None, key: str) -> float | None:
    if not tel or not tel.get(key):
        return None
    vals = list(tel[key].values())
    return round(sum(vals) / len(vals), 2) if vals else None


# ---------------------------------------------------------------------------
# dtype probe
# ---------------------------------------------------------------------------

def dtype_table() -> pd.DataFrame:
    rows = []
    for r in load("01_dtype_*.json"):
        d, res = r["device"], r["results"]
        row = {
            "gpu": d["name"],
            "capability": d["capability"],
            "sm_count": d["sm_count"],
            "vram_gb": round(d["total_mem_mib"] / 1024, 1),
            "is_bf16_supported_flag": r["flags"]["is_bf16_supported"],
        }
        for dt in ("fp32", "fp16", "bf16"):
            row[f"{dt}_tflops"] = res.get(dt, {}).get("tflops")
        row["fp16_speedup"] = r.get("fp16_vs_fp32_speedup")
        row["bf16_speedup"] = r.get("bf16_vs_fp32_speedup")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gpu") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# LLM training (single + multi-GPU share one schema)
# ---------------------------------------------------------------------------

def llm_table() -> pd.DataFrame:
    rows = []
    # Explicit prefixes, NOT a [ACDE]_* character class: that also matched
    # C_comm_*.json (the interconnect records), which have no perf/config
    # schema and leaked in as all-NaN rows.
    files = []
    for pat in ("A_*.json", "C_ddp_*.json", "D_*.json", "E_*.json"):
        files.extend(load(pat))
    for r in files:
        c, perf, mem = r.get("config", {}), r.get("perf", {}), r.get("memory", {})
        if not c or not perf:
            continue
        tel = r.get("telemetry")
        tokens_total = (perf.get("tokens_per_step") or 0) * (c.get("steps") or 0)
        # energy_total_j is already summed across GPUs by the sampler.
        energy = tel.get("energy_total_j") if tel else None
        rows.append(
            {
                "track": r["_file"].split("_")[0],
                "mode": c.get("mode"),
                "zero_stage": c.get("zero_stage"),
                "group": c.get("group"),
                "precision": c.get("precision"),
                "world_size": c.get("world_size"),
                "micro_batch_per_gpu": c.get("micro_batch_per_gpu"),
                "global_batch": c.get("global_batch"),
                "seq_len": c.get("seq_len"),
                "status": r.get("status"),
                "tokens_per_s": perf.get("tokens_per_s"),
                "tokens_per_s_per_gpu": perf.get("tokens_per_s_per_gpu"),
                "step_time_median_s": perf.get("step_time_median_s"),
                "wall_time_s": perf.get("wall_time_s"),
                "peak_alloc_mib": mem.get("peak_alloc_mib_rank0"),
                "peak_reserved_mib": mem.get("peak_reserved_mib_rank0"),
                "util_mean_pct": _tel_mean(tel, "util_mean_pct"),
                "power_mean_w_per_gpu": _tel_mean(tel, "power_mean_w"),
                "energy_total_j": energy,
                "energy_per_1m_tokens_j": (
                    round(energy / tokens_total * 1e6, 1)
                    if energy and tokens_total
                    else None
                ),
                "smi_peak_mem_mib_max": (
                    max(tel["mem_max_mib"].values())
                    if tel and tel.get("mem_max_mib")
                    else None
                ),
                "file": r["_file"],
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["mode", "group", "precision", "micro_batch_per_gpu"], na_position="last"
    )


def scaling_table(df: pd.DataFrame) -> pd.DataFrame:
    """Scaling efficiency versus the 1-GPU baseline of the same GPU type.

    Weak scaling: micro-batch per GPU is held constant, so ideal throughput is
    linear in world size. Efficiency = measured / (1-GPU throughput x N).
    """
    if df.empty:
        return pd.DataFrame()
    ok = df[(df["status"] == "ok") & df["tokens_per_s"].notna()].copy()
    if ok.empty:
        return pd.DataFrame()

    # map a group label to its GPU family for baseline lookup
    def family(g: str) -> str:
        if not isinstance(g, str):
            return "unknown"
        if g.startswith("rtx"):
            return "TITAN RTX"
        if g.startswith("p1080"):
            return "GTX 1080 Ti"
        if g.startswith("xp"):
            return "TITAN Xp"
        return "unknown"

    ok["family"] = ok["group"].map(family)
    rows = []
    for (fam, prec, bs), grp in ok.groupby(
        ["family", "precision", "micro_batch_per_gpu"], dropna=False
    ):
        base = grp[(grp["world_size"] == 1) & (grp["mode"] == "single")]
        if base.empty:
            continue
        base_tp = float(base["tokens_per_s"].iloc[0])
        for _, row in grp.iterrows():
            n = int(row["world_size"])
            ideal = base_tp * n
            rows.append(
                {
                    "family": fam,
                    "mode": row["mode"],
                    "zero_stage": row["zero_stage"],
                    "group": row["group"],
                    "precision": prec,
                    "micro_batch_per_gpu": bs,
                    "n_gpus": n,
                    "tokens_per_s": row["tokens_per_s"],
                    "tokens_per_s_per_gpu": row["tokens_per_s_per_gpu"],
                    "baseline_1gpu_tokens_per_s": round(base_tp, 1),
                    "ideal_linear_tokens_per_s": round(ideal, 1),
                    "speedup_vs_1gpu": round(row["tokens_per_s"] / base_tp, 3),
                    "scaling_efficiency_pct": round(
                        100.0 * row["tokens_per_s"] / ideal, 2
                    ),
                    "peak_alloc_mib": row["peak_alloc_mib"],
                    "util_mean_pct": row["util_mean_pct"],
                    "energy_per_1m_tokens_j": row["energy_per_1m_tokens_j"],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["family", "precision", "mode", "n_gpus"], na_position="last"
    )


# ---------------------------------------------------------------------------
# CNN track
# ---------------------------------------------------------------------------

def cnn_table() -> pd.DataFrame:
    rows = []
    for r in load("B_cnn_*.json"):
        c, dev = r["config"], r["device"]
        for row in r["rows"]:
            if row.get("status") != "ok":
                rows.append(
                    {
                        "model": c["model"],
                        "gpu": dev["name"],
                        "batch_size": row.get("batch_size"),
                        "status": row.get("status"),
                    }
                )
                continue
            rows.append(
                {
                    "model": c["model"],
                    "gpu": dev["name"],
                    "capability": dev["capability"],
                    "precision": c["precision"],
                    "batch_size": row["batch_size"],
                    "status": "ok",
                    # column names kept aligned with the prior A6000 CSV
                    "energy_per_image": row["energy_per_image"],
                    "energy_per_image_net": row["energy_per_image_net"],
                    "throughput": row["throughput"],
                    "runtime": row["runtime"],
                    "memory_allocated_gb": row["memory_allocated_gb"],
                    "memory_reserved_gb": row["memory_reserved_gb"],
                    "max_memory_allocated_gb": row["max_memory_allocated_gb"],
                    "achieved_tflops": row["achieved_tflops"],
                    "theoretical_fp32_tflops": dev["theoretical_fp32_tflops"],
                    "compute_util_pct": row.get("compute_util_pct"),
                    "power_mean_w": (
                        round(sum(row["power_mean_w"].values()), 1)
                        if row.get("power_mean_w")
                        else None
                    ),
                    "idle_power_w": row.get("idle_power_w"),
                    "n_images": row["n_images"],
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# inference track
# ---------------------------------------------------------------------------

def infer_table() -> pd.DataFrame:
    """Normalise vLLM and HF-generate rows onto one comparable schema.

    The two engines name things differently -- vLLM reports `concurrency` and
    `output_tokens_per_s`, HF reports `batch_size` and `gen_tokens_per_s` -- so
    both are mapped to `requests` and `output_tokens_per_s` here. They remain
    different serving stacks; the `engine` column keeps that visible.
    """
    rows = []
    for r in load("F_*.json"):
        c = r.get("config", {})
        dev = r.get("device", {})
        if r.get("status") == "unsupported_architecture":
            rows.append(
                {
                    "engine": c.get("engine"),
                    "gpu": dev.get("name"),
                    "capability": dev.get("capability"),
                    "status": "unsupported_architecture",
                    "note": r.get("reason"),
                }
            )
            continue
        for row in r.get("rows", []):
            if row.get("status") != "ok":
                rows.append(
                    {
                        "engine": c.get("engine"),
                        "gpu": dev.get("name"),
                        "requests": row.get("concurrency") or row.get("batch_size"),
                        "status": row.get("status"),
                    }
                )
                continue
            tps = row.get("output_tokens_per_s") or row.get("gen_tokens_per_s")
            e_key = (
                "energy_per_1k_output_tokens_j"
                if "energy_per_1k_output_tokens_j" in row
                else "energy_per_1k_tokens_j"
            )
            rows.append(
                {
                    "engine": c.get("engine"),
                    "gpu": dev.get("name"),
                    "capability": dev.get("capability"),
                    "dtype": c.get("dtype"),
                    "tensor_parallel": c.get("tensor_parallel"),
                    "model": c.get("model"),
                    "requests": row.get("concurrency") or row.get("batch_size"),
                    "status": "ok",
                    "prompt_len": row.get("prompt_len"),
                    "max_new_tokens": row.get("max_new_tokens"),
                    "output_tokens_per_s": tps,
                    "decode_tokens_per_s": row.get("decode_tokens_per_s"),
                    "ttft_ms": (
                        round(row["ttft_s"] * 1000, 2) if row.get("ttft_s") else None
                    ),
                    "elapsed_s": row.get("elapsed_s"),
                    "output_tokens": row.get("output_tokens") or row.get("new_tokens"),
                    "peak_alloc_mib": row.get("peak_alloc_mib"),
                    "energy_per_1k_output_tokens_j": row.get(e_key),
                    "power_mean_w": (
                        round(sum(row["power_mean_w"].values()), 1)
                        if row.get("power_mean_w")
                        else None
                    ),
                    "util_mean_pct": (
                        round(
                            sum(row["util_mean_pct"].values())
                            / len(row["util_mean_pct"]),
                            1,
                        )
                        if row.get("util_mean_pct")
                        else None
                    ),
                    "file": r["_file"],
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["engine", "gpu", "dtype", "tensor_parallel", "requests"], na_position="last"
    )


def comm_table() -> pd.DataFrame:
    """NCCL all-reduce bandwidth per group, with the P2P verdict attached.

    The historical size_mb field actually denotes MiB. The 497 MiB probe is
    near, but not identical to, GPT-2's parameter-sized FP32 gradient volume.
    """
    rows = []
    for r in load("C_comm_*.json"):
        c = r["config"]
        for row in r["rows"]:
            rows.append(
                {
                    "group": c["group"],
                    "n_gpus": c["world_size"],
                    "p2p_all_enabled": r.get("p2p_all_enabled"),
                    "size_mb": row["size_mb"],
                    "time_ms": row["time_ms"],
                    "algbw_gb_s": row["algbw_gb_s"],
                    "busbw_gb_s": row["busbw_gb_s"],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["group", "size_mb"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    wrote = []

    comm = comm_table()
    if not comm.empty:
        # Headline table: the historical 497 MiB buffer setting.
        hl = comm[comm["size_mb"] == 497][
            ["group", "n_gpus", "p2p_all_enabled", "time_ms",
             "algbw_gb_s", "busbw_gb_s"]
        ].sort_values(["n_gpus", "group"])
        hl.to_csv(OUT / "comm_bandwidth.csv", index=False)
        wrote.append(("comm_bandwidth", len(hl), OUT / "comm_bandwidth.csv"))
        comm.to_csv(OUT / "comm_bandwidth_full.csv", index=False)
        wrote.append(
            ("comm_bandwidth_full", len(comm), OUT / "comm_bandwidth_full.csv")
        )

    for name, df in (
        ("dtype_gemm", dtype_table()),
        ("llm_runs", llm_table()),
        ("cnn_resnet50", cnn_table()),
        ("inference", infer_table()),
    ):
        if df is None or df.empty:
            continue
        path = OUT / f"{name}.csv"
        df.to_csv(path, index=False)
        wrote.append((name, len(df), path))

    llm = llm_table()
    sc = scaling_table(llm)
    if not sc.empty:
        path = OUT / "scaling_efficiency.csv"
        sc.to_csv(path, index=False)
        wrote.append(("scaling_efficiency", len(sc), path))

    print("Wrote:")
    for name, n, path in wrote:
        print(f"  {name:<22} {n:>4} rows  -> {path.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
