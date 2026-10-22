#!/usr/bin/env python
"""Build reports/REPORT.md and reports/figures/*.png from the CSVs written by scripts/aggregate.py."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REP = ROOT / "reports"
FIG = REP / "figures"

# Categorical slots in fixed order (validated reference palette); colour follows the framework, never its rank.
FW_COLOR = {"ddp": "#2a78d6", "fsdp": "#eb6834", "zero1": "#1baf7a", "zero2": "#eda100", "zero3": "#e87ba4"}
FW_MARKER = {"ddp": "o", "fsdp": "s", "zero1": "^", "zero2": "D", "zero3": "v"}  # secondary encoding
FW_LABEL = {"ddp": "DDP", "fsdp": "FSDP2", "zero1": "ZeRO-1", "zero2": "ZeRO-2", "zero3": "ZeRO-3"}
SET_COLOR = ["#2a78d6", "#eb6834", "#1baf7a"]
LAYOUT_COLOR = {"TP": "#2a78d6", "DP": "#eb6834"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODEL_LABEL = {"resnet50": "ResNet-50 (224 px)", "qwen2.5-1.5b": "Qwen2.5-1.5B (seq 1024)",
               "llama3.1-8b": "Llama-3.1-8B (seq 1024)", "qwen2.5-0.5b": "Qwen2.5-0.5B"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})


def _unit(model: str) -> tuple[str, str]:
    return ("throughput_samples_s", "images/s") if model == "resnet50" else ("throughput_tokens_s", "tokens/s")


def _fig_legend(fig, axes, ncol):
    """One shared legend row under the suptitle (never on top of the data)."""
    seen = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    order = [FW_LABEL[f] for f in FW_COLOR] + ["fp32 gradient reduce", "bf16 gradient reduce"]
    labels = [lab for lab in order if lab in seen] + [lab for lab in seen if lab not in order]
    fig.legend([seen[lab] for lab in labels], labels, loc="upper left", bbox_to_anchor=(0.01, 0.935),
               ncol=ncol, fontsize=9, handlelength=1.8, columnspacing=1.6)


def fig_train_throughput(st: pd.DataFrame):
    """ResNet-50 and Qwen2.5-1.5B vs GPU count (lines); Llama-3.1-8B only runs on 4 GPUs, so its panel shows
    the two sharded frameworks with fp32 vs bf16 gradient reduction (bars, hatch = bf16)."""
    main = st[st["variant"] == ""]
    line_models = [m for m in ("resnet50", "qwen2.5-1.5b") if m in set(main["model"])]
    has_llama = ((st["model"] == "llama3.1-8b") & (st["status"] == "ok")).any()
    n_pan = len(line_models) + int(has_llama)
    fig, axes = plt.subplots(1, n_pan, figsize=(4.6 * n_pan, 4.1))
    axes = [axes] if n_pan == 1 else list(axes)
    for ax, model in zip(axes, line_models):
        col, unit = _unit(model)
        g = main[(main["model"] == model) & (main["status"] == "ok")]
        for fw in FW_COLOR:
            d = g[g["framework"] == fw].sort_values("n_gpus")
            if d.empty:
                continue
            ax.plot(d["n_gpus"], d[col], color=FW_COLOR[fw], marker=FW_MARKER[fw], label=FW_LABEL[fw],
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(MODEL_LABEL[model], color=INK, loc="left")
        ax.set_xlabel("GPUs (RTX A6000)")
        ax.set_ylabel(unit)
        ax.set_xticks([1, 2, 4])
        ax.set_ylim(0, g[col].max() * 1.12 if len(g) else 1)
    if has_llama:
        ax = axes[-1]
        g = st[(st["model"] == "llama3.1-8b") & (st["status"] == "ok") & (st["n_gpus"] == 4)]
        fws = [f for f in FW_COLOR if f in set(g["framework"])]
        for i, fw in enumerate(fws):
            for k, (variant, hatch) in enumerate((("", None), ("bf16-reduce", "///"))):
                r = g[(g["framework"] == fw) & (g["variant"] == variant)]
                if r.empty:
                    continue
                x = i + (k - 0.5) * 0.38
                v = float(r["throughput_tokens_s"].iloc[0])
                ax.bar(x, v, width=0.34, color=FW_COLOR[fw], hatch=hatch, edgecolor=SURFACE, linewidth=0,
                       label=FW_LABEL[fw] if k == 0 else None)
                ax.text(x, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.bar([0], [0], color="#c3c2b7", label="fp32 gradient reduce")
        ax.bar([0], [0], color="#c3c2b7", hatch="///", edgecolor=SURFACE, linewidth=0, label="bf16 gradient reduce")
        ax.set_xticks(range(len(fws)), [FW_LABEL[f] for f in fws])
        ax.set_title("Llama-3.1-8B (seq 1024), 4 GPUs", color=INK, loc="left")
        ax.set_ylabel("tokens/s")
        ax.set_ylim(0, g["throughput_tokens_s"].max() * 1.18)
        ax.grid(axis="x", visible=False)
        ax.text(0.5, -0.17, "1-2 GPUs: does not fit (see feasibility table)", transform=ax.transAxes,
                ha="center", fontsize=8, color=INK2)
    fig.suptitle("Training throughput (weak scaling: fixed per-GPU batch)", x=0.01, ha="left", color=INK)
    _fig_legend(fig, axes, ncol=7)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIG / "train_throughput.png", dpi=150)
    plt.close(fig)


def fig_scaling_efficiency(st: pd.DataFrame):
    d = st[(st["status"] == "ok") & (st["n_gpus"] > 1) & st["scaling_efficiency"].notna()
           & (st["base_n_gpus"] == 1)]
    models = [m for m in ("resnet50", "qwen2.5-1.5b") if m in set(d["model"])]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 3.9), sharey=True)
    axes = [axes] if len(models) == 1 else list(axes)
    for ax, model in zip(axes, models):
        g = d[d["model"] == model]
        fws = [f for f in FW_COLOR if f in set(g["framework"])]
        ns = sorted(g["n_gpus"].unique())
        w = 0.8 / len(fws)
        for i, fw in enumerate(fws):
            for j, n in enumerate(ns):
                r = g[(g["framework"] == fw) & (g["n_gpus"] == n)]
                if r.empty:
                    continue
                v = float(r["scaling_efficiency"].iloc[0])
                x = j + (i - (len(fws) - 1) / 2) * w
                ax.bar(x, v, width=w * 0.92, color=FW_COLOR[fw], label=FW_LABEL[fw] if j == 0 else None)
                ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
        link = {2: "NVLink pair", 4: "2 pairs over PCIe"}
        ax.set_xticks(range(len(ns)), [f"{n} GPUs\n({link.get(n, '')})" for n in ns])
        ax.set_title(MODEL_LABEL[model], color=INK, loc="left")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("scaling efficiency vs 1 GPU")
    fig.suptitle("Scaling efficiency = throughput(N) / (N x throughput(1 GPU)), same framework",
                 x=0.01, ha="left", color=INK)
    _fig_legend(fig, axes, ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIG / "scaling_efficiency.png", dpi=150)
    plt.close(fig)


def fig_memory(st: pd.DataFrame):
    models = [m for m in ("resnet50", "qwen2.5-1.5b", "llama3.1-8b") if m in set(st["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.1), sharey=True)
    axes = [axes] if len(models) == 1 else list(axes)
    for ax, model in zip(axes, models):
        g = st[st["model"] == model]
        fws = [f for f in FW_COLOR if f in set(g["framework"])]
        ns = sorted(g["n_gpus"].unique())
        w = 0.8 / len(fws)
        for i, fw in enumerate(fws):
            for j, n in enumerate(ns):
                r = g[(g["framework"] == fw) & (g["n_gpus"] == n)]
                if r.empty:
                    continue
                x = j + (i - (len(fws) - 1) / 2) * w
                if r["status"].iloc[0] == "ok":
                    ax.bar(x, r["peak_mem_reserved_gib"].iloc[0], width=w * 0.92, color=FW_COLOR[fw],
                           label=FW_LABEL[fw])
                else:
                    ax.text(x, 0.6, "x", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.axhline(48, color=INK2, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(ns)), [f"{n} GPU" + ("s" if n > 1 else "") for n in ns])
        ax.set_title(MODEL_LABEL[model], color=INK, loc="left")
        ax.set_xlim(-0.5, len(ns) - 0.5)
        ax.set_ylim(0, 52)
        ax.grid(axis="x", visible=False)
    axes[0].text(-0.45, 48.6, "48 GiB per GPU", ha="left", va="bottom", fontsize=8, color=INK2)
    axes[0].set_ylabel("peak reserved memory per GPU (GiB)")
    fig.suptitle("Peak GPU memory per rank   (x = not launched: model states alone exceed the 44 GiB cap)",
                 x=0.01, ha="left", color=INK)
    _fig_legend(fig, axes, ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(FIG / "peak_memory.png", dpi=150)
    plt.close(fig)


def fig_vllm(sv: pd.DataFrame):
    models = [m for m in ("qwen2.5-1.5b", "llama3.1-8b") if m in set(sv["model"])]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 3.8))
    axes = [axes] if len(models) == 1 else axes
    for ax, model in zip(axes, models):
        g = sv[(sv["model"] == model) & (sv["status"] == "ok")].sort_values(["n_gpus", "dp"])
        xs = list(range(len(g)))
        colors = [LAYOUT_COLOR["DP" if r.dp > 1 else "TP"] for r in g.itertuples()]
        ax.bar(xs, g["throughput_output_tokens_s"], color=colors, width=0.7)
        for x, r in zip(xs, g.itertuples()):
            eff = "" if pd.isna(r.scaling_efficiency) else f"\n({float(r.scaling_efficiency):.0%})"
            ax.text(x, r.throughput_output_tokens_s, f"{r.throughput_output_tokens_s:,.0f}{eff}",
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(xs, [f"{r.layout}\n{r.n_gpus} GPU" for r in g.itertuples()])
        ax.set_title(MODEL_LABEL[model].split(" (")[0] + " - vLLM", color=INK, loc="left")
        ax.set_ylabel("output tokens/s")
        ax.set_ylim(0, g["throughput_output_tokens_s"].max() * 1.25)
        ax.grid(axis="x", visible=False)
        ax.bar([0], [0], color=LAYOUT_COLOR["TP"], label="tensor parallel (one engine)")
        ax.bar([0], [0], color=LAYOUT_COLOR["DP"], label="data parallel (independent engines)")
        ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("vLLM offline throughput, 1024 prompts x (512 in / 256 out); (%) = scaling efficiency vs 1 GPU",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "vllm_throughput.png", dpi=150)
    plt.close(fig)


def fig_nccl(nc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, (label, g) in enumerate(nc.groupby("label", sort=False)):
        g = g.sort_values("size_mib")
        ax.plot(g["size_mib"], g["busbw_gbs"], color=SET_COLOR[i % 3], marker="osD"[i % 3],
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.set_xscale("log", base=2)
    sizes = sorted(nc["size_mib"].unique())
    ax.set_xticks(sizes, [str(int(x)) for x in sizes])
    ax.minorticks_off()
    ax.set_xlabel("message size (MiB)")
    ax.set_ylabel("bus bandwidth (GB/s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("NCCL all-reduce (bf16) bus bandwidth", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "nccl_busbw.png", dpi=150)
    plt.close(fig)


def _fmt(v, nd=1, pct=False):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return "-"
    if pct:
        return f"{100 * float(v):.{nd}f}%"
    return f"{float(v):,.{nd}f}"


def train_table(st: pd.DataFrame) -> str:
    lines = ["| Model | Framework | GPUs | Throughput | per GPU | Step (ms) | Scaling eff. | MFU | Peak mem/GPU (GiB) | GPU util | Power/GPU (W) | Runs |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    order = {m: i for i, m in enumerate(["resnet50", "qwen2.5-0.5b", "qwen2.5-1.5b", "llama3.1-8b"])}
    st = st[st["status"] != "skipped_infeasible"].copy()
    st["_m"] = st["model"].map(order)
    st = st.sort_values(["_m", "n_gpus", "framework"], key=lambda c: c.map(
        {"ddp": 0, "fsdp": 1, "zero1": 2, "zero2": 3, "zero3": 4}) if c.name == "framework" else c)
    for r in st.itertuples():
        col, unit = _unit(r.model)
        if r.status != "ok":
            note = {"skipped_infeasible": f"does not fit (states ~{_fmt(r.state_gib_per_gpu_est)} GiB/GPU)",
                    "oom": "OOM"}.get(r.status, r.status)
            lines.append(f"| {r.model} | {FW_LABEL[r.framework]} | {r.n_gpus} | {note} | | | | | | | | 0 |")
            continue
        tput = getattr(r, col)
        unit_short = "img/s" if unit == "images/s" else "tok/s"
        lines.append(
            f"| {r.model} | {FW_LABEL[r.framework]} | {r.n_gpus} | {_fmt(tput, 0)} {unit_short} | "
            f"{_fmt(tput / r.n_gpus, 0)} | {_fmt(r.step_median_ms)} | "
            f"{_fmt(r.scaling_efficiency, 0, pct=True) if r.n_gpus > 1 else '-'} | {_fmt(r.mfu, 1, pct=True)} | "
            f"{_fmt(r.peak_mem_reserved_gib)} | {_fmt(r.gpu_util_mean_pct, 0)}% | {_fmt(r.power_mean_w_per_gpu, 0)} | "
            f"{r.repeats_ok} |")
    return "\n".join(lines)


def vllm_table(sv: pd.DataFrame) -> str:
    lines = ["| Model | Layout | GPUs | Output tok/s | Total tok/s | Req/s | Output tok/s per GPU | Scaling eff. | GPU util | Runs |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sv.itertuples():
        if r.status != "ok":
            lines.append(f"| {r.model} | {r.layout} | {r.n_gpus} | {r.status} | | | | | | 0 |")
            continue
        lines.append(f"| {r.model} | {r.layout} | {r.n_gpus} | {_fmt(r.throughput_output_tokens_s, 0)} | "
                     f"{_fmt(r.throughput_total_tokens_s, 0)} | {_fmt(r.throughput_requests_s, 2)} | "
                     f"{_fmt(r.throughput_output_tokens_s_per_gpu, 0)} | {_fmt(r.scaling_efficiency, 0, pct=True)} | "
                     f"{_fmt(r.gpu_util_mean_pct, 0)}% | {r.repeats_ok} |")
    return "\n".join(lines)


def nccl_table(nc: pd.DataFrame) -> str:
    sizes = sorted(nc["size_mib"].unique())
    lines = ["| GPUs (setting) | Links | " + " | ".join(f"{s} MiB" for s in sizes) + " |",
             "|---|---|" + "---:|" * len(sizes)]
    for label, g in nc.groupby("label", sort=False):
        vals = [g[g["size_mib"] == s]["busbw_gbs"] for s in sizes]
        lines.append(f"| {label} | {g['links'].iloc[0]} | " + " | ".join(_fmt(v.iloc[0]) if len(v) else "-"
                                                                       for v in vals) + " |")
    return "\n".join(lines)


def feasibility_table(st: pd.DataFrame, model: str) -> str:
    g = st[(st["model"] == model) & (st["variant"] == "")]
    ns = sorted(g["n_gpus"].unique())
    lines = ["| Framework | " + " | ".join(f"{n} GPU" + ("s" if n > 1 else "") for n in ns) + " |",
             "|---|" + "---|" * len(ns)]
    for fw in FW_COLOR:
        cells = []
        for n in ns:
            r = g[(g["framework"] == fw) & (g["n_gpus"] == n)]
            if r.empty:
                cells.append("-")
            elif r["status"].iloc[0] == "ok":
                cells.append(f"runs: {_fmt(r['peak_mem_reserved_gib'].iloc[0])} GiB peak")
            elif r["status"].iloc[0] == "skipped_infeasible":
                cells.append(f"no: states ~{_fmt(r['state_gib_per_gpu_est'].iloc[0], 0)} GiB/GPU")
            else:
                cells.append(r["status"].iloc[0])
        if any(c != "-" for c in cells):
            lines.append(f"| {FW_LABEL[fw]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ablation_table(st: pd.DataFrame) -> str:
    """Variant runs next to their baseline (same model, framework, GPU count)."""
    lines = ["| Model | Framework | GPUs | Variant | Throughput (tok/s) | Step (ms) | vs baseline | Peak mem/GPU (GiB) | Runs |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in st[(st["variant"] != "") & (st["status"] == "ok")].itertuples():
        base = st[(st["model"] == r.model) & (st["variant"] == "") & (st["framework"] == r.framework)
                  & (st["n_gpus"] == r.n_gpus) & (st["status"] == "ok")]
        col, _ = _unit(r.model)
        for label, row, rel in ([("baseline (fp32 reduce)", b, None) for b in base.itertuples()]
                                + [(r.variant, r, None)]):
            ratio = getattr(row, col) / getattr(base.iloc[0], col) if len(base) else None
            lines.append(f"| {r.model} | {FW_LABEL[r.framework]} | {r.n_gpus} | {label} | {_fmt(getattr(row, col), 0)} | "
                         f"{_fmt(row.step_median_ms, 0)} | {_fmt(ratio, 2)}x | {_fmt(row.peak_mem_reserved_gib)} | "
                         f"{row.repeats_ok} |")
    return "\n".join(lines)


LLAMA_NOTE = '**Llama-3.1-8B note (audit re-run):** ZeRO-3 shows ~2 allocator retries per step with default DeepSpeed buffers; FSDP2 shows none. A re-run with more memory headroom reproduced both numbers within 1-2 % (`results/rerun/RERUN_ANALYSIS.md`).\n'
VLLM_NOTE = '**Audit re-run:** the TP rows above ran at `max_num_seqs=256` per engine, so DP engines together had more concurrency. The README uses TP results re-run with concurrency matched to the GPU count (512 for TP2, 1024 for TP4): Qwen2.5-1.5B 11,811 / 4,993 and Llama-3.1-8B 3,101 / 1,830 output tok/s. Data parallel beats tensor parallel even at matched concurrency (1.15–1.33× on NVLink, 3.9–6.3× across PCIe). See `results/rerun/RERUN_ANALYSIS.md`.\n'


def run_breakdown(runs: pd.DataFrame) -> str:
    purpose = {"smoke": "plumbing check, 1/2/4 GPUs", "probe_1gpu": "micro-batch / precision selection",
               "probe_4gpu": "Llama-8B micro-batch fit", "vllm_smoke": "vLLM plumbing check",
               "nccl": "interconnect bandwidth", "nccl_diag": "PCIe P2P diagnosis",
               "train_scaling": "training results (this report)", "vllm_tp4_diag": "vLLM TP=4 diagnosis",
               "vllm_scaling": "inference results (this report)"}
    t = runs.groupby(["experiment", "status"]).size().unstack(fill_value=0)
    cols = [c for c in ("ok", "skipped_infeasible", "oom", "failed", "timeout") if c in t.columns]
    lines = ["| Experiment | Purpose | " + " | ".join(cols) + " |", "|---|---|" + "---:|" * len(cols)]
    for exp in [e for e in purpose if e in t.index] + [e for e in t.index if e not in purpose]:
        lines.append(f"| `{exp}` | {purpose.get(exp, '')} | " + " | ".join(str(t.loc[exp, c]) for c in cols) + " |")
    return "\n".join(lines)


def environment_block() -> str:
    for p in sorted((RES / "raw").glob("*/*/result.json")):
        env = json.loads(p.read_text())["env"]
        sw = env["software"]
        return (f"- Host `{env['hostname']}`, Linux {env['kernel']}, NVIDIA driver {env['driver']['driver_version']}\n"
                f"- GPUs: {env['gpus'][0]['name']} ({env['gpus'][0]['mem_total_mib'] / 1024:.0f} GiB), "
                f"runs used physical GPUs 4-7 only\n"
                f"- Python {sw['python']}, PyTorch {sw['torch']} (CUDA {sw['torch_cuda']}, NCCL {sw['nccl']}), "
                f"DeepSpeed {sw['deepspeed']}, vLLM {sw['vllm']}, transformers {sw['transformers']}, "
                f"torchvision {sw['torchvision']}")
    return "-"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(RES / "summary_train.csv") if (RES / "summary_train.csv").exists() else pd.DataFrame()
    if not st.empty:
        st["variant"] = st["variant"].fillna("")
    main_st = st[st["variant"] == ""] if not st.empty else st
    sv = pd.read_csv(RES / "summary_vllm.csv") if (RES / "summary_vllm.csv").exists() else pd.DataFrame()
    nc = pd.read_csv(RES / "nccl.csv") if (RES / "nccl.csv").exists() else pd.DataFrame()
    runs = pd.read_csv(RES / "runs.csv")

    parts = ["# Multi-GPU Benchmark Report - Stage 1 (RTX A6000)\n",
             "**Historical report; interpretation superseded by [CURRENT_REPORT.md](../CURRENT_REPORT.md).** "
             "Causal, capacity, precision-equivalence and inference telemetry claims require the corrections there.\n",
             "_Generated by `scripts/make_report.py` from `results/`. Every number below is a median over the "
             "repeated runs listed in the Runs column; per-run data is in `results/runs.csv` and "
             "`results/raw/`._\n",
             "## Environment\n", environment_block(), ""]
    counts = runs["status"].value_counts().to_dict()
    parts += [f"Runs recorded: {len(runs)} ({', '.join(f'{k}: {v}' for k, v in counts.items())}), "
              "by experiment:\n", run_breakdown(runs), "",
              "The two `failed` runs are the default-settings all-reduce on GPUs 4+6 (the PCIe peer-to-peer hang "
              "described below; once in `nccl`, once with debug logging in `nccl_diag`). The two `oom` runs are "
              "batch-size probes that set the per-GPU micro-batch. `skipped_infeasible` configurations were "
              "recorded without being launched.\n"]

    if not nc.empty:
        fig_nccl(nc)
        parts += ["## Interconnect: NCCL all-reduce bus bandwidth (GB/s)\n", nccl_table(nc), "",
                  "![NCCL bus bandwidth](figures/nccl_busbw.png)\n",
                  "**PCIe peer-to-peer between the NVLink pairs does not work on this node.** A 2-rank all-reduce on "
                  "GPUs 4+6 (link `NODE`) with default settings never completes its first collective: NCCL picks "
                  "`P2P/CUMEM` over PCIe for that pair (`NCCL_DEBUG=INFO`, `results/raw/nccl_diag/`), and the "
                  "NCCL watchdog aborts it after the 5-min timeout. With `NCCL_P2P_DISABLE=1` the same pair works through "
                  "host memory (`SHM`) at ~5.5 GB/s. For the 4-GPU ring NCCL already uses `P2P/CUMEM` only on the "
                  "NVLink hops (4-5, 6-7) and `SHM` for the PCIe hops, which is why 4-GPU training needed no change; "
                  "`NCCL_P2P_LEVEL=NVL` gives the identical transport plan and bandwidth. The likely cause is a host "
                  "setting (IOMMU/ACS peer-to-peer routing), which is outside the scope of an unprivileged user.\n"]
    if not main_st.empty:
        fig_train_throughput(st)
        fig_scaling_efficiency(main_st)
        fig_memory(main_st)
        parts += ["## Training (DDP, FSDP2, DeepSpeed ZeRO)\n",
                  "Weak scaling: fixed per-GPU micro-batch, bf16 compute, AdamW, synthetic data, random init. "
                  "Throughput is measured over the timed steps after warmup; step time is the slowest rank. "
                  "Scaling efficiency compares a framework with itself on 1 GPU. MFU uses model FLOPs "
                  "(6N + 12LdT per token for LMs, counted FLOPs for ResNet-50) against the A6000's 154.8 TFLOPS "
                  "dense BF16 peak; for Llama-3.1-8B the activation-checkpoint recompute is not counted. "
                  "Peak memory is the PyTorch caching-allocator reservation (max over ranks).\n",
                  train_table(main_st), "",
                  "**Reading these numbers.** (1) The 2-GPU runs use GPUs 4-5, an NVLink pair; the 4-GPU runs join "
                  "two NVLink pairs over PCIe, whose all-reduce bus bandwidth is ~4x lower (Interconnect section), so "
                  "the 2 -> 4 GPU step is mostly a test of PCIe communication. (2) NVML `utilization.gpu` stays near "
                  "100 % even when ranks are waiting on NCCL (communication kernels count as busy); mean board power "
                  "is the better stall indicator - see the power column in `results/summary_train.csv`. "
                  "(3) With gradient accumulation (Qwen: 4 micro-steps per optimizer step), DDP and ZeRO-1 communicate "
                  "gradients once per optimizer step, while ZeRO-2 reduce-scatters on every micro-step "
                  "(DeepSpeed `engine.py`: \"ZeRO stage >= 2 communicates during non gradient accumulation "
                  "boundaries as well\") and FSDP2 re-gathers parameters for every micro-step's forward and backward.\n",
                  "### Memory feasibility of full training for Llama-3.1-8B (48 GiB per GPU)\n",
                  "Mixed-precision AdamW needs ~16 bytes/parameter of model state (~120 GiB for 8B) before "
                  "activations. Configurations whose estimated state alone exceeds the 44 GiB cap are recorded, "
                  "not launched (see `safety.py`).\n",
                  feasibility_table(main_st, "llama3.1-8b"), "",
                  LLAMA_NOTE,
                  "![Training throughput](figures/train_throughput.png)\n",
                  "![Scaling efficiency](figures/scaling_efficiency.png)\n",
                  "![Peak memory](figures/peak_memory.png)\n"]
        if (st["variant"] != "").any():
            parts += ["### Ablation: gradient-reduction precision for Llama-3.1-8B\n",
                      "Same runs with bf16 instead of fp32 gradient reduce-scatter (FSDP2 `reduce_dtype`, DeepSpeed "
                      "`communication_data_type`), which halves the gradient bytes on the wire.\n",
                      ablation_table(st), ""]
    if not sv.empty:
        fig_vllm(sv)
        parts += ["## Inference (vLLM)\n",
                  "Strong scaling: the same 1024 requests (512 random prompt tokens, exactly 256 generated tokens, "
                  "greedy, `ignore_eos`) on every layout. TP = one engine split across GPUs; DP = independent "
                  "single-GPU engines that each take an equal share of the prompts and start behind a barrier. "
                  "TP=4 runs use `NCCL_P2P_LEVEL=NVL` (see Interconnect): with default settings vLLM's TP=4 "
                  "communicator hung in its first decode step (`results/raw/vllm_tp4_diag/`).\n",
                  vllm_table(sv), "",
                  VLLM_NOTE,
                  "![vLLM throughput](figures/vllm_throughput.png)\n"]
    (REP / "REPORT.md").write_text("\n".join(parts) + "\n")
    print(f"wrote {REP / 'REPORT.md'} and {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
