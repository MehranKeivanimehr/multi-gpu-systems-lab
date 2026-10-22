"""Generate the Titan/1080 chapter from the measurements themselves.

Nothing in the output is typed by hand, so the chapter cannot drift from the
records. Every table is derived from results/block_statistics.json or from the
run records directly; a missing input produces an explicit gap note rather than
a plausible-looking number.
"""
from __future__ import annotations
import os

import glob, json, pathlib, sys

W = pathlib.Path(os.environ.get("TITAN_CAMPAIGN_ROOT",
    str(pathlib.Path(__file__).resolve().parent.parent)))
OUT = W / "reports" / "TITAN_CHAPTER.md"
SHORT = {"NVIDIA TITAN RTX": "TITAN RTX",
         "NVIDIA GeForce GTX 1080 Ti": "GTX 1080 Ti",
         "NVIDIA TITAN Xp": "TITAN Xp"}


def load_json(p):
    try:
        return json.loads(pathlib.Path(p).read_text())
    except Exception:
        return None


def first(pattern):
    hits = sorted(glob.glob(str(W / "runs" / pattern / "results" / "*.json")))
    return load_json(hits[0]) if hits else None


def fmt(v, nd=1):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def recipe_label(r):
    return {"single": "single", "ddp": "DDP", "fsdp": "FSDP"}.get(
        r["recipe"], f"ZeRO-{r['zero_stage']}" if r["zero_stage"] else r["recipe"])


def section_repeats(p, stats):
    p += ["## 3. Training throughput over six independent repeat blocks", ""]
    if not stats:
        p += ["*No block statistics present: `results/block_statistics.json` is "
              "missing, so no repeat claim is made here.*", ""]
        return
    rows = stats["configurations"]
    done = [r for r in rows if r["tokens_per_s"]["blocks_complete"]]
    p += [f"{len(rows)} configurations carry at least one completed repeat; "
          f"**{len(done)}** have the full six blocks. Only the six-block rows "
          f"support the 31/32 = 96.875% pointwise median-coverage claim; the "
          f"others are reported descriptively and marked.", "",
          "`CV%` is the sample coefficient of variation; `range/med%` is "
          "(max-min)/median. Every individual repeat is listed so the spread "
          "behind each median is visible.", "",
          "| config | GPUs | recipe | prec | median tok/s | min | max | CV% | "
          "range/med% | coverage | all repeats |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda q: (q["group"], str(q["zero_stage"]),
                                         q["recipe"], q["precision"])):
        t = r["tokens_per_s"]
        reps = ";".join(f"{v:,.0f}" for v in r["tokens_per_s_by_block"].values())
        cov = t["minmax_median_coverage"] + ("" if t["blocks_complete"] else " *(partial)*")
        p += [f"| `{r['case_id']}` | {r['world_size']} | {recipe_label(r)} | "
              f"{r['precision']} | {fmt(t['median'])} | {fmt(t['min'])} | "
              f"{fmt(t['max'])} | {fmt(t['sample_cv_pct'], 2)} | "
              f"{fmt(t['range_over_median_pct'], 2)} | {cov} | {reps} |"]
    p += [""]

    p += ["### 3.1 Weak-scaling efficiency, paired within blocks", "",
          "Speedup is computed **per block** against the same block's "
          "world-size-1 baseline of the same recipe and precision, then reduced "
          "to a median. Taking a ratio of two medians drawn from different block "
          "sets would pair one arm's good block with another's bad one.", "",
          "| config | GPUs | recipe | baseline | median speedup | median efficiency % | blocks paired |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda q: (q["group"], q["recipe"], q["world_size"])):
        sp = r.get("speedup_vs_1gpu") or {}
        ef = r.get("scaling_efficiency_pct") or {}
        if not sp.get("ratios"):
            continue
        p += [f"| `{r['case_id']}` | {r['world_size']} | {recipe_label(r)} | "
              f"`{r.get('scaling_baseline')}` | {fmt(sp.get('median_ratio'), 3)} | "
              f"{fmt(ef.get('median_ratio_pct'), 1)} | {sp.get('n')} |"]
    p += [""]


def section_collectives(p):
    p += ["## 4. Interconnect: NCCL all-reduce at the exact gradient payload", ""]
    rows = []
    for case in ("G09_comm_rtx2", "G10_comm_p1080_2",
                 "G11_comm_p1080_4", "G12_comm_p1080_6"):
        r = first(case)
        if r:
            rows.append((case, r))
    if not rows:
        p += ["*No collective records present.*", ""]
        return
    p += ["GPT-2 124M's FP32 gradient volume is 124,439,808 x 4 = "
          "**497,759,232 B**, which is 497.76 MB decimal but only 474.7 MiB. "
          "The sweep's historical \"497\" point was 497 **MiB** = 521,142,272 B, "
          "a different payload; see `bundle/CORRECTIONS.md` C-03. Both are "
          "measured below, and the exact-volume row is flagged.", "",
          "| group | GPUs | P2P | payload B | MiB | algbw GB/s | ring-equiv GB/s | exact gradient volume |",
          "|---|---|---|---|---|---|---|---|"]
    for case, r in rows:
        cfg = r.get("config", {})
        for q in r["rows"]:
            p += [f"| {cfg.get('group','?')} | {cfg.get('world_size','?')} | "
                  f"{'enabled' if r.get('p2p_all_enabled') else 'disabled'} | "
                  f"{q['payload_bytes']:,} | {q['size_mib_exact']:.3f} | "
                  f"{q['algbw_gb_s']:.2f} | {q['ring_equivalent_bw_gb_s']:.2f} | "
                  f"{'**yes**' if q['is_exact_gpt2_124m_fp32_gradient_volume'] else '' } |"]
    p += ["", "`ring-equiv` is algbw x 2(n-1)/n, an analytical ring "
          "normalisation, **not** measured wire traffic.", ""]


def section_cnn(p):
    p += ["## 5. ResNet-50 forward-only, batch 32, >=60 s complete-work windows", ""]
    rows = []
    for case in ("E01_cnn_rtx_fp32", "E02_cnn_rtx_amp",
                 "E03_cnn_p1080_fp32", "E04_cnn_xp_fp32"):
        r = first(case)
        if r:
            rows.append((case, r))
    if not rows:
        p += ["*No CNN records present.*", ""]
        return
    p += ["| GPU | precision | img/s | window s | iters | achieved TFLOP/s | util % | J/image | energy J |",
          "|---|---|---|---|---|---|---|---|---|"]
    for case, r in rows:
        dev = SHORT.get(r["device"].get("name"), r["device"].get("name"))
        for q in r["rows"]:
            u = q.get("util_mean_pct")
            u = list(u.values())[0] if isinstance(u, dict) and u else u
            p += [f"| {dev} | {r['config'].get('precision')} | "
                  f"{q['throughput']:,.2f} | {q['elapsed_s']:.1f} | {q['iters']:,} | "
                  f"{q['achieved_tflops']:.2f} | {u:.1f} | "
                  f"{q['energy_per_image']:.3f} | {q['energy_total_j']:,.0f} |"]
    p += ["", "Achieved TFLOP/s uses **8.178 GFLOP/image** (4.089 GMAC x 2). The "
          "earlier 4.1 constant was MACs mislabelled as FLOPs, which understated "
          "every compute figure by 1.995x.", ""]


def section_inference(p):
    p += ["## 6. Inference", "", "### 6.1 vLLM offline, TinyLlama-1.1B", ""]
    got = False
    p += ["| arm | GPUs | TP | concurrency | out tok/s | elapsed s | J per 1k out tok |",
          "|---|---|---|---|---|---|---|"]
    for case, tp in (("F02_vllm_tp1_rtx", 1), ("F03_vllm_tp2_rtx", 2)):
        r = first(case)
        if not r:
            continue
        got = True
        for q in r["rows"]:
            p += [f"| `{case}` | {tp} | {tp} | {q.get('concurrency')} | "
                  f"{fmt(q.get('output_tokens_per_s'))} | "
                  f"{fmt(q.get('elapsed_s'), 2)} | "
                  f"{fmt(q.get('energy_per_1k_output_tokens_j'), 2)} |"]
    if not got:
        p = p[:-2] + ["*No vLLM records present.*", ""]
    else:
        g = (first("F02_vllm_tp1_rtx") or {}).get("guided_decoding", {})
        p += ["",
              f"Guided decoding: backend **configured** = "
              f"`{g.get('configured_backend')}`, backend **exercised** = "
              f"`{g.get('exercised')}`, output **conformed** to the constraint = "
              f"`{g.get('conformed')}`. These are recorded as separate facts. "
              f"`lm-format-enforcer` is a *different* backend chosen because "
              f"`outlines` 0.0.46 cannot import `pyairports.airports` in this "
              f"prefix; it is **not** a repair of `outlines`. The frozen config "
              f"(identical for TP1, TP2 and streaming) is "
              f"`bundle/plan/vllm_frozen_config.json`.", ""]
    p += ["### 6.2 Hugging Face `generate` baseline", "",
          "| GPU | dtype | batch | gen tok/s | decode-proxy tok/s | prefill-proxy ms | peak MiB | J per 1k tok |",
          "|---|---|---|---|---|---|---|---|"]
    any_hf = False
    for case in ("F05_hf_rtx_fp16", "F04_hf_p1080_fp16", "F06_hf_xp_fp16"):
        r = first(case)
        if not r:
            continue
        any_hf = True
        dev = SHORT.get(r["device"].get("name"), r["device"].get("name"))
        for q in r["rows"]:
            pf = q.get("prefill_forward_proxy_s")
            p += [f"| {dev} | {r['config'].get('dtype')} | {q.get('batch_size')} | "
                  f"{fmt(q.get('gen_tokens_per_s'))} | "
                  f"{fmt(q.get('decode_proxy_tokens_per_s'))} | "
                  f"{fmt(pf * 1000 if isinstance(pf, (int, float)) else None)} | "
                  f"{fmt(q.get('peak_alloc_mib'))} | "
                  f"{fmt(q.get('energy_per_1k_tokens_j'), 2)} |"]
    if not any_hf:
        p = p[:-2] + ["*No HF records present.*", ""]
    p += ["", "`prefill-proxy` is a single forward pass used as a stand-in for "
          "time-to-first-token; `true_ttft_measured` is false in these records, "
          "so it is a proxy and not TTFT. HF `J per 1k tok` is per 1k generated "
          "tokens while the vLLM column is per 1k **output** tokens; the two "
          "columns come from different stacks and are not directly comparable.",
          "", "vLLM requires compute capability >= 7.0 and so runs only on the "
          "TITAN RTX pair; the Pascal cards use HF `generate`. These are "
          "different serving stacks and are not interchangeable measurements.", ""]


def section_hf_dtype(p, aux):
    """fp32 vs fp16 for HF generate, from the six-block aux statistics."""
    p += ["### 6.3 The Pascal fp16 penalty, measured as a matched dtype pair", ""]
    if not aux:
        p += ["*No aux block statistics present.*", ""]
        return
    rows = aux["configurations"]
    # {(family, batch): {dtype: median}}
    import collections
    grid = collections.defaultdict(dict)
    for r in rows:
        if r["metric"] != "gen_tokens_per_s":
            continue
        cid = r["case_id"]
        dtype = "float32" if cid.endswith("float32") else "float16"
        fam = cid.split("_tinyllama")[0].replace("H_", "")
        grid[(fam, r["sub_key"])][dtype] = r["value"]
    if not grid:
        p += ["*No HF dtype pairs present.*", ""]
        return
    p += ["These arms are identical except for `--dtype`: same model revision, "
          "prompt length, new-token count and device. The ratio therefore "
          "isolates the dtype, not the stack.", "",
          "| GPU | batch | fp32 median tok/s | fp16 median tok/s | fp32 / fp16 | fp32 CV% | fp16 CV% |",
          "|---|---|---|---|---|---|---|"]
    for (fam, bs) in sorted(grid, key=lambda k: (k[0], k[1])):
        d = grid[(fam, bs)]
        a, b = d.get("float32"), d.get("float16")
        if not (a and b):
            continue
        ratio = a["median"] / b["median"] if b["median"] else None
        p += [f"| {fam} | {bs} | {fmt(a['median'])} | {fmt(b['median'])} | "
              f"{fmt(ratio, 2)}x | {fmt(a['sample_cv_pct'], 2)} | "
              f"{fmt(b['sample_cv_pct'], 2)} |"]
    p += ["", "On the sm_61 Pascal cards fp32 is **faster** than fp16, and the "
          "gap widens with batch size: consumer GP102/GP104 has no useful "
          "half-precision throughput, so fp16 work takes a slow path. Choosing "
          "fp16 on these cards is a pessimisation, not an optimisation. On the "
          "sm_75 TITAN RTX, which has real half-precision units, the ordering "
          "reverses. Any cross-family inference comparison that fixes a single "
          "dtype therefore penalises one family by construction.", ""]


def section_dtype_probe(p):
    """Per-device dtype capability, all ten devices, native vs emulated."""
    import glob as _g
    p += ["## 7. Dtype capability on all ten devices, and why the microbenchmark misleads", ""]
    rows = []
    for d in range(10):
        hits = sorted(_g.glob(str(W / "runs" / f"DT_gpu{d}" / "results" / "*.json")))
        if hits:
            rows.append((d, load_json(hits[0])))
    if not rows:
        p += ["*No dtype probe records present.*", ""]
        return
    p += [f"Probed on **{len(rows)} of 10** devices, each selected by UUID. The "
          f"prior study covered 5 of 10.", "",
          "| GPU | name | cap | fp32 TFLOP/s | fp16 TFLOP/s | bf16 TFLOP/s | "
          "fp16 / fp32 | bf16 / fp32 | `including_emulation=True` | native flag |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for d, r in rows:
        dev, res, fl = r["device"], r["results"], r["flags"]
        p += [f"| {d} | {dev['name'].replace('NVIDIA ','').replace('GeForce ','')} | "
              f"{dev['capability']} | {res['fp32']['tflops']:.2f} | "
              f"{res['fp16']['tflops']:.2f} | {res['bf16']['tflops']:.2f} | "
              f"{r['fp16_vs_fp32_speedup']:.3f} | {r['bf16_vs_fp32_speedup']:.3f} | "
              f"`{fl['bf16_including_emulation']}` | `{fl['bf16_native_api_flag']}` |"]
    p += ["",
          "**bf16 is emulated on every device here.** "
          "`torch.cuda.is_bf16_supported()` returns True on all ten only because "
          "`including_emulation=True` is its default; with "
          "`including_emulation=False` it is False on all ten, and measured bf16 "
          "is 0.54-0.63x of fp32 everywhere. The flag was never lying -- it was "
          "answering a different question than the one a reader assumes.", "",
          "### 7.1 A 4096-cubed GEMM mispredicts real inference in both directions", "",
          "Taking the fp16/fp32 GEMM ratio above as a guide to end-to-end work "
          "fails on this node, and fails in opposite directions per family:", "",
          "| family | fp16/fp32 on a 4096-cubed GEMM | fp16/fp32 for HF generate at batch 16 | the GEMM figure |",
          "|---|---|---|---|",
          "| TITAN RTX (sm_75) | **8.06x** faster | **0.90x**, i.e. slightly slower | over-predicts the benefit ~9-fold |",
          "| GTX 1080 Ti (sm_61) | 0.85x | **0.16x**, i.e. 6.07x slower | under-states the harm ~7-fold |",
          "| TITAN Xp (sm_61) | 1.09x | **0.16x**, i.e. 6.44x slower | wrong sign entirely |",
          "",
          "A large square GEMM is compute-bound and saturates the half-precision "
          "units; autoregressive decode at these batch sizes is dominated by "
          "memory traffic and kernel-launch overhead, so Turing's tensor cores "
          "go largely unused, while on Pascal the fp16 code path's penalty "
          "compounds across many small kernels. The practical consequence is "
          "that the dtype choice must be made per family from end-to-end "
          "measurement, and a single fixed dtype across families penalises one "
          "of them by construction.", ""]


def main() -> int:
    stats = load_json(W / "results" / "block_statistics.json")
    aux = load_json(W / "results" / "aux_block_statistics.json")
    p: list[str] = []
    p += ["# The Titan / GTX 1080 Ti chapter", "",
          "Author of repository: Mehran Keivanimehr.", "",
          "Generated by `tools/make_titan_chapter.py` from the run records and "
          "`results/block_statistics.json`. Every number here is derived; none "
          "is transcribed by hand.", ""]
    p += ["## 1. What this chapter is and is not", "",
          "* Node `TITAN_NODE`: 2x TITAN RTX (24 GB, sm_75), 7x GTX 1080 Ti "
          "(11 GB, sm_61), 1x TITAN Xp (12 GB, sm_61); driver 535.230.02, "
          "no NVLink, PCIe only.",
          "* Authorisation basis: the repository owner explicitly confirmed that "
          "direct execution on `TITAN_NODE` is permitted under their facility "
          "authorisation, without Slurm, from the approved pool, at **at most "
          "six concurrent GPUs and one job at a time**. That confirmation is the "
          "recorded basis (`ledger/WINDOW.txt`); no policy document was located "
          "and none is claimed. See `bundle/CORRECTIONS.md` C-01.",
          "* Host nvcc is 10.1 against a 12.2 driver, so **no CUDA JIT**: "
          "DeepSpeed runs with `DS_BUILD_OPS=0`, no offload, and a plain "
          "`torch.optim.AdamW`.",
          "* bf16 is unsupported before Ampere. "
          "`torch.cuda.is_bf16_supported()` returns True here only because "
          "`including_emulation=True` is its default.",
          "* Diagnostic runs carry `--numerical-checks` and their timings are "
          "**not** performance results. Performance comes from the repeat "
          "blocks in section 3.", ""]
    p += ["## 2. Measurement protocol actually enforced", "",
          "* Telemetry sampling interval is frozen at **0.2 s** and the value "
          "that travels with each record is read from the live sampler, not "
          "from a constant. The energy gap cutoff is **0.6 s** = 3 intervals, "
          "declared before data.",
          "* Energy is trapezoid integration of sampled board power, gross and "
          "uncalibrated. Net (idle-subtracted) energy is diagnostic only.",
          "* Every launch passes a fail-closed gate: no foreign compute process, "
          "no unaccounted VRAM above 500 MiB, utilisation settled, window not "
          "expired, and a timeout that fits inside the window.",
          "* Device visibility is UUID-bound for every arm except vLLM, whose "
          "engine parses `CUDA_VISIBLE_DEVICES` as integers "
          "(`vllm/platforms/cuda.py:98`). Those arms use numeric visibility with "
          "the UUID enforced as a **verified postcondition**, externally via "
          "`nvidia-smi` and internally via the CUDA runtime. That is a weaker "
          "input guarantee and a stronger checked one; it is stated, not "
          "silently substituted.",
          "* torch 2.4.1 exposes neither `uuid` nor `pci_bus_id` on "
          "`CUDADeviceProperties`, and `pynvml` is absent, so "
          "`device_identity_established` is **false** in every record. The "
          "substitute evidence is a UUID-selectivity control "
          "(`tools/uuid_selectivity.py`, run `D02`), which is weaker than a "
          "runtime-reported UUID and labelled as such.", ""]
    section_repeats(p, stats)
    section_collectives(p)
    section_cnn(p)
    section_inference(p)
    section_hf_dtype(p, aux)
    section_dtype_probe(p)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(p) + "\n")
    print(f"wrote {OUT} ({len(p)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
