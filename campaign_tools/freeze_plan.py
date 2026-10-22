"""Extend the 39-job plan to the frozen 50-configuration matrix, 6 blocks."""
import json, sys
from collections import Counter
prior, out = sys.argv[1], sys.argv[2]
p = json.load(open(prior))
rows = p["configurations"]
by_id = {r["case_id"]: r for r in rows}
UU = {r["device_indices"][0]: r["device_uuids"][0] for r in rows if len(r["device_indices"]) == 1}
NM = {r["device_indices"][0]: r["device_names"][0] for r in rows if len(r["device_indices"]) == 1}
GPT2_REV = by_id["T_p1080_1_single_fp32"]["model_revision"]
PREQ = by_id["T_p1080_1_single_fp32"]["numerical_prerequisites"]
PREQ_DS = by_id["T_p1080_2_zero1_fp32"]["numerical_prerequisites"]

def baseline(group, family, dev, recipe, dtype):
    ds = "deepspeed" if recipe.startswith("zero") else recipe
    return {
      "case_id": f"T_{group}_{recipe}_{dtype}", "stage": "stage2_performance",
      "family": family, "group": group, "topology_worst_link": "N/A",
      "device_indices": [dev], "device_uuids": [UU[dev]], "device_names": [NM[dev]],
      "world_size": 1, "recipe": recipe, "dtype": dtype,
      "model": "gpt2-124m", "model_revision": GPT2_REV,
      "input_policy": "synthetic token IDs; shared model seed; rank-distinct data streams (seed+rank)",
      "workload": {"seq_len": 512, "micro_batch_per_gpu": 4, "global_batch": 4,
                   "scaling": "weak (per-rank work fixed); N=1 baseline"},
      "warmup_steps": 5, "measured_steps": 30,
      "memory": {"memory_fraction": 0.85, "grad_checkpointing": False},
      "per_case_timeout_s": 900,
      "expected_artifacts": by_id["T_p1080_1_single_fp32"]["expected_artifacts"],
      "numerical_prerequisites": (PREQ_DS if recipe.startswith("zero") else PREQ),
      "planned_independent_repeats": 6,
      "repeats_justification": "6 fresh-process blocks; min/max then has 31/32 pointwise median coverage",
      "outcome": "not_run", "authorization": by_id["T_p1080_1_single_fp32"]["authorization"],
      "inventory_provenance": by_id["T_p1080_1_single_fp32"]["inventory_provenance"],
      "baseline_role": f"same-wrapper N=1 baseline for {recipe} {dtype} on {family}",
      "notes": ("SINGLETON FSDP uses no-shard behaviour at world_size 1: the "
                "effective implementation is disclosed and no within-wrapper "
                "efficiency is invented for it"
                if recipe == "fsdp" else
                "singleton wrapper baseline; if the mode is unsupported at N=1 no "
                "within-wrapper efficiency is invented"),
    }

added = []
for r in ("ddp", "fsdp", "zero1", "zero2"):
    added.append(baseline("p1080_1", "GTX 1080 Ti", 1, r, "fp32"))
    added.append(baseline("rtx_1", "TITAN RTX", 0, r, "fp32"))
for r in ("ddp", "zero1", "zero2"):
    added.append(baseline("rtx_1", "TITAN RTX", 0, r, "amp_fp16"))
assert len(added) == 11, len(added)
rows = rows + added
for r in rows:
    r["planned_independent_repeats"] = 6
    r.setdefault("repeats_justification", "")
    r["repeats_justification"] = ("6 fresh-process blocks; [min,max] then has "
                                  "31/32 = 96.875% pointwise median coverage for "
                                  "6 IID continuous observations")

plan = dict(p)
plan["status"] = "FROZEN PROPOSAL - NOT AUTHORIZED, NOT RUN"
plan["blocks"] = 6
plan["configurations"] = rows          # the extended list MUST be written back
plan["configuration_count"] = len(rows)
plan["total_planned_job_launches"] = len(rows) * 6
plan["unit_note"] = ("total_planned_job_launches counts JOB LAUNCHES, not scalar "
                     "observations. A sweep job contains several arms (e.g. HF "
                     "batches 1/4/16); ranks within one job and requests within "
                     "one session are NOT independent repeats.")
plan["superseded_estimate"] = {"prior": "39 configurations / 195 runs",
                               "status": "SUPERSEDED by this frozen 50/300 matrix"}
plan["frozen_defaults"] = {
  "training": {"model": "gpt2-124m", "seq_len": 512, "micro_batch_per_gpu": 4,
    "accumulation_steps": 1, "warmup_attempts": 5, "measured_attempts": 30,
    "optimizer": "torch.optim.AdamW lr=1e-4 (betas/eps/weight_decay at torch defaults)",
    "gradient_clipping": "native paths: none; DeepSpeed config sets gradient_clipping=1.0 "
                         "(a recorded recipe difference, not claimed equivalent)",
    "dropout": "GPT2Config defaults (resid/embd/attn 0.1); RNG controlled and replayed "
               "for the matched reference",
    "precisions": "FP32 everywhere; AMP-fp16 retained on TITAN RTX only",
    "scaling": "global batch = 4N at world size N => WEAK scaling, not fixed-global-batch strong scaling",
    "speedup_def": "T(N)/T(1)", "efficiency_def": "T(N)/(N*T(1))",
    "baseline_rule": "each recipe uses its OWN same-wrapper N=1 baseline, measured afresh in the same blocked campaign",
    "quality_claims": "none; no equal-quality or time-to-quality claim follows"},
  "cnn": {"batch_size": 32, "input": "3x224x224", "arms": 4,
    "measured_window_s_min": 60, "work_convention": "~8.2 GFLOP/image at FMA=2",
    "work_label": "ANALYTICAL work, not occupancy; AMP carries no percentage of FP32 peak",
    "energy_denominator": "actual complete-work interval, NOT an assumed exactly-60-second denominator"},
  "inference": {"model_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "prompt_len": 256, "output_tokens_exact": 128, "greedy": True,
    "corpus": "bundle/inputs/prompt_corpus.json",
    "corpus_sha256": "f2d11005cbce3e5419be77a3c3b05eff7a0b3e56b791e0d3c13327d2ac611350",
    "prompt_ids_sha256": "ab65f4a304b009015fc7d1a05d0f0ded328a771b77723e03df6114d02e228266",
    "enable_prefix_caching": False, "enforce_eager": False,
    "disable_custom_all_reduce": True, "max_model_len": 1024,
    "gpu_memory_utilization": 0.85,
    "vllm_launch_shape": "ONE parent per case, local multiprocessing, not Ray",
    "throughput_def": "total output tokens / matching end-to-end generation interval, "
                      "NOT reciprocal median latency"},
  "collectives": {"payloads_mib": [4, 16, 64, 256, 497],
    "payload_bytes": [m * 1024 * 1024 for m in (4, 16, 64, 256, 497)],
    "gpt2_124m_fp32_gradient_bytes": 124439808 * 4,
    "note_497": "497 MiB = 521142272 B is ~4.7% LARGER than the 497759232 B gradient volume",
    "warmups": 5, "measured_iters": 20,
    "bandwidth_label": "ring-equivalent NORMALISATION, not measured wire traffic; no topology causality"},
  "streaming": {"server": "one fresh loopback-only server per block",
    "requests": 100, "concurrency": 4, "prompt_len": 256, "output_tokens": 64,
    "reporting": "session variation + descriptive request summaries only; no tail "
                 "percentiles, no internal GPU TTFT, no chunk-gap-as-ITL"},
  "timing": {"exclude": "initialization and warmup",
    "sync": "CUDA synchronize at both measurement boundaries",
    "distributed_throughput": "total completed work over the COMMON job interval; "
                              "never summed rank rates nor the fastest rank; per-rank "
                              "timings persisted with their relation to that interval",
    "counting": "completed tokens/images/updates counted explicitly",
    "sampling_interval_s": 0.2,
    "energy_gap_cutoff_s": 0.6,
    "gap_rule": "3 x the CONFIGURED nominal interval, declared before data; never an observed mean",
    "diagnostics": "timed observers declared; audit-instrumented throughput labelled; "
                   "no zero-overhead claim without evidence"},
  "statistics": {"estimand": "median throughput per measurement arm across 6 sessions",
    "report": "every point, median, sample CV, range/median",
    "interval": "[min,max] over 6 IID continuous observations = 31/32 pointwise median coverage",
    "not": "not simultaneous intervals; not a global significance ranking",
    "comparisons": "matched-block ratios, same median method",
    "precision_goal": "endpoints within +/-5% of the median",
    "extension": "no automatic extra blocks beyond six",
    "partial": "n<6 receives no six-block confidence claim",
    "exclusions": "prespecified validity only; a valid slow run is never excluded for being slow"},
}
json.dump(plan, open(out, "w"), indent=1)
print(f"configurations: {plan['configuration_count']}  (added {len(added)} baselines)")
print(f"blocks: 6   planned job launches: {plan['total_planned_job_launches']}")
print("added baseline case_ids:")
for a in added: print("   ", a["case_id"])
print("by world size:", dict(sorted(Counter(r['world_size'] for r in rows).items())))
