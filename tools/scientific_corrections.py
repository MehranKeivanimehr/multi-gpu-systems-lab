#!/usr/bin/env python3
"""Derive corrected interpretations without editing a single historical record."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from .build_report import csv_text, load_json, read_csv
    from .scientific_metrics import cnn_compute, decode_proxy, dispersion, resnet50_conv_linear_macs
except ImportError:
    from build_report import csv_text, load_json, read_csv
    from scientific_metrics import cnn_compute, decode_proxy, dispersion, resnet50_conv_linear_macs

ROOT = Path(__file__).resolve().parents[1]


def artifacts(root=ROOT):
    output = {}
    cnn = []
    for row in read_csv(root / "studies/titan-1080/results/cnn_resnet50.csv"):
        if row["status"] != "ok":
            continue
        metrics = cnn_compute(float(row["throughput"]), row["precision"], float(row["theoretical_fp32_tflops"]))
        cnn.append({key: row[key] for key in ("gpu", "precision", "batch_size", "throughput")}
                   | metrics | {"source": "studies/titan-1080/results/cnn_resnet50.csv",
                                "energy_status": "historical_partial_sample_window_not_validated"})
    output["results/corrections/titan_cnn.csv"] = csv_text(cnn, list(cnn[0]))
    inference = []
    for row in read_csv(root / "studies/titan-1080/results/inference.csv"):
        if row["status"] != "ok":
            continue
        hf = row["engine"] == "hf_transformers_generate"
        prefill = float(row["ttft_ms"]) / 1000 if hf and row["ttft_ms"] else None
        inference.append({key: row[key] for key in ("engine", "gpu", "dtype", "requests", "output_tokens_per_s", "elapsed_s")}
                         | {"prefill_forward_proxy_s": prefill,
                            "decode_proxy_tokens_s": decode_proxy(int(row["output_tokens"]), int(row["requests"]),
                                                                  float(row["elapsed_s"]), prefill) if hf else None,
                            "true_request_ttft_available": False,
                            "source": "studies/titan-1080/results/raw/" + row["file"]})
    output["results/corrections/titan_inference.csv"] = csv_text(inference, list(inference[0]))
    grouped = defaultdict(list)
    sources = defaultdict(list)
    for path in sorted((root / "studies/a6000/results/raw/train_scaling").glob("*/result.json")):
        data = load_json(path)
        if data.get("status") != "ok":
            continue
        spec = data["spec"]
        key = (spec["model"], spec["framework"], spec["n_gpus"], spec.get("variant", ""))
        metric = "throughput_samples_s" if spec["model"] == "resnet50" else "throughput_tokens_s"
        grouped[key].append(data[metric])
        sources[key].append(path.relative_to(root).as_posix())
    repeats = [dict(zip(("model", "framework", "n_gpus", "variant"), key)) | dispersion(values)
               | {"sources": ";".join(sources[key])} for key, values in sorted(grouped.items())]
    output["results/corrections/a6000_repeats.csv"] = csv_text(repeats, list(repeats[0]))
    kv = []
    for path in sorted((root / "studies/a6000/results/raw/vllm_scaling").glob("llama*tp1*/worker_*.log")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
        if match:
            capacity = int(match[1].replace(",", ""))
            spec = load_json(path.parent / "spec.json")
            length = spec["input_len"] + spec["output_len"]
            kv.append({"source": path.relative_to(root).as_posix(), "capacity_tokens": capacity,
                       "actual_request_tokens": length, "token_capacity_over_request_length": capacity / length,
                       "capacity_for_256_requests_sufficient_by_token_count": capacity >= 256 * length,
                       "observed_scheduler_occupancy_available": False})
    if not kv:
        raise ValueError("Missing source KV logs")
    output["results/corrections/a6000_kv_capacity.csv"] = csv_text(kv, list(kv[0]))
    proof = {"schema_version": 1, "historical_records_modified": False,
             "gpu_reruns_performed": False, "cnn_rows": len(cnn), "inference_rows": len(inference),
             "training_repeat_groups": len(repeats),
             "resnet50_conv_linear_macs": resnet50_conv_linear_macs(),
             "resnet50_conv_linear_flops_fma2": 2 * resnet50_conv_linear_macs(),
             "approximation_used_for_correction_gflops": 8.2,
             "approximation_scope": "conv/linear dominant work, not complete hardware operation count",
             "max_sample_cv_pct": max(row["sample_cv_pct"] for row in repeats),
             "max_full_range_over_median_pct": max(row["full_range_over_median_pct"] for row in repeats),
             "full_range_over_one_percent_groups": sum(row["full_range_over_median_pct"] > 1 for row in repeats)}
    proof["kv_log_records_checked"] = len(kv)
    output["results/corrections/proof.json"] = json.dumps(proof, indent=2) + "\n"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for relative, content in artifacts().items():
        path = ROOT / relative
        encoded = content.encode("utf-8")
        if args.check:
            if not path.exists() or path.read_bytes() != encoded:
                raise SystemExit("Stale correction artifact: " + relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
    print("Scientific correction artifacts verified." if args.check else "Scientific correction artifacts generated.")


if __name__ == "__main__":
    main()
