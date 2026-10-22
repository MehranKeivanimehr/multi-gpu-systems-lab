#!/usr/bin/env python3
"""Build a deterministic artifact catalog and report using the standard library."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["campaign", "experiment", "run_id", "kind", "status", "model", "framework",
          "gpu", "n_gpus", "batch_per_gpu", "seq_len", "precision", "source"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def family(group: str) -> str:
    for prefix, name in (("p1080", "GTX 1080 Ti"), ("rtx", "TITAN RTX"), ("xp", "TITAN Xp")):
        if group.startswith(prefix):
            return name
    return ""


def row(**values) -> dict:
    return {field: values.get(field, "") for field in FIELDS}


def catalog(root: Path = ROOT) -> list[dict]:
    output = []
    raw = root / "studies/a6000/results/raw"
    for path in sorted(raw.glob("*/*/status.json")):
        status = load_json(path)
        spec = status.get("spec", {})
        result_path = path.parent / "result.json"
        result = load_json(result_path) if result_path.exists() else {}
        precision = result.get("trainer_info", {}).get("precision", "")
        if not precision:
            precision = spec.get("dtype", "bf16" if spec.get("kind") == "vllm" else "")
        output.append(row(campaign="a6000", experiment=path.parent.parent.name,
                          run_id=spec.get("run_id", path.parent.name), kind=spec.get("kind", ""),
                          status=status["status"], model=spec.get("model", ""),
                          framework=spec.get("framework", ""), gpu="RTX A6000",
                          n_gpus=spec.get("n_gpus", ""), batch_per_gpu=spec.get("micro_batch", ""),
                          seq_len=spec.get("seq_len", ""), precision=precision,
                          source=path.relative_to(root).as_posix()))
    for path in sorted((root / "studies/titan-1080/results/raw").glob("*.json")):
        data = load_json(path)
        config = data.get("config", {})
        name = path.stem
        if name.startswith("00_"):
            kind, status = "preflight", "cleared" if data.get("cleared") else "failed"
        elif name.startswith("01_"):
            kind, status = "dtype", "recorded"
        elif name.startswith("C_comm"):
            kind, status = "nccl", "recorded"
        elif name.startswith("B_") or name.startswith("F_"):
            kind = "cnn_inference" if name.startswith("B_") else "inference"
            statuses = {item.get("status", "unlabelled") for item in data.get("rows", [])}
            status = data.get("status") or ("rows_ok" if statuses == {"ok"} else "mixed_rows")
        else:
            kind, status = "train", data.get("status", "unlabelled")
        output.append(row(campaign="titan-1080", experiment=name.split("_")[0], run_id=name,
                          kind=kind, status=status, model=config.get("model", ""),
                          framework=config.get("mode", config.get("engine", "")),
                          gpu=data.get("device", {}).get("name") or family(config.get("group", "")),
                          n_gpus=config.get("world_size", config.get("tensor_parallel", "")),
                          batch_per_gpu=config.get("micro_batch_per_gpu", ""),
                          seq_len=config.get("seq_len", ""),
                          precision=config.get("precision", config.get("dtype", "")),
                          source=path.relative_to(root).as_posix()))
    for path in sorted((root / "studies/a100").glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            data = json.loads(line)
            output.append(row(campaign="a100", experiment=path.stem, run_id=data["key"],
                              kind=data.get("task", "inference"), status=data["status"],
                              model=data.get("model", ""), framework=data.get("engine", data.get("optimizer", "")),
                              gpu=data.get("gpu", "NVIDIA A100-SXM4-40GB"), n_gpus=1,
                              batch_per_gpu=data.get("batch", ""), seq_len=data.get("seq_len", ""),
                              precision=data.get("precision", ""),
                              source=f"{path.relative_to(root).as_posix()}#L{line_no}"))
    return sorted(output, key=lambda item: (item["campaign"], item["experiment"], item["run_id"]))


def csv_text(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def table(headers: list[str], rows: list[list]) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(clean(value) for value in values) + " |" for values in rows])


def fmt(value, digits: int = 1) -> str:
    parsed = number(value)
    return f"{parsed:,.{digits}f}" if parsed is not None else "unavailable"


def inventory(root: Path, rows: list[dict]) -> dict:
    campaigns = {}
    for campaign in ("a6000", "a100", "titan-1080"):
        selected = [item for item in rows if item["campaign"] == campaign]
        csv_paths = sorted((root / "studies" / campaign).rglob("*.csv"))
        campaigns[campaign] = {
            "catalog_entries": len(selected),
            "catalog_status_counts": dict(sorted(Counter(item["status"] for item in selected).items())),
            "aggregate_csv_rows": {path.relative_to(root).as_posix(): len(read_csv(path)) for path in csv_paths},
        }
    campaigns["a100"]["measurement_rows"] = len(read_csv(root / "studies/a100/all_results.csv"))
    campaigns["titan-1080"]["measurement_rows"] = {
        name: len(read_csv(root / f"studies/titan-1080/results/{name}.csv"))
        for name in ("llm_runs", "cnn_resnet50", "inference", "dtype_gemm", "comm_bandwidth_full")
    }
    return {"schema_version": 1, "catalog_unit": "source run/artifact; see docs/DATA.md",
            "campaigns": campaigns}


def protocol_rows() -> list[dict]:
    fields = ["campaign", "task", "model", "length_or_shape", "precision", "recipe", "comparison_scope"]
    values = [
        ["a6000", "train", "qwen2.5-1.5b DDP", "seq1024", "fp32 persistent params; bf16 autocast", "mb1; accumulation4; AdamW; fp32 gradient reduction", "within-recipe scaling; not precision-matched to ZeRO"],
        ["a6000", "train", "qwen2.5-1.5b FSDP2", "seq1024", "fp32 persistent shards; bf16 compute; fp32 reduction", "mb1; accumulation4; AdamW; no checkpointing", "within-recipe scaling; not precision-matched to ZeRO"],
        ["a6000", "train", "qwen2.5-1.5b ZeRO-1/2", "seq1024", "bf16 model; fp32 optimizer master; fp32 communication", "mb1; accumulation4; DeepSpeed bf16 mode", "complete-recipe comparison; not framework-only effect"],
        ["a6000", "train", "llama3.1-8b", "seq1024", "bf16; fp32/bf16 reduction variants", "mb2; checkpointing; FSDP2/ZeRO3", "campaign-local scaling and audit"],
        ["a6000", "train", "resnet50", "224x224", "bf16 autocast", "AdamW; batch128/GPU", "not Titan inference or A100 SGD"],
        ["a100", "train", "gpt2-124m", "seq1024", "fp32/tf32/fp16/bf16", "fused AdamW; batch8/16/32", "not Titan seq512"],
        ["a100", "train", "qwen2.5-1.5b", "seq512/1024", "bf16 and bf16-pure separated", "fused AdamW; batch/checkpoint sweep", "not A6000 accumulation recipe"],
        ["a100", "train", "resnet50", "224x224", "fp32/tf32/fp16/bf16", "SGD momentum; channels_last", "not Titan inference or A6000 AdamW"],
        ["titan-1080", "train", "gpt2-124m", "seq512", "fp32/amp_fp16", "AdamW; accumulation1", "campaign-local scaling"],
        ["titan-1080", "inference", "resnet50", "224x224", "fp32/amp_fp16", "forward-only batch sweep", "no supplied A6000 inference baseline"],
        ["a6000", "inference", "qwen2.5-1.5b / llama3.1-8b", "prompt512; output256", "bf16", "vLLM TP/DP; 1024 requests; original/audit separate", "campaign-local matched-concurrency audit"],
        ["a100", "inference", "qwen2.5-1.5b-instruct", "prompt128; output256", "bf16", "HF/vLLM; batch1/8/32/128/512", "campaign-local engine comparison"],
        ["titan-1080", "inference", "tinyllama-1.1b", "prompt256; output128", "float32/float16", "HF all families; vLLM TITAN RTX only", "campaign-local GPU/engine comparison"],
    ]
    return [dict(zip(fields, values_row)) for values_row in values]


def report(root: Path, inv: dict) -> str:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_showcase import combined_report
    return combined_report(root, inv)


def artifacts(root: Path = ROOT) -> dict[str, str]:
    rows = catalog(root)
    inv = inventory(root, rows)
    protocols = protocol_rows()
    return {"results/run_index.csv": csv_text(rows, FIELDS),
            "results/inventory.json": json.dumps(inv, indent=2, sort_keys=True) + "\n",
            "results/protocols.csv": csv_text(protocols, list(protocols[0])),
            "reports/REPORT.md": report(root, inv)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if any generated artifact is stale")
    args = parser.parse_args()
    stale = []
    for name, content in artifacts().items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_bytes() != content.encode("utf-8"):
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
            print(f"Wrote {name}")
    if stale:
        print("Stale artifacts: " + ", ".join(stale) + "; run python tools/build_report.py")
        return 1
    if args.check:
        print("Generated catalog and report are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
