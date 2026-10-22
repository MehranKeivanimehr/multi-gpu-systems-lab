#!/usr/bin/env python3
"""Derive the clean Colab reproduction notebook from its historical source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "studies/a100/reproduce.ipynb"


def notebook() -> dict:
    original = json.loads((ROOT / "provenance/a100-original.ipynb").read_text(encoding="utf-8"))
    cells = []

    def add(source: str, kind="code"):
        cell = {"cell_type": kind, "id": f"a100-{len(cells):02d}", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("# A100 40 GB historical-protocol reproduction\n\nAuthor: Mehran Keivanimehr.\n\n"
        "SCIENTIFIC STATUS: this reproduces the historical protocol, including measurement limitations. "
        "It is not the corrected validation protocol. Read CURRENT_REPORT.md and the project scientific "
        "status register before interpreting training updates, TTFT, energy, capacity or MFU. "
        "New A100 experimental validation remains pending.\n\n"
        "Use an exclusive Colab A100 runtime. The recorded main stack was torch 2.11.0+cu130, "
        "Python 3.13.15, Transformers 5.18.0, Accelerate 1.15.0. Colab's base image and "
        "unrecorded transitive dependencies may differ. This notebook has not been GPU-rerun "
        "during repository preparation.\n\n"
        "Run in order. Outputs go to a new local directory. The Qwen dtype audit is retained. "
        "The vLLM setup downloads a separate environment and model weights. Historical "
        "cleanup, retry, account authentication, and remote upload cells are excluded.\n", "markdown")
    for index in range(8):
        source = "".join(original["cells"][index]["source"])
        if index == 1:
            source = source.replace("!pip install -q transformers accelerate nvidia-ml-py",
                                    "%pip install -q transformers==5.18.0 accelerate==1.15.0 nvidia-ml-py pandas tabulate")
            start = source.index("try:\n    from google.colab import drive")
            end = source.index("os.makedirs(RESULTS_DIR", start)
            source = source[:start] + "RESULTS_DIR = '/content/gpu_bench/a100_reproduction'\n" + source[end:]
        add(source)
    add("## Isolated vLLM inference\n\nThe recorded run used vLLM 0.31.0, torch 2.13.0+cu132, "
        "`gpu_memory_utilization=0.6`, and the sampler setting below. Package availability and "
        "future driver compatibility need validation on the target runtime.\n", "markdown")
    add("%pip install -q uv\n"
        "!uv venv /content/vllm-env --python 3.12\n"
        "!uv pip install --python /content/vllm-env/bin/python vllm==0.31.0 torch==2.13.0 nvidia-ml-py ninja --torch-backend=auto\n")
    add("%%writefile /content/vllm_bench.py\n" + (ROOT / "studies/a100/vllm_bench.py").read_text(encoding="utf-8"))
    add("cleanup()\n"
        "VENV_PY = '/content/vllm-env/bin/python'\n"
        "run_env = {**os.environ, 'PATH': '/content/vllm-env/bin:' + os.environ['PATH'],\n"
        "           'VLLM_USE_FLASHINFER_SAMPLER': '0'}\n"
        "log_path = f'{RESULTS_DIR}/vllm_log.txt'\n"
        "with open(log_path, 'w') as run_log:\n"
        "    result = subprocess.run([VENV_PY, '/content/vllm_bench.py', RESULTS_DIR, '0.6'],\n"
        "                            stdout=run_log, stderr=subprocess.STDOUT, env=run_env)\n"
        "print('vLLM exit code:', result.returncode, 'log:', log_path)\n"
        "if result.returncode:\n"
        "    raise RuntimeError(f'vLLM failed; inspect {log_path}')\n")
    add("# Rebuild tables from the new local JSONL records.\n"
        "names = ['train_gpt2', 'train_resnet50', 'train_qwen2.5-1.5b', 'infer_hf', 'infer_vllm']\n"
        "tables = []\n"
        "for name in names:\n"
        "    with open(f'{RESULTS_DIR}/{name}.jsonl') as stream:\n"
        "        records = [json.loads(line) for line in stream if line.strip()]\n"
        "    frame = pd.DataFrame(records)\n"
        "    frame.to_csv(f'{RESULTS_DIR}/{name}.csv', index=False)\n"
        "    tables.append(frame.assign(benchmark=name))\n"
        "pd.concat(tables, ignore_index=True).to_csv(f'{RESULTS_DIR}/all_results.csv', index=False)\n"
        "print('Reproduction outputs:', RESULTS_DIR, sorted(os.listdir(RESULTS_DIR)))\n")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = (json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    if args.check:
        if not TARGET.exists() or TARGET.read_bytes() != content:
            raise SystemExit("Clean notebook is stale; run python tools/prepare_notebook.py")
        print("Clean notebook is current.")
    else:
        TARGET.write_bytes(content)
        print("Wrote studies/a100/reproduce.ipynb")


if __name__ == "__main__":
    main()
