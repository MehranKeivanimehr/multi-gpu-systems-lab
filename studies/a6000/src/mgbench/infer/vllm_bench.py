"""vLLM offline throughput benchmark.

Parent mode (default): splits the run's GPUs into `dp` groups of `tp` GPUs, starts one worker process per group
(each with its own CUDA_VISIBLE_DEVICES), releases them together through a file barrier once every engine is
loaded and warmed up, samples NVML on all GPUs, and aggregates the results.

Worker mode (--worker): one vLLM engine (tensor-parallel over its GPUs) generating a fixed share of the prompts.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from ..config import framework_config, model_registry
from ..gpu_monitor import GpuSampler, aggregate_gpu_summary
from ..results import environment, executed_source_record, read_json, write_json

TOKEN_RANGE = (1000, 100000)  # ordinary-vocabulary ids valid for both Qwen2.5 (151k) and Llama 3.1 (128k)


def make_prompts(n: int, input_len: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(*TOKEN_RANGE) for _ in range(input_len)] for _ in range(n)]


def worker(spec: dict, rank: int, out: Path) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    entry = model_registry()[spec["model"]]
    fw = framework_config("vllm")
    prompts = make_prompts(spec["num_prompts"], spec["input_len"], spec["seed"])[rank::spec["dp"]]
    warm = make_prompts(spec["warmup_prompts"], spec["input_len"], spec["seed"] + 1)

    t_load = time.time()
    llm = LLM(
        model=entry["hf_repo"], revision=entry["hf_revision"], tensor_parallel_size=spec["tp"],
        dtype=fw["dtype"], gpu_memory_utilization=fw["gpu_memory_utilization"],
        max_model_len=spec["input_len"] + spec["output_len"] + 64, max_num_seqs=spec.get("max_num_seqs", fw["max_num_seqs"]),
        enable_prefix_caching=fw["enable_prefix_caching"], seed=spec["seed"],
    )
    load_s = time.time() - t_load

    def params(n_out):
        return SamplingParams(temperature=0.0, max_tokens=n_out, min_tokens=n_out,
                              ignore_eos=fw["ignore_eos"], detokenize=False)

    llm.generate([TokensPrompt(prompt_token_ids=p) for p in warm], params(32), use_tqdm=False)

    # Barrier: signal ready, wait for the parent's go.
    (out / f"ready_{rank}").touch()
    while not (out / "go").exists():
        time.sleep(0.05)

    t0 = time.perf_counter()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params(spec["output_len"]),
                        use_tqdm=False)
    t1 = time.perf_counter()
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    n_in = sum(len(o.prompt_token_ids) for o in outs)
    write_json(out / f"worker_{rank}.json", {
        "executed_source": executed_source_record(),
        "rank": rank, "t_start": t0, "t_end": t1, "load_s": load_s, "requests": len(outs),
        "input_tokens": n_in, "output_tokens": n_out,
        "timebase": "time.perf_counter, host-local",
    })
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001 - best effort; the runner also cleans up marked processes
        pass


def parent(spec: dict, out: Path, timeout_s: float) -> int:
    tp, dp, gpus = spec["tp"], spec["dp"], spec["gpus"]
    assert len(gpus) == tp * dp
    for f in out.glob("ready_*"):
        f.unlink()
    (out / "go").unlink(missing_ok=True)
    spec_path = out / "spec.json"

    sampler = GpuSampler(gpus).start()
    procs = []
    for r in range(dp):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus[r * tp:(r + 1) * tp]))
        log = open(out / f"worker_{r}.log", "w")
        procs.append((subprocess.Popen(
            [sys.executable, "-m", "mgbench.infer.vllm_bench", "--worker", str(r), "--spec", str(spec_path),
             "--out", str(out)], env=env, stdout=log, stderr=subprocess.STDOUT), log))

    deadline = time.time() + timeout_s
    while len(list(out.glob("ready_*"))) < dp:
        if any(p.poll() is not None for p, _ in procs):
            print("[mgbench] a vLLM worker exited before becoming ready", flush=True)
            break
        if time.time() > deadline:
            print("[mgbench] timeout waiting for vLLM workers", flush=True)
            break
        time.sleep(0.2)
    else:
        sampler.mark_start()
        (out / "go").touch()

    codes = []
    for p, log in procs:
        try:
            codes.append(p.wait(timeout=max(10, deadline - time.time())))
        except subprocess.TimeoutExpired:
            p.terminate()
            codes.append(p.wait(timeout=60))
        log.close()
    sampler.mark_stop()
    sampler.stop()

    workers = [read_json(f) for f in sorted(out.glob("worker_*.json"))]
    if len(workers) != dp or any(codes):
        write_json(out / "error_parent.json", {"status": "failed", "exit_codes": codes,
                                               "workers_reported": len(workers)})
        return 1
    # Throughput over the window in which the replicas actually generated (max end - min start).
    t0 = min(w["t_start"] for w in workers)
    t1 = max(w["t_end"] for w in workers)
    # Retrospectively select precisely the workers' generation makespan.
    # Sampling continues through setup/teardown for provenance, but those
    # readings must not contaminate generation utilization or mean power.
    sampler.t_start, sampler.t_stop = t0, t1
    window = t1 - t0
    n_out = sum(w["output_tokens"] for w in workers)
    n_in = sum(w["input_tokens"] for w in workers)
    n_req = sum(w["requests"] for w in workers)
    gpu = aggregate_gpu_summary(sampler.summary())
    write_json(out / "telemetry_raw.json", sampler.raw_record())
    result = {
        "status": "ok", "spec": spec, "framework_config": framework_config("vllm"),
        "protocol_version": "scientific-v2",
        "telemetry_window": {"start_s": t0, "stop_s": t1, "timebase": "time.perf_counter",
                             "scope": "union makespan of worker generation; no setup/teardown"},
        "model_entry": model_registry()[spec["model"]], "window_s": window,
        "requests": n_req, "input_tokens": n_in, "output_tokens": n_out,
        "throughput_requests_s": n_req / window,
        "throughput_output_tokens_s": n_out / window,
        "throughput_total_tokens_s": (n_in + n_out) / window,
        "throughput_output_tokens_s_per_gpu": n_out / window / len(gpus),
        "load_s_max": max(w["load_s"] for w in workers),
        "workers": workers, "gpu": gpu, "gpu_per_device": sampler.summary(), "env": environment(gpus),
        "executed_source": executed_source_record(),
        "telemetry_complete": set(sampler.summary()) == {str(g) for g in gpus},
    }
    write_json(out / "result.json", result)
    print(f"[mgbench] {spec['run_id']}: {result['throughput_output_tokens_s']:.0f} output tok/s, "
          f"{result['throughput_total_tokens_s']:.0f} total tok/s, {result['throughput_requests_s']:.2f} req/s, "
          f"window {window:.1f} s, util {gpu.get('gpu_util_mean_pct', float('nan')):.0f}%", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=1100)
    args = ap.parse_args()
    out = Path(args.out)
    spec = read_json(args.spec)
    if args.worker is not None:
        worker(spec, args.worker, out)
        os._exit(0)  # do not linger in interpreter teardown; GPU memory is released on exit
    sys.exit(parent(spec, out, args.timeout))


if __name__ == "__main__":
    main()
