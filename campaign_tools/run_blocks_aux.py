"""RETIRED driver: run_blocks_aux.py

This driver invoked tools/gpu_attempt.sh, bypassing the audited Python launcher
(studies/titan-1080/bench/launch.py). The shell path was shown -- with a MOCKED
nvidia-smi, no device access -- to LAUNCH when the device query exited 9, when it
returned no output, and when it returned an UNREQUESTED row (widening
CUDA_VISIBLE_DEVICES to a device that was never requested). It also held no
device reservation, revalidated only utilisation during settling, and never
confirmed ownership-scoped descendant cleanup.

Use tools/run_blocks_routed.py, which routes every training, auxiliary and
engine adapter through launch.run_case. This module refuses to run.
"""
import sys as _retired_sys

print("REFUSED: run_blocks_aux.py is retired; it bypassed the audited launcher.",
      file=_retired_sys.stderr)
print("         Use tools/run_blocks_routed.py (routes through launch.run_case).",
      file=_retired_sys.stderr)
_retired_sys.exit(95)

_ORIGINAL_SOURCE_RETAINED_BELOW = """
\"\"\"Repeat blocks for the non-training plan recipes: CNN, HF, vLLM, collectives.

Same discipline as the training runner: fresh process and fresh directory per
repeat, randomised interleaved order with a recorded seed, cutoff re-checked
before every launch, existing directories never overwritten.

The vLLM arms need their own interpreter prefix and numeric device visibility
(the engine parses CUDA_VISIBLE_DEVICES as integers), so they carry the UUID as
a verified postcondition through tools/vllm_arm.py. That substitution is
recorded per run in uuid_binding_receipt.json.
\"\"\"
from __future__ import annotations

import argparse, json, os, pathlib, random, subprocess, sys, time

W = pathlib.Path(os.environ.get("TITAN_CAMPAIGN_ROOT",
    str(pathlib.Path(__file__).resolve().parent.parent)))
T = W / "src/multigpu-bench/studies/titan-1080"
BENCH = T / "bench"
VLLM_PY = W / "env/vllm-n3/bin/python"
REV = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
EXPIRY = 1792161707
CUTOFF = EXPIRY - 1800

AUX = ("cnn_forward_only", "hf_generate", "nccl_allreduce_fixed_payload",
       "vllm_offline_tp1", "vllm_offline_tp2", "vllm_loopback_streaming")


def spec(cfg: dict, case: str):
    \"\"\"-> (adapter, script, argv, env_overrides)\"\"\"
    r, wl = cfg["recipe"], cfg.get("workload") or {}
    if r == "cnn_forward_only":
        return ("script", BENCH / "bench_cnn.py", [
            "--group", cfg["group"], "--label", case,
            "--batches", str(wl.get("batch_size", 32)),
            "--target-secs", "60",
            "--precision", "amp_fp16" if cfg["dtype"] == "amp_fp16" else "fp32",
        ], {})
    if r == "hf_generate":
        return ("script", BENCH / "infer_hf.py", [
            "--revision", REV, "--group", cfg["group"], "--label", case,
            "--batches", ",".join(str(b) for b in wl.get("batch_sizes", [1, 4, 16])),
            "--prompt-len", str(wl.get("prompt_len", 256)),
            "--max-new-tokens", str(wl.get("max_new_tokens", 128)),
            "--dtype", cfg["dtype"],
        ], {})
    if r == "nccl_allreduce_fixed_payload":
        return ("script_dist", BENCH / "comm_bench.py", [
            "--group", cfg["group"], "--iters", str(wl.get("iters", 20)),
        ], {})
    if r in ("vllm_offline_tp1", "vllm_offline_tp2"):
        tp = 1 if r.endswith("tp1") else 2
        return ("script", W / "tools/vllm_arm.py", [
            str(BENCH / "infer_vllm.py"),
            "--revision", REV, "--group", cfg["group"], "--label", case,
            "--tp", str(tp),
            "--prompt-len", str(wl.get("prompt_len", 256)),
            "--max-new-tokens", str(wl.get("max_new_tokens", 128)),
            "--concurrency", ",".join(str(c) for c in wl.get("concurrency", [1, 4, 16, 64])),
            "--gpu-mem-util", "0.85", "--max-model-len", "1024",
            "--guided-decoding-backend", "lm-format-enforcer", "--guided-probe",
        ], {"BENCH_PYTHON": str(VLLM_PY), "BENCH_NUMERIC_VISIBILITY": "1"})
    if r == "vllm_loopback_streaming":
        return ("script", W / "tools/vllm_arm.py", [
            str(BENCH / "infer_vllm_stream.py"),
            "--revision", REV, "--group", cfg["group"], "--label", case,
            "--requests", str(wl.get("requests", 100)),
            "--concurrency", str(wl.get("concurrency", 4)),
            "--prompt-len", str(wl.get("prompt_len", 256)),
            "--output-len", str(wl.get("output_len", 64)),
            "--gpu-mem-util", "0.85", "--max-model-len", "1024",
            "--guided-decoding-backend", "lm-format-enforcer",
        ], {"BENCH_PYTHON": str(VLLM_PY), "BENCH_NUMERIC_VISIBILITY": "1"})
    raise ValueError(f"unhandled recipe {r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="1,2,3,4,5,6")
    ap.add_argument("--recipes", default=",".join(AUX))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan = json.loads((W / "bundle/plan/run_plan.json").read_text())
    want = set(a.recipes.split(","))
    cfgs = [c for c in plan["configurations"] if c["recipe"] in want]
    blocks = [int(b) for b in a.blocks.split(",")]
    ledger = W / "ledger" / "blocks_aux.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    print(f"configs={len(cfgs)} blocks={blocks} launches={len(cfgs)*len(blocks)}")
    launched = skipped = refused = failed = 0

    for b in blocks:
        order = list(cfgs)
        rng = random.Random(40261016 + b)
        rng.shuffle(order)
        print(f"\\n===== aux block {b}  seed={40261016 + b}  n={len(order)} =====", flush=True)
        for k, cfg in enumerate(order, 1):
            case = f"Q{b}_{cfg['case_id']}"
            if (W / "runs" / case).exists():
                skipped += 1
                continue
            now, tmo = int(time.time()), int(cfg["per_case_timeout_s"])
            if now >= CUTOFF or now + tmo + 120 >= EXPIRY:
                print(f"  STOP: launch cutoff reached before {case}", flush=True)
                with open(ledger, "a") as fh:
                    fh.write(json.dumps({"event": "cutoff_stop", "block": b,
                                         "case": case}, sort_keys=True) + "\\n")
                print(f"\\nlaunched={launched} skipped={skipped} "
                      f"refused={refused} failed={failed}")
                return 0
            adapter, script, extra, envov = spec(cfg, case)
            devs = ",".join(str(x) for x in cfg["device_indices"])
            argv = ["bash", str(W / "tools/gpu_attempt.sh"), case, devs,
                    str(tmo), adapter, str(script)] + extra
            if a.dry_run:
                print(f"  [{k}/{len(order)}] DRY {case} devs={devs} adapter={adapter}")
                continue
            env = dict(os.environ)
            for q in ("BENCH_PYTHON", "BENCH_NUMERIC_VISIBILITY"):
                env.pop(q, None)
            env.update(envov)
            t0 = time.time()
            subprocess.run(argv, capture_output=True, text=True, env=env)
            el = time.time() - t0
            rc_file = W / "runs" / case / "exit_code"
            rc = int(rc_file.read_text().strip()) if rc_file.exists() else None
            if rc == 0:
                launched += 1; tag = "ok"
            elif rc == 90:
                refused += 1; tag = "REFUSED"
            else:
                failed += 1; tag = f"rc={rc}"
            print(f"  [{k}/{len(order)}] {case:40s} {tag:9s} {el:5.0f}s", flush=True)
            with open(ledger, "a") as fh:
                fh.write(json.dumps({"event": "launch", "block": b, "case": case,
                                     "recipe": cfg["recipe"], "exit_code": rc,
                                     "elapsed_s": round(el, 1),
                                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                          time.gmtime())},
                                    sort_keys=True) + "\\n")

    print(f"\\nlaunched={launched} skipped={skipped} refused={refused} failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
