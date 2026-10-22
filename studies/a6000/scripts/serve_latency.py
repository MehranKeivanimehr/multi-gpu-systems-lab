#!/usr/bin/env python
"""Controlled loopback vLLM serving session for streaming-latency protocols (brought into the source tree from the
pilot's external wrapper `serve_latency_pilot.py`, stage 20261016s).

NOT AUTHORIZED TO RUN under the current approval. It refuses unless ALL of these hold:
  * an authorization file with scope "serving" for this candidate, this GPU index/UUID, unexpired and unrevoked
    (the repository launch_hold stays active; the authorization is the only exception);
  * exactly one GPU from the authorization, bound by UUID (CUDA_VISIBLE_DEVICES=<UUID>);
  * the same fail-closed gate as the training smoke (settings, telemetry validity, idle, host headroom, node power
    budget, MPS, identity), re-validated immediately before launch;
  * the server runs in its own systemd scope with CPU/memory limits and RuntimeMaxSec; cleanup is by that scope's
    cgroup only and must be confirmed empty (otherwise the hold/reservation is retained).

Usage (future, after a separate approval):
  serve_latency.py --authorization A --gpu 4 --out <new dir> --client tag:requests:concurrency:logprobs ...
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.request
from pathlib import Path

A6000 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A6000 / "src"))

from mgbench import launch_controls as lc  # noqa: E402
from mgbench import scopes, smoke  # noqa: E402
from mgbench.config import cluster_config, load_yaml  # noqa: E402

MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ALIAS = "qwen2.5-1.5b-instruct@989aa79"
VOCAB_UPPER = 151643
PROMPT_LEN, OUTPUT_LEN = 512, 256
SCOPE = "serving"


def preflight(auth: Path | None, gpu: int, pins: dict, cases_done: int = 0) -> list[str]:
    """Everything that must hold before a server may start. Pure apart from reading files/telemetry."""
    problems = smoke.settings_problems(cluster_config())
    problems += lc.authorization_problems(auth, scope=SCOPE, candidate_root=A6000.parents[1], gpus=[gpu],
                                          uuids=pins, n_cases_done=cases_done)
    return problems


def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorization", type=Path, default=None)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--client", action="append", required=True, help="tag:requests:concurrency:logprobs")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--startup-timeout", type=float, default=600)
    ap.add_argument("--total-timeout", type=float, default=1800)
    args = ap.parse_args(argv)
    smoke_cfg = load_yaml(A6000 / "configs/smoke_gpu4.yaml")
    pins = {int(k): v for k, v in smoke_cfg["gpu_uuids"].items()}
    problems = preflight(args.authorization, args.gpu, pins)
    if problems:
        print("NOT LAUNCHING: " + "; ".join(problems), file=sys.stderr)
        return 6
    # ---- below: only reachable with a valid serving authorization (none exists under the current approval)
    from mgbench.safety import host_problems, idle_problems, mps_daemon_problems, node_guard_problems

    idle = cluster_config()["idle"]
    gate = idle_problems([args.gpu], idle) + host_problems(idle) + node_guard_problems([args.gpu], idle) + \
        mps_daemon_problems() + preflight(args.authorization, args.gpu, pins)
    if gate:
        print("NOT LAUNCHING (gate): " + "; ".join(gate), file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    cmd = ["numactl", "--cpunodebind=1", "--preferred=1", sys.executable, "-s", "-m", "vllm.entrypoints.cli.main",
           "serve", MODEL_REPO, "--revision", REVISION, "--tokenizer-revision", REVISION, "--served-model-name", ALIAS,
           "--dtype", "bfloat16", "--host", "127.0.0.1", "--port", str(port),
           "--max-model-len", str(PROMPT_LEN + OUTPUT_LEN + 64), "--max-num-seqs", "256",
           "--gpu-memory-utilization", "0.85", "--no-enable-prefix-caching", "--seed", str(args.seed),
           "--generation-config", "vllm"]
    env = smoke.env_for_worker(__import__("os").environ, src=A6000 / "src", uuid=pins[args.gpu], index=args.gpu,
                               threads=8, cache=args.out / "cache", run_id=f"serve/{args.out.name}")
    unit = f"mgbench-serve-{args.out.name}"
    lim = smoke_cfg["limits"]
    record = {"unit": unit, "command": cmd, "port": port, "clients": []}
    with open(args.out / "server.log", "x") as lf:
        proc, handle = scopes.start_scope(unit, cmd, env=env, cwd=str(A6000), stdout=lf, stderr=lf,
                                          cpu_quota_pct=lim["worker_cpu_quota_pct"], memory_max=lim["worker_memory"],
                                          runtime_max_s=int(args.total_timeout), stop_timeout_s=60)
        record["identity"] = {"cgroup": handle.cgroup, "leader_pid": handle.leader_pid,
                              "leader_start_time": handle.leader_start_time, "error": handle.identity_error}
        try:
            t0 = time.time()
            while proc.poll() is None and time.time() - t0 < args.startup_timeout:
                try:
                    if http_get(base + "/health", timeout=5)[0] == 200:
                        break
                except Exception:  # noqa: BLE001
                    time.sleep(2)
            else:
                raise RuntimeError("server did not become healthy before the startup timeout")
            if proc.poll() is not None:
                raise RuntimeError(f"server exited during startup ({proc.returncode})")
            import subprocess
            for spec in args.client:
                tag, n, conc, lp = spec.split(":")
                remaining = args.total_timeout - (time.time() - t0) - 90
                if remaining < 30:
                    raise RuntimeError("no time left for further clients inside the server's RuntimeMaxSec")
                (args.out / f"metrics_before_{tag}.txt").write_text(http_get(base + "/metrics")[1])
                ccmd = [sys.executable, "-s", str(A6000.parents[1] / "validation/stream_latency.py"), "--base-url",
                        base, "--model", ALIAS, "--requests", n, "--concurrency", conc, "--prompt-len",
                        str(PROMPT_LEN), "--output-len", str(OUTPUT_LEN), "--vocab-upper", str(VOCAB_UPPER),
                        "--seed", str(args.seed), "--timeout", "300", "--logprobs", lp,
                        "--out", str(args.out / f"events_{tag}.jsonl")]
                res = subprocess.run(ccmd, capture_output=True, text=True, timeout=remaining,
                                     env={**env, "CUDA_VISIBLE_DEVICES": ""})
                (args.out / f"metrics_after_{tag}.txt").write_text(http_get(base + "/metrics")[1])
                (args.out / f"client_{tag}.log").write_text(res.stdout + res.stderr)
                record["clients"].append({"tag": tag, "command": ccmd, "exit_code": res.returncode})
        except Exception as exc:  # noqa: BLE001
            record["error"] = repr(exc)
        finally:
            record["cleanup"] = scopes.cleanup_scope(handle, grace_s=60)
    lc.create_exclusive(args.out / "session_record.json", json.loads(json.dumps(record, default=str)))
    return 0 if record["cleanup"]["confirmed"] else 3


if __name__ == "__main__":
    sys.exit(main())
