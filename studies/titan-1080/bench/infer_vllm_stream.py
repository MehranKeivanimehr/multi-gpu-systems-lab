"""vLLM loopback streaming latency: true TTFT and inter-token latency.

The offline `LLM.generate` path cannot measure time-to-first-token, because it
returns only completed sequences; the HF arm records a single forward pass as a
`prefill_forward_proxy` and marks `true_ttft_measured` false. This arm measures
TTFT for real: one vLLM OpenAI server is started on loopback, and a streaming
client timestamps the arrival of every token chunk.

Structure and limits:
  * ONE controller owns ONE engine. The server is this process's only child and
    is terminated by pid at the end; no broad kills, no process-name matching.
  * The engine configuration is the frozen one shared with the TP1 and TP2 arms
    (bundle/plan/vllm_frozen_config.json), including the guided-decoding
    backend, so the three arms differ only in transport and concurrency.
  * 100 requests at concurrency 4 support a MEDIAN. They do not support a tail
    percentile, and none is reported.
  * Server startup is outside the measured window and outside the energy window.
"""
from __future__ import annotations

import argparse, asyncio, json, os, signal, socket, statistics, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SAMPLE_INTERVAL_S, GpuSampler, phys_devices, stats_to_dict, write_result  # noqa: E402


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def wait_ready(base: str, proc: subprocess.Popen, timeout: float) -> float:
    import aiohttp
    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as sess:
        while time.perf_counter() - t0 < timeout:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early with code {proc.returncode}")
            try:
                async with sess.get(f"{base}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        return time.perf_counter() - t0
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"server not ready within {timeout}s")


async def one_request(sess, base: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Timestamp every streamed chunk. TTFT is the first chunk carrying text."""
    import aiohttp
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True}
    t0 = time.perf_counter()
    ttft = None
    arrivals: list[float] = []
    text = []
    async with sess.post(f"{base}/v1/completions", json=body,
                         timeout=aiohttp.ClientTimeout(total=600)) as r:
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            chunk = (obj.get("choices") or [{}])[0].get("text") or ""
            if not chunk:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            arrivals.append(now)
            text.append(chunk)
    total = time.perf_counter() - t0
    itl = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return {"ttft_s": ttft, "total_s": total, "n_chunks": len(arrivals),
            "chars": sum(len(x) for x in text),
            "inter_token_latencies_s": itl,
            "median_itl_s": statistics.median(itl) if itl else None}


async def drive(base: str, model: str, prompt: str, n: int, conc: int,
                max_tokens: int) -> list[dict]:
    import aiohttp
    sem = asyncio.Semaphore(conc)
    out: list[dict] = []
    async with aiohttp.ClientSession() as sess:
        async def run_one():
            async with sem:
                out.append(await one_request(sess, base, model, prompt, max_tokens))
        await asyncio.gather(*[run_one() for _ in range(n)])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--revision", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--guided-decoding-backend", default="lm-format-enforcer",
                   choices=["lm-format-enforcer", "outlines"])
    p.add_argument("--startup-timeout", type=float, default=900.0)
    a = p.parse_args()

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--host", "127.0.0.1", "--port", str(port),
            "--model", a.model, "--revision", a.revision,
            "--tokenizer-revision", a.revision,
            "--dtype", "float16", "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", str(a.gpu_mem_util),
            "--max-model-len", str(a.max_model_len),
            "--guided-decoding-backend", a.guided_decoding_backend,
            "--disable-custom-all-reduce", "--seed", "1234",
            "--disable-log-requests"]

    log = open(os.path.join(os.environ.get("BENCH_RESULTS_DIR", "."),
                            "vllm_server.log"), "w")
    # start_new_session keeps the engine's own group separate, so the only thing
    # this process ever signals is the pid it started.
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    rec: dict = {"status": "error", "error": None}
    try:
        startup_s = asyncio.run(wait_ready(base, proc, a.startup_timeout))
        # Deterministic prompt of the declared token length, built from the
        # tokenizer so prompt_len is tokens and not characters.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
        ids = [(i * 7919) % 30000 + 5 for i in range(a.prompt_len)]
        prompt = tok.decode(ids)
        prompt_tokens = len(tok(prompt)["input_ids"])

        # one untimed warm request, then the measured window
        asyncio.run(drive(base, a.model, prompt, 1, 1, 8))
        sampler = GpuSampler(phys_devices(), interval=SAMPLE_INTERVAL_S)
        with sampler:
            t0 = sampler.mark_start()
            rows = asyncio.run(drive(base, a.model, prompt, a.requests,
                                     a.concurrency, a.output_len))
            wall = sampler.mark_stop() - t0
        tel = sampler.stats()

        ttfts = sorted(r["ttft_s"] for r in rows if r["ttft_s"] is not None)
        itls = [x for r in rows for x in r["inter_token_latencies_s"]]
        chunks = sum(r["n_chunks"] for r in rows)
        energy = tel.energy_total_j
        rec = {
            "status": "ok",
            "config": {
                "engine": "vllm_loopback_streaming",
                "vllm_version": __import__("vllm").__version__,
                "model": a.model, "model_revision": a.revision,
                "tokenizer_revision": a.revision, "dtype": "float16",
                "tensor_parallel": 1, "group": a.group, "label": a.label,
                "requests": a.requests, "concurrency": a.concurrency,
                "prompt_len_requested": a.prompt_len,
                "prompt_len_tokens_actual": prompt_tokens,
                "output_len": a.output_len,
                "gpu_memory_utilization": a.gpu_mem_util,
                "max_model_len": a.max_model_len,
                "enable_prefix_caching": False,
                "disable_custom_all_reduce": True,
                "guided_decoding_backend_configured": a.guided_decoding_backend,
                "guided_backend_exercised_in_this_arm": False,
                "guided_note": "this arm streams UNCONSTRAINED completions; the "
                               "backend is configured identically to the TP1/TP2 "
                               "arms but is not exercised here",
                "protocol_version": "scientific-v2",
                "transport": "loopback HTTP, one controller per engine",
                "controller_pid": os.getpid(), "engine_pid": proc.pid,
                "server_port": port,
            },
            "startup_s_outside_measurement": round(startup_s, 2),
            "perf": {
                "wall_time_s": round(wall, 4),
                "requests_per_s": round(a.requests / wall, 3),
                "streamed_chunks": chunks,
                "chunks_per_s": round(chunks / wall, 2),
                "true_ttft_measured": True,
                "ttft_median_s": statistics.median(ttfts) if ttfts else None,
                "ttft_min_s": ttfts[0] if ttfts else None,
                "ttft_max_s": ttfts[-1] if ttfts else None,
                "itl_median_s": statistics.median(itls) if itls else None,
                "itl_n": len(itls),
                "tail_percentiles_reported": False,
                "tail_note": f"{a.requests} requests support a median; a tail "
                             f"percentile is NOT claimed from this sample size",
            },
            "energy": {
                "energy_total_j": energy,
                "energy_per_1k_streamed_chunks_j":
                    round(energy / chunks * 1000, 3) if chunks else None,
                "scope": "gross sampled board energy over the measured window only",
            },
            "telemetry": stats_to_dict(tel),
            "rows": rows,
        }
    except Exception as exc:                      # noqa: BLE001 - recorded
        rec = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        # Terminate exactly the pid we started, then its group, then give up
        # loudly. Never a name-based or user-wide kill.
        try:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=30)
        except (ProcessLookupError, PermissionError) as exc:
            print(f"  note: engine already gone ({exc})", flush=True)
        log.close()

    if rec["status"] == "ok":
        pf = rec["perf"]
        print(f"[ok] streaming {a.group} {a.requests} req @ conc {a.concurrency}: "
              f"TTFT median {pf['ttft_median_s']*1000:.1f} ms, "
              f"ITL median {pf['itl_median_s']*1000:.1f} ms, "
              f"{pf['requests_per_s']:.2f} req/s, "
              f"{rec['energy']['energy_total_j']:.0f} J", flush=True)
    else:
        print(f"[error] {rec['error']}", flush=True)
    path = write_result(rec, f"S_stream_{a.label}")
    print(f"  -> {path}", flush=True)
    return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
