# A100 offline reproduction: one-token batch duration is a proxy, NOT request TTFT.
# No corrected physical-energy claim; A100 scientific-v2 host validation pending.
import json, os, sys, time, random, statistics, threading, datetime
import pynvml, torch, vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

RESULTS_DIR = sys.argv[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
IN_LEN, OUT_LEN = 128, 256
NUM_PROMPTS = [1, 8, 32, 128, 512]
GPU_UTIL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
OUT = f"{RESULTS_DIR}/infer_vllm.jsonl"

pynvml.nvmlInit()
_H = pynvml.nvmlDeviceGetHandleByIndex(0)

class GPUSampler:
    def __init__(self, interval=0.1):
        self.interval, self.util, self.power, self.mem = interval, [], [], []
        self._stop = threading.Event()
    def _run(self):
        while not self._stop.is_set():
            self.util.append(pynvml.nvmlDeviceGetUtilizationRates(_H).gpu)
            self.power.append(pynvml.nvmlDeviceGetPowerUsage(_H) / 1000)
            self.mem.append(pynvml.nvmlDeviceGetMemoryInfo(_H).used / 2**30)
            time.sleep(self.interval)
    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self._t.join()
    def stats(self):
        avg = lambda x: round(sum(x) / len(x), 1) if x else None
        return {"gpu_util_avg": avg(self.util), "gpu_util_max": max(self.util, default=None),
                "power_avg_w": avg(self.power), "power_max_w": round(max(self.power, default=0), 1),
                "nvml_mem_max_gb": round(max(self.mem, default=0), 2)}

def key(n): return f"vllm|qwen2.5-1.5b-instruct|bf16|n{n}|in{IN_LEN}|out{OUT_LEN}"

def main():
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["key"] for l in open(OUT) if l.strip()}
    todo = [n for n in NUM_PROMPTS if key(n) not in done]
    if not todo:
        print("all vLLM configs already done"); return

    t0 = time.perf_counter()
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=GPU_UTIL, max_model_len=1024,
              enable_prefix_caching=False, seed=0)
    load_s = time.perf_counter() - t0

    rng = random.Random(0)
    def prompts(n):   # distinct random-token prompts of exactly IN_LEN tokens
        return [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 150000) for _ in range(IN_LEN)])
                for _ in range(n)]
    sp_full = SamplingParams(temperature=0.0, max_tokens=OUT_LEN, min_tokens=OUT_LEN, ignore_eos=True)
    sp_one = SamplingParams(temperature=0.0, max_tokens=1)

    llm.generate(prompts(8), sp_full, use_tqdm=False)          # warmup

    for n in todo:
        ps = prompts(n)
        ttfts = []
        for _ in range(3):
            t = time.perf_counter(); llm.generate(ps, sp_one, use_tqdm=False)
            ttfts.append(time.perf_counter() - t)
        ttft = statistics.median(ttfts)
        times = []
        with GPUSampler() as s:
            for _ in range(2):
                t = time.perf_counter(); outs = llm.generate(ps, sp_full, use_tqdm=False)
                times.append(time.perf_counter() - t)
        assert all(len(o.outputs[0].token_ids) == OUT_LEN for o in outs)
        t, gpu = statistics.median(times), s.stats()
        out_tok_s = n * OUT_LEN / t
        row = {"key": key(n), "gpu": torch.cuda.get_device_name(0), "engine": "vllm",
               "vllm": vllm.__version__, "torch": torch.__version__,
               "ts": datetime.datetime.now().isoformat(timespec="seconds"),
               "model": "qwen2.5-1.5b-instruct", "task": "inference", "precision": "bf16",
               "batch": n, "input_len": IN_LEN, "output_len": OUT_LEN, "decoding": "greedy(ignore_eos)",
               "gpu_memory_utilization": GPU_UTIL, "prefix_caching": False, "load_s": round(load_s, 1),
               "status": "ok", "one_token_batch_completion_proxy_ms": round(1000 * ttft, 1), "e2e_s": round(t, 2),
               "protocol_version": "a100-proxy-label-correction-not-validated",
               "one_token_call_times_s": ttfts, "full_batch_call_times_s": times,
               "true_request_ttft_measured": False,
               "output_tokens_per_s": round(out_tok_s, 1),
               "total_tokens_per_s": round(n * (IN_LEN + OUT_LEN) / t, 1),
               "per_seq_decode_proxy_tok_s": round((OUT_LEN - 1) / (t - ttft), 1) if t > ttft else None, **gpu,
               "tokens_per_joule": None,
               "energy_status": "not_validated; no aligned integrated energy recorded"}
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno())
        print(f"n={n:<4} one-token-batch proxy {row['one_token_batch_completion_proxy_ms']} ms  "
              f"{row['output_tokens_per_s']} out tok/s  "
              f"({row['per_seq_decode_proxy_tok_s']} proxy tok/s/seq)  {gpu['power_avg_w']} W")

if __name__ == "__main__":
    main()
