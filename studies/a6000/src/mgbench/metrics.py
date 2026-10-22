"""Pure metric math (no GPU needed, unit-tested)."""
from __future__ import annotations

import statistics

# Dense BF16 tensor-core peak from the NVIDIA RTX A6000 datasheet (309.7 TFLOPS is the sparse figure).
PEAK_BF16_TFLOPS = {"NVIDIA RTX A6000": 154.8}
GIB = 1024**3


def step_stats(times_s: list[float]) -> dict:
    xs = sorted(times_s)
    n = len(xs)

    def pct(p):
        if n == 1:
            return xs[0]
        k = (n - 1) * p
        lo, hi = int(k), min(int(k) + 1, n - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {
        "n": n,
        "median_ms": 1e3 * statistics.median(xs),
        "mean_ms": 1e3 * statistics.fmean(xs),
        "std_ms": 1e3 * (statistics.stdev(xs) if n > 1 else 0.0),
        "p10_ms": 1e3 * pct(0.10),
        "p90_ms": 1e3 * pct(0.90),
        "min_ms": 1e3 * xs[0],
        "max_ms": 1e3 * xs[-1],
    }


def lm_matmul_params(hf_cfg: dict) -> int:
    """Parameters that take part in matmuls (all Linear layers incl. lm_head; excludes embeddings and norms)."""
    h = hf_cfg["hidden_size"]
    nh = hf_cfg["num_attention_heads"]
    nkv = hf_cfg.get("num_key_value_heads", nh)
    hd = hf_cfg.get("head_dim") or h // nh
    inter = hf_cfg["intermediate_size"]
    layers = hf_cfg["num_hidden_layers"]
    attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h
    mlp = 3 * h * inter
    return layers * (attn + mlp) + h * hf_cfg["vocab_size"]


def lm_total_params(hf_cfg: dict) -> int:
    """Total parameter count (embeddings counted once when tied). Biases and norms included approximately."""
    h = hf_cfg["hidden_size"]
    nh = hf_cfg["num_attention_heads"]
    nkv = hf_cfg.get("num_key_value_heads", nh)
    hd = hf_cfg.get("head_dim") or h // nh
    layers = hf_cfg["num_hidden_layers"]
    v = hf_cfg["vocab_size"]
    per_layer = (h * nh * hd + 2 * h * nkv * hd + nh * hd * h) + 3 * h * hf_cfg["intermediate_size"] + 2 * h
    if hf_cfg.get("attention_bias", False) or "Qwen2" in str(hf_cfg.get("architectures", "")):
        per_layer += nh * hd + 2 * nkv * hd  # Qwen2 has q/k/v biases
    emb = v * h
    head = 0 if hf_cfg.get("tie_word_embeddings", False) else v * h
    return layers * per_layer + emb + head + h


def lm_train_flops_per_token(hf_cfg: dict, seq_len: int) -> float:
    """Model FLOPs per trained token: 6*N_matmul + 12*L*d*T (PaLM appendix B convention, full causal attention).

    Activation-checkpoint recomputation is not counted (that would be hardware FLOPs, not model FLOPs).
    """
    return 6.0 * lm_matmul_params(hf_cfg) + 12.0 * hf_cfg["num_hidden_layers"] * hf_cfg["hidden_size"] * seq_len


def mfu(flops_per_sample: float, samples_per_s: float, n_gpus: int, gpu_name: str) -> float | None:
    peak = PEAK_BF16_TFLOPS.get(gpu_name)
    if not peak or not flops_per_sample:
        return None
    return flops_per_sample * samples_per_s / (n_gpus * peak * 1e12)


def scaling_efficiency(throughput_n: float, throughput_base: float, n: int, n_base: int = 1) -> float:
    """Weak/strong scaling efficiency relative to a baseline at n_base GPUs: T_n / ((n/n_base) * T_base)."""
    return throughput_n / ((n / n_base) * throughput_base)


def bus_bandwidth_allreduce(size_bytes: float, seconds: float, n: int) -> tuple[float, float]:
    """(algorithm bandwidth, bus bandwidth) in GB/s, nccl-tests convention: busbw = algbw * 2(n-1)/n."""
    algbw = size_bytes / seconds / 1e9
    return algbw, algbw * 2 * (n - 1) / n
