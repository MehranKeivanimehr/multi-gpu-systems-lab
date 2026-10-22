"""
HuggingFace `generate` inference baseline -- the Pascal substitute for vLLM.

vLLM requires compute capability >= 7.0, so it cannot run on the GTX 1080 Ti or
TITAN Xp (both 6.1) at all. Rather than leave 8 of 10 GPUs without an inference
number, this measures the same model and the same prompt/decode shape through
plain transformers, so Pascal and Turing are at least on one common axis. The
report is explicit that HF-generate and vLLM are different serving stacks and
that the Pascal-vs-Turing inference comparison must be read on this path, not
against vLLM's continuous-batching numbers.

Both float32 and float16 are recorded by the supplied driver. The dtype GEMM
probe alone does not predict decode performance; compare the inference arms
at matching prompt lengths and batch sizes.

The standalone forward duration is a prefill proxy, NOT streamed TTFT.
Subtracting it from a separate generation call gives only a decode proxy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from common import (
    SAMPLE_INTERVAL_S,  # noqa: E402
    GpuSampler,
    apply_memory_cap,
    guard_devices,
    phys_devices,
    stats_to_dict,
    torch_env,
    wait_for_idle,
    write_result,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--revision", required=True, help="exact model/tokenizer commit, recorded with the run")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--group", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--memory-fraction", type=float, default=0.85)
    args = p.parse_args()

    guard_devices(phys_devices())
    wait_for_idle(phys_devices())
    apply_memory_cap(args.memory_fraction)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)
    dtype = getattr(torch, args.dtype)

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    n_params = sum(p_.numel() for p_ in model.parameters())
    rows = []

    for bs in [int(b) for b in args.batches.split(",")]:
        # Fixed synthetic prompt of exactly prompt_len tokens so prefill cost is
        # identical across GPUs and runs.
        # CPU generation makes the same IDs reproducible across GPU families.
        generator = torch.Generator(device="cpu").manual_seed(args.seed + bs)
        ids_cpu = torch.randint(0, model.config.vocab_size - 1, (bs, args.prompt_len), generator=generator)
        import hashlib
        input_sha256 = hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest()
        ids = ids_cpu.to(device)
        attn = torch.ones_like(ids)

        try:
            # warmup, including one real generate to build the KV cache path
            with torch.no_grad():
                model.generate(
                    input_ids=ids, attention_mask=attn, max_new_tokens=8,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

            # --- standalone forward/prefill proxy; not TTFT ---
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn)
            torch.cuda.synchronize(device)
            ttft = time.perf_counter() - t0
            del out

            # --- full generate ---
            sampler = GpuSampler(phys_devices(), interval=SAMPLE_INTERVAL_S)
            idle = sampler.measure_idle(1.0)
            with sampler:
                t0 = sampler.mark_start()
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=ids, attention_mask=attn,
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=args.max_new_tokens,
                        do_sample=False, pad_token_id=tok.pad_token_id,
                    )
                torch.cuda.synchronize(device)
                elapsed = sampler.mark_stop() - t0
            st = sampler.stats()
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:<3} OOM (recorded, stopping sweep)", flush=True)
            rows.append({"batch_size": bs, "status": "oom"})
            torch.cuda.empty_cache()
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"  bs={bs:<3} OOM (recorded, stopping sweep)", flush=True)
                rows.append({"batch_size": bs, "status": "oom"})
                torch.cuda.empty_cache()
                break
            raise

        new_tokens = int(gen.shape[1] - ids.shape[1]) * bs
        gen_tps = new_tokens / elapsed
        # Decode-only rate: subtract the prefill from the wall time.
        decode_tps = (new_tokens - bs) / (elapsed - ttft) if elapsed > ttft and new_tokens > bs else None
        idle_w = sum(idle.values())
        e_gross = st.energy_total_j
        e_net = e_gross - idle_w * st.duration_s

        row = {
            "batch_size": bs,
            "status": "ok",
            "prompt_len": args.prompt_len,
            "max_new_tokens": args.max_new_tokens,
            "prefill_forward_proxy_s": round(ttft, 5),
            "input_ids_sha256": input_sha256,
            "true_ttft_measured": False,
            "elapsed_s": round(elapsed, 4),
            "new_tokens": new_tokens,
            "gen_tokens_per_s": round(gen_tps, 2),
            "decode_proxy_tokens_per_s": round(decode_tps, 2) if decode_tps is not None else None,
            "peak_alloc_mib": round(
                torch.cuda.max_memory_allocated(device) / 1024**2, 1
            ),
            "energy_total_j": round(e_gross, 2),
            "energy_per_1k_tokens_j": round(e_gross / new_tokens * 1000, 3),
            "energy_per_1k_tokens_net_j": round(e_net / new_tokens * 1000, 3),
            "power_mean_w": st.power_mean_w,
            "util_mean_pct": st.util_mean_pct,
            "telemetry": stats_to_dict(st),
        }
        rows.append(row)
        print(
            f"  bs={bs:<3} prefill-proxy {ttft*1000:>8.1f} ms  "
            f"gen {gen_tps:>8.1f} tok/s  decode-proxy {str(decode_tps):>8} tok/s  "
            f"peak {row['peak_alloc_mib']:>7.1f} MiB  "
            f"{row['energy_per_1k_tokens_j']:>6.2f} J/ktok",
            flush=True,
        )
        del ids, attn, gen
        torch.cuda.empty_cache()

    record = {
        "config": {
            "engine": "hf_transformers_generate",
            "model": args.model,
            "model_revision": args.revision,
            "seed": args.seed,
            "protocol_version": "scientific-v2",
            "n_params": n_params,
            "dtype": args.dtype,
            "group": args.group,
            "label": args.label,
            "tensor_parallel": 1,
            "memory_fraction": args.memory_fraction,
        },
        "device": {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_mem_mib": round(props.total_memory / 1024**2, 1),
        },
        "rows": rows,
        "torch_env": torch_env(),
    }
    path = write_result(record, f"F_hf_{args.label}_{args.dtype}")
    print(f"  -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
