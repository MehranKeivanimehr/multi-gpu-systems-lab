"""
vLLM inference benchmark -- TITAN RTX only.

vLLM requires compute capability >= 7.0. The GTX 1080 Ti and TITAN Xp are 6.1,
so vLLM cannot run on 8 of this cluster's 10 GPUs at all; those GPUs are
covered by infer_hf.py instead. This script therefore only ever targets the
two TITAN RTX (sm_75), at TP=1 and TP=2.

Turing specifics:
  * dtype must be float16. bf16 needs sm_80+, and the dtype probe measured bf16
    at 0.63x of fp32 on this GPU anyway.
  * FlashAttention needs sm_80+, so vLLM selects the XFormers backend here.
  * gpu_memory_utilization is set to the same 0.85 cap the rest of the suite
    uses, so vLLM's KV-cache pool cannot crowd the node.
  * custom all-reduce is disabled explicitly for TP=2: CUDA P2P is unavailable
    between these two cards (measured), and vLLM's custom all-reduce kernel
    assumes peer access.

Quantization is not used: TinyLlama-1.1B in fp16 is ~2.2 GB against 24 GB of
VRAM, so there is no capacity pressure to relieve, and AWQ/Marlin kernels would
add a confound rather than information.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    SAMPLE_INTERVAL_S,  # noqa: E402
    GpuSampler,
    guard_devices,
    phys_devices,
    stats_to_dict,
    wait_for_idle,
    write_result,
)


def guided_probe(llm, backend: str) -> dict:
    """Establish whether the CONFIGURED guided backend was actually EXERCISED.

    Configuring a backend and exercising it are different facts, and a run that
    reports only the first invites the reader to assume the second. One guided
    request is issued against a trivial regex; the verdict records whether the
    engine accepted the guided parameters and whether the output conformed.
    Failure here never alters the unguided performance rows -- it is reported
    beside them.
    """
    import re as _re
    out = {"configured_backend": backend, "exercised": False, "conformed": None,
           "output": None, "error": None,
           "note": "a configured backend is not an exercised backend; these are "
                   "recorded separately and never merged"}
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        pattern = r"(red|green|blue)"
        try:
            gd = GuidedDecodingParams(regex=pattern, backend=backend)
            sp = SamplingParams(temperature=0.0, max_tokens=8, guided_decoding=gd)
        except (ImportError, TypeError):
            # Older surface: guided options are passed alongside the request.
            sp = SamplingParams(temperature=0.0, max_tokens=8)
            out["error"] = "GuidedDecodingParams unavailable on this vllm surface"
            return out
        res = llm.generate(["Pick one colour: "], sp)
        text = res[0].outputs[0].text if res and res[0].outputs else ""
        out["exercised"] = True
        out["output"] = text
        out["conformed"] = bool(_re.fullmatch(pattern, text.strip()))
    except Exception as exc:                      # noqa: BLE001 - recorded
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--revision", required=True, help="exact model and tokenizer commit")
    p.add_argument("--group", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--concurrency", default="1,4,16,64")
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument(
        "--guided-decoding-backend", default="lm-format-enforcer",
        choices=["lm-format-enforcer", "outlines"],
        help="engine-level guided-decoding backend. Default is lm-format-enforcer "
             "because outlines 0.0.46 cannot import pyairports.airports in this "
             "prefix. This is a different backend, NOT a repair of outlines.")
    p.add_argument(
        "--guided-probe", action="store_true",
        help="additionally issue one guided request to establish whether the "
             "configured backend was actually exercised")
    args = p.parse_args()

    guard_devices(phys_devices())
    wait_for_idle(phys_devices())

    import torch
    from vllm import LLM, SamplingParams

    props = torch.cuda.get_device_properties(0)
    if props.major < 7:
        print(
            f"  SKIP: {props.name} is capability {props.major}.{props.minor}; "
            f"vLLM requires >= 7.0"
        )
        write_result(
            {
                "config": {"engine": "vllm", "group": args.group, "tp": args.tp},
                "device": {"name": props.name,
                           "capability": f"{props.major}.{props.minor}"},
                "status": "unsupported_architecture",
                "reason": "vLLM requires compute capability >= 7.0",
            },
            f"F_vllm_{args.label}_tp{args.tp}",
        )
        return 0

    t_load = time.perf_counter()
    llm = LLM(
        model=args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        enable_prefix_caching=False,
        dtype="float16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        disable_custom_all_reduce=True,  # no CUDA P2P between these two cards
        enforce_eager=False,
        seed=1234,
        guided_decoding_backend=args.guided_decoding_backend,
    )
    load_s = time.perf_counter() - t_load

    tok = llm.get_tokenizer()
    vocab = min(getattr(tok, "vocab_size", 30000), 30000)
    # Fixed-length synthetic prompts so prefill cost is identical everywhere.
    base_ids = [((i * 7919) % (vocab - 100)) + 10 for i in range(args.prompt_len)]

    rows = []
    for conc in [int(c) for c in args.concurrency.split(",")]:
        prompts = [{"prompt_token_ids": list(base_ids)} for _ in range(conc)]
        sp = SamplingParams(
            max_tokens=args.max_new_tokens,
            min_tokens=args.max_new_tokens,
            temperature=0.0,
            ignore_eos=True,
        )
        # warmup at this concurrency
        llm.generate(prompts, SamplingParams(max_tokens=8, ignore_eos=True),
                     use_tqdm=False)

        sampler = GpuSampler(phys_devices(), interval=SAMPLE_INTERVAL_S)
        idle = sampler.measure_idle(1.0)
        with sampler:
            t0 = sampler.mark_start()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            elapsed = sampler.mark_stop() - t0
        st = sampler.stats()

        out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        in_tokens = conc * args.prompt_len
        idle_w = sum(idle.values())
        e_gross = st.energy_total_j
        e_net = e_gross - idle_w * st.duration_s

        row = {
            "concurrency": conc,
            "status": "ok",
            "prompt_len": args.prompt_len,
            "max_new_tokens": args.max_new_tokens,
            "elapsed_s": round(elapsed, 4),
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "output_tokens_per_s": round(out_tokens / elapsed, 2),
            "total_tokens_per_s": round((in_tokens + out_tokens) / elapsed, 2),
            "batch_completion_duration_s": round(elapsed, 4),
            "true_request_latency_measured": False,
            "energy_total_j": round(e_gross, 2),
            "energy_per_1k_output_tokens_j": round(e_gross / out_tokens * 1000, 3),
            "energy_per_1k_output_tokens_net_j": round(e_net / out_tokens * 1000, 3),
            "power_mean_w": st.power_mean_w,
            "util_mean_pct": st.util_mean_pct,
            "smi_mem_max_mib": st.mem_max_mib,
            "telemetry": stats_to_dict(st),
        }
        rows.append(row)
        print(
            f"  conc={conc:<4} {row['output_tokens_per_s']:>9.1f} out tok/s  "
            f"{elapsed:>7.2f} s  "
            f"{row['energy_per_1k_output_tokens_j']:>7.2f} J/ktok",
            flush=True,
        )

    record = {
        "config": {
            "engine": "vllm",
            "vllm_version": __import__("vllm").__version__,
            "model": args.model,
            "model_revision": args.revision,
            "tokenizer_revision": args.revision,
            "enable_prefix_caching": False,
            "prompt_token_ids": base_ids,
            "protocol_version": "scientific-v2",
            "dtype": "float16",
            "tensor_parallel": args.tp,
            "group": args.group,
            "label": args.label,
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_model_len": args.max_model_len,
            "disable_custom_all_reduce": True,
            # Configured, which is a weaker fact than exercised. See
            # guided_decoding below for whether it actually ran.
            "guided_decoding_backend_configured": args.guided_decoding_backend,
            "guided_decoding_backend_is_not_outlines_repair":
                "outlines 0.0.46 cannot import pyairports.airports in this "
                "prefix; lm-format-enforcer is a DIFFERENT backend selected in "
                "its place, not a fix for outlines",
        },
        "device": {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_mem_mib": round(props.total_memory / 1024**2, 1),
        },
        "model_load_s": round(load_s, 2),
        "guided_decoding": (guided_probe(llm, args.guided_decoding_backend)
                            if args.guided_probe else
                            {"configured_backend": args.guided_decoding_backend,
                             "exercised": False, "conformed": None,
                             "error": "probe not requested for this arm"}),
        "rows": rows,
    }
    path = write_result(record, f"F_vllm_{args.label}_tp{args.tp}")
    print(f"  -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
