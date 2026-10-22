"""Pure, device-free construction of the RETAINED recipe's model/input/optimizer.

Why this module exists
----------------------
The earlier diagnostic built `GPT2Config(n_positions=512)` (124,046,592
parameters) and drew inputs inline. The retained benchmark
(`studies/titan-1080/bench/train_llm.py`) builds `GPT2Config()` **defaults**
and draws inputs through `synthetic_batch`. Those are different models and
different input paths, so evidence from the old diagnostic could not bind to the
retained runs.

This module reproduces the retained construction with the device factored out,
so the identical code path can be exercised on CPU here and on a device later.
It changes no benchmark semantics: the config, the optimizer construction, the
generator seeding rule and the sampling call are the benchmark's.

Resolved and recorded, never assumed: architecture, positional capacity, unique
trainable parameters, tied aliases, dtypes, optimizer class and hyperparameters,
accumulation, clipping source, dropout, RNG handling, warmup/measured split.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# The retained recipe, transcribed from train_llm.py and the ds_zero*.json configs.
RETAINED = {
    "model_builder": "transformers.GPT2LMHeadModel(transformers.GPT2Config())",
    "config_overrides": {},                 # none: pure defaults
    "input_builder": "torch.randint(0, 50257, (bs, seq_len), generator=gen, dtype=torch.long)",
    "generator_rule": "torch.Generator(device=<train device>) seeded seed+rank",
    "model_seed_rule": "torch.manual_seed(seed)",
    "optimizer_builder": "torch.optim.AdamW(model.parameters(), lr=1e-4)",
    "optimizer_explicit_args": {"lr": 1e-4},
    "gradient_accumulation_steps": 1,
    "gradient_clipping_source": "DeepSpeed config ds_zero{1,2}.json: gradient_clipping=1.0",
    "dropout_source": "GPT2Config defaults (not overridden by the benchmark)",
    "warmup_measured_split": "args.warmup untimed steps, then args.steps measured",
    "amp_recipe": 'torch.amp.autocast("cuda", float16) + GradScaler("cuda")',
    "fp32_recipe": "no autocast, no scaler",
}


def build_model(*, seed: int, device: str = "cpu", torch_module=None):
    """The retained model. `device` is the ONLY difference from train_llm.py."""
    torch = torch_module or __import__("torch")
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    cfg = GPT2Config()                       # defaults, exactly as retained
    model = GPT2LMHeadModel(cfg)
    return model.to(device), cfg


def build_optimizer(model, *, torch_module=None):
    """The retained optimizer: only lr is given, so every other value is a
    torch DEFAULT and must be read back rather than assumed."""
    torch = torch_module or __import__("torch")
    return torch.optim.AdamW(model.parameters(), lr=1e-4)


def build_generator(*, seed: int, rank: int, device: str = "cpu", torch_module=None):
    """The retained per-rank generator. train_llm.py uses device=<train device>,
    so a CPU run here and a CUDA run later draw DIFFERENT streams; that is a
    recorded property of the recipe, not something this module changes."""
    torch = torch_module or __import__("torch")
    g = torch.Generator(device=device)
    g.manual_seed(seed + rank)
    return g


def synthetic_batch(bs: int, seq_len: int, device, gen, *, torch_module=None):
    """Byte-identical to the retained generator call."""
    torch = torch_module or __import__("torch")
    return torch.randint(0, 50257, (bs, seq_len), device=device,
                         generator=gen, dtype=torch.long)


def resolve_optimizer_semantics(opt) -> dict:
    """Read the RESOLVED values from the optimizer. AdamW's default
    weight_decay is 0.01, not 0, and assuming otherwise was a real defect."""
    g = opt.param_groups[0]
    return {
        "class": type(opt).__name__,
        "module": type(opt).__module__,
        "decoupled_weight_decay": type(opt).__name__ == "AdamW",
        "lr": g["lr"], "beta1": g["betas"][0], "beta2": g["betas"][1],
        "eps": g["eps"], "weight_decay": g["weight_decay"],
        "amsgrad": bool(g.get("amsgrad", False)),
        "maximize": bool(g.get("maximize", False)),
        "foreach": g.get("foreach"), "capturable": g.get("capturable"),
        "fused": g.get("fused"),
        "n_param_groups": len(opt.param_groups),
        "n_params_in_groups": sum(len(x["params"]) for x in opt.param_groups),
    }


def resolve_model_identity(model, cfg) -> dict:
    """Unique trainable parameters versus tied aliases, resolved by storage."""
    import torch
    by_storage: dict[int, list[str]] = {}
    total_listed = 0
    for name, p in model.named_parameters():
        total_listed += 1
        key = p.untyped_storage().data_ptr() if hasattr(p, "untyped_storage") \
            else p.storage().data_ptr()
        by_storage.setdefault(key, []).append(name)
    tied = {tuple(sorted(v)) for v in by_storage.values() if len(v) > 1}
    unique_numel = sum(next(iter(g)).numel() for g in
                       ([ [dict(model.named_parameters())[n] for n in names]
                          for names in ([v for v in by_storage.values()]) ]))
    sum_numel = sum(p.numel() for _n, p in model.named_parameters())
    dtypes = sorted({str(p.dtype) for _n, p in model.named_parameters()})
    return {
        "architecture": type(model).__name__,
        "config_class": type(cfg).__name__,
        "n_positions": cfg.n_positions, "n_embd": cfg.n_embd,
        "n_layer": cfg.n_layer, "n_head": cfg.n_head, "vocab_size": cfg.vocab_size,
        "resid_pdrop": cfg.resid_pdrop, "attn_pdrop": cfg.attn_pdrop,
        "embd_pdrop": cfg.embd_pdrop,
        "named_parameter_entries": total_listed,
        "distinct_storages": len(by_storage),
        "tied_alias_groups": [list(t) for t in sorted(tied)],
        "sum_of_named_parameter_numel": sum_numel,
        "unique_trainable_numel": unique_numel,
        "param_dtypes": dtypes,
    }


DIGEST_CHUNK_ELEMS = 1_048_576      # float64 workspace bound: 8 MiB + copy


def state_digest(model, opt=None) -> dict:
    """Digest of the initial state through the SAME path a device run will use."""
    import torch
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode() + b"\0")
        # Streamed in bounded chunks. The byte sequence is unchanged: the same
        # C-contiguous float64 elements in the same order, so the digest value
        # is identical to the previous whole-parameter conversion. The
        # representation is NOT versioned because it did not change; a test
        # asserts equality against the original expression.
        flat = p.detach().reshape(-1)
        n = int(flat.numel())
        for off in range(0, n, DIGEST_CHUNK_ELEMS):
            k = min(DIGEST_CHUNK_ELEMS, n - off)
            h.update(flat.narrow(0, off, k)
                     .to("cpu", torch.float64).numpy().tobytes())
    out = {"parameter_state_sha256": h.hexdigest()}
    if opt is not None:
        ho = hashlib.sha256()
        ho.update(json.dumps(resolve_optimizer_semantics(opt), sort_keys=True).encode())
        out["optimizer_semantics_sha256"] = ho.hexdigest()
    return out


def input_digest(ids) -> dict:
    """Digest of the REALIZED inputs, through the same executed path."""
    import torch
    a = ids.detach().to("cpu", torch.int64).numpy()
    return {"realized_input_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "shape": list(ids.shape), "dtype": str(ids.dtype),
            "min_token": int(a.min()), "max_token": int(a.max())}


AMP_SOURCE = "gpu_bench_stage1/bench/train_llm.py:144"


def resolve_amp_config(precision: str, ds_cfg: dict) -> tuple[dict, dict]:
    """Apply the retained benchmark's own AMP configuration. No invention.

    The retained source sets, for DeepSpeed mode:
        ds_cfg["fp16"] = {"enabled": True, "loss_scale": 0,
                          "initial_scale_power": 12}
    `loss_scale: 0` selects DeepSpeed's DYNAMIC loss scaling. The same source
    constructs `torch.amp.GradScaler` only when mode != "deepspeed" (:188), so
    native GradScaler semantics are NOT assumed to apply here.

    Returns (ds_cfg, amp_semantics).
    """
    if precision not in ("fp32", "amp_fp16"):
        raise ValueError(f"unsupported precision {precision!r}")
    cfg = dict(ds_cfg)
    if precision == "amp_fp16":
        cfg["fp16"] = {"enabled": True, "loss_scale": 0,
                       "initial_scale_power": 12}
        cfg["bf16"] = {"enabled": False}
        sem = {
            "source": AMP_SOURCE + ", the retained benchmark's own configuration",
            "precision": precision,
            "fp16_enabled": True, "loss_scale": 0, "initial_scale_power": 12,
            "scaler": "DeepSpeed dynamic loss scaling",
            "native_gradscaler_used": False,
            "native_gradscaler_semantics_assumed": False,
            "skip_policy": ("overflow skips are a natural consequence of "
                            "dynamic scaling and are reported as observed; no "
                            "fault injection and no manual scale tuning"),
            "inferred_from_absent_hook": False,
        }
    else:
        cfg["fp16"] = {"enabled": False}
        cfg["bf16"] = {"enabled": False}
        sem = {"source": AMP_SOURCE, "precision": precision,
               "fp16_enabled": False, "loss_scale": None,
               "scaler": None, "native_gradscaler_used": False,
               "native_gradscaler_semantics_assumed": False,
               "inferred_from_absent_hook": False}
    return cfg, sem
