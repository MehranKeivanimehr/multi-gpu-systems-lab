"""Model construction with random weights (architecture from the vendored config, no checkpoint loading)."""
from __future__ import annotations

import json

import torch
import torch.nn as nn

from .config import ROOT


def load_hf_config_dict(entry: dict) -> dict:
    with open(ROOT / entry["hf_config_dir"] / "config.json") as f:
        return json.load(f)


def build_causal_lm(entry: dict, device, dtype: torch.dtype) -> nn.Module:
    """Build an HF causal LM directly on `device` ('cuda:N' or 'meta') with random init."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(ROOT / entry["hf_config_dir"])
    cfg.use_cache = False
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation="sdpa")
    model.train()
    return model


def build_vision(entry: dict, device, dtype: torch.dtype = torch.float32) -> nn.Module:
    import torchvision

    with torch.device(device):
        model = getattr(torchvision.models, entry["torchvision_name"])(weights=None, num_classes=entry["num_classes"])
    # Weights stay contiguous (FSDP2 rejects non-contiguous params); inputs are channels_last, which is what
    # makes cuDNN pick NHWC tensor-core kernels.
    model = model.to(dtype=dtype)
    model.train()
    return model


def shard_units(model: nn.Module) -> list[nn.Module]:
    """Sub-modules that become separate FSDP units: transformer decoder layers or ResNet bottlenecks."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    units = []
    for name in ("layer1", "layer2", "layer3", "layer4"):
        if hasattr(model, name):
            units.extend(getattr(model, name).children())
    return units


def has_tied_params(model: nn.Module) -> bool:
    seen, total = set(), 0
    for _, p in model.named_parameters(remove_duplicate=False):
        total += 1
        seen.add(id(p))
    return len(seen) != total


@torch.no_grad()
def materialize_meta_model(model: nn.Module, device, seed: int) -> None:
    """After fully_shard on a meta-device model: allocate local shards on `device` and fill them with random
    values (normal(0, 0.02) for matrices, 1 for norm weights, 0 for biases). Rotary buffers are recomputed."""
    from torch.distributed.tensor import DTensor

    model.to_empty(device=device)
    gen = torch.Generator(device=device).manual_seed(seed + torch.distributed.get_rank())
    for name, p in model.named_parameters():
        local = p.to_local() if isinstance(p, DTensor) else p
        if name.endswith("bias"):
            local.zero_()
        elif local.dim() == 1:
            local.fill_(1.0)
        else:
            local.normal_(0.0, 0.02, generator=gen)
    reset_rotary_buffers(model, device)


@torch.no_grad()
def reset_rotary_buffers(model: nn.Module, device) -> None:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    for mod in model.modules():
        if hasattr(mod, "inv_freq") and hasattr(mod, "rope_type") and hasattr(mod, "config"):
            if mod.rope_type == "default":
                inv, _ = mod.compute_default_rope_parameters(mod.config)
            else:
                inv, _ = ROPE_INIT_FUNCTIONS[mod.rope_type](mod.config)
            mod.inv_freq.copy_(inv.to(device))
            if hasattr(mod, "original_inv_freq"):
                mod.original_inv_freq.copy_(inv.to(device))


def vision_train_flops_per_sample(entry: dict) -> float:
    """Forward+backward FLOPs per image, counted on the meta device (no compute)."""
    from torch.utils.flop_counter import FlopCounterMode

    model = build_vision(entry, "meta")
    x = torch.empty(1, 3, entry["image_size"], entry["image_size"], device="meta")
    counter = FlopCounterMode(display=False)
    with counter:
        model(x).sum().backward()
    return float(counter.get_total_flops())
