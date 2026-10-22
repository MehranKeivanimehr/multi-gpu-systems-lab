"""Seeded synthetic batches, generated once on the GPU so data loading never enters the timed region."""
from __future__ import annotations

import itertools

import torch


def _digest(t: torch.Tensor) -> str:
    import hashlib

    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape), str(t.dtype))).encode() + t.view(torch.uint8).numpy().tobytes()).hexdigest()


class SyntheticTokens:
    def __init__(self, vocab_size: int, seq_len: int, micro_batch: int, device, seed: int, pool: int = 4):
        g = torch.Generator(device="cpu").manual_seed(seed)
        cpu = [torch.randint(0, vocab_size, (micro_batch, seq_len), generator=g) for _ in range(pool)]
        # Actual inputs, identified by content (CPU bytes before the copy to the device), in cycle order.
        self.input_digests = {"generator": "torch.Generator(cpu).manual_seed", "seed": seed, "pool": pool,
                              "cycle": "batches are reused in this order", "sha256": [_digest(x) for x in cpu]}
        self.batches = [x.to(device, non_blocking=True) for x in cpu]
        self._it = itertools.cycle(self.batches)
        self.samples_per_batch = micro_batch
        self.tokens_per_batch = micro_batch * seq_len

    def next(self):
        x = next(self._it)
        return {"input_ids": x, "labels": x}


class SyntheticImages:
    def __init__(self, image_size: int, num_classes: int, micro_batch: int, device, seed: int,
                 dtype=torch.float32, pool: int = 4):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.batches = []
        for _ in range(pool):
            x = torch.randn(micro_batch, 3, image_size, image_size, generator=g)
            y = torch.randint(0, num_classes, (micro_batch,), generator=g)
            x = x.to(device, dtype=dtype).contiguous(memory_format=torch.channels_last)
            self.batches.append({"images": x, "labels": y.to(device)})
        self._it = itertools.cycle(self.batches)
        self.samples_per_batch = micro_batch
        self.tokens_per_batch = None

    def next(self):
        return next(self._it)


class GlobalSynthetic:
    """Rank-independent global inputs for numerical reference diagnostics (A6000 validation, session 20261014a).

    Every optimizer step uses one global batch of ``micro_batch * grad_accum * world`` samples drawn from a single
    CPU generator seeded independently of the rank. Rank ``r`` processes, at accumulation micro-step ``i``, global
    rows ``[(i*world + r)*micro_batch, (i*world + r + 1)*micro_batch)``. Any (world, grad_accum) factorisation of the
    same global batch therefore sees exactly the same samples, grouped into the same micro-batches when
    ``micro_batch`` is equal. ``steps`` distinct global batches are generated and then cycled.
    """

    def __init__(self, kind: str, *, micro_batch: int, grad_accum: int, world: int, rank: int, device, seed: int,
                 steps: int, vocab_size: int | None = None, seq_len: int | None = None, image_size: int | None = None,
                 num_classes: int | None = None, dtype=torch.float32):
        g = torch.Generator(device="cpu").manual_seed(seed)
        global_batch = micro_batch * grad_accum * world
        self.micro = []                                   # list over steps of list over micro-steps
        self.input_digests = {"generator": "torch.Generator(cpu).manual_seed", "seed": seed, "global_batch": global_batch,
                              "labels": "labels = input_ids (causal LM shifts internally)" if kind == "causal_lm" else
                              "class labels", "attention_mask": None, "per_step_microbatch_sha256": [],
                              "per_step_global_rows": [], "world": world, "rank": rank}
        for _ in range(steps):
            if kind == "causal_lm":
                full = torch.randint(0, vocab_size, (global_batch, seq_len), generator=g)
            else:
                images = torch.randn(global_batch, 3, image_size, image_size, generator=g)
                labels = torch.randint(0, num_classes, (global_batch,), generator=g)
            per_step = []
            digests = []
            self.input_digests["per_step_global_rows"].append([(i * world + rank) * micro_batch for i in range(grad_accum)])
            for i in range(grad_accum):
                lo = (i * world + rank) * micro_batch
                rows = slice(lo, lo + micro_batch)
                if kind == "causal_lm":
                    digests.append(_digest(full[rows]))
                    x = full[rows].to(device)
                    per_step.append({"input_ids": x, "labels": x})
                else:
                    x = images[rows].to(device, dtype=dtype).contiguous(memory_format=torch.channels_last)
                    per_step.append({"images": x, "labels": labels[rows].to(device)})
            self.micro.append(per_step)
            self.input_digests["per_step_microbatch_sha256"].append(digests)
        self._flat = itertools.cycle([b for step in self.micro for b in step])
        self.samples_per_batch = micro_batch
        self.global_batch = global_batch

    def next(self):
        return next(self._flat)
