"""Where the on-node vLLM environment and the pinned model cache live.

These two contracts (`test_guided_dispatch`, `test_prometheus_init`) drive the
INSTALLED vLLM stack. They cannot run anywhere else, and they must FAIL rather
than skip when it is absent: a skipped repair check is not evidence that a
repair works.

Both values come from the environment, with the on-node paths as defaults, so
the published copy of these tests needs only the two variables set and carries
no operator path.
"""
from __future__ import annotations

import os
import pathlib

VLLM_PYTHON = os.environ.get(
    "ZD_VLLM_PYTHON",
    os.environ.get("ZERODIAG_VLLM_PYTHON",
                   "$PROJECT_ROOT/gpu_bench_stage1/envs/"
                   "bench-vllm/bin/python"))
HF_HOME = os.environ.get(
    "ZD_HF_HOME",
    os.environ.get("ZERODIAG_HF_HOME",
                   "$PROJECT_ROOT/gpu_bench_stage1/cache/hf"))

OFFLINE_ENV = {"HF_HOME": HF_HOME, "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}


def require() -> None:
    """Fail, with an actionable message, if the environment is not here."""
    if not (VLLM_PYTHON and pathlib.Path(VLLM_PYTHON).is_file()):
        raise AssertionError(
            f"ZD_VLLM_PYTHON must name an interpreter with vllm 0.6.3.post1 "
            f"and prometheus_client 0.26.0 installed (looked at "
            f"{VLLM_PYTHON!r}). This contract FAILS rather than skips: an "
            f"unverified repair is not a verified one. The on-node receipts "
            f"are in the private evidence.")
    cache = pathlib.Path(HF_HOME)
    if not cache.is_dir():
        raise AssertionError(
            f"ZD_HF_HOME must name a Hugging Face cache holding the pinned "
            f"TinyLlama revision (looked at {HF_HOME!r}). The probe loads the "
            f"tokenizer offline; it never downloads a model.")
