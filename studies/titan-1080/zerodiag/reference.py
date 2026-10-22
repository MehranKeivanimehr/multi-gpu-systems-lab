"""The two singleton FP32 cross-family reference cases.

Design (frozen before any new result): one TITAN RTX singleton and one GTX 1080
Ti singleton, same pinned model revision, same frozen prompt corpus, same
greedy decode, FP32, no warmup phase distinction beyond the declared one. The
control asks whether two different GPU families, given *verified identical
realized inputs from a verified identical initial state*, produce agreeing
outputs.

What this control does NOT do, stated here so it cannot drift:

* Loss or logit agreement is **not** independent full-gradient verification.
* Seeds alone are **not** proof of identical inputs or identical dropout masks;
  realized token ids and the initial-state digest are compared directly, and a
  mismatch refuses.
* The historical reference limitations are preserved. This control is
  **informed**, not blind: the operator knew the hypothesis when designing it.

Any amendment to this protocol must be declared prospectively, before the
affected result exists.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

PROTOCOL = {
    "schema": "titan_reference_protocol_v1",
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "model_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "recipe": "hf_generate",
    "dtype": "float32",
    "greedy": True,
    "do_sample": False,
    "prompt_len_tokens": 256,
    "output_tokens_exact": 64,
    "corpus": "protocols/prompt_corpus.json",
    "corpus_sha256": ("f2d11005cbce3e5419be77a3c3b05eff7a0b3e56b791e0d3c13"
                      "327d2ac611350"),
    "phases": {"warmup": "1 untimed generate call, discarded",
               "measurement": "1 recorded generate call per singleton"},
    "matching_rules": {
        "realized_input": "sha256 over the exact tokenized input id tensor",
        "initial_state": "sha256 over float64 parameter bytes, streamed",
        "comparison": ("greedy output token ids compared exactly; final-token "
                       "logits compared with the oracle's dtype-derived "
                       "tolerance policy"),
        "refuse_on": ("any realized-input or initial-state digest mismatch "
                      "between the two singletons"),
    },
    "thresholds": {
        "output_ids": "exact equality required",
        "logits_rtol": "64u for the compared dtype (oracle tolerance policy)",
        "note": "no tolerance is widened to accommodate a disagreement",
    },
    "establishes": ("cross-family output agreement on verified identical "
                    "realized inputs from a verified identical initial state"),
    "does_not_establish": ("independent full-gradient verification; training "
                           "quality; any historical throughput or energy claim"),
    "status": "INFORMED control, not blind",
    "amendments": [],
}


def protocol_digest() -> str:
    return hashlib.sha256(
        json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def corpus_self_digest(obj: dict) -> str:
    """The frozen corpus digest convention, stated explicitly.

    The campaign plan's `corpus_sha256` is NOT the digest of the file bytes. It
    is the digest of the corpus object with its own `corpus_sha256` field
    removed, serialised as compact sorted JSON. Excluding the field from its own
    digest is what makes the value verifiable instead of circular; the file
    bytes additionally carry whitespace and the field itself, so they hash
    differently (254bee77... for the on-node copy). Both facts are recorded so
    neither is mistaken for corruption.
    """
    o = {k: v for k, v in obj.items() if k != "corpus_sha256"}
    return hashlib.sha256(
        json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_corpus(path: str | pathlib.Path) -> dict:
    """Verify the corpus against the frozen digest, by the frozen convention."""
    p = pathlib.Path(path)
    if not p.is_file():
        return {"ok": False, "reason": f"corpus not found at {p}"}
    raw = p.read_bytes()
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        return {"ok": False, "reason": f"corpus is not readable JSON: {exc}"}
    got = corpus_self_digest(obj)
    declared = obj.get("corpus_sha256")
    ok = (got == PROTOCOL["corpus_sha256"] and declared == PROTOCOL["corpus_sha256"])
    return {
        "ok": ok,
        "self_digest": got, "expected": PROTOCOL["corpus_sha256"],
        "field_declared": declared,
        "file_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "convention": ("digest of the object with corpus_sha256 removed, "
                       "compact sorted JSON; the file-bytes digest differs by "
                       "construction and is recorded, not compared"),
        "n_prompts": obj.get("n_prompts"), "prompt_len": obj.get("prompt_len"),
        "policy": obj.get("policy"),
        "reason": (None if ok else
                   "corpus does not match the frozen digest under its own "
                   "declared convention"),
    }


def verify_realized_match(a: dict, b: dict) -> dict:
    """Compare two singletons' realized bindings. Seeds are not accepted.

    `a` and `b` are each {"realized_input_sha256", "initial_state_sha256",
    "input_ids_shape", "dtype", "revision", "device_name"}.
    """
    checks = {}
    for key in ("realized_input_sha256", "initial_state_sha256",
                "input_ids_shape", "dtype", "revision"):
        checks[key] = a.get(key) is not None and a.get(key) == b.get(key)
    same_family = a.get("device_name") == b.get("device_name")
    ok = all(checks.values()) and not same_family
    return {
        "matched": ok, "checks": checks,
        "cross_family": not same_family,
        "devices": [a.get("device_name"), b.get("device_name")],
        "basis": ("realized token-id digest and initial-state digest, compared "
                  "directly; a shared seed alone is not accepted as proof"),
        "reason": (None if ok else
                   ("realized bindings differ, or both singletons are the same "
                    "family, so the cross-family control is not established")),
        "dropout_note": ("generation runs in eval mode with dropout inactive; "
                         "no claim is made about matching dropout masks"),
    }


def declare_amendment(reason: str, *, prospective: bool) -> dict:
    """Record a protocol amendment. Retrospective amendments are refused."""
    if not prospective:
        raise ValueError(
            "a controlled protocol amendment must be declared PROSPECTIVELY, "
            "before the affected result exists; refusing to record it after "
            "the fact")
    return {"reason": reason, "prospective": True}
