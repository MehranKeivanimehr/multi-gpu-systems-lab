"""Executing runner for one singleton FP32 cross-family reference case.

Launched per-rank by the production launcher (world size one per case), so it
inherits the same ownership, aggregate accounting and bounded cleanup as the
training workers.

What it records, and nothing it does not:

* the pinned model and tokenizer revisions, resolved from the local cache;
* the ACTUAL tokenized input length and a digest of the exact input id tensor;
* a digest of the initial parameter state, streamed in bounded chunks;
* the resolved precision, the observation phase, and whether dropout was
  active (generation runs in eval mode, so it is not);
* greedy output token ids and the COMPLETE final-step logit array, retained
  losslessly (raw little-endian binary + JSON schema + checksum) so the
  frozen 64u comparison can be recomputed offline, without a GPU or a model
  cache, by `zerodiag/ref_compare.py`. Digests are retained as integrity
  metadata only: a digest comparison is NOT the numerical outcome.

It does not claim matching dropout masks from a matching seed, and it does not
present loss or logit agreement as gradient verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from zerodiag import reference as RF            # noqa: E402
from zerodiag import lossless as LL             # noqa: E402
from zerodiag.containment import apply_affinity, MonitorThread  # noqa: E402

MONITOR = None


def _digest_tensor(t) -> str:
    import torch
    flat = t.detach().reshape(-1).contiguous()
    h = hashlib.sha256()
    n = int(flat.numel())
    step = 1 << 20
    for off in range(0, n, step):
        k = min(step, n - off)
        h.update(flat.narrow(0, off, k).to("cpu", torch.float64).numpy().tobytes()
                 if flat.dtype.is_floating_point else
                 flat.narrow(0, off, k).to("cpu").numpy().tobytes())
    return h.hexdigest()


REQUIRED_ARRAYS = ("final_logits", "input_ids", "attention_mask",
                   "generated_ids")


def arrays_complete(arr_dir) -> dict:
    """Every required array must load and verify before success is reported."""
    rep = {"required": list(REQUIRED_ARRAYS), "verified": {}, "ok": False}
    for n in REQUIRED_ARRAYS:
        try:
            _, rec = LL.load_array(arr_dir, n)
            rep["verified"][n] = {"sha256": rec["sha256"], "shape": rec["shape"],
                                  "dtype": rec["dtype"],
                                  "n_bytes": rec["n_bytes"]}
        except Exception as exc:
            rep["verified"][n] = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    rep["ok"] = all(isinstance(v, dict) for v in rep["verified"].values())
    return rep


def build_comparison_metadata(ev: dict, *, case_id: str,
                              pad_token_id: int) -> dict:
    """The offline comparator's input contract, built from recorded evidence.

    Pure and separately tested: a KeyError here would otherwise only surface
    on a GPU, after the measurement had already been taken.
    """
    return {
        "case_id": case_id,
        "criterion": "titan_reference_logit_criterion_v1",
        "clarification": "protocols/REFERENCE_CRITERION_CLARIFICATION.md",
        "arrays_dir": "arrays",
        "arrays": {k: {"file": v["file"], "sha256": v["sha256"],
                       "shape": v["shape"], "dtype": v["dtype"]}
                   for k, v in ev["retained"].items()},
        "realized_input_sha256": ev["realized_input"]["realized_input_sha256"],
        "attention_mask_sha256": ev["realized_input"]["attention_mask_sha256"],
        "initial_state_sha256": ev["initial_state"]["parameter_state_sha256"],
        "input_shape": ev["realized_input"]["shape"],
        "model": RF.PROTOCOL["model"],
        "resolved_revision": ev["resolved_revision"],
        "model_dtype": ev["model_dtype"],
        "precision_policy": RF.PROTOCOL.get("precision"),
        "eval_settings": {"module_training_mode": ev["dropout_active"],
                          "grad_enabled": False,
                          "do_sample": False, "num_beams": 1,
                          "max_new_tokens":
                              RF.PROTOCOL["output_tokens_exact"],
                          "pad_token_id": pad_token_id},
        "dropout_active": ev["dropout_active"],
        "child_device_name": ev["child_device_name"],
        "child_device_capability": ev["child_device_capability"],
        "child_visible_devices": ev["child_visible_devices"],
        "torch_version": ev["torch_version"],
        "protocol_digest": RF.protocol_digest(),
    }


def gate_completed(ev: dict, arr_dir) -> str:
    """Decide whether a reference case may be reported complete.

    A reference case is complete only if the retained arrays exist and verify,
    because the frozen comparison is evaluated FROM them. Mutates `ev` with the
    gate record and returns the stage that may be written.
    """
    gate = arrays_complete(arr_dir)
    ev["retained_arrays"] = gate
    if not gate["ok"]:
        ev["reason"] = ("retained logit arrays are missing or failed "
                        "verification; the numerical comparison cannot be "
                        "recomputed, so this case is not complete")
        ev["reference_case_complete"] = False
        return "refused"
    ev["reference_case_complete"] = True
    return "completed"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-cpus", type=int, default=4)
    ap.add_argument("--memory-budget-gib", type=float, default=8.0)
    ap.add_argument("--monitor-interval-s", type=float, default=0.5)
    ap.add_argument("--n-prompts", type=int, default=8)
    a = ap.parse_args(argv)
    if a.world_size != 1 or a.rank != 0:
        print("REFUSED: a reference singleton is world size one, rank zero",
              file=sys.stderr)
        return 95

    t0 = time.perf_counter()
    aff = apply_affinity(a.max_cpus)
    global MONITOR
    MONITOR = MonitorThread(
        budget_bytes=int(a.memory_budget_gib * 1073741824),
        interval_s=a.monitor_interval_s).start()

    out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    arr = out / "arrays"
    ev = {"case_id": a.case_id, "kind": "reference", "stage": "started",
          "rank": a.rank, "world_size": a.world_size,
          "containment": {"affinity": aff, "monitor": MONITOR.report()},
          "child_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
          "protocol": RF.PROTOCOL, "protocol_digest": RF.protocol_digest(),
          "device_contact": "pending"}

    def dump(stage, **kw):
        ev["stage"] = stage
        ev["elapsed_s"] = round(time.perf_counter() - t0, 3)
        ev["memory_monitor"] = MONITOR.report() if MONITOR else None
        if MONITOR is None:
            ev["verification_successful"] = False
            ev["refusal"] = "no memory monitor was running"
        if stage == "completed":
            stage = ev["stage"] = gate_completed(ev, arr)
        ev.update(kw)
        (out / "reference_evidence.json").write_text(
            json.dumps(ev, indent=2, sort_keys=True, default=str))

    try:
        corpus = RF.verify_corpus(a.corpus)
        ev["corpus_check"] = corpus
        if not corpus["ok"]:
            dump("refused", reason=f"corpus check failed: {corpus['reason']}")
            return 90

        import torch
        ev["torch_version"] = torch.__version__
        if not torch.cuda.is_available():
            dump("refused", reason="no CUDA device visible to the child")
            return 90
        props = torch.cuda.get_device_properties(0)
        ev["child_device_name"] = props.name
        ev["child_device_capability"] = f"{props.major}.{props.minor}"
        ev["device_contact"] = "one device, read via torch properties"

        from transformers import AutoModelForCausalLM, AutoTokenizer
        rev = RF.PROTOCOL["model_revision"]
        name = RF.PROTOCOL["model"]
        tok = AutoTokenizer.from_pretrained(name, revision=rev,
                                            local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            name, revision=rev, local_files_only=True,
            torch_dtype=torch.float32)
        model.eval().to("cuda")
        ev["resolved_revision"] = rev
        ev["model_dtype"] = str(next(model.parameters()).dtype)
        ev["dropout_active"] = bool(model.training)
        ev["observation_phase"] = ("model.eval(); generation only. Dropout is "
                                   "inactive, so no claim is made about "
                                   "matching dropout masks.")
        ev["initial_state"] = {"parameter_state_sha256":
                               _digest_tensor(torch.cat(
                                   [p.detach().reshape(-1)
                                    for p in model.parameters()]))}
        ev["n_parameters"] = sum(p.numel() for p in model.parameters())

        # The frozen corpus stores PRE-TOKENIZED id lists, not text. They are
        # consumed as token ids through the model's own input interface; the
        # tokenizer is never asked to tokenize them (that was the earlier
        # defect, which raised ValueError on a list of ints).
        corpus_obj = json.loads(pathlib.Path(a.corpus).read_text())
        prompts = corpus_obj["prompts"]
        want = RF.PROTOCOL["prompt_len_tokens"]
        ev["corpus_representation"] = {
            "prompts_are_pre_tokenized_id_lists": all(
                isinstance(x, list) and all(isinstance(i, int) for i in x)
                for x in prompts[:4]),
            "declared_prompt_len": corpus_obj.get("prompt_len"),
            "n_prompts": len(prompts),
            "consumed_as": "token ids via model(input_ids=...)",
            "tokenizer_used_to_tokenize": False,
            "special_tokens_added_by_us": False,
            "excluded_special_ids": corpus_obj.get("excluded_special_ids"),
        }
        sel = prompts[:a.n_prompts]
        ids_list, lengths = [], []
        for row in sel:
            if not (isinstance(row, list) and all(isinstance(i, int) for i in row)):
                dump("refused", reason=("corpus prompt is not a list of token "
                                        "ids; refusing to guess its encoding"))
                return 90
            lengths.append(len(row))
            ids_list.append(torch.tensor(row[:want], dtype=torch.long))
        ev["consumed_token_lengths"] = lengths
        ev["consumed_lengths_all_equal_protocol"] = all(n == want for n in lengths)
        ev["length_deviations"] = [(i, n) for i, n in enumerate(lengths) if n != want]

        import torch as T
        keep = [i for i, n in zip(ids_list, lengths) if n >= want]
        if not keep:
            dump("refused", reason=(f"no corpus prompt reaches {want} token ids; "
                                    f"lengths were {sorted(set(lengths))}"))
            return 90
        batch = T.stack(keep).to("cuda")
        mask = T.ones_like(batch)
        ev["realized_input"] = {
            "realized_input_sha256": _digest_tensor(batch),
            "attention_mask_sha256": _digest_tensor(mask),
            "attention_mask_policy": "all ones; no padding is present",
            "shape": list(batch.shape), "dtype": str(batch.dtype),
            "n_prompts_used": int(batch.shape[0]),
            "first_ids": batch[0, :8].tolist(),
            "last_ids": batch[0, -8:].tolist()}
        ev["retained"] = {
            "input_ids": LL.save_array(
                batch, arr, "input_ids",
                note="exact token ids consumed via model(input_ids=...)"),
            "attention_mask": LL.save_array(
                mask, arr, "attention_mask",
                note="all ones; no padding present"),
        }

        with T.no_grad():
            o = model(input_ids=batch, attention_mask=mask)
            last = o.logits[:, -1, :]
            ev["final_logits"] = {
                "sha256": _digest_tensor(last),
                "shape": list(last.shape),
                "dtype": str(o.logits.dtype),
                "token_position": int(batch.shape[1] - 1),
                "token_position_convention": ("zero-based index into the input "
                                              "sequence; the last input "
                                              "position, i.e. the logits that "
                                              "predict the first new token"),
                "digest_role": ("integrity metadata only; the verdict is "
                                "computed from the retained array")}
            ev["retained"]["final_logits"] = LL.save_array(
                last, arr, "final_logits",
                note=(f"complete final-step logits, position "
                      f"{int(batch.shape[1] - 1)} (last input token), "
                      f"shape [n_prompts, vocab]"))
            gen = model.generate(input_ids=batch, attention_mask=mask,
                                 max_new_tokens=RF.PROTOCOL["output_tokens_exact"],
                                 do_sample=False, num_beams=1,
                                 pad_token_id=tok.eos_token_id)
        new = gen[:, batch.shape[1]:]
        ev["retained"]["generated_ids"] = LL.save_array(
            new, arr, "generated_ids",
            note="greedy new token ids only, input prefix removed")
        ev["generated"] = {
            "output_ids_sha256": _digest_tensor(new),
            "shape": list(new.shape),
            "n_new_tokens_per_prompt": int(new.shape[1]),
            "exact_output_tokens_requested": RF.PROTOCOL["output_tokens_exact"]}
        ev["cuda_peak_alloc_mib"] = round(
            T.cuda.max_memory_allocated() / 1048576, 1)
        meta = build_comparison_metadata(
            ev, case_id=a.case_id, pad_token_id=int(tok.eos_token_id))
        (out / "comparison_metadata.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True))
        ev["comparison_metadata_written"] = True
        ev["establishes"] = RF.PROTOCOL["establishes"]
        ev["does_not_establish"] = RF.PROTOCOL["does_not_establish"]
        ev["status"] = RF.PROTOCOL["status"]
        dump("completed")
        return 0
    except Exception as exc:                       # preserved, never swallowed
        import traceback
        dump("error", reason=f"{type(exc).__name__}: {exc}",
             traceback=traceback.format_exc())
        return 91
    finally:
        if MONITOR is not None:
            MONITOR.stop()


if __name__ == "__main__":
    sys.exit(main())
