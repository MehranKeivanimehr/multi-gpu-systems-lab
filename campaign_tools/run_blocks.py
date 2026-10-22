"""RETIRED driver: run_blocks.py

This driver invoked tools/gpu_attempt.sh, bypassing the audited Python launcher
(studies/titan-1080/bench/launch.py). The shell path was shown -- with a MOCKED
nvidia-smi, no device access -- to LAUNCH when the device query exited 9, when it
returned no output, and when it returned an UNREQUESTED row (widening
CUDA_VISIBLE_DEVICES to a device that was never requested). It also held no
device reservation, revalidated only utilisation during settling, and never
confirmed ownership-scoped descendant cleanup.

Use tools/run_blocks_routed.py, which routes every training, auxiliary and
engine adapter through launch.run_case. This module refuses to run.
"""
import sys as _retired_sys

print("REFUSED: run_blocks.py is retired; it bypassed the audited launcher.",
      file=_retired_sys.stderr)
print("         Use tools/run_blocks_routed.py (routes through launch.run_case).",
      file=_retired_sys.stderr)
_retired_sys.exit(95)

_ORIGINAL_SOURCE_RETAINED_BELOW = """
\"\"\"Execute the frozen plan as independent repeat blocks, one job at a time.

Design points that matter for the statistics:

* Each repeat is a FRESH PROCESS in a fresh directory. Nothing is reused
  between repeats, so the six observations per configuration are independent
  in the sense the 31/32 median-coverage claim requires.
* Within a block, configuration order is RANDOMISED and interleaved across
  groups and recipes, with the seed recorded. Running every repeat of one
  configuration back to back would confound repeat variance with thermal drift.
* The launch cutoff is re-checked before every launch, and a block stops cleanly
  rather than starting a job it cannot finish inside the window.
* An existing attempt directory is never overwritten: the runner skips it, so a
  resumed campaign preserves earlier outcomes including failures.
\"\"\"
from __future__ import annotations

import argparse, json, os, pathlib, random, subprocess, sys, time

W = pathlib.Path(os.environ.get("TITAN_CAMPAIGN_ROOT",
    str(pathlib.Path(__file__).resolve().parent.parent)))
T = W / "src/multigpu-bench/studies/titan-1080"
EXPIRY = 1792161707          # 2026-10-16T14:41:47Z
CUTOFF = EXPIRY - 1800       # stop launching 30 min before expiry

RECIPE_ARGS = {
    "single": ["--mode", "single"],
    "ddp":    ["--mode", "ddp"],
    "fsdp":   ["--mode", "fsdp"],
    "zero1":  ["--mode", "deepspeed", "--zero-stage", "1"],
    "zero2":  ["--mode", "deepspeed", "--zero-stage", "2"],
}
TRAINING = tuple(RECIPE_ARGS)


# Engine-aware DeepSpeed evidence, matched to (stage, world_size, family). A
# ZeRO performance arm carries the pointer to the artifact that validated THAT
# shape; an unmatched shape carries no pointer and is recorded unvalidated
# rather than borrowing another shape's evidence.
DS_EVIDENCE = {
    (1, 1, "p1080"): "D35_ds_zero1_observe_1gpu_p1080/results/deepspeed_observe_zero1.json",
    (2, 1, "p1080"): "D36_ds_zero2_observe_1gpu_p1080/results/deepspeed_observe_zero2.json",
    (1, 1, "rtx"):   "D37_ds_zero1_observe_1gpu_rtx/results/deepspeed_observe_zero1.json",
    (2, 1, "rtx"):   "D38_ds_zero2_observe_1gpu_rtx/results/deepspeed_observe_zero2.json",
    (1, 2, "p1080"): "D23_ds_zero1_observe_2gpu/results/deepspeed_observe_zero1.json",
    (2, 2, "p1080"): "D24_ds_zero2_observe_2gpu/results/deepspeed_observe_zero2.json",
    (1, 4, "p1080"): "D25_ds_zero1_observe_4gpu/results/deepspeed_observe_zero1.json",
    (2, 4, "p1080"): "D26_ds_zero2_observe_4gpu/results/deepspeed_observe_zero2.json",
    (1, 6, "p1080"): "D27_ds_zero1_observe_6gpu/results/deepspeed_observe_zero1.json",
    (2, 6, "p1080"): "D28_ds_zero2_observe_6gpu/results/deepspeed_observe_zero2.json",
    (1, 2, "rtx"):   "D29_ds_zero1_observe_rtx2/results/deepspeed_observe_zero1.json",
    (2, 2, "rtx"):   "D30_ds_zero2_observe_rtx2/results/deepspeed_observe_zero2.json",
}


def ds_evidence_path(cfg: dict) -> str:
    if cfg["recipe"] not in ("zero1", "zero2"):
        return ""
    stage = int(cfg["recipe"][-1])
    family = "rtx" if cfg["group"].startswith("rtx") else "p1080"
    rel = DS_EVIDENCE.get((stage, int(cfg["world_size"]), family))
    if not rel:
        return ""
    q = W / "runs" / rel
    return str(q) if q.exists() else ""


def adapter_for(recipe: str, world_size: int) -> str:
    \"\"\"torchrun for anything that builds a process group; plain otherwise.

    world_size 1 still uses torchrun for the wrapper recipes, because a DDP or
    FSDP singleton must be the real wrapper, not a relabelled single-GPU run.
    \"\"\"
    if recipe == "single":
        return "plain"
    return recipe


def build_argv(cfg: dict, case: str) -> list[str]:
    wl = cfg["workload"]
    argv = list(RECIPE_ARGS[cfg["recipe"]])
    if cfg["recipe"] in ("zero1", "zero2"):
        stage = cfg["recipe"][-1]
        argv += ["--ds-config", str(T / f"configs/ds_zero{stage}.json")]
    argv += [
        "--precision", "amp_fp16" if cfg["dtype"] == "amp_fp16" else "fp32",
        "--batch-size", str(wl["micro_batch_per_gpu"]),
        "--seq-len", str(wl["seq_len"]),
        "--warmup", str(cfg["warmup_steps"]),
        "--steps", str(cfg["measured_steps"]),
        "--memory-fraction", str(cfg["memory"]["memory_fraction"]),
        "--group", cfg["group"], "--out-name", case,
    ]
    return argv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="1,2,3,4,5,6")
    ap.add_argument("--recipes", default=",".join(TRAINING))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan = json.loads((W / "bundle/plan/run_plan.json").read_text())
    want = set(a.recipes.split(","))
    cfgs = [c for c in plan["configurations"] if c["recipe"] in want]
    blocks = [int(b) for b in a.blocks.split(",")]
    ledger = W / "ledger" / "blocks.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)

    print(f"configs={len(cfgs)} blocks={blocks} launches={len(cfgs)*len(blocks)}")
    launched = skipped = refused = failed = 0

    for b in blocks:
        order = list(cfgs)
        # Seed from the block index alone: the order is reproducible and the
        # seed is recorded, so the interleaving is auditable rather than ad hoc.
        rng = random.Random(20261016 + b)
        rng.shuffle(order)
        print(f"\\n===== block {b}  seed={20261016 + b}  n={len(order)} =====", flush=True)
        for k, cfg in enumerate(order, 1):
            case = f"P{b}_{cfg['case_id']}"
            d = W / "runs" / case
            if d.exists():
                skipped += 1
                continue
            now = int(time.time())
            tmo = int(cfg["per_case_timeout_s"])
            if now >= CUTOFF or now + tmo + 120 >= EXPIRY:
                print(f"  STOP: launch cutoff reached before {case}", flush=True)
                _write(ledger, {"event": "cutoff_stop", "block": b, "case": case,
                                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                print(f"\\nlaunched={launched} skipped={skipped} refused={refused} failed={failed}")
                return 0
            devs = ",".join(str(x) for x in cfg["device_indices"])
            argv = ["bash", str(W / "tools/gpu_attempt.sh"), case, devs,
                    str(tmo), adapter_for(cfg["recipe"], cfg["world_size"])] + \\
                   build_argv(cfg, case)
            if a.dry_run:
                print(f"  [{k}/{len(order)}] DRY {case} devs={devs}")
                continue
            env = dict(os.environ)
            ev = ds_evidence_path(cfg)
            if ev:
                env["BENCH_DS_EVIDENCE"] = ev
            else:
                env.pop("BENCH_DS_EVIDENCE", None)
            t0 = time.time()
            r = subprocess.run(argv, capture_output=True, text=True, env=env)
            el = time.time() - t0
            rc_file = d / "exit_code"
            rc = int(rc_file.read_text().strip()) if rc_file.exists() else None
            tag = "ok" if rc == 0 else ("REFUSED" if rc == 90 else f"rc={rc}")
            if rc == 0:
                launched += 1
            elif rc == 90:
                refused += 1
            else:
                failed += 1
            print(f"  [{k}/{len(order)}] {case:36s} {tag:9s} {el:5.0f}s", flush=True)
            _write(ledger, {"event": "launch", "block": b, "case": case,
                            "devices": cfg["device_indices"], "recipe": cfg["recipe"],
                            "exit_code": rc, "elapsed_s": round(el, 1),
                            "ds_evidence": ev or None,
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    print(f"\\nlaunched={launched} skipped={skipped} refused={refused} failed={failed}")
    return 0


def _write(path: pathlib.Path, obj: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\\n")


if __name__ == "__main__":
    sys.exit(main())

"""
