"""Repeat-block driver routed through the AUDITED Python launcher.

This replaces the retired tools/gpu_attempt.sh bypass. Every case -- training,
auxiliary and engine adapters alike -- goes through
`studies/titan-1080/bench/launch.py: run_case`, which:

  * reserves the case directory before writing anything, so a later refusal
    cannot have touched prior evidence;
  * gates through `launch_gate.gate`, which refuses on a non-zero query exit, on
    empty or malformed output, on NaN/inf/out-of-range values, on unrequested or
    missing rows, on a changed device identity, on unaccounted VRAM, on an
    exceeded concurrency cap, on an unauthorized device and on an expired
    authorization -- and revalidates the FULL state on every settle poll;
  * holds a `DeviceReservation` with an aggregate lease, which the shell path
    had no equivalent of;
  * confirms ownership-scoped group cleanup, never killing by username, process
    name or broad ancestry, and never reclaiming an abandoned lock merely
    because a launcher died.

Nothing here launches anything by itself: `--execute` is required, and without
an explicit authorization (devices, expiry, UUID bindings) `run_case` refuses.
"""
from __future__ import annotations

import argparse, json, os, pathlib, random, sys, time

HERE = pathlib.Path(__file__).resolve().parent

# Explicit, ordered bindings for the layouts this driver actually ships in.
# No absolute paths and no ambiguous guessing: each candidate is recorded and
# the one used is printed, so a wrong binding is visible rather than silent.
BENCH_CANDIDATES = (
    HERE.parent / "source/studies/titan-1080/bench",        # publication tree
    HERE.parent / "multigpu-bench/studies/titan-1080/bench",  # correction candidate
    HERE.parent / "studies/titan-1080/bench",                 # delivered archive
)
PLAN_CANDIDATES = (
    "protocols/run_plan.json",        # publication tree
    "bundle/plan/run_plan.json",      # delivered archive
)


def resolve_bench() -> pathlib.Path:
    for c in BENCH_CANDIDATES:
        if (c / "launch.py").is_file():
            return c
    raise SystemExit(
        "REFUSED: no bench package with launch.py found. Tried:\n  "
        + "\n  ".join(str(c) for c in BENCH_CANDIDATES))


def resolve_plan(root: pathlib.Path) -> pathlib.Path:
    for rel in PLAN_CANDIDATES:
        if (root / rel).is_file():
            return root / rel
    raise SystemExit(
        f"REFUSED: no run plan found under {root}. Tried:\n  "
        + "\n  ".join(PLAN_CANDIDATES))


BENCH = resolve_bench()
sys.path.insert(0, str(BENCH))

RECIPE_ARGS = {
    "single": ["--mode", "single"], "ddp": ["--mode", "ddp"],
    "fsdp": ["--mode", "fsdp"],
    "zero1": ["--mode", "deepspeed", "--zero-stage", "1"],
    "zero2": ["--mode", "deepspeed", "--zero-stage", "2"],
}
TRAINING = tuple(RECIPE_ARGS)


def build_args(cfg: dict, case: str, bench: pathlib.Path) -> list[str]:
    wl = cfg["workload"]
    argv = list(RECIPE_ARGS[cfg["recipe"]])
    if cfg["recipe"] in ("zero1", "zero2"):
        argv += ["--ds-config", str(bench.parent / f"configs/ds_zero{cfg['recipe'][-1]}.json")]
    argv += ["--precision", "amp_fp16" if cfg["dtype"] == "amp_fp16" else "fp32",
             "--batch-size", str(wl["micro_batch_per_gpu"]),
             "--seq-len", str(wl["seq_len"]),
             "--warmup", str(cfg["warmup_steps"]),
             "--steps", str(cfg["measured_steps"]),
             "--memory-fraction", str(cfg["memory"]["memory_fraction"]),
             "--group", cfg["group"], "--out-name", case]
    return argv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="campaign root containing bundle/plan")
    ap.add_argument("--output", required=True, help="attempt output root")
    ap.add_argument("--reservation-root", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--blocks", default="1,2,3,4,5,6")
    ap.add_argument("--recipes", default=",".join(TRAINING))
    ap.add_argument("--concurrency-limit", type=int, required=True)
    ap.add_argument("--aggregate-limit", type=int, required=True)
    ap.add_argument("--authorized-devices", required=True,
                    help="comma-separated indices the allocation actually covers")
    ap.add_argument("--authorization-expiry-epoch", type=float, required=True)
    ap.add_argument("--uuid-binding", action="append", default=[],
                    help="INDEX=GPU-… ; required, one per authorized device")
    ap.add_argument("--execute", action="store_true",
                    help="without this the driver plans only and launches nothing")
    a = ap.parse_args(argv)

    import launch as L  # the audited launcher
    root = pathlib.Path(a.root).resolve()
    out = pathlib.Path(a.output).resolve()
    plan_path = resolve_plan(root)
    import hashlib
    print(f"bench package: {BENCH}")
    print(f"  launch.py sha256: {hashlib.sha256((BENCH/'launch.py').read_bytes()).hexdigest()}")
    print(f"run plan: {plan_path}")
    print(f"  plan sha256: {hashlib.sha256(plan_path.read_bytes()).hexdigest()}")
    plan = json.loads(plan_path.read_text())
    want = set(a.recipes.split(","))
    cfgs = [c for c in plan["configurations"] if c["recipe"] in want]
    authorized = [int(x) for x in a.authorized_devices.split(",")]
    uuids = L._parse_uuid_bindings(a.uuid_binding)
    bench = BENCH

    planned = 0
    inventory: list[dict] = []
    stopped = False
    stop_reason = None
    sequence = []
    for b in (int(x) for x in a.blocks.split(",")):
        order = list(cfgs)
        random.Random(20261016 + b).shuffle(order)
        for cfg in order:
            sequence.append((b, cfg))
    for b, cfg in sequence:
            case = f"P{b}_{cfg['case_id']}"
            devices = list(cfg["device_indices"])
            adapter = L.adapter_for_recipe(cfg["recipe"])
            planned += 1
            if not a.execute:
                print(f"  PLAN {case:38s} devices={devices} adapter={adapter}")
                inventory.append({"case": case, "outcome": "planned_only",
                                  "devices": devices, "adapter": adapter})
                continue
            if stopped:
                # Every planned case stays visible with an explicit disposition.
                inventory.append({"case": case, "outcome": "not_run_after_stop",
                                  "devices": devices, "adapter": adapter,
                                  "stop_reason": stop_reason})
                print(f"  {case:38s} NOT RUN (sequence stopped: {stop_reason})")
                continue
            try:
                rec = L.run_case(
                    case=case, out_root=str(out), devices=devices,
                    concurrency_limit=a.concurrency_limit,
                    authorized_devices=authorized, expected_uuids=uuids,
                    authorization_expiry_epoch=a.authorization_expiry_epoch,
                    python_exe=a.python, script=str(bench / "train_llm.py"),
                    script_args=build_args(cfg, case, bench),
                    reservation_root=a.reservation_root,
                    aggregate_limit=a.aggregate_limit,
                    adapter=adapter, timeout_s=float(cfg["per_case_timeout_s"]))
                cl = (rec.get("cleanup") or {}).get("group_absent_confirmed")
                print(f"  {case:38s} exit={rec.get('exit_code')} "
                      f"cleanup_confirmed={cl}")
                rc = rec.get("exit_code")
                bad_exit = (rc is None) or (not isinstance(rc, int)) or rc != 0
                uncertain_cleanup = cl is not True
                inventory.append({"case": case, "outcome": "attempted",
                                  "exit_code": rc,
                                  "cleanup_confirmed": cl,
                                  "devices": devices, "adapter": adapter})
                if bad_exit or uncertain_cleanup:
                    stopped = True
                    stop_reason = (f"case {case}: exit_code={rc!r}, "
                                   f"cleanup_confirmed={cl!r}")
                    print(f"  STOP: {stop_reason}")
            except Exception as exc:
                print(f"  {case:38s} REFUSED {type(exc).__name__}: {exc}")
                inventory.append({"case": case, "outcome": "refused",
                                  "error": f"{type(exc).__name__}: {exc}",
                                  "devices": devices, "adapter": adapter})
                stopped = True
                stop_reason = f"case {case} refused: {type(exc).__name__}"
                print(f"  STOP: {stop_reason}")
    attempted = [x for x in inventory if x["outcome"] == "attempted"]
    refused = [x for x in inventory if x["outcome"] == "refused"]
    notrun = [x for x in inventory if x["outcome"] == "not_run_after_stop"]
    print(f"\n{'planned' if not a.execute else 'attempted'} cases: {planned}")
    if not a.execute:
        print("nothing was launched: --execute was not supplied")
        return 0
    print(f"inventory: attempted={len(attempted)} refused={len(refused)} "
          f"not_run_after_stop={len(notrun)}")
    inv = pathlib.Path(a.output) / "case_inventory.json"
    inv.parent.mkdir(parents=True, exist_ok=True)
    inv.write_text(json.dumps(
        {"planned": planned, "stopped": stopped, "stop_reason": stop_reason,
         "inventory": inventory}, indent=2, sort_keys=True, default=str))
    print(f"  -> {inv}")
    if stopped:
        print(f"CAMPAIGN STOPPED: {stop_reason}")
        return 1
    if any(x.get("exit_code") != 0 or x.get("cleanup_confirmed") is not True
           for x in attempted):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
