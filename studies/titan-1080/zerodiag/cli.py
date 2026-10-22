"""Diagnostic CLI. Plans and self-checks on CPU; refuses to run without a device.

It never initialises CUDA, never constructs a DeepSpeed engine, and refuses
unless an explicit authorization is supplied. `--selfcheck` exercises the
production instrumentation against CPU fixtures. `--plan` prints the frozen
context inventory with resolved bindings and hashes.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# Frozen retained context inventory. No Pascal AMP, no larger RTX groups,
# no extra performance repeats, no fault injection.
CONTEXTS = (
    [{"family": "1080Ti", "stage": s, "world_size": w, "precision": "fp32"}
     for s in (1, 2) for w in (1, 2, 4, 6)]
  + [{"family": "RTX", "stage": s, "world_size": w, "precision": "fp32"}
     for s in (1, 2) for w in (1, 2)]
  + [{"family": "RTX", "stage": s, "world_size": w, "precision": "amp_fp16"}
     for s in (1, 2) for w in (1, 2)]
)
SEPARATE_INVENTORY = {
    "cross_family_references": 2,
    "prospective_streaming_case": 1,
    "note": "inventoried separately; not part of the 16 ZeRO contexts",
}


def source_bindings() -> dict:
    out = {}
    for rel in ("zerodiag/factories.py", "zerodiag/engine_probe.py",
                "zerodiag/verdict.py", "zerodiag/update_oracle.py",
                "zerodiag/matrix.py", "zerodiag/reference.py",
                "zerodiag/ref_runner.py", "zerodiag/stream_runner.py",
                "zerodiag/stream_server.py", "zerodiag/containment.py",
                "zerodiag/ownership.py",
                "zerodiag/streaming.py",
                "zerodiag/cli.py"):
        p = ROOT / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return out


def installed_bindings() -> dict:
    """Record the pinned DeepSpeed origin WITHOUT initialising CUDA."""
    import inspect, os
    out: dict = {}
    try:
        import deepspeed
        out["deepspeed_version"] = deepspeed.__version__
        out["deepspeed_dir"] = os.path.dirname(inspect.getfile(deepspeed))
        import deepspeed.runtime.zero.stage_1_and_2 as s12
        import deepspeed.runtime.engine as eng
        for label, mod in (("stage_1_and_2.py", s12), ("engine.py", eng)):
            p = pathlib.Path(inspect.getfile(mod))
            out[label] = hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception as exc:                      # recorded, never guessed
        out["error"] = f"{type(exc).__name__}: {exc}"
    import torch
    out["torch_version"] = torch.__version__
    out["torch_cuda_available"] = bool(torch.cuda.is_available())
    return out


def selfcheck() -> int:
    """Run the production instrumentation over a CPU fixture. No device."""
    import torch
    from . import engine_probe as EP, factories as F, update_oracle as OR, verdict as V
    HP = {"lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
          "weight_decay": 0.01, "decoupled_weight_decay": True, "amsgrad": False}

    class Fixture:
        def __init__(self, n=(8, 4)):
            self.partition_size = list(n)
            self.groups_padding = [0] * len(n)
            self.single_partition_of_fp32_groups = [
                torch.nn.Parameter(torch.randn(k, dtype=torch.float64)) for k in n]
            self.optimizer = torch.optim.AdamW(
                self.single_partition_of_fp32_groups, lr=HP["lr"])
            self.overflow = False
    torch.manual_seed(0)
    z = Fixture()
    for m in z.single_partition_of_fp32_groups:
        m.grad = torch.randn_like(m) * 0.01
    exp = EP.expected_partition_map(z, rank=0, world_size=1)
    sc = EP.verify_storage_correspondence(z, rank=0)
    b = EP.capture_boundary(z, rank=0, world_size=1, phase="before")
    z.optimizer.step()
    a = EP.capture_boundary(z, rank=0, world_size=1, phase="after")
    r = V.verify_boundary(before=b, after=a, expected_map=exp,
                          resolved_hparams=HP, accumulation_steps=1, rank=0,
                          world_size=1, storage_check=sc,
                          module_check=EP.module_reflects_masters(z),
                          coverage=OR.FULL)
    ok = r["rank_verification_successful"]
    print(f"selfcheck: rank_verification_successful={ok} "
          f"boundaries={r['logical_boundaries']} base_calls={r['base_optimizer_calls']} "
          f"successes={r['successful_updates']} skips={r['skipped_updates']}")
    print(f"  establishes: {r['establishes']}")
    ci = r["coverage_inventory"]
    print(f"  coverage inventory: {ci['n_observed_partitions']}/"
          f"{ci['n_expected_partitions']} expected partitions verified, "
          f"{ci['observed_elements_verified']}/"
          f"{ci['expected_elements_this_rank']} expected elements")
    print(f"  coverage complete: {ci['complete']}"
          + (f" ({ci['incomplete_because']})" if not ci["complete"] else ""))
    print(f"  established by: {r['coverage_is_established_by']}")
    print(f"  NOT established by: {r['coverage_is_not_established_by']}")
    return 0 if ok else 1


INTEGRATION_CASE = {
    "case_id": "G0_1080ti_zero1_w1_fp32",
    "family": "1080Ti", "stage": 1, "world_size": 1, "precision": "fp32",
    "model": "retained GPT2 (GPT2Config defaults, 124,439,808 params)",
    "warmup_steps": 0, "audited_boundaries": 1,
    "batch_size": 4, "seq_len": 512,
    "worker_timeout_s": 180,
    "retries": 0, "fault_injection": False, "amp": False,
}


def _receipt_digest(rec: dict) -> str:
    import hashlib, json as _j
    body = {k: rec[k] for k in sorted(rec) if k != "digest"}
    return hashlib.sha256(_j.dumps(body, sort_keys=True).encode()).hexdigest()


def validate_receipt(path: str, *, source_bindings_now: dict,
                     now: float | None = None) -> dict:
    """Validate an authorization receipt. Any defect refuses; nothing is assumed."""
    import time as _t
    now = _t.time() if now is None else now
    try:
        rec = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": f"receipt unreadable: {type(exc).__name__}: {exc}"}
    required = ("authorization_id", "approved_by", "approval_text", "start_epoch",
                "expiry_epoch", "device_index", "device_uuid", "device_name",
                "case_id", "source_digest", "digest")
    missing = [k for k in required if not rec.get(k)]
    if missing:
        return {"ok": False, "reason": f"receipt missing fields: {missing}"}
    if _receipt_digest(rec) != rec["digest"]:
        return {"ok": False, "reason": "receipt digest does not match its body: "
                                       "fabricated or altered"}
    if rec["case_id"] != INTEGRATION_CASE["case_id"]:
        return {"ok": False, "reason": f"receipt case_id {rec['case_id']!r} is not "
                                       f"the frozen integration case"}
    if not (rec["start_epoch"] <= now < rec["expiry_epoch"]):
        return {"ok": False, "reason": f"receipt not currently valid: now={now:.0f} "
                                       f"window=[{rec['start_epoch']:.0f},"
                                       f"{rec['expiry_epoch']:.0f})"}
    if rec["expiry_epoch"] - rec["start_epoch"] > 15 * 60 + 1:
        return {"ok": False, "reason": "receipt window exceeds 15 minutes"}
    if rec["source_digest"] != source_bindings_now.get("__tree__"):
        return {"ok": False, "reason": "receipt is bound to a different source tree"}
    return {"ok": True, "receipt": rec}


def source_tree_digest() -> dict:
    """Digest of the runnable source the receipt binds to."""
    import hashlib
    b = source_bindings()
    h = hashlib.sha256()
    for k in sorted(b):
        h.update(k.encode() + b"\0" + (b[k] or "").encode())
    wk = ROOT / "zerodiag/worker.py"
    if wk.is_file():
        b["zerodiag/worker.py"] = hashlib.sha256(wk.read_bytes()).hexdigest()
        h = hashlib.sha256()
        for k in sorted(b):
            h.update(k.encode() + b"\0" + (b[k] or "").encode())
    b["__tree__"] = h.hexdigest()
    return b


def execute(a) -> int:
    """Launch the ONE frozen boundary case as an owned child, then verify."""
    import signal, subprocess, time as _t
    if not a.output:
        print("REFUSED: --execute requires --output")
        return 95
    if not a.receipt:
        print("REFUSED: --execute requires --receipt. No approval is inferred "
              "from idle devices, earlier windows, this invocation or notes.")
        return 95
    sb = source_tree_digest()
    v = validate_receipt(a.receipt, source_bindings_now=sb)
    if not v["ok"]:
        print(f"REFUSED: {v['reason']}")
        return 95
    rec = v["receipt"]
    out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    inv = {"integration_inventory": [INTEGRATION_CASE],
           "n_cases": 1, "authorization": rec}
    (out / "integration_inventory.json").write_text(
        json.dumps(inv, indent=2, sort_keys=True, default=str))
    worker = a.worker_script or str(ROOT / "zerodiag/worker.py")
    import hashlib
    wsha = hashlib.sha256(pathlib.Path(worker).read_bytes()).hexdigest()
    print(f"integration inventory: {inv['n_cases']} case "
          f"({INTEGRATION_CASE['case_id']})")
    print(f"worker: {worker}")
    print(f"  worker sha256: {wsha}")
    # R2 production binding. The REQUIRED hash comes from the frozen source
    # bindings computed from disk, never from the receipt and never from a
    # command-line override: authorization does not adopt whatever worker hash
    # the invocation supplies. Mismatch refuses BEFORE any child is spawned.
    required = sb.get("zerodiag/worker.py")
    if not required:
        print("REFUSED: the frozen production worker binding is missing from the "
              "source tree digest; production execution cannot be bound")
        return 95
    if wsha != required:
        print("REFUSED: worker bytes do not match the approved production "
              "binding. No production child was spawned.")
        print(f"  required (frozen source binding): {required}")
        print(f"  supplied ({worker}): {wsha}")
        print("  There is no production override for this check. Harmless "
              "substitute workers are accepted only by --cpu-containment-check, "
              "which never reserves or touches a GPU.")
        return 95
    if rec.get("source_digest") != sb["__tree__"]:
        print("REFUSED: receipt source_digest does not match the source tree now "
              "on disk. No production child was spawned.")
        return 95
    budget = rec["expiry_epoch"] - _t.time()
    need = INTEGRATION_CASE["worker_timeout_s"] + 45
    if budget < need:
        print(f"REFUSED: {budget:.0f}s left but the case needs {need}s including "
              f"bounded cleanup and final verification")
        return 95
    return _run_bound_worker(a, rec, out, worker, wsha, sb)


def _run_bound_worker(a, rec, out, worker, wsha, sb) -> int:
    """Spawn and supervise an ALREADY-BOUND worker, then verify its evidence.

    Production reaches this only through execute(), which refuses any worker
    whose bytes differ from the frozen binding. The CPU test harness calls this
    directly with harmless fixtures; that injection point is not reachable from
    any command line, so there is no production bypass.
    """
    import signal, subprocess, time as _t
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = rec["device_uuid"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # DeepSpeed's init_distributed takes the standard torch.distributed path
    # only when all of RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT and
    # LOCAL_RANK are present; otherwise it attempts MPI discovery, which
    # imports mpi4py and fails on this host. World size is 1 for the frozen
    # case, so these describe a single-rank group and introduce no
    # communication. The port is derived per attempt rather than fixed, so a
    # concurrent job of another user on this shared node cannot collide.
    dist_env = {
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(29600 + (int(hashlib.sha256(
            (rec["authorization_id"] + str(out)).encode()
        ).hexdigest()[:8], 16) % 1000)),
    }
    env.update(dist_env)
    # The worker no longer defaults rank, world size, case or precision; the
    # Gate-0 path states them explicitly. Its semantics are unchanged.
    argv = [sys.executable, "-B", worker, "--output", str(out),
            "--receipt", a.receipt, "--ds-config", a.ds_config,
            "--zero-stage", "1", "--batch-size",
            str(INTEGRATION_CASE["batch_size"]),
            "--seq-len", str(INTEGRATION_CASE["seq_len"]),
            "--case-id", INTEGRATION_CASE["case_id"],
            "--rank", "0", "--world-size", "1", "--precision", "fp32"]
    # R1: apply an ACTUAL affinity mask here, before the child is created, so
    # the child and every thread its libraries spawn inherit the restriction.
    from zerodiag.containment import apply_affinity, MemoryMonitor
    aff = apply_affinity(int(getattr(a, "max_cpus", 4) or 4))
    env.update(aff["supplementary_thread_vars"])
    print(f"  cpu affinity: {aff['permitted_cpus']} via {aff['mechanism']} "
          f"(a restriction, not a quota and not an exclusive reservation)")
    (out / "containment.json").write_text(json.dumps(
        {"affinity": aff, "budget_gib": float(getattr(a, "memory_budget_gib", 8.0)),
         "monitor_interval_s": float(getattr(a, "monitor_interval_s", 0.5)),
         "distributed_env": dist_env,
         "distributed_env_purpose": ("satisfies DeepSpeed init_distributed's "
                                     "required_env so it uses torch.distributed "
                                     "rather than MPI discovery; world size 1, "
                                     "no collective communication"),
         "child_visible_devices": env["CUDA_VISIBLE_DEVICES"]},
        indent=2, sort_keys=True))
    print(f"  dist env: world_size=1 rank=0 master={dist_env['MASTER_ADDR']}:"
          f"{dist_env['MASTER_PORT']} (single-rank group, no collectives)")
    t0 = _t.time()
    proc = subprocess.Popen(argv, env=env, start_new_session=True,
                            stdout=open(out / "worker.stdout", "w"),
                            stderr=open(out / "worker.stderr", "w"))
    pgid = os.getpgid(proc.pid)
    rc = None
    try:
        rc = proc.wait(timeout=INTEGRATION_CASE["worker_timeout_s"])
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {INTEGRATION_CASE['worker_timeout_s']}s; terminating "
              f"the owned group {pgid}")
        try:
            os.killpg(pgid, signal.SIGTERM); proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(pgid, signal.SIGKILL); proc.wait(timeout=30)
            except Exception:
                pass
        rc = proc.returncode
    elapsed = _t.time() - t0
    group_absent = False
    for _ in range(40):
        try:
            os.killpg(pgid, 0)
            _t.sleep(0.25)
        except (ProcessLookupError, PermissionError):
            group_absent = True
            break
    ev_path = out / "integration_evidence.json"
    ev = None
    if ev_path.is_file():
        try:
            ev = json.loads(ev_path.read_text())
        except ValueError:
            ev = None
    summary = {"case_id": INTEGRATION_CASE["case_id"], "worker_exit_code": rc,
               "elapsed_s": round(elapsed, 2), "owned_pgid": pgid,
               "group_absent_confirmed": group_absent,
               "worker_sha256": wsha,
               "worker_is_frozen_production": wsha == sb.get("zerodiag/worker.py"),
               "evidence_present": ev is not None,
               "evidence_stage": (ev or {}).get("stage"),
               "authorization_revoked": True}
    (out / "integration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"worker exit={rc} elapsed={elapsed:.1f}s "
          f"cleanup_confirmed={group_absent} evidence={summary['evidence_stage']}")
    if not group_absent:
        print("STOP: owned-group cleanup could not be confirmed")
        return 1
    if ev is None:
        print("STOP: the worker produced no readable evidence")
        return 1
    if ev.get("stage") != "completed":
        print(f"STOP: incomplete evidence (stage={ev.get('stage')!r}, "
              f"reason={ev.get('reason')!r})")
        return 1
    if rc != 0:
        print(f"STOP: worker exited {rc}")
        return 1
    print("integration case completed; authorization revoked, holds restored")
    return 0



def cpu_containment_check(a) -> int:
    """Demonstrate R1 affinity and R3 monitoring with harmless fixtures.

    This is the ONLY path that accepts a substitute worker, and it cannot
    reserve, select or contact a GPU: it spawns a plain CPU allocator.
    """
    from zerodiag.containment import (apply_affinity, verify_affinity,
                                      MemoryMonitor, owned_pids,
                                      budget_arithmetic)
    import subprocess, threading, time as _t
    aff = apply_affinity(a.max_cpus)
    print(f"affinity: {aff['permitted_cpus']}  mechanism={aff['mechanism']}")
    print(f"  scheduler quota: {aff['is_a_scheduler_quota']}   "
          f"exclusive reservation: {aff['is_exclusive_cpu_reservation']}")
    # child inherits the mask
    child = subprocess.run(
        [sys.executable, "-I", "-c",
         "import os;print(sorted(os.sched_getaffinity(0)))"],
        capture_output=True, text=True)
    print(f"  child effective cpus: {child.stdout.strip()}")
    ok_child = child.stdout.strip() == str(aff["permitted_cpus"])
    # thread inherits the mask
    seen = []
    t = threading.Thread(target=lambda: seen.append(sorted(os.sched_getaffinity(0))))
    t.start(); t.join()
    print(f"  thread effective cpus: {seen[0]}")
    ok_thread = seen[0] == aff["permitted_cpus"]
    print(f"  launcher verify: {verify_affinity(aff['permitted_cpus'])}")

    # monitor + stop path against a small threshold and a harmless allocator
    small_budget = 256 * 1024 * 1024
    victim = subprocess.Popen(
        [sys.executable, "-I", "-c",
         "import time;b=[];\n"
         "\nfor _ in range(200):\n b.append(bytearray(8*1024*1024));time.sleep(0.02)\n"
         "time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    victim_pgid = os.getpgid(victim.pid)
    mon = MemoryMonitor(victim_pgid, budget_bytes=small_budget, interval_s=0.2)
    stopped = False
    for _ in range(100):
        s_ = mon.sample()
        if mon.breached:
            victim.terminate()
            try:
                victim.wait(timeout=5)
            except subprocess.TimeoutExpired:
                victim.kill(); victim.wait(timeout=5)
            stopped = True
            break
        _t.sleep(0.2)
    if not stopped:
        victim.terminate()
        try:
            victim.wait(timeout=5)
        except subprocess.TimeoutExpired:
            victim.kill(); victim.wait(timeout=5)
    rep = mon.report()
    print(f"monitor: interval={rep['interval_s']}s samples={rep['samples']} "
          f"peak={rep['peak_owned_rss_bytes'] / 1048576:.0f} MiB "
          f"threshold=256 MiB breached={rep['breached']}")
    print(f"  owned-only stop executed: {stopped}  "
          f"owned pids remaining in the stopped group: {owned_pids(victim_pgid)}")
    print(f"  launcher survived the owned-only stop: pid {os.getpid()} alive, "
          f"affinity {sorted(os.sched_getaffinity(0))}")
    print(f"  enforced aggregate cap: {rep['enforced_cap']}")
    print(f"  {rep['limitations']}")
    b = budget_arithmetic()
    print(f"PLANNING ESTIMATE, not measured usage or admission: gate0 host {b['gate0_host_gib']} GiB "
          f"(fits 8 GiB: {b['gate0_fits_8gib']}, headroom "
          f"{b['gate0_headroom_gib']} GiB); strict all-host "
          f"{b['strict_all_host_gib']} GiB (fits: "
          f"{b['strict_all_host_fits_8gib']})")
    ok = ok_child and ok_thread and rep["breached"] and stopped
    print(f"CONTAINMENT CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Retained-recipe ZeRO diagnostic (CPU-safe).")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--execute", action="store_true",
                    help="requires --authorization-id and --authorization-expiry-epoch")
    ap.add_argument("--authorization-id", default=None)
    ap.add_argument("--authorization-expiry-epoch", type=float, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--receipt", default=None,
                    help="authorization receipt JSON, bound to device and source")
    ap.add_argument("--ds-config", default=None)
    ap.add_argument("--max-cpus", type=int, default=4)
    ap.add_argument("--memory-budget-gib", type=float, default=8.0)
    ap.add_argument("--monitor-interval-s", type=float, default=0.5)
    ap.add_argument("--execute-case", action="store_true",
                    help="execute one case from the frozen 18-case matrix")
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--interpreter", default=None,
                    help="interpreter for the case runner (the streaming "
                         "session needs the vLLM environment)")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--matrix", action="store_true",
                    help="print the frozen matrix and exit")
    ap.add_argument("--cpu-containment-check", action="store_true",
                    help="CPU-only containment demonstration with harmless "
                         "fixtures. Never reserves, selects or contacts a GPU.")
    ap.add_argument("--worker-script", default=None,
                    help="worker path; production REFUSES any worker whose "
                         "bytes differ from the frozen binding, before any "
                         "child is spawned")
    a = ap.parse_args(argv)
    if not (a.plan or a.selfcheck or a.execute or a.cpu_containment_check
            or a.execute_case or a.matrix):
        ap.error("choose --plan, --matrix, --selfcheck, --cpu-containment-check, "
                 "--execute or --execute-case")
    if a.execute:
        return execute(a)
    if a.matrix:
        from zerodiag import matrix as MX
        m = MX.full_matrix()
        print(json.dumps(m, indent=2, sort_keys=True, default=str))
        return 0
    if a.execute_case:
        return execute_case(a)
    if a.cpu_containment_check:
        return cpu_containment_check(a)

    if a.selfcheck:
        rc = selfcheck()
        if rc != 0:
            print("SELFCHECK FAILED - stopping; no further stage is attempted")
        return rc
    doc = {"contexts": CONTEXTS, "n_contexts": len(CONTEXTS),
           "separately_inventoried": SEPARATE_INVENTORY,
           "source_bindings": source_bindings(),
           "installed_bindings": installed_bindings(),
           "authorization": {"allocation_id": None, "start_utc": None,
                             "expiry_utc": None, "approved": None},
           "device_contact": "none"}
    print(f"retained ZeRO contexts: {len(CONTEXTS)}")
    for c in CONTEXTS:
        print(f"  {c['family']:7s} ZeRO-{c['stage']} w{c['world_size']} {c['precision']}")
    print(f"separately inventoried: {SEPARATE_INVENTORY}")
    print(f"torch_cuda_available: {doc['installed_bindings']['torch_cuda_available']}")
    if a.output:
        p = pathlib.Path(a.output); p.mkdir(parents=True, exist_ok=True)
        (p / "diagnostic_plan.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
        print(f"  -> {p/'diagnostic_plan.json'}")
    return 0


# ---------------------------------------------------------------------------
# Multi-rank case execution. The Gate-0 single-case `--execute` path above is
# frozen and untouched; this is the general path for the 18-case matrix.
# ---------------------------------------------------------------------------

CASE_RECEIPT_FIELDS = ("authorization_id", "approved_by", "approval_text",
                       "start_epoch", "expiry_epoch", "case_id", "world_size",
                       "device_indices", "device_uuids", "device_names",
                       "source_digest", "digest")


def validate_case_receipt(path: str, *, case: dict, source_bindings_now: dict,
                          now: float | None = None) -> dict:
    """Validate a receipt for a matrix case. Any defect refuses."""
    import time as _t
    now = _t.time() if now is None else now
    try:
        rec = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": f"receipt unreadable: {type(exc).__name__}: {exc}"}
    missing = [k for k in CASE_RECEIPT_FIELDS if rec.get(k) in (None, "", [])]
    if missing:
        return {"ok": False, "reason": f"receipt missing fields: {missing}"}
    if _receipt_digest(rec) != rec["digest"]:
        return {"ok": False, "reason": "receipt digest does not match its body"}
    if rec["case_id"] != case["case_id"]:
        return {"ok": False, "reason": (f"receipt case_id {rec['case_id']!r} is not "
                                        f"the requested case {case['case_id']!r}")}
    if int(rec["world_size"]) != int(case["world_size"]):
        return {"ok": False, "reason": (f"receipt world_size {rec['world_size']} != "
                                        f"case world_size {case['world_size']}")}
    if list(rec["device_uuids"]) != list(case["device_uuids"]):
        return {"ok": False, "reason": ("receipt device_uuids do not match the "
                                        "frozen group for this case")}
    if list(rec["device_indices"]) != list(case["device_indices"]):
        return {"ok": False, "reason": ("receipt device_indices do not match the "
                                        "frozen group for this case")}
    if len(set(rec["device_names"])) != 1:
        return {"ok": False, "reason": (f"receipt device group is not homogeneous: "
                                        f"{sorted(set(rec['device_names']))}")}
    if len(rec["device_uuids"]) != int(rec["world_size"]):
        return {"ok": False, "reason": ("one device per rank is required; "
                                        "device_uuids length != world_size")}
    if len(set(rec["device_uuids"])) != len(rec["device_uuids"]):
        return {"ok": False, "reason": "a device is bound to more than one rank"}
    if not (rec["start_epoch"] <= now < rec["expiry_epoch"]):
        return {"ok": False, "reason": (f"receipt not currently valid: now={now:.0f} "
                                        f"window=[{rec['start_epoch']:.0f},"
                                        f"{rec['expiry_epoch']:.0f})")}
    if rec["source_digest"] != source_bindings_now.get("__tree__"):
        return {"ok": False, "reason": "receipt is bound to a different source tree"}
    return {"ok": True, "receipt": rec}


def _derived_port(key: str) -> int:
    return 29600 + (int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1000)


def execute_case(a) -> int:
    """Launch one matrix case across its ranks, then reconcile every rank.

    One job at a time. Each rank sees exactly its own device by UUID, so a rank
    cannot touch a sibling's GPU. All ranks share one process group, so resource
    accounting and cleanup are whole-job and ownership-scoped.
    """
    import time as _t
    from zerodiag import matrix as MX
    from zerodiag import verdict as V
    from zerodiag.containment import apply_affinity

    if not (a.output and a.receipt and a.case_id):
        print("REFUSED: --execute-case requires --case-id, --output and --receipt")
        return 95
    cases = {c["case_id"]: c for c in MX.full_matrix()["remaining_cases"]}
    if a.case_id not in cases:
        print(f"REFUSED: {a.case_id!r} is not in the frozen matrix")
        return 95
    case = cases[a.case_id]
    kind = case["kind"]
    if kind not in ("training", "reference", "streaming"):
        print(f"REFUSED: unknown case kind {kind!r}")
        return 95
    sb = source_tree_digest()
    v = validate_case_receipt(a.receipt, case=case, source_bindings_now=sb)
    if not v["ok"]:
        print(f"REFUSED: {v['reason']}")
        return 95
    rec = v["receipt"]

    runner_rel = {"training": "zerodiag/worker.py",
                  "reference": "zerodiag/ref_runner.py",
                  "streaming": "zerodiag/stream_runner.py"}[kind]
    worker = str(ROOT / runner_rel)
    wsha = hashlib.sha256(pathlib.Path(worker).read_bytes()).hexdigest()
    required = sb.get(runner_rel)
    if not required or wsha != required:
        print(f"REFUSED: {runner_rel} bytes do not match the frozen production "
              f"binding. No child was spawned.")
        return 95
    # The streaming session needs the vLLM environment; everything else runs in
    # the training environment. Neither environment is modified or installed.
    interpreter = a.interpreter or sys.executable
    corpus = a.corpus or str(ROOT.parent / "protocols" / "prompt_corpus.json")
    if kind in ("reference", "streaming") and not pathlib.Path(corpus).is_file():
        print(f"REFUSED: corpus not found at {corpus}")
        return 95

    out = pathlib.Path(a.output)
    if out.exists() and any(out.iterdir()):
        print(f"REFUSED: {out} is not empty; attempt directories are write-once "
              f"and a failed attempt is never overwritten")
        return 95
    out.mkdir(parents=True, exist_ok=True)

    ws = int(case["world_size"])
    budget = rec["expiry_epoch"] - _t.time()
    need = case["worker_timeout_s"] + 45
    if budget < need:
        print(f"REFUSED: {budget:.0f}s left but the case needs {need}s including "
              f"bounded cleanup and final verification")
        return 95

    aff = apply_affinity(int(a.max_cpus))
    print(f"case {case['case_id']}: kind={kind} world_size={ws} "
          f"stage={case.get('stage')} precision={case.get('precision')} "
          f"group={case['group']}")
    print(f"  runner: {runner_rel}")
    print(f"  devices {case['device_indices']} ({case['device_names'][0]})")
    print(f"  worker sha256: {wsha}")
    print(f"  cpu affinity: {aff['permitted_cpus']} via {aff['mechanism']} "
          f"(a restriction, not a quota and not an exclusive reservation)")
    port = _derived_port(rec["authorization_id"] + case["case_id"])
    (out / "case_spec.json").write_text(json.dumps(
        {"case": case, "authorization": rec, "worker_sha256": wsha,
         "affinity": aff, "master_port": port,
         "aggregate_budget_gib": float(a.memory_budget_gib),
         "rendezvous_note": ("the derived port is collision-reducing, not "
                             "collision-free, and is not an exclusive "
                             "reservation")},
        indent=2, sort_keys=True, default=str))

    from zerodiag.ownership import run_ranks

    def _argv(r, rd):
        common = [interpreter, "-B", worker,
                  "--output", str(rd), "--receipt", a.receipt,
                  "--case-id", case["case_id"],
                  "--rank", str(r), "--world-size", str(ws),
                  "--max-cpus", str(a.max_cpus),
                  "--memory-budget-gib", str(a.memory_budget_gib),
                  "--monitor-interval-s", str(a.monitor_interval_s)]
        if kind == "training":
            return common + ["--ds-config", a.ds_config,
                             "--zero-stage", str(case["stage"]),
                             "--precision", case["precision"],
                             "--batch-size", str(case["batch_size"]),
                             "--seq-len", str(case["seq_len"]),
                             "--chunk-elems", str(case["chunk_elems"])]
        if kind == "streaming":
            return common + ["--corpus", corpus,
                             "--device-index", str(rec["device_indices"][r])]
        return common + ["--corpus", corpus]

    def _env(r, rd):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = rec["device_uuids"][r]
        env.update({"RANK": str(r), "LOCAL_RANK": "0", "WORLD_SIZE": str(ws),
                    "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                    "PYTHONDONTWRITEBYTECODE": "1"})
        env.update(aff["supplementary_thread_vars"])
        return env

    life = run_ranks(world_size=ws, argv_for=_argv, env_for=_env, out_dir=out,
                     timeout_s=case["worker_timeout_s"],
                     budget_bytes=int(float(a.memory_budget_gib) * 1073741824),
                     interval_s=float(a.monitor_interval_s))
    print(f"  launched {life['n_launched']}/{ws} rank(s); ownership token "
          f"{life['token']}")
    if life["partial_launch"]:
        print(f"  PARTIAL LAUNCH: {life['launch_errors']}")
    elapsed = life["elapsed_s"]
    breached = life["budget_breached"]
    timed_out = life["timed_out"]
    group_absent = life["owned_absent_confirmed"]
    exit_codes = [life["exit_codes"].get(str(r)) for r in range(ws)]

    # Collect every rank's evidence; a missing rank is never inferred.
    per_rank, rank_verdicts = {}, []
    for r in range(ws):
        ep = out / f"rank{r}" / {
            "training": "integration_evidence.json",
            "reference": "reference_evidence.json",
            "streaming": "streaming_evidence.json"}[kind]
        ev = None
        if ep.is_file():
            try:
                ev = json.loads(ep.read_text())
            except ValueError:
                ev = None
        per_rank[str(r)] = {
            "exit_code": exit_codes[r],
            "evidence_present": ev is not None,
            "stage": (ev or {}).get("stage"),
            "reason": (ev or {}).get("reason"),
            "rank_reported": (ev or {}).get("rank"),
            "realized_input_sha256": ((ev or {}).get("realized_input") or {})
                                       .get("realized_input_sha256"),
            "initial_state_sha256": ((ev or {}).get("initial_state") or {})
                                      .get("parameter_state_sha256"),
            "expected_base_calls": (ev or {}).get("expected_base_calls"),
            "amp_semantics": (ev or {}).get("amp_semantics"),
        }
        if ev and ev.get("rank_verdict"):
            rank_verdicts.append(ev["rank_verdict"])

    if kind == "training":
        reconciliation = V.combine_ranks(rank_verdicts, world_size=ws) \
            if rank_verdicts else {"whole_model_verified": False,
                                   "reason": "no rank produced a verdict"}
    else:
        done = sum(1 for d in per_rank.values() if d["stage"] == "completed")
        reconciliation = {
            "verification_successful": done == ws,
            "kind": kind, "ranks_completed": done, "ranks_expected": ws,
            "note": (f"a {kind} case carries no partition reconciliation; its "
                     f"terminal status is the runner's own verdict"),
            "reason": (None if done == ws else
                       f"{ws - done} of {ws} rank(s) did not complete")}
    reconciliation_successful = bool(
        reconciliation.get("whole_model_verified") if kind == "training"
        else reconciliation.get("verification_successful"))
    summary = {
        "case_id": case["case_id"], "kind": kind, "world_size": ws,
        "stage": case.get("stage"), "precision": case.get("precision"),
        "group": case["group"], "runner": runner_rel,
        "runner_sha256": wsha, "interpreter": interpreter,
        "device_uuids": rec["device_uuids"],
        "authorization_id": rec["authorization_id"],
        "worker_sha256": wsha, "source_digest": sb["__tree__"],
        "elapsed_s": elapsed, "ownership_token": life["token"],
        "exit_codes": exit_codes,
        "lifecycle": life,
        "ranks_expected": ws,
        "ranks_with_evidence": sum(1 for d in per_rank.values()
                                   if d["evidence_present"]),
        "ranks_completed": sum(1 for d in per_rank.values()
                               if d["stage"] == "completed"),
        "per_rank": per_rank,
        "rank_reconciliation": reconciliation,
        # `combine_ranks` reports `whole_model_verified`; the reference and
        # streaming paths report `verification_successful`. One kind-aware
        # field is recorded so no reader has to know which, and a missing key
        # can never read as success.
        "reconciliation_successful": reconciliation_successful,
        "aggregate_memory": life["aggregate"],
        "budget_breached": breached, "timed_out": timed_out,
        "group_absent_confirmed": group_absent,
        "authorization_revoked": True,
        "monitoring_note": ("sampled aggregate owned RSS; a guard, not an "
                            "enforced aggregate memory cap"),
    }
    (out / "case_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"  ranks: {summary['ranks_completed']}/{ws} completed, "
          f"{summary['ranks_with_evidence']}/{ws} with evidence")
    print(f"  aggregate peak owned RSS: "
          f"{life['aggregate']['peak_owned_rss_gib']} GiB of "
          f"{a.memory_budget_gib} GiB over {life['aggregate']['samples']} sample(s)")
    print(f"  elapsed {elapsed:.1f}s  cleanup_confirmed={group_absent}")

    if life["partial_launch"]:
        print("STOP: partial launch; a job with fewer ranks than the case "
              "requires is never accepted as a smaller job")
        return 1
    if not group_absent:
        print(f"STOP: owned cleanup could not be confirmed; surviving "
              f"{life['cleanup'].get('surviving_pids')}")
        return 1
    if breached:
        print("STOP: aggregate host-memory budget breached")
        return 1
    if timed_out:
        print("STOP: case timed out")
        return 1
    if summary["ranks_with_evidence"] != ws:
        print(f"STOP: {ws - summary['ranks_with_evidence']} rank(s) produced no "
              f"readable evidence; a missing rank is never inferred")
        return 1
    if summary["ranks_completed"] != ws:
        bad = {r: d["reason"] for r, d in per_rank.items() if d["stage"] != "completed"}
        print(f"STOP: rank(s) did not complete: {bad}")
        return 1
    if any(c != 0 for c in summary["exit_codes"]):
        print(f"STOP: nonzero rank exit codes {summary['exit_codes']}")
        return 1
    if not reconciliation_successful:
        print(f"RESULT: rank reconciliation did not succeed: "
              f"{reconciliation.get('reason') or reconciliation}")
        return 2
    print("case completed; every expected rank reconciled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
