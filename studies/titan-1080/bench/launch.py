"""Guarded launcher. Replaces the shell wrapper entirely.

Review R03/R04 found, in the previous shell wrapper:
  * `BENCH_EXPECT_UUIDS` was expanded unquoted, so it could append competing
    `--devices`/`--concurrency-limit` options that the CLI then honoured, while
    CUDA visibility was still exported from the ORIGINAL device list -- gate and
    workload could refer to different devices;
  * a refused launch still truncated an existing case's `gate.log`;
  * a case name of `../escaped` wrote outside the intended output root;
  * job control was enabled and the child's PGID was looked up only AFTER
    waiting, so a leader that had already exited could not be identified;
  * the signal condition required the child PGID to equal the PARENT's PGID;
  * there were no interruption/timeout traps, and the tested `terminate_owned`
    helper was not used by the shell path at all.

This module removes the shell from the path. Arguments are built as validated
argv arrays, so no string can inject or override an option. The case directory is
reserved atomically with O_EXCL semantics BEFORE any log file is opened, so a
refusal cannot touch prior evidence. The child is started in its own session via
`start_new_session=True`, its process-group id is captured immediately, and
normal exit, timeout and interruption all route through bounded,
ownership-scoped cleanup that refuses to signal the launcher's own group.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from typing import Any, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from launch_gate import (  # noqa: E402
    DeviceReservation, GateRefusal, gate, terminate_owned_group,
    authorization_remaining_s,
)

CASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DEFAULT_GRACE_S = 10.0


class LaunchRefusal(RuntimeError):
    """Raised before anything is written or executed."""


def validate_case_name(case: str) -> str:
    """A case identity must be a single safe path segment.

    `../escaped`, `a/b`, absolute paths, dot-names and empty strings all refuse.
    """
    if not isinstance(case, str) or not CASE_RE.match(case):
        raise LaunchRefusal(
            f"invalid case identity {case!r}: must match {CASE_RE.pattern} "
            f"(one path segment, no separators, no parent references)")
    if case in (".", "..") or os.sep in case or (os.altsep and os.altsep in case):
        raise LaunchRefusal(f"invalid case identity {case!r}")
    return case


def reserve_case_dir(out_root: str, case: str) -> str:
    """Atomically reserve a FRESH case directory, refusing any existing one.

    os.mkdir is atomic and fails with EEXIST, so this both reserves and refuses
    in one step. It runs before any log file is created, which is what keeps a
    refusal from truncating prior evidence.
    """
    validate_case_name(case)
    root = os.path.abspath(out_root)
    target = os.path.abspath(os.path.join(root, case))
    if os.path.commonpath([root, target]) != root:
        raise LaunchRefusal(
            f"case path {target!r} escapes the output root {root!r}")
    os.makedirs(root, exist_ok=True)
    try:
        os.mkdir(target, 0o700)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise LaunchRefusal(
                f"case directory already exists: {target!r}. Refusing so prior "
                f"evidence is never overwritten; choose a new attempt identity.")
        raise LaunchRefusal(f"cannot reserve case directory {target!r}: {exc}")
    return target


#: Launch adapters. The previous launcher ALWAYS built torchrun arguments, which
#: is correct only for distributed training/NCCL. HF and CNN are single-process
#: controllers, and vLLM must have exactly ONE parent even at TP=2 because the
#: engine spawns its own workers -- launching two copies of a two-GPU engine
#: would start two engines, each claiming both GPUs.
ADAPTERS = ("torchrun", "plain", "vllm_parent", "loopback_server")

#: expected OS child processes this launcher itself starts, per adapter
ADAPTER_PARENT_PROCESSES = {"torchrun": 1, "plain": 1,
                            "vllm_parent": 1, "loopback_server": 1}


def adapter_for_recipe(recipe: str) -> str:
    """Map a frozen-matrix recipe to its launch adapter."""
    # "single" never initialises a process group, so it needs no rendezvous.
    # The singleton DDP/FSDP/ZeRO baselines DO need one even at world size 1.
    if recipe in ("ddp", "fsdp", "zero1", "zero2", "deepspeed",
                  "nccl_allreduce_fixed_payload"):
        return "torchrun"
    if recipe in ("single", "cnn_forward_only", "hf_generate"):
        return "plain"
    if recipe.startswith("vllm_offline"):
        return "vllm_parent"
    if recipe == "vllm_loopback_streaming":
        return "loopback_server"
    raise LaunchRefusal(f"no launch adapter defined for recipe {recipe!r}")


def build_child_argv(python_exe: str, script: str, *,
                     nproc: int, master_port: int,
                     script_args: Sequence[str],
                     adapter: str = "torchrun") -> list[str]:
    """Build the child argv as a validated array. No shell, no string splitting.

    `adapter` selects the launch shape. Only "torchrun" multiplies processes with
    the device count; every other adapter starts exactly one parent regardless of
    how many devices are visible to it.
    """
    if adapter not in ADAPTERS:
        raise LaunchRefusal(f"unknown launch adapter {adapter!r}; expected one of {ADAPTERS}")
    if not isinstance(python_exe, str) or not os.path.isabs(python_exe):
        raise LaunchRefusal(f"interpreter must be an absolute path: {python_exe!r}")
    if not isinstance(script, str) or not os.path.isabs(script):
        raise LaunchRefusal(f"script must be an absolute path: {script!r}")
    if not isinstance(nproc, int) or nproc < 1:
        raise LaunchRefusal(f"nproc must be a positive int: {nproc!r}")
    for item in script_args:
        if not isinstance(item, str):
            raise LaunchRefusal(f"script argument is not a string: {item!r}")
        if "\x00" in item:
            raise LaunchRefusal("script argument contains a NUL byte")

    if adapter == "torchrun":
        if not isinstance(master_port, int) or not (1024 <= master_port <= 65535):
            raise LaunchRefusal(f"master_port out of range: {master_port!r}")
        return [python_exe, "-m", "torch.distributed.run",
                "--nnodes=1", f"--nproc_per_node={nproc}",
                "--master_addr=127.0.0.1", f"--master_port={master_port}",
                script, *script_args]

    # Single-parent adapters. torchrun arguments must NOT appear: a second engine
    # or controller copy would double-claim the reserved devices.
    if nproc != 1 and adapter in ("plain",):
        raise LaunchRefusal(
            f"adapter {adapter!r} is single-device; refusing nproc={nproc}")
    return [python_exe, "-B", script, *script_args]
def bind_receipt_to_plan(receipt: dict, *, devices: Sequence[int],
                         expected_uuids: dict[int, str]) -> dict:
    """The receipt must describe exactly the device namespace the child will see."""
    planned = list(devices)
    if list(receipt.get("devices") or []) != planned:
        raise LaunchRefusal(
            f"gate receipt devices {receipt.get('devices')!r} do not match the "
            f"planned child devices {planned!r}")
    # UUID visibility avoids assuming CUDA ordinal order equals nvidia-smi's
    # physical index order. BENCH_PHYS_DEVICES remains numeric for telemetry.
    visible = ",".join(expected_uuids[d] for d in planned)
    receipt_uuids = {int(k): v for k, v in (receipt.get("uuids") or {}).items()}
    for d in planned:
        if d not in receipt_uuids:
            raise LaunchRefusal(f"gate receipt has no UUID for planned device {d}")
        if expected_uuids.get(d) != receipt_uuids[d]:
            raise LaunchRefusal(
                f"planned UUID for device {d} ({expected_uuids.get(d)!r}) does not "
                f"match the gate receipt ({receipt_uuids[d]!r})")
    plan = {
        "cuda_visible_devices": visible,
        "cuda_device_order": "PCI_BUS_ID",
        "physical_devices": planned,
        "uuid_plan": {str(d): expected_uuids[d] for d in planned},
        "child_local_rank_to_physical": {str(i): d for i, d in enumerate(planned)},
    }
    receipt["child_visibility_plan"] = plan
    receipt["receipt_bound_to_plan"] = True
    return receipt


class _Interrupted(Exception):
    pass


def run_case(*, case: str, out_root: str, devices: Sequence[int],
             concurrency_limit: int, authorized_devices: Sequence[int],
             expected_uuids: dict[int, str], authorization_expiry_epoch: float,
             python_exe: str, script: str, script_args: Sequence[str],
             reservation_root: str, aggregate_limit: int,
             adapter: str = "torchrun",
             timeout_s: float = 1200.0, grace_s: float = DEFAULT_GRACE_S,
             gate_kwargs: dict | None = None,
             popen: Any = None, env: dict | None = None) -> dict:
    """Reserve, gate, launch, supervise and clean up. Returns an attempt record."""
    for name, value in (("timeout_s", timeout_s), ("grace_s", grace_s)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise LaunchRefusal(f"{name} must be finite and positive")
    if (type(aggregate_limit) is not int or type(concurrency_limit) is not int
            or not 1 <= aggregate_limit <= concurrency_limit):
        raise LaunchRefusal("aggregate limit must not exceed the authorized concurrency limit")
    authorization_remaining_s(authorization_expiry_epoch)
    # 1) identity and evidence protection FIRST: nothing is written before this
    #    succeeds, so a later refusal cannot have touched prior evidence.
    case_dir = reserve_case_dir(out_root, case)

    record: dict[str, Any] = {
        "case": case, "case_dir": case_dir,
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "devices": list(devices), "concurrency_limit": concurrency_limit,
        "adapter": adapter,
        "expected_parent_processes": ADAPTER_PARENT_PROCESSES[adapter],
        "torchrun_used": adapter == "torchrun",
        "aggregate_limit": aggregate_limit,
        "interpreter": python_exe, "script": script,
        "source_metadata": {
            "launcher": os.path.abspath(__file__),
            "launcher_sha256": _file_sha256(os.path.abspath(__file__)),
            "gate_module": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "launch_gate.py"),
        },
    }
    try:
        record["source_metadata"]["gate_sha256"] = _file_sha256(
            record["source_metadata"]["gate_module"])
        record["source_metadata"]["script_sha256"] = _file_sha256(script)
    except OSError:
        pass

    if adapter not in ADAPTERS:
        raise LaunchRefusal(f"unknown launch adapter {adapter!r}")
    argv = build_child_argv(python_exe, script,
                            adapter=adapter,
                            nproc=(len(devices) if adapter == "torchrun" else 1),
                            master_port=_port_for(case), script_args=script_args)
    record["argv"] = list(argv)

    # 2) reservation: a snapshot is not a reservation
    reservation = DeviceReservation(reservation_root, devices,
                                    aggregate_limit=aggregate_limit,
                                    owner=f"{case}:pid{os.getpid()}")
    with reservation:
        record["reservation"] = {"root": reservation_root,
                                 "held": list(reservation.held)}
        # 3) gate, with bindings and expiry mandatory on the launch path
        receipt = gate(list(devices), concurrency_limit=concurrency_limit,
                       authorized_devices=list(authorized_devices),
                       expected_uuids=expected_uuids, require_bindings=True,
                       authorization_expiry_epoch=authorization_expiry_epoch,
                       **(gate_kwargs or {}))
        receipt["reservation"] = {"root": reservation_root,
                                  "devices": list(devices)}
        receipt = bind_receipt_to_plan(receipt, devices=devices,
                                       expected_uuids=expected_uuids)
        record["gate_receipt"] = receipt
        _write_json(os.path.join(case_dir, "gate_receipt.json"), receipt)
        _write_json(os.path.join(case_dir, "command.json"), {
            "argv": argv, "utc_start": record["utc_start"],
            "devices": list(devices),
            "cuda_visible_devices": receipt["child_visibility_plan"]["cuda_visible_devices"],
            "source_metadata": record["source_metadata"],
        })

        child_env = dict(os.environ if env is None else env)
        child_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        child_env["CUDA_VISIBLE_DEVICES"] = \
            receipt["child_visibility_plan"]["cuda_visible_devices"]
        child_env["BENCH_PHYS_DEVICES"] = ",".join(str(d) for d in devices)

        out_path = os.path.join(case_dir, "stdout.log")
        err_path = os.path.join(case_dir, "stderr.log")
        spawn = popen or subprocess.Popen
        cleanup: dict[str, Any] = {"performed": False}
        child_pid = child_pgid = None
        previous_handlers: dict[int, Any] = {}

        def _on_signal(signum, frame):
            raise _Interrupted(f"received signal {signum}")

        proc = None
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.signal(sig, _on_signal)
            with open(out_path, "w") as out_fh, open(err_path, "w") as err_fh:
                # Leave a bounded termination/reap allowance inside the grant.
                remaining = authorization_remaining_s(authorization_expiry_epoch)
                cleanup_allowance = grace_s + max(1.0, grace_s) + 1.0
                effective_timeout = min(timeout_s, remaining - cleanup_allowance)
                if effective_timeout <= 0:
                    raise LaunchRefusal("insufficient authorization time for workload and cleanup")
                record["effective_timeout_s"] = effective_timeout
                record["cleanup_allowance_s"] = cleanup_allowance
                started = time.monotonic()
                # Fail closed on launcher failure: never free a lease until
                # the dedicated group is confirmed absent, not merely signalled.
                reservation.retain_on_exit = True
                proc = spawn(argv, stdout=out_fh, stderr=err_fh, env=child_env,
                             start_new_session=True, cwd=case_dir)
                # Capture immediately; metadata failures after this point MUST
                # take the same finally cleanup path as a wait failure.
                child_pid = child_pgid = proc.pid
                record["child_pid"] = child_pid
                record["child_pgid"] = child_pgid
                _write_json(os.path.join(case_dir, "child_identity.json"),
                            {"pid": child_pid, "pgid": child_pgid,
                             "own_pgid": os.getpgid(0), "dedicated_session": True})
                try:
                    wait_s = min(effective_timeout - (time.monotonic() - started),
                                 authorization_remaining_s(authorization_expiry_epoch)
                                 - cleanup_allowance)
                    if wait_s <= 0:
                        raise subprocess.TimeoutExpired(argv, effective_timeout)
                    rc = proc.wait(timeout=wait_s)
                    record["outcome"] = "completed"
                    record["exit_code"] = rc
                except subprocess.TimeoutExpired:
                    record["outcome"] = "timeout"
                except _Interrupted as interruption:
                    record["outcome"] = "interrupted"
                    record["interruption"] = str(interruption)
        finally:
            try:
                if child_pgid is not None:
                    cleanup = _bounded_cleanup(child_pgid, grace_s)
                    if proc is not None and record.get("outcome") != "completed":
                        record["exit_code"] = _reap(proc, grace_s)
                    cleanup["group_absent_confirmed"] = _confirm_group_absent(child_pgid)
                    reservation.retain_on_exit = not cleanup["group_absent_confirmed"]
                    cleanup["leases_retained_for_review"] = reservation.retain_on_exit
                    if reservation.retain_on_exit:
                        record["process_exit_code"] = record.get("exit_code")
                        record["exit_code"] = None
                        record["outcome"] = "cleanup_unverified"
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
        record["cleanup"] = cleanup
    record["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(os.path.join(case_dir, "attempt.json"), record)
    return record


def _bounded_cleanup(child_pgid: int | None, grace_s: float) -> dict:
    """Ownership-scoped, bounded cleanup of the dedicated child group."""
    result: dict[str, Any] = {"performed": True, "child_pgid": child_pgid,
                              "own_pgid": os.getpgid(0), "steps": []}
    if child_pgid is None:
        result["steps"].append({"action": "skipped",
                                "reason": "child identity unknown"})
        return result
    term = terminate_owned_group(child_pgid, signal_number=signal.SIGTERM)
    result["steps"].append({"signal": "SIGTERM", **term})
    if term.get("action") != "signalled":
        return result
    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        try:
            os.killpg(child_pgid, 0)
        except ProcessLookupError:
            result["steps"].append({"action": "group_gone"})
            return result
        except PermissionError:
            result["steps"].append({"action": "unverifiable",
                                    "reason": "permission denied on probe"})
            return result
        time.sleep(0.1)
    kill = terminate_owned_group(child_pgid, signal_number=signal.SIGKILL)
    result["steps"].append({"signal": "SIGKILL", **kill})
    return result


def _reap(proc, grace_s: float):
    try:
        return proc.wait(timeout=max(1.0, grace_s))
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def _confirm_group_absent(pgid, *, wait_s=1.0):
    """A signal receipt is not proof of termination. Unknown keeps leases."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _file_sha256(path: str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _port_for(case: str) -> int:
    import zlib
    return 29600 + (zlib.crc32(case.encode("utf-8")) % 900)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _parse_uuid_bindings(items: Sequence[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in items:
        index, _, uuid = item.partition("=")
        if not uuid:
            raise LaunchRefusal(
                f"--expect-uuid needs INDEX=GPU-UUID, got {item!r}")
        try:
            key = int(index)
        except ValueError:
            raise LaunchRefusal(f"--expect-uuid index is not an int: {index!r}")
        if key in out:
            raise LaunchRefusal(f"duplicate UUID binding for device {key}")
        out[key] = uuid
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--devices", required=True)
    p.add_argument("--concurrency-limit", required=True, type=int)
    p.add_argument("--authorized-devices", required=True)
    p.add_argument("--expect-uuid", action="append", default=[], required=True,
                   metavar="INDEX=GPU-UUID")
    p.add_argument("--authorization-expiry-epoch", required=True, type=float)
    p.add_argument("--reservation-root", required=True)
    p.add_argument("--aggregate-limit", required=True, type=int)
    p.add_argument("--python-exe", required=True)
    p.add_argument("--script", required=True)
    p.add_argument("--adapter", default="torchrun", choices=list(ADAPTERS))
    p.add_argument("--timeout", type=float, default=1200.0)
    p.add_argument("script_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)

    if os.environ.get("BENCH_GPU_AUTHORIZED") != "1":
        print("REFUSING: BENCH_GPU_AUTHORIZED is not set to 1. Set it only when a "
              "written allocation/concurrency/time approval is in force.", flush=True)
        return 91
    try:
        devices = [int(x) for x in args.devices.split(",") if x.strip()]
        allowed = [int(x) for x in args.authorized_devices.split(",") if x.strip()]
        uuids = _parse_uuid_bindings(args.expect_uuid)
        script_args = list(args.script_args)
        if script_args[:1] == ["--"]:
            script_args = script_args[1:]
        record = run_case(case=args.case, out_root=args.out_root, devices=devices,
                          concurrency_limit=args.concurrency_limit,
                          authorized_devices=allowed, expected_uuids=uuids,
                          authorization_expiry_epoch=args.authorization_expiry_epoch,
                          python_exe=args.python_exe, script=args.script,
                          script_args=script_args,
                          adapter=args.adapter,
                          reservation_root=args.reservation_root,
                          aggregate_limit=args.aggregate_limit,
                          timeout_s=args.timeout)
    except LaunchRefusal as refusal:
        print(f"LAUNCH REFUSED: {refusal}", flush=True)
        return 92
    except GateRefusal as refusal:
        print(f"GATE REFUSED: {refusal}", flush=True)
        return 90
    print(json.dumps(record, indent=2, sort_keys=True, default=str), flush=True)
    return 0 if record.get("exit_code") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
