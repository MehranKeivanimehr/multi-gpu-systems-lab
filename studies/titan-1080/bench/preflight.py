"""
Stage-1 preflight gate. Fails loudly and early rather than mid-suite.

Checks, in order:
  1. torch sees CUDA at all.
  2. The wheel carries kernels usable by every compute capability on this box:
     6.1 (Pascal: 7x 1080 Ti + TITAN Xp) and 7.5 (Turing: 2x TITAN RTX).
     Note this is NOT a literal "sm_61 in arch_list" test. CUDA guarantees
     binary compatibility across *minor* revisions within a major revision, so
     a cubin built for sm_60 executes on a 6.1 device. torch 2.4.1+cu121 ships
     sm_50/60/70/75/80/86/90 and therefore covers 6.1 via sm_60 and 7.5
     exactly. The check below applies that rule per device; step 4 then proves
     it empirically, which is the authoritative gate.
  3. bf16 support, so the report can state plainly why fp32/fp16 was chosen.
  4. Every GPU can actually execute a kernel -- a real matmul per device,
     which is what catches a missing-arch fallback that only shows up at launch.
  5. All 10 GPUs are idle enough to use (no foreign compute processes).
  6. Topology groups resolve as expected.

Exit code 0 means the suite is cleared to run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from common import (  # noqa: E402
    FOREIGN_MEM_LIMIT_MIB,
    compute_processes,
    query_gpus,
    torch_env,
    write_result,
)
from topology import inventory, parse_topo, resolve_groups  # noqa: E402

def arch_covers(cap: str, archs: list[str]) -> tuple[bool, str]:
    """Can any compiled arch in `archs` run on a device of capability `cap`?

    CUDA rule: a cubin for sm_{M}{m} runs on a device sm_{M}{n} when the major
    version M matches and m <= n. PTX in the wheel can also JIT upward, but we
    only credit real cubins here -- JIT would show up as a slow first launch.
    """
    major, minor = (int(x) for x in cap.split("."))
    best = None
    for a in archs:
        if not a.startswith("sm_"):
            continue
        digits = a[3:]
        if not digits.isdigit():
            continue
        am, an = int(digits[0]), int(digits[1:] or 0)
        if am == major and an <= minor:
            if best is None or an > best[1]:
                best = (am, an)
    if best is None:
        return False, "none"
    exact = best[1] == minor
    return True, f"sm_{best[0]}{best[1]}" + ("" if exact else " (minor-compat)")

GREEN, RED, YEL, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}PASS{OFF}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{OFF}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YEL}NOTE{OFF}  {msg}")


def main() -> int:
    report: dict[str, object] = {}
    failures: list[str] = []

    print("\n=== 1. torch / CUDA ===")
    env = torch_env()
    report["torch_env"] = env
    print(
        f"  torch {env['torch']}  cuda {env['cuda_runtime']}  "
        f"cudnn {env['cudnn']}  nccl {env['nccl']}"
    )
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
        failures.append("no_cuda")
        print(json.dumps(report, indent=2, default=str))
        return 1
    ok(f"CUDA available, {env['device_count']} devices visible")

    print("\n=== 2. compiled architectures (HARD GATE) ===")
    archs = env["arch_list"]
    print(f"  arch_list = {archs}")
    caps_present = sorted({g.compute_cap for g in inventory()})
    report["arch_coverage"] = {}
    for cap in caps_present:
        covered, via = arch_covers(cap, archs)
        report["arch_coverage"][cap] = {"covered": covered, "via": via}
        n_dev = sum(1 for g in inventory() if g.compute_cap == cap)
        if covered:
            ok(f"capability {cap} ({n_dev} GPUs) served by {via}")
        else:
            fail(
                f"capability {cap} ({n_dev} GPUs) has NO usable cubin -- "
                f"re-pin torch before running"
            )
            failures.append(f"uncovered_cap_{cap}")

    print("\n=== 3. dtype capability ===")
    bf16 = torch.cuda.is_bf16_supported()
    report["bf16_supported"] = bool(bf16)
    if bf16:
        warn("bf16 emulation-inclusive flag is true on the current device; not a native-speed claim")
    else:
        ok("bf16 unsupported as expected -> suite uses FP32 + AMP-fp16 only")

    print("\n=== 4. per-GPU kernel execution ===")
    gpus = inventory()
    dev_results = []
    for g in gpus:
        t0 = time.time()
        try:
            with torch.cuda.device(g.index):
                dtype_flags = {"bf16_including_emulation": bool(torch.cuda.is_bf16_supported(including_emulation=True)),
                               "bf16_native_api_flag": bool(torch.cuda.is_bf16_supported(including_emulation=False))}
            a = torch.randn(512, 512, device=f"cuda:{g.index}")
            b = torch.randn(512, 512, device=f"cuda:{g.index}")
            c = (a @ b).sum().item()
            torch.cuda.synchronize(g.index)
            dt = time.time() - t0
            finite = c == c and abs(c) != float("inf")
            if finite:
                ok(f"GPU{g.index} {g.short:<18} cap {g.compute_cap}  matmul OK ({dt:.2f}s)")
                dev_results.append({"index": g.index, "name": g.short, "ok": True, "dtype_flags": dtype_flags})
            else:
                fail(f"GPU{g.index} {g.short} produced non-finite result")
                failures.append(f"gpu{g.index}_nonfinite")
                dev_results.append({"index": g.index, "name": g.short, "ok": False})
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:
            fail(f"GPU{g.index} {g.short}: {type(exc).__name__}: {exc}")
            failures.append(f"gpu{g.index}_error")
            dev_results.append(
                {"index": g.index, "name": g.short, "ok": False, "error": str(exc)}
            )
    report["per_gpu"] = dev_results

    print("\n=== 5. cluster idle check ===")
    mine = os.getpid()
    foreign = [p for p in compute_processes() if p["pid"] != mine]
    report["foreign_procs"] = foreign
    if foreign:
        for p in foreign:
            warn(f"GPU{p['gpu']} foreign compute PID {p['pid']} ({p['used_mib']:.0f} MiB)")
    else:
        ok("no foreign compute processes on any GPU")
    snap = query_gpus()
    report["snapshot_pre"] = snap
    busy = [r for r in snap if (r["memory.used"] or 0) > FOREIGN_MEM_LIMIT_MIB]
    if busy:
        for r in busy:
            warn(f"GPU{r['index']} holds {r['memory.used']:.0f} MiB")
    else:
        ok(f"all {len(snap)} GPUs below {FOREIGN_MEM_LIMIT_MIB} MiB used")

    print("\n=== 6. topology groups ===")
    links = parse_topo()
    report["links"] = {f"{a}-{b}": v for (a, b), v in links.items()}
    groups = resolve_groups()
    report["groups"] = groups
    for key, info in groups.items():
        if "error" in info:
            fail(f"{key}: {info['error']}")
            failures.append(f"group_{key}")
            continue
        ok(
            f"{key:<10} {info['model']:<18} n={info['n_gpus']} "
            f"devices={info['cuda_visible_devices']:<12} worst_link={info['worst_link']}"
        )

    report["failures"] = failures
    report["cleared"] = not failures
    path = write_result(report, "00_preflight")

    print(f"\n  report -> {path}")
    if failures:
        print(f"\n{RED}PREFLIGHT FAILED{OFF}: {failures}\n")
        return 1
    print(f"\n{GREEN}PREFLIGHT CLEARED{OFF} -- suite may run.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
