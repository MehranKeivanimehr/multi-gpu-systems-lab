#!/usr/bin/env python
"""Lifecycle self-test of scope ownership, limits and cleanup with HARMLESS CPU processes only (sleep / short busy
loops). No GPU, no CUDA, no real OOM, no hardware fault. Every child runs in its own transient user scope with a
RuntimeMaxSec backstop. An unrelated sentinel (same UID, same MGBENCH_RUN_ID marker, outside every test scope) must
survive all cleanups.

Usage: lifecycle_selftest.py <new output json>
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

A6000 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A6000 / "src"))
from mgbench import scopes  # noqa: E402

RUN = f"mgbench-lifecycle-{os.getpid()}"
MARKER = {"MGBENCH_RUN_ID": "lifecycle-selftest-shared-marker"}
results = []


def record(name, ok, **detail):
    results.append({"scenario": name, "pass": bool(ok), **detail})
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, default=str)[:300]}", flush=True)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    st = Path(f"/proc/{pid}/stat").read_text() if Path(f"/proc/{pid}/stat").exists() else ""
    return bool(st) and st[st.rindex(")") + 2] != "Z"


def start(tag, cmd, **lim):
    lim = {"cpu_quota_pct": 100, "memory_max": "256M", "runtime_max_s": 60, "stop_timeout_s": 5, **lim}
    env = {**os.environ, **MARKER}
    return scopes.start_scope(f"{RUN}-{tag}", cmd, env=env, cwd="/tmp", stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, **lim)


def wait_pids(h, n, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        try:
            pids = scopes.cgroup_pids(h.cgroup)
            if len(pids) >= n:
                return pids
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.1)
    return scopes.cgroup_pids(h.cgroup) if h.cgroup else []


def main():
    out = Path(sys.argv[1])
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)       # never overwrite an earlier log
    sentinel = subprocess.Popen(["sleep", "900"], env={**os.environ, **MARKER}, start_new_session=True)
    record("sentinel started outside every test scope", alive(sentinel.pid), pid=sentinel.pid,
           cgroup=scopes.proc_cgroup(sentinel.pid))

    # 1. limits, affinity and CPU quota enforcement (4 busy threads for 3 s under a 1.5-CPU quota)
    probe = ("import os,threading,time,json;from pathlib import Path;"
             "cg=[l[3:] for l in open('/proc/self/cgroup').read().splitlines() if l.startswith('0::')][0];"
             "b=Path('/sys/fs/cgroup'+cg);u=lambda:int(dict(l.split() for l in (b/'cpu.stat').read_text().splitlines())['usage_usec']);"
             "u0,t0=u(),time.time();end=t0+3\n"
             "def spin():\n  while time.time()<end: pass\n"
             "ts=[threading.Thread(target=spin) for _ in range(4)];[t.start() for t in ts];[t.join() for t in ts]\n"
             "print(json.dumps({'cgroup':cg,'cpu.max':(b/'cpu.max').read_text().strip(),'memory.max':(b/'memory.max').read_text().strip(),"
             "'memory.swap.max':(b/'memory.swap.max').read_text().strip(),'affinity':sorted(os.sched_getaffinity(0)),"
             "'cpu_rate':(u()-u0)/1e6/(time.time()-t0)}))")
    r = subprocess.run(scopes.scope_command(f"{RUN}-limits", ["numactl", "--cpunodebind=1", sys.executable, "-c", probe],
                                            cpu_quota_pct=150, memory_max="256M", runtime_max_s=30, stop_timeout_s=5),
                       capture_output=True, text=True, timeout=60)
    try:
        got = json.loads(r.stdout.strip().splitlines()[-1])
        node1 = set(range(64, 128)) | set(range(192, 256))
        ok = (got["cpu.max"] == "150000 100000" and got["memory.max"] == str(256 * 2**20)
              and got["memory.swap.max"] == "0" and set(got["affinity"]) <= node1 and got["cpu_rate"] <= 1.5 * 1.15)
        record("limits enforced: cpu.max, memory.max, swap 0, NUMA-1 affinity, CPU rate under quota", ok, **got)
    except Exception as exc:  # noqa: BLE001
        record("limits enforced", False, error=repr(exc), stdout=r.stdout[-500:], stderr=r.stderr[-500:])

    # 2. leader exits first, descendants keep running inside the scope
    p, h = start("leaderexit", ["sh", "-c", "sleep 300 & sleep 300 & exit 0"])
    pids = wait_pids(h, 2)
    p.wait(timeout=10)
    rc = scopes.cleanup_scope(h, grace_s=5)
    record("leader exits first; descendants cleaned via cgroup", rc["confirmed"] and not any(alive(x) for x in pids),
           identity=h.cgroup, pids=pids, receipt=rc["steps"][-1])

    # 3. descendants in new sessions
    p, h = start("setsid", ["sh", "-c", "setsid sleep 300 & setsid sh -c 'sleep 300 & wait' & wait"])
    pids = wait_pids(h, 4)
    sids = {x: os.getsid(x) for x in pids if alive(x)}
    rc = scopes.cleanup_scope(h, grace_s=5)
    record("descendants in new sessions cleaned via cgroup", rc["confirmed"] and len(set(sids.values())) >= 2
           and not any(alive(x) for x in pids), sessions=sids)

    # 4. timeout backstop: RuntimeMaxSec ends the scope without any cleanup call
    p, h = start("timeout", ["sleep", "300"], runtime_max_s=3)
    pids = wait_pids(h, 1)
    t0 = time.time()
    while time.time() - t0 < 20 and any(alive(x) for x in pids):
        time.sleep(0.2)
    took = time.time() - t0
    rc = scopes.cleanup_scope(h, grace_s=1)
    record("RuntimeMaxSec backstop ends the scope by itself", rc["confirmed"] and took < 15, seconds=round(took, 1))

    # 5. interruption: a controller receiving SIGINT cleans its scope
    ctl = ("import sys,signal,time,subprocess;sys.path.insert(0,%r);from mgbench import scopes;"
           "p,h=scopes.start_scope(%r,['sleep','300'],env=None,cwd='/tmp',stdout=None,stderr=None,cpu_quota_pct=100,"
           "memory_max='64M',runtime_max_s=60,stop_timeout_s=5);print(h.cgroup,flush=True)\n"
           "def stop(n,f):\n  r=scopes.cleanup_scope(h,grace_s=5);print('CONFIRMED' if r['confirmed'] else 'UNCONFIRMED',flush=True);sys.exit(0)\n"
           "signal.signal(signal.SIGINT,stop);time.sleep(60)") % (str(A6000 / "src"), f"{RUN}-interrupt")
    c = subprocess.Popen([sys.executable, "-c", ctl], stdout=subprocess.PIPE, text=True)
    cg = c.stdout.readline().strip()
    child = wait_pids(scopes.ScopeHandle(unit="x", leader_pid=0, cgroup=cg), 1) if cg and cg != "None" else []
    c.send_signal(signal.SIGINT)
    try:
        outp = c.communicate(timeout=30)[0]
    except subprocess.TimeoutExpired:
        c.kill()
        outp = "controller did not finish within 30 s"
    record("interruption: controller cleans its scope on SIGINT", "CONFIRMED" in outp and not any(alive(x) for x in child),
           cgroup=cg, output=outp.strip())

    # 6. metadata error: a wrong recorded identity is never acted on and reports unconfirmed
    p, h = start("metadata", ["sleep", "300"])
    pids = wait_pids(h, 1)
    wrong = scopes.ScopeHandle(unit=h.unit, leader_pid=h.leader_pid, cgroup=h.cgroup + "-tampered")
    rc_wrong = scopes.cleanup_scope(wrong, grace_s=1)
    still = all(alive(x) for x in pids)
    rc = scopes.cleanup_scope(h, grace_s=5)
    record("metadata mismatch -> unconfirmed, nothing signalled; real identity then cleans",
           not rc_wrong["confirmed"] and still and rc["confirmed"], wrong_receipt=rc_wrong["steps"][-1])

    # 7. supervisor dies (SIGKILL); its worker scope still ends at its own RuntimeMaxSec
    sup = ("import sys,time,subprocess;sys.path.insert(0,%r);from mgbench import scopes;"
           "p,h=scopes.start_scope(%r,['sleep','300'],env=None,cwd='/tmp',stdout=None,stderr=None,cpu_quota_pct=100,"
           "memory_max='64M',runtime_max_s=6,stop_timeout_s=2);print(h.cgroup,flush=True);time.sleep(120)") % (
        str(A6000 / "src"), f"{RUN}-orphan")
    sp, sh = start("sup", [sys.executable, "-c", sup], memory_max="256M")
    time.sleep(2)
    worker_cg = f"/{'/'.join(sh.cgroup.strip('/').split('/')[:-1])}/{RUN}-orphan.scope"
    wpids = scopes.cgroup_pids(worker_cg)
    scopes.run_systemctl(["kill", "--kill-whom=all", "--signal=SIGKILL", scopes.unit_name(sh.unit)])
    t0 = time.time()
    while time.time() - t0 < 25 and (any(alive(x) for x in wpids) or scopes.cgroup_pids(worker_cg)):
        time.sleep(0.2)
    record("supervisor death: worker scope still bounded by its own RuntimeMaxSec",
           bool(wpids) and not any(alive(x) for x in wpids) and time.time() - t0 < 20,
           worker_pids=wpids, seconds=round(time.time() - t0, 1))
    scopes.cleanup_scope(sh, grace_s=2)

    # 8. aggregate accounting with the smoke's real limits (idle sleepers only)
    a, ha = start("agg-sup", ["sleep", "30"], cpu_quota_pct=100, memory_max="2G")
    b, hb = start("agg-worker", ["sleep", "30"], cpu_quota_pct=1500, memory_max="46G")
    wait_pids(ha, 1), wait_pids(hb, 1)
    la, lb = scopes.cgroup_limits(ha.cgroup), scopes.cgroup_limits(hb.cgroup)
    cpu = sum(int(x["cpu.max"].split()[0]) for x in (la, lb)) / 100000
    mem = sum(int(x["memory.max"]) for x in (la, lb))
    ra, rb = scopes.cleanup_scope(ha, 3), scopes.cleanup_scope(hb, 3)
    record("aggregate supervisor+worker limits = 16 CPU-equivalents and 48 GiB, separate failure domains",
           cpu == 16 and mem == 48 * 2**30 and ha.cgroup != hb.cgroup and ra["confirmed"] and rb["confirmed"],
           cpu_equivalents=cpu, memory_bytes=mem, cgroups=[ha.cgroup, hb.cgroup])

    # 9. the sentinel with the same UID and marker survived everything
    record("unrelated sentinel (same UID and run marker) survived every cleanup", alive(sentinel.pid), pid=sentinel.pid)
    sentinel.terminate()
    sentinel.wait(timeout=10)
    leftovers = []
    for u in scopes.run_systemctl(["list-units", "--all", "--no-legend", "--plain", f"{RUN}*"]).stdout.splitlines():
        if u.strip():
            leftovers.append(u.split()[0])
    record("no test scope left behind", not [u for u in leftovers if " active " in f" {u} "], units=leftovers)
    with os.fdopen(fd, "w") as f:
        json.dump({"run": RUN, "results": results, "all_pass": all(r["pass"] for r in results)}, f, indent=1,
                  default=str)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
