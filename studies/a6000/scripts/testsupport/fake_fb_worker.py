"""Harmless CPU stand-in for one first-boundary worker (Candidate 19), used ONLY via the supervisor's stub-gated test
hook. One child "rank" process runs inside the SAME worker scope; it checks its capacity with the candidate's
window_contract.worker_capacity_check, then the launcher writes a first-boundary result.json from a test template.
No GPU, CUDA, NCCL or training. Usage: fake_fb_worker.py <attempt dir> <world=1>; MGBENCH_TEST_SCENARIO selects
normal | timeout | ignore_term | metadata_timeout | change_identity | slow."""
import json
import os
import subprocess
import sys
from pathlib import Path

import mgbench

attempt = Path(sys.argv[1])
scenario = os.environ.get("MGBENCH_TEST_SCENARIO", "normal")
(attempt / "identity_prestep.json").write_text(json.dumps({"stage": "TEST HOOK stand-in for the pre-step", "problems": []}))
if scenario == "metadata_timeout":
    Path(os.environ["MGBENCH_TEST_SYSTEMCTL_DELAY"]).write_text("5")
RANK = r"""
import json, os, signal, sys, time
from pathlib import Path
from mgbench.window_contract import worker_capacity_check
attempt, scen = Path(sys.argv[1]), sys.argv[2]
st = json.loads(open(os.environ["MGBENCH_TEST_NVML_STATE"]).read())
tt = (st.get("torch_total_bytes") or {}).get("4", 50897289216)
cap = worker_capacity_check(json.loads(os.environ["MGBENCH_CAPACITY_CONTRACT"]), 0, tt)
(attempt / "rank0_capacity.json").write_text(json.dumps(cap, default=str))
(attempt / "rank0.cgroup").write_text(open("/proc/self/cgroup").read())
if not cap["pass"]:
    sys.exit(3)                                   # stops before any allocation
if scen == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if scen in ("timeout", "ignore_term", "metadata_timeout"):
    time.sleep(600)
if scen == "slow":
    time.sleep(6)
"""
rc = subprocess.Popen([sys.executable, "-B", "-s", "-c", RANK, str(attempt), scenario]).wait()
res = json.loads(Path(os.environ["MGBENCH_TEST_FB_TEMPLATE"]).read_text())
spec = json.loads((attempt / "spec.json").read_text())
res.update(mem_fraction=spec.get("mem_fraction"), process={"mgbench_file": mgbench.__file__, "sys_executable": sys.executable},
           label="TEST HOOK: harmless CPU stand-in; no GPU, no training")
(attempt / "result.json").write_text(json.dumps(res))
if scenario == "change_identity":                 # GPU 4's identity changes after this case (mocked observation)
    Path(os.environ["MGBENCH_TEST_LATE_FLAG"]).write_text("on")
sys.exit(rc)
