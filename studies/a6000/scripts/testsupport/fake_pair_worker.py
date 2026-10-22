"""Harmless CPU stand-in for one R5/C5 worker, used ONLY via the supervisor's stub-gated test hook (Candidate 17).
It starts `world` child "rank" processes inside the SAME worker scope (they inherit it), which write rank-identified
records taken from the expected-identity environment the supervisor really set. No GPU, NCCL or training happens.
Usage: fake_pair_worker.py <attempt dir> <world>; MGBENCH_TEST_SCENARIO selects normal | leader_first | timeout |
missing_rank | duplicate_rank | inconsistent_counts | foreign_on_5; Candidate 18 adds ignore_term | delayed_metadata |
metadata_timeout | alter_record | duplicate_record | remove_record | change_after_r5 | slow (each rank checks its own
capacity with the candidate's window_contract.worker_capacity_check and a file barrier before any "step")."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

attempt, world = Path(sys.argv[1]), int(sys.argv[2])
scenario = os.environ.get("MGBENCH_TEST_SCENARIO", "normal")
if world > 1:
    uuids = os.environ["MGBENCH_EXPECTED_GPU_UUIDS"].split(",")
    pcis = os.environ["MGBENCH_EXPECTED_GPU_PCIS"].split(",")
    idx = [int(x) for x in os.environ["MGBENCH_EXPECTED_GPU_INDICES"].split(",")]
else:
    uuids, pcis, idx = ([os.environ["MGBENCH_EXPECTED_GPU_UUID"]], [os.environ["MGBENCH_EXPECTED_GPU_PCI"]],
                        [int(os.environ["MGBENCH_EXPECTED_GPU_INDEX"])])
window = attempt.parent.parent
edir = window / "enrollment"
if world == 1 and scenario == "alter_record":          # tamper with the sealed record (detected at the next gate)
    rp = edir / "ENROLLMENT_RECORD.json"; os.chmod(rp, 0o644); rp.write_text(rp.read_text().replace("GPU-0", "GPU-9"))
if world == 1 and scenario == "duplicate_record":
    (edir / "ENROLLMENT_RECORD_2.json").write_text("{}")
if world == 1 and scenario == "remove_record":
    (edir / "ENROLLMENT_RECORD.json").unlink()
if world == 1 and scenario == "change_after_r5":
    Path(os.environ["MGBENCH_TEST_LATE_FLAG"]).write_text("on")
if scenario in ("delayed_metadata", "metadata_timeout"):
    Path(os.environ["MGBENCH_TEST_SYSTEMCTL_DELAY"]).write_text("2" if scenario == "delayed_metadata" else "5")
(attempt / "identity_prestep.json").write_text(json.dumps({"stage": "TEST HOOK stand-in for the pre-step", "problems": []}))
tmpl = json.loads(Path(os.environ[f"MGBENCH_TEST_CONTROL_TEMPLATE_W{world}"]).read_text())
if scenario == "inconsistent_counts" and world > 1:
    tmpl["per_rank"][1]["measured"]["counter_updates"] = 4
RANK = r"""
import json, os, signal, sys, time
from pathlib import Path
from mgbench.window_contract import worker_capacity_check
attempt, r, world, scen = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
row = json.loads(sys.argv[5])
(attempt / f"rank{r}.cgroup").write_text(open("/proc/self/cgroup").read())
st = json.loads(open(os.environ["MGBENCH_TEST_NVML_STATE"]).read())
tt = (st.get("torch_total_bytes") or {}).get(str(row["nvml_index"]), 50897289216)
cap = worker_capacity_check(json.loads(os.environ["MGBENCH_CAPACITY_CONTRACT"]), r, tt)
(attempt / f"rank{r}_capacity.json").write_text(json.dumps(cap, default=str))
end = time.time() + 10                         # barrier: no rank "steps" until every rank's capacity record exists
while time.time() < end and not all((attempt / f"rank{k}_capacity.json").exists() for k in range(world)):
    time.sleep(0.05)
caps = [json.loads((attempt / f"rank{k}_capacity.json").read_text()) for k in range(world)]
if not all(c.get("pass") is True for c in caps):
    sys.exit(3)                                # stops before any optimizer step
if scen == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if scen in ("timeout", "leader_first", "ignore_term", "delayed_metadata", "metadata_timeout"):
    time.sleep(600)
if scen == "slow":
    time.sleep(6)
if scen == "missing_rank" and r == 1:
    sys.exit(0)
claim = 0 if (scen == "duplicate_rank" and r == 1) else r
rec = {"rank": claim, "identity_row": dict(row, rank=claim), "update_state": "verified_sampled_every_base_call",
       "identity_ok": True}
(attempt / f"rank{r}_record.json").write_text(json.dumps(rec))
if scen == "foreign_on_5":
    time.sleep(8)
"""
kids = []
for r in range(world):
    row = {"rank": r, "nvml_index": idx[r], "uuid": uuids[r], "pci_bus_id": pcis[r]}
    kids.append(subprocess.Popen([sys.executable, "-B", "-s", "-c", RANK, str(attempt), str(r), str(world), scenario,
                                  json.dumps(row)]))
if scenario == "leader_first":
    sys.exit(0)                                  # the launcher exits first; its ranks keep running in the scope
if scenario == "foreign_on_5" and world > 1:      # appears only while the pair (GPUs 4-5) is running
    time.sleep(1)
    Path(os.environ["MGBENCH_TEST_FOREIGN_FLAG"]).write_text("on")
rc = max(k.wait() for k in kids)
if True:                                         # rank-0 outputs are always written; missing_rank drops ONLY rank 1
    (attempt / "grad_b1.f32").write_bytes(b"\0" * 16)
    (attempt / "dtheta_final.f32").write_bytes(b"\0" * 16)
    (attempt / "control_record.json").write_text(json.dumps(tmpl))
sys.exit(rc)
