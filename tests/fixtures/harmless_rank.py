"""A harmless rank fixture. Imports no model worker, never touches CUDA."""
import argparse, json, os, pathlib, sys, time
ap = argparse.ArgumentParser()
for f in ("--output", "--rank", "--world-size", "--case-id"):
    ap.add_argument(f, required=True)
ap.add_argument("--mode", default="ok")
ap.add_argument("--sleep", type=float, default=0.0)
a = ap.parse_args()
out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
rec = {"rank": int(a.rank), "world_size": int(a.world_size),
       "case_id": a.case_id, "pid": os.getpid(), "pgid": os.getpgid(0),
       "sid": os.getsid(0), "ppid": os.getppid(),
       "job_token": os.environ.get("ZERODIAG_JOB_TOKEN"),
       "env_seen": {k: os.environ.get(k) for k in
                    ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")},
       "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
(out / "rank_record.json").write_text(json.dumps(rec, sort_keys=True))
if a.mode == "child":
    import subprocess
    subprocess.Popen([sys.executable, "-c", "import time;time.sleep(600)"])
    (out / "spawned_child.flag").write_text("1")
if a.sleep:
    time.sleep(a.sleep)
if a.mode == "fail":
    print(f"rank {a.rank} failing on purpose", file=sys.stderr)
    sys.exit(7)
(out / "done.flag").write_text("1")
sys.exit(0)
