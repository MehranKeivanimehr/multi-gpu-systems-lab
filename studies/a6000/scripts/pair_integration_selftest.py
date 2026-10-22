#!/usr/bin/env python
"""Candidate 17 integrated pair-path self-test: runs the REAL supervisor (a scratch copy of this candidate's
studies/a6000) in real bounded user scopes, with a DECLARED NVML test stub and the stub-gated harmless fake worker.
No GPU, CUDA, NCCL or training. Usage: pair_integration_selftest.py <new output dir>"""
import datetime as dt, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

A = Path(__file__).resolve().parents[1]
E = A.parents[3]
PY = sys.executable
OUT = Path(sys.argv[1]); OUT.mkdir()
WORK = OUT / "cand"; (WORK / "studies").mkdir(parents=True)
shutil.copytree(A, WORK / "studies/a6000", ignore=shutil.ignore_patterns("results", "__pycache__"))
WA = WORK / "studies/a6000"
sys.path.insert(0, str(WA / "src"))
from mgbench.launch_controls import candidate_digest, create_exclusive  # noqa: E402
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
U4, U5 = "GPU-00000000-0000-0000-0000-000000000109", "GPU-00000000-0000-0000-0000-000000000103"   # U5 = CPU FIXTURE ONLY
uuids = {str(i): f"GPU-0000000{i}-0000-0000-0000-00000000000{i}" for i in range(8)}; uuids.update({"4": U4, "5": U5})
pcis = {str(i): f"00000000:{0x10 + i:02x}:00.0" for i in range(8)}; pcis.update({"4": "00000000:81:00.0", "5": "00000000:82:00.0"})
FEAS = E / "36_c16_design/pair_memory_feasibility.json"; MAP = E / "36_c16_design/global_input_map_r5_c5.json"
digs = json.loads(MAP.read_text())["global_row_digests"]
RECIPE = {"model": "qwen2.5-1.5b", "seq_len": 1024, "micro_batch": 1}


def template(world, ga):
    pr = [{"rank": r, "update_state": "verified_sampled_every_base_call", "identity_ok": True, "warmup": {"logical_attempts": 0},
           "measured": {"logical_attempts": 5, "microbatches": 5 * ga, "counter_updates": 5, "base_calls_per_attempt": [1] * 5,
                        "boundary_coverage": ["complete"] * 5, "microbatch_losses_all_finite": True}} for r in range(world)]
    gm = [{"step": k, "rank": r, "microstep": i, "global_row": i * world + r, "digest": digs[k][i * world + r], "raw_loss": 12.0}
          for k in range(5) for r in range(world) for i in range(ga)]
    return {"result_kind": "full_control_5update", "spec": dict(RECIPE, n_gpus=world, grad_accum=ga, measure_steps=5,
            warmup_steps=0, fp32_reference=True), "per_rank": pr, "global_sample_map": gm, "canonical_state_matches": True,
            "snapshots": {"writer_rank": 0, "files": ["dtheta_final.f32", "grad_b1.f32"], "dtheta_after_update": 5}}


for w, ga in ((1, 4), (2, 2)):
    (OUT / f"control_template_w{w}.json").write_text(json.dumps(template(w, ga)))
import yaml  # noqa: E402
cfg = yaml.safe_load((A / "configs/control_pair.yaml").read_text())
cfg.update(gpu_uuids={4: U4, 5: U5}, gpu_pci={4: pcis["4"], 5: pcis["5"]}, numa_node=1, reference_budget_bytes=25_000_000_000,
           min_free_disk_bytes=100_000_000_000)
cfg["limits"].update(worker_cpu_quota_pct=200, worker_memory="1G", worker_memory_bytes=2**30,
                     threads_per_rank={"R5_fp32_world1_ga4": 1, "C5_fp32_world2_ga2": 1})   # CPU-stage budget, aggregate
cfg["timing"].update(gate_s=30, child_s=25, grace_s=10, reap_s=5, final_reserve_s=30, telemetry_timeout_s=5)
(WA / "configs/test_pair.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
digest = candidate_digest(WORK)
bound = {str(FEAS): sha(FEAS), str(MAP): sha(MAP)}
exp = {"bound_files": bound, "global_row_digests": digs, "recipe": RECIPE}
sentinel = subprocess.Popen(["sleep", "900"], env=dict(os.environ, MGBENCH_RUN_ID="pair-selftest-shared-marker"))
results = []


def state(**kw):
    s = {"uuids": uuids, "pci": pcis, "foreign_pid": sentinel.pid, "foreign_gpu": 5}
    s.update(kw)
    return s


def run(name, scenario="normal", auth_over=None, auth_raw=None, expires_s=1200, st=None, pre_reserve=False,
        exp_sha_override=None, check=None):
    d = OUT / name; d.mkdir()
    (d / "nvml_state.json").write_text(json.dumps(st or state(foreign_flag=str(d / "foreign.flag"))))
    create_exclusive(d / "expectations.json", exp)
    now = dt.datetime.now(dt.timezone.utc)
    auth = {"id": f"selftest-{name}", "scope": "train-control-pair", "candidate_root": str(WORK), "candidate_digest": digest,
            "gpu_indices": [4, 5], "gpu_uuids": {"4": U4, "5": U5}, "case_gpus": cfg["case_gpus"], "max_gpus": 2,
            "max_concurrent_jobs": 1, "max_cases": 2, "issued_utc": (now - dt.timedelta(seconds=2)).isoformat(),
            "expires_utc": (now + dt.timedelta(seconds=expires_s)).isoformat(), "nonrenewable": True, "bound_files": bound,
            "expectations_sha256": sha(d / "expectations.json"), "config_sha256": sha(WA / "configs/test_pair.yaml")}
    auth.update(auth_over or {})
    if auth_raw is not None:
        (d / "authorization.json").write_text(auth_raw)
    else:
        create_exclusive(d / "authorization.json", auth)
    if name == "refuse_revoked":
        create_exclusive(Path(str(d / "authorization.json") + ".revoked"), {"revoked": True})
    out = d / "window"; out.mkdir()
    if pre_reserve:
        (out / "RESERVATION_gpu4_gpu5").write_text("someone else's cooperative marker\n")
    before = (out / "RESERVATION_gpu4_gpu5").read_text() if pre_reserve else None
    unit = f"mgbench-pair-sup-st-{name.replace('_', '-')}-{os.getpid()}"
    env = {k: os.environ[k] for k in ("HOME", "USER", "PATH", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS") if k in os.environ}
    pc = d / "private_cache"
    env.update(PYTHONPATH=str(WA / "scripts/testsupport/stub_nvml"), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", MGBENCH_TEST_NVML_STATE=str(d / "nvml_state.json"), MGBENCH_TEST_SCENARIO=scenario,
               MGBENCH_SUPERVISOR_TEST_WORKER=str(WA / "scripts/testsupport/fake_pair_worker.py"),
               MGBENCH_TEST_CONTROL_TEMPLATE_W1=str(OUT / "control_template_w1.json"),
               MGBENCH_TEST_CONTROL_TEMPLATE_W2=str(OUT / "control_template_w2.json"),
               MGBENCH_TEST_FOREIGN_FLAG=str(d / "foreign.flag"),
               **{k: str(pc / v) for k, v in (("TRITON_CACHE_DIR", "t"), ("TRITON_HOME", "th"), ("XDG_CACHE_HOME", "x"),
                                             ("TORCHINDUCTOR_CACHE_DIR", "i"), ("MPLCONFIGDIR", "m"))})
    cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}", "-p", "CPUQuota=100%", "-p",
           "MemoryMax=2G", "-p", "MemorySwapMax=0", "-p", "RuntimeMaxSec=400", "-p", "TimeoutStopSec=30", "--", PY, "-B", "-s",
           str(WA / "scripts/smoke_supervisor.py"), "--authorization", str(d / "authorization.json"), "--expectations",
           str(d / "expectations.json"), "--expectations-sha256", exp_sha_override or sha(d / "expectations.json"),
           "--out", str(out), "--sup-unit", unit, "--config", "configs/test_pair.yaml"]
    t0 = time.monotonic()
    with open(d / "console.log", "x") as log:
        rc = subprocess.run(cmd, cwd=WA, env=env, stdout=log, stderr=log, timeout=420).returncode
    summ = json.loads((out / "window_summary.json").read_text()) if (out / "window_summary.json").exists() else {}
    act = subprocess.run(["systemctl", "--user", "list-units", "--all", "--no-legend", "--plain", f"mgbench-pair-{out.name}*",
                          f"mgbench-pair-window*"], capture_output=True, text=True).stdout
    r = {"name": name, "scenario": scenario, "rc": rc, "seconds": round(time.monotonic() - t0, 1),
         "stopped_because": summ.get("stopped_because"), "reservation": summ.get("reservation"),
         "cases": [(c["case"], c["outcome"]) for c in summ.get("cases", [])], "test_hook_used": summ.get("test_hook_used"),
         "all_cleanups_confirmed": summ.get("all_cleanups_confirmed"), "revoked": summ.get("authorization_revoked_now"),
         "sentinel_alive": sentinel.poll() is None, "worker_units_active": " active " in f" {act} ",
         "pre_reservation_untouched": (before == (out / "RESERVATION_gpu4_gpu5").read_text()) if pre_reserve else None,
         "launched_case_dirs": sorted(p.name for p in (out / "cases").iterdir()) if (out / "cases").exists() else []}
    ok, why = check(r, out)
    r.update(pass_=bool(ok) and r["sentinel_alive"] and not r["worker_units_active"] and bool(summ), why=why)
    results.append(r)
    print(("PASS" if r["pass_"] else "FAIL"), name, rc, r["seconds"], "s |", str(r["stopped_because"])[:150], r["cases"], flush=True)


def refused(r, out, words):
    return (r["rc"] == 2 and str(r["stopped_because"]).startswith("preconditions:") and not r["launched_case_dirs"]
            and r["reservation"] == "never created" and words in str(r["stopped_because"])), words


def cgroup_shared(out, case_dir):
    cg = [(out / "cases" / case_dir / f"rank{r}.cgroup").read_text().strip() for r in range(2)]
    return cg[0] == cg[1] and cg[0].endswith(".scope") and "mgbench-pair-" in cg[0]


try:
    run("normal", check=lambda r, out: (r["cases"] == [("R5_fp32_world1_ga4", "completed_valid"), ("C5_fp32_world2_ga2", "completed_valid")]
        and r["test_hook_used"] is True and r["reservation"] == "released" and (out / "RESERVATION_gpu4_gpu5.released").exists()
        and cgroup_shared(out, "02_C5_fp32_world2_ga2"), "both ranks in ONE worker scope; joint reservation released"))
    run("leader_first_exit", "leader_first", check=lambda r, out: (r["cases"][0] == ("R5_fp32_world1_ga4", "unexpected_result")
        and r["all_cleanups_confirmed"] is True and r["cases"][1][1] == "not_run", "orphaned rank killed via cgroup; no success"))
    run("timeout", "timeout", check=lambda r, out: (r["cases"][0][1] == "stopped" and "time limit" in str(r["stopped_because"])
        and r["all_cleanups_confirmed"] is True, "child limit reached; cleanup confirmed; no success"))
    for scen in ("missing_rank", "duplicate_rank", "inconsistent_counts"):
        run(scen, scen, check=lambda r, out: (r["cases"] == [("R5_fp32_world1_ga4", "completed_valid"),
                                                             ("C5_fp32_world2_ga2", "unexpected_result")], "C5 refused"))
    run("foreign_on_gpu5", "foreign_on_5", check=lambda r, out: (r["cases"][1][1] == "stopped" and "gpu5" in str(r["stopped_because"]),
        "foreign process seen on GPU 5 during C5; ours stopped; the foreign (sentinel) process untouched"))
    run("gpu5_unsafe_during_settlement", st=state(unsafe_after_calls={"5": 16}), check=lambda r, out: (r["cases"][0][1] == "completed_valid" and
        r["cases"][1][1] == "not_run" and "gate" in str(r["stopped_because"]), "C5 never launched"))
    run("budget_exhausted_before_launch", expires_s=90, check=lambda r, out: (not r["launched_case_dirs"] and
        "remaining time" in str(r["stopped_because"]), "no launch"))
    run("preexisting_joint_reservation", pre_reserve=True, check=lambda r, out: (r["rc"] == 2 and not r["launched_case_dirs"]
        and r["pre_reservation_untouched"] is True and not (out / "RESERVATION_gpu4_gpu5.released").exists(),
        "refused; foreign marker untouched; nothing partial"))
    run("refuse_null", auth_raw=json.dumps({k: None for k in ("id", "scope", "gpu_indices")}), check=lambda r, o: refused(r, o, "unusable"))
    run("refuse_expired", auth_over={"issued_utc": "2026-10-16T00:00:00+00:00", "expires_utc": "2026-10-16T01:00:00+00:00"},
        check=lambda r, o: refused(r, o, "not valid now"))
    run("refuse_revoked", check=lambda r, o: refused(r, o, "revoked"))
    run("refuse_malformed", auth_raw="{not json", check=lambda r, o: refused(r, o, "unusable"))
    run("refuse_wrong_scope", auth_over={"scope": "train-first-boundary"}, check=lambda r, o: refused(r, o, "does not cover"))
    run("refuse_hash_mismatch", exp_sha_override="0" * 64, check=lambda r, o: refused(r, o, "expectations file differs"))
    run("refuse_uuid_swap", auth_over={"gpu_uuids": {"4": U5, "5": U4}}, check=lambda r, o: refused(r, o, "UUID binding"))
    run("refuse_old_single_gpu_authorization", auth_raw=None, auth_over={"scope": "train-first-boundary", "gpu_index": 4,
        "gpu_uuid": U4, "max_gpus": 1, "gpu_indices": None, "gpu_uuids": None, "case_gpus": None},
        check=lambda r, o: refused(r, o, "malformed or null"))
finally:
    alive = sentinel.poll() is None
    sentinel.send_signal(signal.SIGTERM); sentinel.wait(timeout=10)       # our own sentinel, ended by us
    rep = {"candidate_copy_digest": digest, "sentinel_alive_until_end": alive, "results": results,
           "label": "CPU-only: declared NVML stub + stub-gated fake worker; real supervisor, scopes, cleanup; no GPU/NCCL"}
    create_exclusive(OUT / "pair_integration_selftest.json", json.loads(json.dumps(rep, default=str)))
    print("sentinel alive until end:", alive, "| passed", sum(r["pass_"] for r in results), "of", len(results))
