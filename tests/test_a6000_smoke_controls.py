"""CPU tests for the GPU-4 smoke-stage controls (stage 20261016s). NVML and systemctl are stubbed; nothing here
touches a GPU, starts a scope or signals a process. These show decision logic, not hardware behaviour."""
import datetime as dt
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
A6000 = ROOT / "studies/a6000"
SRC = A6000 / "src"
UUID4 = "GPU-00000000-0000-0000-0000-000000000109"

try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


class NVMLError(Exception):
    pass


def nvml(n=8, power=20.0, temp=30, util=0, mem_used=2**20, uuids=None, procs=None, fail=(), sleep=None,
         limit_mw=300000):
    """pynvml stand-in; values may be per-index dicts. `procs`: {(index, kind): [pids]}."""
    if uuids is None:
        uuids = {i: f"GPU-0000000{i}-0000-0000-0000-00000000000{i}" for i in range(8)}
        uuids[4] = UUID4
    procs = procs or {}
    pick = lambda v, h: v.get(h, 20.0) if isinstance(v, dict) else v
    s = types.ModuleType("pynvml")
    s.NVMLError, s.nvmlMemory_v2, s.NVML_TEMPERATURE_GPU, s.NVML_CLOCK_SM = NVMLError, 2, 0, 1

    def call(name, fn):
        def w(*a, **k):
            if sleep and name in sleep:
                time.sleep(sleep[name])
            if name in fail:
                raise NVMLError(name)
            return fn(*a, **k)
        return w
    lst = lambda kind: (lambda h: [types.SimpleNamespace(pid=p, usedGpuMemory=2**30) for p in procs.get((h, kind), [])])
    s.nvmlInit = lambda: None
    s.nvmlDeviceGetCount = call("count", lambda: n)
    s.nvmlDeviceGetHandleByIndex = call("handle", lambda i: i)
    s.nvmlDeviceGetMemoryInfo = lambda h, version=None: types.SimpleNamespace(used=pick(mem_used, h), total=48 * 2**30)
    s.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=pick(util, h), memory=0)
    s.nvmlDeviceGetComputeRunningProcesses = call("compute", lst("compute"))
    s.nvmlDeviceGetGraphicsRunningProcesses = call("graphics", lst("graphics"))
    s.nvmlDeviceGetMPSComputeRunningProcesses = call("mps", lst("mps"))
    s.nvmlDeviceGetName = lambda h: b"NVIDIA RTX A6000"
    s.nvmlDeviceGetPciInfo = lambda h: types.SimpleNamespace(busId=b"00000000:81:00.0")
    s.nvmlDeviceGetPowerUsage = call("power", lambda h: pick(power, h) * 1000)
    s.nvmlDeviceGetEnforcedPowerLimit = call("limit", lambda h: limit_mw)
    s.nvmlDeviceGetTemperature = call("temp", lambda h, sensor: pick(temp, h))
    s.nvmlDeviceGetPerformanceState = lambda h: 8
    s.nvmlDeviceGetClockInfo = lambda h, c: 210
    s.nvmlDeviceGetUUID = call("uuid", lambda h: uuids[h])
    return s


class Stubbed(unittest.TestCase):
    def load(self, **kw):
        p = mock.patch.dict(sys.modules, {"pynvml": nvml(**kw)})
        p.start()
        self.addCleanup(p.stop)
        for name in [m for m in sys.modules if m == "mgbench" or m.startswith("mgbench.")]:
            del sys.modules[name]
        sys.path.insert(0, str(SRC))
        self.addCleanup(sys.path.remove, str(SRC))
        import mgbench.gpu_monitor as gm
        gm._initialized = False
        import mgbench.safety as safety
        return gm, safety


IDLE = {"poll_s": 0, "consecutive_polls": 1, "max_wait_s": 0, "max_mem_mib": 100, "max_util_pct": 0,
        "max_power_w": 60, "max_temp_c": 50, "min_free_ram_gib": 0, "query_timeout_s": 2.0,
        "node_guard": {"planned_w_per_gpu": 300, "start_budget_w": 1500, "abort_total_w": 1700, "abort_after_s": 5,
                       "max_load1": 10**6, "max_gpus_per_run": 2, "node_gpu_count": 8}}


class TelemetryRefusals(Stubbed):
    def test_value_rules(self):
        _, s = self.load()
        for bad in (None, float("nan"), float("inf"), -1.0, 10_000.0, True, "20"):
            self.assertIsNotNone(s.value_problem("power_w", bad), bad)
        self.assertIsNone(s.value_problem("power_w", 0.0))          # zero is a legitimate reading
        self.assertIsNone(s.value_problem("temp_c", 26))

    def test_idle_gate_refuses_nan_negative_implausible(self):
        for kw in ({"power": float("nan")}, {"temp": -5}, {"util": 150}, {"mem_used": 60 * 2**30},
                   {"power": float("inf")}):
            _, s = self.load(**kw)
            problems = s.idle_problems([4], IDLE)
            self.assertTrue(any("do not launch" in p for p in problems), (kw, problems))

    def test_mps_process_or_query_error_refuses(self):
        _, s = self.load(procs={(4, "mps"): [999]})
        self.assertTrue(any("processes" in p for p in s.idle_problems([4], IDLE)))
        _, s = self.load(fail=("mps",))
        self.assertTrue(any("unknown" in p for p in s.idle_problems([4], IDLE)))

    def test_mps_daemon_and_env_detected(self):
        _, s = self.load()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "4242").mkdir()
            (Path(d) / "4242/comm").write_text("nvidia-cuda-mps-control\n")
            (Path(d) / "17").mkdir()
            (Path(d) / "17/comm").write_text("python\n")
            self.assertTrue(s.mps_daemon_problems(Path(d)))
            with mock.patch.dict("os.environ", {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/x"}):
                (Path(d) / "4242/comm").write_text("bash\n")
                self.assertTrue(s.mps_daemon_problems(Path(d)))
            (Path(d) / "4242/comm").write_text("bash\n")
            with mock.patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
                os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)
                self.assertEqual(s.mps_daemon_problems(Path(d)), [])

    def test_hung_query_is_bounded_and_refuses(self):
        gm, s = self.load(sleep={"compute": 3.0})
        t0 = time.time()
        problems = s.idle_problems([4], dict(IDLE, query_timeout_s=0.3))
        self.assertLess(time.time() - t0, 2.0)
        self.assertTrue(any("unreadable" in p for p in problems))
        with self.assertRaises(gm.TelemetryTimeout):
            gm.bounded(time.sleep, 0.1, 1.0)

    def test_node_telemetry_rules(self):
        gm, s = self.load()
        snap = gm.node_snapshot(2.0)
        self.assertEqual(s.node_telemetry_problems(snap, 8, {4: UUID4}), [])
        self.assertTrue(s.node_telemetry_problems(snap, 8, {4: "GPU-00000000-0000-0000-0000-000000000110"}))
        bad = json.loads(json.dumps(snap))
        bad["gpus"] = {int(k): v for k, v in bad["gpus"].items()}
        bad["gpus"][2]["power_w"] = float("nan")
        self.assertTrue(any("gpu2" in p for p in s.node_telemetry_problems(bad, 8)))
        bad["gpus"][2]["power_w"] = 20.0
        bad["gpus"][6]["uuid"] = "GPU-not-a-uuid"
        self.assertTrue(any("malformed UUID" in p for p in s.node_telemetry_problems(bad, 8)))
        del bad["gpus"][7]
        self.assertTrue(any("expected 8" in p for p in s.node_telemetry_problems(bad, 8)))
        _, s7 = self.load(n=7)
        import mgbench.gpu_monitor as gm7
        self.assertTrue(s7.node_telemetry_problems(gm7.node_snapshot(2.0), 8))
        _, s_err = self.load(fail=("temp",))
        import mgbench.gpu_monitor as gme
        self.assertTrue(any("unreadable" in p for p in s_err.node_telemetry_problems(gme.node_snapshot(2.0), 8)))

    def test_power_watch_and_node_power_refuse_invalid(self):
        _, s = self.load(power={i: (float("nan") if i == 1 else 20.0) for i in range(8)})
        with self.assertRaises(ValueError):
            s.node_gpu_power()
        self.assertTrue(any("unreadable" in p for p in s.node_guard_problems([4], IDLE)))
        w = s.PowerWatch(IDLE["node_guard"])
        self.assertTrue(w.update(float("nan")))
        self.assertTrue(s.PowerWatch(IDLE["node_guard"]).update(-3.0))
        self.assertFalse(s.PowerWatch(IDLE["node_guard"]).update(1200.0))


def smoke_module():
    for name in [m for m in sys.modules if m == "mgbench" or m.startswith("mgbench.")]:
        del sys.modules[name]
    sys.path.insert(0, str(SRC))
    import mgbench.smoke as smoke
    return smoke


@unittest.skipUnless(HAVE_YAML, "yaml not installed")
class SmokeSettingsAndBudget(Stubbed):
    def setUp(self):
        self.load()
        import yaml
        self.yaml = yaml
        self.smoke = __import__("mgbench.smoke", fromlist=["x"])
        self.cluster = yaml.safe_load((A6000 / "configs/cluster.yaml").read_text())
        self.cfg = yaml.safe_load((A6000 / "configs/smoke_gpu4.yaml").read_text())

    def test_candidate_configs_match_the_approved_rules(self):
        self.assertEqual(self.smoke.settings_problems(self.cluster), [])
        self.assertEqual(self.smoke.smoke_config_problems(self.cfg), [])

    def test_changed_threshold_or_lifted_hold_refuses(self):
        c = json.loads(json.dumps(self.cluster))
        c["idle"]["min_free_ram_gib"] = 100
        self.assertTrue(self.smoke.settings_problems(c))
        c = json.loads(json.dumps(self.cluster))
        c["idle"]["node_guard"]["start_budget_w"] = 1600
        self.assertTrue(self.smoke.settings_problems(c))
        c = json.loads(json.dumps(self.cluster))
        c["launch_hold"]["active"] = False
        self.assertTrue(any("launch_hold" in p for p in self.smoke.settings_problems(c)))

    def test_smoke_config_refusals(self):
        for mutate in (lambda c: c.update(allowed_gpus=[4, 5]), lambda c: c.update(gpu_uuids={4: "GPU-bad"}),
                       lambda c: c["cases"].append({"id": "x", "framework": "ddp"}),
                       lambda c: c["cases"][0].update(diagnostic_fault={"kind": "nonfinite_loss"}),
                       lambda c: c["limits"].update(worker_cpu_quota_pct=1600),
                       lambda c: c["limits"].update(worker_memory_bytes=47 * 2**30),
                       lambda c: c["timing"].update(child_s=300),
                       lambda c: c["timing"].update(window_s=2700),
                       lambda c: c["cases"].reverse(),
                       lambda c: c.update(gpu_pci={5: "00000000:A1:00.0"})):
            c = json.loads(json.dumps(self.cfg))
            mutate(c)
            self.assertTrue(self.smoke.smoke_config_problems(c))

    def test_budget_is_the_approved_1720_seconds_and_refuses_cases_that_cannot_fit(self):
        t = self.cfg["timing"]
        total = t["startup_s"] + 4 * (t["gate_s"] + t["child_s"] + t["grace_s"] + t["reap_s"]) + t["final_reserve_s"]
        self.assertEqual(total, 1720)
        self.assertLessEqual(total, t["window_s"])
        self.assertTrue(self.smoke.case_budget(0, 370 + 120, t)["fits"])
        self.assertFalse(self.smoke.case_budget(0, 370 + 119, t)["fits"])
        self.assertGreater(4 * (420 + 60), 1800)          # the superseded 420-s plan cannot fit 30 minutes

    def test_busy_versus_fatal(self):
        busy, fatal = self.smoke.classify(["gpu4: temp 55 C", "node: other GPUs draw 1300 W + 1 x 300 W planned = 1600 W > start budget 1500 W",
                                           "gpu4: process list unavailable (x) -> unknown, do not launch",
                                           "authorization revoked", "MPS active (1:nvidia-cuda-mps-control) -> do not launch"])
        self.assertEqual(len(busy), 2)
        self.assertEqual(len(fatal), 3)

    def test_power_limit_must_equal_the_guard_assumption(self):
        snap = {"gpus": {4: {"power_limit_w": 300.0}}}
        self.assertEqual(self.smoke.gpu_power_limit_problems(snap, 4, 300), [])
        self.assertTrue(self.smoke.gpu_power_limit_problems({"gpus": {4: {"power_limit_w": 350.0}}}, 4, 300))
        self.assertTrue(self.smoke.gpu_power_limit_problems({"gpus": {4: {}}}, 4, 300))

    def test_worker_env_exposes_exactly_one_device_by_uuid(self):
        env = self.smoke.env_for_worker({"HOME": "/h", "CUDA_MPS_PIPE_DIRECTORY": "/x", "LD_PRELOAD": "evil"},
                                        src=SRC, uuid=UUID4, index=4, threads=8, cache=Path("/tmp/c"), run_id="r")
        # DeepSpeed 0.19.7 cuda_accelerator._get_nvml_gpu_id does list(map(int, CUDA_VISIBLE_DEVICES.split(",")));
        # the UUID form raised ValueError in smoke 20261016s case 3 (regression test).
        self.assertEqual(list(map(int, env["CUDA_VISIBLE_DEVICES"].split(","))), [4])
        self.assertEqual(env["MGBENCH_EXPECTED_GPU_INDEX"], "4")
        self.assertEqual(env["MGBENCH_EXPECTED_GPU_UUID"], UUID4)
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(env["MGBENCH_REQUIRE_SINGLE_DEVICE"], "1")
        self.assertNotIn("CUDA_MPS_PIPE_DIRECTORY", env)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")


@unittest.skipUnless(HAVE_YAML, "yaml not installed")
class AuthorizationBinding(Stubbed):
    def setUp(self):
        self.load()
        import mgbench.launch_controls as lc
        self.lc = lc
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.cand = self.tmp / "cand"
        (self.cand / "studies/a6000/src").mkdir(parents=True)
        (self.cand / "studies/a6000/src/x.py").write_text("a = 1\n")
        now = dt.datetime.now(dt.timezone.utc)
        self.auth = {"id": "t", "scope": "train-smoke", "candidate_root": str(self.cand),
                     "candidate_digest": lc.candidate_digest(self.cand), "gpu_index": 4, "gpu_uuid": UUID4,
                     "max_gpus": 1, "max_concurrent_jobs": 1, "max_cases": 4,
                     "issued_utc": (now - dt.timedelta(minutes=1)).isoformat(),
                     "expires_utc": (now + dt.timedelta(minutes=44)).isoformat(), "nonrenewable": True}

    def write(self, auth, name="auth.json"):
        p = self.tmp / name
        self.lc.create_exclusive(p, auth)
        return p

    def problems(self, path, **kw):
        args = dict(scope="train-smoke", candidate_root=self.cand, gpus=[4], uuids={4: UUID4})
        args.update(kw)
        return self.lc.authorization_problems(path, **args)

    def test_valid_authorization_permits(self):
        self.assertEqual(self.problems(self.write(self.auth)), [])

    def test_each_binding_refuses(self):
        p = self.write(self.auth)
        late = dt.datetime.fromisoformat(self.auth["expires_utc"]) + dt.timedelta(seconds=1)
        self.assertTrue(self.problems(p, now=late))                               # expiry, no controller needed
        early = dt.datetime.fromisoformat(self.auth["issued_utc"]) - dt.timedelta(seconds=1)
        self.assertTrue(self.problems(p, now=early))
        self.assertTrue(self.problems(p, scope="serving"))
        self.assertTrue(self.problems(p, gpus=[5]))
        self.assertTrue(self.problems(p, gpus=[4, 5]))
        self.assertTrue(self.problems(p, uuids={4: "GPU-00000000-0000-0000-0000-000000000110"}))
        self.assertTrue(self.problems(p, candidate_root=self.tmp))
        self.assertTrue(self.problems(p, n_cases_done=4))
        self.assertTrue(self.problems(p, expected_sha256="0" * 64))
        self.assertTrue(self.problems(None))

    def test_candidate_change_or_renewable_or_malformed_refuses(self):
        p = self.write(self.auth)
        (self.cand / "studies/a6000/src/x.py").write_text("a = 2\n")
        self.assertTrue(any("digest" in x for x in self.problems(p)))
        a = dict(self.auth, nonrenewable=False, candidate_digest=self.lc.candidate_digest(self.cand))
        self.assertTrue(self.problems(self.write(a, "renewable.json")))
        a = {k: v for k, v in self.auth.items() if k != "expires_utc"}
        self.assertTrue(self.problems(self.write(a, "malformed.json")))
        a = dict(self.auth, expires_utc="2026-10-16T05:00:00", candidate_digest=self.lc.candidate_digest(self.cand))
        self.assertTrue(self.problems(self.write(a, "naive.json")))

    def test_revocation_is_exclusive_and_records_are_never_overwritten(self):
        p = self.write(self.auth)
        self.assertTrue(self.lc.revoke(p, "done"))
        self.assertFalse(self.lc.revoke(p, "again"))
        self.assertTrue(any("revoked" in x for x in self.problems(p)))
        with self.assertRaises(FileExistsError):
            self.lc.create_exclusive(p, {"x": 1})


class FakeSystemctl:
    """Records systemctl calls; simulates one scope whose cgroup empties after `empties_after` kills."""

    def __init__(self, cg_root: Path, cgroup: str, pids, empties_after=1, show_fails=False):
        self.calls, self.kills = [], 0
        self.cg = cg_root / cgroup.lstrip("/")
        self.cg.mkdir(parents=True)
        (self.cg / "cgroup.procs").write_text("\n".join(map(str, pids)))
        self.cgroup, self.empties_after, self.show_fails = cgroup, empties_after, show_fails

    def __call__(self, args, timeout_s=10.0):
        self.calls.append(args)
        if args[0] == "show":
            if self.show_fails:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="bus error")
            empty = not (self.cg / "cgroup.procs").read_text().strip()
            return types.SimpleNamespace(returncode=0, stderr="", stdout=(
                f"ActiveState={'inactive' if empty else 'active'}\nControlGroup={'' if empty else self.cgroup}\n"
                f"LoadState={'not-found' if empty else 'loaded'}\nSubState=x\nMainPID=0\n"))
        if args[0] == "kill":
            self.kills += 1
            if self.kills >= self.empties_after:
                (self.cg / "cgroup.procs").write_text("")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


class ScopeCleanupDecisions(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(SRC))
        self.addCleanup(sys.path.remove, str(SRC))
        import mgbench.scopes as scopes
        self.scopes = scopes
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def handle(self, cg):
        return self.scopes.ScopeHandle(unit="mgbench-smoke-t-01", leader_pid=1, cgroup=cg)

    def test_scope_command_enforces_limits_and_control_group_kill(self):
        cmd = self.scopes.scope_command("u", ["sleep", "1"], cpu_quota_pct=1500, memory_max="46G", runtime_max_s=425,
                                        stop_timeout_s=60)
        for p in ("CPUQuota=1500%", "MemoryMax=46G", "MemorySwapMax=0", "RuntimeMaxSec=425", "TimeoutStopSec=60",
                  "KillMode=control-group"):
            self.assertIn(p, cmd)
        self.assertEqual(cmd[:3], ["systemd-run", "--user", "--scope"])

    def test_cleanup_signals_only_the_unit_and_confirms_empty(self):
        cg = "/user.slice/u/app.slice/mgbench-smoke-t-01.scope"
        fake = FakeSystemctl(self.tmp, cg, [111, 222])
        with mock.patch.object(self.scopes, "CGROUP_ROOT", self.tmp), \
                mock.patch.object(self.scopes, "run_systemctl", fake):
            r = self.scopes.cleanup_scope(self.handle(cg), grace_s=2)
        self.assertTrue(r["confirmed"])
        kills = [c for c in fake.calls if c[0] == "kill"]
        self.assertTrue(kills and all(c[-1] == "mgbench-smoke-t-01.scope" and "--kill-whom=all" in c for c in kills))
        shows = [c for c in fake.calls if c[0] == "show"]
        self.assertTrue(shows and all(c[1] == "mgbench-smoke-t-01.scope" for c in shows))   # never the bare name

    def test_metadata_error_or_missing_identity_is_unconfirmed(self):
        cg = "/user.slice/u/app.slice/mgbench-smoke-t-01.scope"
        fake = FakeSystemctl(self.tmp, cg, [111], show_fails=True)
        with mock.patch.object(self.scopes, "CGROUP_ROOT", self.tmp), \
                mock.patch.object(self.scopes, "run_systemctl", fake):
            self.assertFalse(self.scopes.cleanup_scope(self.handle(cg), grace_s=1)["confirmed"])
            r = self.scopes.cleanup_scope(self.scopes.ScopeHandle(unit="x", leader_pid=1, identity_error="lost"), 1)
        self.assertFalse(r["confirmed"])
        self.assertFalse([c for c in fake.calls if c[0] == "kill"])                # nothing signalled

    def test_survivors_after_sigkill_leave_cleanup_unconfirmed(self):
        cg = "/user.slice/u/app.slice/mgbench-smoke-t-01.scope"
        fake = FakeSystemctl(self.tmp, cg, [111], empties_after=99)
        with mock.patch.object(self.scopes, "CGROUP_ROOT", self.tmp), \
                mock.patch.object(self.scopes, "run_systemctl", fake), mock.patch.object(self.scopes.time, "sleep"):
            r = self.scopes.cleanup_scope(self.handle(cg), grace_s=0.01)
        self.assertFalse(r["confirmed"])
        self.assertTrue(any("--signal=SIGKILL" in c for c in fake.calls if c[0] == "kill"))


def good_result(framework="ddp", checked=True):
    measured = {"logical_attempts": 3, "microbatches": 6, "microbatch_losses_all_finite": True, "microbatch_losses": [12.0] * 6,
                "counter_updates": 3, "framework_reported_skips": 0, "base_call_multiplicity_matches": True,
                "boundary_coverage": ["complete"] * 3 if checked else ["not_checked"] * 3,
                "inspected_boundaries": 3 if checked else 0, "update_evidence_verdicts": ["verified_sampled"] * 3,
                "sample_plan_complete_each_boundary": True if checked else None,
                "discriminating_samples_per_call": [150, 150, 150] if checked else [],
                "steps": [{"module_bf16_vs_master": {"pass": True}}] * 3}
    warm = {"logical_attempts": 1, "microbatches": 2, "microbatch_losses_all_finite": True, "microbatch_losses": [12.0] * 2}
    layout = ({"unit_map": {"unique_trainable": 290}, "problems": []} if framework != "zero2" else
              {"model_to_partition_map": [{"params": 290, "padding": 0}], "accounting": {"group0": 10}})
    audit = {"boundary_ledger": {"warmup": warm, "measured": measured, "layout": layout}}
    if framework == "zero2":
        audit["deepspeed_layout"] = {"optimizer_class": "deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer",
                                     "accounting": {"group0": 10}, "model_to_partition_map": [{"params": 290}]}
    return {
        "status": "ok", "mem_fraction": 0.5,
        "assessment": {"execution": "ok", "measurement_eligibility": "ineligible: smoke test, not a performance measurement",
                       "gradient": ({"status": "complete_at_every_measured_boundary", "gradient_validity": True}
                                    if checked else {"status": "not_checked", "gradient_validity": None}),
                       "update_state": {"status": "verified_sampled_every_base_call" if checked else "not_collected"}},
        "optimizer_update_audit_per_rank": [audit],
        "runtime_devices": [{"stage": s, "device_count": 1, "uuid": UUID4, "current_device": 0,
                             "pci_bus_id": "0000:81:00.0"} for s in ("a", "b", "c")],
        "identity_chain": {"checked": True, "problems": [], "driver": {"count": 1},
                           "nvml": {"index": 4, "uuid": UUID4, "pci_bus_id": "00000000:81:00.0"},
                           "deepspeed": ({"logical": 0, "nvml_index": 4, "expected_index": 4, "uuid": UUID4}
                                         if framework == "zero2" else "not applicable")},
        "process": {"mgbench_file": str(SRC / "mgbench/__init__.py"), "sys_executable": "/py",
                    "cgroup": {"cpu.max": "1500000 100000", "memory.max": "49392123904"}}}


FROZEN = {"mem_fraction": 0.5, "warmup_attempts": 1, "measured_attempts": 3, "grad_accum": 2}


class FrozenExpectations(unittest.TestCase):
    def setUp(self):
        self.smoke = smoke_module()

    def ev(self, result, framework="ddp", checked=True, rc=0):
        return self.smoke.evaluate_case(result, rc, {"framework": framework, "numerical_checks": checked}, FROZEN,
                                        candidate_src=SRC, interpreter="/py", uuid=UUID4,
                                        worker_cpu_max="1500000 100000", worker_memory_max="49392123904")

    def test_expected_results_pass(self):
        for fw, chk in (("ddp", True), ("fsdp", True), ("zero2", True), ("ddp", False)):
            self.assertTrue(self.ev(good_result(fw, chk), fw, chk)["pass"], (fw, chk))

    def test_runtime_sample_plan_must_equal_the_frozen_plan(self):
        import hashlib
        plan = [{"name": "model.norm.weight", "local_elements": 896, "indices": [0, 1, 2]}]
        frozen = dict(FROZEN, native_sample_plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest())
        r = good_result()
        r["optimizer_update_audit_per_rank"][0]["boundary_ledger"]["layout"]["sample_plan"] = plan
        ev = lambda res: self.smoke.evaluate_case(res, 0, {"framework": "ddp", "numerical_checks": True}, frozen,
                                                  candidate_src=SRC, interpreter="/py", uuid=UUID4,
                                                  worker_cpu_max="1500000 100000", worker_memory_max="49392123904")
        self.assertTrue(ev(r)["pass"])
        r["optimizer_update_audit_per_rank"][0]["boundary_ledger"]["layout"]["sample_plan"] = [dict(plan[0], indices=[0, 1, 3])]
        self.assertFalse(ev(r)["pass"])

    def test_deviations_fail(self):
        def mutated(fn, fw="ddp", chk=True):
            r = good_result(fw, chk)
            fn(r)
            return self.ev(r, fw, chk)["pass"]
        led = lambda r: r["optimizer_update_audit_per_rank"][0]["boundary_ledger"]
        self.assertFalse(mutated(lambda r: led(r)["measured"].update(boundary_coverage=["complete", "partial", "complete"])))
        self.assertFalse(mutated(lambda r: r["runtime_devices"][1].update(device_count=2)))
        self.assertFalse(mutated(lambda r: r["runtime_devices"][2].update(uuid="GPU-other")))
        self.assertFalse(mutated(lambda r: r["process"].update(mgbench_file="/old/editable/mgbench/__init__.py")))
        self.assertFalse(mutated(lambda r: led(r)["measured"].update(microbatches=3)))
        self.assertFalse(mutated(lambda r: led(r)["warmup"].update(microbatch_losses_all_finite=False)))
        self.assertFalse(mutated(lambda r: r["assessment"]["update_state"].update(status="failed")))
        self.assertFalse(mutated(lambda r: r["process"]["cgroup"].update({"memory.max": "max"})))
        # an unchecked case must not report gradient validity
        self.assertFalse(mutated(lambda r: r["assessment"]["gradient"].update(status="complete_at_every_measured_boundary",
                                                                              gradient_validity=True), chk=False))
        self.assertFalse(mutated(lambda r: led(r)["measured"]["steps"][1].update(module_bf16_vs_master={"pass": False}),
                                 fw="zero2"))
        self.assertFalse(mutated(lambda r: r["identity_chain"].update(problems=["driver ordinal 0: UUID differs"])))
        self.assertFalse(mutated(lambda r: r["identity_chain"]["deepspeed"].update(nvml_index=5), fw="zero2"))
        self.assertFalse(mutated(lambda r: led(r)["measured"].update(sample_plan_complete_each_boundary=False)))
        self.assertFalse(mutated(lambda r: led(r)["measured"].update(discriminating_samples_per_call=[150, 0, 150])))
        self.assertFalse(mutated(lambda r: r["runtime_devices"][0].update(pci_bus_id=None)))
        self.assertFalse(self.ev(None)["pass"])
        self.assertFalse(self.ev(good_result(), rc=1)["pass"])


@unittest.skipUnless(HAVE_YAML, "yaml not installed")
class ServingWrapperIsControlled(Stubbed):
    def test_refuses_without_serving_authorization_and_writes_nothing(self):
        self.load()
        spec = importlib.util.spec_from_file_location("tested_serve_latency", A6000 / "scripts/serve_latency.py")
        sv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sv)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "session"
            with mock.patch("sys.stderr"):
                self.assertEqual(sv.main(["--gpu", "4", "--out", str(out), "--client", "a:1:1:1"]), 6)
            self.assertFalse(out.exists())
        self.assertTrue(any("authorization" in p for p in sv.preflight(None, 4, {4: UUID4})))

    def test_source_has_no_marker_or_pgid_cleanup(self):
        text = (A6000 / "scripts/serve_latency.py").read_text()
        self.assertNotIn("terminate_run_processes", text)
        self.assertNotIn("killpg", text)
        self.assertIn("scopes.cleanup_scope", text)


class NoAllDeviceCudaPaths(unittest.TestCase):
    def test_worker_and_supervisor_sources(self):
        main = (SRC / "mgbench/train/__main__.py").read_text()
        self.assertNotIn("range(torch.cuda.device_count())", main)
        self.assertIn('MGBENCH_REQUIRE_SINGLE_DEVICE") == "1" and torch.cuda.device_count() != 1', main)
        self.assertIn('_runtime_device_record("after set_per_process_memory_fraction")', main)
        sup = (A6000 / "scripts/smoke_supervisor.py").read_text()
        self.assertIn("scopes.cleanup_scope", sup)
        self.assertNotIn("terminate_run_processes", sup)
        self.assertNotIn("killpg", sup)


if __name__ == "__main__":
    unittest.main()
