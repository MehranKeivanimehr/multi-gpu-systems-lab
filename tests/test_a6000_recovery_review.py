"""Regression tests for the A6000 recovery review (session 20261015r).

CPU only. NVML is replaced by an in-memory stub, so nothing here queries or touches a GPU. These tests show the
launcher's decision logic and the update ledger's accounting; they are not evidence of CUDA, NCCL or DeepSpeed
behaviour on the node.
"""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
A6000 = ROOT / "studies/a6000"
A6000_SRC = A6000 / "src"

try:
    import torch  # noqa: F401
    import yaml  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # the offline CI has no torch/yaml; these run with the study environment
    HAVE_TORCH = False


class NVMLError(Exception):
    pass


def nvml_stub(n_gpus=8, power=20.0, compute=None, graphics=None, uuids=None, fail=()):
    """Minimal pynvml stand-in. `compute`/`graphics`: {index: [pids]}; `fail`: names of calls that raise."""
    compute, graphics, uuids = compute or {}, graphics or {}, uuids or {}
    stub = types.ModuleType("pynvml")
    stub.NVMLError = NVMLError
    stub.nvmlMemory_v2 = 2
    stub.NVML_TEMPERATURE_GPU = 0
    stub.NVML_CLOCK_SM = 1

    def call(name, fn):
        def wrapped(*args, **kwargs):
            if name in fail:
                raise NVMLError(f"stub failure in {name}")
            return fn(*args, **kwargs)
        return wrapped

    procs = lambda table: (lambda h: [types.SimpleNamespace(pid=p, usedGpuMemory=2**30) for p in table.get(h, [])])
    stub.nvmlInit = lambda: None
    stub.nvmlDeviceGetCount = call("count", lambda: n_gpus)
    stub.nvmlDeviceGetHandleByIndex = call("handle", lambda i: i)
    stub.nvmlDeviceGetMemoryInfo = lambda h, version=None: types.SimpleNamespace(used=2**20, total=48 * 2**30)
    stub.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=0, memory=0)
    stub.nvmlDeviceGetComputeRunningProcesses = call("compute", procs(compute))
    stub.nvmlDeviceGetGraphicsRunningProcesses = call("graphics", procs(graphics))
    stub.nvmlDeviceGetMPSComputeRunningProcesses = call("mps", procs({}))   # stage 20261016s: MPS is queried too
    stub.nvmlDeviceGetName = lambda h: b"NVIDIA RTX A6000"
    stub.nvmlDeviceGetPciInfo = lambda h: types.SimpleNamespace(busId=b"00000000:81:00.0")
    stub.nvmlDeviceGetPowerUsage = call("power", lambda h: int(power * 1000))
    stub.nvmlDeviceGetTemperature = lambda h, sensor: 30
    stub.nvmlDeviceGetPerformanceState = lambda h: 8
    stub.nvmlDeviceGetClockInfo = lambda h, clock: 210
    stub.nvmlDeviceGetUUID = call("uuid", lambda h: uuids.get(h, f"GPU-stub-{h}"))
    return stub


class StubbedNVML(unittest.TestCase):
    """Load a fresh mgbench (gpu_monitor, safety) against an NVML stub."""

    def load(self, **stub_kwargs):
        patcher = mock.patch.dict(sys.modules, {"pynvml": nvml_stub(**stub_kwargs)})
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in [m for m in sys.modules if m == "mgbench" or m.startswith("mgbench.")]:
            del sys.modules[name]
        sys.path.insert(0, str(A6000_SRC))
        self.addCleanup(sys.path.remove, str(A6000_SRC))
        import mgbench.gpu_monitor as gm
        import mgbench.safety as safety
        gm._initialized = False
        return gm, safety


IDLE = {"poll_s": 0, "consecutive_polls": 1, "max_wait_s": 0, "max_mem_mib": 100, "max_util_pct": 0,
        "max_power_w": 60, "max_temp_c": 50, "min_free_ram_gib": 0,
        "node_guard": {"planned_w_per_gpu": 300, "start_budget_w": 1500, "abort_total_w": 1700, "abort_after_s": 5,
                       "max_load1": 10**6, "max_gpus_per_run": 2, "node_gpu_count": 8}}


class ProcessQueryFailsClosed(StubbedNVML):
    """gpu_monitor used to turn an NVML process-query exception into an empty list, i.e. an idle GPU."""

    def test_compute_query_error_is_unknown_not_empty(self):
        gm, _ = self.load(fail=("compute",))
        r = gm.read_gpu(4)
        self.assertIsNone(r["processes"])
        self.assertIn("nvmlDeviceGetComputeRunningProcesses", r["process_query_error"])

    def test_graphics_query_error_or_missing_binding_is_unknown(self):
        gm, _ = self.load(fail=("graphics",))
        self.assertIsNone(gm.read_gpu(4)["processes"])
        gm, _ = self.load()
        del sys.modules["pynvml"].nvmlDeviceGetGraphicsRunningProcesses        # older binding without the call
        r = gm.read_gpu(4)
        self.assertIsNone(r["processes"])
        self.assertIn("AttributeError", r["process_query_error"])

    def test_successful_queries_list_compute_and_graphics_processes(self):
        gm, _ = self.load(compute={4: [111]}, graphics={4: [222]})
        r = gm.read_gpu(4)
        self.assertEqual(sorted((p["pid"], p["type"]) for p in r["processes"]), [(111, "compute"), (222, "graphics")])
        self.assertIsNone(r["process_query_error"])
        self.assertEqual(gm.read_gpu(5)["processes"], [])

    def test_idle_gate_refuses_unknown_process_list(self):
        _, safety = self.load(fail=("compute",))
        problems = safety.idle_problems([4], IDLE)
        self.assertTrue(problems)
        self.assertTrue(any("unknown" in p and "do not launch" in p for p in problems))

    def test_idle_gate_refuses_unreadable_gpu(self):
        _, safety = self.load(fail=("handle",))
        self.assertTrue(any("unreadable" in p for p in safety.idle_problems([4, 5], IDLE)))

    def test_idle_gate_flags_graphics_process(self):
        _, safety = self.load(graphics={5: [333]})
        self.assertTrue(any(p.startswith("gpu5: processes") for p in safety.idle_problems([4, 5], IDLE)))

    def test_idle_gate_still_passes_a_genuinely_idle_gpu(self):
        _, safety = self.load()
        self.assertEqual(safety.idle_problems([4, 5], IDLE), [])

    def test_wait_for_idle_never_reports_idle_when_process_query_fails(self):
        _, safety = self.load(fail=("compute",))
        with mock.patch.object(safety.os, "getloadavg", return_value=(1.0, 1.0, 1.0)):
            ok, _, problems = safety.wait_for_idle([4], IDLE, log=lambda *_: None)
        self.assertFalse(ok)
        self.assertTrue(any("unknown" in p for p in problems))


class NodeGuardFailsClosed(StubbedNVML):
    def test_missing_guard_blocks(self):
        _, safety = self.load()
        self.assertTrue(safety.node_guard_problems([4], {}))

    def test_unreadable_power_blocks(self):
        _, safety = self.load(fail=("power",))
        self.assertTrue(any("unreadable" in p for p in safety.node_guard_problems([4], IDLE)))

    def test_missing_gpu_blocks(self):
        _, safety = self.load(n_gpus=7)
        with mock.patch.object(safety.os, "getloadavg", return_value=(1.0, 1.0, 1.0)):
            self.assertTrue(any("expected 8" in p for p in safety.node_guard_problems([4], IDLE)))

    def test_full_node_with_idle_gpus_passes(self):
        _, safety = self.load()
        with mock.patch.object(safety.os, "getloadavg", return_value=(1.0, 1.0, 1.0)):
            self.assertEqual(safety.node_guard_problems([4, 5], IDLE), [])


PINS = {4: "GPU-00000000-0000-0000-0000-000000000109", 5: "GPU-00000000-0000-0000-0000-000000000112"}


class DeviceIdentity(StubbedNVML):
    def test_matching_uuids_pass(self):
        _, safety = self.load(uuids=PINS)
        self.assertEqual(safety.device_identity_problems([4, 5], {"gpu_uuids": PINS}), [])

    def test_swapped_or_unknown_uuid_blocks(self):
        _, safety = self.load(uuids={4: PINS[5], 5: PINS[4]})
        self.assertEqual(len(safety.device_identity_problems([4, 5], {"gpu_uuids": PINS})), 2)

    def test_unreadable_uuid_or_missing_pin_blocks(self):
        _, safety = self.load(uuids=PINS, fail=("uuid",))
        self.assertTrue(any("unreadable" in p for p in safety.device_identity_problems([4], {"gpu_uuids": PINS})))
        _, safety = self.load(uuids=PINS)
        self.assertTrue(safety.device_identity_problems([6], {"gpu_uuids": PINS}))
        self.assertTrue(safety.device_identity_problems([4], {}))

    def test_cluster_config_pins_only_gpus_4_to_7(self):
        text = (A6000 / "configs/cluster.yaml").read_text()
        for uuid in ("GPU-00000000-0000-0000-0000-000000000109", "GPU-00000000-0000-0000-0000-000000000112",
                     "GPU-00000000-0000-0000-0000-000000000116", "GPU-00000000-0000-0000-0000-000000000108"):
            self.assertIn(uuid, text)
        self.assertIn("allowed_gpus: [4, 5, 6, 7]", text)
        self.assertIn("node_gpu_count: 8", text)


class LaunchHold(StubbedNVML):
    def test_hold_semantics(self):
        _, safety = self.load()
        self.assertTrue(safety.launch_hold_problems({}))                                  # missing -> held
        self.assertTrue(safety.launch_hold_problems({"launch_hold": {"reason": "x"}}))    # no flag -> held
        self.assertTrue(safety.launch_hold_problems({"launch_hold": {"active": True}}))
        self.assertTrue(safety.launch_hold_problems({"launch_hold": {"active": "false"}}))  # only a real False lifts
        self.assertEqual(safety.launch_hold_problems({"launch_hold": {"active": False}}), [])

    def test_delivered_cluster_config_is_on_hold(self):
        text = (A6000 / "configs/cluster.yaml").read_text()
        self.assertIn("launch_hold:\n  active: true", text)


@unittest.skipUnless(HAVE_TORCH, "torch/yaml not installed")
class RunnerHoldAndEvidence(StubbedNVML):
    """Drive scripts/run_matrix.py main() with every launch path patched to fail the test if reached."""

    def runner(self, tmp: Path, hold_active: bool):
        self.load()
        spec = importlib.util.spec_from_file_location("tested_run_matrix", A6000 / "scripts/run_matrix.py")
        rm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rm)
        cluster = rm.cluster_config()
        cluster["launch_hold"] = {"active": hold_active, "since": "test", "reason": "unit test"}
        rm.cluster_config = lambda: cluster
        root = tmp / "studies/a6000"
        rm.ROOT, rm.CACHE = root, root / ".cache"
        rm.STOP_FILE = rm.CACHE / "STOP"

        def forbidden(*a, **k):
            raise AssertionError("a launch path was reached")
        rm.launch = rm.wait_for_idle = rm.nvidia_smi_snapshot = forbidden
        return rm, cluster

    def experiment(self, tmp: Path, name: str) -> Path:
        path = tmp / "exp.yaml"
        path.write_text(f"name: {name}\nkind: train\nrepeats: 1\ngpu_counts: [4]\n"
                        "defaults: {warmup_steps: 0, measure_steps: 2, seed: 1}\n"
                        "workloads:\n  - {model: resnet50, micro_batch: 8, frameworks: [fsdp]}\n")
        return path

    def test_active_hold_exits_6_before_writing_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            rm, _ = self.runner(tmp, hold_active=True)
            exp = self.experiment(tmp, "validation_v2_holdtest")
            with mock.patch.object(sys, "argv", ["run_matrix.py", str(exp)]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    rm.main()
            self.assertEqual(cm.exception.code, 6)
            self.assertFalse((tmp / "studies").exists())          # no results dir, log or lock was created

    def test_interrupted_attempt_is_not_written_into(self):
        """Like resnet50__fsdp__g4__r0 from 2026-10-14 22:14: four 0-byte files and no status.json."""
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            rm, cluster = self.runner(tmp, hold_active=False)
            name = "validation_v2_preservetest"
            exp = self.experiment(tmp, name)
            run_id = rm.expand_experiment(rm.load_yaml(exp), cluster)[0]["run_id"]
            crashed = tmp / "studies/a6000/results/raw" / name / run_id
            crashed.mkdir(parents=True)
            for f in ("spec.json", "command.txt", "nvidia_smi_before.txt", "run.log"):
                (crashed / f).write_bytes(b"")
            before = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in crashed.iterdir())
            with mock.patch.object(sys, "argv", ["run_matrix.py", str(exp)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                rm.main()
            after = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in crashed.iterdir())
            self.assertEqual(before, after)
            self.assertIn("preserved unchanged", (crashed.parent / "runner.log").read_text())

    def test_status_writable_rules(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            rm, _ = self.runner(tmp, hold_active=True)
            d = tmp / "run"
            self.assertTrue(rm.status_writable(d))
            d.mkdir()
            self.assertTrue(rm.status_writable(d))
            (d / "status.json").write_text(json.dumps({"status": "deferred_safety"}))
            self.assertTrue(rm.status_writable(d))
            (d / "status.json").write_text(json.dumps({"status": "failed"}))
            self.assertFalse(rm.status_writable(d))
            (d / "status.json").write_text(json.dumps({"status": "deferred_safety"}))
            (d / "run.log").write_bytes(b"")
            self.assertFalse(rm.status_writable(d))


def ledger_module():
    spec = importlib.util.spec_from_file_location("tested_update_ledger", A6000_SRC / "mgbench/train/update_ledger.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_for(summary, losses_finite=True):
    return {"verification": summary, "losses_all_finite": losses_finite, "nonfinite_gradient_steps": 0}


class UpdateLedgerAccounting(unittest.TestCase):
    """Attempted steps, observed gradients, logical updates and base optimizer calls stay distinct."""

    def setUp(self):
        self.L = ledger_module()

    def native(self, steps=5, skip_at=None, nan_at=None, checks=True):
        led = self.L.UpdateLedger("ddp", checks, "in_step")
        for i in range(steps):
            led.begin_step()
            led.observe(i != nan_at, 3, "in-step", "full_local")          # inspected whether or not opt.step() runs
            called = i != skip_at
            if called:
                led.base_call()
            led.end_step(called, "native optimizer post-hook")
        return led.summary(steps)

    def deepspeed(self, steps=5, overflow_at=None, base_calls_per_step=1, expected=1):
        led = self.L.UpdateLedger("zero2", True, "base_optimizer_pre_hook")
        for i in range(steps):
            led.begin_step()
            skipped = i == overflow_at
            if not skipped:                         # the pre-hook only runs when DeepSpeed calls the base optimizer
                for _ in range(base_calls_per_step):
                    led.observe(True, 1, "base optimizer pre-hook", "full_local" if expected else "unknown")
                    led.base_call()
            led.end_step(not skipped, "engine counters", framework_reported_skip=skipped,
                         expected_observations=expected)
        return led.summary(steps)

    def test_clean_native_run_is_verified(self):
        s = self.native()
        self.assertEqual((s["attempted_steps"], s["gradient_observed_steps"], s["logical_updates"],
                          s["base_optimizer_calls"]), (5, 5, 5, 5))
        self.assertEqual(s["gradient_finiteness"], "verified_finite")
        self.assertTrue(s["logical_updates_complete"])
        self.assertEqual(self.L.run_status([audit_for(s)], True)[0], "ok")

    def test_native_skipped_optimizer_call_still_has_observed_gradients(self):
        s = self.native(skip_at=2)
        self.assertEqual((s["attempted_steps"], s["gradient_observed_steps"], s["logical_updates"],
                          s["base_optimizer_calls"]), (5, 5, 4, 4))
        self.assertEqual(s["gradient_finiteness"], "verified_finite")
        self.assertFalse(s["logical_updates_complete"])
        self.assertEqual(self.L.run_status([audit_for(s)], True)[0], "invalid_numerics_or_updates")

    def test_deepspeed_skipped_boundary_stays_unverified(self):
        s = self.deepspeed(overflow_at=2)
        self.assertEqual((s["attempted_steps"], s["gradient_observed_steps"], s["logical_updates"],
                          s["base_optimizer_calls"], s["framework_reported_skips"]), (5, 4, 4, 4, 1))
        self.assertEqual(s["gradient_unverified_steps"], [2])
        self.assertEqual(s["gradient_finiteness"], "unverified")           # never "finite" for the unseen step
        status, reasons = self.L.run_status([audit_for(s)], True)
        self.assertEqual(status, "invalid_numerics_or_updates")
        self.assertTrue(any("[2]" in r and "unobserved" in r for r in reasons))

    def test_deepspeed_clean_run_with_established_coverage_is_verified(self):
        s = self.deepspeed()
        self.assertEqual(s["gradient_finiteness"], "verified_finite")
        self.assertEqual(self.L.run_status([audit_for(s)], True)[0], "ok")

    def test_zero3_subgroup_calls_are_not_updates_and_coverage_is_unverified(self):
        s = self.deepspeed(base_calls_per_step=3, expected=None)
        self.assertEqual((s["base_optimizer_calls"], s["logical_updates"]), (15, 5))
        self.assertEqual(s["gradient_finiteness"], "unverified")
        self.assertEqual(self.L.run_status([audit_for(s)], True)[0], "unverified_gradients")

    def test_nonfinite_gradient_is_detected(self):
        s = self.native(nan_at=3)
        self.assertEqual(s["gradient_finiteness"], "nonfinite_detected")
        self.assertEqual(s["gradient_nonfinite_steps"], [3])
        self.assertEqual(self.L.run_status([audit_for(s)], True)[0], "invalid_numerics_or_updates")

    def test_throughput_run_lists_gradients_as_unverified(self):
        s = self.native(checks=False)
        self.assertEqual(s["gradient_finiteness"], "unverified")
        self.assertEqual(s["gradient_observed_steps"], 0)
        status, reasons = self.L.run_status([audit_for(s)], False)
        self.assertEqual(status, "ok")
        self.assertTrue(any("not requested" in r for r in reasons))

    def test_observation_without_gradients_is_not_a_pass(self):
        led = self.L.UpdateLedger("ddp", True, "in_step")
        led.begin_step()
        led.observe(None, 0, "in-step", "full_local")
        led.base_call()
        led.end_step(True, "post-hook")
        s = led.summary(1)
        self.assertEqual(s["steps"][0]["gradient"], "no_gradients")
        self.assertEqual(s["gradient_finiteness"], "unverified")

    def test_unknown_update_and_rank_disagreement(self):
        led = self.L.UpdateLedger("zero2", True, "base_optimizer_pre_hook")
        led.begin_step()
        led.observe(True, 1, "hook", "full_local")
        led.end_step(None, "two boundaries in one attempted step")
        s = led.summary(1)
        self.assertEqual(s["logical_updates_unknown"], 1)
        self.assertFalse(s["logical_updates_complete"])
        good = self.native(steps=1)
        self.assertEqual(self.L.run_status([audit_for(good), audit_for(s)], True)[0], "invalid_numerics_or_updates")

    def test_end_without_begin_is_an_error(self):
        with self.assertRaises(RuntimeError):
            self.L.UpdateLedger("ddp", True, "in_step").end_step(True, "x")


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class LedgerWithRealTorchOptimizer(unittest.TestCase):
    """The ledger fed by observe_optimizer's real hooks on a CPU AdamW, as the trainers wire it."""

    def setUp(self):
        sys.path.insert(0, str(A6000_SRC))
        self.addCleanup(sys.path.remove, str(A6000_SRC))
        import torch
        from mgbench.train.frameworks import observe_optimizer
        from mgbench.train.update_ledger import UpdateLedger, scan_gradients
        self.torch, self.observe, self.Ledger, self.scan = torch, observe_optimizer, UpdateLedger, scan_gradients

    def run_steps(self, observe_in_hook, skip_at=None, nan_at=None, steps=4):
        torch = self.torch
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ledger = self.Ledger("test", True, "base_optimizer_pre_hook" if observe_in_hook else "in_step")
        audit, _ = self.observe(opt, model, {"numerical_checks": True}, ledger=ledger, observe_in_hook=observe_in_hook,
                                hook_coverage="full_local")
        for i in range(steps):
            ledger.begin_step()
            loss = model(torch.randn(3, 4)).pow(2).mean() * (float("nan") if i == nan_at else 1.0)
            loss.backward()
            if not observe_in_hook:
                finite, count = self.scan(p.grad for p in model.parameters())
                ledger.observe(finite, count, "in-step", "full_local")
            before = audit["optimizer_calls"]
            if i != skip_at:
                opt.step()
            opt.zero_grad(set_to_none=True)
            ledger.end_step(audit["optimizer_calls"] > before, "post-hook")
        return ledger.summary(steps), audit

    def test_native_style_observation_covers_a_skipped_call(self):
        s, audit = self.run_steps(observe_in_hook=False, skip_at=1)
        self.assertEqual((s["attempted_steps"], s["gradient_observed_steps"], s["logical_updates"],
                          s["base_optimizer_calls"]), (4, 4, 3, 3))
        self.assertEqual(audit["gradient_checks"], 3)          # the base-call counter alone would have seen only 3
        self.assertEqual(s["gradient_finiteness"], "verified_finite")

    def test_hook_only_observation_leaves_a_skipped_call_unverified(self):
        s, _ = self.run_steps(observe_in_hook=True, skip_at=1)
        self.assertEqual(s["gradient_unverified_steps"], [1])
        self.assertEqual(s["gradient_finiteness"], "unverified")

    def test_nan_gradient_found_by_in_step_scan_even_when_the_call_is_skipped(self):
        s, audit = self.run_steps(observe_in_hook=False, skip_at=2, nan_at=2)
        self.assertEqual(s["gradient_finiteness"], "nonfinite_detected")
        self.assertEqual(audit["nonfinite_gradient_steps"], 0)  # the hook never saw it


# TrainerWiring (recovery review 20261015r) checked the UpdateLedger wiring, which stage 20261016s replaced with the
# BoundaryLedger; its successor is TrainerWiringV2 in tests/test_a6000_boundary_ledger.py.


if __name__ == "__main__":
    unittest.main()
