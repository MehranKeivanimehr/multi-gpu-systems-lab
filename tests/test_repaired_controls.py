"""CPU regression tests for the repaired local controls (session 20261015d).

Scope, stated explicitly: these are mock/CPU tests of control logic. They do not
execute CUDA, do not validate DeepSpeed on any host, and do not measure overhead
or numerical equivalence. Every review counterexample below is expressed as a
NEGATIVE acceptance test: the named condition must REFUSE or must NOT certify.
"""
import ast
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "studies/titan-1080/bench"


def module_at(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# RETAINED: repair 1 - apply_memory_cap device restoration
# ===========================================================================
class FakeCuda:
    def __init__(self, n_devices=4, current=0, raise_on=None):
        self._n, self._current, self._raise_on = n_devices, current, raise_on
        self.fraction_calls, self.set_device_calls = [], []

    def is_available(self): return True
    def device_count(self): return self._n
    def current_device(self): return self._current

    def set_device(self, d):
        self._current = d; self.set_device_calls.append(d)

    def set_per_process_memory_fraction(self, fraction, device):
        self._current = device                      # the real side effect
        self.fraction_calls.append((fraction, device))
        if self._raise_on is not None and device == self._raise_on:
            raise RuntimeError(f"simulated allocator failure on device {device}")


def fake_torch(cuda):
    module = types.ModuleType("torch"); module.cuda = cuda; return module


class MemoryCapRestorationTests(unittest.TestCase):
    def setUp(self):
        self.common = module_at("rc_common", "studies/titan-1080/bench/common.py")

    def test_current_device_restored_after_normal_completion(self):
        for start in range(4):
            cuda = FakeCuda(n_devices=4, current=start)
            self.common.apply_memory_cap(0.85, torch_module=fake_torch(cuda))
            self.assertEqual(cuda.current_device(), start)

    def test_current_device_restored_when_a_cap_call_raises(self):
        cuda = FakeCuda(n_devices=4, current=2, raise_on=1)
        with self.assertRaises(RuntimeError):
            self.common.apply_memory_cap(0.85, torch_module=fake_torch(cuda))
        self.assertEqual(cuda.current_device(), 2)

    def test_intended_cap_value_and_device_coverage_preserved(self):
        cuda = FakeCuda(n_devices=6, current=3)
        self.common.apply_memory_cap(0.85, torch_module=fake_torch(cuda))
        self.assertEqual([d for _, d in cuda.fraction_calls], list(range(6)))
        self.assertEqual({f for f, _ in cuda.fraction_calls}, {0.85})

    def test_no_op_when_cuda_unavailable(self):
        cuda = FakeCuda(current=1); cuda.is_available = lambda: False
        self.common.apply_memory_cap(0.85, torch_module=fake_torch(cuda))
        self.assertEqual(cuda.fraction_calls, [])
        self.assertEqual(cuda.set_device_calls, [])


# ===========================================================================
# R01 - trainer validity: the FINAL status must not certify missing evidence
# ===========================================================================
class FakeGrad:
    def __init__(self, finite=True): self.finite = finite


class FakeParam:
    def __init__(self, grad=None): self.grad = grad


class FakeOptimizer:
    def __init__(self, params): self.param_groups = [{"params": params}]; self._post = []
    def register_step_post_hook(self, hook): self._post.append(hook)
    def step(self):
        for hook in self._post: hook(self, (), {})


def finite_fn(gradients): return all(g.finite for g in gradients)


class TrainerCertificationTests(unittest.TestCase):
    """Exercises run_report.certify, which is the status the trainer emits."""

    def setUp(self):
        self.obs = module_at("rc_observer", "studies/titan-1080/bench/update_observer.py")
        self.rep = module_at("rc_report", "studies/titan-1080/bench/run_report.py")

    def _native(self, **over):
        base = {"attempted_steps": 5, "native_successful_logical_updates": 5,
                "base_optimizer_calls": 5, "gradient_coverage_complete": True,
                "missing_gradient_steps": 0, "unobserved_attempt_steps": 0,
                "nonfinite_gradient_steps": 0}
        base.update(over); return base

    def _certify(self, mode="single", checks=True, expected=5,
                 native=None, engine=None, losses=None):
        record = self.rep.build_update_record(
            mode=mode, numerical_checks=checks, expected_logical_updates=expected,
            native_summary=native, engine_counters=engine,
            losses=[1.0] * 5 if losses is None else losses)
        return record, self.rep.certify(record)

    # --- the three reported counterexamples ---------------------------------
    def test_counterexample_missing_gradients_must_not_certify(self):
        _, v = self._certify(native=self._native(missing_gradient_steps=5,
                                                 gradient_coverage_complete=False))
        self.assertNotEqual(v["status"], "ok")
        self.assertFalse(v["certified"])
        self.assertFalse(v["numerical_diagnostic_certified"])
        self.assertTrue(any("missing_gradient_steps" in r for r in v["reasons"]))

    def test_counterexample_no_gradient_observations_must_not_certify(self):
        _, v = self._certify(native=self._native(unobserved_attempt_steps=5,
                                                 gradient_coverage_complete=False))
        self.assertNotEqual(v["status"], "ok")
        self.assertTrue(any("unobserved_attempt_steps" in r for r in v["reasons"]))

    def test_counterexample_deepspeed_zero_attempts_must_not_claim_coverage(self):
        na = self.obs.not_applicable_native_summary("engine.step() is opaque")
        record, v = self._certify(
            mode="deepspeed",
            native=na,
            engine={"engine_update_boundaries": 5, "engine_skipped_updates": 0,
                    "engine_base_optimizer_calls": 5})
        self.assertNotEqual(v["status"], "ok")
        self.assertEqual(v["status"], "numerics_unobservable_on_engine_path")
        # no vacuous 0 == 0 coverage pass
        self.assertIsNone(record["gradient_coverage_complete"])
        self.assertEqual(record["attempted_steps"], self.obs.UNOBSERVED)
        # no contradictory alias
        self.assertNotIn("successful_updates", record)
        # engine and native counters stay distinct
        self.assertEqual(record["engine_successful_updates"], 5)
        self.assertEqual(record["native_successful_logical_updates"],
                         self.obs.NOT_APPLICABLE)
        # must not imply hardware validation
        self.assertFalse(record["deepspeed_hardware_validated"])

    # --- things that must still certify / still fail ------------------------
    def test_clean_native_run_still_certifies(self):
        _, v = self._certify(native=self._native())
        self.assertEqual(v["status"], "ok")
        self.assertTrue(v["certified"])

    def test_nonfinite_gradients_do_not_certify(self):
        _, v = self._certify(native=self._native(nonfinite_gradient_steps=2,
                                                 gradient_coverage_complete=False))
        self.assertNotEqual(v["status"], "ok")

    def test_update_shortfall_does_not_certify(self):
        _, v = self._certify(native=self._native(native_successful_logical_updates=4))
        self.assertEqual(v["status"], "updates_unverified")

    def test_nonfinite_losses_do_not_certify(self):
        _, v = self._certify(native=self._native(), losses=[1.0, float("nan")])
        self.assertEqual(v["status"], "updates_unverified")

    def test_deepspeed_without_diagnostics_can_verify_updates_only(self):
        na = self.obs.not_applicable_native_summary("opaque")
        record, v = self._certify(
            mode="deepspeed", checks=False, native=na,
            engine={"engine_update_boundaries": 5, "engine_skipped_updates": 0,
                    "engine_base_optimizer_calls": 5})
        self.assertEqual(v["status"], "ok")
        self.assertTrue(v["update_claim_verified"])
        self.assertIsNone(v["numerical_diagnostic_certified"])
        self.assertIsNone(record["gradient_coverage_complete"])
        self.assertFalse(record["deepspeed_hardware_validated"])

    def test_deepspeed_skipped_updates_do_not_certify(self):
        na = self.obs.not_applicable_native_summary("opaque")
        _, v = self._certify(mode="deepspeed", checks=False, native=na,
                             engine={"engine_update_boundaries": 5,
                                     "engine_skipped_updates": 1,
                                     "engine_base_optimizer_calls": 5})
        self.assertEqual(v["status"], "updates_unverified")


class ObserverCoverageTests(unittest.TestCase):
    def setUp(self):
        self.mod = module_at("rc_obs2", "studies/titan-1080/bench/update_observer.py")

    def _observer(self, checks=True):
        return self.mod.UpdateObserver(numerical_checks=checks, finite_fn=finite_fn)

    def test_empty_gradients_are_not_coverage(self):
        opt = FakeOptimizer([FakeParam(None)])
        obs = self._observer(); obs.attach(opt)
        for _ in range(5):
            obs.begin_step(); obs.observe_gradients(opt, scaling="none")
            opt.step(); obs.end_step()
        s = obs.summary()
        self.assertEqual(s["missing_gradient_steps"], 5)
        self.assertFalse(s["gradient_coverage_complete"])

    def test_unobserved_attempts_are_not_coverage(self):
        opt = FakeOptimizer([FakeParam(FakeGrad(True))])
        obs = self._observer(); obs.attach(opt)
        for _ in range(5):
            obs.begin_step(); opt.step(); obs.end_step()      # never observed
        s = obs.summary()
        self.assertEqual(s["unobserved_attempt_steps"], 5)
        self.assertFalse(s["gradient_coverage_complete"])

    def test_zero_attempts_is_unknown_not_complete(self):
        obs = self._observer()
        self.assertIsNone(obs.summary()["gradient_coverage_complete"])

    def test_skipped_step_still_observed(self):
        params = [FakeParam(FakeGrad(True))]
        opt = FakeOptimizer(params)
        obs = self._observer(); obs.attach(opt)
        for i in range(6):
            bad = i in (2, 5)
            params[0].grad = FakeGrad(not bad)
            obs.begin_step(); obs.observe_gradients(opt, scaling="unscaled")
            if not bad: opt.step()
            obs.record_scale(65536.0, 32768.0 if bad else 65536.0)
            obs.end_step()
        s = obs.summary()
        self.assertEqual(s["attempted_steps"], 6)
        self.assertEqual(s["native_successful_logical_updates"], 4)
        self.assertEqual(s["nonfinite_gradient_steps"], 2)
        self.assertTrue(s["gradient_coverage_complete"])

    def test_scale_change_alone_is_not_success(self):
        opt = FakeOptimizer([FakeParam(FakeGrad(True))])
        obs = self._observer(); obs.attach(opt)
        obs.begin_step(); obs.observe_gradients(opt, scaling="unscaled")
        obs.record_scale(65536.0, 32768.0); obs.end_step()
        s = obs.summary()
        self.assertEqual(s["native_successful_logical_updates"], 0)
        self.assertEqual(s["native_skipped_steps"], 1)

    def test_base_calls_and_logical_updates_separate(self):
        opt = FakeOptimizer([FakeParam(FakeGrad(True))])
        obs = self._observer(); obs.attach(opt)
        obs.begin_step(); obs.observe_gradients(opt, scaling="none")
        opt.step(); opt.step(); opt.step(); obs.end_step()
        s = obs.summary()
        self.assertEqual(s["native_successful_logical_updates"], 1)
        self.assertEqual(s["base_optimizer_calls"], 3)

    def test_unknown_scaling_rejected(self):
        opt = FakeOptimizer([FakeParam(FakeGrad(True))])
        obs = self._observer(); obs.attach(opt)
        obs.begin_step()
        with self.assertRaises(ValueError):
            obs.observe_gradients(opt, scaling="probably-fine")


# ===========================================================================
# R05 - diagnostic-only operations must not run in a clean timing path
# ===========================================================================
class DiagnosticsScopeTests(unittest.TestCase):
    """AST proof that every diagnostic-only call in the trainer is guarded by
    args.numerical_checks. Structural, so it cannot drift silently."""

    DIAGNOSTIC_CALLS = ("get_scale", "unscale_", "observe_gradients", "record_scale")

    def test_diagnostic_calls_are_guarded_by_numerical_checks(self):
        tree = ast.parse((ROOT / "studies/titan-1080/bench/train_llm.py").read_text())

        def guards(node):
            return (isinstance(node, ast.If)
                    and any(isinstance(n, ast.Attribute) and n.attr == "numerical_checks"
                            for n in ast.walk(node.test)))

        unguarded = []

        def walk(node, guarded):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.If) and guards(child):
                    for sub in child.body:
                        walk(sub, True)
                    for sub in child.orelse:
                        walk(sub, guarded)
                    continue
                if isinstance(child, ast.Call):
                    func = child.func
                    name = getattr(func, "attr", getattr(func, "id", None))
                    if name in self.DIAGNOSTIC_CALLS and not guarded:
                        unguarded.append((name, child.lineno))
                walk(child, guarded)

        walk(tree, False)
        self.assertEqual(unguarded, [],
                         f"diagnostic-only calls outside a numerical_checks guard: "
                         f"{unguarded}")

    def test_pre_step_hook_is_not_reintroduced(self):
        text = (ROOT / "studies/titan-1080/bench/train_llm.py").read_text()
        self.assertNotIn("register_step_pre_hook", text)

    def test_unscale_precedes_observe_precedes_step(self):
        text = (ROOT / "studies/titan-1080/bench/train_llm.py").read_text()
        i_u = text.index("scaler.unscale_(optimizer)")
        i_o = text.index('observer.observe_gradients(optimizer, scaling="unscaled")')
        i_s = text.index("scaler.step(optimizer)", i_o)
        self.assertLess(i_u, i_o)
        self.assertLess(i_o, i_s)


# ===========================================================================
# R02 - gate: unknown or stale state must refuse
# ===========================================================================
GOOD = {0: "GPU-00000000-0000-0000-0000-000000000113",
        1: "GPU-00000000-0000-0000-0000-000000000115"}


def rows_text(devices, rows):
    return "\n".join(
        f"{d}, {rows[d]['uuid']}, {rows[d]['mem']}, {rows[d]['util']}"
        for d in devices if d in rows)


def ok_rows(util=0.0, mem=6.0):
    return {d: {"uuid": GOOD[d], "mem": mem, "util": util} for d in (0, 1)}


def make_query(rows, rc=0, stderr="", raise_exc=None):
    def runner(devices, timeout):
        if raise_exc is not None: raise raise_exc
        return rc, rows_text(devices, rows), stderr
    return runner


def no_apps(timeout): return 0, "", ""


class GateRefusalTests(unittest.TestCase):
    def setUp(self):
        self.mod = module_at("rc_gate", "studies/titan-1080/bench/launch_gate.py")
        self.Refusal = self.mod.GateRefusal

    def _gate(self, devices=(0, 1), **kw):
        kw.setdefault("concurrency_limit", 2)
        kw.setdefault("query_runner", make_query(ok_rows()))
        kw.setdefault("apps_runner", no_apps)
        kw.setdefault("sleep_fn", lambda _s: None)
        kw.setdefault("own_pids", {1234})
        return self.mod.gate(list(devices), **kw)

    def test_baseline_passes(self):
        r = self._gate()
        self.assertEqual(r["gate"], "passed")
        self.assertTrue(r["revalidated_after_settlement"])

    # --- the six reported gate counterexamples ------------------------------
    def test_counterexample_nan_telemetry_refuses(self):
        rows = {0: {"uuid": GOOD[0], "mem": "nan", "util": "nan"},
                1: {"uuid": GOOD[1], "mem": "nan", "util": "nan"}}
        with self.assertRaises(self.Refusal) as cm:
            self._gate(query_runner=make_query(rows))
        self.assertIn("non-finite", str(cm.exception))

    def test_counterexample_infinite_telemetry_refuses(self):
        rows = {0: {"uuid": GOOD[0], "mem": "inf", "util": "0"},
                1: {"uuid": GOOD[1], "mem": "6", "util": "0"}}
        with self.assertRaises(self.Refusal) as cm:
            self._gate(query_runner=make_query(rows))
        self.assertIn("non-finite", str(cm.exception))

    def test_counterexample_negative_telemetry_refuses(self):
        rows = {0: {"uuid": GOOD[0], "mem": "-1", "util": "-1"},
                1: {"uuid": GOOD[1], "mem": "6", "util": "0"}}
        with self.assertRaises(self.Refusal) as cm:
            self._gate(query_runner=make_query(rows))
        self.assertIn("outside the plausible range", str(cm.exception))

    def test_counterexample_out_of_range_utilization_refuses(self):
        rows = {0: {"uuid": GOOD[0], "mem": "6", "util": "250"},
                1: {"uuid": GOOD[1], "mem": "6", "util": "0"}}
        with self.assertRaises(self.Refusal):
            self._gate(query_runner=make_query(rows))

    def test_counterexample_missing_bindings_refuses_in_launch_path(self):
        """Neither allocation nor UUID bindings supplied: the launch path must
        refuse rather than silently skip both checks."""
        with self.assertRaises(self.Refusal) as cm:
            self._gate(require_bindings=True,
                       authorization_expiry_epoch=4_000_000_000.0)
        self.assertIn("authorized device allocation", str(cm.exception))
        with self.assertRaises(self.Refusal) as cm:
            self._gate(require_bindings=True, authorized_devices=(0, 1),
                       authorization_expiry_epoch=4_000_000_000.0)
        self.assertIn("UUID binding", str(cm.exception))

    def test_counterexample_memory_and_process_appearing_during_settle_refuses(self):
        state = {"n": 0}
        def runner(devices, timeout):
            state["n"] += 1
            if state["n"] == 1:
                return 0, rows_text(devices, ok_rows(util=99.0, mem=6.0)), ""
            return 0, rows_text(devices, ok_rows(util=0.0, mem=9000.0)), ""
        def apps(timeout):
            return (0, "", "") if state["n"] <= 1 else \
                   (0, f"{GOOD[1]}, 4321, 1024", "")
        with self.assertRaises(self.Refusal) as cm:
            self._gate(query_runner=runner, apps_runner=apps, settle_sleep_s=0.0)
        msg = str(cm.exception)
        self.assertTrue("foreign compute process" in msg or "unaccounted VRAM" in msg,
                        f"must detect mid-settle change, got: {msg}")

    def test_counterexample_uuid_change_during_settle_refuses(self):
        state = {"n": 0}
        def runner(devices, timeout):
            state["n"] += 1
            rows = ok_rows(util=99.0 if state["n"] == 1 else 0.0)
            if state["n"] > 1:
                rows[1] = dict(rows[1], uuid="GPU-00000000-0000-0000-0000-000000000118")
            return 0, rows_text(devices, rows), ""
        with self.assertRaises(self.Refusal) as cm:
            self._gate(query_runner=runner, expected_uuids=GOOD, settle_sleep_s=0.0)
        self.assertIn("UUID mismatch", str(cm.exception))

    def test_counterexample_malformed_compute_app_uuid_refuses(self):
        def apps(timeout): return 0, "not-a-gpu-uuid, 4321, 1024", ""
        with self.assertRaises(self.Refusal) as cm:
            self._gate(apps_runner=apps)
        self.assertIn("implausible GPU identity", str(cm.exception))

    # --- retained v2 refusals ----------------------------------------------
    def test_failed_query_refuses(self):
        with self.assertRaises(self.Refusal):
            self._gate(query_runner=make_query(ok_rows(), rc=9, stderr="boom"))

    def test_timeout_refuses(self):
        exc = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=15)
        with self.assertRaises(self.Refusal):
            self._gate(query_runner=make_query(ok_rows(), raise_exc=exc))

    def test_malformed_row_refuses(self):
        def bad(devices, timeout): return 0, f"0, {GOOD[0]}, 6", ""
        with self.assertRaises(self.Refusal):
            self._gate(query_runner=bad)

    def test_incomplete_query_refuses(self):
        rows = {0: {"uuid": GOOD[0], "mem": 6.0, "util": 0.0}}
        with self.assertRaises(self.Refusal):
            self._gate(query_runner=make_query(rows))

    def test_unsettled_utilization_refuses(self):
        with self.assertRaises(self.Refusal) as cm:
            self._gate(query_runner=make_query(ok_rows(util=99.0)),
                       settle_attempts=20, settle_sleep_s=0.0)
        self.assertIn("did not settle", str(cm.exception))

    def test_concurrency_breach_refuses(self):
        rows = dict(ok_rows())
        for d in (2, 3):
            rows[d] = {"uuid": f"GPU-dddddddd-0000-0000-0000-00000000000{d}",
                       "mem": 6.0, "util": 0.0}
        with self.assertRaises(self.Refusal) as cm:
            self._gate(devices=(0, 1, 2, 3), concurrency_limit=2,
                       query_runner=make_query(rows))
        self.assertIn("does not raise the concurrency ceiling", str(cm.exception))

    def test_invalid_config_values_refuse(self):
        for kw in ({"concurrency_limit": 0}, {"concurrency_limit": "2"},
                   {"concurrency_limit": True},
                   {"util_threshold_pct": float("nan")},
                   {"util_threshold_pct": -5.0},
                   {"settle_attempts": 0}, {"query_timeout_s": 0.0},
                   {"max_foreign_mem_mib": float("inf")}):
            with self.assertRaises(self.Refusal, msg=f"must refuse {kw}"):
                self._gate(**kw)

    def test_invalid_device_indices_refuse(self):
        with self.assertRaises(self.Refusal):
            self._gate(devices=(0, -1))
        with self.assertRaises(self.Refusal):
            self._gate(devices=(0, 0))
        with self.assertRaises(self.Refusal):
            self._gate(devices=())

    def test_expired_authorization_refuses(self):
        with self.assertRaises(self.Refusal) as cm:
            self._gate(authorization_expiry_epoch=1000.0, now_fn=lambda: 2000.0)
        self.assertIn("authorization expired", str(cm.exception))

    def test_nonfinite_expiry_refuses(self):
        with self.assertRaises(self.Refusal):
            self._gate(authorization_expiry_epoch=float("nan"))

    def test_launch_path_requires_expiry(self):
        with self.assertRaises(self.Refusal) as cm:
            self._gate(require_bindings=True, authorized_devices=(0, 1),
                       expected_uuids=GOOD)
        self.assertIn("authorization expiry", str(cm.exception))

    def test_receipt_reports_final_validated_state(self):
        state = {"n": 0}
        def runner(devices, timeout):
            state["n"] += 1
            return 0, rows_text(devices, ok_rows(
                util=99.0 if state["n"] == 1 else 0.0, mem=6.0 + state["n"])), ""
        receipt = self._gate(query_runner=runner, settle_sleep_s=0.0)
        self.assertGreater(receipt["settle_polls"], 1)
        self.assertEqual(receipt["memory_used_mib"][0], 6.0 + state["n"],
                         "receipt must carry the FINAL observation, not the first")

    def test_full_launch_path_passes_with_complete_bindings(self):
        receipt = self._gate(require_bindings=True, authorized_devices=(0, 1),
                             expected_uuids=GOOD,
                             authorization_expiry_epoch=4_000_000_000.0)
        self.assertEqual(receipt["gate"], "passed")
        self.assertTrue(receipt["bindings_required"])


class ReservationTests(unittest.TestCase):
    """A snapshot is not a reservation."""

    def setUp(self):
        self.mod = module_at("rc_resv", "studies/titan-1080/bench/launch_gate.py")

    def test_concurrent_start_race_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            first = self.mod.DeviceReservation(root, [0, 1], aggregate_limit=4)
            first.acquire()
            second = self.mod.DeviceReservation(root, [1, 2], aggregate_limit=4)
            with self.assertRaises(self.mod.ReservationRefusal) as cm:
                second.acquire()
            self.assertIn("already reserved", str(cm.exception))
            self.assertEqual(second.held, [], "a failed acquire must hold nothing")
            first.release()

    def test_aggregate_concurrency_limit_enforced(self):
        with tempfile.TemporaryDirectory() as root:
            held = self.mod.DeviceReservation(root, [0, 1], aggregate_limit=2)
            held.acquire()
            other = self.mod.DeviceReservation(root, [2, 3], aggregate_limit=2)
            with self.assertRaises(self.mod.ReservationRefusal) as cm:
                other.acquire()
            self.assertIn("aggregate concurrency", str(cm.exception))
            held.release()

    def test_release_allows_reacquisition(self):
        with tempfile.TemporaryDirectory() as root:
            with self.mod.DeviceReservation(root, [0], aggregate_limit=2):
                pass
            self.mod.DeviceReservation(root, [0], aggregate_limit=2).acquire()

    def test_dead_launcher_lease_requires_manual_review(self):
        with tempfile.TemporaryDirectory() as root:
            dead = self.mod.DeviceReservation(root, [0], aggregate_limit=2,
                                              pid_alive=lambda pid: False)
            dead.acquire(); dead.held.clear()          # simulate crash, lease left
            fresh = self.mod.DeviceReservation(root, [0], aggregate_limit=2,
                                              pid_alive=lambda pid: False)
            with self.assertRaises(self.mod.ReservationRefusal):
                fresh.acquire()

    def test_expired_lease_of_live_owner_is_not_reclaimed(self):
        with tempfile.TemporaryDirectory() as root:
            clock = {"t": 1000.0}
            old = self.mod.DeviceReservation(root, [0], aggregate_limit=2, ttl_s=1.0,
                                             now_fn=lambda: clock["t"])
            old.acquire(); old.held.clear()
            clock["t"] = 5000.0
            fresh = self.mod.DeviceReservation(root, [0], aggregate_limit=2,
                                               now_fn=lambda: clock["t"])
            with self.assertRaises(self.mod.ReservationRefusal):
                fresh.acquire()

    def test_unverifiable_pid_keeps_lease_live(self):
        """Fail closed: if liveness cannot be established, do not steal the lease."""
        with tempfile.TemporaryDirectory() as root:
            def unverifiable(pid): raise OSError("cannot verify")
            first = self.mod.DeviceReservation(root, [0], aggregate_limit=2)
            first.acquire(); first.held.clear()
            second = self.mod.DeviceReservation(
                root, [0], aggregate_limit=2,
                pid_alive=self.mod.DeviceReservation._default_alive)
            with self.assertRaises(self.mod.ReservationRefusal):
                second.acquire()


class CleanupHelperTests(unittest.TestCase):
    def setUp(self):
        self.mod = module_at("rc_cleanup", "studies/titan-1080/bench/launch_gate.py")

    def test_signals_only_own_group(self):
        pgids = {11: 100, 12: 100, 13: 999}
        killed = []
        out = self.mod.terminate_owned([11, 12, 13], own_pgid=100,
                                       getpgid=lambda pid: pgids[pid],
                                       kill=lambda pid, sig: killed.append(pid))
        self.assertEqual(sorted(out["signalled"]), [11, 12])
        self.assertEqual(out["refused_foreign"], [13])
        self.assertEqual(sorted(killed), [11, 12])

    def test_not_found_and_permission_denied_are_distinct(self):
        """R04: a PermissionError must NOT be reported as 'already gone'."""
        def getpgid(pid):
            if pid == 1: raise ProcessLookupError(pid)
            raise PermissionError(pid)
        killed = []
        out = self.mod.terminate_owned([1, 2], own_pgid=7, getpgid=getpgid,
                                       kill=lambda p, s: killed.append(p))
        self.assertEqual(out["not_found"], [1])
        self.assertEqual(out["unverifiable"], [2])
        self.assertNotIn(2, out["not_found"])
        self.assertEqual(killed, [], "nothing unverifiable may be signalled")

    def test_group_helper_refuses_own_group(self):
        calls = []
        out = self.mod.terminate_owned_group(
            100, own_pgid=100, killpg=lambda p, s: calls.append(p))
        self.assertEqual(out["action"], "refused")
        self.assertIn("own process group", out["reason"])
        self.assertEqual(calls, [])

    def test_group_helper_refuses_invalid_pgid(self):
        for bad in (0, -5, None, "7"):
            out = self.mod.terminate_owned_group(bad, own_pgid=100,
                                                 killpg=lambda p, s: None)
            self.assertEqual(out["action"], "refused")

    def test_group_helper_distinguishes_outcomes(self):
        def gone(p, s): raise ProcessLookupError(p)
        def denied(p, s): raise PermissionError(p)
        self.assertEqual(self.mod.terminate_owned_group(5, own_pgid=1, killpg=gone)["action"],
                         "not_found")
        self.assertEqual(self.mod.terminate_owned_group(5, own_pgid=1, killpg=denied)["action"],
                         "unverifiable")
        self.assertEqual(self.mod.terminate_owned_group(5, own_pgid=1,
                                                        killpg=lambda p, s: None)["action"],
                         "signalled")


if __name__ == "__main__":
    unittest.main()
