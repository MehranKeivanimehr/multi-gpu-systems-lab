"""CPU regressions for independently reproduced session-d counterexamples.

All queries and child processes are fake. No CUDA, nvidia-smi, or actual signals.
The separate POSIX lifecycle suite still requires a Linux host.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "studies/titan-1080/bench"
sys.path.insert(0, str(BENCH))
import launch_gate
import launch

GOOD = "GPU-00000000-0000-0000-0000-000000000113"


class ReservationRegressionTests(unittest.TestCase):
    def test_disjoint_transactions_cannot_exceed_aggregate_cap(self):
        """Pause A inside its transaction; B must not snapshot or acquire."""
        with tempfile.TemporaryDirectory() as root:
            entered, resume = threading.Event(), threading.Event()
            class Paused(launch_gate.DeviceReservation):
                def _acquire_locked(self):
                    entered.set()
                    if not resume.wait(3):
                        raise AssertionError("test transaction did not resume")
                    return super()._acquire_locked()
            first = Paused(root, [0], aggregate_limit=1)
            errors = []
            def acquire():
                try:
                    first.acquire()
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=acquire)
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
                second = launch_gate.DeviceReservation(root, [1], aggregate_limit=1)
                with self.assertRaises(launch_gate.ReservationRefusal):
                    second.acquire()
            finally:
                resume.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(list(Path(root).glob("*.lease"))), 1)
            with self.assertRaises(launch_gate.ReservationRefusal):
                second.acquire()
            first.release()
            second.acquire()
            second.release()

    def test_incomplete_lease_is_not_reclaimed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "device-0.lease"
            path.touch()
            before = path.stat().st_mtime_ns
            res = launch_gate.DeviceReservation(root, [0], aggregate_limit=1)
            with self.assertRaises(launch_gate.ReservationRefusal):
                res.acquire()
            self.assertEqual(path.read_bytes(), b"")
            self.assertEqual(path.stat().st_mtime_ns, before)

    def test_live_owner_cannot_be_stolen_after_ttl(self):
        with tempfile.TemporaryDirectory() as root:
            clock = {"t": 1000}
            first = launch_gate.DeviceReservation(root, [0], aggregate_limit=1,
                ttl_s=1, now_fn=lambda: clock["t"], pid_alive=lambda pid: True)
            first.acquire()
            before = (Path(root) / "device-0.lease").read_bytes()
            clock["t"] = 99999
            second = launch_gate.DeviceReservation(root, [0], aggregate_limit=1,
                now_fn=lambda: clock["t"], pid_alive=lambda pid: True)
            with self.assertRaises(launch_gate.ReservationRefusal):
                second.acquire()
            self.assertEqual((Path(root) / "device-0.lease").read_bytes(), before)
            first.release()

    def test_release_cannot_delete_a_replacement_lease(self):
        with tempfile.TemporaryDirectory() as root:
            first = launch_gate.DeviceReservation(root, [0], aggregate_limit=1)
            first.acquire()
            path = Path(root) / "device-0.lease"
            payload = json.loads(path.read_text())
            payload["token"] = "different-owner"
            path.write_text(json.dumps(payload))
            before = path.read_bytes()
            with self.assertRaises(launch_gate.ReservationRefusal):
                first.release()
            self.assertEqual(path.read_bytes(), before)

    def test_unknown_liveness_cannot_reclaim(self):
        with tempfile.TemporaryDirectory() as root:
            first = launch_gate.DeviceReservation(root, [0], aggregate_limit=2)
            first.acquire()
            def unknown(pid):
                raise PermissionError("unverifiable")
            second = launch_gate.DeviceReservation(root, [0], aggregate_limit=2,
                                                   pid_alive=unknown)
            with self.assertRaises(launch_gate.ReservationRefusal):
                second.acquire()
            first.release()

    def test_abandoned_transaction_lock_refuses(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / ".reservation.lock").mkdir()
            res = launch_gate.DeviceReservation(root, [0], aggregate_limit=1)
            with self.assertRaises(launch_gate.ReservationRefusal):
                res.acquire()
            self.assertEqual(list(Path(root).glob("*.lease")), [])


class ExpiryRegressionTests(unittest.TestCase):
    def test_expiry_during_settlement_refuses(self):
        clock, polls = {"t": 1000.0}, {"n": 0}
        def query(*args):
            polls["n"] += 1
            return 0, f"0, {GOOD}, 6, {99 if polls['n'] == 1 else 0}", ""
        def advance(seconds):
            clock["t"] += seconds
        with self.assertRaises(launch_gate.GateRefusal):
            launch_gate.gate([0], concurrency_limit=1, authorized_devices=[0],
                expected_uuids={0: GOOD}, require_bindings=True,
                authorization_expiry_epoch=1001, now_fn=lambda: clock["t"],
                query_runner=query, apps_runner=lambda t: (0, "", ""),
                sleep_fn=advance, settle_sleep_s=2)

    def test_expiry_while_querying_refuses_even_if_devices_idle(self):
        clock = {"t": 1000.0}
        def query(*args):
            clock["t"] = 1002
            return 0, f"0, {GOOD}, 6, 0", ""
        with self.assertRaises(launch_gate.GateRefusal):
            launch_gate.gate([0], concurrency_limit=1,
                authorization_expiry_epoch=1001, now_fn=lambda: clock["t"],
                query_runner=query, apps_runner=lambda t: (0, "", ""))

    def test_invalid_expiry_and_clock_refuse(self):
        for value in (True, None, float("nan"), float("inf")):
            with self.assertRaises(launch_gate.GateRefusal):
                launch_gate.authorization_remaining_s(value)
        with self.assertRaises(launch_gate.GateRefusal):
            launch_gate.authorization_remaining_s(4000000000, now_fn=lambda: float("nan"))


class IntegratedLauncherRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.kw = dict(case="case", out_root=self.tmp.name, devices=[0],
            concurrency_limit=1, authorized_devices=[0], expected_uuids={0: GOOD},
            authorization_expiry_epoch=time.time()+100,
            python_exe=sys.executable, script=str(BENCH / "launch.py"), script_args=[],
            reservation_root=str(Path(self.tmp.name)/"leases"), aggregate_limit=1,
            grace_s=1, timeout_s=1200,
            gate_kwargs={"query_runner": lambda *a: (0, f"0, {GOOD}, 6, 0", ""),
                         "apps_runner": lambda t: (0, "", "")})
        self.waits, self.environments = [], []
        waits = self.waits
        class Proc:
            pid = 20000
            def wait(self, timeout):
                waits.append(timeout)
                return 0
        def spawn(*args, **kwargs):
            self.environments.append(kwargs["env"])
            return Proc()
        self.kw["popen"] = spawn
        self.cleanup = mock.patch.object(launch, "_bounded_cleanup",
                                         return_value={"performed": True}).start()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(launch.os, "getpgid", create=True, return_value=100).start()
        mock.patch.object(launch, "_confirm_group_absent", return_value=True).start()

    def test_child_timeout_fits_authorization_with_cleanup_allowance(self):
        record = launch.run_case(**self.kw)
        self.assertGreater(self.waits[0], 0)
        self.assertLess(self.waits[0], 97)
        self.assertEqual(record["cleanup_allowance_s"], 3)
        # _reap has a one-second minimum even when grace_s is smaller.
        self.kw.update(case="short_grace", grace_s=0.1)
        small = launch.run_case(**self.kw)
        self.assertEqual(small["cleanup_allowance_s"], 2.1)

    def test_insufficient_time_refuses_before_spawn(self):
        self.kw["authorization_expiry_epoch"] = time.time()+1
        with self.assertRaises(launch.LaunchRefusal):
            launch.run_case(**self.kw)
        self.assertEqual(self.environments, [])

    def test_post_spawn_metadata_failure_still_cleans_up(self):
        real_write = launch._write_json
        def fail_identity(path, payload):
            if Path(path).name == "child_identity.json":
                raise OSError("injected post-spawn write failure")
            return real_write(path, payload)
        with mock.patch.object(launch, "_write_json", side_effect=fail_identity):
            with self.assertRaises(OSError):
                launch.run_case(**self.kw)
        self.cleanup.assert_called_once_with(20000, 1)

    def test_uuid_visibility_separate_from_numeric_telemetry_indices(self):
        launch.run_case(**self.kw)
        self.assertEqual(self.environments[0]["CUDA_VISIBLE_DEVICES"], GOOD)
        self.assertEqual(self.environments[0]["BENCH_PHYS_DEVICES"], "0")

    def test_unconfirmed_cleanup_retains_lease_and_refuses_success(self):
        with mock.patch.object(launch, "_confirm_group_absent", return_value=False):
            record = launch.run_case(**self.kw)
        self.assertIsNone(record["exit_code"])
        self.assertEqual(record["outcome"], "cleanup_unverified")
        self.assertTrue(record["cleanup"]["leases_retained_for_review"])
        self.assertEqual(len(list((Path(self.tmp.name)/"leases").glob("*.lease"))), 1)

    def test_invalid_budget_refuses_before_creating_case(self):
        for name, value in (("timeout_s", float("nan")), ("timeout_s", -1),
                            ("grace_s", float("inf")), ("aggregate_limit", 2)):
            kwargs = dict(self.kw, **{name: value})
            with self.assertRaises(launch.LaunchRefusal):
                launch.run_case(**kwargs)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
