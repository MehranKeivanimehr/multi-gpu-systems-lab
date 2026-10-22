"""Launcher boundary and POSIX lifecycle tests (review R03 / R04).

Process tests use ONLY bounded, harmless dummy workers (`time.sleep` in a child
interpreter) and an unrelated sentinel. No benchmark process, no CUDA, no device
query, and no broad killing is involved. Every spawned pid is bounded by a short
sleep and reaped in tearDown.

If the POSIX session/group primitives this requires are unavailable, the affected
tests SKIP (reported as untested) rather than passing vacuously.
"""
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "studies/titan-1080/bench"

POSIX_OK = (os.name == "posix" and hasattr(os, "setsid")
            and hasattr(os, "killpg") and hasattr(os, "getpgid"))

GOOD = {0: "GPU-00000000-0000-0000-0000-000000000113",
        1: "GPU-00000000-0000-0000-0000-000000000115"}
FUTURE = 4_000_000_000.0


def module_at(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rows_text(devices):
    return "\n".join(f"{d}, {GOOD[d]}, 6.0, 0.0" for d in devices)


def clean_gate_kwargs():
    return {"query_runner": lambda devices, timeout: (0, rows_text(devices), ""),
            "apps_runner": lambda timeout: (0, "", ""),
            "sleep_fn": lambda _s: None,
            "own_pids": {os.getpid()}}


class LauncherBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.mod = module_at("ll_launch", "studies/titan-1080/bench/launch.py")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    # --- case identity / path escape ---------------------------------------
    def test_case_path_escape_refused(self):
        for bad in ("../escaped", "a/b", "/abs", "..", ".", "", "x" * 80,
                    "has space", "semi;colon"):
            with self.assertRaises(self.mod.LaunchRefusal, msg=f"must refuse {bad!r}"):
                self.mod.reserve_case_dir(self.root, bad)
        self.assertEqual(sorted(os.listdir(self.root)), [],
                         "a refused case must not create anything")

    def test_fresh_case_directory_reserved_atomically(self):
        path = self.mod.reserve_case_dir(self.root, "caseA")
        self.assertTrue(os.path.isdir(path))
        with self.assertRaises(self.mod.LaunchRefusal) as cm:
            self.mod.reserve_case_dir(self.root, "caseA")
        self.assertIn("already exists", str(cm.exception))

    def test_refusal_does_not_touch_prior_evidence(self):
        """R03: a refused launch previously truncated an existing gate.log."""
        path = self.mod.reserve_case_dir(self.root, "caseB")
        evidence = os.path.join(path, "gate_receipt.json")
        with open(evidence, "w") as fh:
            fh.write('{"original":"evidence"}')
        before_bytes = open(evidence, "rb").read()
        before_mtime = os.stat(evidence).st_mtime_ns
        with self.assertRaises(self.mod.LaunchRefusal):
            self.mod.run_case(
                case="caseB", out_root=self.root, devices=[0, 1],
                concurrency_limit=2, authorized_devices=[0, 1],
                expected_uuids=GOOD, authorization_expiry_epoch=FUTURE,
                python_exe=sys.executable, script=str(BENCH / "launch.py"),
                script_args=[], reservation_root=os.path.join(self.root, "resv"),
                aggregate_limit=2, gate_kwargs=clean_gate_kwargs())
        self.assertEqual(open(evidence, "rb").read(), before_bytes)
        self.assertEqual(os.stat(evidence).st_mtime_ns, before_mtime)

    # --- argument array construction ---------------------------------------
    def test_uuid_binding_cannot_inject_options(self):
        """R03: BENCH_EXPECT_UUIDS could append competing --devices/--limit."""
        with self.assertRaises(self.mod.LaunchRefusal):
            self.mod._parse_uuid_bindings(["--devices 8,9 --concurrency-limit 6"])
        parsed = self.mod._parse_uuid_bindings(["0=--devices"])
        self.assertEqual(parsed, {0: "--devices"},
                         "a hostile value stays a VALUE; it can never become an option")

    def test_duplicate_uuid_binding_refused(self):
        with self.assertRaises(self.mod.LaunchRefusal):
            self.mod._parse_uuid_bindings([f"0={GOOD[0]}", f"0={GOOD[1]}"])

    def test_child_argv_is_a_validated_array(self):
        argv = self.mod.build_child_argv(sys.executable, str(BENCH / "launch.py"),
                                         nproc=2, master_port=29600,
                                         script_args=["--mode", "ddp"])
        self.assertIsInstance(argv, list)
        self.assertTrue(all(isinstance(a, str) for a in argv))
        self.assertEqual(argv[0], sys.executable)
        self.assertIn("--nproc_per_node=2", argv)
        # a hostile script arg is still exactly one argv element
        argv2 = self.mod.build_child_argv(
            sys.executable, str(BENCH / "launch.py"), nproc=1, master_port=29600,
            script_args=["--devices 8,9 --concurrency-limit 6"])
        self.assertIn("--devices 8,9 --concurrency-limit 6", argv2)
        self.assertEqual(sum(1 for a in argv2 if a.startswith("--devices")), 1)

    def test_child_argv_rejects_bad_inputs(self):
        for kw in ({"python_exe": "python"}, {"script": "rel/path.py"},
                   {"nproc": 0}, {"master_port": 80}, {"master_port": 70000}):
            base = dict(python_exe=sys.executable, script=str(BENCH / "launch.py"),
                        nproc=1, master_port=29600, script_args=[])
            base.update(kw)
            with self.assertRaises(self.mod.LaunchRefusal, msg=f"must refuse {kw}"):
                self.mod.build_child_argv(base.pop("python_exe"), base.pop("script"),
                                          **base)
        with self.assertRaises(self.mod.LaunchRefusal):
            self.mod.build_child_argv(sys.executable, str(BENCH / "launch.py"),
                                      nproc=1, master_port=29600,
                                      script_args=["ok", b"bytes"])

    # --- receipt binding ---------------------------------------------------
    def test_receipt_must_match_planned_devices_and_uuids(self):
        receipt = {"devices": [0, 1], "uuids": {0: GOOD[0], 1: GOOD[1]}}
        bound = self.mod.bind_receipt_to_plan(dict(receipt), devices=[0, 1],
                                              expected_uuids=GOOD)
        self.assertEqual(bound["child_visibility_plan"]["cuda_visible_devices"],
                         f"{GOOD[0]},{GOOD[1]}")
        self.assertTrue(bound["receipt_bound_to_plan"])
        with self.assertRaises(self.mod.LaunchRefusal):
            self.mod.bind_receipt_to_plan({"devices": [8, 9], "uuids": {}},
                                          devices=[0, 1], expected_uuids=GOOD)
        wrong = {"devices": [0, 1], "uuids": {0: GOOD[0], 1: "GPU-00000000-0000-0000-0000-000000000117"}}
        with self.assertRaises(self.mod.LaunchRefusal) as cm:
            self.mod.bind_receipt_to_plan(wrong, devices=[0, 1], expected_uuids=GOOD)
        self.assertIn("does not match the gate receipt", str(cm.exception))

    def test_cli_refuses_without_authorization_env(self):
        env = dict(os.environ); env.pop("BENCH_GPU_AUTHORIZED", None)
        proc = subprocess.run(
            [sys.executable, str(BENCH / "launch.py"), "--case", "c",
             "--out-root", self.root, "--devices", "0", "--concurrency-limit", "1",
             "--authorized-devices", "0", "--expect-uuid", f"0={GOOD[0]}",
             "--authorization-expiry-epoch", str(FUTURE),
             "--reservation-root", self.root, "--aggregate-limit", "1",
             "--python-exe", sys.executable, "--script", str(BENCH / "launch.py")],
            capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(proc.returncode, 91)
        self.assertIn("BENCH_GPU_AUTHORIZED", proc.stdout)


@unittest.skipUnless(POSIX_OK, "POSIX session/group primitives unavailable - UNTESTED")
class LauncherLifecycleTests(unittest.TestCase):
    """Real processes, bounded dummy workers only."""

    def setUp(self):
        self.mod = module_at("ll_launch2", "studies/titan-1080/bench/launch.py")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.resv = os.path.join(self.root, "resv")
        self.spawned = []

    def tearDown(self):
        for pid in self.spawned:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        self.tmp.cleanup()

    def _dummy(self, seconds, pidfile=None, spawn_grandchild=False):
        """A harmless bounded worker. Optionally spawns a grandchild and exits."""
        if spawn_grandchild:
            code = (
                "import os,subprocess,sys,time\n"
                f"p=subprocess.Popen([sys.executable,'-c','import time;time.sleep({seconds})'])\n"
                f"open({pidfile!r},'w').write(str(p.pid))\n"
                "sys.exit(0)\n")        # leader exits immediately, child outlives it
        else:
            code = f"import time\ntime.sleep({seconds})\n"
        return [sys.executable, "-c", code]

    def _run(self, case, argv_override, timeout_s=30.0, grace_s=3.0, popen=None):
        captured = {}
        def spawn(argv, **kw):
            proc = subprocess.Popen(argv_override, **kw)
            captured["pid"] = proc.pid
            self.spawned.append(proc.pid)
            return proc
        return self.mod.run_case(
            case=case, out_root=self.root, devices=[0, 1], concurrency_limit=2,
            authorized_devices=[0, 1], expected_uuids=GOOD,
            authorization_expiry_epoch=FUTURE, python_exe=sys.executable,
            script=str(BENCH / "launch.py"), script_args=[],
            reservation_root=self.resv, aggregate_limit=2,
            timeout_s=timeout_s, grace_s=grace_s,
            gate_kwargs=clean_gate_kwargs(), popen=popen or spawn), captured

    def test_normal_exit_records_dedicated_session_and_cleans_up(self):
        record, _ = self._run("normal", self._dummy(0.2))
        self.assertEqual(record["outcome"], "completed")
        self.assertEqual(record["exit_code"], 0)
        self.assertNotEqual(record["child_pgid"], os.getpgid(0),
                            "child must be in its OWN group, not the launcher's")
        self.assertEqual(record["child_pgid"], record["child_pid"],
                          "start_new_session makes the child its own group leader")
        self.assertTrue(record["cleanup"]["performed"])
        identity = json.load(open(os.path.join(record["case_dir"],
                                               "child_identity.json")))
        self.assertTrue(identity["dedicated_session"])
        self.assertNotEqual(identity["pgid"], identity["own_pgid"])

    def test_child_identity_captured_before_waiting(self):
        """R04: the previous shell path looked up the PGID only AFTER waiting, so
        a leader that had already exited could not be identified."""
        pidfile = os.path.join(self.root, "grandchild.pid")
        record, _ = self._run(
            "leader_exits", self._dummy(25, pidfile=pidfile, spawn_grandchild=True),
            timeout_s=20.0, grace_s=3.0)
        self.assertEqual(record["outcome"], "completed")
        self.assertIsNotNone(record["child_pgid"])
        # the identity file exists even though the leader exited immediately
        identity = json.load(open(os.path.join(record["case_dir"],
                                               "child_identity.json")))
        self.assertEqual(identity["pgid"], record["child_pid"])
        # the grandchild, which outlived its leader, was swept by group signal
        deadline = time.time() + 10
        gpid = int(open(pidfile).read().strip())
        self.spawned.append(gpid)
        reaped = False
        while time.time() < deadline:
            try:
                os.kill(gpid, 0)
            except ProcessLookupError:
                reaped = True; break
            time.sleep(0.2)
        self.assertTrue(reaped,
                        "a grandchild outliving its leader must still be cleaned up")

    def test_timeout_triggers_bounded_group_cleanup(self):
        record, cap = self._run("timeout", self._dummy(60), timeout_s=1.0, grace_s=2.0)
        self.assertEqual(record["outcome"], "timeout")
        steps = record["cleanup"]["steps"]
        self.assertTrue(any(s.get("signal") == "SIGTERM" for s in steps))
        deadline = time.time() + 10
        gone = False
        while time.time() < deadline:
            try:
                os.kill(cap["pid"], 0)
            except ProcessLookupError:
                gone = True; break
            time.sleep(0.2)
        self.assertTrue(gone, "timed-out worker must be terminated")

    def test_interruption_triggers_bounded_group_cleanup(self):
        def interrupt_soon():
            time.sleep(1.0)
            os.kill(os.getpid(), signal.SIGINT)
        thread = threading.Thread(target=interrupt_soon, daemon=True)
        thread.start()
        record, cap = self._run("interrupted", self._dummy(60),
                                timeout_s=40.0, grace_s=2.0)
        self.assertEqual(record["outcome"], "interrupted")
        self.assertTrue(record["cleanup"]["performed"])
        deadline = time.time() + 10
        gone = False
        while time.time() < deadline:
            try:
                os.kill(cap["pid"], 0)
            except ProcessLookupError:
                gone = True; break
            time.sleep(0.2)
        self.assertTrue(gone, "interrupted worker must be terminated")

    def test_unrelated_sentinel_is_never_touched(self):
        sentinel = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            start_new_session=True)
        self.spawned.append(sentinel.pid)
        try:
            record, _ = self._run("sentinel", self._dummy(60), timeout_s=1.0,
                                  grace_s=2.0)
            self.assertEqual(record["outcome"], "timeout")
            time.sleep(1.0)
            self.assertIsNone(sentinel.poll(),
                              "an unrelated process in its own session must survive")
            self.assertNotEqual(os.getpgid(sentinel.pid), record["child_pgid"])
        finally:
            sentinel.kill(); sentinel.wait(timeout=10)

    def test_launcher_never_signals_its_own_group(self):
        record, _ = self._run("ownguard", self._dummy(0.2))
        self.assertNotEqual(record["cleanup"]["child_pgid"],
                            record["cleanup"]["own_pgid"])
        for step in record["cleanup"]["steps"]:
            self.assertNotEqual(step.get("pgid"), os.getpgid(0))

    def test_reservation_released_so_next_attempt_can_run(self):
        self._run("resv1", self._dummy(0.2))
        record, _ = self._run("resv2", self._dummy(0.2))
        self.assertEqual(record["outcome"], "completed")

    def test_attempt_metadata_recorded_per_attempt(self):
        record, _ = self._run("meta", self._dummy(0.2))
        for name in ("attempt.json", "command.json", "gate_receipt.json",
                     "child_identity.json", "stdout.log", "stderr.log"):
            self.assertTrue(os.path.isfile(os.path.join(record["case_dir"], name)),
                            f"missing {name}")
        command = json.load(open(os.path.join(record["case_dir"], "command.json")))
        self.assertIn("argv", command)
        self.assertEqual(command["cuda_visible_devices"], f"{GOOD[0]},{GOOD[1]}")
        self.assertIn("launcher_sha256", command["source_metadata"])


if __name__ == "__main__":
    unittest.main()
