"""Lifecycle acceptance on the ROUTED path, with real POSIX processes.

These exercise `launch.run_case` -- the entry point the routed driver uses and
that future jobs will actually use -- not a parallel test-only path. Device
queries are injected; the child processes are real, harmless `sleep`-like
Python processes, and an unrelated sentinel must survive every case.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "studies/titan-1080/bench"
UUID0 = "GPU-00000000-0000-0000-0000-000000000104"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def clean_gate_kwargs(devices=(0,)):
    row = "\n".join(f"{d}, {UUID0}, 6.0, 0.0" for d in devices) + "\n"
    return {"query_runner": lambda d, t: (0, row, ""),
            "apps_runner": lambda t: (0, "", ""),
            "settle_attempts": 2, "settle_sleep_s": 0.0,
            "sleep_fn": lambda s: None}


class _Sentinel:
    """An unrelated harmless process that must never be signalled."""
    def __enter__(self):
        self.p = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True)
        return self

    def alive(self):
        return self.p.poll() is None

    def __exit__(self, *a):
        try:
            self.p.terminate(); self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
        return False


class RoutedLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = load("launch_rt", BENCH / "launch.py")
        cls.base = pathlib.Path(tempfile.mkdtemp())
        cls._n = 0

    def dirs(self):
        type(self)._n += 1
        out = self.base / f"out{self._n}"; res = self.base / f"res{self._n}"
        out.mkdir(parents=True, exist_ok=True); res.mkdir(parents=True, exist_ok=True)
        return str(out), str(res)

    def run_case(self, *, script_body, timeout_s=20.0, case=None, devices=(0,),
                 out=None, res=None, **kw):
        out_root, res_root = (out, res) if out else self.dirs()
        src = self.base / f"child{self._n}.py"
        src.write_text(script_body)
        params = dict(
            case=case or f"C{self._n}", out_root=out_root, devices=list(devices),
            concurrency_limit=6, authorized_devices=list(devices),
            expected_uuids={d: UUID0 for d in devices},
            authorization_expiry_epoch=time.time() + 3600.0,
            python_exe=sys.executable, script=str(src), script_args=[],
            reservation_root=res_root, aggregate_limit=len(devices),
            adapter="plain", timeout_s=timeout_s,
            gate_kwargs=clean_gate_kwargs(devices))
        params.update(kw)
        return self.L.run_case(**params), out_root, res_root

    # -- normal completion ---------------------------------------------------
    def test_a_clean_child_completes_and_cleanup_is_confirmed(self):
        with _Sentinel() as s:
            rec, _, _ = self.run_case(script_body="print('done')\n")
            self.assertEqual(rec.get("exit_code"), 0)
            self.assertTrue(rec.get("cleanup", {}).get("group_absent_confirmed"),
                            f"cleanup not confirmed: {rec.get('cleanup')}")
            self.assertTrue(s.alive(), "the unrelated sentinel was signalled")

    # -- timeout -------------------------------------------------------------
    def test_a_timeout_terminates_the_owned_group_and_confirms_absence(self):
        with _Sentinel() as s:
            rec, _, _ = self.run_case(
                script_body="import time\ntime.sleep(600)\n", timeout_s=2.0)
            self.assertNotEqual(rec.get("exit_code"), 0)
            self.assertTrue(rec.get("cleanup", {}).get("group_absent_confirmed"),
                            f"owned group not confirmed absent: {rec.get('cleanup')}")
            self.assertTrue(s.alive())

    def test_descendants_of_the_child_are_also_reaped(self):
        """The child spawns a grandchild in the same group; both must go."""
        body = (
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
            "print('GRANDCHILD', p.pid, flush=True)\n"
            "time.sleep(600)\n")
        with _Sentinel() as s:
            rec, out_root, _ = self.run_case(script_body=body, timeout_s=3.0)
            self.assertTrue(rec.get("cleanup", {}).get("group_absent_confirmed"))
            txt = ""
            for p in pathlib.Path(out_root).rglob("*"):
                if p.is_file() and p.suffix in (".log", ".txt", ".out"):
                    txt += p.read_text(errors="replace")
            gpid = None
            for line in txt.splitlines():
                if line.startswith("GRANDCHILD"):
                    gpid = int(line.split()[1])
            if gpid:
                for _ in range(50):
                    if not pathlib.Path(f"/proc/{gpid}").exists():
                        break
                    time.sleep(0.1)
                self.assertFalse(pathlib.Path(f"/proc/{gpid}").exists(),
                                 "a descendant survived cleanup")
            self.assertTrue(s.alive())

    def test_leader_exiting_first_still_reaps_the_group(self):
        """The leader exits immediately but leaves a long-lived descendant."""
        body = (
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
            "print('LEADER_EXITING', flush=True)\n")
        with _Sentinel() as s:
            rec, _, _ = self.run_case(script_body=body, timeout_s=10.0)
            self.assertTrue(rec.get("cleanup", {}).get("group_absent_confirmed"),
                            f"group not confirmed absent after leader exit: "
                            f"{rec.get('cleanup')}")
            self.assertTrue(s.alive())

    # -- evidence protection -------------------------------------------------
    def test_an_existing_case_directory_is_never_overwritten(self):
        out_root, res_root = self.dirs()
        rec, _, _ = self.run_case(script_body="print('first')\n",
                                  case="DUP", out=out_root, res=res_root)
        self.assertEqual(rec.get("exit_code"), 0)
        with self.assertRaises(Exception):
            self.run_case(script_body="print('second')\n", case="DUP",
                          out=out_root, res=res_root)

    def test_a_refusal_writes_nothing_into_a_fresh_case_directory(self):
        out_root, res_root = self.dirs()
        before = {p for p in pathlib.Path(out_root).rglob("*")}
        with self.assertRaises(Exception):
            self.run_case(script_body="print('x')\n", case="REFUSED_CASE",
                          out=out_root, res=res_root,
                          authorization_expiry_epoch=time.time() - 1.0)
        after = {p for p in pathlib.Path(out_root).rglob("*")}
        self.assertEqual(before, after,
                         "a refused launch created or modified evidence")

    # -- reservations --------------------------------------------------------
    def test_the_aggregate_lease_is_enforced(self):
        out_root, res_root = self.dirs()
        with self.assertRaises(Exception):
            self.run_case(script_body="print('x')\n", devices=(0, 2),
                          out=out_root, res=res_root, aggregate_limit=1,
                          concurrency_limit=6)

    def test_an_abandoned_lock_is_not_reclaimed_automatically(self):
        """A stale lease left by a dead launcher must block, not be stolen."""
        out_root, res_root = self.dirs()
        gate = load("lg_rt", BENCH / "launch_gate.py")
        r = gate.DeviceReservation(root=res_root, devices=[0], aggregate_limit=1)
        r.acquire()
        try:
            with self.assertRaises(Exception):
                self.run_case(script_body="print('x')\n", devices=(0,),
                              out=out_root, res=res_root, aggregate_limit=1)
        finally:
            r.release()

    def test_releasing_a_lease_permits_a_later_case(self):
        out_root, res_root = self.dirs()
        gate = load("lg_rt2", BENCH / "launch_gate.py")
        r = gate.DeviceReservation(root=res_root, devices=[0], aggregate_limit=1)
        r.acquire(); r.release()
        rec, _, _ = self.run_case(script_body="print('ok')\n", devices=(0,),
                                  out=out_root, res=res_root, aggregate_limit=1)
        self.assertEqual(rec.get("exit_code"), 0)

    # -- the gate is really in the routed path -------------------------------
    def test_a_failing_device_query_refuses_on_the_routed_path(self):
        out_root, res_root = self.dirs()
        bad = dict(clean_gate_kwargs(), query_runner=lambda d, t: (9, "", "driver error"))
        with self.assertRaises(Exception):
            self.run_case(script_body="print('x')\n", out=out_root, res=res_root,
                          gate_kwargs=bad)

    def test_an_unrequested_row_refuses_on_the_routed_path(self):
        out_root, res_root = self.dirs()
        rows = (f"0, {UUID0}, 6.0, 0.0\n"
                f"7, GPU-00000000-0000-0000-0000-000000000119, 6.0, 0.0\n")
        bad = dict(clean_gate_kwargs(), query_runner=lambda d, t: (0, rows, ""))
        with self.assertRaises(Exception):
            self.run_case(script_body="print('x')\n", out=out_root, res=res_root,
                          gate_kwargs=bad)

    def test_the_record_identifies_the_launcher_and_gate_by_hash(self):
        rec, _, _ = self.run_case(script_body="print('ok')\n")
        sm = rec.get("source_metadata") or {}
        self.assertTrue(sm.get("launcher_sha256"))
        self.assertTrue(sm.get("gate_sha256"))
        self.assertTrue(sm.get("script_sha256"))


def _find_retired_bypass() -> tuple[pathlib.Path | None, list[str]]:
    """Locate the retired shell driver across the layouts it legitimately has.

    The campaign drivers sit beside the benchmark package in the delivered
    archive (`tools/` next to `src/multigpu-bench/`) and inside the correction
    candidate (`campaign_tools/` next to `multigpu-bench/`). Searching both,
    and reporting which were tried, keeps the test EXECUTED rather than skipped
    in the reconstructed bundle layout.
    """
    tried = []
    for rel in ("campaign_tools/gpu_attempt.sh",          # candidate layout
                "../tools/gpu_attempt.sh",                 # delivered archive layout
                "../../tools/gpu_attempt.sh"):             # reconstructed src/ layout
        # Both the publication layout (data dirs beside `source/`) and the
        # packaged layout (data dirs inside the repository root) are supported.
        for base in (ROOT.parent, ROOT):
            cand = (base / rel).resolve()
            tried.append(str(cand))
            if cand.exists():
                return cand, tried
        cand = (ROOT.parent / rel).resolve()
        if cand.exists():
            return cand, tried
    return None, tried


class RetiredBypassTests(unittest.TestCase):
    def test_the_shell_bypass_refuses_and_names_the_replacement(self):
        sh, tried = _find_retired_bypass()
        if sh is None:
            self.fail("the retired bypass was not found in any supported layout; "
                      "tried: " + "; ".join(tried))
        r = subprocess.run(["bash", str(sh), "X", "0", "60", "plain"],
                           capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("retired", (r.stdout + r.stderr).lower())
        self.assertIn("launch.py", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
