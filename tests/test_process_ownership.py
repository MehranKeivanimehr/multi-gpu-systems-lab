"""Ownership scoping for the launch guard.

D10-D13 failed because `_is_ours` tested the process group only, and torchrun
puts every rank in its own group, so rank 0 reported rank 1 as foreign compute
load and refused to launch. These tests pin both directions: a sibling rank of
the same launcher is ours, and nothing else gets promoted to ours.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "studies" / "titan-1080" / "bench"


def module_at(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class ProcessOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = module_at("po_common", "studies/titan-1080/bench/common.py")
        cls.spawned: list[subprocess.Popen] = []

    @classmethod
    def tearDownClass(cls):
        for p in cls.spawned:
            try:
                p.terminate(); p.wait(timeout=10)
            except Exception:
                pass

    def _spawn(self, argv, **kw):
        p = subprocess.Popen(argv, **kw)
        type(self).spawned.append(p)
        return p

    # -- positive ------------------------------------------------------------
    def test_our_own_pid_is_ours(self):
        self.assertTrue(self.common._is_ours(os.getpid()))

    def test_a_child_in_its_own_session_is_still_ours(self):
        """start_new_session=True gives a different pgid; descent still holds."""
        c = self._spawn([sys.executable, "-c", "import time; time.sleep(30)"],
                        start_new_session=True)
        self.assertNotEqual(os.getpgid(c.pid), os.getpgid(0))
        self.assertTrue(self.common._is_ours(c.pid))

    def test_a_sibling_rank_of_the_same_launcher_is_ours(self):
        """Models torchrun exactly: an agent spawns rank A and rank B. From
        rank B's view, rank A shares no pgid and is not a descendant -- only the
        shared agent parent identifies it."""
        agent = self._spawn(
            [sys.executable, "-c",
             "import subprocess, sys, time;"
             "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],"
             " start_new_session=True);"
             "print(p.pid, flush=True); time.sleep(30)"],
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        sibling_pid = int(agent.stdout.readline().strip())
        self.addCleanup(lambda: subprocess.run(["kill", str(sibling_pid)],
                                               stderr=subprocess.DEVNULL))
        # Without the patch we are the agent's parent, so descent already holds;
        # pretend we are a rank, i.e. that the agent is OUR parent.
        with mock.patch.object(os, "getppid", return_value=agent.pid):
            self.assertTrue(self.common._shares_launcher_ancestry(sibling_pid))
            self.assertTrue(self.common._is_ours(sibling_pid))

    # -- negative ------------------------------------------------------------
    def test_our_own_ancestor_is_not_ours(self):
        """An ancestor shares our UID but is not part of this job. If UID alone
        decided ownership this would wrongly pass, and the guard would stop
        protecting us from a second job of our own.

        Fixture note (repaired 2026-10-20): this test previously read the real
        grandparent directly. `_is_ours` checks SAME PROCESS GROUP before it
        consults ancestry, and under an ordinary shell invocation
        (`bash -c 'python -m unittest ...'`) the grandparent shares our process
        group, so the guard correctly reported it as ours and the test failed --
        12 of 12 standalone runs. The guard was right; the fixture collided with
        its first check. The test now moves itself into its own process group so
        that the ancestry path is the one actually under test, and restores the
        original group afterwards.
        """
        grandparent = self.common._proc_ppid(os.getppid())
        if not grandparent or grandparent <= 1:
            self.skipTest("no inspectable grandparent process")
        original_pgid = os.getpgid(0)
        try:
            try:
                os.setpgid(0, 0)        # leave the shell's process group
            except OSError as exc:
                self.skipTest(f"cannot create an own process group: {exc}")
            self.assertNotEqual(
                os.getpgid(grandparent), os.getpgid(0),
                "precondition: the ancestor must not share our process group, "
                "or the pgid check would legitimately claim it")
            self.assertEqual(
                self.common._proc_uid(grandparent), os.getuid(),
                "precondition: same UID, so only ancestry can reject it")
            self.assertFalse(self.common._shares_launcher_ancestry(grandparent))
            self.assertFalse(self.common._is_ours(grandparent))
        finally:
            try:
                os.setpgid(0, original_pgid)
            except OSError:
                pass

    def test_pid_one_is_not_ours(self):
        self.assertFalse(self.common._is_ours(1))

    def test_a_process_of_another_uid_is_never_ours(self):
        with mock.patch.object(self.common, "_proc_uid", return_value=os.getuid() + 12345), \
             mock.patch.object(self.common, "_shares_launcher_ancestry", return_value=True):
            c = self._spawn([sys.executable, "-c", "import time; time.sleep(30)"],
                            start_new_session=True)
            self.assertFalse(self.common._is_ours(c.pid),
                             "ancestry must never override a UID mismatch")

    def test_ancestry_is_not_consulted_for_a_uid_mismatch(self):
        calls = []

        def spy(pid, **kw):
            calls.append(pid); return True

        with mock.patch.object(self.common, "_proc_uid", return_value=os.getuid() + 999), \
             mock.patch.object(self.common, "_shares_launcher_ancestry", spy):
            c = self._spawn([sys.executable, "-c", "import time; time.sleep(30)"],
                            start_new_session=True)
            self.common._is_ours(c.pid)
        self.assertEqual(calls, [], "UID must be checked before ancestry")

    def test_a_vanished_pid_is_not_ours(self):
        c = self._spawn([sys.executable, "-c", ""], start_new_session=True)
        c.wait(timeout=30)
        for _ in range(50):
            if not Path(f"/proc/{c.pid}").exists():
                break
            time.sleep(0.1)
        self.assertFalse(self.common._is_ours(c.pid))

    def test_ancestry_walk_is_depth_bounded(self):
        with mock.patch.object(self.common, "_proc_ppid", side_effect=lambda q: q + 1):
            self.assertFalse(self.common._shares_launcher_ancestry(10 ** 6, max_depth=4))

    def test_a_self_parenting_pid_terminates_the_walk(self):
        with mock.patch.object(self.common, "_proc_ppid", return_value=424242):
            self.assertFalse(self.common._shares_launcher_ancestry(424242))


if __name__ == "__main__":
    unittest.main()
