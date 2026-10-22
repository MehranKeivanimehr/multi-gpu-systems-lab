"""Harmless Linux process tests only; never import Torch, vLLM or CUDA."""
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies/titan-1080"))
from zerodiag import ownership, containment


@unittest.skipUnless(sys.platform == "linux", "requires real Linux /proc and sessions")
class StreamingAggregate(unittest.TestCase):
    def test_server_and_escaped_descendant_counted_and_cleaned_without_sentinel(self):
        own = ownership.JobOwnership("cpu-stream-" + os.urandom(16).hex(),
                                     excluded_pgids=(os.getpgid(0),))
        child_code = "import time; payload=bytearray(16*1024*1024); time.sleep(30)"
        parent_code = ("import subprocess,sys,time; payload=bytearray(16*1024*1024); "
                       f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
                       "print(p.pid,flush=True); time.sleep(30)")
        sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        server = None
        try:
            server = subprocess.Popen([sys.executable, "-c", parent_code],
                                      env=own.env_for_child(), start_new_session=True,
                                      stdout=subprocess.PIPE, text=True)
            own.register(0, server.pid)
            deadline = time.monotonic() + 5
            pids = []
            while time.monotonic() < deadline:
                pids = own.owned_pids()
                if len(pids) >= 2 and own.accounting()["total_bytes"] > 30 * 1024 * 1024:
                    break
                time.sleep(.05)
            self.assertGreaterEqual(len(pids), 2)
            self.assertNotIn(os.getpid(), pids)
            self.assertNotIn(sentinel.pid, pids)
            with mock.patch.object(containment, "owned_pids", return_value=[os.getpid()]):
                m = containment.MemoryMonitor(os.getpgid(0), budget_bytes=128 * 1024 * 1024,
                                              additional_pids=own.owned_pids)
                s = m.sample()
            self.assertTrue(set(pids).issubset(s["pids"]))
            self.assertGreater(s["bytes"], 30 * 1024 * 1024)
            report = own.terminate(grace_s=1)
            server.wait(timeout=5)
            self.assertTrue(report["owned_absent_confirmed"])
            self.assertEqual(report["foreign_processes_signalled"], 0)
            self.assertIsNone(sentinel.poll())
        finally:
            own.terminate(grace_s=1)
            if server is not None:
                server.wait(timeout=5)
                if server.stdout:
                    server.stdout.close()
            sentinel.terminate()
            sentinel.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
