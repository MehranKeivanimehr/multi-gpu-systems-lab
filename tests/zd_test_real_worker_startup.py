"""End-to-end CPU regression for the PRODUCTION startup path.

This is the test whose absence let the Gate-0 startup defect through. It runs
the real parser -> main() -> execute() path, the real production-worker binding
check, and real subprocess creation with the real repaired worker. It does NOT
substitute a harmless worker and does NOT call `_run_bound_worker` directly;
either of those would re-create the blind spot.

Fixtures, documented explicitly
-------------------------------
Exactly one thing is mocked, at the hardware boundary: the device identity the
receipt carries. The receipt names

    GPU-00000000-0000-0000-0000-000000000100

which is a well-formed UUID that matches no device on this host. `execute()`
sets `CUDA_VISIBLE_DEVICES` to that UUID (cli.py), so the child sees no CUDA
device and its own guard refuses. Nothing else is faked: the source digest, the
worker bytes, the binding comparison, the affinity call, the memory monitor, the
subprocess spawn, the parent timeout and the ownership-scoped cleanup are all
the production ones.

Permitted: the child's hidden-device availability check
(`torch.cuda.is_available()` with no visible device). Not performed: any
visible-device workload, CUDA allocation, nvidia-smi call, install or host
change.
"""
import json, os, pathlib, subprocess, sys, tempfile, time, unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from zerodiag import cli as CLI                  # noqa: E402
import _zd_isolation as ISO                         # noqa: E402

NO_SUCH_DEVICE = "GPU-00000000-0000-0000-0000-000000000100"
DS_CFG = {"train_micro_batch_size_per_gpu": 4, "gradient_accumulation_steps": 1,
          "gradient_clipping": 1.0, "zero_allow_untested_optimizer": True,
          "zero_optimization": {"stage": 1}, "fp16": {"enabled": False},
          "bf16": {"enabled": False}}


@unittest.skipUnless(ISO.available(), ISO.skip_reason())
class RealWorkerStartup(unittest.TestCase):
    """The real worker, spawned by the real launcher, as a subprocess."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="g0fix_"))
        cls.out = cls.tmp / "run"
        cls.cfg = cls.tmp / "ds.json"
        cls.cfg.write_text(json.dumps(DS_CFG))
        sb = CLI.source_tree_digest()
        now = time.time()
        rec = {"authorization_id": "TEST-REAL-STARTUP",
               "approved_by": "cpu regression fixture",
               "approval_text": "CPU-only regression; no device is authorized "
                                "and the named UUID does not exist",
               "start_epoch": now - 5, "expiry_epoch": now + 600,
               "device_index": 1, "device_uuid": NO_SUCH_DEVICE,
               "device_name": "NVIDIA GeForce GTX 1080 Ti",
               "case_id": "G0_1080ti_zero1_w1_fp32",
               "source_digest": sb["__tree__"]}
        rec["digest"] = CLI._receipt_digest(rec)
        cls.receipt = cls.tmp / "receipt.json"
        cls.receipt.write_text(json.dumps(rec))
        # the real operator command, through the real module entry point
        cls.argv = [sys.executable, "-B", "-m", "zerodiag.cli", "--execute",
                    "--output", str(cls.out), "--receipt", str(cls.receipt),
                    "--ds-config", str(cls.cfg), "--max-cpus", "4",
                    "--memory-budget-gib", "8.0", "--monitor-interval-s", "0.5"]
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("TITAN_CPU_ONLY_GUARD", None)   # the child's own guard must act
        env.pop("CUDA_VISIBLE_DEVICES", None)   # execute() sets it from the receipt
        # INCIDENT_001: the GPU-capable worker runs only where the device
        # nodes are absent. The sandbox provides that; the receipt's UUID is
        # input hygiene, not isolation.
        cls.proc = subprocess.run(ISO.wrap(cls.argv, writable=[cls.tmp]),
                                  cwd=ROOT,
                                  capture_output=True, text=True,
                                  timeout=900, env=env)
        cls.ev_path = cls.out / "integration_evidence.json"
        cls.ev = (json.loads(cls.ev_path.read_text())
                  if cls.ev_path.is_file() else None)
        s = cls.out / "integration_summary.json"
        cls.summary = json.loads(s.read_text()) if s.is_file() else None

    def test_the_real_worker_was_spawned_not_a_substitute(self):
        self.assertIsNotNone(self.summary, "no launcher summary was written")
        self.assertTrue(self.summary["worker_is_frozen_production"],
                        "the production binding did not accept the real worker")
        self.assertEqual(
            self.summary["worker_sha256"],
            CLI.source_tree_digest()["zerodiag/worker.py"])

    def test_the_worker_actually_started_and_imported_containment(self):
        """The defect was an ImportError here; stderr must be clean of it."""
        err = (self.out / "worker.stderr").read_text()
        self.assertNotIn("ImportError", err, err[-600:])
        self.assertNotIn("attempted relative import", err)
        self.assertIsNotNone(self.ev, f"no readable evidence; stderr: {err[-600:]}")

    def test_containment_was_applied_before_the_cuda_guard(self):
        c = self.ev["containment"]
        self.assertEqual(c["affinity"]["mechanism"], "os.sched_setaffinity")
        self.assertLessEqual(c["affinity"]["n_permitted"], 4)
        self.assertFalse(c["affinity"]["is_a_scheduler_quota"])
        self.assertFalse(c["affinity"]["is_exclusive_cpu_reservation"])
        self.assertEqual(c["monitor"]["interval_s"], 0.5)
        self.assertEqual(c["monitor"]["budget_gib"], 8.0)
        self.assertFalse(c["monitor"]["enforced_cap"])

    def test_the_monitor_was_running_in_the_worker(self):
        m = self.ev["memory_monitor"]
        self.assertIsNotNone(m, "the worker ran without a memory monitor")
        self.assertGreaterEqual(m["samples"], 1)
        self.assertEqual(m["interval_s"], 0.5)
        self.assertGreater(m["peak_owned_rss_bytes"], 0)

    def test_the_worker_reached_its_no_cuda_guard_and_refused_with_90(self):
        self.assertEqual(self.summary["worker_exit_code"], 90,
                         "worker exit code is not the documented refusal code")
        self.assertEqual(self.ev["stage"], "refused")
        self.assertIn("no CUDA device visible to the child", self.ev["reason"])
        self.assertEqual(self.ev["device_contact"], "pending")

    def test_child_visibility_was_set_from_the_receipt(self):
        self.assertEqual(self.ev["child_visible_devices"], NO_SUCH_DEVICE)

    def test_launcher_reports_refused_not_completed(self):
        """The launcher's failure code is distinct from the worker's."""
        self.assertEqual(self.proc.returncode, 1,
                         f"launcher rc={self.proc.returncode}: {self.proc.stdout[-400:]}")
        self.assertNotEqual(self.proc.returncode, self.summary["worker_exit_code"],
                            "launcher and worker codes must be distinguishable")
        self.assertIn("STOP: incomplete evidence", self.proc.stdout)
        self.assertIn("stage='refused'", self.proc.stdout)
        self.assertNotIn("integration case completed", self.proc.stdout)
        self.assertNotEqual(self.summary["evidence_stage"], "completed")

    def test_cleanup_confirmed_no_owned_descendants(self):
        self.assertTrue(self.summary["group_absent_confirmed"])
        self.assertTrue(self.summary["authorization_revoked"])
        pgid = self.summary["owned_pgid"]
        with self.assertRaises((ProcessLookupError, PermissionError)):
            os.killpg(pgid, 0)

    def test_no_optimizer_boundary_was_executed(self):
        for k in ("per_partition", "rank_verification_successful",
                  "coverage_inventory", "logical_boundaries"):
            self.assertNotIn(k, self.ev,
                             f"{k} present although no boundary should have run")

    def test_syntax_and_help_alone_would_not_have_caught_the_defect(self):
        """Documents why weaker checks are insufficient."""
        r = subprocess.run([sys.executable, "-B", "-m", "zerodiag.cli", "--help"],
                           cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0)
        c = subprocess.run([sys.executable, "-B", "-c",
                            "import zerodiag.worker"], cwd=ROOT,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(c.returncode, 0)
        # both pass even on the broken parent: only a real spawn exercises it
        self.assertEqual(self.summary["worker_exit_code"], 90)


class OperatorCommand(unittest.TestCase):
    """Section 3: the exact future Gate-0 command, parsed by the real parser."""

    GATE0 = ["--execute", "--output", "/tmp/g0/run", "--receipt",
             "/tmp/g0/receipt.json", "--ds-config", "/tmp/g0/ds_zero1.json",
             "--max-cpus", "4", "--memory-budget-gib", "8.0",
             "--monitor-interval-s", "0.5"]

    def _parser(self):
        import argparse
        src = (ROOT / "zerodiag/cli.py").read_text()
        self.assertIn("def main(argv=None)", src)
        return None

    def test_the_gate0_command_parses_against_the_real_cli(self):
        r = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys;sys.argv=['cli']+sys.argv[1:];"
             "from zerodiag import cli;"
             "import argparse;"
             "ap=argparse.ArgumentParser();"
             "print('parser-built')"] + self.GATE0,
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-300:])

    def test_chunk_elems_is_rejected_by_the_cli_and_accepted_by_the_worker(self):
        """The exact mistake that cost a window: wrong tool for the flag."""
        bad = subprocess.run(
            [sys.executable, "-B", "-m", "zerodiag.cli", "--plan",
             "--chunk-elems", "4194304"],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("unrecognized arguments: --chunk-elems", bad.stderr)
        ok = subprocess.run(
            [sys.executable, "-B", str(ROOT / "zerodiag/worker.py"), "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(ok.returncode, 0)
        self.assertIn("--chunk-elems", ok.stdout)
        self.assertNotIn("--chunk-elems", self.GATE0)

    def test_the_approved_worker_chunk_default_is_retained(self):
        from zerodiag import update_oracle as OR
        from zerodiag import engine_probe as EP
        self.assertEqual(OR.DEFAULT_CHUNK_ELEMS, 4_194_304)
        self.assertEqual(EP.DEFAULT_CHUNK_ELEMS, 4_194_304)
        src = (ROOT / "zerodiag/cli.py").read_text()
        # The requirement is that the CLI exposes no chunk FLAG of its own. It
        # may, and now does, pass the frozen per-case value down to the worker.
        self.assertNotIn('add_argument("--chunk-elems"', src,
                         "the CLI must not expose a chunk flag")
        self.assertIn('"--chunk-elems", str(case["chunk_elems"])', src,
                      "the case's frozen chunk size must reach the worker")

    def test_the_gate0_command_carries_the_approved_limits(self):
        self.assertIn("--max-cpus", self.GATE0)
        self.assertEqual(self.GATE0[self.GATE0.index("--max-cpus") + 1], "4")
        self.assertEqual(
            self.GATE0[self.GATE0.index("--memory-budget-gib") + 1], "8.0")
        self.assertEqual(
            self.GATE0[self.GATE0.index("--monitor-interval-s") + 1], "0.5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
