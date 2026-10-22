"""The launcher must satisfy DeepSpeed's init_distributed requirements.

Why this test exists: the first repaired Gate-0 attempt reached
`deepspeed.initialize` and failed with `ModuleNotFoundError: No module named
'mpi4py'`. DeepSpeed takes the standard torch.distributed path only when every
name in its own `required_env` list is present in the environment; otherwise it
falls back to MPI discovery, which imports mpi4py.

The required names are read out of the INSTALLED DeepSpeed source rather than
hardcoded here, so this test tracks the actual dependency instead of a copy of
it that could drift.
"""
import ast, json, os, pathlib, re, subprocess, sys, tempfile, time, unittest

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


def installed_required_env():
    """Read `required_env` from the installed deepspeed/comm/comm.py."""
    import deepspeed.comm.comm as dc
    src = pathlib.Path(dc.__file__).read_text()
    m = re.search(r"required_env\s*=\s*(\[[^\]]*\])", src)
    if not m:
        raise AssertionError("could not locate required_env in the installed "
                             "deepspeed source; this test must be revisited")
    return ast.literal_eval(m.group(1))


class DistEnvContract(unittest.TestCase):

    def test_installed_deepspeed_still_gates_on_required_env(self):
        req = installed_required_env()
        self.assertEqual(sorted(req), sorted(
            ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]))

    def test_mpi4py_is_absent_so_the_fallback_would_still_fail(self):
        """Documents that the fallback is not merely slower: it is fatal here."""
        r = subprocess.run([sys.executable, "-c", "import mpi4py"],
                           capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("No module named 'mpi4py'", r.stderr)

    @unittest.skipUnless(ISO.available(), ISO.skip_reason())
    def test_the_launcher_sets_every_required_name(self):
        """End-to-end: run the real launcher, read what it recorded."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="g0dist_"))
        out = tmp / "run"
        cfg = tmp / "ds.json"; cfg.write_text(json.dumps(DS_CFG))
        sb = CLI.source_tree_digest()
        now = time.time()
        rec = {"authorization_id": "TEST-DIST-ENV",
               "approved_by": "cpu regression fixture",
               "approval_text": "CPU-only; the named UUID does not exist",
               "start_epoch": now - 5, "expiry_epoch": now + 600,
               "device_index": 1, "device_uuid": NO_SUCH_DEVICE,
               "device_name": "NVIDIA GeForce GTX 1080 Ti",
               "case_id": "G0_1080ti_zero1_w1_fp32",
               "source_digest": sb["__tree__"]}
        rec["digest"] = CLI._receipt_digest(rec)
        rp = tmp / "receipt.json"; rp.write_text(json.dumps(rec))
        env = dict(os.environ); env["PYTHONDONTWRITEBYTECODE"] = "1"
        for v in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                  "LOCAL_RANK", "CUDA_VISIBLE_DEVICES", "TITAN_CPU_ONLY_GUARD"):
            env.pop(v, None)          # the launcher must supply them itself
        p = subprocess.run(
            ISO.wrap([sys.executable, "-B", "-m", "zerodiag.cli", "--execute",
             "--output", str(out), "--receipt", str(rp), "--ds-config", str(cfg),
             "--max-cpus", "4", "--memory-budget-gib", "8.0",
             "--monitor-interval-s", "0.5"], writable=[tmp]),
            cwd=ROOT, capture_output=True, text=True, timeout=900, env=env)
        c = json.loads((out / "containment.json").read_text())
        de = c["distributed_env"]
        for name in installed_required_env():
            self.assertIn(name, de, f"launcher does not set {name}")
            self.assertTrue(de[name], f"{name} is empty")
        self.assertEqual(de["WORLD_SIZE"], "1")
        self.assertEqual(de["RANK"], "0")
        self.assertEqual(de["LOCAL_RANK"], "0")
        self.assertEqual(de["MASTER_ADDR"], "127.0.0.1")
        self.assertTrue(29600 <= int(de["MASTER_PORT"]) <= 30599,
                        f"port {de['MASTER_PORT']} outside the per-attempt range")
        # the child still refuses on the absent device, with its own code
        s = json.loads((out / "integration_summary.json").read_text())
        self.assertEqual(s["worker_exit_code"], 90)
        self.assertEqual(p.returncode, 1)
        self.assertTrue(s["group_absent_confirmed"])
        err = (out / "worker.stderr").read_text()
        self.assertNotIn("mpi4py", err)

    def test_port_is_derived_per_attempt_not_fixed(self):
        """Two different attempts must not be forced onto one port."""
        src = (ROOT / "zerodiag/cli.py").read_text()
        self.assertIn("29600 + (int(hashlib.sha256(", src)
        self.assertIn('rec["authorization_id"]', src)
        self.assertNotIn('"MASTER_PORT": "29500"', src)

    def test_world_size_one_claims_no_collectives(self):
        src = (ROOT / "zerodiag/cli.py").read_text()
        self.assertIn("no collective communication", src)
        self.assertIn("world size 1", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
