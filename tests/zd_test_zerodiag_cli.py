"""The actual diagnostic CLI from the final layout: bindings and fail-stop."""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"

def run(*args, env_extra=None):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_extra: env.update(env_extra)
    return subprocess.run([sys.executable, "-B", "-m", "zerodiag.cli", *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)

class CliTests(unittest.TestCase):
    def test_plan_lists_exactly_the_sixteen_retained_contexts(self):
        r = run("--plan")
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        self.assertIn("retained ZeRO contexts: 16", r.stdout)
        self.assertEqual(r.stdout.count("1080Ti"), 8)
        self.assertEqual(r.stdout.count("RTX"), 8)
        self.assertEqual(r.stdout.count("amp_fp16"), 4)

    def test_plan_excludes_what_must_not_be_added(self):
        r = run("--plan")
        self.assertNotIn("1080Ti  ZeRO-1 w1 amp", r.stdout)
        for bad in ("w4 amp", "w6 amp", "RTX    ZeRO-1 w4", "RTX    ZeRO-1 w6"):
            self.assertNotIn(bad, r.stdout)

    def test_plan_records_source_and_installed_bindings(self):
        out = pathlib.Path(tempfile.mkdtemp())
        r = run("--plan", "--output", str(out))
        self.assertEqual(r.returncode, 0)
        d = json.loads((out / "diagnostic_plan.json").read_text())
        for k in ("zerodiag/factories.py", "zerodiag/engine_probe.py",
                  "zerodiag/verdict.py", "zerodiag/update_oracle.py"):
            self.assertRegex(d["source_bindings"][k], r"^[0-9a-f]{64}$")
        self.assertEqual(d["installed_bindings"]["deepspeed_version"], "0.15.1")
        self.assertRegex(d["installed_bindings"]["stage_1_and_2.py"], r"^[0-9a-f]{64}$")
        self.assertFalse(d["installed_bindings"]["torch_cuda_available"])
        self.assertEqual(d["device_contact"], "none")

    def test_all_authorization_fields_are_null(self):
        out = pathlib.Path(tempfile.mkdtemp())
        run("--plan", "--output", str(out))
        d = json.loads((out / "diagnostic_plan.json").read_text())
        self.assertTrue(all(v is None for v in d["authorization"].values()))

    def test_selfcheck_passes_on_cpu_without_device_contact(self):
        r = run("--selfcheck")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr[-300:])
        self.assertIn("rank_verification_successful=True", r.stdout)
        self.assertIn("conditional", r.stdout.lower())

    def test_execute_without_a_receipt_is_refused(self):
        """The execute path is now implemented, so the refusal moved from 'this
        build cannot execute' to 'no receipt was supplied'. Approval is never
        inferred from idle devices, earlier windows or this invocation."""
        r = run("--execute", "--output", "/tmp/zd_no_receipt")
        self.assertEqual(r.returncode, 95)
        self.assertIn("REFUSED", r.stdout)
        self.assertIn("No approval is inferred", r.stdout)

    def test_execute_with_only_a_fabricated_id_is_still_refused(self):
        """Legacy flags alone are not an authorization: a bound receipt is."""
        r = run("--execute", "--output", "/tmp/zd_fab",
                "--authorization-id", "MADE-UP",
                "--authorization-expiry-epoch", "99999999999")
        self.assertEqual(r.returncode, 95)
        self.assertIn("--receipt", r.stdout)

    def test_execute_without_an_output_is_refused(self):
        r = run("--execute")
        self.assertEqual(r.returncode, 95)
        self.assertIn("--output", r.stdout)

    def test_no_mode_is_an_error(self):
        self.assertNotEqual(run().returncode, 0)

    def test_the_guard_blocks_device_entry_points(self):
        r = run("--selfcheck", env_extra={"TITAN_CPU_ONLY_GUARD": "1",
                "PYTHONPATH": str(ROOT.parent / "guards")})
        self.assertEqual(r.returncode, 0, "the selfcheck must not need a device")

if __name__ == "__main__":
    unittest.main()
