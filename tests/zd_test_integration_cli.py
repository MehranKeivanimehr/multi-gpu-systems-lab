"""The real --execute path, exercised with mocked devices and harmless workers.

No production bypass: the CLI always launches a worker SCRIPT and always records
that script's sha256, flagging any worker that is not the frozen production one.
Tests supply a harmless stand-in worker through that same recorded mechanism, so
substitution is visible in the evidence rather than hidden behind a test flag.
"""
from __future__ import annotations

import hashlib, json, os, pathlib, subprocess, sys, tempfile, time, unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
from zerodiag import cli as CLI   # noqa: E402

DS_CFG = {"train_micro_batch_size_per_gpu": 4, "gradient_accumulation_steps": 1,
          "gradient_clipping": 1.0, "zero_allow_untested_optimizer": True,
          "zero_optimization": {"stage": 1}, "fp16": {"enabled": False},
          "bf16": {"enabled": False}}

# A harmless worker: writes the evidence shape the CLI verifies, touches no device.
GOOD_WORKER = '''
import argparse, json, pathlib, sys
ap = argparse.ArgumentParser()
for f in ("--output","--receipt","--ds-config"): ap.add_argument(f, required=True)
for f in ("--case-id","--rank","--world-size","--precision"): ap.add_argument(f)
ap.add_argument("--zero-stage"); ap.add_argument("--batch-size"); ap.add_argument("--seq-len")
ap.add_argument("--max-cpus"); ap.add_argument("--memory-budget-gib")
ap.add_argument("--monitor-interval-s"); ap.add_argument("--chunk-elems")
a = ap.parse_args()
out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
(out/"integration_evidence.json").write_text(json.dumps(
    {"stage": "completed", "device_contact": "mocked: no device touched",
     "verdict": {"rank_verification_successful": True}}, indent=2))
sys.exit(0)
'''
BAD_WORKERS = {
    "nonzero_exit": 'import sys, json, pathlib, argparse\n'
        'ap=argparse.ArgumentParser()\n'
        '[ap.add_argument(f, required=True) for f in ("--output","--receipt","--ds-config")]\n'
        '[ap.add_argument(f) for f in ("--case-id","--rank","--world-size","--precision",'
        '"--max-cpus","--memory-budget-gib","--monitor-interval-s","--chunk-elems")]\n'
        'ap.add_argument("--zero-stage");ap.add_argument("--batch-size");ap.add_argument("--seq-len")\n'
        'a=ap.parse_args();p=pathlib.Path(a.output);p.mkdir(parents=True,exist_ok=True)\n'
        '(p/"integration_evidence.json").write_text(json.dumps({"stage":"completed"}))\n'
        'sys.exit(3)\n',
    "no_evidence": 'import sys, argparse\n'
        'ap=argparse.ArgumentParser()\n'
        '[ap.add_argument(f, required=True) for f in ("--output","--receipt","--ds-config")]\n'
        '[ap.add_argument(f) for f in ("--case-id","--rank","--world-size","--precision",'
        '"--max-cpus","--memory-budget-gib","--monitor-interval-s","--chunk-elems")]\n'
        'ap.add_argument("--zero-stage");ap.add_argument("--batch-size");ap.add_argument("--seq-len")\n'
        'ap.parse_args();sys.exit(0)\n',
    "incomplete_evidence": 'import sys, json, pathlib, argparse\n'
        'ap=argparse.ArgumentParser()\n'
        '[ap.add_argument(f, required=True) for f in ("--output","--receipt","--ds-config")]\n'
        '[ap.add_argument(f) for f in ("--case-id","--rank","--world-size","--precision",'
        '"--max-cpus","--memory-budget-gib","--monitor-interval-s","--chunk-elems")]\n'
        'ap.add_argument("--zero-stage");ap.add_argument("--batch-size");ap.add_argument("--seq-len")\n'
        'a=ap.parse_args();p=pathlib.Path(a.output);p.mkdir(parents=True,exist_ok=True)\n'
        '(p/"integration_evidence.json").write_text(json.dumps({"stage":"refused","reason":"ProbeRefusal: layout"}))\n'
        'sys.exit(90)\n',
    "orphan_child": 'import sys, json, pathlib, argparse, subprocess\n'
        'ap=argparse.ArgumentParser()\n'
        '[ap.add_argument(f, required=True) for f in ("--output","--receipt","--ds-config")]\n'
        '[ap.add_argument(f) for f in ("--case-id","--rank","--world-size","--precision",'
        '"--max-cpus","--memory-budget-gib","--monitor-interval-s","--chunk-elems")]\n'
        'ap.add_argument("--zero-stage");ap.add_argument("--batch-size");ap.add_argument("--seq-len")\n'
        'a=ap.parse_args();p=pathlib.Path(a.output);p.mkdir(parents=True,exist_ok=True)\n'
        'subprocess.Popen([sys.executable,"-c","import time;time.sleep(400)"])\n'
        '(p/"integration_evidence.json").write_text(json.dumps({"stage":"completed"}))\n'
        'sys.exit(0)\n',
}


class IntegrationCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.cfg = self.tmp / "ds.json"; self.cfg.write_text(json.dumps(DS_CFG))
        self.out = self.tmp / "out"
        self.sb = CLI.source_tree_digest()

    def worker(self, body, name="w.py"):
        p = self.tmp / name; p.write_text(body); return p

    def receipt(self, **over):
        now = time.time()
        rec = {"authorization_id": "TEST-SYNTH-1", "approved_by": "synthetic-fixture",
               "approval_text": "synthetic test approval, not a real authorization",
               "start_epoch": now - 5, "expiry_epoch": now + 600,
               "device_index": 1, "device_uuid": "GPU-00000000-0000-0000-0000-000000000101",
               "device_name": "NVIDIA GeForce GTX 1080 Ti",
               "case_id": CLI.INTEGRATION_CASE["case_id"],
               "source_digest": self.sb["__tree__"]}
        rec.update(over)
        rec["digest"] = CLI._receipt_digest(rec)
        if over.get("_break_digest"):
            rec["digest"] = "0" * 64
        p = self.tmp / f"rec_{abs(hash(json.dumps(rec, sort_keys=True, default=str)))}.json"
        p.write_text(json.dumps(rec, default=str))
        return p

    def run_cli(self, *args):
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run([sys.executable, "-B", "-m", "zerodiag.cli", *args],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=600, env=env)


    def run_harness(self, worker_path, receipt_path):
        """Drive a HARMLESS worker through the post-bind path only.

        The production bind in execute() refuses substituted worker bytes, so
        injection happens here, in the test harness, by calling the post-bind
        entry point directly. No command-line flag reaches it.
        """
        driver = self.tmp / "harness_driver.py"
        driver.write_text(
            "import argparse, hashlib, json, pathlib, sys\n"
            "sys.path.insert(0, %r)\n"
            "from zerodiag import cli as CLI\n"
            "a = argparse.Namespace(output=sys.argv[1], receipt=sys.argv[2],\n"
            "    ds_config=sys.argv[4], worker_script=sys.argv[3],\n"
            "    max_cpus=4, memory_budget_gib=8.0, monitor_interval_s=0.5)\n"
            "sb = CLI.source_tree_digest()\n"
            "v = CLI.validate_receipt(a.receipt, source_bindings_now=sb)\n"
            "if not v['ok']:\n"
            "    print('REFUSED: ' + v['reason']); raise SystemExit(95)\n"
            "out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)\n"
            "inv = {'integration_inventory': [CLI.INTEGRATION_CASE], 'n_cases': 1,\n"
            "       'authorization': v['receipt']}\n"
            "(out / 'integration_inventory.json').write_text(\n"
            "    json.dumps(inv, indent=2, sort_keys=True, default=str))\n"
            "wsha = hashlib.sha256(pathlib.Path(a.worker_script).read_bytes()).hexdigest()\n"
            "raise SystemExit(CLI._run_bound_worker(\n"
            "    a, v['receipt'], out, a.worker_script, wsha, sb))\n"
            % str(ROOT))
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, "-B", str(driver), str(self.out), str(receipt_path),
             str(worker_path), str(self.cfg)],
            cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)

    def test_production_execute_refuses_a_substituted_worker(self):
        """R2: the production path refuses BEFORE spawning any child."""
        w = self.worker(GOOD_WORKER, "sub.py")
        r = self.run_cli("--execute", "--output", str(self.out),
                         "--receipt", str(self.receipt()),
                         "--ds-config", str(self.cfg), "--worker-script", str(w))
        self.assertEqual(r.returncode, 95, r.stdout + r.stderr[-300:])
        self.assertIn("do not match the approved production binding", r.stdout)
        self.assertIn("No production child was spawned", r.stdout)
        self.assertFalse((self.out / "worker.stdout").exists())
        self.assertFalse((self.out / "integration_evidence.json").exists())

    # ---- the valid path reaches the injected worker ----
    def test_valid_synthetic_approval_reaches_the_worker(self):
        w = self.worker(GOOD_WORKER)
        r = self.run_harness(w, self.receipt())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr[-300:])
        ev = json.loads((self.out / "integration_evidence.json").read_text())
        self.assertEqual(ev["stage"], "completed")
        s = json.loads((self.out / "integration_summary.json").read_text())
        self.assertTrue(s["group_absent_confirmed"])
        self.assertTrue(s["authorization_revoked"])
        self.assertFalse(s["worker_is_frozen_production"],
                         "a substituted worker must be recorded as such")
        self.assertFalse(s["worker_is_frozen_production"])

    def test_the_inventory_contains_exactly_one_case(self):
        w = self.worker(GOOD_WORKER)
        self.run_harness(w, self.receipt())
        inv = json.loads((self.out / "integration_inventory.json").read_text())
        self.assertEqual(inv["n_cases"], 1)
        self.assertEqual(len(inv["integration_inventory"]), 1)
        c = inv["integration_inventory"][0]
        self.assertEqual(c["world_size"], 1)
        self.assertEqual(c["stage"], 1)
        self.assertEqual(c["precision"], "fp32")
        self.assertEqual(c["warmup_steps"], 0)
        self.assertEqual(c["audited_boundaries"], 1)
        self.assertEqual(c["retries"], 0)
        self.assertFalse(c["amp"])
        self.assertFalse(c["fault_injection"])
        self.assertEqual(c["worker_timeout_s"], 180)

    # ---- approval refusals ----
    def test_absent_receipt_refuses(self):
        r = self.run_cli("--execute", "--output", str(self.out),
                         "--ds-config", str(self.cfg))
        self.assertEqual(r.returncode, 95)
        self.assertIn("No approval is inferred", r.stdout)

    def test_fabricated_receipt_digest_refuses(self):
        r = self.run_cli("--execute", "--output", str(self.out), "--ds-config", str(self.cfg),
                         "--receipt", str(self.receipt(_break_digest=True)))
        self.assertEqual(r.returncode, 95)
        self.assertIn("digest does not match", r.stdout)

    def test_expired_receipt_refuses(self):
        now = time.time()
        r = self.run_cli("--execute", "--output", str(self.out), "--ds-config", str(self.cfg),
                         "--receipt", str(self.receipt(start_epoch=now-900, expiry_epoch=now-10)))
        self.assertEqual(r.returncode, 95)
        self.assertIn("not currently valid", r.stdout)

    def test_not_yet_valid_receipt_refuses(self):
        now = time.time()
        r = self.run_cli("--execute", "--output", str(self.out), "--ds-config", str(self.cfg),
                         "--receipt", str(self.receipt(start_epoch=now+300, expiry_epoch=now+600)))
        self.assertEqual(r.returncode, 95)

    def test_a_window_longer_than_fifteen_minutes_refuses(self):
        now = time.time()
        r = self.run_cli("--execute", "--output", str(self.out), "--ds-config", str(self.cfg),
                         "--receipt", str(self.receipt(start_epoch=now-5, expiry_epoch=now+3600)))
        self.assertEqual(r.returncode, 95)
        self.assertIn("exceeds 15 minutes", r.stdout)

    def test_mismatched_source_digest_refuses(self):
        r = self.run_cli("--execute", "--output", str(self.out), "--ds-config", str(self.cfg),
                         "--receipt", str(self.receipt(source_digest="f"*64)))
        self.assertEqual(r.returncode, 95)
        self.assertIn("different source tree", r.stdout)

    def test_mismatched_case_id_refuses(self):
        r = self.run_cli("--execute", "--output", str(self.out), "--ds-config", str(self.cfg),
                         "--receipt", str(self.receipt(case_id="G3_everything")))
        self.assertEqual(r.returncode, 95)
        self.assertIn("not\n" if False else "frozen integration case", r.stdout)

    def test_each_missing_required_field_refuses(self):
        for f in ("authorization_id", "approved_by", "approval_text", "device_uuid",
                  "device_name", "case_id", "source_digest"):
            with self.subTest(field=f):
                r = self.run_cli("--execute", "--output", str(self.out),
                                 "--ds-config", str(self.cfg),
                                 "--receipt", str(self.receipt(**{f: ""})))
                self.assertEqual(r.returncode, 95)

    def test_insufficient_remaining_window_refuses(self):
        now = time.time()
        r = self.run_cli("--execute", "--output", str(self.out), "--ds-config", str(self.cfg),
                         "--receipt", str(self.receipt(start_epoch=now-5, expiry_epoch=now+30)))
        self.assertEqual(r.returncode, 95)
        self.assertIn("including bounded cleanup", r.stdout)

    # ---- child failures stop and return nonzero ----
    def test_worker_nonzero_exit_stops(self):
        r = self.run_harness(self.worker(BAD_WORKERS["nonzero_exit"], "b1.py"),
                             self.receipt())
        self.assertEqual(r.returncode, 1)
        self.assertIn("worker exited 3", r.stdout)

    def test_missing_evidence_stops(self):
        r = self.run_harness(self.worker(BAD_WORKERS["no_evidence"], "b2.py"),
                             self.receipt())
        self.assertEqual(r.returncode, 1)
        self.assertIn("no readable evidence", r.stdout)

    def test_incomplete_evidence_stops(self):
        r = self.run_harness(self.worker(BAD_WORKERS["incomplete_evidence"], "b3.py"),
                             self.receipt())
        self.assertEqual(r.returncode, 1)
        self.assertIn("incomplete evidence", r.stdout)

    def test_unconfirmed_cleanup_stops(self):
        r = self.run_harness(self.worker(BAD_WORKERS["orphan_child"], "b4.py"),
                             self.receipt())
        self.assertEqual(r.returncode, 1)
        self.assertIn("cleanup could not be confirmed", r.stdout)
        subprocess.run(["pkill", "-f", "time.sleep(400)"], capture_output=True)

    def test_plan_still_lists_sixteen_contexts(self):
        r = self.run_cli("--plan")
        self.assertEqual(r.returncode, 0)
        self.assertIn("retained ZeRO contexts: 16", r.stdout)

    def test_selfcheck_still_passes(self):
        self.assertEqual(self.run_cli("--selfcheck").returncode, 0)


if __name__ == "__main__":
    unittest.main()
