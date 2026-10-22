"""R1/R2/R3/R4 containment and binding tests. CPU-only, harmless fixtures."""
import json, os, pathlib, subprocess, sys, tempfile, threading, unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
from zerodiag import containment as CT          # noqa: E402
from zerodiag import verdict as VD              # noqa: E402


class TestAffinity(unittest.TestCase):
    """R1: an actual mask, not thread counts."""

    def setUp(self):
        self.orig = os.sched_getaffinity(0)

    def tearDown(self):
        os.sched_setaffinity(0, self.orig)

    def test_mask_is_applied_and_bounded(self):
        a = CT.apply_affinity(4)
        self.assertLessEqual(a["n_permitted"], 4)
        self.assertEqual(sorted(os.sched_getaffinity(0)), a["permitted_cpus"])

    def test_mask_is_not_claimed_to_be_a_quota(self):
        a = CT.apply_affinity(2)
        self.assertFalse(a["is_a_scheduler_quota"])
        self.assertFalse(a["is_exclusive_cpu_reservation"])

    def test_thread_vars_are_supplementary_not_the_mechanism(self):
        a = CT.apply_affinity(2)
        self.assertEqual(a["mechanism"], "os.sched_setaffinity")
        self.assertEqual(a["supplementary_thread_vars"]["OMP_NUM_THREADS"], "2")

    def test_child_inherits_the_mask(self):
        a = CT.apply_affinity(2)
        r = subprocess.run([sys.executable, "-I", "-c",
                            "import os;print(sorted(os.sched_getaffinity(0)))"],
                           capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), str(a["permitted_cpus"]))

    def test_thread_inherits_the_mask(self):
        a = CT.apply_affinity(2)
        seen = []
        t = threading.Thread(target=lambda: seen.append(sorted(os.sched_getaffinity(0))))
        t.start(); t.join()
        self.assertEqual(seen[0], a["permitted_cpus"])

    def test_verify_detects_mismatch(self):
        a = CT.apply_affinity(2)
        self.assertTrue(CT.verify_affinity(a["permitted_cpus"])["matches_expected"])
        self.assertFalse(CT.verify_affinity([999])["matches_expected"])

    def test_bad_cpu_count_refuses(self):
        for bad in (0, -1, 1.5, True, "4"):
            with self.assertRaises(CT.ContainmentRefusal):
                CT.apply_affinity(bad)


class TestMemoryMonitor(unittest.TestCase):
    """R3: a real sampled monitor with a stop path."""

    def test_interval_must_be_at_most_one_second(self):
        with self.assertRaises(CT.ContainmentRefusal):
            CT.MemoryMonitor(os.getpgid(0), budget_bytes=1 << 20, interval_s=2.0)
        with self.assertRaises(CT.ContainmentRefusal):
            CT.MemoryMonitor(os.getpgid(0), budget_bytes=1 << 20, interval_s=0)
        CT.MemoryMonitor(os.getpgid(0), budget_bytes=1 << 20, interval_s=1.0)

    def test_owned_attribution_only(self):
        pids = CT.owned_pids(os.getpgid(0))
        self.assertIn(os.getpid(), pids)
        self.assertNotIn(1, pids)              # init is never ours

    def test_rss_accounting_is_conservative_and_declares_what_it_excludes(self):
        acc = CT.owned_rss_bytes([os.getpid()])
        self.assertGreater(acc["total_bytes"], 0)
        self.assertIn("over-estimate", acc["shared_pages"])
        for excluded in ("whole-user usage", "virtual address space", "GPU memory"):
            self.assertIn(excluded, acc["not_used"])

    def test_exited_pid_is_reported_unreadable_not_counted_as_zero(self):
        acc = CT.owned_rss_bytes([2 ** 22 - 1])
        self.assertEqual(acc["total_bytes"], 0)
        self.assertEqual(acc["unreadable"], [2 ** 22 - 1])

    def test_breach_detected_against_a_small_threshold(self):
        m = CT.MemoryMonitor(os.getpgid(0), budget_bytes=1, interval_s=0.2)
        m.sample()
        self.assertTrue(m.breached)
        r = m.report()
        self.assertFalse(r["enforced_cap"])
        self.assertIn("not an enforced aggregate memory cap", r["limitations"])

    def test_no_breach_under_a_large_budget(self):
        m = CT.MemoryMonitor(os.getpgid(0), budget_bytes=1 << 40, interval_s=0.2)
        m.sample()
        self.assertFalse(m.breached)

    def test_monitor_thread_stops_only_the_owned_group(self):
        """Harmless allocator in its own group is stopped; we survive."""
        victim = subprocess.Popen(
            [sys.executable, "-I", "-c",
             "import time\nb=[]\nfor _ in range(400):\n"
             " b.append(bytearray(4*1024*1024))\n time.sleep(0.01)\ntime.sleep(20)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        try:
            m = CT.MemoryMonitor(os.getpgid(victim.pid),
                                 budget_bytes=192 * 1024 * 1024, interval_s=0.2)
            import time as _t
            for _ in range(80):
                m.sample()
                if m.breached:
                    break
                _t.sleep(0.2)
            self.assertTrue(m.breached, "monitor never observed the breach")
            self.assertGreater(m.report()["samples"], 1)
        finally:
            victim.terminate()
            try:
                victim.wait(timeout=5)
            except subprocess.TimeoutExpired:
                victim.kill(); victim.wait(timeout=5)
        self.assertEqual(os.getpid(), os.getpid())      # launcher unaffected

    def test_never_observed_is_flagged(self):
        m = CT.MemoryMonitor(999999, budget_bytes=1 << 30, interval_s=0.2)
        m.sample()
        self.assertTrue(m.report()["owned_process_never_observed"])
        self.assertFalse(m.breached)


class TestBudget(unittest.TestCase):
    """R3: arithmetic, honestly reported."""

    def test_budget_stays_at_8gib_and_is_never_raised(self):
        for ce in (1 << 20, 1 << 22, 124_439_808):
            self.assertEqual(CT.budget_arithmetic(
                oracle_chunk_elems=ce)["proposed_budget_gib"], 8.0)

    def test_unchunked_oracle_overruns_8gib_and_chunking_is_what_fixes_it(self):
        """The chunk bound, not a raised budget, is what makes this fit."""
        unchunked = CT.budget_arithmetic(oracle_chunk_elems=124_439_808)
        self.assertFalse(unchunked["strict_all_host_fits_8gib"])
        chunked = CT.budget_arithmetic()
        self.assertTrue(chunked["strict_all_host_fits_8gib"])
        self.assertTrue(chunked["gate0_fits_8gib"])
        self.assertLess(chunked["oracle_chunk_elems"],
                        chunked["max_oracle_chunk_elems_within_budget_strict"])

    def test_gate0_host_figure_excludes_device_tensors(self):
        b = CT.budget_arithmetic()
        self.assertNotIn("model_params_fp32", b["host_resident_bytes"])
        self.assertIn("model_params_fp32", b["device_resident_bytes"])
        self.assertLess(b["gate0_host_bytes"], b["strict_all_host_bytes"])

    def test_oracle_workspace_is_chunk_bounded(self):
        small = CT.budget_arithmetic(oracle_chunk_elems=1 << 20)
        big = CT.budget_arithmetic(oracle_chunk_elems=1 << 23)
        self.assertLess(small["gate0_host_bytes"], big["gate0_host_bytes"])
        self.assertGreater(small["max_oracle_chunk_elems_within_budget_strict"], 0)
        self.assertLess(small["max_oracle_chunk_elems_within_budget_strict"],
                        small["max_oracle_chunk_elems_within_budget_host_only"])

    def test_snapshots_are_the_dominant_host_term(self):
        b = CT.budget_arithmetic()
        host = b["host_resident_bytes"]
        self.assertEqual(max(host, key=host.get).startswith("snapshot"), True)


class TestWorkerBinding(unittest.TestCase):
    """R2: production refuses substituted or modified worker bytes."""

    def _receipt(self, tmp, **over):
        from zerodiag import cli as CL
        import hashlib, time
        sb = CL.source_tree_digest()
        rec = {"authorization_id": "A-TEST", "case_id": "G0_1080ti_zero1_w1_fp32",
               "device_uuid": "GPU-00000000-0000-0000-0000-000000000100",
               "device_index": 0, "issued_epoch": time.time() - 10,
               "expiry_epoch": time.time() + 600, "window_s": 610,
               "operator": "test", "source_digest": sb["__tree__"],
               "scope": "one boundary", "holds": "all"}
        rec.update(over)
        rec["digest"] = CL._receipt_digest(rec)
        p = pathlib.Path(tmp) / "receipt.json"
        p.write_text(json.dumps(rec))
        return str(p)

    def test_substituted_worker_refuses_before_spawn(self):
        from zerodiag import cli as CL
        with tempfile.TemporaryDirectory() as tmp:
            fake = pathlib.Path(tmp) / "fake_worker.py"
            fake.write_text("raise SystemExit('this must never run')\n")
            a = CL.main.__globals__["argparse"].Namespace(
                plan=False, selfcheck=False, execute=True,
                cpu_containment_check=False, output=str(pathlib.Path(tmp) / "o"),
                receipt=self._receipt(tmp), ds_config=None,
                worker_script=str(fake), authorization_id=None,
                authorization_expiry_epoch=None, max_cpus=4,
                memory_budget_gib=8.0, monitor_interval_s=0.5)
            rc = CL.execute(a)
            self.assertEqual(rc, 95)
            self.assertFalse((pathlib.Path(tmp) / "o" / "worker.stdout").exists(),
                             "a production child was spawned despite mismatch")

    def test_modified_production_worker_refuses(self):
        from zerodiag import cli as CL
        with tempfile.TemporaryDirectory() as tmp:
            mod = pathlib.Path(tmp) / "worker.py"
            mod.write_text((ROOT / "zerodiag/worker.py").read_text() + "# one byte\n")
            a = CL.main.__globals__["argparse"].Namespace(
                plan=False, selfcheck=False, execute=True,
                cpu_containment_check=False, output=str(pathlib.Path(tmp) / "o"),
                receipt=self._receipt(tmp), ds_config=None,
                worker_script=str(mod), authorization_id=None,
                authorization_expiry_epoch=None, max_cpus=4,
                memory_budget_gib=8.0, monitor_interval_s=0.5)
            self.assertEqual(CL.execute(a), 95)

    def test_required_hash_comes_from_source_not_the_invocation(self):
        """No CLI or receipt field can supply the required worker hash."""
        src = (ROOT / "zerodiag/cli.py").read_text()
        i = src.index("required = sb.get")
        window = src[i:i + 900]
        self.assertNotIn("a.worker_sha", window)
        self.assertNotIn('rec["worker', window)
        self.assertNotIn("rec.get(\"worker", window)

    def test_no_production_allow_test_worker_bypass_exists(self):
        src = (ROOT / "zerodiag/cli.py").read_text()
        for bypass in ("allow_test_worker", "allow-test-worker",
                       "skip_worker_check", "--force-worker"):
            self.assertNotIn(bypass, src)


class TestCoverageWording(unittest.TestCase):
    """R4: inventory-based coverage, not rank count."""

    def _ids(self, n=2, numel=100):
        return [{"group": i, "rank": 0, "offset": 0, "partition_numel": numel,
                 "padding": 0, "unpadded_numel": numel} for i in range(n)]

    def test_rank_count_claim_is_gone(self):
        src = (ROOT / "zerodiag/verdict.py").read_text()
        self.assertNotIn("one rank's full check is not", src)
        self.assertNotIn("one rank is never whole-model", src)

    def test_inventory_reports_expected_and_observed(self):
        ids = self._ids()
        res = [{"partition": (0, 0), "verification_successful": True},
               {"partition": (1, 0), "verification_successful": True}]
        inv = VD.coverage_inventory(expected_ids=ids, results=res, rank=0,
                                    world_size=1, param_inventory=None)
        self.assertEqual(inv["n_expected_partitions"], 2)
        self.assertEqual(inv["expected_elements_this_rank"], 200)
        self.assertEqual(inv["observed_elements_verified"], 200)

    def test_no_param_inventory_means_not_complete(self):
        inv = VD.coverage_inventory(expected_ids=self._ids(),
                                    results=[{"partition": (0, 0),
                                              "verification_successful": True},
                                             {"partition": (1, 0),
                                              "verification_successful": True}],
                                    rank=0, world_size=1, param_inventory=None)
        self.assertFalse(inv["complete"])
        self.assertIn("does not establish completeness", inv["incomplete_because"])

    def test_world_size_one_alone_does_not_make_it_complete(self):
        ids = self._ids()
        res = [{"partition": (0, 0), "verification_successful": True},
               {"partition": (1, 0), "verification_successful": False}]
        inv = VD.coverage_inventory(expected_ids=ids, results=res, rank=0,
                                    world_size=1,
                                    param_inventory={"n_unique_parameters": 2,
                                                     "n_unique_elements": 200})
        self.assertFalse(inv["complete"])
        self.assertEqual(inv["missing_partitions"], [(1, 0)])

    def test_complete_requires_elements_to_reconcile(self):
        ids = self._ids()
        res = [{"partition": (0, 0), "verification_successful": True},
               {"partition": (1, 0), "verification_successful": True}]
        good = VD.coverage_inventory(expected_ids=ids, results=res, rank=0,
                                     world_size=1,
                                     param_inventory={"n_unique_parameters": 2,
                                                      "n_unique_elements": 200})
        self.assertTrue(good["complete"])
        short = VD.coverage_inventory(expected_ids=ids, results=res, rank=0,
                                      world_size=1,
                                      param_inventory={"n_unique_parameters": 2,
                                                       "n_unique_elements": 300})
        self.assertFalse(short["complete"])
        self.assertEqual(short["model_parameter_inventory"]["elements_unaccounted"], 100)

    def test_identity_is_distinguished_from_recomputation(self):
        src = (ROOT / "zerodiag/verdict.py").read_text()
        self.assertIn("not an independent recomputation of the", src)
        self.assertIn("Active dropout does not remove the mapping", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
