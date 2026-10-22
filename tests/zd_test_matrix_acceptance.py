"""Engineering acceptance for the 18-case matrix and its multi-rank launcher.

The multi-process lifecycle tests drive the REAL production path: the real
parser, the real worker-binding check, real `execute_case`, real multi-process
spawn into one shared process group, real aggregate monitoring, real per-rank
evidence collection and real rank-set reconciliation.

Documented fixture, the only thing mocked: the receipt names device UUIDs of
the form GPU-00000000-... which exist on no host, so every rank reaches its own
no-CUDA guard and refuses with code 90. Nothing else is faked.

CPU fixtures here are NOT CUDA, NCCL or AMP hardware validation.
"""
import json, os, pathlib, subprocess, sys, tempfile, time, unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from zerodiag import cli as CLI                    # noqa: E402
from zerodiag import matrix as MX                  # noqa: E402
from zerodiag import factories as F                # noqa: E402
from zerodiag import reference as RF               # noqa: E402
from zerodiag import streaming as ST               # noqa: E402
import _zd_isolation as ISO                           # noqa: E402

DS_CFG = {"train_micro_batch_size_per_gpu": 4, "gradient_accumulation_steps": 1,
          "gradient_clipping": 1.0, "zero_allow_untested_optimizer": True,
          "zero_optimization": {"stage": 1}, "fp16": {"enabled": False},
          "bf16": {"enabled": False}}


def fake_uuid(i):
    return f"GPU-{i:08d}-0000-0000-0000-000000000000"


class MatrixExpansion(unittest.TestCase):

    def test_all_eighteen_cases_expand(self):
        m = MX.full_matrix()
        self.assertEqual(m["counts"],
                         {"training": 15, "reference": 2, "streaming": 1,
                          "remaining_total": 18, "completed": 1,
                          "all_contexts": 19})

    def test_the_fifteen_training_contexts_are_exactly_the_agreed_set(self):
        want = set()
        for ws in (2, 4, 6):
            want.add(("GTX 1080 Ti", 1, ws, "fp32"))
        for ws in (1, 2, 4, 6):
            want.add(("GTX 1080 Ti", 2, ws, "fp32"))
        for stage in (1, 2):
            for ws in (1, 2):
                want.add(("TITAN RTX", stage, ws, "fp32"))
                want.add(("TITAN RTX", stage, ws, "amp_fp16"))
        got = {(c["family"], c["stage"], c["world_size"], c["precision"])
               for c in MX.training_cases()}
        self.assertEqual(got, want)
        self.assertEqual(len(got), 15)

    def test_gate0_counts_once_and_is_not_re_executed(self):
        m = MX.full_matrix()
        g = m["completed_elsewhere"][0]
        self.assertEqual(g["case_id"], "G0_1080ti_zero1_w1_fp32")
        self.assertEqual(g["disposition"], "completed")
        self.assertIn("f45b26062044bb08a560223524f23ca2d8a9be05ab926d25fb870b36733508b1",
                      g["evidence_sha256"])
        ids = {c["case_id"] for c in m["remaining_cases"]}
        self.assertNotIn("G0_1080ti_zero1_w1_fp32", ids)

    def test_every_group_is_homogeneous_and_xp_is_never_used(self):
        for c in MX.full_matrix()["remaining_cases"]:
            self.assertEqual(len(set(c["device_names"])), 1, c["case_id"])
            for n in c["device_names"]:
                self.assertNotIn("Xp", n, f"{c['case_id']} uses a TITAN Xp")
            self.assertEqual(len(c["device_uuids"]), c["world_size"])
            self.assertEqual(len(set(c["device_uuids"])), c["world_size"])

    def test_device_identities_come_from_the_frozen_plan(self):
        g = MX.device_groups()
        self.assertIn("source_sha256", g)
        self.assertIn("run_plan.json", g["source"])
        self.assertEqual(g["groups"]["p1080_6"]["device_indices"], [1, 3, 4, 5, 8, 9])
        self.assertEqual(g["groups"]["rtx_2"]["device_indices"], [0, 2])

    def test_expected_base_calls_are_derived_not_fixed_at_one(self):
        for c in MX.training_cases():
            self.assertIn("derived from", str(c["expected_base_calls"]))
        src = (ROOT / "zerodiag/worker.py").read_text()
        self.assertIn("len(EP._req(zopt, \"single_partition_of_fp32_groups\"))", src)

    def test_the_matrix_prints_through_the_real_cli(self):
        r = subprocess.run([sys.executable, "-B", "-m", "zerodiag.cli", "--matrix"],
                           cwd=ROOT, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        self.assertEqual(json.loads(r.stdout)["counts"]["remaining_total"], 18)


class CaseReceiptRefusals(unittest.TestCase):
    """Rank/device mismatches and source/config substitutions must refuse."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="mx_"))
        self.case = {c["case_id"]: c for c in
                     MX.full_matrix()["remaining_cases"]}["D_p1080_2_zero1_fp32"]
        self.sb = CLI.source_tree_digest()

    def _receipt(self, **over):
        now = time.time()
        rec = {"authorization_id": "TEST-MX", "approved_by": "fixture",
               "approval_text": "cpu fixture; no device is authorized",
               "start_epoch": now - 5, "expiry_epoch": now + 600,
               "case_id": self.case["case_id"],
               "world_size": self.case["world_size"],
               "device_indices": list(self.case["device_indices"]),
               "device_uuids": list(self.case["device_uuids"]),
               "device_names": list(self.case["device_names"]),
               "source_digest": self.sb["__tree__"]}
        rec.update(over)
        rec["digest"] = CLI._receipt_digest(rec)
        if over.get("_break_digest"):
            rec["digest"] = "0" * 64
        p = self.tmp / f"r{abs(hash(json.dumps(rec, sort_keys=True)))}.json"
        p.write_text(json.dumps(rec))
        return str(p)

    def _v(self, **over):
        return CLI.validate_case_receipt(self._receipt(**over), case=self.case,
                                         source_bindings_now=self.sb)

    def test_a_correct_receipt_validates(self):
        self.assertTrue(self._v()["ok"])

    def test_wrong_case_id_refuses(self):
        r = self._v(case_id="D_rtx_1_zero1_fp32")
        self.assertFalse(r["ok"]); self.assertIn("not the requested case", r["reason"])

    def test_world_size_mismatch_refuses(self):
        r = self._v(world_size=4)
        self.assertFalse(r["ok"]); self.assertIn("world_size", r["reason"])

    def test_substituted_device_uuid_refuses(self):
        r = self._v(device_uuids=[fake_uuid(1), fake_uuid(2)])
        self.assertFalse(r["ok"]); self.assertIn("device_uuids", r["reason"])

    def test_substituted_device_index_refuses(self):
        r = self._v(device_indices=[1, 7])
        self.assertFalse(r["ok"]); self.assertIn("device_indices", r["reason"])

    def test_heterogeneous_group_refuses(self):
        r = self._v(device_names=["NVIDIA GeForce GTX 1080 Ti", "NVIDIA TITAN Xp"])
        self.assertFalse(r["ok"])
        self.assertIn("not homogeneous", r["reason"])

    def test_duplicate_device_for_two_ranks_refuses(self):
        u = self.case["device_uuids"][0]
        case2 = dict(self.case, device_uuids=[u, u])
        rec = self._receipt(device_uuids=[u, u])
        r = CLI.validate_case_receipt(rec, case=case2, source_bindings_now=self.sb)
        self.assertFalse(r["ok"])
        self.assertIn("more than one rank", r["reason"])

    def test_source_substitution_refuses(self):
        r = self._v(source_digest="0" * 64)
        self.assertFalse(r["ok"]); self.assertIn("different source tree", r["reason"])

    def test_forged_digest_refuses(self):
        r = self._v(_break_digest=True)
        self.assertFalse(r["ok"]); self.assertIn("digest", r["reason"])

    def test_expired_window_refuses(self):
        now = time.time()
        r = self._v(start_epoch=now - 900, expiry_epoch=now - 10)
        self.assertFalse(r["ok"]); self.assertIn("not currently valid", r["reason"])


@unittest.skipUnless(ISO.available(), ISO.skip_reason())
class MultiProcessLifecycle(unittest.TestCase):
    """Real execute_case and real multi-rank spawn against SYNTHETIC devices.

    SAFETY CONTRACT, and the reason this is written the way it is: an earlier
    version of this test built a receipt with fake UUIDs and then overwrote them
    with the frozen case's REAL UUIDs, because `validate_case_receipt` requires
    the receipt to match the case. The launcher therefore set
    CUDA_VISIBLE_DEVICES to a real GPU and the worker initialised CUDA and began
    building the model - an unapproved CUDA workload on a shared device. See
    `out/evidence/INCIDENT_001_unapproved_cuda.md`.

    The fix is to replace the frozen-groups INPUT, not to add a bypass to
    production code: `zerodiag.matrix._GROUPS_PATH` is pointed at a synthetic
    groups file whose UUIDs exist on no host. The case and the receipt then
    agree on devices that cannot be opened, so every rank reaches its own
    no-CUDA guard. No real device is ever named anywhere in this test.
    """

    SYNTH = {
        "source": "SYNTHETIC test fixture, not a real device inventory",
        "source_sha256": "0" * 64,
        "groups": {
            "p1080_1": {"device_indices": [901], "device_uuids": [fake_uuid(901)],
                        "device_names": ["SYNTHETIC GTX 1080 Ti"],
                        "family": "GTX 1080 Ti", "topology_worst_link": "N/A"},
            "p1080_2": {"device_indices": [901, 902],
                        "device_uuids": [fake_uuid(901), fake_uuid(902)],
                        "device_names": ["SYNTHETIC GTX 1080 Ti"] * 2,
                        "family": "GTX 1080 Ti", "topology_worst_link": "PIX"},
            "p1080_4": {"device_indices": [901, 902, 903, 904],
                        "device_uuids": [fake_uuid(i) for i in (901, 902, 903, 904)],
                        "device_names": ["SYNTHETIC GTX 1080 Ti"] * 4,
                        "family": "GTX 1080 Ti", "topology_worst_link": "PIX"},
            "p1080_6": {"device_indices": [901, 902, 903, 904, 905, 906],
                        "device_uuids": [fake_uuid(i) for i in range(901, 907)],
                        "device_names": ["SYNTHETIC GTX 1080 Ti"] * 6,
                        "family": "GTX 1080 Ti", "topology_worst_link": "PHB"},
            "rtx_1": {"device_indices": [911], "device_uuids": [fake_uuid(911)],
                      "device_names": ["SYNTHETIC TITAN RTX"],
                      "family": "TITAN RTX", "topology_worst_link": "N/A"},
            "rtx_2": {"device_indices": [911, 912],
                      "device_uuids": [fake_uuid(911), fake_uuid(912)],
                      "device_names": ["SYNTHETIC TITAN RTX"] * 2,
                      "family": "TITAN RTX", "topology_worst_link": "PHB"},
        },
    }

    @classmethod
    def setUpClass(cls):
        cls.synth_dir = pathlib.Path(tempfile.mkdtemp(prefix="mxsynth_"))
        cls.synth = cls.synth_dir / "device_groups_frozen.json"
        cls.synth.write_text(json.dumps(cls.SYNTH))

    def _run(self, case_id, *, reuse_out=None):
        """Drive the real CLI with the synthetic groups file substituted."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="mxlife_"))
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["ZERODIAG_GROUPS_PATH"] = str(self.synth)
        for v in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR",
                  "MASTER_PORT", "TITAN_CPU_ONLY_GUARD", "CUDA_VISIBLE_DEVICES"):
            env.pop(v, None)
        # resolve the case through the SAME override the child will use
        probe = subprocess.run(
            [sys.executable, "-B", "-c",
             "import json,sys;from zerodiag import matrix as MX;"
             "print(json.dumps({c['case_id']:c for c in "
             "MX.full_matrix()['remaining_cases']}[sys.argv[1]]))", case_id],
            cwd=ROOT, capture_output=True, text=True, timeout=180, env=env)
        self.assertEqual(probe.returncode, 0, probe.stderr[-400:])
        case = json.loads(probe.stdout)
        for u in case["device_uuids"]:
            self.assertTrue(u.startswith("GPU-0000"),
                            f"SAFETY: case names a non-synthetic device {u!r}")
        sb_probe = subprocess.run(
            [sys.executable, "-B", "-c",
             "import json;from zerodiag.cli import source_tree_digest as D;"
             "print(json.dumps(D()))"],
            cwd=ROOT, capture_output=True, text=True, timeout=180, env=env)
        sb = json.loads(sb_probe.stdout)
        now = time.time()
        rec = {"authorization_id": f"TEST-LIFE-{case_id}",
               "approved_by": "cpu fixture",
               "approval_text": ("cpu fixture against SYNTHETIC devices; no real "
                                 "device is named and no CUDA workload can run"),
               "start_epoch": now - 5, "expiry_epoch": now + 600,
               "case_id": case_id, "world_size": case["world_size"],
               "device_indices": list(case["device_indices"]),
               "device_uuids": list(case["device_uuids"]),
               "device_names": list(case["device_names"]),
               "source_digest": sb["__tree__"]}
        rec["digest"] = CLI._receipt_digest(rec)
        rp = tmp / "receipt.json"; rp.write_text(json.dumps(rec))
        cfg = tmp / "ds.json"; cfg.write_text(json.dumps(DS_CFG))
        out = pathlib.Path(reuse_out) if reuse_out else (tmp / "run")
        p = subprocess.run(
            ISO.wrap([sys.executable, "-B", "-m", "zerodiag.cli",
                      "--execute-case", "--case-id", case_id,
                      "--output", str(out), "--receipt", str(rp),
                      "--ds-config", str(cfg), "--max-cpus", "4",
                      "--memory-budget-gib", "8.0",
                      "--monitor-interval-s", "0.5"],
                     extra_env={"ZERODIAG_GROUPS_PATH": str(self.synth)},
                     writable=[tmp, self.synth_dir]),
            cwd=ROOT, capture_output=True, text=True, timeout=900, env=env)
        s = out / "case_summary.json"
        return p, (json.loads(s.read_text()) if s.is_file() else None), out, case, rp, cfg

    def test_single_rank_case_lifecycle(self):
        p, summ, out, case, _, _ = self._run("D_p1080_1_zero2_fp32")
        self.assertIsNotNone(summ, p.stdout[-900:] + p.stderr[-400:])
        self.assertEqual(summ["world_size"], 1)
        self.assertEqual(summ["stage"], 2)
        self.assertEqual(summ["ranks_expected"], 1)
        self.assertEqual(summ["ranks_with_evidence"], 1)
        self.assertEqual(summ["ranks_completed"], 0)
        self.assertEqual(summ["exit_codes"], [90])
        self.assertTrue(summ["group_absent_confirmed"])
        self.assertEqual(p.returncode, 1)
        ev = json.loads((out / "rank0" / "integration_evidence.json").read_text())
        self.assertEqual(ev["stage"], "refused")
        self.assertIn("no CUDA device visible to the child", ev["reason"])

    def test_two_rank_case_lifecycle_and_shared_process_group(self):
        p, summ, out, case, _, _ = self._run("D_p1080_2_zero1_fp32")
        self.assertIsNotNone(summ, p.stdout[-900:] + p.stderr[-400:])
        self.assertEqual(summ["world_size"], 2)
        self.assertEqual(summ["ranks_expected"], 2)
        self.assertEqual(summ["ranks_with_evidence"], 2)
        self.assertEqual(summ["exit_codes"], [90, 90])
        ports = {}
        for r in range(2):
            ev = json.loads((out / f"rank{r}" / "integration_evidence.json").read_text())
            self.assertEqual(ev["rank"], r)
            self.assertEqual(ev["world_size"], 2)
            self.assertEqual(ev["case_id"], "D_p1080_2_zero1_fp32")
            de = ev["distributed_env_seen"]
            self.assertEqual(de["RANK"], str(r))
            self.assertEqual(de["WORLD_SIZE"], "2")
            self.assertEqual(de["LOCAL_RANK"], "0")
            ports[r] = de["MASTER_PORT"]
            self.assertEqual(ev["child_visible_devices"], case["device_uuids"][r])
        self.assertEqual(ports[0], ports[1], "ranks must share one rendezvous")
        self.assertTrue(summ["group_absent_confirmed"])
        self.assertEqual(p.returncode, 1)

    def test_four_and_six_rank_cases_launch_every_rank(self):
        for cid, ws in (("D_p1080_4_zero1_fp32", 4), ("D_p1080_6_zero1_fp32", 6)):
            p, summ, out, case, _, _ = self._run(cid)
            self.assertIsNotNone(summ, p.stdout[-600:])
            self.assertEqual(summ["ranks_expected"], ws, cid)
            self.assertEqual(summ["ranks_with_evidence"], ws, cid)
            self.assertEqual(summ["exit_codes"], [90] * ws, cid)
            self.assertTrue(summ["group_absent_confirmed"], cid)

    def test_aggregate_resource_accounting_is_whole_job(self):
        p, summ, out, case, _, _ = self._run("D_p1080_2_zero1_fp32")
        agg = summ["aggregate_memory"]
        self.assertEqual(agg["budget_gib"], 8.0)
        self.assertFalse(agg["enforced_cap"])
        self.assertGreaterEqual(agg["samples"], 1)
        self.assertIn("token sweep", agg["discovery"])
        self.assertIn("VmRSS", agg["method"])
        self.assertIn("over-estimate", agg["shared_pages"])
        self.assertIn("Monitoring, not a cap", agg["limitations"])

    def test_a_missing_rank_is_never_inferred(self):
        p, summ, out, case, _, _ = self._run("D_p1080_2_zero1_fp32")
        self.assertEqual(summ["ranks_expected"], 2)
        self.assertEqual(summ["ranks_completed"], 0)
        self.assertFalse(summ["rank_reconciliation"].get("verification_successful"))
        self.assertIn("no rank produced a verdict",
                      str(summ["rank_reconciliation"].get("reason")))

    def test_write_once_attempt_directory(self):
        p, summ, out, case, rp, cfg = self._run("D_p1080_1_zero2_fp32")
        env = dict(os.environ)
        env["ZERODIAG_GROUPS_PATH"] = str(self.synth)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        again = subprocess.run(
            ISO.wrap([sys.executable, "-B", "-m", "zerodiag.cli",
                      "--execute-case", "--case-id", "D_p1080_1_zero2_fp32",
                      "--output", str(out), "--receipt", str(rp),
                      "--ds-config", str(cfg), "--max-cpus", "4",
                      "--memory-budget-gib", "8.0",
                      "--monitor-interval-s", "0.5"],
                     extra_env={"ZERODIAG_GROUPS_PATH": str(self.synth)},
                     writable=[out.parent, self.synth_dir]),
            cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
        self.assertEqual(again.returncode, 95, again.stdout[-400:])
        self.assertIn("write-once", again.stdout)
        self.assertIn("never overwritten", again.stdout)


class AmpBookkeeping(unittest.TestCase):
    """Clearly labelled CPU fixtures. Not AMP hardware validation."""

    def test_amp_config_comes_from_the_retained_benchmark(self):
        cfg, sem = F.resolve_amp_config("amp_fp16", {"zero_optimization": {}})
        self.assertEqual(cfg["fp16"], {"enabled": True, "loss_scale": 0,
                                       "initial_scale_power": 12})
        self.assertFalse(cfg["bf16"]["enabled"])
        self.assertIn("train_llm.py:144", sem["source"])

    def test_native_gradscaler_is_neither_used_nor_assumed(self):
        _, sem = F.resolve_amp_config("amp_fp16", {})
        self.assertFalse(sem["native_gradscaler_used"])
        self.assertFalse(sem["native_gradscaler_semantics_assumed"])
        self.assertEqual(sem["scaler"], "DeepSpeed dynamic loss scaling")

    def test_fp32_disables_both_half_precisions(self):
        cfg, sem = F.resolve_amp_config("fp32", {})
        self.assertFalse(cfg["fp16"]["enabled"])
        self.assertFalse(cfg["bf16"]["enabled"])
        self.assertIsNone(sem["scaler"])

    def test_skips_are_never_inferred_from_an_absent_hook(self):
        _, sem = F.resolve_amp_config("amp_fp16", {})
        self.assertFalse(sem["inferred_from_absent_hook"])
        self.assertIn("no fault injection", sem["skip_policy"])
        self.assertIn("no manual scale tuning", sem["skip_policy"])

    def test_unsupported_precision_refuses(self):
        with self.assertRaises(ValueError):
            F.resolve_amp_config("bf16", {})


class ReferenceProtocol(unittest.TestCase):

    def test_corpus_verifies_under_its_own_declared_convention(self):
        r = RF.verify_corpus(ROOT / "protocols" / "prompt_corpus.json")
        self.assertTrue(r["ok"], r.get("reason"))
        self.assertEqual(r["self_digest"], RF.PROTOCOL["corpus_sha256"])
        self.assertNotEqual(r["file_bytes_sha256"], r["self_digest"])
        self.assertEqual(r["n_prompts"], 64)
        self.assertEqual(r["prompt_len"], 256)

    def test_seeds_alone_do_not_establish_matching(self):
        a = {"realized_input_sha256": "aa", "initial_state_sha256": "bb",
             "input_ids_shape": [1, 256], "dtype": "torch.float32",
             "revision": RF.PROTOCOL["model_revision"],
             "device_name": "NVIDIA TITAN RTX"}
        b = dict(a, realized_input_sha256="cc")      # same seed, different ids
        r = RF.verify_realized_match(a, b)
        self.assertFalse(r["matched"])
        self.assertFalse(r["checks"]["realized_input_sha256"])
        self.assertIn("seed alone is not accepted", r["basis"])

    def test_same_family_pair_is_not_a_cross_family_control(self):
        a = {"realized_input_sha256": "aa", "initial_state_sha256": "bb",
             "input_ids_shape": [1, 256], "dtype": "torch.float32",
             "revision": RF.PROTOCOL["model_revision"],
             "device_name": "NVIDIA TITAN RTX"}
        r = RF.verify_realized_match(a, dict(a))
        self.assertFalse(r["matched"])
        self.assertFalse(r["cross_family"])

    def test_a_correct_cross_family_pair_matches(self):
        a = {"realized_input_sha256": "aa", "initial_state_sha256": "bb",
             "input_ids_shape": [1, 256], "dtype": "torch.float32",
             "revision": RF.PROTOCOL["model_revision"],
             "device_name": "NVIDIA TITAN RTX"}
        b = dict(a, device_name="NVIDIA GeForce GTX 1080 Ti")
        r = RF.verify_realized_match(a, b)
        self.assertTrue(r["matched"])
        self.assertTrue(r["cross_family"])

    def test_loss_agreement_is_not_gradient_verification(self):
        d = RF.PROTOCOL["does_not_establish"]
        self.assertIn("independent full-gradient verification", d)
        self.assertIn("training quality", d)
        # the per-case record carries the loss-agreement caveat explicitly
        case = MX.reference_cases()[0]
        self.assertIn("loss agreement is not gradient correctness",
                      case["does_not_establish"])
        self.assertIn("INFORMED", RF.PROTOCOL["status"])

    def test_retrospective_amendment_is_refused(self):
        RF.declare_amendment("widen the window", prospective=True)
        with self.assertRaises(ValueError):
            RF.declare_amendment("explain a disagreement", prospective=False)


class StreamingProtocol(unittest.TestCase):

    def test_protocol_is_256_in_64_out_one_session(self):
        self.assertEqual(ST.PROTOCOL["prompt_len_tokens"], 256)
        self.assertEqual(ST.PROTOCOL["output_tokens"], 64)
        self.assertEqual(ST.PROTOCOL["sessions"], 1)
        self.assertIn("not six independent", ST.PROTOCOL["sessions_note"])

    def test_actual_tokenized_lengths_are_checked(self):
        self.assertTrue(ST.verify_prompt_lengths([256] * 64)["ok"])
        bad = ST.verify_prompt_lengths([256, 278, 256])
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["n_offending"], 1)
        self.assertFalse(ST.verify_prompt_lengths([])["ok"])

    def test_pinned_revision_is_required_for_model_and_tokenizer(self):
        rev = ST.PROTOCOL["model_revision"]
        self.assertTrue(ST.verify_revision(rev, rev)["ok"])
        self.assertFalse(ST.verify_revision(rev, "main")["ok"])
        self.assertFalse(ST.verify_revision(None, rev)["ok"])

    def test_text_chunks_do_not_become_token_timestamps(self):
        events = [{"t": 0.10, "text": "He"}, {"t": 0.20, "text": "llo w"},
                  {"t": 0.31, "text": "orld"}]
        a = ST.token_attribution(events)
        self.assertFalse(a["identity_established"])
        self.assertFalse(a["per_token_timing_reportable"])
        self.assertIn("NOT token times", a["reason"])

    def test_per_token_timing_only_with_one_to_one_attribution(self):
        ok = [{"t": 0.1, "token_ids": [15]}, {"t": 0.2, "token_ids": [16]}]
        self.assertTrue(ST.token_attribution(ok)["per_token_timing_reportable"])
        multi = [{"t": 0.1, "token_ids": [15, 16]}, {"t": 0.2, "token_ids": [17]}]
        self.assertFalse(ST.token_attribution(multi)["per_token_timing_reportable"])
        untimed = [{"token_ids": [15]}, {"t": 0.2, "token_ids": [16]}]
        self.assertFalse(ST.token_attribution(untimed)["per_token_timing_reportable"])

    def test_session_summary_names_chunk_metrics_as_chunk_metrics(self):
        s = ST.summarize_session(
            {"r0": [{"t": 0.1, "text": "a"}, {"t": 0.25, "text": "b"}]},
            completion_tokens={"r0": 64}, status={"r0": "ok"})
        self.assertEqual(s["sessions"], 1)
        self.assertEqual(s["requests_ok"], 1)
        d = s["per_request"]["r0"]
        self.assertEqual(d["n_chunks"], 2)
        self.assertEqual(d["server_reported_completion_tokens"], 64)
        self.assertIsNone(d["per_token_timing"])
        self.assertFalse(s["per_token_timing_reported"])
        self.assertIn("CHUNK metrics", s["metric_names"])
        self.assertIn("one session", s["no_tail_percentiles"])

    def test_failed_requests_are_counted_not_dropped(self):
        s = ST.summarize_session(
            {"r0": [{"t": 0.1, "text": "a"}], "r1": []},
            completion_tokens={"r0": 64, "r1": None},
            status={"r0": "ok", "r1": "error"})
        self.assertEqual(s["requests_recorded"], 2)
        self.assertEqual(s["requests_ok"], 1)
        self.assertEqual(s["requests_failed"], 1)

    def test_history_is_not_rewritten(self):
        self.assertIn("278-token", ST.PROTOCOL["historical_untouched"])
        self.assertIn("64/128", ST.PROTOCOL["historical_untouched"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
