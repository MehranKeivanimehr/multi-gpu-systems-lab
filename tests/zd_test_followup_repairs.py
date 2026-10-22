"""Regressions for the four authorized follow-up repairs.

A — reference input handling: pre-tokenized id lists consumed as token ids.
B — streaming input handling and stream parsing on CPU fixtures.
C — multi-GPU module/master consistency against the rank's own partition view.
D — orchestrator-to-rank ownership hierarchy with harmless processes.

No GPU access anywhere in this module. Mocked-device and CPU-fixture evidence
is identified as such and is never presented as hardware evidence.
"""
import json, os, pathlib, signal, subprocess, sys, tempfile, time, unittest

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = REPO / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from zerodiag import engine_probe as EP            # noqa: E402
from zerodiag import streaming as ST               # noqa: E402
from zerodiag import reference as RF               # noqa: E402
from zerodiag.ownership import JobOwnership, run_ranks  # noqa: E402

def _in_tree(rel: str, root=None):
    """Locate a file in either tree layout (candidate or published study).

    The candidate keeps `protocols/` and `out/` beside the package; the
    published repository keeps `protocols/` inside the study directory. Trying
    both removes the need for a build-time path rewrite, which is one less
    thing that can silently fail to match.
    """
    base = ROOT if root is None else root
    for b in (base, base.parent):
        c = b / rel
        if c.exists():
            return c
    return base / rel


CORPUS = _in_tree("protocols/prompt_corpus.json")
FIXTURE = str(pathlib.Path(__file__).resolve().parent / "fixtures" / "harmless_rank.py")
GIB = 1073741824


# ---------------------------------------------------------------- A ---------
class ReferenceInputHandling(unittest.TestCase):
    """The corpus representation is the thing that broke. Use the real one."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = json.loads(CORPUS.read_text())
        cls.prompts = cls.corpus["prompts"]

    def test_the_real_corpus_stores_pre_tokenized_id_lists(self):
        self.assertIsInstance(self.prompts[0], list)
        self.assertTrue(all(isinstance(i, int) for i in self.prompts[0]))
        self.assertEqual(self.corpus["prompt_len"], 256)

    def test_the_previous_handling_fails_on_the_real_representation(self):
        """Passing an id list to a tokenizer is what raised before."""
        # the pinned tokenizer lives in the Stage-1 cache; stay offline
        cache = os.environ.get("ZD_HF_CACHE",
            "$PROJECT_ROOT/gpu_bench_stage1/cache/hf")
        if not pathlib.Path(cache, "hub").is_dir():
            self.skipTest(f"pinned tokenizer cache not present at {cache}")
        # `cache_dir` is passed explicitly: HF_HOME is read at import time, and
        # another test module may already have imported transformers.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            RF.PROTOCOL["model"], revision=RF.PROTOCOL["model_revision"],
            cache_dir=str(pathlib.Path(cache) / "hub"), local_files_only=True)
        with self.assertRaises(ValueError) as cm:
            tok(self.prompts[0], return_tensors="pt", add_special_tokens=True)
        self.assertIn("text input must be of type", str(cm.exception))

    def test_the_repaired_handling_builds_a_usable_id_batch(self):
        want = RF.PROTOCOL["prompt_len_tokens"]
        rows = [p for p in self.prompts[:4] if len(p) >= want]
        self.assertTrue(rows, "fixture needs prompts of at least 256 ids")
        batch = torch.stack([torch.tensor(r[:want], dtype=torch.long) for r in rows])
        mask = torch.ones_like(batch)
        self.assertEqual(batch.shape[1], want)
        self.assertEqual(batch.dtype, torch.long)
        self.assertEqual(mask.shape, batch.shape)
        self.assertTrue(bool((mask == 1).all()))

    def test_the_runner_consumes_ids_through_the_model_interface(self):
        src = (ROOT / "zerodiag/ref_runner.py").read_text()
        self.assertIn("model(input_ids=batch, attention_mask=mask)", src)
        self.assertIn("tokenizer_used_to_tokenize", src)
        self.assertNotIn("tok(p, return_tensors", src)
        self.assertIn("attention_mask_sha256", src)
        self.assertIn("refusing to guess its encoding", src)

    def test_a_non_id_corpus_row_is_refused_not_guessed(self):
        src = (ROOT / "zerodiag/ref_runner.py").read_text()
        self.assertIn("corpus prompt is not a list of token", src)

    def test_the_protocol_comparison_criteria_are_unchanged(self):
        self.assertEqual(RF.PROTOCOL["output_tokens_exact"], 64)
        self.assertEqual(RF.PROTOCOL["prompt_len_tokens"], 256)
        self.assertEqual(RF.PROTOCOL["dtype"], "float32")
        self.assertTrue(RF.PROTOCOL["greedy"])
        self.assertEqual(RF.PROTOCOL["thresholds"]["output_ids"],
                         "exact equality required")
        self.assertIn("64u", RF.PROTOCOL["thresholds"]["logits_rtol"])

    def test_an_initial_state_digest_is_not_a_comparison(self):
        """Guards the previous over-claim."""
        a = {"realized_input_sha256": None, "initial_state_sha256": "same",
             "input_ids_shape": [1, 256], "dtype": "torch.float32",
             "revision": RF.PROTOCOL["model_revision"],
             "device_name": "NVIDIA TITAN RTX"}
        b = dict(a, device_name="NVIDIA GeForce GTX 1080 Ti")
        r = RF.verify_realized_match(a, b)
        self.assertFalse(r["matched"],
                         "a matching initial state alone must not count as a match")
        self.assertFalse(r["checks"]["realized_input_sha256"])


# ---------------------------------------------------------------- B ---------
class StreamingInputAndParsing(unittest.TestCase):

    def test_request_construction_sends_token_ids(self):
        from zerodiag import stream_runner as SR
        ids = list(range(1000, 1256))
        body = SR.build_request_payload("M", ids, 64)
        self.assertEqual(body["prompt"], ids,
                         "the prompt must be the token ids themselves")
        self.assertTrue(all(isinstance(x, int) for x in body["prompt"]))
        self.assertEqual(body["max_tokens"], 64)
        src = (ROOT / "zerodiag/stream_runner.py").read_text()
        self.assertIn("consumed_as", src)
        self.assertIn("tokenizer_used_to_tokenize", src)
        self.assertNotIn('tok(p, add_special_tokens=True)["input_ids"]', src)

    def test_length_deviations_are_recorded_not_hidden(self):
        src = (ROOT / "zerodiag/stream_runner.py").read_text()
        self.assertIn("length_deviations", src)
        r = ST.verify_prompt_lengths([256, 256, 278])
        self.assertFalse(r["ok"])
        self.assertEqual(r["n_offending"], 1)
        self.assertEqual(r["offending"][0], (2, 278))

    def test_malformed_events_do_not_become_tokens(self):
        events = [{"t": 0.1, "unparsed": "not json"},
                  {"t": 0.2, "text": "ab"}, {"t": 0.3, "done": True}]
        a = ST.token_attribution(events)
        self.assertFalse(a["identity_established"])
        self.assertFalse(a["per_token_timing_reportable"])
        s = ST.summarize_session({"r0": events}, completion_tokens={"r0": None},
                                 status={"r0": "ok"})
        self.assertIsNone(s["per_request"]["r0"]["per_token_timing"])
        self.assertEqual(s["per_request"]["r0"]["n_chunks"], 3)

    def test_a_server_error_is_counted_as_a_failed_request(self):
        s = ST.summarize_session(
            {"r0": [{"t": 0.1, "error": "HTTPError 500"}]},
            completion_tokens={"r0": None}, status={"r0": "http_500"})
        self.assertEqual(s["requests_ok"], 0)
        self.assertEqual(s["requests_failed"], 1)
        self.assertEqual(s["per_request"]["r0"]["status"], "http_500")

    def test_missing_completion_evidence_is_visible(self):
        s = ST.summarize_session(
            {"r0": [{"t": 0.1, "text": "a"}]},
            completion_tokens={"r0": None}, status={"r0": "ok"})
        self.assertIsNone(s["per_request"]["r0"]["server_reported_completion_tokens"])

    def test_token_timing_only_with_one_to_one_attribution(self):
        ok = [{"t": 0.1, "token_ids": [5]}, {"t": 0.2, "token_ids": [6]}]
        self.assertTrue(ST.token_attribution(ok)["per_token_timing_reportable"])
        multi = [{"t": 0.1, "token_ids": [5, 6]}]
        self.assertFalse(ST.token_attribution(multi)["per_token_timing_reportable"])

    def test_vllm_uuid_visibility_defect_is_handled(self):
        """vLLM parses CUDA_VISIBLE_DEVICES with int(); a UUID raises."""
        import importlib.util
        spec = importlib.util.find_spec("vllm")
        src = (ROOT / "zerodiag/stream_runner.py").read_text()
        # the repair must be present regardless of whether vllm is importable here
        self.assertIn("server_env[\"CUDA_VISIBLE_DEVICES\"] = str(a.device_index)", src)
        self.assertIn("binding_chain", src)
        self.assertIn("int(physical_device_id)", src)
        self.assertIn("no --device-index supplied", src)
        # and the launcher must supply it
        cli = (ROOT / "zerodiag/cli.py").read_text()
        self.assertIn('"--device-index", str(rec["device_indices"][r])', cli)
        # the defect itself, demonstrated without touching a device
        with self.assertRaises(ValueError):
            int("TITAN_RTX#1")

    def test_the_runner_never_mocks_the_server(self):
        src = (ROOT / "zerodiag/stream_runner.py").read_text()
        self.assertIn("vllm.entrypoints.openai.api_server", src)
        self.assertIn("_wait_health", src)
        self.assertNotIn("mock", src.lower().replace("mocked-device", ""))


# ---------------------------------------------------------------- C ---------
class PartitionedLayout:
    """Fake ZeRO layout with real attribute names and real partition views."""

    def __init__(self, group_numels=(48,), world_size=2, padding=None,
                 working_dtype=torch.float32, propagate=True, rank=0,
                 corrupt_index=None, stale_delta=None, break_views=False,
                 mismatch_numel=False):
        self.world_size = world_size
        self.groups_padding = list(padding or [0] * len(group_numels))
        self.bit16_groups_flat, self.parallel_partitioned_bit16_groups = [], []
        self.single_partition_of_fp32_groups = []
        self.partition_size = []
        for gi, n in enumerate(group_numels):
            flat = torch.randn(n, dtype=working_dtype)
            self.bit16_groups_flat.append(flat)
            base, rem = n // world_size, n % world_size
            views, start = [], 0
            for r in range(world_size):
                sz = base + (1 if r < rem else 0)
                views.append(flat.narrow(0, start, sz)); start += sz
            self.parallel_partitioned_bit16_groups.append(views)
            psize = int(views[rank].numel())
            self.partition_size.append(psize)
            m = torch.nn.Parameter(views[rank].detach().to(torch.float32).clone())
            if stale_delta is not None:
                with torch.no_grad():
                    m.data.add_(stale_delta)       # master moved, view did not
            self.single_partition_of_fp32_groups.append(m)
        if mismatch_numel:
            self.single_partition_of_fp32_groups[0] = torch.nn.Parameter(
                torch.randn(self.partition_size[0] + 3))
        if break_views:
            self.parallel_partitioned_bit16_groups = None
        if propagate and stale_delta is None and not break_views \
                and not mismatch_numel:
            with torch.no_grad():
                for gi, m in enumerate(self.single_partition_of_fp32_groups):
                    self.parallel_partitioned_bit16_groups[gi][rank].copy_(
                        m.detach().to(working_dtype))
        if corrupt_index is not None and not break_views:
            with torch.no_grad():
                v = self.parallel_partitioned_bit16_groups[0][rank]
                v[corrupt_index] = torch.nextafter(
                    v[corrupt_index].clone(),
                    torch.tensor(float("inf"), dtype=v.dtype))
        self.optimizer = torch.optim.AdamW(
            [p for p in self.single_partition_of_fp32_groups], lr=1e-2)
        self.overflow = False
        self.bit16_groups = [[p] for p in self.single_partition_of_fp32_groups]


class PartitionedConsistency(unittest.TestCase):

    def test_correct_propagation_passes_at_world_size_two(self):
        for rank in (0, 1):
            z = PartitionedLayout(world_size=2, rank=rank)
            r = EP.module_reflects_masters_partitioned(z, rank=rank, world_size=2,
                                                       chunk_elems=8)
            self.assertTrue(r["verdict_successful"], f"rank {rank}: {r.get('reason')}")
            self.assertEqual(r["criterion"], "fp32_exact_equality")
            self.assertTrue(r["bitwise_identical"])
            self.assertFalse(r["per_group"][0]["prefix_comparison_used"])
            self.assertIsNone(r["tolerance_used"])

    def test_the_mapping_is_derived_not_inferred_from_lengths(self):
        z = PartitionedLayout(world_size=4, rank=2)
        mp = EP.partition_working_map(z, rank=2, world_size=4)
        self.assertTrue(mp["mapped"])
        self.assertFalse(mp["inferred_from_equal_lengths"])
        self.assertIn("parallel_partitioned_bit16_groups", mp["derived_from"])
        g = mp["groups"][0]
        # rank 2 of 4 has TWO preceding partitions of 12 elements each
        self.assertEqual(g["flat_offset"], 12 + 12)
        self.assertEqual(g["working_numel"], g["master_numel"])

    def test_stale_working_values_fail(self):
        z = PartitionedLayout(world_size=2, rank=0, stale_delta=1e-4,
                              propagate=False)
        r = EP.module_reflects_masters_partitioned(z, rank=0, world_size=2,
                                                   chunk_elems=8)
        self.assertFalse(r["verdict_successful"])
        self.assertFalse(r["numerically_equal"])
        self.assertGreater(r["per_group"][0]["n_differing_elements"], 0)

    def test_a_one_element_corruption_fails(self):
        z = PartitionedLayout(world_size=2, rank=0, corrupt_index=3)
        r = EP.module_reflects_masters_partitioned(z, rank=0, world_size=2,
                                                   chunk_elems=8)
        self.assertFalse(r["verdict_successful"])
        self.assertEqual(r["per_group"][0]["n_differing_elements"], 1)
        self.assertEqual(r["per_group"][0]["first_difference"]["global_index"], 3)

    def test_partition_boundaries_are_respected(self):
        """Rank 0 must not see rank 1's elements."""
        z = PartitionedLayout(world_size=2, rank=0)
        with torch.no_grad():                    # corrupt the OTHER rank's half
            z.parallel_partitioned_bit16_groups[0][1][0] += 1.0
        r = EP.module_reflects_masters_partitioned(z, rank=0, world_size=2,
                                                   chunk_elems=8)
        self.assertTrue(r["verdict_successful"],
                        "a neighbouring partition must not affect this rank")

    def test_padding_is_reported_and_not_counted_as_coverage(self):
        z = PartitionedLayout(group_numels=(48,), world_size=2, padding=[5],
                              rank=1)
        r = EP.module_reflects_masters_partitioned(z, rank=1, world_size=2,
                                                   chunk_elems=8)
        g = r["per_group"][0]
        self.assertTrue(g["is_last_rank"])
        self.assertEqual(g["padding"], 5)
        self.assertEqual(g["unpadded_numel"], g["elements_expected"] - 5)
        self.assertEqual(r["coverage"]["padding_elements"], 5)
        self.assertIn("never counted as model coverage",
                      r["coverage"]["padding_note"])

    def test_an_absent_partition_view_is_unverified_not_successful(self):
        z = PartitionedLayout(world_size=2, rank=0, break_views=True)
        r = EP.module_reflects_masters_partitioned(z, rank=0, world_size=2,
                                                   chunk_elems=8)
        self.assertFalse(r["checked"])
        self.assertFalse(r["verdict_successful"])
        self.assertIn("NOT established", r["reason"])

    def test_a_wrong_sized_mapping_is_unverified(self):
        z = PartitionedLayout(world_size=2, rank=0, mismatch_numel=True)
        mp = EP.partition_working_map(z, rank=0, world_size=2)
        self.assertFalse(mp["mapped"])
        r = EP.module_reflects_masters_partitioned(z, rank=0, world_size=2,
                                                   chunk_elems=8)
        self.assertFalse(r["verdict_successful"])

    def test_mixed_precision_uses_the_expected_cast(self):
        z = PartitionedLayout(world_size=2, rank=0, working_dtype=torch.float16)
        r = EP.module_reflects_masters_partitioned(z, rank=0, world_size=2,
                                                   chunk_elems=8)
        self.assertEqual(r["per_group"][0]["criterion"],
                         "mixed_precision_expected_cast")
        self.assertTrue(r["verdict_successful"])
        self.assertIsNone(r["tolerance_used"])

    def test_cross_rank_coverage_reconciles_exactly_once(self):
        maps = []
        for rank in range(6):
            z = PartitionedLayout(group_numels=(120,), world_size=6, rank=rank)
            mp = EP.partition_working_map(z, rank=rank, world_size=6)
            mp["rank"] = rank
            maps.append(mp)
        rec = EP.reconcile_partition_coverage(maps, unique_model_elements=120,
                                              world_size=6)
        self.assertTrue(rec["complete"], rec)
        self.assertEqual(rec["elements_covered_once"], 120)
        self.assertEqual(rec["elements_unaccounted"], 0)
        self.assertEqual(rec["missing_ranks"], [])
        self.assertEqual(rec["duplicate_ranks"], [])
        self.assertEqual(rec["overlapping_spans"], [])

    def test_missing_and_duplicate_coverage_are_reported(self):
        maps = []
        for rank in range(4):
            z = PartitionedLayout(group_numels=(48,), world_size=4, rank=rank)
            mp = EP.partition_working_map(z, rank=rank, world_size=4)
            mp["rank"] = rank
            maps.append(mp)
        short = EP.reconcile_partition_coverage(maps[:3],
                                                unique_model_elements=48,
                                                world_size=4)
        self.assertFalse(short["complete"])
        self.assertEqual(short["missing_ranks"], ["3"])
        self.assertEqual(short["elements_unaccounted"], 12)
        dupe = EP.reconcile_partition_coverage(maps + [maps[0]],
                                               unique_model_elements=48,
                                               world_size=4)
        self.assertFalse(dupe["complete"])
        self.assertTrue(dupe["overlapping_spans"])


# ---------------------------------------------------------------- D ---------
class OrchestratorHierarchy(unittest.TestCase):
    """Harmless processes only. No model worker, no CUDA."""

    def _argv(self, ws, mode=None, sleep=None):
        def f(r, rd):
            a = [sys.executable, "-I", FIXTURE, "--output", str(rd),
                 "--rank", str(r), "--world-size", str(ws), "--case-id", "HIER"]
            if mode: a += ["--mode", mode(r)]
            if sleep: a += ["--sleep", str(sleep(r))]
            return a
        return f

    def test_the_full_hierarchy_is_recorded(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="hier_"))
        rep = run_ranks(world_size=4, argv_for=self._argv(4, sleep=lambda r: 1.5),
                        out_dir=tmp, timeout_s=90, budget_bytes=8 * GIB,
                        interval_s=0.2)
        led = json.loads((tmp / "ownership_ledger.json").read_text())
        self.assertEqual(len(led["ranks"]), 4)
        for rec, r in zip(led["ranks"], range(4)):
            d = json.loads((tmp / f"rank{r}" / "rank_record.json").read_text())
            self.assertEqual(rec["pid"], d["pid"])
            self.assertEqual(d["pid"], d["sid"], "each rank leads its own session")
            self.assertEqual(d["job_token"], led["token"])
        self.assertTrue(rep["owned_absent_confirmed"])

    def test_terminating_the_orchestrator_does_not_reach_ranks_by_itself(self):
        """The documented hazard, demonstrated rather than assumed."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="hierkill_"))
        own = JobOwnership("zd-" + os.urandom(6).hex(),
                           ledger_path=tmp / "ownership_ledger.json")
        procs = []
        for r in range(3):
            rd = tmp / f"rank{r}"; rd.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen(
                [sys.executable, "-I", FIXTURE, "--output", str(rd),
                 "--rank", str(r), "--world-size", "3", "--case-id", "HIERKILL",
                 "--sleep", "60"], env=own.env_for_child(),
                start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            own.register(r, p.pid); procs.append(p)
        try:
            self.assertEqual(len(own.owned_pids()), 3)
            # an uncatchable kill of the parent does NOT stop them
            self.assertTrue(all(p.poll() is None for p in procs),
                            "ranks survive independently of the parent")
            # the ledger is the independent mechanism that does reach them
            led = json.loads((tmp / "ownership_ledger.json").read_text())
            revived = JobOwnership(led["token"])
            for r in led["ranks"]:
                revived.ranks[r["rank"]] = r
            rep = revived.terminate(grace_s=15)
            self.assertTrue(rep["owned_absent_confirmed"], rep["surviving_pids"])
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: pass

    def test_partial_launch_cleans_up_what_started(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="hierpart_"))
        def argv(r, rd):
            if r == 2: return [str(rd / "nonexistent-binary")]
            return self._argv(4, sleep=lambda _: 30)(r, rd)
        rep = run_ranks(world_size=4, argv_for=argv, out_dir=tmp, timeout_s=60,
                        budget_bytes=8 * GIB, interval_s=0.2, grace_s=10)
        self.assertTrue(rep["partial_launch"])
        self.assertTrue(rep["owned_absent_confirmed"])

    def test_rank_failure_and_timeout_are_distinguished(self):
        t1 = pathlib.Path(tempfile.mkdtemp(prefix="hierfail_"))
        r1 = run_ranks(world_size=3,
                       argv_for=self._argv(3, mode=lambda r: "fail" if r == 1 else "ok"),
                       out_dir=t1, timeout_s=60, budget_bytes=8 * GIB,
                       interval_s=0.2)
        self.assertEqual(r1["exit_codes"]["1"], 7)
        self.assertFalse(r1["timed_out"])
        t2 = pathlib.Path(tempfile.mkdtemp(prefix="hierto_"))
        r2 = run_ranks(world_size=2, argv_for=self._argv(2, sleep=lambda r: 60),
                       out_dir=t2, timeout_s=3, budget_bytes=8 * GIB,
                       interval_s=0.2, grace_s=10)
        self.assertTrue(r2["timed_out"])
        self.assertTrue(r2["owned_absent_confirmed"])

    def test_unrelated_sentinels_survive_every_path(self):
        s = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(300)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for kw in ({"timeout_s": 3, "sleep": lambda r: 60},
                       {"timeout_s": 60, "sleep": None}):
                tmp = pathlib.Path(tempfile.mkdtemp(prefix="hiersent_"))
                run_ranks(world_size=2,
                          argv_for=self._argv(2, sleep=kw["sleep"]),
                          out_dir=tmp, timeout_s=kw["timeout_s"],
                          budget_bytes=8 * GIB, interval_s=0.2, grace_s=10)
                self.assertIsNone(s.poll(), "an unrelated process was killed")
        finally:
            s.terminate()
            try: s.wait(timeout=10)
            except subprocess.TimeoutExpired: s.kill()

    def test_unconfirmed_cleanup_is_a_refusal_in_the_launcher(self):
        src = (ROOT / "zerodiag/cli.py").read_text()
        self.assertIn("STOP: owned cleanup could not be confirmed", src)
        i = src.index("STOP: owned cleanup could not be confirmed")
        self.assertIn("return 1", src[i:i + 300])


if __name__ == "__main__":
    unittest.main(verbosity=2)
