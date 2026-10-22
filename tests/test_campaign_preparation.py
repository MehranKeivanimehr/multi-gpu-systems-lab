"""Tests for the campaign-preparation changes (session 20261016).

CPU/mock only. These prove control and metadata behaviour; they execute no CUDA
and start no benchmark process.
"""
import ast, importlib.util, json, os, sys, tempfile, types, unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "studies/titan-1080/bench"
GOOD = "GPU-00000000-0000-0000-0000-000000000113"
FUTURE = 4_000_000_000.0


def module_at(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


# --------------------------------------------------------------------------
# 3.5 one frozen sampling interval, used everywhere and persisted
# --------------------------------------------------------------------------
class SamplingIntervalTests(unittest.TestCase):
    def setUp(self):
        self.common = module_at("cp_common", "studies/titan-1080/bench/common.py")

    def test_frozen_constant_is_zero_point_two(self):
        self.assertEqual(self.common.SAMPLE_INTERVAL_S, 0.2)

    def test_sampler_default_derives_from_the_constant(self):
        s = self.common.GpuSampler([0])
        self.assertEqual(s.interval, 0.2)

    def test_no_sampled_path_hardcodes_a_different_interval(self):
        """The three inference/CNN paths used to hardcode 0.1 while the protocol
        said 0.2, which would have mis-declared the energy gap rule."""
        offenders = []
        for name in ("bench_cnn.py", "infer_hf.py", "infer_vllm.py", "train_llm.py"):
            text = (BENCH / name).read_text()
            for node in ast.walk(ast.parse(text)):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if kw.arg in ("interval", "sample_interval") and \
                            isinstance(kw.value, ast.Constant) and \
                            isinstance(kw.value.value, (int, float)) and \
                            kw.value.value != 0.2:
                        offenders.append((name, node.lineno, kw.value.value))
        self.assertEqual(offenders, [], f"hardcoded non-0.2 intervals: {offenders}")

    def test_every_sampled_module_references_the_constant(self):
        for name in ("bench_cnn.py", "infer_hf.py", "infer_vllm.py", "train_llm.py"):
            self.assertIn("SAMPLE_INTERVAL_S", (BENCH / name).read_text(), name)

    def test_gap_rule_follows_the_configured_nominal_not_an_observed_mean(self):
        self.assertAlmostEqual(3 * self.common.SAMPLE_INTERVAL_S, 0.6, places=9)

    def _stats_of(self, interval=None):
        """stats() without nvidia-smi: synthetic rows dense enough to satisfy the
        same fail-closed coverage rule a real run must satisfy."""
        step = 0.2 if interval is None else interval
        s = self.common.GpuSampler([0], interval=interval)
        s.t_start = 100.0
        n = int(round(1.0 / step)) + 1
        s._rows = [
            (100.0 + k * step,
             [{"index": 0, "utilization.gpu": 90.0, "power.draw": 120.0,
               "memory.used": 1024.0, "clocks.sm": 1500.0, "temperature.gpu": 60.0}])
            for k in range(n)
        ]
        s.t_stop = s._rows[-1][0]
        return s.stats()

    def test_serialized_telemetry_carries_the_interval_actually_used(self):
        """Declaring 0.2 in metadata is not enough: the number that travels with
        a measurement must come from the sampler that produced it."""
        d = self.common.stats_to_dict(self._stats_of())
        self.assertEqual(d["sample_interval_s"], 0.2)
        self.assertEqual(d["sample_interval_frozen_default_s"], 0.2)
        self.assertIs(d["sample_interval_matches_frozen"], True)
        self.assertEqual(d["max_permitted_gap_s"], 0.6)  # exact: the audit tool compares to a 0.6 literal

    def test_a_nondefault_interval_is_reported_as_a_mismatch_not_silently_as_frozen(self):
        d = self.common.stats_to_dict(self._stats_of(interval=0.5))
        self.assertEqual(d["sample_interval_s"], 0.5)
        self.assertEqual(d["sample_interval_frozen_default_s"], 0.2)
        self.assertIs(d["sample_interval_matches_frozen"], False)


# --------------------------------------------------------------------------
# 3.3 real device binding; absence of UUID must not certify
# --------------------------------------------------------------------------
def fake_torch(*, available=True, current=1, name="NVIDIA TITAN RTX", uuid=GOOD):
    props = types.SimpleNamespace(name=name)
    if uuid is not None:
        props.uuid = uuid
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        current_device=lambda: current,
        get_device_properties=lambda i: props)
    return types.SimpleNamespace(cuda=cuda)


class RuntimeDeviceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.common = module_at("cp_common2", "studies/titan-1080/bench/common.py")

    def test_records_live_current_device_and_uuid(self):
        rec = self.common.runtime_device_identity(1, torch_module=fake_torch())
        self.assertEqual(rec["current_device"], 1)
        self.assertEqual(rec["uuid"], GOOD)
        self.assertTrue(rec["uuid_established"])
        self.assertIn("get_device_properties", rec["uuid_source"])

    def test_missing_uuid_does_not_certify(self):
        rec = self.common.runtime_device_identity(0, torch_module=fake_torch(uuid=None))
        self.assertIsNone(rec["uuid"])
        self.assertFalse(rec["uuid_established"],
                         "identity must NOT be certified when the UUID is unavailable")

    def test_cuda_unavailable_does_not_certify(self):
        rec = self.common.runtime_device_identity(0, torch_module=fake_torch(available=False))
        self.assertFalse(rec["uuid_established"])
        self.assertEqual(rec["uuid_source"], "cuda_unavailable")

    def test_env_echo_is_recorded_but_is_not_the_evidence(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2"}):
            rec = self.common.runtime_device_identity(0, torch_module=fake_torch(uuid=None))
        self.assertEqual(rec["visible_devices_env"], "0,2")
        self.assertFalse(rec["uuid_established"],
                         "echoing the env var must never establish identity")


# --------------------------------------------------------------------------
# 3.1 launch adapters
# --------------------------------------------------------------------------
class LaunchAdapterTests(unittest.TestCase):
    def setUp(self):
        self.launch = module_at("cp_launch", "studies/titan-1080/bench/launch.py")

    def test_recipe_to_adapter_mapping(self):
        f = self.launch.adapter_for_recipe
        for r in ("ddp", "fsdp", "zero1", "zero2",
                  "nccl_allreduce_fixed_payload"):
            self.assertEqual(f(r), "torchrun", r)
        # "single" initialises no process group, so it is a plain controller;
        # singleton ddp/fsdp/zero baselines still need torchrun.
        for r in ("single", "cnn_forward_only", "hf_generate"):
            self.assertEqual(f(r), "plain", r)
        self.assertEqual(f("vllm_offline_tp1"), "vllm_parent")
        self.assertEqual(f("vllm_offline_tp2"), "vllm_parent")
        self.assertEqual(f("vllm_loopback_streaming"), "loopback_server")
        with self.assertRaises(self.launch.LaunchRefusal):
            f("not_a_recipe")

    def test_torchrun_adapter_multiplies_processes_with_devices(self):
        argv = self.launch.build_child_argv(
            sys.executable, str(BENCH / "launch.py"), nproc=6, master_port=29600,
            script_args=[], adapter="torchrun")
        self.assertIn("-m", argv); self.assertIn("torch.distributed.run", argv)
        self.assertIn("--nproc_per_node=6", argv)

    def test_single_parent_adapters_never_emit_torchrun(self):
        for adapter in ("plain", "vllm_parent", "loopback_server"):
            argv = self.launch.build_child_argv(
                sys.executable, str(BENCH / "launch.py"), nproc=1,
                master_port=29600, script_args=["--x"], adapter=adapter)
            self.assertNotIn("torch.distributed.run", argv, adapter)
            self.assertNotIn("--nnodes=1", argv, adapter)
            self.assertEqual(argv[0], sys.executable)
            self.assertEqual(self.launch.ADAPTER_PARENT_PROCESSES[adapter], 1)

    def test_vllm_tp2_still_starts_exactly_one_parent(self):
        """Two copies of a two-GPU engine would start two engines, each claiming
        both GPUs."""
        argv = self.launch.build_child_argv(
            sys.executable, str(BENCH / "launch.py"), nproc=1, master_port=29600,
            script_args=["--tp", "2"], adapter="vllm_parent")
        self.assertNotIn("torch.distributed.run", argv)
        self.assertEqual(self.launch.ADAPTER_PARENT_PROCESSES["vllm_parent"], 1)

    def test_plain_adapter_refuses_multi_device_nproc(self):
        with self.assertRaises(self.launch.LaunchRefusal):
            self.launch.build_child_argv(
                sys.executable, str(BENCH / "launch.py"), nproc=2,
                master_port=29600, script_args=[], adapter="plain")

    def test_unknown_adapter_refuses(self):
        with self.assertRaises(self.launch.LaunchRefusal):
            self.launch.build_child_argv(
                sys.executable, str(BENCH / "launch.py"), nproc=1,
                master_port=29600, script_args=[], adapter="magic")

    def _run(self, adapter, devices, uuids):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        started = []
        class Proc:
            pid = 31000
            def wait(self, timeout): return 0
        def spawn(argv, **kw):
            started.append((argv, kw.get("env", {})))
            return Proc()
        rows = "\n".join(f"{d}, {uuids[d]}, 6.0, 0.0" for d in devices)
        rec = self.launch.run_case(
            case="c", out_root=tmp.name, devices=devices, concurrency_limit=6,
            authorized_devices=devices, expected_uuids=uuids,
            authorization_expiry_epoch=FUTURE, python_exe=sys.executable,
            script=str(BENCH / "launch.py"), script_args=[],
            reservation_root=os.path.join(tmp.name, "leases"), aggregate_limit=6,
            adapter=adapter, grace_s=1, timeout_s=60, popen=spawn,
            gate_kwargs={"query_runner": lambda *a: (0, rows, ""),
                         "apps_runner": lambda t: (0, "", ""),
                         "sleep_fn": lambda s: None,
                         "own_pids": {os.getpid()}})
        return rec, started

    def test_each_adapter_starts_exactly_one_launcher_child(self):
        u = {0: GOOD, 2: "GPU-00000000-0000-0000-0000-000000000114"}
        with mock.patch.object(self.launch, "_bounded_cleanup",
                               return_value={"performed": True}), \
             mock.patch.object(self.launch, "_confirm_group_absent", return_value=True):
            for adapter, devices in (("torchrun", [0, 2]), ("vllm_parent", [0, 2]),
                                     ("plain", [0]), ("loopback_server", [0])):
                rec, started = self._run(adapter, devices, u)
                self.assertEqual(len(started), 1,
                                 f"{adapter}: launcher must start exactly one child")
                argv, env = started[0]
                self.assertEqual(rec["adapter"], adapter)
                self.assertEqual(rec["torchrun_used"], adapter == "torchrun")
                # visibility is by UUID for every adapter
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"],
                                 ",".join(u[d] for d in devices))
                self.assertEqual(env["BENCH_PHYS_DEVICES"],
                                 ",".join(str(d) for d in devices))
                if adapter == "torchrun":
                    self.assertIn(f"--nproc_per_node={len(devices)}", argv)
                else:
                    self.assertNotIn("torch.distributed.run", argv)

    def test_cleanup_runs_for_every_adapter(self):
        u = {0: GOOD}
        with mock.patch.object(self.launch, "_confirm_group_absent", return_value=True), \
             mock.patch.object(self.launch, "_bounded_cleanup",
                               return_value={"performed": True}) as cl:
            for adapter in ("plain", "vllm_parent", "loopback_server"):
                cl.reset_mock()
                self._run(adapter, [0], u)
                self.assertTrue(cl.called, f"{adapter}: cleanup must run")


# --------------------------------------------------------------------------
# 3.6 frozen shared prompt corpus
# --------------------------------------------------------------------------
_corpus_candidates = [
    Path(__file__).resolve().parent.parent / "protocols" / "prompt_corpus.json",
    Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
        / "protocols" / "prompt_corpus.json",
]
CORPUS = Path(os.environ["TITAN_CORPUS"]) if os.environ.get("TITAN_CORPUS") else next(
    (c for c in _corpus_candidates if c.is_file()), _corpus_candidates[0])
_UNUSED_CORPUS = None


@unittest.skipUnless(CORPUS.is_file(), "frozen corpus not present")
class PromptCorpusTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads(CORPUS.read_text())

    def test_shape_and_validity(self):
        self.assertEqual(self.c["prompt_len"], 256)
        self.assertEqual(len(self.c["prompts"]), self.c["n_prompts"])
        self.assertTrue(all(len(p) == 256 for p in self.c["prompts"]))
        v = self.c["vocab_size"]
        self.assertTrue(all(0 <= t < v for p in self.c["prompts"] for t in p))
        special = set(self.c["excluded_special_ids"])
        self.assertFalse(any(t in special for p in self.c["prompts"] for t in p))

    def test_revision_and_policy_frozen(self):
        self.assertEqual(self.c["revision"],
                         "fe8a4ea1ffedaf415f4da2f062534de366a451e6")
        self.assertEqual(self.c["tokenizer_revision"], self.c["revision"])
        pol = self.c["policy"]
        self.assertTrue(pol["greedy"]); self.assertEqual(pol["temperature"], 0.0)
        self.assertEqual(pol["exact_output_tokens"], 128)
        self.assertFalse(pol["enable_prefix_caching"])

    def test_hf_and_vllm_arms_share_prompts_and_64_is_vllm_only(self):
        m = self.c["arm_mapping"]
        self.assertEqual(m["hf_batches"], [1, 4, 16])
        self.assertEqual(m["vllm_request_batches"], [1, 4, 16, 64])
        self.assertNotIn(64, m["hf_batches"])
        for b in m["hf_batches"]:
            self.assertIn(b, m["vllm_request_batches"])
        self.assertGreaterEqual(self.c["n_prompts"], 64)

    def test_corpus_hash_is_reproducible(self):
        import hashlib
        body = {k: v for k, v in self.c.items() if k != "corpus_sha256"}
        blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(blob).hexdigest(), self.c["corpus_sha256"])


# --------------------------------------------------------------------------
# 3.7 collective units
# --------------------------------------------------------------------------
class CollectiveUnitTests(unittest.TestCase):
    def setUp(self):
        self.text = (BENCH / "comm_bench.py").read_text()

    def test_payloads_are_labelled_mebibytes(self):
        self.assertIn("SIZES_MIB", self.text)
        self.assertNotIn("SIZES_MB =", self.text)

    def test_exact_bytes_recorded_and_gradient_claim_is_explicit(self):
        self.assertIn("payload_bytes", self.text)
        self.assertIn("is_exact_gpt2_124m_fp32_gradient_volume", self.text)
        self.assertIn("GPT2_124M_FP32_GRADIENT_BYTES", self.text)

    def test_497_mib_is_not_the_gradient_volume(self):
        self.assertEqual(497 * 1024 * 1024, 521_142_272)
        self.assertEqual(124_439_808 * 4, 497_759_232)
        self.assertNotEqual(497 * 1024 * 1024, 124_439_808 * 4)
        ratio = (497 * 1024 * 1024) / (124_439_808 * 4)
        self.assertAlmostEqual(ratio, 1.047, places=3)

    def test_ring_normalisation_is_not_called_measured_wire_traffic(self):
        self.assertIn("ring_equivalent_bw_gb_s", self.text)
        self.assertIn("not measured wire traffic", self.text)

    def test_unknown_p2p_entry_cannot_yield_all_enabled(self):
        self.assertIn("p2p_unknown_entries", self.text)
        self.assertIn("None if any(v is None for v in p2p.values())", self.text)


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# frozen statistics and the independent audit validator
# --------------------------------------------------------------------------
TOOLS = ROOT / "tools"


def _tool(name):
    spec = importlib.util.spec_from_file_location(f"cp_{name}", TOOLS / f"{name}.py")
    m = importlib.util.module_from_spec(spec); sys.modules[f"cp_{name}"] = m
    spec.loader.exec_module(m); return m


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.ts = _tool("titan_stats")

    def test_minmax_median_coverage_at_six_is_31_over_32(self):
        from fractions import Fraction
        self.assertEqual(self.ts.minmax_median_coverage(6), Fraction(31, 32))
        self.assertAlmostEqual(float(self.ts.minmax_median_coverage(6)), 0.96875, places=9)

    def test_coverage_matches_brute_force_enumeration(self):
        """Independent check: coverage fails only when all n points land on one
        side of the median."""
        from fractions import Fraction
        from itertools import product
        for n in (1, 2, 3, 6, 8):
            bad = sum(1 for s in product([0, 1], repeat=n) if sum(s) in (0, n))
            self.assertEqual(self.ts.minmax_median_coverage(n),
                             Fraction(2 ** n - bad, 2 ** n), n)

    def test_coverage_is_not_claimable_below_six_blocks(self):
        s = self.ts.summarize([1.0, 2.0, 3.0, 4.0])
        self.assertFalse(s["coverage_claimable"])
        self.assertIsNone(s["precision_goal_met"])
        self.assertIn("no six-block confidence claim", s["qualification"])

    def test_summary_reports_points_median_cv_and_range(self):
        s = self.ts.summarize([100.0, 101.0, 99.5, 100.5, 100.2, 99.8])
        for k in ("points", "median", "sample_cv_pct", "range_over_median_pct",
                  "interval", "minmax_median_coverage"):
            self.assertIn(k, s)
        self.assertEqual(len(s["points"]), 6)
        self.assertTrue(s["coverage_claimable"])

    def test_precision_goal_flags_a_wide_interval(self):
        self.assertFalse(self.ts.summarize([100, 100, 100, 100, 100, 200])["precision_goal_met"])

    def test_matched_block_ratio_uses_only_shared_blocks(self):
        r = self.ts.matched_block_ratio({1: 10.0, 2: 20.0, 3: 30.0},
                                        {1: 5.0, 2: 10.0, 4: 1.0})
        self.assertEqual(r["blocks_used"], [1, 2])
        self.assertEqual(r["median_ratio"], 2.0)

    def test_matched_block_ratio_with_no_shared_blocks_is_unestablished(self):
        r = self.ts.matched_block_ratio({1: 1.0}, {2: 1.0})
        self.assertEqual(r["n"], 0)
        self.assertIsNone(r["median_ratio"])


class AuditValidatorTests(unittest.TestCase):
    def setUp(self):
        self.au = _tool("titan_audit")

    def test_gap_cutoff_is_three_times_the_frozen_nominal(self):
        self.assertEqual(self.au.NOMINAL_INTERVAL_S, 0.2)
        self.assertAlmostEqual(self.au.GAP_CUTOFF_S, 0.6, places=9)

    def test_integrator_exact_on_constant_and_linear_traces(self):
        ts = [i * 0.1 for i in range(101)]
        j, cov = self.au.integrate(ts, [100.0] * 101, 0.0, 10.0)
        self.assertTrue(cov); self.assertAlmostEqual(j, 1000.0, places=6)
        j2, _ = self.au.integrate(ts, [10.0 * t for t in ts], 0.0, 10.0)
        self.assertAlmostEqual(j2, 500.0, places=9)

    def test_integrator_refuses_an_uncovered_window(self):
        _, cov = self.au.integrate([1.0, 2.0], [50.0, 50.0], 0.5, 2.5)
        self.assertFalse(cov)

    def test_self_test_rejects_every_corruption(self):
        """Runs the validator against synthetic valid and corrupted records."""
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.au.self_test()
        self.assertEqual(rc, 0, buf.getvalue())
        text = buf.getvalue()
        self.assertNotIn("FAIL  ", text)
        for expected in ("rejects_corrupt_energy_value", "rejects_truncated_coverage",
                         "rejects_oversized_gap", "rejects_wrong_interval",
                         "rejects_silent_query_failures",
                         "rejects_inflated_successful_updates",
                         "rejects_empty_gradients_claiming_coverage",
                         "rejects_bad_units"):
            self.assertIn(expected, text)

    def test_497_mib_is_flagged_as_not_the_gradient_volume(self):
        self.assertEqual(self.au.GPT2_124M_FP32_GRADIENT_BYTES, 497_759_232)
        self.assertNotEqual(497 * 1024 * 1024, self.au.GPT2_124M_FP32_GRADIENT_BYTES)


class CollectivePayloadTests(unittest.TestCase):
    """The sweep claimed to measure "the 497 MB gradient payload" but its 497
    point is 497 MiB = 521,142,272 B, while GPT-2 124M's FP32 gradient volume is
    497,759,232 B = 474.7 MiB. Every row therefore reported
    is_exact_gpt2_124m_fp32_gradient_volume = false and the real DDP payload went
    unmeasured. These tests pin the unit distinction so it cannot recur."""

    def setUp(self):
        self.cb = module_at("cp_comm", "studies/titan-1080/bench/comm_bench.py")

    def test_the_gradient_constant_is_parameters_times_four(self):
        self.assertEqual(self.cb.GPT2_124M_FP32_GRADIENT_BYTES, 124_439_808 * 4)
        self.assertEqual(self.cb.GPT2_124M_FP32_GRADIENT_BYTES, 497_759_232)

    def test_the_gradient_volume_is_not_497_mebibytes(self):
        """The conflation itself, stated as a test."""
        self.assertNotEqual(self.cb.GPT2_124M_FP32_GRADIENT_BYTES, 497 * 1024 * 1024)
        self.assertAlmostEqual(
            self.cb.GPT2_124M_FP32_GRADIENT_BYTES / 1024 / 1024, 474.700, places=3)
        self.assertAlmostEqual(
            self.cb.GPT2_124M_FP32_GRADIENT_BYTES / 1e6, 497.759, places=3)

    def test_exactly_one_measured_payload_is_the_gradient_volume(self):
        pts = self.cb.measurement_payloads()
        exact = [b for _, b in pts if b == self.cb.GPT2_124M_FP32_GRADIENT_BYTES]
        self.assertEqual(len(exact), 1, f"payloads: {pts}")

    def test_the_497_mebibyte_point_is_retained_and_still_not_exact(self):
        pts = dict((b, mb) for mb, b in self.cb.measurement_payloads())
        self.assertIn(497 * 1024 * 1024, pts, "the original point must not vanish")
        self.assertNotEqual(497 * 1024 * 1024, self.cb.GPT2_124M_FP32_GRADIENT_BYTES)

    def test_every_payload_is_a_whole_number_of_fp32_elements(self):
        for mb, b in self.cb.measurement_payloads():
            self.assertEqual(b % 4, 0, f"{mb}: {b} is not divisible by 4")

    def test_payloads_are_unique_and_ascending(self):
        bs = [b for _, b in self.cb.measurement_payloads()]
        self.assertEqual(bs, sorted(bs))
        self.assertEqual(len(bs), len(set(bs)))
