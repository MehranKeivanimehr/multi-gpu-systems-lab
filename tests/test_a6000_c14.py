"""CPU tests for Candidate 14: the cushion-preserving admission predicate in the admission path, per-case private caches
(seed verification, pre-load refusal, child inheritance), the SDPA dispatch recorder, DeepSpeed boundary-gradient
extraction (fake optimizer), and the first-boundary result kind end to end on tiny CPU models (gloo, world 1).
Nothing here touches a GPU, starts a scope or loads any existing cache file."""
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
A6000 = ROOT / "studies/a6000"
SRC = A6000 / "src"
DESIGN = ROOT.parents[1] / "33_s1a_design"
UUID4 = "GPU-00000000-0000-0000-0000-000000000109"
try:
    import torch
    import torch.distributed as dist
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
try:
    import transformers  # noqa: F401
    import yaml
    HAVE_HF = True
except ImportError:
    HAVE_HF = False
TINY = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "vocab_size": 128, "hidden_size": 32,
        "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 64, "tie_word_embeddings": True, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "torch_dtype": "float32", "attention_dropout": 0.0, "use_cache": False}


def mg(name):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return __import__(name, fromlist=["x"])


class Admission(unittest.TestCase):
    def setUp(self):
        self.s = mg("mgbench.smoke")

    def test_a4_counterexample_is_refused_not_clamped(self):
        a = self.s.admission(43.48, 47.402, 0.8)
        self.assertFalse(a["eligible"])
        self.assertIsNone(a["fraction"])
        self.assertEqual(a["required_fraction"], "0.94")
        self.assertIn("exceeds the cap", a["reasons"][0])

    def test_valid_cases_keep_the_cushion(self):
        for pred, f in ((41.96, 0.91), (38.66, 0.84), (35.55, 0.77), (41.71, 0.90)):
            a = self.s.admission(pred, 47.402, 0.8)
            self.assertTrue(a["eligible"], a)
            self.assertEqual(a["fraction"], f)
            self.assertGreaterEqual(float(a["allowance_gib"]), 1.02 * pred - 1e-9)

    def test_headroom_and_malformed_inputs_refuse(self):
        self.assertFalse(self.s.admission(41.0, 47.402, 5.0)["eligible"])           # headroom < 2 GiB
        for bad in (True, "43.0", float("nan"), float("inf"), -1.0, 0, None):
            self.assertFalse(self.s.admission(bad, 47.402, 0.8)["eligible"], bad)
            self.assertFalse(self.s.admission(30.0, bad, 0.8)["eligible"], bad)
            self.assertFalse(self.s.admission(30.0, 47.402, bad)["eligible"], bad)

    def test_feasibility_evidence_missing_tampered_or_inconsistent_refuses(self):
        feas = {"torch_total_memory_gib": 47.402, "non_allocator_gib": 0.8,
                "cases": {"x": {"predicted_reserved_gib": 38.66, "fraction": 0.84, "eligible": True}}}
        case = {"id": "x", "mem_fraction": 0.84}
        self.assertEqual(self.s.feasibility_problems(feas, [case]), [])
        self.assertTrue(self.s.feasibility_problems(None, [case]))
        self.assertTrue(self.s.feasibility_problems({"cases": "x"}, [case]))
        self.assertTrue(self.s.feasibility_problems(feas, [{"id": "y", "mem_fraction": 0.84}]))   # no record
        for mut in (lambda f: f["cases"]["x"].update(fraction=0.83), lambda f: f["cases"]["x"].update(eligible=False),
                    lambda f: f["cases"]["x"].update(predicted_reserved_gib=43.48),
                    lambda f: f.update(torch_total_memory_gib=True)):
            g = copy.deepcopy(feas)
            mut(g)
            self.assertTrue(self.s.feasibility_problems(g, [case]))
        self.assertTrue(self.s.feasibility_problems(feas, [{"id": "x", "mem_fraction": 0.92}]))    # config mismatch
        self.assertTrue(self.s.feasibility_problems(feas, [{"id": "x", "mem_fraction": True}]))

    @unittest.skipUnless(HAVE_HF, "yaml not installed")
    def test_sealed_s1a_feasibility_refuses_exactly_a4_in_the_admission_path(self):
        p = DESIGN / "memory_time_feasibility_v2.json"
        if not p.exists():
            self.skipTest("frozen S1a design files are on the node only")
        cfg = yaml.safe_load((A6000 / "configs/s1a_qwen15b.yaml").read_text())
        probs = self.s.diag_config_problems(cfg, json.loads(p.read_text()))
        self.assertEqual(len(probs), 1, probs)
        self.assertIn("a4_zero2_bf16", probs[0])

    def test_null_or_mismatched_authorization_never_permits(self):
        lc = mg("mgbench.launch_controls")
        self.assertTrue(lc.authorization_problems(None, scope="train-first-boundary", candidate_root=ROOT, gpus=[4],
                                                  uuids={4: UUID4}))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "auth.json"
            p.write_text(json.dumps({"scope": None, "gpu_index": None, "issued_utc": None, "expires_utc": None}))
            self.assertTrue(lc.authorization_problems(p, scope="train-first-boundary", candidate_root=ROOT, gpus=[4],
                                                      uuids={4: UUID4}))


class PerCaseCaches(unittest.TestCase):
    def setUp(self):
        self.s = mg("mgbench.smoke")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.seed = self.tmp / "seed" / "KEY"
        self.seed.mkdir(parents=True)
        (self.seed / "cuda_utils.so").write_bytes(b"\x7fELF fake helper")
        self.sha = hashlib.sha256(b"\x7fELF fake helper").hexdigest()
        self.root = self.tmp / "cache"
        self.root.mkdir(mode=0o700)

    def test_each_case_gets_its_own_cache_with_only_the_verified_binary(self):
        a = self.s.make_case_cache(self.root, "01_a", self.seed, self.sha)
        b = self.s.make_case_cache(self.root, "02_b", self.seed, self.sha)
        self.assertEqual(a["problems"] + b["problems"], [])
        self.assertEqual(oct(Path(a["dir"]).stat().st_mode & 0o777), "0o700")
        (Path(a["dir"]) / "triton/Fp16Matmul_2d_kernel.pickle").write_bytes(b"\x80\x04}\x94.")   # a's runtime output
        self.assertEqual(self.s.cache_preload_problems(Path(b["dir"]), b["expected_manifest"]), [])
        self.assertTrue(self.s.cache_preload_problems(Path(a["dir"]), a["expected_manifest"]))   # refuse before load
        files = [k for k, v in b["expected_manifest"].items() if v != "dir"]
        self.assertEqual(files, ["triton/KEY/cuda_utils.so"])
        self.assertEqual(b["seed"]["sha256_copy"], self.sha)

    def test_unverified_or_pickle_bearing_seed_is_refused(self):
        self.assertTrue(self.s.make_case_cache(self.root, "01_bad", self.seed, "0" * 64)["problems"])
        (self.seed / "x.pickle").write_bytes(b"\x80\x04}\x94.")
        self.assertTrue(self.s.make_case_cache(self.root, "02_pk", self.seed, self.sha)["problems"])

    def test_symlink_or_changed_binary_refuses(self):
        a = self.s.make_case_cache(self.root, "01_a", self.seed, self.sha)
        so = Path(a["dir"]) / "triton/KEY/cuda_utils.so"
        so.write_bytes(b"changed")
        self.assertTrue(self.s.cache_preload_problems(Path(a["dir"]), a["expected_manifest"]))
        b = self.s.make_case_cache(self.root, "02_b", self.seed, self.sha)
        os.symlink("/tmp", Path(b["dir"]) / "xdg/link")
        self.assertTrue(self.s.cache_preload_problems(Path(b["dir"]), b["expected_manifest"]))

    def test_worker_env_points_every_cache_into_the_case_dir_and_children_inherit_it(self):
        a = self.s.make_case_cache(self.root, "01_a", None, None)
        env = self.s.env_for_worker({"HOME": "/h", "TRITON_CACHE_DIR": "/shared"}, src=SRC, uuid=UUID4, index=4,
                                    threads=2, cache=Path(a["dir"]), run_id="r")
        for k in ("TRITON_CACHE_DIR", "TRITON_HOME", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR",
                  "XDG_CACHE_HOME", "MPLCONFIGDIR"):
            self.assertTrue(env[k].startswith(a["dir"] + "/"), k)
        env["CUDA_VISIBLE_DEVICES"] = ""
        code = "import os, json, subprocess, sys; print(subprocess.run([sys.executable, '-s', '-c', " \
               "'import os, json; print(json.dumps({k: os.environ.get(k) for k in (\"TRITON_CACHE_DIR\", \"XDG_CACHE_HOME\")}))']," \
               " capture_output=True, text=True).stdout.strip())"
        out = subprocess.run([sys.executable, "-s", "-c", code], env=env, capture_output=True, text=True, timeout=60)
        got = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(got["TRITON_CACHE_DIR"], env["TRITON_CACHE_DIR"])                     # grandchild
        self.assertEqual(got["XDG_CACHE_HOME"], env["XDG_CACHE_HOME"])

    def test_supervisor_uses_per_case_caches_and_checks_before_launch(self):
        sup = (A6000 / "scripts/smoke_supervisor.py").read_text()
        self.assertIn("smoke.make_case_cache(cache, f\"{n:02d}_{case['id']}\", seed_dir, seed_sha)", sup)
        self.assertLess(sup.index("smoke.cache_preload_problems(case_cache"), sup.index("scopes.start_scope(unit"))
        self.assertNotIn("shutil.copytree(src_dir, dst)", sup)                                   # no shared seeding


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class DeepSpeedBoundaryExtraction(unittest.TestCase):
    def setUp(self):
        self.fw = mg("mgbench.train.frameworks")

    def fake(self, pre=False, **kw):
        a, b = torch.arange(6, dtype=torch.bfloat16), torch.arange(3, dtype=torch.bfloat16) + 10
        zo = types.SimpleNamespace(bit16_groups=[[a, b]], averaged_gradients={0: [a, b]}, partition_size=[10],
                                   clip_grad=0.0, loss_scale=1.0,
                                   single_partition_of_fp32_groups=[torch.zeros(10, dtype=torch.float32)])
        zo.flatten_dense_tensors_aligned = lambda ts, n: torch.cat([torch.cat([t.reshape(-1) for t in ts]),
                                                                    torch.zeros(n - 9, dtype=ts[0].dtype)])
        zo._get_preflattened_grad_partition = (lambda i: zo.flatten_dense_tensors_aligned(zo.averaged_gradients[0], 10)) \
            if pre else (lambda i: None)
        for k, v in kw.items():
            setattr(zo, k, v)
        return zo

    def test_matches_steps_flattening_and_cast_without_transform(self):
        for pre in (False, True):
            g = self.fw.deepspeed_boundary_gradient(self.fake(pre=pre))
            self.assertEqual(g.dtype, torch.float32)
            self.assertEqual(g.tolist(), [0, 1, 2, 3, 4, 5, 10, 11, 12, 0])                # padding last, exact cast
        slices = [("a", 0, 0, 6, (6,)), ("b", 0, 6, 3, (3,))]
        items = self.fw.flat_slices(g, slices)
        self.assertEqual([n for n, _ in items], ["a", "b"])
        self.assertEqual(items[1][1].tolist(), [10, 11, 12])                                 # offset identity, no pad

    def test_refuses_transforms_missing_reduction_or_extra_groups(self):
        for kw in ({"clip_grad": 1.0}, {"loss_scale": 2.0}, {"averaged_gradients": {}}, {"bit16_groups": [[1], [2]]}):
            with self.assertRaises(RuntimeError, msg=kw):
                self.fw.deepspeed_boundary_gradient(self.fake(**kw))
        zo = self.fake(pre=True)          # a pre-flattened buffer whose size disagrees with partition_size
        zo.partition_size = [11]
        with self.assertRaises(RuntimeError):
            self.fw.deepspeed_boundary_gradient(zo)

    def test_step_is_not_called_at_the_boundary(self):
        src = (SRC / "mgbench/train/frameworks.py").read_text()
        body = src[src.index("def setup_deepspeed"):]
        loop = body[body.index("for i, b in enumerate(micro_batches):"):body.index("ledger.record_microbatch_losses(raw)")]
        self.assertLess(loop.index("if first_boundary(spec) and i == n - 1:"), loop.index("engine.step()"))
        for setup in ("def setup_ddp", "def setup_fsdp"):
            b = src[src.index(setup):]
            b = b[:b.index("\ndef ", 1)]
            self.assertLess(b.index("if first_boundary(spec):"), b.index("opt.step()"))


@unittest.skipUnless(HAVE_TORCH and HAVE_HF, "torch/transformers not installed")
class FirstBoundaryEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        if dist.is_initialized():
            self.skipTest("a process group already exists")
        dist.init_process_group("gloo", store=dist.FileStore(str(self.tmp / "store"), 1), rank=0, world_size=1)
        self.addCleanup(dist.destroy_process_group)
        import torch.nn.parallel as tnp
        base = tnp.DistributedDataParallel

        class CpuDDP(base):
            def __init__(self, module, device_ids=None, **kw):
                super().__init__(module, device_ids=None, **kw)
        p = mock.patch.object(tnp, "DistributedDataParallel", CpuDDP)
        p.start()
        self.addCleanup(p.stop)
        self.fw, self.main = mg("mgbench.train.frameworks"), mg("mgbench.train.__main__")
        self.cfg, self.data = mg("mgbench.config"), mg("mgbench.data")
        d = self.tmp / "tiny"
        d.mkdir()
        (d / "config.json").write_text(json.dumps(TINY))
        self.entry = {"kind": "causal_lm", "hf_config_dir": str(d)}
        self.ref = self.tmp / "reference"
        self.ref.mkdir()

    def spec(self, framework, **kw):
        s = dict(model="tiny", seq_len=16, micro_batch=1, grad_accum=4, seed=1234, warmup_steps=0, measure_steps=3,
                 numerical_checks=True, canonical_init=True, canonical_start="bf16_representable", global_inputs=True,
                 reference_dir=str(self.ref), framework=framework, n_gpus=1)
        s.update(kw)
        return s

    def batches(self, steps):
        return self.data.GlobalSynthetic("causal_lm", micro_batch=1, grad_accum=4, world=1, rank=0,
                                         device=torch.device("cpu"), seed=1234000, steps=steps, vocab_size=128,
                                         seq_len=16, image_size=None, num_classes=None, dtype=torch.float32)

    def write_reference(self):
        spec = self.spec("ddp", fp32_reference=True, reference_role="write")
        tr = self.fw.SETUP["ddp"](spec, self.entry, self.cfg.resolved_framework_config(spec), torch.device("cpu"))
        data = self.batches(3)
        tr.ledger.start_measured()
        for _ in range(3):
            tr.step([data.next() for _ in range(4)])
        self.main.finalize_reference(tr, tr.ledger.summary(3, 0, 4)["measured"])

    def first_boundary(self, framework, **kw):
        out = self.tmp / f"case_{framework}_{len(list(self.tmp.glob('case_*')))}"
        out.mkdir()
        spec = self.spec(framework, result_kind="first_boundary_gradient", measure_steps=1, _out_dir=str(out), **kw)
        tr = self.fw.SETUP[framework](spec, self.entry, self.cfg.resolved_framework_config(spec), torch.device("cpu"))
        before = tr.state_fingerprint()
        canon = self.main._compare_canonical(tr.canonical, before, spec)
        data = self.batches(1)
        tr.ledger.start_measured()
        rec = self.main.SdpaDispatchRecorder()
        with rec:
            tr.step([data.next() for _ in range(4)])
        summ = tr.ledger.summary(1, 0, 4)
        after = tr.state_fingerprint()
        comp = tr.reference.finish(summ["measured"]["microbatch_losses"])
        a = self.main.assess_first_boundary(summ, 4, after == before, comp)
        return a, summ, comp, canon, rec.record(), out, data, tr

    def test_first_boundary_kinds(self):
        self.write_reference()
        # identical FP32 recipe: exactly zero difference, no update, parameters unchanged
        a, summ, comp, canon, sd, out, data, tr = self.first_boundary("ddp", fp32_reference=True)
        m = summ["measured"]
        self.assertEqual((m["logical_attempts"], m["microbatches"], m["base_optimizer_calls"], m["counter_updates"]),
                         (1, 4, 0, 0))
        self.assertEqual(a["execution"], "ok", a)
        self.assertEqual(a["update_state"]["status"], "not_applicable")
        self.assertTrue(a["parameters_unchanged"])
        self.assertEqual(a["gradient"]["status"], "complete_at_the_boundary")
        v = comp["snapshots"]["vs_sealed_r0_grad_b1"]
        self.assertTrue(v["coverage_complete"] and v["nonzero_certifiable"])
        self.assertEqual(v["global_rel_l2"], 0.0)
        self.assertEqual(comp["loss_rel"], [0.0] * 4)
        self.assertEqual(comp["grad_norm_ratio_global"], 1.0)
        self.assertTrue((out / "first_boundary_grad.f32").exists())
        self.assertEqual(len(data.input_digests["per_step_microbatch_sha256"]), 1)
        self.assertGreater(sd["calls"], 0)
        self.assertEqual(tr.update_audit["optimizer_calls"], 0)
        with self.assertRaises(RuntimeError):                                     # exactly one boundary
            tr.reference.at_boundary(0, [])
        # native bf16 parameters (G3): realized bf16 canonical start, bf16 grads, small nonzero difference
        a3, s3, c3, canon3, _, _, _, tr3 = self.first_boundary("ddp", param_dtype="bf16")
        self.assertTrue(canon3["module_state_matches_canonical"])
        self.assertEqual(canon3["expected_module_state"].split()[0], "canonical_bf16")
        self.assertTrue(a3["parameters_unchanged"] and a3["execution"] == "ok")
        g3 = c3["snapshots"]["vs_sealed_r0_grad_b1"]
        self.assertTrue(g3["coverage_complete"] and 0 < g3["global_rel_l2"] < 1)
        # FSDP2 (bf16 params policy) first boundary
        a2, s2, c2, _, _, _, _, _ = self.first_boundary("fsdp")
        self.assertEqual(a2["execution"], "ok", a2)
        self.assertTrue(c2["snapshots"]["vs_sealed_r0_grad_b1"]["coverage_complete"])

    def test_zero_equals_zero_is_never_certified(self):
        ref = self.tmp / "zref"
        ref.mkdir()
        r = mg("mgbench.train.reference")
        w = r.SnapshotWriter(ref / "grad_b1.f32")
        w.add("w", torch.zeros(5))
        w.close()
        (ref / "reference_scalars.json").write_text(json.dumps({"microbatch_losses_measured": [1.0] * 4,
                                                                "grad_norms_per_boundary": [{"w": 0.0}]}))
        out = self.tmp / "zout"
        out.mkdir()
        s = r.FirstBoundarySession(out, ref)
        s.at_boundary(0, [("w", torch.zeros(5))])
        v = s.finish([1.0] * 4)["snapshots"]["vs_sealed_r0_grad_b1"]
        self.assertFalse(v["nonzero_certifiable"])
        summ = {"measured": {"logical_attempts": 1, "microbatches": 4, "base_optimizer_calls": 0, "counter_updates": 0,
                             "microbatch_losses_all_finite": True, "boundary_coverage": ["complete"]},
                "warmup": {"logical_attempts": 0}}
        a = self.main.assess_first_boundary(summ, 4, True, {"snapshots": {"vs_sealed_r0_grad_b1": v}})
        self.assertTrue(a["numerical_comparison"].startswith("not_certifiable"))

    def test_sdpa_policy_and_recorder(self):
        pol = {"tf32": False, "bf16_reduced_precision_reduction": False, "sdpa": "math"}
        b = torch.backends.cuda
        saved = (b.flash_sdp_enabled(), b.mem_efficient_sdp_enabled(), b.math_sdp_enabled(), b.cudnn_sdp_enabled(),
                 b.matmul.allow_bf16_reduced_precision_reduction, b.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.get_float32_matmul_precision(), torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)

        def restore():
            b.enable_flash_sdp(saved[0]); b.enable_mem_efficient_sdp(saved[1]); b.enable_math_sdp(saved[2])
            b.enable_cudnn_sdp(saved[3]); b.matmul.allow_bf16_reduced_precision_reduction = saved[4]
            b.matmul.allow_tf32 = saved[5]; torch.backends.cudnn.allow_tf32 = saved[6]
            torch.set_float32_matmul_precision(saved[7]); torch.backends.cudnn.benchmark = saved[8]
            torch.backends.cudnn.deterministic = saved[9]
        self.addCleanup(restore)
        rec = self.main.SdpaDispatchRecorder()
        q = torch.randn(1, 2, 4, 8)
        with rec:
            torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        self.assertEqual(rec.record()["dispatch"], "NOT observed (several backends enabled)")
        self.main.apply_matched_policy({"matched_policy": pol})
        rec2 = self.main.SdpaDispatchRecorder()
        with rec2:
            torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        self.assertEqual(rec2.record()["dispatch"], "constrained to math by flags")
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, rec2.orig)            # unwrapped afterwards
        self.assertFalse(b.matmul.allow_bf16_reduced_precision_reduction)
        with self.assertRaises(ValueError):
            self.main.apply_matched_policy({"matched_policy": dict(pol, sdpa="flash_forced")})


@unittest.skipUnless(HAVE_HF, "yaml not installed")
class FirstBoundaryScope(unittest.TestCase):
    def setUp(self):
        self.s = mg("mgbench.smoke")
        self.cfg = yaml.safe_load((A6000 / "configs/fb_qwen15b.yaml").read_text())
        fp = ROOT.parents[1] / "35_c15_design/fb_memory_feasibility.json"
        if not fp.exists():
            self.skipTest("Candidate 15 design files are on the node only")
        self.feas = json.loads(fp.read_text())

    def test_proposed_config_is_admissible_and_fits(self):
        self.assertEqual(self.s.fb_config_problems(self.cfg, self.feas), [])
        wc = mg("mgbench.window_contract")         # Candidate 19: the Candidate 18 timing model replaces the 2090-s sum
        self.assertEqual(wc.timing_model(self.cfg["timing"], n_gpus_watched=1, n_cases=5)["inclusive_window_s"], 2459)
        self.assertEqual(self.cfg["cases"][-1]["id"], "X1_fp32_order_permuted")      # exploratory, NOT control C
        self.assertFalse(self.feas["cases"]["G2_ds_autocast"]["eligible"])
        self.assertFalse(self.feas["cases"]["G5_ds_bf16_fp32accum"]["eligible"])

    def test_refusals(self):
        muts = [lambda c: c.update(scope="train-diagnostics"), lambda c: c["cases"].reverse(),
                lambda c: c["cases"].append({"id": "G2_ds_autocast", "framework": "zero2", "ds_precision": "autocast",
                                             "mem_fraction": 0.92}),
                lambda c: c["cases"][1].update(mem_fraction=0.92), lambda c: c["cases"][2].pop("param_dtype"),
                lambda c: c["cases"][3].update(ds_grad_accum_dtype="fp32"), lambda c: c["common"].pop("matched_policy"),
                lambda c: c["common"].update(measure_steps=3), lambda c: c["timing"].update(child_s=300),
                lambda c: c.update(reference_budget_bytes=True), lambda c: c.update(allowed_gpus=[4, 5]),
                lambda c: c["cases"][4].update(microbatch_order=[0, 1, 2, 3]), lambda c: c["cases"].pop()]
        for i, m in enumerate(muts):
            c = copy.deepcopy(self.cfg)
            m(c)
            self.assertTrue(self.s.fb_config_problems(c, self.feas), i)
        self.assertTrue(self.s.fb_config_problems(self.cfg, None))                       # missing evidence

    def test_specs_expand_to_one_boundary_without_update(self):
        cfgmod = mg("mgbench.config")
        cluster = dict(cfgmod.cluster_config(), allowed_gpus=[4], gpu_sets={1: [4]})
        for case in self.cfg["cases"]:
            (spec,) = cfgmod.expand_experiment(self.s.fb_experiment(case, self.cfg, Path("/sealed/ref")), cluster)
            self.assertEqual((spec["result_kind"], spec["measure_steps"], spec["warmup_steps"], spec["grad_accum"]),
                             ("first_boundary_gradient", 1, 0, 4))
            self.assertEqual(spec["gpus"], [4])
            for k in ("fp32_reference", "param_dtype"):
                self.assertEqual(spec.get(k), case.get(k))

    def test_evaluator_refuses_any_sign_of_an_update_or_uncertifiable_numbers(self):
        case = self.cfg["cases"][3]
        dev = lambda st: {"stage": st, "device_count": 1, "uuid": UUID4, "pci_bus_id": "0000:81:00.0"}
        good = {"result_kind": "first_boundary_gradient", "parameters_unchanged": True, "mem_fraction": 0.85,
                "assessment": {"execution": "ok", "gradient": {"status": "complete_at_the_boundary"},
                               "update_state": {"status": "not_applicable"}, "numerical_comparison": "comparable"},
                "canonical_state_check": {"module_state_matches_canonical": True, "optimizer_master_matches_canonical": True},
                "input_digests": {"per_step_microbatch_sha256": [["a", "b", "c", "d"]]},
                "runtime_devices": [dev("after set_device"), dev("after set_per_process_memory_fraction"), dev("end of run")],
                "identity_chain": {"checked": True, "problems": []},
                "process": {"mgbench_file": str(SRC / "mgbench/__init__.py"), "sys_executable": "/py"},
                "precision": {"matched_policy": self.s.FB_COMMON["matched_policy"], "legacy": {"cuda_matmul_allow_tf32": False}},
                "sdpa_dispatch": {"dispatch": "constrained to math by flags"}}
        frozen = {"input_per_step_microbatch_sha256": [["a", "b", "c", "d"], ["e", "f", "g", "h"]]}
        ev = lambda r, rc=0: self.s.evaluate_fb_case(r, rc, case, frozen, uuid=UUID4, candidate_src=SRC, interpreter="/py")
        self.assertTrue(ev(good)["pass"], [c for c in ev(good)["checks"] if not c["pass"]])
        for m in (lambda r: r.update(parameters_unchanged=False),
                  lambda r: r["assessment"]["update_state"].update(status="verified_sampled_every_base_call"),
                  lambda r: r["assessment"].update(numerical_comparison="not_certifiable (zero vector)"),
                  lambda r: r["input_digests"].update(per_step_microbatch_sha256=[["a", "b", "c", "x"]]),
                  lambda r: r["sdpa_dispatch"].update(dispatch="NOT observed (several backends enabled)"),
                  lambda r: r["canonical_state_check"].update(optimizer_master_matches_canonical=False),
                  lambda r: r.update(mem_fraction=0.92), lambda r: r["runtime_devices"][2].update(uuid="GPU-x")):
            r = copy.deepcopy(good)
            m(r)
            self.assertFalse(ev(r)["pass"])
        self.assertFalse(ev(good, rc=4)["pass"])
        self.assertFalse(ev(None)["pass"])


if __name__ == "__main__":
    unittest.main()
