"""CPU tests for stage S1a (single-GPU numerical diagnostics): the streaming reference writer/comparer, the reference
session, the declared common representable start, DeepSpeed slice mapping, AdamW state/bias-correction errors, the
diagnostics configuration, budget, five-case authorization and case evaluation. CUDA is never used; tiny models run
the production DDP / FSDP2 trainers on CPU with a world-size-1 gloo group. These show logic, not GPU behaviour."""
import copy
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
A6000 = ROOT / "studies/a6000"
SRC = A6000 / "src"
UUID4 = "GPU-00000000-0000-0000-0000-000000000109"
DESIGN = ROOT.parents[1] / "33_s1a_design"          # evidence root design files (present on the node only)

try:
    import torch
    import torch.distributed as dist
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
try:
    import transformers  # noqa: F401
    HAVE_HF = True
except ImportError:
    HAVE_HF = False
try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

TINY = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "vocab_size": 128, "hidden_size": 32,
        "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 64, "tie_word_embeddings": True, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "torch_dtype": "float32", "attention_dropout": 0.0, "use_cache": False}


def mg(name):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return __import__(name, fromlist=["x"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class SnapshotFixtures(unittest.TestCase):
    def setUp(self):
        self.r = mg("mgbench.train.reference")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        g = torch.Generator().manual_seed(0)
        self.ref = {"a": torch.randn(5, 7, generator=g), "b": torch.randn(33, generator=g),
                    "c": torch.randn(2, 3, 4, generator=g)}

    def write(self, tensors, name="snap.f32"):
        w = self.r.SnapshotWriter(self.tmp / name)
        for k, v in tensors.items():
            w.add(k, v)
        return w.close(), self.tmp / name

    def compare(self, observed, name="snap.f32", **kw):
        c = self.r.SnapshotComparer(self.tmp / name)
        for k, v in observed.items():
            c.compare(k, v, **kw)
        return c.summary()

    def test_identical_round_trip_is_exact_and_complete(self):
        meta, _ = self.write(self.ref)
        self.assertEqual((meta["tensors"], meta["elements"], meta["bytes"]), (3, 35 + 33 + 24, 4 * 92))
        s = self.compare(self.ref)
        self.assertTrue(s["coverage_complete"])
        self.assertEqual((s["global_rel_l2"], s["global_cosine"], s["global_norm_ratio"]), (0.0, 1.0, 1.0))
        self.assertEqual(s["compared_elements"], 92)
        with self.assertRaises(ValueError):                               # duplicate names are refused at write
            w = self.r.SnapshotWriter(self.tmp / "dup.f32")
            w.add("a", self.ref["a"])
            w.add("a", self.ref["a"])
        with self.assertRaises(FileExistsError):                           # records are never overwritten
            self.r.SnapshotWriter(self.tmp / "snap.f32")

    def test_tampered_reference_bytes_are_rejected(self):
        _, path = self.write(self.ref)
        raw = bytearray(path.read_bytes())
        raw[4 * 40] ^= 0x01                                                # one bit inside tensor "b"
        path.write_bytes(bytes(raw))
        c = self.r.SnapshotComparer(path)
        c.compare("a", self.ref["a"])                                      # untouched tensor still verifies
        with self.assertRaises(ValueError):
            c.compare("b", self.ref["b"])

    def test_missing_duplicate_unexpected_and_resized_tensors_break_coverage(self):
        self.write(self.ref)
        s = self.compare({"a": self.ref["a"], "b": self.ref["b"]})
        self.assertFalse(s["coverage_complete"])
        self.assertEqual(s["missing"], ["c"])
        c = self.r.SnapshotComparer(self.tmp / "snap.f32")
        for k in ("a", "b", "c", "a"):
            c.compare(k, self.ref[k])
        c.compare("lm_head.weight", self.ref["a"])                        # an alias compared again under its own name
        c2 = self.r.SnapshotComparer(self.tmp / "snap.f32")
        c2.compare("a", self.ref["a"])
        c2.compare("b", self.ref["b"][:-1])
        c2.compare("c", self.ref["c"])
        s, s2 = c.summary(), c2.summary()
        self.assertFalse(s["coverage_complete"])
        self.assertTrue(any("duplicate" in p for p in s["problems"]))
        self.assertTrue(any("unexpected" in p for p in s["problems"]))
        self.assertFalse(s2["coverage_complete"])
        self.assertTrue(any("elements" in p for p in s2["problems"]))

    def test_wrong_normalization_and_sign_are_caught(self):
        self.write(self.ref)
        s4 = self.compare({k: 4 * v for k, v in self.ref.items()})        # a missing 1/GA with GA = 4
        self.assertAlmostEqual(s4["global_rel_l2"], 3.0, places=12)
        self.assertAlmostEqual(s4["global_norm_ratio"], 4.0, places=12)
        self.assertAlmostEqual(s4["global_cosine"], 1.0, places=12)      # cosine alone cannot see a scale error
        s_q = self.compare({k: v / 4 for k, v in self.ref.items()})
        self.assertAlmostEqual(s_q["global_rel_l2"], 0.75, places=12)
        neg = self.compare({k: -v for k, v in self.ref.items()})
        self.assertAlmostEqual(neg["global_cosine"], -1.0, places=12)
        for cat in ("bf16_compute", "bf16_weights_and_accumulation"):
            lim = {"bf16_compute": (0.05, 0.15), "bf16_weights_and_accumulation": (0.08, 0.24)}[cat]
            self.assertEqual(self.r.grade(s4["global_rel_l2"], *lim), "fail")
            self.assertEqual(self.r.grade(s_q["global_rel_l2"], *lim), "fail")

    def test_zero_and_tiny_reference_norms(self):
        ref = {"z": torch.zeros(6), "t": torch.full((6,), 1e-30), "n": torch.ones(4)}
        self.write(ref)
        s = self.compare({"z": torch.zeros(6), "t": torch.full((6,), 1e-30) * (1 + 1e-3), "n": torch.ones(4)})
        per = {r["name"]: r for r in s["per_tensor"]}
        self.assertIsNone(per["z"]["rel_l2"])                              # zero denominator -> no relative value
        self.assertEqual(s["zero_reference_tensors"], ["z"])
        self.assertAlmostEqual(per["t"]["rel_l2"], 1e-3, delta=2e-6)      # float64 accumulation; no underflow
        self.write({"z": torch.zeros(3)}, "allzero.f32")
        s0 = self.compare({"z": torch.zeros(3)}, "allzero.f32")
        self.assertIsNone(s0["global_rel_l2"])
        self.assertEqual(self.r.grade(s0["global_rel_l2"], 0.05, 0.15), "missing")   # never a pass
        self.assertEqual(self.r.grade(float("nan"), 0.05, 0.15), "missing")

    def test_nonfinite_observed_values_fail_coverage(self):
        self.write(self.ref)
        bad = dict(self.ref, b=self.ref["b"].clone())
        bad["b"][3] = float("inf")
        s = self.compare(bad)
        self.assertFalse(s["coverage_complete"])
        self.assertTrue(any("non-finite" in p for p in s["problems"]))

    def test_chunked_statistics_equal_unchunked_and_witness_index(self):
        self.write(self.ref)
        obs = {k: v.clone() for k, v in self.ref.items()}
        obs["b"][29] += 0.5                                                # worst element in a later chunk
        whole = self.compare(obs)
        with mock.patch.object(self.r, "CHUNK", 7):
            chunked = self.compare(obs)
        for k in ("global_rel_l2", "global_cosine", "global_norm_ratio"):
            self.assertAlmostEqual(whole[k], chunked[k], places=13)
        self.assertEqual((chunked["max_abs_diff"]["name"], chunked["max_abs_diff"]["flat_index"]), ("b", 29))
        self.assertAlmostEqual(chunked["max_abs_diff"]["value"], 0.5, places=6)
        self.assertAlmostEqual(self.r.norm64(torch.arange(10.0)), math.sqrt(285.0))

    def test_baseline_difference_uses_the_writer_definition_and_catches_no_op(self):
        theta0 = {k: v.clone() for k, v in self.ref.items()}
        theta1 = {k: v + 1e-4 * torch.sign(v) + 1e-9 for k, v in self.ref.items()}
        self.write({k: theta1[k] - theta0[k] for k in theta0})            # writer: float32 subtraction
        c = self.r.SnapshotComparer(self.tmp / "snap.f32")
        for k in theta1:
            c.compare(k, theta1[k], baseline=theta0[k])
        s = c.summary()
        self.assertEqual(s["global_rel_l2"], 0.0)                          # identical definition -> exactly zero
        c = self.r.SnapshotComparer(self.tmp / "snap.f32")
        for k in theta0:
            c.compare(k, theta0[k], baseline=theta0[k])                    # no-op update
        s = c.summary()
        self.assertEqual(s["global_norm_ratio"], 0.0)
        self.assertIsNone(s["global_cosine"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class ReferenceSessionFixtures(unittest.TestCase):
    def setUp(self):
        self.r = mg("mgbench.train.reference")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        g = torch.Generator().manual_seed(1)
        self.names = ["w1", "w2", "b"]
        self.g = [{n: torch.randn(9, generator=g) for n in self.names} for _ in range(3)]
        self.theta0 = {n: torch.randn(9, generator=g) for n in self.names}
        self.dt = {n: 1e-3 * torch.randn(9, generator=g) for n in self.names}
        self.mv = {n: (torch.randn(9, generator=g), torch.rand(9, generator=g)) for n in self.names}
        self.losses = [11.9 + 0.01 * i for i in range(12)]

    def session(self, role, scale_g=(1, 1, 1), scale_m=1.0, scale_v=1.0, dt_scale=1.0, losses=None):
        s = self.r.ReferenceSession(role, self.tmp)
        for k in range(3):
            s.at_boundary(k, [(n, scale_g[k] * self.g[k][n]) for n in self.names])
        s.final([(n, self.theta0[n] + dt_scale * self.dt[n], scale_m * self.mv[n][0], scale_v * self.mv[n][1])
                 for n in self.names], self.theta0)
        return s.finish(losses if losses is not None else self.losses)

    def test_write_then_compare_identical_and_scaled(self):
        w = self.session("write")
        self.assertTrue(w["scalars_written"])
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()),
                         sorted(mg("mgbench.smoke").REFERENCE_FILES))
        c = self.session("compare")
        self.assertEqual(c["snapshots"]["grad_b1"]["global_rel_l2"], 0.0)
        self.assertEqual(c["snapshots"]["dtheta_final"]["global_rel_l2"], 0.0)
        self.assertEqual(max(c["loss_rel"]), 0.0)
        self.assertEqual([g["global_norm_ratio"] for g in c["grad_norm_ratios"]], [1.0, 1.0, 1.0])
        bad = self.session("compare", scale_g=(1, 4, 1), scale_m=4.0, scale_v=16.0, dt_scale=0.0,
                           losses=[x * 1.02 for x in self.losses])
        self.assertAlmostEqual(bad["grad_norm_ratios"][1]["global_norm_ratio"], 4.0)
        self.assertAlmostEqual(bad["moment_norm_ratios"]["global"]["m"], 4.0)
        self.assertAlmostEqual(bad["moment_norm_ratios"]["global"]["v"], 16.0)
        self.assertEqual(bad["snapshots"]["dtheta_final"]["global_norm_ratio"], 0.0)   # no-op update
        self.assertAlmostEqual(max(bad["loss_rel"]), 0.02, places=9)
        short = self.session("compare", losses=self.losses[:11])
        self.assertFalse(short["loss_count_match"])

    def test_scalars_are_written_once(self):
        self.session("write")
        with self.assertRaises(FileExistsError):
            self.session("write")


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class AdamWStateAndBiasCorrectionErrors(unittest.TestCase):
    def run_opt(self, cls):
        ue = mg("mgbench.train.update_evidence")
        torch.manual_seed(3)
        model = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.Linear(6, 3))
        opt = cls(model.parameters(), lr=1e-3, weight_decay=0.01)
        smp = ue.AdamWCallSampler(ue.build_plan(list(model.named_parameters())))
        opt.register_step_pre_hook(lambda o, a, k: smp.pre(o))
        opt.register_step_post_hook(lambda o, a, k: smp.post(o))
        for _ in range(2):
            model(torch.randn(4, 8)).pow(2).mean().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        return [c["verdict"] for c in smp.calls]

    @staticmethod
    def variant(bias_correction=True, v_scale=1.0):
        class Variant(torch.optim.AdamW):
            @torch.no_grad()
            def step(self, closure=None):
                for g in self.param_groups:
                    b1, b2 = g["betas"]
                    for p in g["params"]:
                        if p.grad is None:
                            continue
                        st = self.state[p]
                        if not st:
                            st.update(step=torch.tensor(0.0), exp_avg=torch.zeros_like(p),
                                      exp_avg_sq=torch.zeros_like(p))
                        st["step"] += 1
                        t = float(st["step"])
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                        st["exp_avg"].lerp_(p.grad, 1 - b1)
                        st["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=(1 - b2) * v_scale)
                        bc1, bc2 = (1 - b1 ** t, 1 - b2 ** t) if bias_correction else (1.0, 1.0)
                        denom = (st["exp_avg_sq"].sqrt() / math.sqrt(bc2)).add_(g["eps"])
                        p.addcdiv_(st["exp_avg"], denom, value=-g["lr"] / bc1)
        return Variant

    def test_discriminating_relative_error_is_reported_separately(self):
        ue = mg("mgbench.train.update_evidence")
        r = {"betas": (0.9, 0.999), "lr": 1e-4, "eps": 1e-8, "wd": 0.0, "step": 0.0,
             "theta": [1.0, 1.0], "grad": [1e-3, 0.0], "m": [0.0, 0.0], "v": [0.0, 0.0]}
        exp0 = 1.0 - 1e-4 * 1e-3 / (1e-3 + 1e-8)                           # discriminating sample (change ~1e-4)
        chk = ue.check_adamw(r, [exp0 + 1e-9, 1.0], [1e-4, 0.0], [1e-9, 0.0], 1.0)
        self.assertTrue(chk["pass"], chk.get("reason"))
        self.assertEqual(chk["discriminating"], 1)                          # the zero-gradient sample is not
        self.assertAlmostEqual(chk["max_rel_err_param_discriminating"], 1e-9 / abs(exp0 - 1.0), delta=1e-9)
        self.assertGreaterEqual(chk["max_rel_err_param"], chk["max_rel_err_param_discriminating"])
        none = ue.check_adamw(dict(r, grad=[0.0, 0.0]), [1.0, 1.0], [0.0] * 2, [0.0] * 2, 1.0)
        self.assertIsNone(none["max_rel_err_param_discriminating"])        # unavailable, never reported as 0

    def test_reimplemented_adamw_passes_and_errors_fail(self):
        self.assertEqual(self.run_opt(self.variant()), ["verified_sampled"] * 2)          # positive control
        self.assertEqual(self.run_opt(torch.optim.AdamW), ["verified_sampled"] * 2)
        self.assertIn("failed", self.run_opt(self.variant(bias_correction=False)))         # bias correction omitted
        self.assertIn("failed", self.run_opt(self.variant(v_scale=2.0)))                   # second-moment state error


def _gloo(testcase):
    if dist.is_initialized():
        raise unittest.SkipTest("a process group already exists in this interpreter")
    tmp = Path(tempfile.mkdtemp())
    dist.init_process_group("gloo", store=dist.FileStore(str(tmp / "store"), 1), rank=0, world_size=1)
    testcase.addCleanup(shutil.rmtree, tmp, True)
    testcase.addCleanup(dist.destroy_process_group)


def _tiny_entry(tmp: Path) -> dict:
    d = tmp / "tiny_qwen2"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(TINY))
    return {"kind": "causal_lm", "hf_config_dir": str(d)}


@unittest.skipUnless(HAVE_TORCH and HAVE_HF, "torch/transformers not installed")
class CanonicalStartAndDtypes(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        _gloo(self)
        self.diag = mg("mgbench.train.diagnostics")
        self.fw = mg("mgbench.train.frameworks")
        self.main = mg("mgbench.train.__main__")
        self.entry = _tiny_entry(self.tmp)

    def build(self, **spec):
        return self.diag.canonical_model(self.entry, dict({"seed": 1234}, **spec), self.fw._build)

    def test_representable_start_survives_bf16_casts_and_keeps_ties(self):
        plain, _ = self.build()
        model, rec = self.build(canonical_start="bf16_representable")
        changed = sum(int((p.to(torch.bfloat16).float() != p).sum()) for p in plain.parameters())
        self.assertGreater(changed, 0)                                     # a plain fp32 start is not representable
        for p in model.parameters():
            self.assertTrue(torch.equal(p, p.to(torch.bfloat16).float()))  # bf16 weights hold it exactly
            master = p.to(torch.bfloat16).to(torch.float32)               # DeepSpeed's fp32 master from bf16
            self.assertTrue(torch.equal(master, p))
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        bufs = dict(model.named_buffers())
        for n, b in plain.named_buffers():
            self.assertTrue(torch.equal(bufs[n], b), n)                    # buffers are not rounded
        self.assertEqual(rec["canonical_start"], "bf16_representable")
        theta0 = self.fw.canonical_theta0(model, {"reference_role": "write"})
        self.assertNotIn("lm_head.weight", theta0)                         # tied alias held once
        self.assertEqual(len(theta0), len(list(model.parameters())))

    def test_wrong_initialization_is_detected(self):
        model, rec = self.build(canonical_start="bf16_representable")
        spec = {"framework": "ddp"}
        good = self.diag.gathered_state_fingerprints("ddp", model, deepspeed=False)
        self.assertTrue(self.main._compare_canonical(rec, good, spec)["module_state_matches_canonical"])
        other, _ = self.diag.canonical_model(self.entry, {"seed": 99, "canonical_start": "bf16_representable"},
                                             self.fw._build)
        wrong = self.diag.gathered_state_fingerprints("ddp", other, deepspeed=False)
        self.assertFalse(self.main._compare_canonical(rec, wrong, spec)["module_state_matches_canonical"])


@unittest.skipUnless(HAVE_TORCH and HAVE_HF and HAVE_YAML, "torch/transformers/yaml not installed")
class TinyEndToEnd(unittest.TestCase):
    """Production DDP and FSDP2 trainers on CPU (gloo, world size 1) with the full S1a reference path."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        _gloo(self)
        import torch.nn.parallel as tnp

        base = tnp.DistributedDataParallel

        class CpuDDP(base):                                                # CPU modules take no device_ids
            def __init__(self, module, device_ids=None, **kw):
                super().__init__(module, device_ids=None, **kw)
        p = mock.patch.object(tnp, "DistributedDataParallel", CpuDDP)
        p.start()
        self.addCleanup(p.stop)
        self.fw = mg("mgbench.train.frameworks")
        self.main = mg("mgbench.train.__main__")
        self.cfg = mg("mgbench.config")
        self.data = mg("mgbench.data")
        self.smoke = mg("mgbench.smoke")
        self.entry = _tiny_entry(self.tmp)
        self.ref = self.tmp / "reference"
        self.ref.mkdir()

    def run_case(self, framework, role, fp32):
        spec = dict(model="tiny", seq_len=16, micro_batch=1, grad_accum=4, seed=1234, warmup_steps=0,
                    measure_steps=3, numerical_checks=True, canonical_init=True, canonical_start="bf16_representable",
                    global_inputs=True, fp32_reference=fp32, reference_role=role, reference_dir=str(self.ref),
                    framework=framework, n_gpus=1)
        tr = self.fw.SETUP[framework](spec, self.entry, self.cfg.resolved_framework_config(spec), torch.device("cpu"))
        canon = self.main._compare_canonical(tr.canonical, tr.state_fingerprint(), spec)
        data = self.data.GlobalSynthetic("causal_lm", micro_batch=1, grad_accum=4, world=1, rank=0,
                                         device=torch.device("cpu"), seed=1234000, steps=3, vocab_size=128,
                                         seq_len=16, image_size=None, num_classes=None, dtype=torch.float32)
        tr.ledger.start_measured()
        for _ in range(3):
            tr.step([data.next() for _ in range(4)])
        summ = tr.ledger.summary(3, 0, 4)
        rc, fc = self.main.finalize_reference(tr, summ["measured"])
        return tr, summ, rc, canon, data

    def test_control_writes_full_snapshots_and_identical_recipe_compares_exactly(self):
        tr, summ, rc, canon, data = self.run_case("ddp", "write", True)
        n_t = summ["layout"]["unit_map"]["unique_trainable"]
        self.assertTrue(canon["module_state_matches_canonical"])
        m = summ["measured"]
        self.assertEqual((m["logical_attempts"], m["microbatches"], m["counter_updates"]), (3, 12, 3))
        self.assertEqual(m["boundary_coverage"], ["complete"] * 3)
        self.assertEqual(m["update_evidence_verdicts"], ["verified_sampled"] * 3)
        self.assertEqual(summ["warmup"]["logical_attempts"], 0)
        self.assertTrue(summ["warmup"]["microbatch_losses_all_finite"])  # nothing expected, nothing missing
        self.assertEqual(rc["snapshots"]["grad_b1"]["tensors"], n_t)
        self.assertEqual(summ["layout"]["tied_aliases"], {"model.embed_tokens.weight": ["lm_head.weight"]})
        names = [e["name"] for e in json.loads((self.ref / "grad_b1.f32.index.json").read_text())["index"]]
        self.assertNotIn("lm_head.weight", names)                          # tied alias written once
        self.assertEqual(len(data.input_digests["per_step_microbatch_sha256"]), 3)
        self.assertIsNotNone(tr.update_audit["optimizer_branch"]["branch"])
        receipt = self.smoke.reference_receipt(self.ref, 10**9, expected_tensors=n_t,
                                               expected_elements=rc["snapshots"]["grad_b1"]["elements"])
        self.assertTrue(receipt["valid"], receipt["problems"])
        # identical recipe against itself: exactly zero difference everywhere (positive control of the machinery)
        _, s2, same, canon2, _ = self.run_case("ddp", "compare", True)
        self.assertTrue(canon2["module_state_matches_canonical"])
        for snap in ("grad_b1", "dtheta_final"):
            self.assertTrue(same["snapshots"][snap]["coverage_complete"])
            self.assertEqual(same["snapshots"][snap]["global_rel_l2"], 0.0)
        self.assertEqual(max(same["loss_rel"]), 0.0)
        self.assertEqual(same["moment_norm_ratios"]["global"], {"m": 1.0, "v": 1.0})
        # FSDP2 with bf16 parameters (CPU): complete coverage, a small but non-zero difference
        _, s3, f, canon3, _ = self.run_case("fsdp", "compare", False)
        self.assertTrue(canon3["module_state_matches_canonical"])
        g1 = f["snapshots"]["grad_b1"]
        self.assertTrue(g1["coverage_complete"] and f["snapshots"]["dtheta_final"]["coverage_complete"])
        self.assertGreater(g1["global_rel_l2"], 0.0)
        self.assertLess(g1["global_rel_l2"], 0.15)
        # receipts and rechecks
        self.assertFalse(self.smoke.reference_receipt(self.ref, 1000, expected_tensors=n_t,
                                                      expected_elements=g1["expected_elements"])["valid"])
        self.assertTrue(self.smoke.reference_recheck(self.ref, None))
        self.assertEqual(self.smoke.reference_recheck(self.ref, receipt), [])
        idx = self.ref / "grad_b1.f32.index.json"
        text = idx.read_text()
        sha = json.loads(text)["index"][0]["sha256"]
        idx.write_text(text.replace(sha, sha[::-1], 1))                    # same size, different content
        self.assertTrue(any("changed" in p for p in self.smoke.reference_recheck(self.ref, receipt)))
        idx.write_text(text + " ")
        self.assertTrue(any("resized" in p for p in self.smoke.reference_recheck(self.ref, receipt)))
        (self.ref / "stray.bin").write_bytes(b"x")
        self.assertFalse(self.smoke.reference_receipt(self.ref, 10**9, expected_tensors=n_t,
                                                      expected_elements=g1["expected_elements"])["valid"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class DeepSpeedSliceMapping(unittest.TestCase):
    def setUp(self):
        self.fw = mg("mgbench.train.frameworks")

    def fake(self, shift=0, offsets=True):
        class DeepSpeedZeroOptimizer:
            pass
        emb = torch.nn.Embedding(5, 3, dtype=torch.bfloat16)
        head = torch.nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
        head.weight = emb.weight                                           # tied
        norm = torch.nn.Linear(1, 4, bias=False, dtype=torch.bfloat16)
        module = torch.nn.ModuleDict({"emb": emb, "head": head, "norm": norm})
        params = [module["emb"].weight, module["norm"].weight]             # DeepSpeed holds the tied tensor once
        n = sum(p.numel() for p in params)
        flat = torch.zeros(n + 1, dtype=torch.bfloat16)                    # one alignment element at the end
        off = 0
        for p in params:
            flat[off:off + p.numel()].copy_(p.detach().reshape(-1))
            p.data = flat[off + shift:off + shift + p.numel()].view(p.shape)
            off += p.numel()
        zo = DeepSpeedZeroOptimizer()
        zo.bit16_groups, zo.bit16_groups_flat, zo.partition_size = [params], [flat], [n + 1]
        if offsets:
            zo.round_robin_bit16_groups, zo.round_robin_bit16_offsets = [params], [[0, params[0].numel()]]
        zo.nccl_start_alignment_factor = 2
        master = flat.float()
        zo.single_partition_of_fp32_groups = [master]
        zo.real_dp_process_group = [None]
        opt = types.SimpleNamespace(param_groups=[{"params": [master]}])
        engine = types.SimpleNamespace(optimizer=zo, module=module)
        with mock.patch.object(self.fw.dist, "get_rank", lambda group=None: 0), \
                mock.patch.object(self.fw.dist, "get_world_size", lambda group=None: 1):
            return self.fw.deepspeed_zero12_layout(engine, opt), module, master

    def test_slices_follow_flattening_order_and_are_storage_verified(self):
        lay, module, master = self.fake()
        self.assertTrue(lay["param_slices_storage_verified"])
        self.assertEqual([s[0] for s in lay["param_slices"]], ["emb.weight", "norm.weight"])   # first names only
        named = dict(module.named_parameters())
        for name, view in self.fw.flat_slices(master, lay["param_slices"]):
            self.assertTrue(torch.equal(view, named[name].float()))
        ok = self.fw.module_vs_master_full(master, module, lay["param_slices"])
        self.assertTrue(ok["pass"])
        self.assertEqual(ok["compared"], 15 + 4)
        bumped = master.clone()
        bumped[16] += 1.0                                                  # one element of norm.weight
        bad = self.fw.module_vs_master_full(bumped, module, lay["param_slices"])
        self.assertEqual((bad["mismatched"], bad["first_mismatched_tensor"], bad["pass"]), (1, "norm.weight", False))

    def test_misplaced_view_is_not_verified(self):
        lay, _, _ = self.fake(shift=1)
        self.assertFalse(lay["param_slices_storage_verified"])
        self.assertTrue(self.fake(offsets=False)[0]["param_slices_storage_verified"])   # cumulative fallback


@unittest.skipUnless(HAVE_YAML, "yaml not installed")
class DiagnosticsConfigBudgetAndAuthorization(unittest.TestCase):
    def setUp(self):
        self.smoke = mg("mgbench.smoke")
        self.lc = mg("mgbench.launch_controls")
        self.cfg = yaml.safe_load((A6000 / "configs/s1a_qwen15b.yaml").read_text())
        # Candidate 14: admission recomputes the cushioned fraction, so the fixture carries predictions; a4's is a
        # hypothetical 42.5 GiB that fits 0.92 WITH the cushion (the sealed S1a value 43.48 does not - see C17).
        pred = {"r0_fp32_control": 41.96, "a1_ddp_bf16_autocast": 38.66, "a2_fsdp2_bf16": 35.55, "a3_zero1_bf16": 41.71,
                "a4_zero2_bf16": 42.5}
        self.feas = {"torch_total_memory_gib": 47.402, "non_allocator_gib": 0.8,
                     "cases": {c["id"]: {"eligible": True, "fraction": c["mem_fraction"],
                                         "predicted_reserved_gib": pred[c["id"]]} for c in self.cfg["cases"]}}
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_frozen_config_is_accepted_and_budget_is_2390_seconds(self):
        self.assertEqual(self.smoke.diag_config_problems(self.cfg, self.feas), [])
        self.assertEqual(self.smoke.diag_budget_seconds(self.cfg["timing"], 5), 2390)
        t = self.cfg["timing"]
        self.assertTrue(self.smoke.case_budget(0, 430 + 120, t)["fits"])
        self.assertFalse(self.smoke.case_budget(0, 430 + 119, t)["fits"])
        self.assertGreater(self.smoke.diag_budget_seconds(t, 6), t["window_s"])           # a sixth case cannot fit
        if (DESIGN / "memory_time_feasibility_v2.json").exists():
            real = json.loads((DESIGN / "memory_time_feasibility_v2.json").read_text())
            probs = self.smoke.diag_config_problems(self.cfg, real)       # C17: the sealed a4 admission is refused now
            self.assertEqual(len(probs), 1, probs)
            self.assertIn("a4_zero2_bf16", probs[0])

    def test_refusals(self):
        muts = [lambda c: c.update(scope="train-smoke"), lambda c: c.update(allowed_gpus=[4, 5]),
                lambda c: c.update(gpu_uuids={4: "GPU-bad"}), lambda c: c.update(gpu_pci={5: "x"}),
                lambda c: c["cases"].append(dict(c["cases"][1], id="a5_extra")),
                lambda c: c.update(max_cases=6), lambda c: c["cases"].reverse(),
                lambda c: c["cases"][4].update(mem_fraction=0.93), lambda c: c["cases"][4].update(mem_fraction=True),
                lambda c: c["cases"][1].update(mem_fraction=0.85),          # differs from the frozen feasibility value
                lambda c: c["cases"][0].update(reference_role="compare"),
                lambda c: c["cases"][0].update(fp32_reference=False),
                lambda c: c["cases"][2].pop("depends_on"), lambda c: c["cases"][3].update(fp32_reference=True),
                lambda c: c["cases"][2].update(framework="ddp"),
                lambda c: c["cases"][1].update(diagnostic_fault={"kind": "nonfinite_loss"}),
                lambda c: c["common"].update(grad_accum=2), lambda c: c["common"].update(warmup_steps=1),
                lambda c: c["common"].update(activation_checkpointing=True),
                lambda c: c["common"].update(model="llama-3.1-8b"), lambda c: c["common"].pop("canonical_start"),
                lambda c: c["timing"].update(child_s=301), lambda c: c["timing"].update(window_s=2701),
                lambda c: c["limits"].update(worker_cpu_quota_pct=1600),
                lambda c: c["limits"].update(worker_memory_bytes=47 * 2**30),
                lambda c: c.update(reference_budget_bytes=14 * 10**9), lambda c: c.update(min_free_disk_bytes=10**9)]
        for i, mutate in enumerate(muts):
            c = copy.deepcopy(self.cfg)
            mutate(c)
            self.assertTrue(self.smoke.diag_config_problems(c, self.feas), i)
        feas = copy.deepcopy(self.feas)
        feas["cases"]["a4_zero2_bf16"]["eligible"] = False                 # an infeasible case is not launched
        self.assertTrue(self.smoke.diag_config_problems(self.cfg, feas))

    def test_case_specs_expand_to_the_frozen_recipe_on_gpu4(self):
        cfgmod = mg("mgbench.config")
        cluster = dict(cfgmod.cluster_config(), allowed_gpus=[4], gpu_sets={1: [4]})
        specs = []
        for case in self.cfg["cases"]:
            out = cfgmod.expand_experiment(self.smoke.diag_experiment(case, self.cfg, self.tmp / "reference"), cluster)
            self.assertEqual(len(out), 1)
            specs.append(out[0])
        for case, spec in zip(self.cfg["cases"], specs):
            self.assertEqual(spec["gpus"], [4])
            self.assertEqual((spec["framework"], spec["mem_fraction"], spec["reference_role"]),
                             (case["framework"], case["mem_fraction"], case["reference_role"]))
            self.assertEqual(spec["fp32_reference"], case["id"] == "r0_fp32_control")
            for k, v in self.smoke.DIAG_COMMON.items():
                self.assertEqual(spec[k], v, k)
            self.assertEqual(spec["reference_dir"], str(self.tmp / "reference"))
        self.assertEqual(len({s["run_id"] for s in specs}), 5)

    def auth(self, **kw):
        cand = self.tmp / "cand"
        (cand / "studies/a6000/src").mkdir(parents=True, exist_ok=True)
        (cand / "studies/a6000/src/x.py").write_text("a = 1\n")
        now = dt.datetime.now(dt.timezone.utc)
        a = {"id": "s1a-test", "scope": "train-diagnostics", "candidate_root": str(cand),
             "candidate_digest": self.lc.candidate_digest(cand), "gpu_index": 4, "gpu_uuid": UUID4, "max_gpus": 1,
             "max_concurrent_jobs": 1, "max_cases": 5, "issued_utc": (now - dt.timedelta(minutes=1)).isoformat(),
             "expires_utc": (now + dt.timedelta(seconds=2700 - 60)).isoformat(), "nonrenewable": True}
        a.update(kw)
        p = self.tmp / f"auth_{len(list(self.tmp.glob('auth_*')))}.json"
        self.lc.create_exclusive(p, a)
        return p, cand, a

    def test_five_case_authorization_expiry_and_refusals(self):
        p, cand, a = self.auth()
        prob = lambda path=p, **kw: self.lc.authorization_problems(
            path, **dict(dict(scope="train-diagnostics", candidate_root=cand, gpus=[4], uuids={4: UUID4}), **kw))
        for done in range(5):
            self.assertEqual(prob(n_cases_done=done), [], done)
        self.assertTrue(any("ceiling" in x for x in prob(n_cases_done=5)))           # no sixth case
        late = dt.datetime.fromisoformat(a["expires_utc"]) + dt.timedelta(seconds=1)
        self.assertTrue(any("not valid now" in x for x in prob(now=late)))            # expiry without a controller
        self.assertTrue(prob(scope="train-smoke"))                                      # earlier scopes do not carry
        self.assertTrue(prob(gpus=[5]))
        p2, _, _ = self.auth(scope="train-smoke")
        self.assertTrue(prob(path=p2))                                                  # smoke authorization refused
        self.assertTrue(self.lc.revoke(p, "test"))
        self.assertTrue(any("revoked" in x for x in prob()))

    def test_bound_file_changes_refuse(self):
        f = self.tmp / "tolerances.json"
        f.write_text("{}")
        bound = {str(f): hashlib.sha256(b"{}").hexdigest()}
        self.assertEqual(self.smoke.bound_file_problems(bound), [])
        f.write_text("{ }")
        problems = self.smoke.bound_file_problems(bound)
        self.assertTrue(problems)
        self.assertEqual(self.smoke.classify(problems)[1], problems)                   # fatal, never "busy"
        self.assertTrue(self.smoke.bound_file_problems({}))


def good_diag_result(case, n_t=338, n_e=1_543_714_304):
    plan = [[f"{s}{k}" for k in range(4)] for s in range(3)]
    fw = case["framework"]
    dev = lambda st: {"stage": st, "device_count": 1, "current_device": 0, "uuid": UUID4, "pci_bus_id": "0000:81:00.0"}
    steps = [{"module_bf16_vs_master": {"pass": True}}] * 3
    led = {"warmup": {"logical_attempts": 0, "microbatches": 0, "microbatch_losses_all_finite": True},
           "measured": {"logical_attempts": 3, "microbatches": 12, "microbatch_losses_all_finite": True,
                        "counter_updates": 3, "framework_reported_skips": 0, "base_call_multiplicity_matches": True,
                        "boundary_coverage": ["complete"] * 3, "sample_plan_complete_each_boundary": True,
                        "discriminating_samples_per_call": [150, 150, 150], "steps": steps},
           "layout": {"problems": [], "unit_map": {"unique_trainable": n_t}, "sample_plan": [{"name": "x"}],
                      "model_to_partition_map": [{}]}}
    fp32 = bool(case.get("fp32_reference"))
    snap_w = {"tensors": n_t, "elements": n_e, "bytes": 4 * n_e, "data_sha256": "0" * 64}
    snap_c = lambda rel, cos, ratio: {"coverage_complete": True, "compared_tensors": n_t, "compared_elements": n_e,
                                      "problems": [], "global_rel_l2": rel, "global_cosine": cos,
                                      "global_norm_ratio": ratio, "worst_tensor_rel_l2": {"name": "w", "rel_l2": rel},
                                      "max_abs_diff": {"value": 1e-3}, "zero_reference_tensors": []}
    rc = ({"role": "write", "snapshots": {"grad_b1": snap_w, "dtheta_final": snap_w}, "scalars_written": True} if
          case.get("reference_role") == "write" else
          {"role": "compare", "snapshots": {"grad_b1": snap_c(0.02, 0.9998, 1.0), "dtheta_final": snap_c(0.2, 0.98, 1.0)},
           "loss_rel": [1e-3] * 12, "loss_count_match": True,
           "grad_norm_ratios": [{"boundary": k, "tensors": n_t, "missing": [], "global_norm_ratio": 1.0} for k in range(3)],
           "moment_norm_ratios": {"global": {"m": 1.0, "v": 1.0}, "tensors": n_t, "missing": [], "min": 0.9, "max": 1.1}})
    return {"status": "ok", "assessment": {"execution": "ok", "gradient": {"status": "complete_at_every_measured_boundary"},
                                           "update_state": {"status": "verified_sampled_every_base_call"},
                                           "measurement_eligibility": "ineligible: diagnostic instrumentation inside the timed path"},
            "runtime_devices": [dev("after set_device"), dev("after set_per_process_memory_fraction"), dev("end of run")],
            "process": {"mgbench_file": str(SRC / "mgbench/__init__.py"), "sys_executable": "/py",
                        "cgroup": {"cpu.max": "1500000 100000", "memory.max": "49392123904"}},
            "mem_fraction": case["mem_fraction"],
            "identity_chain": {"checked": True, "problems": [], "driver": {"uuid": UUID4},
                               "deepspeed": {"logical": 0, "nvml_index": 4, "expected_index": 4, "uuid": UUID4}},
            "optimizer_update_audit_per_rank": [{"boundary_ledger": led, "optimizer_branch": {"branch": "foreach"},
                                                 "deepspeed_layout": {"optimizer_class": "x.DeepSpeedZeroOptimizer",
                                                                      "accounting": {"group0": n_e},
                                                                      "model_to_partition_map": [{}]}}],
            "canonical_start": "bf16_representable",
            "canonical_state_check": {"module_state_matches_canonical": True,
                                      **({"optimizer_master_matches_canonical": True} if fw.startswith("zero") else {})},
            "input_digests": {"per_step_microbatch_sha256": plan},
            "precision": {"legacy": {"cuda_matmul_allow_tf32": not fp32, "cudnn_allow_tf32": not fp32,
                                     "float32_matmul_precision": "highest" if fp32 else "high",
                                     "cudnn_deterministic": fp32, "cudnn_benchmark": not fp32},
                          "sdpa_enabled": {"flash": not fp32, "mem_efficient": not fp32, "math": True, "cudnn": not fp32},
                          "env_overrides": {"NVIDIA_TF32_OVERRIDE": None}, "autocast_bf16": fw == "ddp" and not fp32},
            "reference_comparison": rc,
            "final_checks": ({"module_bf16_vs_master_full": {"pass": True, "compared": n_e}} if fw.startswith("zero")
                             else None)}


@unittest.skipUnless(HAVE_YAML, "yaml not installed")
class DiagnosticsEvaluation(unittest.TestCase):
    def setUp(self):
        self.smoke = mg("mgbench.smoke")
        self.cases = {c["id"]: c for c in yaml.safe_load((A6000 / "configs/s1a_qwen15b.yaml").read_text())["cases"]}
        tol_path = DESIGN / "tolerances.json"
        self.tol = json.loads(tol_path.read_text()) if tol_path.exists() else None
        if self.tol is None:
            self.skipTest("frozen tolerances are on the node only")
        self.frozen = {"warmup_attempts": 0, "measured_attempts": 3, "grad_accum": 4,
                       "eligibility_prefix": "ineligible: diagnostic",
                       "input_per_step_microbatch_sha256": [[f"{s}{k}" for k in range(4)] for s in range(3)],
                       "schema": {"unique_trainable_tensors": 338, "trainable_elements": 1_543_714_304}}

    def ev(self, cid, fn=None, rc=0):
        r = good_diag_result(self.cases[cid])
        if fn:
            fn(r)
        return self.smoke.evaluate_diag_case(r, rc, self.cases[cid], self.frozen, self.tol, candidate_src=SRC,
                                             interpreter="/py", uuid=UUID4, worker_cpu_max="1500000 100000",
                                             worker_memory_max="49392123904")

    def test_expected_results_pass_and_grade(self):
        e = self.ev("r0_fp32_control")
        self.assertTrue(e["pass_"], [c for c in e["integrity_checks"] if not c["pass"]])
        self.assertEqual(e["overall_grade"], "reference")
        for cid in ("a1_ddp_bf16_autocast", "a2_fsdp2_bf16", "a3_zero1_bf16", "a4_zero2_bf16"):
            e = self.ev(cid)
            self.assertTrue(e["pass_"], (cid, [c for c in e["integrity_checks"] + e["gross_checks"] if not c["pass"]]))
            self.assertEqual(e["overall_grade"], "pass", cid)

    def test_graded_fail_is_recorded_without_failing_integrity(self):
        e = self.ev("a1_ddp_bf16_autocast", lambda r: r["reference_comparison"]["snapshots"]["grad_b1"].update(
            global_rel_l2=0.2))
        self.assertTrue(e["pass_"])
        self.assertEqual(e["grades"]["grad_b1_global_rel_l2"]["grade"], "fail")
        self.assertEqual(e["overall_grade"], "fail")
        e = self.ev("a3_zero1_bf16", lambda r: r["reference_comparison"]["snapshots"]["grad_b1"].update(
            global_rel_l2=0.2))
        self.assertEqual(e["grades"]["grad_b1_global_rel_l2"]["grade"], "marginal")   # looser bf16-weight limits

    def test_integrity_and_gross_failures_stop(self):
        rcm = lambda r: r["reference_comparison"]
        bad = {
            "r0_fp32_control": [lambda r: r["precision"]["legacy"].update(cuda_matmul_allow_tf32=True),
                                lambda r: r["precision"]["sdpa_enabled"].update(flash=True),
                                lambda r: r["precision"]["env_overrides"].update(NVIDIA_TF32_OVERRIDE="1"),
                                lambda r: rcm(r)["snapshots"]["grad_b1"].update(tensors=337),
                                lambda r: r["canonical_state_check"].update(module_state_matches_canonical=False),
                                lambda r: r["input_digests"]["per_step_microbatch_sha256"][2].reverse(),
                                lambda r: r["runtime_devices"][2].update(uuid="GPU-other"),
                                lambda r: r["runtime_devices"][1].update(device_count=2),
                                lambda r: r.update(mem_fraction=0.92)],
            "a2_fsdp2_bf16": [lambda r: rcm(r)["snapshots"]["grad_b1"].update(coverage_complete=False),
                              lambda r: rcm(r)["snapshots"]["dtheta_final"].update(global_norm_ratio=0.0),
                              lambda r: rcm(r)["snapshots"]["dtheta_final"].update(global_cosine=0.1),
                              lambda r: rcm(r)["grad_norm_ratios"][2].update(global_norm_ratio=4.0),
                              lambda r: rcm(r)["moment_norm_ratios"]["global"].update(v=16.0),
                              lambda r: rcm(r)["moment_norm_ratios"]["global"].update(m=float("nan")),
                              lambda r: rcm(r).update(loss_rel=[1e-3] * 11),
                              lambda r: r["precision"].update(autocast_bf16=True)],
            "a4_zero2_bf16": [lambda r: r["final_checks"]["module_bf16_vs_master_full"].update({"pass": False}),
                              lambda r: r["canonical_state_check"].update(optimizer_master_matches_canonical=False),
                              lambda r: r["identity_chain"]["deepspeed"].update(nvml_index=5)]}
        for cid, muts in bad.items():
            for i, fn in enumerate(muts):
                self.assertFalse(self.ev(cid, fn)["pass_"], (cid, i))
        self.assertFalse(self.ev("a1_ddp_bf16_autocast", rc=1)["pass_"])
        none = self.smoke.evaluate_diag_case(None, 1, self.cases["a1_ddp_bf16_autocast"], self.frozen, self.tol,
                                             candidate_src=SRC, interpreter="/py", uuid=UUID4,
                                             worker_cpu_max="1500000 100000", worker_memory_max="49392123904")
        self.assertFalse(none["pass_"])

    def test_identity_receipts_are_separated_by_stage(self):
        names = [c["check"] for c in self.ev("a1_ddp_bf16_autocast")["integrity_checks"]]
        for s in ("pre-allocation torch identity receipt", "identity recheck after the allocator cap",
                  "identity recheck at the end of the run"):
            self.assertTrue(any(n.startswith(s) for n in names), s)


class SupervisorSource(unittest.TestCase):
    def test_generalized_supervisor_keeps_scope_cleanup_and_blocks_dependents(self):
        sup = (A6000 / "scripts/smoke_supervisor.py").read_text()
        self.assertIn("scopes.cleanup_scope", sup)
        self.assertNotIn("killpg", sup)
        self.assertIn('"blocked_invalid_reference" if case.get("depends_on") and control_failed', sup)
        self.assertIn("smoke.reference_recheck(ref_dir", sup)
        self.assertIn('default="configs/smoke_gpu4.yaml"', sup)            # smoke behaviour unchanged by default
        self.assertIn('smoke.active_units("mgbench-", ctx.get("meta_s", 10.0))', sup)   # one job at a time across both
        # prefixes (Candidate 18: the listing is bounded by the timing model's metadata-call limit)


if __name__ == "__main__":
    unittest.main()
