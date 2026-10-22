"""CPU fixture tests for the Candidate 15 first-boundary summarizer. Fixtures are written independently with numpy +
hashlib (not the product writer); no torch, GPU or cache file is involved."""
import copy
import hashlib
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "studies/a6000/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from mgbench import fb_summary as fs  # noqa: E402

DIG = [["d0", "d1", "d2", "d3"], ["e0", "e1", "e2", "e3"]]
RECIPE = {"model": "tiny", "seq_len": 16, "micro_batch": 1, "grad_accum": 4}


def write_snap(path, tensors, shapes=None, tamper=None):
    """Independent writer: little-endian float32 bytes + index (name, shape, numel, offset, sha256)."""
    idx, off, h, raw_all = [], 0, hashlib.sha256(), b""
    for name, arr in tensors.items():
        a = np.asarray(arr, dtype="<f4").reshape(-1)
        raw = a.tobytes()
        idx.append({"name": name, "shape": list((shapes or {}).get(name, a.shape)), "numel": int(a.size),
                    "offset": off, "sha256": hashlib.sha256(raw).hexdigest()})
        raw_all += raw
        off += len(raw)
    if tamper is not None:
        raw_all = bytearray(raw_all); raw_all[tamper] ^= 1; raw_all = bytes(raw_all)
    Path(path).write_bytes(raw_all)
    Path(str(path) + ".index.json").write_text(json.dumps({"bytes": off, "elements": off // 4, "tensors": len(idx),
                                                           "data_sha256": hashlib.sha256(raw_all).hexdigest(),
                                                           "index": idx}))


def good_result(framework="ddp", **over):
    steps = [{"counter_source": "first-boundary result kind: optimizer intentionally not called"}]
    if framework.startswith("zero"):
        steps[0].update(engine_global_steps=0, engine_micro_steps=3, boundary_flags_before_backward=[False, False, False, True])
    r = {"result_kind": "first_boundary_gradient", "parameters_unchanged": True,
         "assessment": {"execution": "ok", "gradient": {"status": "complete_at_the_boundary"},
                        "update_state": {"status": "not_applicable"}, "parameters_unchanged": True},
         "boundary_ledger": {"warmup": {"logical_attempts": 0},
                             "measured": {"logical_attempts": 1, "microbatches": 4, "base_optimizer_calls": 0,
                                          "counter_updates": 0, "microbatch_losses": [12.0, 12.1, 12.2, 12.3], "steps": steps}},
         "update_audit": {"optimizer_calls": 0}, "input_digests": {"per_step_microbatch_sha256": DIG[:1], "attention_mask": None},
         "canonical_state_check": {"module_state_matches_canonical": True, "optimizer_master_matches_canonical": True},
         "spec": dict(RECIPE, framework=framework)}
    r.update(over)
    return r


class Summarizer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        rng = np.random.default_rng(0)
        self.ref = {"model.embed_tokens.weight": rng.normal(size=(7, 3)), "model.norm.weight": rng.normal(size=5),
                    "model.layers.0.zero_bias": np.zeros(4)}
        self.rdir = self.tmp / "ref"
        self.rdir.mkdir()
        write_snap(self.rdir / "grad_b1.f32", self.ref)
        (self.rdir / "reference_scalars.json").write_text(json.dumps({"microbatch_losses_measured": [12.0, 12.1, 12.2, 12.3, 9, 9]}))
        self.frozen = {"input_per_step_microbatch_sha256": DIG, "recipe": RECIPE,
                       "tied_aliases": {"model.embed_tokens.weight": ["lm_head.weight"]}}
        self.n = 0

    def window(self, cases):
        self.n += 1
        w = self.tmp / f"w{self.n}"
        for i, (cid, (tensors, result, kw)) in enumerate(cases.items(), 1):
            d = w / "cases" / f"{i:02d}_{cid}"
            d.mkdir(parents=True)
            if tensors is not None:
                write_snap(d / "first_boundary_grad.f32", tensors, **kw)
            if result is not None:
                (d / "result.json").write_text(json.dumps(result))
        return w

    def run_case(self, tensors, result=None, chunk=3, **kw):
        w = self.window({"G0_fp32_control": (tensors, good_result() if result is None else result, kw)})
        before = {p: p.read_bytes() for p in list(w.rglob("*")) + list(self.rdir.rglob("*")) if p.is_file()}
        s = fs.summarize(w, self.rdir, self.tmp / f"out{self.n}", self.frozen, chunk=chunk)
        after = {p: p.read_bytes() for p in before}
        self.assertEqual(before, after)                                      # inputs never modified
        return s["cases"]["G0_fp32_control"]

    def test_identical_inputs_and_chunking(self):
        for chunk in (1, 3, 1 << 20):
            c = self.run_case(dict(self.ref), chunk=chunk)
            self.assertEqual(c["status"], "summarized", c["record_problems"])
            g = c["vs_sealed_r0"]["global"]
            self.assertEqual((g["rel_l2"], g["norm_ratio"]), (0.0, 1.0))
            self.assertAlmostEqual(g["cosine"], 1.0, places=6)               # float32 data, float64 sums
            self.assertEqual(c["update_verification"], "not_applicable")
            self.assertEqual(c["vs_sealed_r0"]["zero_reference_tensors"], ["model.layers.0.zero_bias"])   # legit zero
            self.assertEqual(c["raw_loss_differences"][0]["abs"], 0.0)

    def test_known_perturbation_and_wrong_normalization(self):
        t = {k: np.array(v, dtype=np.float32) for k, v in self.ref.items()}
        p = dict(t); p["model.norm.weight"] = t["model.norm.weight"] + np.float32(0.5) * np.eye(5, dtype=np.float32)[2]
        c = self.run_case(p)
        ref2 = sum(float((v.astype(np.float64) ** 2).sum()) for v in t.values())
        self.assertAlmostEqual(c["vs_sealed_r0"]["global"]["rel_l2"], 0.5 / math.sqrt(ref2), places=6)
        w = c["vs_sealed_r0"]["largest_abs_difference"]
        self.assertEqual((w["name"], w["flat_index"]), ("model.norm.weight", 2))
        c4 = self.run_case({k: 4 * v for k, v in t.items()})                 # missing 1/GA
        g = c4["vs_sealed_r0"]["global"]
        self.assertAlmostEqual(g["rel_l2"], 3.0, places=6)
        self.assertAlmostEqual(g["norm_ratio"], 4.0, places=6)
        self.assertLess(g["identity_check_abs"], 1e-9)                         # e^2 = r^2 + 1 - 2rc

    def test_missing_duplicate_extra_alias_shape_refuse(self):
        t = dict(self.ref)
        self.assertNotEqual(self.run_case({k: v for k, v in t.items() if k != "model.norm.weight"})["status"], "summarized")
        self.assertNotEqual(self.run_case(dict(t, extra=np.ones(2)))["status"], "summarized")
        self.assertNotEqual(self.run_case(dict(t, **{"lm_head.weight": t["model.embed_tokens.weight"]}))["status"], "summarized")
        self.assertNotEqual(self.run_case(t, shapes={"model.norm.weight": (1, 5)})["status"], "summarized")
        w = self.window({"G0_fp32_control": (t, good_result(), {})})
        ip = w / "cases/01_G0_fp32_control/first_boundary_grad.f32.index.json"
        meta = json.loads(ip.read_text()); meta["index"].append(dict(meta["index"][0])); ip.write_text(json.dumps(meta))
        s = fs.summarize(w, self.rdir, self.tmp / "dup_out", self.frozen)
        self.assertNotEqual(s["cases"]["G0_fp32_control"]["status"], "summarized")

    def test_nonfinite_tamper_and_degenerate_refuse(self):
        t = dict(self.ref)
        bad = dict(t); bad["model.norm.weight"] = np.array([1, np.nan, 2, 3, np.inf])
        self.assertNotEqual(self.run_case(bad)["status"], "summarized")
        self.assertNotEqual(self.run_case(t, tamper=5)["status"], "summarized")
        z = self.run_case({k: np.zeros_like(np.asarray(v)) for k, v in t.items()})
        self.assertTrue(z["vs_sealed_r0"]["degenerate_endpoint"])
        self.assertNotEqual(z["status"], "summarized")

    def test_wrong_inputs_recipe_phase_or_update_claims_refuse(self):
        t = dict(self.ref)
        muts = [lambda r: r["input_digests"].update(per_step_microbatch_sha256=[["d0", "d1", "d2", "x"]]),
                lambda r: r["input_digests"].update(attention_mask="ones"),
                lambda r: r["spec"].update(grad_accum=2),
                lambda r: r["boundary_ledger"]["measured"].update(microbatches=3),
                lambda r: r["boundary_ledger"]["measured"].update(counter_updates=1),
                lambda r: r["boundary_ledger"]["measured"].update(base_optimizer_calls=1),
                lambda r: r["update_audit"].update(optimizer_calls=1),
                lambda r: r["assessment"]["update_state"].update(status="verified_sampled_every_base_call"),
                lambda r: r.update(parameters_unchanged=False),
                lambda r: r["boundary_ledger"]["measured"].update(microbatch_losses=[12.0, float("nan"), 1, 1]),
                lambda r: r["boundary_ledger"]["measured"]["steps"][0].update(counter_source="native optimizer post-hook"),
                lambda r: r["canonical_state_check"].update(module_state_matches_canonical=False),
                lambda r: r.update(result_kind="numerical_diagnostic")]
        for i, m in enumerate(muts):
            r = good_result(); m(r)
            self.assertNotEqual(self.run_case(t, result=r)["status"], "summarized", i)
        zr = good_result("zero2")
        self.assertEqual(self.run_case(t, result=zr)["status"], "summarized")
        zr["boundary_ledger"]["measured"]["steps"][0].update(engine_micro_steps=4, engine_global_steps=1)
        self.assertNotEqual(self.run_case(t, result=zr)["status"], "summarized")       # DeepSpeed step reached
        zf = good_result("zero2")
        zf["boundary_ledger"]["measured"]["steps"][0]["boundary_flags_before_backward"] = [False, True, False, True]
        self.assertNotEqual(self.run_case(t, result=zf)["status"], "summarized")       # boundary not the 4th only
        w = self.window({"G0_fp32_control": (t, None, {})})                         # result.json missing
        self.assertNotEqual(fs.summarize(w, self.rdir, self.tmp / "nores", self.frozen)["cases"]["G0_fp32_control"]["status"],
                            "summarized")

    def test_order_permuted_case_maps_losses_to_generation_order(self):
        r = good_result(microbatch_order=[3, 2, 1, 0])
        r["spec"]["microbatch_order"] = [3, 2, 1, 0]
        r["boundary_ledger"]["measured"]["microbatch_losses"] = [12.3, 12.2, 12.1, 12.0]   # applied (reversed) order
        c = self.run_case(dict(self.ref), result=r)
        self.assertEqual(c["status"], "summarized", c["record_problems"])
        self.assertEqual([d["abs"] for d in c["raw_loss_differences"]], [0.0] * 4)
        r2 = copy.deepcopy(r); r2["microbatch_order"] = [0, 1, 2, 3]                       # record != declared
        self.assertNotEqual(self.run_case(dict(self.ref), result=r2)["status"], "summarized")

    def test_pairwise_and_fresh_output_root(self):
        t = {k: np.array(v, dtype=np.float32) for k, v in self.ref.items()}
        u = {k: 2 * v for k, v in t.items()}
        w = self.window({"G0_fp32_control": (t, good_result(), {}), "G1_ddp_autocast": (u, good_result(), {}),
                         "G3_ddp_bf16_params": (t, good_result(lineage=1), {})})
        s = fs.summarize(w, self.rdir, self.tmp / "pw", self.frozen,
                         pairs=[("G1_ddp_autocast", "G0_fp32_control"), ("G3_ddp_bf16_params", "missing")])
        self.assertAlmostEqual(s["pairwise"]["G1_ddp_autocast_vs_G0_fp32_control"]["global"]["norm_ratio"], 2.0, places=6)
        self.assertIn("not computed", s["pairwise"]["G3_ddp_bf16_params_vs_missing"]["status"])
        with self.assertRaises(FileExistsError):
            fs.summarize(w, self.rdir, self.tmp / "pw", self.frozen)                   # never reuse an output root


class MicrobatchOrder(unittest.TestCase):
    def test_permutation_is_validated_and_losses_map_back(self):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed")
        m = __import__("mgbench.train.__main__", fromlist=["x"])
        self.assertEqual(m.apply_microbatch_order(["a", "b", "c", "d"], [3, 2, 1, 0]), (["d", "c", "b", "a"], [3, 2, 1, 0]))
        self.assertEqual(m.apply_microbatch_order(["a", "b"], None), (["a", "b"], [0, 1]))
        for bad in ([0, 1, 2], [0, 0, 1, 2], [True, 1, 2, 3], "3210"):
            with self.assertRaises(RuntimeError):
                m.apply_microbatch_order(["a", "b", "c", "d"], bad)


class DeepSpeedSourcePin(unittest.TestCase):
    """Binds the boundary bookkeeping the first-boundary path relies on to the INSTALLED DeepSpeed files (read as text
    via find_spec; nothing is imported). A different installation fails here instead of being assumed equivalent."""
    PINS = {"runtime/engine.py": ("2ec177075887663b31be6e14cf1d519e26fcb98fff2524b8160197bcc45ab216", [
                "return (self.micro_steps + 1) % self.gradient_accumulation_steps() == 0",
                "loss = loss / self.gradient_accumulation_steps() if scale_wrt_gas else loss",
                "        self.micro_steps += 1", "        if self.is_gradient_accumulation_boundary():\n            self.gas_boundary_ctr += 1"]),
            "runtime/constants.py": ("4617caee65e337fe2d3396dac9e4abd7270f6dc40d2515469e796d1368179d3b",
                                     ["MANAGED_GRADIENT_ACCUMULATION_DEFAULT = True"]),
            "runtime/zero/stage_1_and_2.py": ("f878e5677ab1da63e905e221e0a4d6f7e8f06851adbc03f016b80bb4dd315a7f", [
                "if self.clip_grad == 0.0 and self.loss_scale == 1.0:\n            return",
                "single_grad_partition = flat_grad_partition.to(self.single_partition_of_fp32_groups[i].dtype)",
                "flat_grad_partition = self.flatten_dense_tensors_aligned(self.averaged_gradients[i],"])}

    def test_installed_sources_match_the_pins(self):
        import importlib.util
        spec = importlib.util.find_spec("deepspeed")
        if spec is None or spec.origin is None:
            self.skipTest("deepspeed not installed")
        base = Path(spec.origin).parent
        for rel, (sha, needles) in self.PINS.items():
            text = (base / rel).read_bytes()
            self.assertEqual(hashlib.sha256(text).hexdigest(), sha, rel)
            for nd in needles:
                self.assertIn(nd, text.decode(), (rel, nd))


if __name__ == "__main__":
    unittest.main()
