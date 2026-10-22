"""CPU tests for Candidate 16: the pre-registered control-C comparator (independent numpy fixtures), the two-device
pair controls (authorization, case/GPU binding, rank identity, per-GPU admission from SUPPLIED snapshots, bounded rank
records), and a tiny gloo world-1 (GA 4) vs world-2 (GA 2) end-to-end run of the production DDP trainer on CPU.
No GPU, device query, cache file or network is involved."""
import copy
import datetime as dt
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
from mgbench import control_c as cc  # noqa: E402
from mgbench import pair_controls as pc  # noqa: E402

U4, U5 = "GPU-00000000-0000-0000-0000-000000000109", "GPU-00000000-0000-0000-0000-000000000103"
NAMES = cc.slice_names(2) + ["model.embed_tokens.weight", "model.layers.1.self_attn.k_proj.bias"]
RECIPE = {"model": "tiny", "seq_len": 16, "micro_batch": 1}
DIG = [[f"s{k}r{j}" for j in range(4)] for k in range(5)]


def write_snap(path, tensors):
    idx, off, blob = [], 0, b""
    for name, a in tensors.items():
        raw = np.asarray(a, dtype="<f4").reshape(-1).tobytes()
        idx.append({"name": name, "shape": list(np.asarray(a).shape), "numel": int(np.asarray(a).size), "offset": off,
                    "sha256": hashlib.sha256(raw).hexdigest()})
        blob += raw
        off += len(raw)
    Path(path).write_bytes(blob)
    Path(str(path) + ".index.json").write_text(json.dumps({"bytes": off, "elements": off // 4, "index": idx}))


def record(world, ga, losses=None, **over):
    per_rank = [{"rank": r, "update_state": "verified_sampled_every_base_call", "identity_ok": True,
                 "warmup": {"logical_attempts": 0},
                 "measured": {"logical_attempts": 5, "microbatches": 5 * ga, "counter_updates": 5,
                              "base_calls_per_attempt": [1] * 5, "boundary_coverage": ["complete"] * 5,
                              "microbatch_losses_all_finite": True}} for r in range(world)]
    losses = losses or [[10.0 + k + 0.1 * j for j in range(4)] for k in range(5)]
    gm = [{"step": k, "rank": r, "microstep": i, "global_row": i * world + r, "digest": DIG[k][i * world + r],
           "raw_loss": losses[k][i * world + r]} for k in range(5) for r in range(world) for i in range(ga)]
    rec = {"result_kind": "full_control_5update",
           "spec": dict(RECIPE, n_gpus=world, grad_accum=ga, measure_steps=5, warmup_steps=0, fp32_reference=True),
           "per_rank": per_rank, "global_sample_map": gm, "canonical_state_matches": True,
           "snapshots": {"writer_rank": 0, "files": ["dtheta_final.f32", "grad_b1.f32"], "dtheta_after_update": 5}}
    rec.update(over)
    return rec


class Comparator(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        rng = np.random.default_rng(1)
        self.g = {n: rng.normal(size=(5, 3)) for n in NAMES}
        self.u = {n: 1e-4 * np.sign(rng.normal(size=(5, 3))) for n in NAMES}
        self.frozen = {"recipe": RECIPE, "n_layers": 2, "global_row_digests": DIG,
                       "tied_aliases": {"model.embed_tokens.weight": ["lm_head.weight"]}}

    def pair(self, c_g=None, c_u=None, r5=None, c5=None, tag="p"):
        dr, dc = self.tmp / f"{tag}_R5", self.tmp / f"{tag}_C5"
        for d, g, u, rec in ((dr, self.g, self.u, r5 or record(1, 4)), (dc, c_g or self.g, c_u or self.u, c5 or record(2, 2))):
            d.mkdir()
            write_snap(d / "grad_b1.f32", g)
            write_snap(d / "dtheta_final.f32", u)
            (d / "control_record.json").write_text(json.dumps(rec))
        return cc.compare_control(dr, dc, self.frozen, chunk=4)

    def test_matching_runs_meet_c_exactly(self):
        out = self.pair()
        self.assertEqual(out["problems"], [])
        self.assertTrue(out["control_C_met"], out["verdicts"])
        self.assertEqual(out["endpoints_preregistered"]["G1_rel_l2"], 0.0)
        self.assertEqual(out["endpoints_preregistered"]["L_max_k0_4_rel"], 0.0)
        self.assertEqual(out["preregistration_sha256"], "cd367fc553cb56d38cd7d17673c442253757ddc42923dfac85a8bcf67a668ce3")
        self.assertTrue(out["informed_by_prior_results"])

    def test_verdict_rules_follow_the_text_and_never_triple_a_cosine(self):
        self.assertEqual(cc.verdict("G1_rel_l2", 1e-4), "pass")
        self.assertEqual(cc.verdict("G1_rel_l2", 2.9e-4), "marginal (investigate)")
        self.assertEqual(cc.verdict("G1_rel_l2", 3.1e-4), "fail")
        self.assertEqual(cc.verdict("U_rel_l2", 0.15), "marginal (investigate)")
        self.assertTrue(cc.verdict("G1_cosine", 0.9999).startswith("not_pass (marginal/fail band undefined"))
        self.assertEqual(cc.verdict("U_cosine", 0.999), "pass")
        for bad in (None, float("nan"), float("inf"), "0"):
            self.assertEqual(cc.verdict("L_k0_rel", bad), "missing")
        self.assertEqual(cc.LIMITS["L_k0_rel"], ("upper", 1e-6))          # RELATIVE, as pre-registered
        self.assertTrue(any("RELATIVE" in a for a in cc.AMBIGUITIES))

    def test_known_violations_stop_interpretation(self):
        g = {k: v.copy() for k, v in self.g.items()}
        g["model.norm.weight"] = g["model.norm.weight"] * (1 + 1e-3)       # inside the pre-registered slices
        out = self.pair(c_g=g, tag="g")
        self.assertIn(out["verdicts"]["G1_rel_l2"], ("marginal (investigate)", "fail"))
        self.assertFalse(out["control_C_met"])
        self.assertTrue(out["recipe_interpretation"].startswith("NOT interpreted"))
        u = {k: (-v if k == "model.layers.0.mlp.down_proj.weight" else v) for k, v in self.u.items()}   # sign flips
        out = self.pair(c_u=u, tag="u")
        self.assertTrue(out["verdicts"]["U_cosine"].startswith("not_pass"))
        losses = [[10.0 + k + 0.1 * j for j in range(4)] for k in range(5)]
        losses[3][2] += 2e-3                                                # drift at k = 3 (relative 3.8e-5 on the mean)
        out = self.pair(c5=record(2, 2, losses=losses), tag="l")
        self.assertEqual(out["verdicts"]["L_k0_rel"], "pass")
        self.assertEqual(out["verdicts"]["L_max_k0_4_rel"], "fail")

    def test_contract_and_alignment_failures(self):
        muts = {"missing tensor": lambda: self.pair(c_g={k: v for k, v in self.g.items() if k != NAMES[2]}, tag="m"),
                "extra tensor": lambda: self.pair(c_g=dict(self.g, extra=np.ones(2)), tag="e"),
                "tied alias stored": lambda: self.pair(c_g=dict(self.g, **{"lm_head.weight": self.g["model.embed_tokens.weight"]}), tag="t"),
                "shape misaligned": lambda: self.pair(c_g=dict(self.g, **{NAMES[0]: self.g[NAMES[0]].reshape(3, 5)}), tag="s"),
                "nan": lambda: self.pair(c_u=dict(self.u, **{NAMES[1]: np.full((5, 3), np.nan)}), tag="n"),
                "missing rank": lambda: self.pair(c5=dict(record(2, 2), per_rank=record(2, 2)["per_rank"][:1]), tag="r"),
                "inconsistent counts": lambda: self.pair(c5=self._mut(record(2, 2), lambda r: r["per_rank"][1]["measured"].update(counter_updates=4)), tag="c"),
                "update not applicable": lambda: self.pair(c5=self._mut(record(2, 2), lambda r: r["per_rank"][0].update(update_state="not_applicable")), tag="na"),
                "wrong endpoint step": lambda: self.pair(c5=self._mut(record(2, 2), lambda r: r["snapshots"].update(dtheta_after_update=3)), tag="w"),
                "rank-1 writer": lambda: self.pair(c5=self._mut(record(2, 2), lambda r: r["snapshots"].update(writer_rank=1)), tag="wr"),
                "rank-seeded data": lambda: self.pair(c5=self._mut(record(2, 2), lambda r: r["global_sample_map"][1].update(digest="other")), tag="d"),
                "wrong sample weighting": lambda: self.pair(c5=self._mut(record(2, 2), lambda r: r["global_sample_map"].pop()), tag="sw"),
                "row/rank mapping": lambda: self.pair(c5=self._mut(record(2, 2), lambda r: r["global_sample_map"][0].update(rank=1)), tag="rm"),
                "R5 recipe": lambda: self.pair(r5=self._mut(record(1, 4), lambda r: r["spec"].update(grad_accum=2)), tag="rr")}
        for name, fn in muts.items():
            out = fn()
            self.assertTrue(out["problems"], name)
            self.assertFalse(out["control_C_met"], name)

    @staticmethod
    def _mut(rec, fn):
        rec = copy.deepcopy(rec)
        fn(rec)
        return rec


class PairControls(unittest.TestCase):
    def setUp(self):
        lc = __import__("mgbench.launch_controls", fromlist=["x"])
        self.lc = lc
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.cand = self.tmp / "cand"
        (self.cand / "studies").mkdir(parents=True)
        (self.cand / "studies/x.py").write_text("a = 1\n")
        now = dt.datetime.now(dt.timezone.utc)
        self.good = {"id": "t", "scope": pc.PAIR_SCOPE, "candidate_root": str(self.cand),
                     "candidate_digest": lc.candidate_digest(self.cand), "gpu_indices": [4, 5],
                     "gpu_uuids": {"4": U4, "5": U5}, "case_gpus": pc.CASE_GPUS, "max_gpus": 2, "max_concurrent_jobs": 1,
                     "max_cases": 2, "issued_utc": (now - dt.timedelta(minutes=1)).isoformat(),
                     "expires_utc": (now + dt.timedelta(minutes=40)).isoformat(), "nonrenewable": True}

    def auth(self, **kw):
        p = self.tmp / f"a{len(list(self.tmp.glob('a*.json')))}.json"
        self.lc.create_exclusive(p, dict(self.good, **kw))
        return p

    def prob(self, p, **kw):
        return pc.pair_authorization_problems(p, **dict(dict(candidate_root=self.cand, uuids={4: U4, 5: U5}), **kw))

    def test_pair_authorization_refusals(self):
        p = self.auth()
        self.assertEqual(self.prob(p), [])
        self.assertTrue(self.prob(None))
        self.assertTrue(self.prob(self.auth(**{k: None for k in pc.REQUIRED})))                      # null
        self.assertTrue(self.prob(self.auth(expires_utc="2026-10-16T00:00:00+00:00")))              # expired
        self.assertTrue(self.prob(self.auth(scope="train-first-boundary")))                          # wrong scope
        self.assertTrue(self.prob(p, n_cases_done=2))                                                # ceiling
        self.assertTrue(self.prob(p, expected_sha256="0" * 64))                                      # hash
        self.assertTrue(self.prob(self.auth(gpu_uuids={"4": U5, "5": U4})))                          # device swap
        self.assertTrue(self.prob(self.auth(gpu_indices=[4, 6])))
        self.assertTrue(self.prob(self.auth(case_gpus=dict(pc.CASE_GPUS, X1=[4]), max_cases=3)))     # extra case
        self.assertTrue(self.lc.revoke(p, "t"))
        self.assertTrue(self.prob(p))

    def test_old_single_gpu_authorizations_never_admit_world_two(self):
        now = dt.datetime.now(dt.timezone.utc)
        a = {"id": "s", "scope": "train-first-boundary", "candidate_root": str(self.cand),
             "candidate_digest": self.lc.candidate_digest(self.cand), "gpu_index": 4, "gpu_uuid": U4, "max_gpus": 1,
             "max_concurrent_jobs": 1, "max_cases": 5, "issued_utc": (now - dt.timedelta(minutes=1)).isoformat(),
             "expires_utc": (now + dt.timedelta(minutes=9)).isoformat(), "nonrenewable": True}
        p = self.tmp / "single.json"
        self.lc.create_exclusive(p, a)
        self.assertEqual(self.lc.authorization_problems(p, scope="train-first-boundary", candidate_root=self.cand,
                                                        gpus=[4], uuids={4: U4}), [])
        self.assertTrue(self.lc.authorization_problems(p, scope="train-first-boundary", candidate_root=self.cand,
                                                       gpus=[4, 5], uuids={4: U4, 5: U5}))
        self.assertTrue(self.prob(p))                                                                # not a pair auth

    def test_case_binding_rank_identity_and_admission(self):
        self.assertEqual(pc.case_gpu_problems("C5_fp32_world2_ga2", [4, 5]), [])
        for cid, g in (("C5_fp32_world2_ga2", [5, 4]), ("R5_fp32_world1_ga4", [5]), ("X1_fp32_order_permuted", [4])):
            self.assertTrue(pc.case_gpu_problems(cid, g))
        exp = [{"rank": 0, "nvml_index": 4, "uuid": U4, "pci_bus_id": "0000:81:00.0"},
               {"rank": 1, "nvml_index": 5, "uuid": U5, "pci_bus_id": "0000:82:00.0"}]
        self.assertEqual(pc.rank_identity_problems(exp, copy.deepcopy(exp)), [])
        swapped = [dict(exp[0], rank=1), dict(exp[1], rank=0)]
        self.assertTrue(pc.rank_identity_problems(exp, swapped))
        self.assertTrue(pc.rank_identity_problems(exp, exp[:1]))                                      # missing rank
        self.assertTrue(pc.rank_identity_problems(exp, exp + [dict(exp[1], rank=2)]))                # extra row
        self.assertTrue(pc.rank_identity_problems(exp, [exp[0], exp[0]]))                            # duplicate
        idle = lambda w=20.0: {"mem_used_mib": 1, "util_pct": 0, "power_w": w, "temp_c": 25, "power_limit_w": 300, "processes": []}
        snap = {"gpus": {i: idle() for i in range(8)}}
        self.assertEqual(pc.pair_admission_problems(snap, [4, 5]), [])
        hot = copy.deepcopy(snap)
        for i in (0, 1, 2):
            hot["gpus"][i]["power_w"] = 300.0                                  # 900 + 3 x 20 + 600 > 1500
        self.assertTrue(pc.pair_admission_problems(hot, [4, 5]))
        self.assertEqual(pc.pair_admission_problems(hot, [4]), [])            # one GPU: 960 + 300 <= 1500
        for mut in (lambda s: s["gpus"][5].update(processes=[123]), lambda s: s["gpus"][5].update(power_limit_w=350),
                    lambda s: s["gpus"][4].update(mem_used_mib=500), lambda s: s["gpus"][5].update(temp_c=float("nan")),
                    lambda s: s["gpus"].pop(7), lambda s: s["gpus"][5].update(processes=None)):
            s = copy.deepcopy(snap); mut(s)
            self.assertTrue(pc.pair_admission_problems(s, [4, 5]))

    def test_proposed_pair_config_admission_and_shared_limits(self):
        import yaml
        cfg = yaml.safe_load((ROOT / "studies/a6000/configs/control_pair.yaml").read_text())
        fp = ROOT.parents[1] / "36_c16_design/pair_memory_feasibility.json"
        if not fp.exists():
            self.skipTest("Candidate 16 design files are on the node only")
        feas = json.loads(fp.read_text())
        self.assertEqual(pc.pair_config_problems(cfg, feas), [])
        muts = [lambda c: c["cases"][1].update(grad_accum=4), lambda c: c["cases"][1].update(mem_fraction=0.92),
                lambda c: c["limits"].update(worker_memory_bytes=2 * 49392123904),       # never 46 GiB per rank
                lambda c: c["common"].update(measure_steps=3), lambda c: c.update(case_gpus={"R5_fp32_world1_ga4": [5],
                                                                                           "C5_fp32_world2_ga2": [4, 5]}),
                lambda c: c["cases"].append({"id": "X1_fp32_order_permuted"}), lambda c: c["timing"].update(window_s=3600)]
        for i, m in enumerate(muts):
            c = copy.deepcopy(cfg); m(c)
            self.assertTrue(pc.pair_config_problems(c, feas), i)
        self.assertTrue(pc.pair_config_problems(cfg, None))

    def test_bounded_rank_record_collection(self):
        d = self.tmp / "run"
        d.mkdir()
        (d / "rank0_record.json").write_text(json.dumps({"rank": 0}))
        recs, probs = pc.collect_rank_records(d, 2)
        self.assertTrue(any("rank 1: record missing" in p for p in probs))
        (d / "rank1_record.json").write_text(json.dumps({"rank": 0}))
        self.assertTrue(any("claims rank" in p for p in pc.collect_rank_records(d, 2)[1]))
        (d / "rank1_record.json").write_text(json.dumps({"rank": 1}))
        self.assertEqual(pc.collect_rank_records(d, 2)[1], [])
        (d / "rank2_record.json").write_text("{}")
        self.assertTrue(any("unexpected" in p for p in pc.collect_rank_records(d, 2)[1]))
        (d / "rank2_record.json").unlink()
        (d / "rank1_record.json").write_text(json.dumps({"rank": 1, "x": "y" * 2000}))
        self.assertTrue(any("larger" in p for p in pc.collect_rank_records(d, 2, max_bytes=1000)[1]))


TINY = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "vocab_size": 128, "hidden_size": 32,
        "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 64, "tie_word_embeddings": True, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "torch_dtype": "float32", "attention_dropout": 0.0, "use_cache": False}


def _gloo_worker(rank, world, ga, store, cfg_dir, out_dir):
    sys.path.insert(0, str(SRC))
    import torch
    import torch.distributed as dist
    import torch.nn.parallel as tnp
    torch.set_num_threads(1)
    dist.init_process_group("gloo", store=dist.FileStore(store, world), rank=rank, world_size=world)
    base = tnp.DistributedDataParallel

    class CpuDDP(base):
        def __init__(self, module, device_ids=None, **kw):
            super().__init__(module, device_ids=None, **kw)
    tnp.DistributedDataParallel = CpuDDP
    from mgbench.config import resolved_framework_config
    from mgbench.data import GlobalSynthetic
    from mgbench.train import frameworks
    from mgbench.train.__main__ import _compare_canonical, build_control_record, finalize_reference, rank_payload
    from mgbench.train.boundary_ledger import assess
    from mgbench.results import write_json
    spec = dict(model="tiny", seq_len=16, micro_batch=1, grad_accum=ga, seed=1234, warmup_steps=0, measure_steps=5,
                numerical_checks=True, canonical_init=True, canonical_start="bf16_representable", global_inputs=True,
                fp32_reference=True, reference_role="write", reference_dir=out_dir, framework="ddp", n_gpus=world,
                result_kind="full_control_5update")
    entry = {"kind": "causal_lm", "hf_config_dir": cfg_dir}
    tr = frameworks.SETUP["ddp"](spec, entry, resolved_framework_config(spec), torch.device("cpu"))
    gathered = tr.state_fingerprint()
    canon = _compare_canonical(tr.canonical, gathered, spec) if rank == 0 else None
    data = GlobalSynthetic("causal_lm", micro_batch=1, grad_accum=ga, world=world, rank=rank, device=torch.device("cpu"),
                           seed=1234000, steps=5, vocab_size=128, seq_len=16, image_size=None, num_classes=None)
    tr.ledger.start_measured()
    for _ in range(5):
        tr.step([data.next() for _ in range(ga)])
    summ = tr.ledger.summary(5, 0, ga)
    finalize_reference(tr, summ["measured"])
    upd = assess([summ], checks_requested=True, update_evidence_requested=True, measurement_role="diagnostic",
                 diagnostics_in_timed_path=True)["update_state"]["status"]
    payload = rank_payload(rank, summ, data.input_digests, upd, True)
    write_json(Path(out_dir) / f"rank{rank}_record.json", payload)
    allp = [None] * world
    dist.all_gather_object(allp, payload)
    if rank == 0:
        import os
        snaps = {"writer_rank": 0, "dtheta_after_update": 5,
                 "files": sorted(p for p in ("grad_b1.f32", "dtheta_final.f32") if os.path.exists(Path(out_dir) / p))}
        write_json(Path(out_dir) / "control_record.json",
                   build_control_record(spec, allp, canonical_ok=canon["module_state_matches_canonical"], snapshots=snaps))
    dist.destroy_process_group()


class GlooWorldOneVersusTwo(unittest.TestCase):
    def test_tiny_r5_c5_end_to_end(self):
        try:
            import torch
            import torch.multiprocessing as mp
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("torch/transformers not installed")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        (tmp / "tiny").mkdir()
        (tmp / "tiny/config.json").write_text(json.dumps(TINY))
        dirs = {}
        for world, ga in ((1, 4), (2, 2)):
            d = tmp / f"w{world}"
            d.mkdir()
            mp.spawn(_gloo_worker, args=(world, ga, str(tmp / f"store{world}"), str(tmp / "tiny"), str(d)),
                     nprocs=world, join=True)
            dirs[world] = d
        self.assertEqual(sorted(p.name for p in dirs[2].iterdir()),
                         sorted(["control_record.json", "dtheta_final.f32", "dtheta_final.f32.index.json", "grad_b1.f32",
                                 "grad_b1.f32.index.json", "rank0_record.json", "rank1_record.json",
                                 "reference_scalars.json"]))                        # rank 1 wrote only its record
        sys.path.insert(0, str(SRC))
        from mgbench.data import GlobalSynthetic
        g = GlobalSynthetic("causal_lm", micro_batch=1, grad_accum=4, world=1, rank=0, device=torch.device("cpu"),
                            seed=1234000, steps=5, vocab_size=128, seq_len=16, image_size=None, num_classes=None)
        frozen = {"recipe": {"model": "tiny", "seq_len": 16, "micro_batch": 1}, "n_layers": 2,
                  "global_row_digests": g.input_digests["per_step_microbatch_sha256"],
                  "tied_aliases": {"model.embed_tokens.weight": ["lm_head.weight"]}}
        out = cc.compare_control(dirs[1], dirs[2], frozen)
        self.assertEqual(out["problems"], [], out["problems"])
        c5 = json.loads((dirs[2] / "control_record.json").read_text())
        self.assertEqual(sorted(s["global_row"] for s in c5["global_sample_map"] if s["rank"] == 1 and s["step"] == 0), [1, 3])
        for k, v in out["endpoints_preregistered"].items():
            self.assertTrue(v is not None and math.isfinite(v), k)
        print("\n[gloo R5 vs C5 tiny]", json.dumps({k: out["endpoints_preregistered"][k] for k in cc.LIMITS}),
              json.dumps(out["verdicts"]))


if __name__ == "__main__":
    unittest.main()
