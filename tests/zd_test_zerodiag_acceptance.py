"""CPU acceptance for the ZeRO diagnostic instrumentation.

What these tests DO cover: the production extraction, reconciliation, oracle and
verdict code, driven against a structurally faithful stand-in that exposes the
REAL DeepSpeed 0.15.1 attribute names and is stepped by a REAL torch AdamW over
master partitions.

What they DO NOT cover, stated plainly: they do not validate CUDA kernels, live
AMP skip behaviour, NCCL collectives, or the binding of this code to an actual
`DeepSpeedZeroOptimizer` instance. Those are hardware-only unknowns.

No device access: CUDA is hidden and the guard refuses `torch.cuda` entry points.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
from zerodiag import engine_probe as EP          # noqa: E402
from zerodiag import factories as F              # noqa: E402
from zerodiag import update_oracle as OR         # noqa: E402
from zerodiag import verdict as V                # noqa: E402

HP = {"lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
      "weight_decay": 0.01, "decoupled_weight_decay": True, "amsgrad": False}


class FakeZeroOptimizer:
    """Structurally faithful stand-in: the real attribute names, a real AdamW.

    Deliberately NOT presented as engine validation. It exists so the production
    extraction code runs on CPU against the layout it will meet.
    """

    def __init__(self, *, group_numels=(16, 8), world_size=1, padding=None,
                 dtype=torch.float64, working_dtype=None):
            self.world_size = world_size
            self.partition_size = [n for n in group_numels]
            self.groups_padding = list(padding or [0] * len(group_numels))
            self.single_partition_of_fp32_groups = [
                torch.nn.Parameter(torch.randn(n, dtype=dtype))
                for n in group_numels]
            self.optimizer = torch.optim.AdamW(
                self.single_partition_of_fp32_groups, lr=HP["lr"])
            self.overflow = False
            self.bit16_groups = [[p] for p in self.single_partition_of_fp32_groups]
            if working_dtype is not None:
                self.fp16_groups_flat = [
                    m.detach().to(working_dtype).clone()
                    for m in self.single_partition_of_fp32_groups]

    def set_grads(self, scale=0.01, nonfinite=False, none=False):
        for m in self.single_partition_of_fp32_groups:
            if none:
                m.grad = None
            else:
                g = torch.randn_like(m) * scale
                if nonfinite:
                    g[0] = float("nan")
                m.grad = g

    def step(self):
        self.optimizer.step()
        if hasattr(self, "fp16_groups_flat"):
            for i, m in enumerate(self.single_partition_of_fp32_groups):
                self.fp16_groups_flat[i] = m.detach().to(
                    self.fp16_groups_flat[i].dtype).clone()

    def resolved(self):
        return F.resolve_optimizer_semantics(self.optimizer)


def run_boundary(z, *, rank=0, world_size=1, hp=None, accumulation=1,
                 coverage=OR.FULL):
    exp = EP.expected_partition_map(z, rank=rank, world_size=world_size)
    sc = EP.verify_storage_correspondence(z, rank=rank)
    before = EP.capture_boundary(z, rank=rank, world_size=world_size, phase="before")
    z.step()
    after = EP.capture_boundary(z, rank=rank, world_size=world_size, phase="after")
    mc = EP.module_reflects_masters(z)
    return V.verify_boundary(before=before, after=after, expected_map=exp,
                             resolved_hparams=hp or dict(HP),
                             accumulation_steps=accumulation, rank=rank,
                             world_size=world_size, storage_check=sc,
                             module_check=mc, coverage=coverage)


class RetainedFactoryEquivalence(unittest.TestCase):
    def test_the_retained_model_is_reproduced_on_cpu(self):
        m, cfg = F.build_model(seed=1234, device="cpu")
        ident = F.resolve_model_identity(m, cfg)
        self.assertEqual(ident["n_positions"], 1024)
        self.assertEqual(ident["sum_of_named_parameter_numel"], 124_439_808)
        self.assertEqual(ident["named_parameter_entries"], 148)
        self.assertEqual(ident["distinct_storages"], 148)

    def test_the_lm_head_tie_exists_and_is_counted_once(self):
        m, _ = F.build_model(seed=1234, device="cpu")
        self.assertIs(m.lm_head.weight, m.transformer.wte.weight)
        names = [n for n, _ in m.named_parameters()]
        self.assertNotIn("lm_head.weight", names)

    def test_dropout_is_active_in_the_retained_recipe(self):
        m, cfg = F.build_model(seed=1234, device="cpu")
        self.assertTrue(m.training)
        self.assertEqual(cfg.resid_pdrop, 0.1)

    def test_resolved_optimizer_weight_decay_is_not_zero(self):
        m, _ = F.build_model(seed=1234, device="cpu")
        sem = F.resolve_optimizer_semantics(F.build_optimizer(m))
        self.assertEqual(sem["weight_decay"], 0.01)
        self.assertTrue(sem["decoupled_weight_decay"])

    def test_model_construction_is_seed_deterministic(self):
        a = F.state_digest(F.build_model(seed=7, device="cpu")[0])
        b = F.state_digest(F.build_model(seed=7, device="cpu")[0])
        c = F.state_digest(F.build_model(seed=8, device="cpu")[0])
        self.assertEqual(a["parameter_state_sha256"], b["parameter_state_sha256"])
        self.assertNotEqual(a["parameter_state_sha256"], c["parameter_state_sha256"])

    def test_realized_input_digest_is_rank_dependent(self):
        g0 = F.build_generator(seed=1234, rank=0, device="cpu")
        g1 = F.build_generator(seed=1234, rank=1, device="cpu")
        d0 = F.input_digest(F.synthetic_batch(2, 8, "cpu", g0))
        d1 = F.input_digest(F.synthetic_batch(2, 8, "cpu", g1))
        self.assertNotEqual(d0["realized_input_sha256"], d1["realized_input_sha256"])
        self.assertEqual(d0["shape"], [2, 8])


class PositiveBoundary(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_fp32_boundary_verifies(self):
        z = FakeZeroOptimizer(); z.set_grads()
        r = run_boundary(z)
        self.assertTrue(r["rank_verification_successful"], r["per_partition"][0].get("reasons"))
        self.assertEqual(r["logical_boundaries"], 2)
        self.assertEqual(r["base_optimizer_calls"], 2)
        self.assertEqual(r["successful_updates"], 2)
        self.assertEqual(r["skipped_updates"], 0)
        self.assertTrue(r["partition_reconciliation"]["identities_reconciled"])

    def test_multi_rank_combination_requires_every_rank(self):
        per = []
        for rk in range(2):
            z = FakeZeroOptimizer(world_size=2); z.set_grads()
            per.append(run_boundary(z, rank=rk, world_size=2))
        whole = V.combine_ranks(per, world_size=2)
        self.assertTrue(whole["whole_model_verified"])
        self.assertEqual(whole["missing_ranks"], [])

    def test_mixed_precision_working_copy_matches_the_expected_cast(self):
        """Updated for the repaired criterion.

        The fixture's step() refreshes the fp16 working copy from the updated
        masters, mirroring DeepSpeed's propagation, so the expected-cast
        criterion passes. This is discriminating here: 6 of 16 fp16 elements
        actually change across the step, so a copy left unrefreshed is
        detectable (see the stale test below).
        """
        z = FakeZeroOptimizer(working_dtype=torch.float16); z.set_grads()
        r = run_boundary(z)
        mc = r["module_reflects_masters"]
        self.assertEqual(mc["per_group"][0]["criterion"],
                         "mixed_precision_expected_cast")
        self.assertTrue(mc["verdict_successful"])
        self.assertTrue(mc["bitwise_identical"])
        self.assertIsNone(mc["tolerance_used"])
        self.assertNotIn("within_tolerance", mc)

    def test_stale_mixed_precision_working_copy_fails(self):
        """A working copy that was NOT refreshed must fail the criterion."""
        z = FakeZeroOptimizer(working_dtype=torch.float16); z.set_grads()
        stale = [f.clone() for f in z.fp16_groups_flat]
        run_boundary(z)                    # masters move; fixture refreshes
        with torch.no_grad():              # undo the propagation
            for f, old in zip(z.fp16_groups_flat, stale):
                f.copy_(old)
        mc = EP.module_reflects_masters(z)
        self.assertFalse(mc["verdict_successful"])
        self.assertFalse(mc["bitwise_identical"])
        self.assertGreater(mc["per_group"][0]["n_differing_elements"], 0)

    def test_coverage_is_inventory_based_not_rank_based(self):
        """R4: coverage is established by the inventory, never by rank count."""
        z = FakeZeroOptimizer(); z.set_grads()
        r = run_boundary(z)
        ci = r["coverage_inventory"]
        self.assertEqual(ci["n_expected_partitions"], len(ci["expected_partitions"]))
        self.assertGreater(ci["expected_elements_this_rank"], 0)
        self.assertEqual(ci["world_size"], 1)
        # world size one does not by itself confer completeness
        self.assertFalse(ci["complete"])
        self.assertIn("no model parameter inventory", ci["incomplete_because"])
        self.assertIn("number of ranks", r["coverage_is_not_established_by"])
        self.assertIn("world size one", r["coverage_is_not_established_by"])

    def test_identity_coverage_is_not_gradient_recomputation(self):
        z = FakeZeroOptimizer(); z.set_grads()
        r = run_boundary(z)
        t = r["gradient_identity_and_coverage_vs_recomputation"]
        self.assertIn("OBSERVED", t)
        self.assertIn("not an independent recomputation", t)
        self.assertIn("Active dropout does not remove", t)
        self.assertIn("CONDITIONAL on the observed", r["establishes"])

    def test_the_verdict_declares_what_it_does_not_establish(self):
        z = FakeZeroOptimizer(); z.set_grads()
        r = run_boundary(z)
        joined = " ".join(r["does_not_establish"]).lower()
        self.assertIn("gradient correctness", joined)
        self.assertIn("historical run", joined)
        self.assertIn("conditional", r["establishes"].lower())


class RequiredNegatives(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_wrong_model_binding_is_detected(self):
        m512 = F.build_model(seed=1234, device="cpu")[0]
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(1234)
        other = GPT2LMHeadModel(GPT2Config(n_positions=512))
        a = F.resolve_model_identity(m512, GPT2Config())
        b = F.resolve_model_identity(other, GPT2Config(n_positions=512))
        self.assertNotEqual(a["sum_of_named_parameter_numel"],
                            b["sum_of_named_parameter_numel"])
        self.assertEqual(b["sum_of_named_parameter_numel"], 124_046_592)

    def test_wrong_input_binding_is_detected(self):
        g = F.build_generator(seed=1234, rank=0, device="cpu")
        good = F.input_digest(F.synthetic_batch(2, 8, "cpu", g))
        g2 = F.build_generator(seed=999, rank=0, device="cpu")
        bad = F.input_digest(F.synthetic_batch(2, 8, "cpu", g2))
        self.assertNotEqual(good["realized_input_sha256"], bad["realized_input_sha256"])

    def test_wrong_accumulation_normalisation_is_refused(self):
        z = FakeZeroOptimizer(); z.set_grads()
        with self.assertRaises(EP.ProbeRefusal):
            run_boundary(z, accumulation=4)

    def test_missing_gradient_is_refused(self):
        z = FakeZeroOptimizer(); z.set_grads(none=True)
        with self.assertRaises(EP.ProbeRefusal):
            run_boundary(z)

    def test_nonfinite_gradient_is_refused(self):
        z = FakeZeroOptimizer(); z.set_grads(nonfinite=True)
        with self.assertRaises(EP.ProbeRefusal):
            run_boundary(z)

    def test_padding_exceeding_partition_size_is_refused(self):
        z = FakeZeroOptimizer(group_numels=(8, 8), padding=[99, 0])
        z.set_grads()
        with self.assertRaises(EP.ProbeRefusal):
            run_boundary(z, world_size=1)

    def test_partition_numel_mismatch_is_refused(self):
        z = FakeZeroOptimizer(group_numels=(16, 8)); z.set_grads()
        z.partition_size = [99, 8]                 # layout lies about the size
        with self.assertRaises(EP.ProbeRefusal):
            run_boundary(z)

    def test_absent_partition_size_is_refused_not_inferred(self):
        z = FakeZeroOptimizer(); z.set_grads()
        del z.partition_size
        with self.assertRaises(EP.ProbeRefusal):
            EP.expected_partition_map(z, rank=0, world_size=1)

    def test_absent_padding_is_refused(self):
        z = FakeZeroOptimizer(); z.set_grads()
        del z.groups_padding
        with self.assertRaises(EP.ProbeRefusal):
            EP.expected_partition_map(z, rank=0, world_size=1)

    def test_unsupported_layout_without_masters_is_refused(self):
        z = FakeZeroOptimizer(); z.single_partition_of_fp32_groups = []
        with self.assertRaises(EP.ProbeRefusal):
            EP.describe_optimizer(z)

    def test_storage_correspondence_failure_is_refused(self):
        z = FakeZeroOptimizer(); z.set_grads()
        exp = EP.expected_partition_map(z, rank=0, world_size=1)
        before = EP.capture_boundary(z, rank=0, world_size=1, phase="before")
        z.step()
        after = EP.capture_boundary(z, rank=0, world_size=1, phase="after")
        bad = {"base_optimizer_sees_the_master_partitions": False,
               "masters_absent_from_base_optimizer": [0]}
        with self.assertRaises(EP.ProbeRefusal):
            V.verify_boundary(before=before, after=after, expected_map=exp,
                              resolved_hparams=dict(HP), accumulation_steps=1,
                              rank=0, world_size=1, storage_check=bad,
                              module_check={"checked": False})

    def test_storage_correspondence_detects_a_decoy_optimizer(self):
        """Same lengths, same class name, different tensors."""
        z = FakeZeroOptimizer()
        decoy = [torch.nn.Parameter(torch.randn_like(m))
                 for m in z.single_partition_of_fp32_groups]
        z.optimizer = torch.optim.AdamW(decoy, lr=HP["lr"])
        sc = EP.verify_storage_correspondence(z, rank=0)
        self.assertFalse(sc["base_optimizer_sees_the_master_partitions"])

    def test_noop_update_is_detected(self):
        z = FakeZeroOptimizer(); z.set_grads()
        exp = EP.expected_partition_map(z, rank=0, world_size=1)
        sc = EP.verify_storage_correspondence(z, rank=0)
        before = EP.capture_boundary(z, rank=0, world_size=1, phase="before")
        z.optimizer.step()                        # advance state
        for i, m in enumerate(z.single_partition_of_fp32_groups):
            with torch.no_grad():
                m.copy_(before["groups"][i]["master"])   # then undo the movement
        after = EP.capture_boundary(z, rank=0, world_size=1, phase="after")
        r = V.verify_boundary(before=before, after=after, expected_map=exp,
                              resolved_hparams=dict(HP), accumulation_steps=1,
                              rank=0, world_size=1, storage_check=sc,
                              module_check={"checked": False})
        self.assertFalse(r["rank_verification_successful"])

    def test_corrupted_optimizer_state_is_detected(self):
        z = FakeZeroOptimizer(); z.set_grads()
        exp = EP.expected_partition_map(z, rank=0, world_size=1)
        sc = EP.verify_storage_correspondence(z, rank=0)
        before = EP.capture_boundary(z, rank=0, world_size=1, phase="before")
        z.step()
        after = EP.capture_boundary(z, rank=0, world_size=1, phase="after")
        after["groups"][0]["exp_avg"] = torch.full_like(after["groups"][0]["exp_avg"], 5.0)
        r = V.verify_boundary(before=before, after=after, expected_map=exp,
                              resolved_hparams=dict(HP), accumulation_steps=1,
                              rank=0, world_size=1, storage_check=sc,
                              module_check={"checked": False})
        self.assertFalse(r["rank_verification_successful"])

    def test_wrong_decay_mode_is_detected(self):
        z = FakeZeroOptimizer(); z.set_grads()
        r = run_boundary(z, hp={**HP, "decoupled_weight_decay": False})
        self.assertFalse(r["rank_verification_successful"])

    def test_unresolved_optimizer_options_are_refused(self):
        z = FakeZeroOptimizer(); z.set_grads()
        for drop in ("weight_decay", "decoupled_weight_decay", "lr", "eps"):
            hp = {k: v for k, v in HP.items() if k != drop}
            with self.subTest(drop=drop), self.assertRaises(OR.OracleRefusal):
                run_boundary(FakeZeroOptimizer(), hp=hp) if False else None
                z2 = FakeZeroOptimizer(); z2.set_grads()
                run_boundary(z2, hp=hp)

    def test_a_skipped_base_call_is_not_a_success(self):
        z = FakeZeroOptimizer(); z.set_grads()
        exp = EP.expected_partition_map(z, rank=0, world_size=1)
        sc = EP.verify_storage_correspondence(z, rank=0)
        before = EP.capture_boundary(z, rank=0, world_size=1, phase="before")
        z.optimizer.step()
        after = EP.capture_boundary(z, rank=0, world_size=1, phase="after")
        after["groups"][0]["state_step"] = before["groups"][0]["state_step"] or 0
        r = V.verify_boundary(before=before, after=after, expected_map=exp,
                              resolved_hparams=dict(HP), accumulation_steps=1,
                              rank=0, world_size=1, storage_check=sc,
                              module_check={"checked": False})
        self.assertGreaterEqual(r["skipped_updates"], 1)
        self.assertFalse(r["rank_verification_successful"])

    def test_non_discriminating_result_cannot_be_promoted(self):
        """A movement below the resolution of the comparison.

        A tiny GRADIENT does not produce a tiny update: Adam normalises by the
        second moment, so grad=1e-8 at |p|=1e6 still moves about 1.0 once
        decoupled decay is included. The non-discriminating regime needs a tiny
        intended MOVEMENT, i.e. an lr below the comparison tolerance but above
        the representable step at that magnitude. At |p|=1e6 in float64 the
        tolerance is about 1.42e-8 and the ULP about 2.3e-10, so lr=1e-9 lands
        strictly between: nonzero, yet unresolvable. An lr of 1e-20 would
        underflow to exactly zero and is correctly treated as a genuinely zero
        update instead.
        """
        z = FakeZeroOptimizer(group_numels=(4,))
        with torch.no_grad():
            z.single_partition_of_fp32_groups[0].fill_(1e6)
        for g in z.optimizer.param_groups:
            g["lr"] = 1e-9
            g["weight_decay"] = 0.0
        z.set_grads(scale=0.01)
        hp = {**HP, "lr": 1e-9, "weight_decay": 0.0}
        r = run_boundary(z, hp=hp)
        part = r["per_partition"][0]
        self.assertGreater(part["intended_movement"], 0.0,
                           "a zero intended update is a different, passing case")
        self.assertLess(part["intended_movement"], part["tolerance"],
                        "precondition: the intended movement must sit below the "
                        "comparison resolution")
        self.assertFalse(part["discriminating"])
        self.assertFalse(part["verification_successful"],
                         "agreement that cannot distinguish a no-op is not success")
        self.assertEqual(r["non_discriminating_partitions"], [(0, 0)])
        self.assertFalse(r["rank_verification_successful"])

    def test_a_large_parameter_with_a_tiny_gradient_still_discriminates(self):
        """The companion fact: Adam's normalisation keeps such a step resolvable,
        so it must NOT be labelled non-discriminating."""
        z = FakeZeroOptimizer(group_numels=(4,))
        with torch.no_grad():
            z.single_partition_of_fp32_groups[0].fill_(1e6)
        z.set_grads(scale=1e-8)
        r = run_boundary(z)
        self.assertTrue(r["per_partition"][0]["discriminating"])
        self.assertTrue(r["rank_verification_successful"])

    def test_missing_rank_refuses_whole_model(self):
        z = FakeZeroOptimizer(world_size=4); z.set_grads()
        one = run_boundary(z, rank=0, world_size=4)
        whole = V.combine_ranks([one], world_size=4)
        self.assertFalse(whole["whole_model_verified"])
        self.assertEqual(whole["missing_ranks"], [1, 2, 3])

    def test_duplicate_rank_refuses_whole_model(self):
        rs = []
        for _ in range(2):
            z = FakeZeroOptimizer(world_size=2); z.set_grads()
            rs.append(run_boundary(z, rank=0, world_size=2))
        whole = V.combine_ranks(rs, world_size=2)
        self.assertFalse(whole["whole_model_verified"])
        self.assertEqual(whole["duplicate_ranks"], [0])

    def test_unexpected_rank_refuses_whole_model(self):
        z = FakeZeroOptimizer(world_size=2); z.set_grads()
        r = run_boundary(z, rank=0, world_size=2)
        r["rank"] = 7
        whole = V.combine_ranks([r], world_size=2)
        self.assertFalse(whole["whole_model_verified"])
        self.assertEqual(whole["unexpected_ranks"], [7])

    def test_sampled_coverage_refuses_whole_model(self):
        z = FakeZeroOptimizer(); z.set_grads()
        r = run_boundary(z, coverage=OR.SAMPLED)
        whole = V.combine_ranks([r], world_size=1)
        self.assertFalse(whole["whole_model_verified"])
        self.assertEqual(whole["sampled_ranks"], [0])

    def test_module_not_reflecting_masters_is_reported(self):
        z = FakeZeroOptimizer(working_dtype=torch.float16); z.set_grads()
        exp = EP.expected_partition_map(z, rank=0, world_size=1)
        sc = EP.verify_storage_correspondence(z, rank=0)
        before = EP.capture_boundary(z, rank=0, world_size=1, phase="before")
        z.optimizer.step()                        # masters move, working copy not refreshed
        after = EP.capture_boundary(z, rank=0, world_size=1, phase="after")
        with torch.no_grad():
            z.fp16_groups_flat[0].fill_(0.0)
        mc = EP.module_reflects_masters(z)
        self.assertTrue(mc["checked"])
        self.assertFalse(mc["verdict_successful"])

    def test_absent_working_view_is_unchecked_not_passed(self):
        z = FakeZeroOptimizer(); z.set_grads()
        mc = EP.module_reflects_masters(z)
        self.assertFalse(mc["checked"])
        self.assertIn("NOT established", mc["reason"])

    def test_bad_phase_label_is_refused(self):
        z = FakeZeroOptimizer(); z.set_grads()
        with self.assertRaises(EP.ProbeRefusal):
            EP.capture_boundary(z, rank=0, world_size=1, phase="midway")

    def test_group_count_change_across_the_step_is_refused(self):
        z = FakeZeroOptimizer(); z.set_grads()
        exp = EP.expected_partition_map(z, rank=0, world_size=1)
        sc = EP.verify_storage_correspondence(z, rank=0)
        before = EP.capture_boundary(z, rank=0, world_size=1, phase="before")
        z.step()
        after = EP.capture_boundary(z, rank=0, world_size=1, phase="after")
        after["groups"] = after["groups"][:1]
        with self.assertRaises(EP.ProbeRefusal):
            V.verify_boundary(before=before, after=after, expected_map=exp,
                              resolved_hparams=dict(HP), accumulation_steps=1,
                              rank=0, world_size=1, storage_check=sc,
                              module_check={"checked": False})


if __name__ == "__main__":
    unittest.main()
