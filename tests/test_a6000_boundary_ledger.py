"""CPU counterexamples for the boundary ledger and sampled update evidence (stage 20261016s).

Real torch.optim.AdamW on CPU tensors; no CUDA, NCCL or DeepSpeed execution. These show that the accounting refuses
to call something verified when it is not - they do not show behaviour on the GPU node.
"""
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "studies/a6000/src"

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class Base(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(SRC))
        self.addCleanup(sys.path.remove, str(SRC))
        from mgbench.train import boundary_ledger as bl
        from mgbench.train import update_evidence as ue
        self.bl, self.ue = bl, ue
        torch.manual_seed(0)

    def model_opt(self, wd=0.01, cls=None):
        model = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Linear(5, 3))
        opt = (cls or torch.optim.AdamW)(model.parameters(), lr=1e-3, weight_decay=wd)
        return model, opt

    def native_ledger(self, opt, checks=True, sampler=True):
        units = {(gi, pi): p for gi, g in enumerate(opt.param_groups) for pi, p in enumerate(g["params"])}
        led = self.bl.BoundaryLedger("ddp", checks, "in_step", {k: p.numel() for k, p in units.items()}, 1, "native")
        smp = self.ue.AdamWCallSampler() if sampler else None

        def pre(o, a, k):
            if smp:
                smp.pre(o)

        def post(o, a, k):
            ev = None
            if smp:
                smp.post(o)
                ev = smp.calls[-1]
            led.base_call(ev)
        opt.register_step_pre_hook(pre)
        opt.register_step_post_hook(post)
        return led, units

    def run_steps(self, model, opt, led, units, *, warmup=1, measured=3, ga=2, skip=(), drop_grad=None, call_twice=(),
            record_losses=True):
        for phase, n in (("warmup", warmup), ("measured", measured)):
            if phase == "measured":
                led.start_measured()
            for i in range(n):
                led.begin_step()
                raw = []
                for _ in range(ga):
                    loss = model(torch.randn(4, 6)).pow(2).mean()
                    (loss / ga).backward()
                    raw.append(loss.detach())
                if drop_grad is not None and phase == "measured":
                    list(model.parameters())[drop_grad].grad = None
                if record_losses:
                    led.record_microbatch_losses(raw)
                if led.gradient_checks:
                    led.observe_units(self.bl.scan_units(units), "in-step")
                calls = 0
                if not (phase == "measured" and i in skip):
                    opt.step()
                    calls += 1
                    if phase == "measured" and i in call_twice:
                        opt.step()
                opt.zero_grad(set_to_none=True)
                led.end_step(calls > 0, "post-hook")
        return led.summary(measured, warmup, ga)

    def assess(self, summary, checks=True):
        return self.bl.assess([summary], checks_requested=checks, update_evidence_requested=checks,
                              measurement_role="smoke", diagnostics_in_timed_path=checks)


class Accounting(Base):
    def test_clean_run_counts_every_category_separately(self):
        model, opt = self.model_opt()
        led, units = self.native_ledger(opt)
        s = self.run_steps(model, opt, led, units)
        w, m = s["warmup"], s["measured"]
        self.assertEqual((w["logical_attempts"], w["microbatches"]), (1, 2))
        self.assertEqual((m["logical_attempts"], m["microbatches"], m["inspected_boundaries"],
                          m["base_optimizer_calls"], m["counter_updates"]), (3, 6, 3, 3, 3))
        self.assertEqual(m["boundary_coverage"], ["complete"] * 3)
        self.assertEqual(s["expected_units"], 4)
        a = self.assess(s)
        self.assertEqual(a["execution"], "ok")
        self.assertEqual(a["gradient"]["status"], "complete_at_every_measured_boundary")
        self.assertEqual(a["update_state"]["status"], "verified_sampled_every_base_call")
        self.assertTrue(a["measurement_eligibility"].startswith("ineligible"))

    def test_missing_gradient_is_partial_not_complete(self):
        model, opt = self.model_opt()
        led, units = self.native_ledger(opt)
        s = self.run_steps(model, opt, led, units, drop_grad=1)
        self.assertEqual(s["measured"]["boundary_coverage"], ["partial"] * 3)
        self.assertEqual(self.assess(s)["gradient"]["status"], "incomplete_or_unaccounted")

    def test_empty_and_short_partition_units(self):
        led = self.bl.BoundaryLedger("zero2", True, "base_optimizer_pre_hook", {"group0": 100}, 1, "impl")
        for units, want in (({}, "empty"), ({"group0": (0, None)}, "empty"), ({"group0": (60, True)}, "partial"),
                            ({"group0": (100, True), ("unexpected", 1): (5, True)}, "partial"),
                            ({"group0": (100, True)}, "complete")):
            led.begin_step()
            led.observe_units(units, "hook")
            led.end_step(True, "engine")
            self.assertEqual(led.phases["warmup"][-1]["coverage"]["state"], want, units)

    def test_nonempty_tensors_without_accounting_are_not_complete(self):
        led = self.bl.BoundaryLedger("zero3", True, "base_optimizer_pre_hook", None, None, "no accounting")
        led.begin_step()
        led.observe_units({"g": (10, True)}, "hook")
        led.end_step(True, "engine")
        self.assertEqual(led.phases["warmup"][0]["coverage"]["state"], "no_expected_accounting")

    def test_repeated_subgroup_call_is_flagged_and_breaks_multiplicity(self):
        led = self.bl.BoundaryLedger("zero2", True, "base_optimizer_pre_hook", {"group0": 10}, 1, "impl")
        led.start_measured()
        for i in range(3):
            led.begin_step()
            led.record_microbatch_losses([1.0, 1.0])
            for _ in range(2 if i == 1 else 1):
                led.observe_units({"group0": (10, True)}, "hook")
                led.base_call({"verdict": "verified_sampled", "sampled_elements": 1, "stepped_elements": 10})
            led.end_step(True, "engine")
        led.phases["warmup"] = [{"microbatch_losses": [1.0, 1.0], "coverage": {"state": "complete"},
                                 "base_optimizer_calls": 1, "update_evidence": [], "counter_update": True,
                                 "framework_reported_skip": False, "observations": []}]
        s = led.summary(3, 1, 2)
        self.assertEqual(s["measured"]["boundary_coverage"][1], "repeated")
        a = self.assess(s)
        self.assertEqual(a["execution"], "problem_detected")
        self.assertTrue(any("multiplicity" in r for r in a["execution_reasons"]))

    def test_finite_zero_gradients_are_present_gradients(self):
        model, opt = self.model_opt()
        led, units = self.native_ledger(opt)
        led.start_measured()
        led.begin_step()
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        self.assertEqual(self.bl.scan_units(units)[(0, 0)], (30, True))
        led.observe_units(self.bl.scan_units(units), "in-step")
        opt.step()
        led.end_step(True, "post-hook")
        self.assertEqual(led.phases["measured"][0]["coverage"]["state"], "complete")
        self.assertEqual(led.phases["measured"][0]["update_evidence"][0]["verdict"], "verified_sampled")

    def test_returned_mean_alone_is_not_microbatch_observation(self):
        model, opt = self.model_opt()
        led, units = self.native_ledger(opt)
        s = self.run_steps(model, opt, led, units, record_losses=False)
        a = self.assess(s)
        self.assertEqual(a["execution"], "problem_detected")
        self.assertEqual(s["measured"]["microbatches"], 0)

    def test_skipped_call_is_not_an_update(self):
        model, opt = self.model_opt()
        led, units = self.native_ledger(opt)
        s = self.run_steps(model, opt, led, units, skip=(1,))
        self.assertEqual((s["measured"]["inspected_boundaries"], s["measured"]["counter_updates"],
                          s["measured"]["base_optimizer_calls"]), (3, 2, 2))
        self.assertEqual(self.assess(s)["execution"], "problem_detected")

    def test_unchecked_timing_path_reports_null_validity(self):
        model, opt = self.model_opt()
        led, units = self.native_ledger(opt, checks=False, sampler=False)
        s = self.run_steps(model, opt, led, units)
        a = self.bl.assess([s], checks_requested=False, update_evidence_requested=False, measurement_role="measurement",
                           diagnostics_in_timed_path=False)
        self.assertEqual(a["gradient"]["status"], "not_checked")
        self.assertIsNone(a["gradient"]["gradient_validity"])
        self.assertEqual(a["update_state"]["status"], "not_collected")
        self.assertTrue(a["measurement_eligibility"].startswith("unvalidated"))
        self.assertEqual(s["measured"]["inspected_boundaries"], 0)


class UpdateEvidence(Base):
    def test_no_op_optimizer_with_firing_hooks_fails(self):
        class NoOpAdamW(torch.optim.AdamW):
            @torch.no_grad()
            def step(self, closure=None):        # hooks still fire around this call
                return None
        model, opt = self.model_opt(cls=NoOpAdamW)
        led, units = self.native_ledger(opt)
        s = self.run_steps(model, opt, led, units)
        self.assertEqual(s["measured"]["base_optimizer_calls"], 3)            # equal call counts...
        self.assertEqual(self.assess(s)["update_state"]["status"], "failed")   # ...do not prove the update

    def test_zero_change_passes_when_the_recomputed_change_is_zero(self):
        model, opt = self.model_opt(wd=0.0)
        smp = self.ue.AdamWCallSampler()
        opt.register_step_pre_hook(lambda o, a, k: smp.pre(o))
        opt.register_step_post_hook(lambda o, a, k: smp.post(o))
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()                                     # zero grad, zero moments, no decay: zero change is correct
        self.assertEqual(smp.calls[-1]["verdict"], "verified_sampled")
        self.assertGreater(smp.calls[-1]["checks"][0]["zero_change_elements"], 0)

    def test_corrupted_update_fails(self):
        model, opt = self.model_opt()
        smp = self.ue.AdamWCallSampler()
        opt.register_step_pre_hook(lambda o, a, k: smp.pre(o))
        opt.register_step_post_hook(lambda o, a, k: [p.data.add_(1e-3) for g in o.param_groups for p in g["params"]])
        opt.register_step_post_hook(lambda o, a, k: smp.post(o))     # runs after the corruption
        model(torch.randn(2, 6)).sum().backward()
        opt.step()
        self.assertEqual(smp.calls[-1]["verdict"], "failed")

    def test_sampling_is_labelled_with_exact_coverage(self):
        model, opt = self.model_opt()
        smp = self.ue.AdamWCallSampler()
        opt.register_step_pre_hook(lambda o, a, k: smp.pre(o))
        opt.register_step_post_hook(lambda o, a, k: smp.post(o))
        model(torch.randn(2, 6)).sum().backward()
        opt.step()
        c = smp.calls[-1]
        self.assertEqual(c["stepped_elements"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(c["sampled_tensors"], 3)
        self.assertLessEqual(c["sampled_elements"], c["stepped_elements"])
        self.assertEqual(c["verdict"], "verified_sampled")

    def test_unsupported_optimizer_is_unverified_not_passed(self):
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        smp = self.ue.AdamWCallSampler()
        opt.register_step_pre_hook(lambda o, a, k: smp.pre(o))
        opt.register_step_post_hook(lambda o, a, k: smp.post(o))
        model(torch.randn(2, 3)).sum().backward()
        opt.step()
        self.assertEqual(smp.calls[-1]["verdict"], "unverified")


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class DeepSpeedLayoutAccounting(unittest.TestCase):
    """deepspeed_zero12_layout() on a stand-in object shaped like the installed DeepSpeedZeroOptimizer."""

    def setUp(self):
        sys.path.insert(0, str(SRC))
        self.addCleanup(sys.path.remove, str(SRC))
        from mgbench.train import frameworks
        self.fw = frameworks

    def fake(self, mismatch=False):
        # stage 20261016s2: the layout also needs an independent model -> bit16-group map (engine.module)
        class DeepSpeedZeroOptimizer:
            pass
        zo = DeepSpeedZeroOptimizer()
        module = torch.nn.ModuleDict({"a": torch.nn.Linear(3, 2, bias=False), "b": torch.nn.Linear(7, 1, bias=False)})
        pa, pb = module["a"].weight, module["b"].weight                  # 6 and 7 elements
        parts = [torch.zeros(6), torch.zeros(7)]
        zo.bit16_groups = [[pa], [pb]]
        zo.bit16_groups_flat = [torch.zeros(6), torch.zeros(7)]
        zo.partition_size = [6, 7]
        zo.nccl_start_alignment_factor = 2
        zo.single_partition_of_fp32_groups = parts
        zo.real_dp_process_group = [None, None]
        opt = types.SimpleNamespace(param_groups=[{"params": [parts[0]]}, {"params": [torch.zeros(7) if mismatch else parts[1]]}])
        return types.SimpleNamespace(optimizer=zo, module=module), opt

    def test_identity_verified_layout_and_multiplicity(self):
        engine, opt = self.fake()
        with mock.patch.object(self.fw.dist, "get_rank", lambda group=None: 0), \
                mock.patch.object(self.fw.dist, "get_world_size", lambda group=None: 1):
            lay = self.fw.deepspeed_zero12_layout(engine, opt)
        self.assertEqual(lay["accounting"], {"group0": 6, "group1": 7})
        self.assertEqual(lay["calls_per_boundary"], 2)

    def test_identity_mismatch_gives_no_accounting(self):
        engine, opt = self.fake(mismatch=True)
        with mock.patch.object(self.fw.dist, "get_rank", lambda group=None: 0), \
                mock.patch.object(self.fw.dist, "get_world_size", lambda group=None: 1):
            lay = self.fw.deepspeed_zero12_layout(engine, opt)
        self.assertIsNone(lay["accounting"])


class TrainerWiringV2(unittest.TestCase):
    def test_native_trainers_record_microbatches_and_scan_before_the_call(self):
        src = (SRC / "mgbench/train/frameworks.py").read_text()
        for setup in ("def setup_ddp", "def setup_fsdp"):
            body = src[src.index(setup):]
            body = body[:body.index("\ndef ", 1)]
            self.assertLess(body.index("ledger.record_microbatch_losses(raw)"), body.index("opt.step()"))
            self.assertLess(body.index("scan_units(units)"), body.index("opt.step()"))
            self.assertIn("attach_boundary_hooks(opt, ledger", body)

    def test_deepspeed_uses_implementation_derived_accounting(self):
        src = (SRC / "mgbench/train/frameworks.py").read_text()
        self.assertIn("deepspeed_zero12_layout(engine, opt)", src)
        self.assertIn("units_in_hook=ds_units", src)
        main = (SRC / "mgbench/train/__main__.py").read_text()
        self.assertIn("trainer.ledger.start_measured()", main)
        self.assertIn('"assessment": assessment,', main)


if __name__ == "__main__":
    unittest.main()
