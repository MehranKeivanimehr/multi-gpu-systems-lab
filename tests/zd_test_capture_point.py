"""The boundary must be captured where the base optimizer consumes the gradient.

The first full hardware attempt refused with

    ProbeRefusal: group 0: master.grad is None before the step, so the gradient
    the optimizer consumes was not observed

and the refusal was CORRECT: in DeepSpeed 0.15.1 ZeRO-1 the master partition's
`.grad` is assigned, clipped, consumed and freed entirely inside
`DeepSpeedZeroOptimizer.step()` (stage_1_and_2.py:1899-1913), so a capture taken
around `engine.step()` can never see it.

`ZeroLikeStep` below reproduces that lifetime on CPU: `.grad` is None outside the
step, assigned just before the base optimizer runs, and cleared immediately
after. The tests show the old capture point refuses against it and the new one
observes the consumed gradient.
"""
import pathlib, sys, unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
from zerodiag import engine_probe as EP          # noqa: E402

HP = {"lr": 1e-2, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
      "weight_decay": 0.01, "decoupled_weight_decay": True, "amsgrad": False}


class ZeroLikeStep:
    """Mimics the real ZeRO-1 gradient lifetime, on CPU, with a real AdamW."""

    def __init__(self, numels=(16, 8), clip=0.5):
        self.world_size = 1
        self.partition_size = list(numels)
        self.groups_padding = [0] * len(numels)
        self.single_partition_of_fp32_groups = [
            torch.nn.Parameter(torch.randn(n, dtype=torch.float64))
            for n in numels]
        self.optimizer = torch.optim.AdamW(
            self.single_partition_of_fp32_groups, lr=HP["lr"],
            weight_decay=HP["weight_decay"])
        self.overflow = False
        self.bit16_groups = [[p] for p in self.single_partition_of_fp32_groups]
        self._pending = [torch.randn_like(m) * 0.05
                         for m in self.single_partition_of_fp32_groups]
        self.clip = clip
        self.clipped = None

    def step(self):
        """Assign -> clip -> base step -> free, exactly as ZeRO-1 does."""
        for m, g in zip(self.single_partition_of_fp32_groups, self._pending):
            m.grad = g.clone()                       # :1899
        self.clipped = []
        for m in self.single_partition_of_fp32_groups:
            m.grad.mul_(self.clip)                   # :1905 unscale_and_clip
            self.clipped.append(m.grad.detach().clone())
        self.optimizer.step()                        # :1911
        for m in self.single_partition_of_fp32_groups:
            m.grad = None                            # :1913


class CapturePoint(unittest.TestCase):

    def test_the_old_capture_point_refuses_as_it_did_on_hardware(self):
        z = ZeroLikeStep()
        self.assertIsNone(z.single_partition_of_fp32_groups[0].grad)
        with self.assertRaises(EP.ProbeRefusal) as cm:
            EP.capture_boundary(z, rank=0, world_size=1, phase="before",
                                chunk_elems=8)
        self.assertIn("master.grad is None before the step", str(cm.exception))

    def test_the_new_capture_point_observes_the_consumed_gradient(self):
        z = ZeroLikeStep()
        with EP.boundary_capture(z, rank=0, world_size=1, chunk_elems=8) as box:
            z.step()
        self.assertEqual(box["base_step_calls"], 1)
        self.assertIsNotNone(box["before"])
        self.assertIsNotNone(box["after"])
        g = box["before"]["groups"][0]["grad"]
        self.assertIsNotNone(g, "the consumed gradient was not observed")
        self.assertEqual(g.device.type, "cpu")

    def test_the_observed_gradient_is_post_clip_not_pre_clip(self):
        z = ZeroLikeStep(clip=0.25)
        with EP.boundary_capture(z, rank=0, world_size=1, chunk_elems=8) as box:
            z.step()
        observed = box["before"]["groups"][0]["grad"]
        self.assertTrue(torch.allclose(observed.double(), z.clipped[0].double()),
                        "observed gradient does not match the clipped value")
        pre_clip = z._pending[0].double()
        self.assertFalse(torch.allclose(observed.double(), pre_clip),
                         "observed gradient equals the PRE-clip value")

    def test_it_is_not_reconstructed_from_averaged_gradients(self):
        z = ZeroLikeStep()
        with EP.boundary_capture(z, rank=0, world_size=1, chunk_elems=8) as box:
            z.step()
        self.assertFalse(box["reconstructed_from_averaged_gradients"])
        self.assertIn("post-unscale, post-clip", box["observed_gradient"])
        self.assertIn("base optimizer", box["capture_point"])

    def test_the_step_is_restored_even_on_failure(self):
        """A bound method is recreated on each access, so compare __func__."""
        z = ZeroLikeStep()
        self.assertIs(z.optimizer.step.__func__, torch.optim.AdamW.step)
        box = None
        try:
            with EP.boundary_capture(z, rank=0, world_size=1,
                                     chunk_elems=8) as b:
                box = b
                self.assertEqual(z.optimizer.step.__name__, "wrapped")
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertIs(z.optimizer.step.__func__, torch.optim.AdamW.step,
                      "the base optimizer's step was not restored")
        n = box["base_step_calls"]
        z.step()
        self.assertEqual(box["base_step_calls"], n,
                         "the wrapper is still intercepting after restore")

    def test_snapshots_are_independent_of_the_live_state(self):
        z = ZeroLikeStep()
        with EP.boundary_capture(z, rank=0, world_size=1, chunk_elems=8) as box:
            z.step()
        before = box["before"]["groups"][0]["master"].clone()
        z.single_partition_of_fp32_groups[0].data.mul_(-7.0)
        self.assertTrue(torch.equal(box["before"]["groups"][0]["master"], before))

    def test_master_moves_between_before_and_after(self):
        """The captured pair must straddle a real update."""
        z = ZeroLikeStep()
        with EP.boundary_capture(z, rank=0, world_size=1, chunk_elems=8) as box:
            z.step()
        b = box["before"]["groups"][0]["master"]
        a = box["after"]["groups"][0]["master"]
        self.assertFalse(torch.equal(a, b), "no update occurred across the boundary")
        self.assertEqual(box["before"]["groups"][0]["state_step"], None)
        self.assertEqual(box["after"]["groups"][0]["state_step"], 1)

    def test_more_than_one_base_call_is_visible_to_the_caller(self):
        """The worker refuses on this; the probe must report it faithfully."""
        z = ZeroLikeStep()
        with EP.boundary_capture(z, rank=0, world_size=1, chunk_elems=8) as box:
            z.step()
            z._pending = [torch.randn_like(m) * 0.05
                          for m in z.single_partition_of_fp32_groups]
            z.step()
        self.assertEqual(box["base_step_calls"], 2)

    def test_worker_refuses_on_a_wrong_base_call_count(self):
        """The expected count is DERIVED per layout, not fixed at one."""
        src = (ROOT / "zerodiag/worker.py").read_text()
        self.assertIn('if box["base_step_calls"] != _n_groups:', src)
        self.assertIn("fp32 master group(s)", src)
        self.assertIn("the base optimizer was never called", src)
        self.assertNotIn('if box["base_step_calls"] != 1:', src,
                         "Gate-0's single-call rule must not be generalised")
        self.assertIn("len(single_partition_of_fp32_groups)", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
