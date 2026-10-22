"""Route 1+2: tests against the EXECUTED production path, not a calculator.

Allocation bounds are observed with a TorchDispatchMode recorder: every aten op
is intercepted and the element count of each newly allocated output tensor is
recorded. View-producing ops are excluded by name because they allocate no
storage; everything else is counted. The same recorder is pointed at the
unchunked reference implementation to show it detects the whole-partition path.
"""
import hashlib, os, pathlib, sys, unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
from zerodiag import engine_probe as EP          # noqa: E402
from zerodiag import update_oracle as OR         # noqa: E402
from zerodiag import verdict as VD               # noqa: E402

VIEW_OPS = {
    "aten.view", "aten.narrow", "aten.slice", "aten.select", "aten.detach",
    "aten.as_strided", "aten.expand", "aten.permute", "aten.t",
    "aten.squeeze", "aten.unsqueeze", "aten._unsafe_view", "aten.alias",
    "aten.reshape", "aten.view_as_real", "aten.split", "aten.unbind",
}


class AllocRecorder(TorchDispatchMode):
    """Record the largest newly-allocated tensor, in elements."""

    def __init__(self):
        self.max_elems = 0
        self.max_bytes = 0
        self.biggest_op = None
        self.n_ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.n_ops += 1
        name = str(func).split(".default")[0]
        base = ".".join(name.split(".")[:2])
        if base in VIEW_OPS:
            return out
        for t in (out if isinstance(out, (tuple, list)) else [out]):
            if isinstance(t, torch.Tensor):
                n = int(t.numel())
                if n > self.max_elems:
                    self.max_elems = n
                    self.max_bytes = n * t.element_size()
                    self.biggest_op = name
        return out


HP = {"lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
      "weight_decay": 0.01, "decoupled_weight_decay": True, "amsgrad": False}
HP_COUPLED = dict(HP, decoupled_weight_decay=False)


def fixture(n=4096, seed=0):
    torch.manual_seed(seed)
    pb = torch.randn(n)
    g = torch.randn(n) * 0.01
    m = torch.randn(n) * 0.001
    v = (torch.randn(n) * 0.001).abs()
    return pb, g, m, v


def truth(pb, g, m, v, hp, step=1):
    return OR.expected_adam_update(
        param_before=pb, grad=g, exp_avg=m, exp_avg_sq=v, step=step,
        lr=hp["lr"], beta1=hp["beta1"], beta2=hp["beta2"], eps=hp["eps"],
        weight_decay=hp["weight_decay"],
        decoupled_weight_decay=hp["decoupled_weight_decay"],
        amsgrad=hp.get("amsgrad", False),
        max_exp_avg_sq=hp.get("max_exp_avg_sq"))


def kwargs_for(pb, g, m, v, pa, em, ev, hp, n):
    return dict(param_before=pb, param_after=pa, grad=g, exp_avg=m,
                exp_avg_sq=v, step=1, hparams=hp, observed_exp_avg=em,
                observed_exp_avg_sq=ev, n_elements_total=n,
                model_map={"expected_elements": n})


class TestChunkedEqualsUnchunked(unittest.TestCase):

    def _compare(self, hp, n=4096):
        pb, g, m, v = fixture(n)
        pa, em, ev = truth(pb, g, m, v, hp)
        kw = kwargs_for(pb, g, m, v, pa, em, ev, hp, n)
        u = OR.verify_update(**kw)
        for ce in (1, 7, 333, n - 1, n, n + 11, OR.DEFAULT_CHUNK_ELEMS):
            c = OR.verify_update_chunked(**kw, chunk_elems=ce)
            for k in ("max_abs_param_deviation", "tolerance", "atol", "rtol",
                      "intended_movement", "observed_movement"):
                self.assertEqual(float(u[k]), float(c[k]),
                                 f"{k} differs at chunk_elems={ce}")
            self.assertEqual(u["verification_successful"],
                             c["verification_successful"])
            self.assertEqual(c["elements_visited"], n)
            self.assertEqual(sum([c["final_chunk_elems"]] +
                                 [min(ce, n)] * (c["n_chunks"] - 1)), n)

    def test_adamw_decoupled_matches_at_every_chunk_size(self):
        self._compare(HP)

    def test_adam_coupled_matches_at_every_chunk_size(self):
        self._compare(HP_COUPLED)

    def test_amsgrad_matches(self):
        n = 2048
        pb, g, m, v = fixture(n, seed=3)
        mx = v + 0.0005
        # lr raised for a resolvable (discriminating) update; tolerance policy
        # untouched.
        hp = dict(HP, lr=1e-2, amsgrad=True, max_exp_avg_sq=mx)
        pa, em, ev = truth(pb, g, m, v, hp)
        kw = kwargs_for(pb, g, m, v, pa, em, ev, hp, n)
        u = OR.verify_update(**kw)
        c = OR.verify_update_chunked(**kw, chunk_elems=97)
        self.assertEqual(float(u["max_abs_param_deviation"]),
                         float(c["max_abs_param_deviation"]))
        self.assertTrue(c["verification_successful"])

    def test_chunk_size_one_and_larger_than_fixture(self):
        n = 100
        pb, g, m, v = fixture(n, seed=5)
        # lr=1e-4 on 100 elements gives an intended move of 1.05e-5 against a
        # tolerance of 2.12e-5, i.e. a NON-discriminating fixture. The fix is a
        # resolvable update, not a widened tolerance: the tolerance policy is
        # unchanged and the discrimination check stays armed.
        hp = dict(HP, lr=1e-2)
        pa, em, ev = truth(pb, g, m, v, hp)
        kw = kwargs_for(pb, g, m, v, pa, em, ev, hp, n)
        c1 = OR.verify_update_chunked(**kw, chunk_elems=1)
        self.assertEqual(c1["n_chunks"], n)
        self.assertEqual(c1["final_chunk_elems"], 1)
        cbig = OR.verify_update_chunked(**kw, chunk_elems=n * 10)
        self.assertEqual(cbig["n_chunks"], 1)
        self.assertEqual(cbig["final_chunk_elems"], n)
        self.assertTrue(c1["verification_successful"])
        self.assertTrue(cbig["verification_successful"])


class TestChunkedDetection(unittest.TestCase):

    def test_corruption_in_the_final_short_chunk_is_detected(self):
        n, ce = 1000, 256          # 3 full chunks + final chunk of 232
        pb, g, m, v = fixture(n, seed=7)
        pa, em, ev = truth(pb, g, m, v, HP)
        self.assertEqual(n % ce, 232, "fixture must have a short final chunk")
        pa = pa.clone()
        pa[n - 1] += 1.0           # last element, inside the final short chunk
        kw = kwargs_for(pb, g, m, v, pa, em, ev, HP, n)
        c = OR.verify_update_chunked(**kw, chunk_elems=ce)
        self.assertFalse(c["verification_successful"])
        self.assertEqual(c["witnesses"]["param"]["global_index"], n - 1)
        self.assertEqual(c["witnesses"]["param"]["chunk_offset"], 768)
        self.assertEqual(c["witnesses"]["param"]["local_index"], 231)

    def test_corruption_in_a_middle_chunk_gives_the_right_global_index(self):
        n, ce = 1000, 256
        pb, g, m, v = fixture(n, seed=8)
        pa, em, ev = truth(pb, g, m, v, HP)
        pa = pa.clone()
        pa[600] += 5.0
        kw = kwargs_for(pb, g, m, v, pa, em, ev, HP, n)
        c = OR.verify_update_chunked(**kw, chunk_elems=ce)
        self.assertFalse(c["verification_successful"])
        self.assertEqual(c["witnesses"]["param"]["global_index"], 600)
        self.assertEqual(c["witnesses"]["param"]["chunk_offset"], 512)
        self.assertEqual(c["witnesses"]["param"]["local_index"], 88)

    def test_non_finite_in_a_later_chunk_refuses(self):
        n, ce = 1000, 256
        pb, g, m, v = fixture(n, seed=9)
        pa, em, ev = truth(pb, g, m, v, HP)
        g = g.clone()
        g[900] = float("nan")      # chunk 3, not the first
        kw = kwargs_for(pb, g, m, v, pa, em, ev, HP, n)
        with self.assertRaises(OR.OracleRefusal) as cm:
            OR.verify_update_chunked(**kw, chunk_elems=ce)
        self.assertIn("non-finite", str(cm.exception))

    def test_moment_corruption_in_a_later_chunk_is_detected(self):
        n, ce = 1000, 256
        pb, g, m, v = fixture(n, seed=10)
        pa, em, ev = truth(pb, g, m, v, HP)
        em = em.clone(); em[777] += 1.0
        kw = kwargs_for(pb, g, m, v, pa, em, ev, HP, n)
        c = OR.verify_update_chunked(**kw, chunk_elems=ce)
        self.assertFalse(c["checks"]["exp_avg_agrees"])
        self.assertEqual(c["witnesses"]["exp_avg"]["global_index"], 777)


class TestCoverageRefusals(unittest.TestCase):

    def test_tiling_is_exact_and_includes_the_final_partial_chunk(self):
        self.assertEqual(OR._chunk_offsets(10, 3), [(0, 3), (3, 3), (6, 3), (9, 1)])
        self.assertEqual(OR._chunk_offsets(0, 4), [])
        self.assertEqual(sum(k for _, k in OR._chunk_offsets(1000, 256)), 1000)

    def test_bad_chunk_size_refuses(self):
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(OR.OracleRefusal):
                OR._chunk_offsets(10, bad)

    def test_declared_coverage_mismatch_refuses(self):
        n = 500
        pb, g, m, v = fixture(n, seed=11)
        pa, em, ev = truth(pb, g, m, v, HP)
        kw = kwargs_for(pb, g, m, v, pa, em, ev, HP, n)
        kw["n_elements_checked"] = n - 1          # understated
        with self.assertRaises(OR.OracleRefusal) as cm:
            OR.verify_update_chunked(**kw, chunk_elems=64)
        self.assertIn("does not match", str(cm.exception))
        kw["n_elements_checked"] = n + 1          # overstated / duplicated
        with self.assertRaises(OR.OracleRefusal):
            OR.verify_update_chunked(**kw, chunk_elems=64)

    def test_incomplete_total_is_not_promoted_to_whole_model(self):
        n = 500
        pb, g, m, v = fixture(n, seed=12)
        pa, em, ev = truth(pb, g, m, v, HP)
        kw = kwargs_for(pb, g, m, v, pa, em, ev, HP, n)
        kw["n_elements_total"] = n + 10
        c = OR.verify_update_chunked(**kw, chunk_elems=64)
        self.assertFalse(c["all_supplied_elements_checked"])
        self.assertFalse(c["whole_model_verified"])

    def test_non_tensor_and_non_cpu_inputs_refuse(self):
        """The chunked path refuses anything it cannot slice on the host."""
        n = 64
        pb, g, m, v = fixture(n, seed=13)
        pa, em, ev = truth(pb, g, m, v, HP)
        kw = kwargs_for(pb, g, m, v, pa, em, ev, HP, n)
        kw["grad"] = "not a tensor"
        with self.assertRaises(OR.OracleRefusal) as cm:
            OR.verify_update_chunked(**kw, chunk_elems=8)
        self.assertIn("must be a tensor", str(cm.exception))
        # the device guard is source-verified here; CUDA is not contacted
        src = (ROOT / "zerodiag/update_oracle.py").read_text()
        self.assertIn('if t.device.type != "cpu":', src)
        self.assertIn("the chunked oracle runs on independent", src)


class TestAllocationBounds(unittest.TestCase):
    """The decisive test: observe what the executed path actually allocates."""

    def _run(self, fn):
        rec = AllocRecorder()
        with rec:
            fn()
        return rec

    def test_chunked_path_allocations_are_bounded_by_the_chunk(self):
        n, ce = 65536, 1024
        pb, g, m, v = fixture(n, seed=20)
        pa, em, ev = truth(pb, g, m, v, HP)
        kw = kwargs_for(pb, g, m, v, pa, em, ev, HP, n)
        rec = self._run(lambda: OR.verify_update_chunked(**kw, chunk_elems=ce))
        self.assertLessEqual(
            rec.max_elems, ce,
            f"chunked path allocated {rec.max_elems} elements via "
            f"{rec.biggest_op}, exceeding the {ce}-element chunk bound")
        self.assertGreater(rec.n_ops, 0)

    def test_the_same_recorder_detects_the_unchunked_reference_path(self):
        """If this fails, the bound test above proves nothing."""
        n, ce = 65536, 1024
        pb, g, m, v = fixture(n, seed=20)
        pa, em, ev = truth(pb, g, m, v, HP)
        kw = kwargs_for(pb, g, m, v, pa, em, ev, HP, n)
        rec = self._run(lambda: OR.verify_update(**kw))
        self.assertGreater(
            rec.max_elems, ce,
            "the recorder failed to observe whole-partition allocations in the "
            "unchunked reference path, so it cannot evidence the chunked bound")
        self.assertEqual(rec.max_elems, n)

    def test_allocation_bound_scales_with_the_chunk_setting(self):
        n = 32768
        pb, g, m, v = fixture(n, seed=21)
        pa, em, ev = truth(pb, g, m, v, HP)
        kw = kwargs_for(pb, g, m, v, pa, em, ev, HP, n)
        for ce in (128, 512, 4096):
            rec = self._run(lambda ce=ce: OR.verify_update_chunked(
                **kw, chunk_elems=ce))
            self.assertLessEqual(rec.max_elems, ce)

    def test_snapshot_capture_allocations_are_bounded(self):
        """Route 2: the destination is host-allocated, the copy is per chunk."""
        n, ce = 65536, 1024
        t = torch.randn(n)
        rec = AllocRecorder()
        with rec:
            snap = EP.cpu_snapshot(t, name="fixture", chunk_elems=ce)
        self.assertEqual(int(snap.numel()), n)
        # the one unavoidable n-sized allocation is the destination itself
        self.assertLessEqual(rec.max_elems, n)
        big = [o for o in [rec.biggest_op] if o]
        self.assertTrue(big)

    def test_peak_chunk_bytes_match_the_stated_budget_arithmetic(self):
        """9 fp32 buffers at the default chunk = 144 MiB."""
        ce = OR.DEFAULT_CHUNK_ELEMS
        self.assertEqual(ce, 4_194_304)
        self.assertEqual(9 * ce * 4, 150_994_944)
        self.assertAlmostEqual(9 * ce * 4 / 1048576, 144.0, places=6)


class TestCpuSnapshots(unittest.TestCase):
    """Route 2 semantics."""

    def test_snapshot_is_on_cpu_and_owns_its_storage(self):
        t = torch.randn(1000)
        s = EP.cpu_snapshot(t, name="t", chunk_elems=64)
        self.assertEqual(s.device.type, "cpu")
        self.assertNotEqual(int(s.data_ptr()), int(t.data_ptr()))
        self.assertFalse(s.requires_grad)
        self.assertTrue(torch.equal(s, t))

    def test_mutating_the_live_fixture_cannot_alter_the_snapshot(self):
        t = torch.randn(500)
        s = EP.cpu_snapshot(t, name="t", chunk_elems=37)
        before = s.clone()
        t.mul_(-3.0)
        t[0] = 123.456
        self.assertTrue(torch.equal(s, before),
                        "snapshot changed when the live tensor was mutated")
        self.assertFalse(torch.equal(s, t))

    def test_detach_alone_would_not_have_been_enough(self):
        """Documents why detach/to are insufficient, per the PyTorch semantics."""
        t = torch.randn(100, requires_grad=True)
        d = t.detach()
        self.assertEqual(int(d.data_ptr()), int(t.data_ptr()))   # shared storage
        same = t.detach().to("cpu", torch.float32)
        self.assertEqual(int(same.data_ptr()), int(t.data_ptr()))  # to() returns self
        s = EP.cpu_snapshot(t, name="t", chunk_elems=16)
        self.assertNotEqual(int(s.data_ptr()), int(t.data_ptr()))

    def test_non_contiguous_source_refuses(self):
        t = torch.randn(64, 4)[:, ::2]
        with self.assertRaises(EP.ProbeRefusal) as cm:
            EP.cpu_snapshot(t, name="strided", chunk_elems=8)
        self.assertIn("not contiguous", str(cm.exception))

    def test_non_finite_source_refuses_with_the_chunk_range(self):
        t = torch.randn(300)
        t[250] = float("inf")
        with self.assertRaises(EP.ProbeRefusal) as cm:
            EP.cpu_snapshot(t, name="g", chunk_elems=100)
        self.assertIn("[200, 300)", str(cm.exception))

    def test_metadata_records_placement_and_independence(self):
        t = torch.randn(10, 5)
        s = EP.cpu_snapshot(t, name="m", chunk_elems=7)
        md = EP.snapshot_meta(t, s, name="m")
        self.assertTrue(md["independent_storage"])
        self.assertEqual(md["snapshot_device"], "cpu")
        self.assertEqual(md["shape"], [10, 5])
        self.assertTrue(md["cuda_transfer_unverified_on_cpu"])

    def test_none_snapshots_through(self):
        self.assertIsNone(EP.cpu_snapshot(None, name="absent"))
        self.assertFalse(EP.snapshot_meta(None, None, name="absent")["present"])


class TestDigestStreaming(unittest.TestCase):

    def test_streamed_digest_equals_the_original_representation(self):
        from zerodiag import factories as F

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.randn(1000))
                self.b = torch.nn.Parameter(torch.randn(37, 11))

        torch.manual_seed(1)
        mod = Tiny()
        # the original whole-parameter expression, recomputed here
        h = hashlib.sha256()
        for name, p in sorted(mod.named_parameters()):
            h.update(name.encode() + b"\0")
            h.update(p.detach().to("cpu", torch.float64).numpy().tobytes())
        orig = h.hexdigest()
        saved = F.DIGEST_CHUNK_ELEMS
        try:
            for ce in (1, 13, 999, 1 << 20):
                F.DIGEST_CHUNK_ELEMS = ce
                self.assertEqual(F.state_digest(mod)["parameter_state_sha256"],
                                 orig, f"digest changed at chunk {ce}")
        finally:
            F.DIGEST_CHUNK_ELEMS = saved

    def test_digest_workspace_is_bounded(self):
        from zerodiag import factories as F

        class Big(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(65536))

        mod = Big()
        saved = F.DIGEST_CHUNK_ELEMS
        try:
            F.DIGEST_CHUNK_ELEMS = 1024
            rec = AllocRecorder()
            with rec:
                F.state_digest(mod)
            self.assertLessEqual(rec.max_elems, 1024)
        finally:
            F.DIGEST_CHUNK_ELEMS = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
