"""Regression checks for node-side patches made during A6000 scientific-v2 validation (session 20261014a).

Stdlib only; modules are loaded by file path, like tests/test_measurement_controls.py.
"""
import hashlib
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A6000_SRC = ROOT / "studies/a6000/src"


def module_at(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExecutedSourceProvenance(unittest.TestCase):
    def test_record_hashes_every_package_source_file(self):
        results = module_at("tested_results", "studies/a6000/src/mgbench/results.py")
        record = results.executed_source_record(A6000_SRC / "mgbench")
        self.assertEqual(Path(record["mgbench_package_dir"]), A6000_SRC / "mgbench")
        expected = sorted(p.relative_to(A6000_SRC).as_posix() for p in (A6000_SRC / "mgbench").rglob("*.py"))
        self.assertEqual(sorted(record["sha256"]), expected)
        for relative, digest in record["sha256"].items():
            self.assertEqual(digest, hashlib.sha256((A6000_SRC / relative).read_bytes()).hexdigest())

    def test_default_points_at_the_module_own_package(self):
        results = module_at("tested_results_default", "studies/a6000/src/mgbench/results.py")
        record = results.executed_source_record()
        self.assertEqual(Path(record["mgbench_package_dir"]), (A6000_SRC / "mgbench").resolve())
        self.assertIn("mgbench/train/frameworks.py", record["sha256"])


FRAMEWORKS = ROOT / "studies/a6000/src/mgbench/train/frameworks.py"
TRAIN_MAIN = ROOT / "studies/a6000/src/mgbench/train/__main__.py"


class LossReportingIsUniform(unittest.TestCase):
    """Defect found in the pilot: with grad_accum > 1, DDP/FSDP2 reported loss/GA of the last micro-batch while
    DeepSpeed reported the raw last micro-batch loss (factor GA apart). Every step function must now return the
    mean of the unscaled micro-batch losses, and only the backward pass may divide by the accumulation count."""

    def test_no_step_function_returns_a_scaled_loss(self):
        source = FRAMEWORKS.read_text()
        self.assertNotIn("return loss.detach()", source)
        # DDP, FSDP2, DeepSpeed, plus Candidate 14's three first-boundary early returns (the same unscaled mean)
        self.assertEqual(source.count("return torch.stack(raw).float().mean()"), 6)
        self.assertNotIn("loss = loss_fn(ddp, b, kind) / len(micro_batches)", source)
        self.assertNotIn("loss = loss_fn(model, b, kind) / len(micro_batches)", source)
        self.assertEqual(source.count("(loss / n).backward()"), 2)                  # DDP, FSDP2 scale backward only

    def test_global_loss_and_definition_are_recorded(self):
        source = TRAIN_MAIN.read_text()
        self.assertIn('"global_loss_per_step": global_loss', source)
        self.assertIn('"loss_definition"', source)


class DiagnosticsAreOptIn(unittest.TestCase):
    def test_performance_runs_do_not_build_diagnostics(self):
        source = FRAMEWORKS.read_text()
        self.assertIn('return any(spec.get(k) for k in ("numerical_checks", "reference_capture", '
                      '"diagnostic_fault", "canonical_init"))', source)
        self.assertEqual(source.count("if diagnostics_enabled(spec) else None"), 3)


try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # the offline CI has no torch; these functional checks run on the GPU node
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class GlobalInputsAndFingerprints(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0, str(A6000_SRC))

    def test_global_inputs_identical_across_world_and_accumulation(self):
        import torch
        from mgbench.data import GlobalSynthetic
        kw = dict(micro_batch=1, device="cpu", seed=7, steps=3, vocab_size=1000, seq_len=8)
        w1 = GlobalSynthetic("causal_lm", grad_accum=4, world=1, rank=0, **kw)
        w2 = [GlobalSynthetic("causal_lm", grad_accum=2, world=2, rank=r, **kw) for r in (0, 1)]
        for s in range(3):
            a = torch.cat([b["input_ids"] for b in w1.micro[s]])
            b = torch.cat([w2[r].micro[s][i]["input_ids"] for i in range(2) for r in range(2)])
            self.assertTrue(torch.equal(a, b))

    def test_fingerprint_detects_single_ulp_change_and_dtype(self):
        import torch
        from mgbench.train.diagnostics import fingerprint_state
        a = torch.arange(10, dtype=torch.float32)
        b = a.clone()
        b[3] = torch.nextafter(b[3], torch.tensor(100.0))
        self.assertEqual(fingerprint_state([("x", a)]), fingerprint_state([("x", a.clone())]))
        self.assertNotEqual(fingerprint_state([("x", a)])["sha256"], fingerprint_state([("x", b)])["sha256"])
        self.assertNotEqual(fingerprint_state([("x", a)])["sha256"],
                            fingerprint_state([("x", a)], cast=torch.bfloat16)["sha256"])

    def test_fingerprint_handles_zero_dim_integer_buffers(self):
        import torch
        from mgbench.train.diagnostics import fingerprint_state
        a = fingerprint_state([("bn.num_batches_tracked", torch.tensor(0, dtype=torch.long))])
        b = fingerprint_state([("bn.num_batches_tracked", torch.tensor(1, dtype=torch.long))])
        self.assertNotEqual(a["sha256"], b["sha256"])

    def test_fault_fires_once_at_the_requested_step(self):
        from mgbench.train.diagnostics import NumericalDiagnostics
        diag = NumericalDiagnostics.__new__(NumericalDiagnostics)
        diag.fault = {"kind": "skip_optimizer_step", "at_measured_step": 2}
        diag.record = {"fault_applied_at": None}
        hits = []
        for i in range(-1, 5):
            diag.measure_index = i
            hits.append(diag.fault_now("skip_optimizer_step"))
            self.assertFalse(diag.fault_now("nonfinite_loss"))
        self.assertEqual(hits, [False, False, False, True, False, False])
        self.assertEqual(diag.record["fault_applied_at"], 2)

    def test_change_detector_reports_nan_as_nonfinite_not_unchanged(self):
        import torch
        from mgbench.train.diagnostics import _change
        before = torch.tensor([1.0, 2.0, 3.0, 4.0])
        after = torch.tensor([1.0, 2.5, float("nan"), float("inf")])
        c = _change(before, after)
        self.assertEqual(c["changed_elems"], 1)
        self.assertEqual(c["nonfinite_after"], 2)
        self.assertEqual(c["max_abs_delta"], 0.5)
        self.assertEqual(_change(before, torch.full((4,), float("nan")))["nonfinite_after"], 4)



class StreamClientLogprobsFlag(unittest.TestCase):
    def _payload(self, logprobs):
        import types
        from unittest import mock
        client = module_at("tested_stream_client", "validation/stream_latency.py")
        args = types.SimpleNamespace(model="m", output_len=4, base_url="http://127.0.0.1:1", timeout=1)
        if logprobs is not None:
            args.logprobs = logprobs
        with mock.patch.object(client.urllib.request, "urlopen", side_effect=OSError("offline test")):
            return client.request_one(0, [1001, 1002], args, 0.0)["request_payload"]

    def test_default_keeps_original_logprobs_1(self):
        self.assertEqual(self._payload(None)["logprobs"], 1)
        self.assertEqual(self._payload(1)["logprobs"], 1)

    def test_zero_omits_logprobs(self):
        self.assertNotIn("logprobs", self._payload(0))



class NodePowerGuard(unittest.TestCase):
    """Node guard added 2026-10-15 after the 2026-10-14 22:14 host failure (start budget + in-run watchdog)."""

    GUARD = {"planned_w_per_gpu": 300, "start_budget_w": 1500, "abort_total_w": 1700, "abort_after_s": 5,
             "max_load1": 96, "max_gpus_per_run": 2}

    def _safety(self, powers):
        import sys
        import types
        from unittest import mock
        stub = types.ModuleType("pynvml")
        stub.nvmlInit = lambda: None
        stub.nvmlDeviceGetCount = lambda: len(powers)
        stub.nvmlDeviceGetHandleByIndex = lambda i: i
        stub.nvmlDeviceGetPowerUsage = lambda h: int(powers[h] * 1000)
        patcher = mock.patch.dict(sys.modules, {"pynvml": stub})
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in [m for m in sys.modules if m == "mgbench" or m.startswith("mgbench.")]:
            del sys.modules[name]
        sys.path.insert(0, str(A6000_SRC))
        self.addCleanup(sys.path.remove, str(A6000_SRC))
        import mgbench.gpu_monitor as gm
        gm._initialized = False
        import mgbench.safety as safety
        return safety

    def test_start_budget_with_another_user_at_full_load(self):
        # GPUs 0-3 busy (as measured 2026-10-15 15:18), GPUs 4-7 idle
        safety = self._safety([282.4, 286.2, 270.7, 277.1, 8.6, 10.2, 13.3, 15.2])
        idle = {"node_guard": self.GUARD}
        from unittest import mock
        with mock.patch.object(safety.os, "getloadavg", return_value=(60.0, 60.0, 55.0)):
            self.assertEqual(safety.node_guard_problems([4], idle), [])          # 1-GPU run may start
            self.assertTrue(safety.node_guard_problems([4, 5], idle))           # 2-GPU run must wait
        with mock.patch.object(safety.os, "getloadavg", return_value=(120.0, 80.0, 60.0)):
            self.assertTrue(any("load" in p for p in safety.node_guard_problems([4], idle)))

    def test_start_budget_when_other_gpus_idle(self):
        safety = self._safety([20.0] * 8)
        from unittest import mock
        with mock.patch.object(safety.os, "getloadavg", return_value=(1.0, 1.0, 1.0)):
            self.assertEqual(safety.node_guard_problems([4, 5], {"node_guard": self.GUARD}), [])

    def test_watchdog_needs_consecutive_seconds_over_limit(self):
        safety = self._safety([0.0])
        w = safety.PowerWatch(self.GUARD)
        self.assertFalse(any(w.update(x) for x in [1800, 1800, 1800, 1800, 1600, 1800, 1800]))
        self.assertFalse(w.update(1800) or w.update(1800))
        self.assertTrue(w.update(1750))                                        # 5th consecutive sample
        self.assertEqual(w.peak, 1800)

    def test_cluster_config_has_the_guard(self):
        text = (ROOT / "studies/a6000/configs/cluster.yaml").read_text()
        for key, value in self.GUARD.items():
            self.assertIn(f"{key}: {value}", text)


if __name__ == "__main__":
    unittest.main()
