"""Exercise real sampler/optimizer-observer code with synthetic hardware readings.

These are CPU regression tests, not proof of CUDA or DeepSpeed execution.
"""
import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import build_report, scientific_status

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MeasurementControlTests(unittest.TestCase):
    def test_a6000_window_excludes_setup_and_teardown(self):
        with patch.dict(sys.modules, {"pynvml": types.ModuleType("pynvml")}):
            monitor = module_at("tested_monitor", "studies/a6000/src/mgbench/gpu_monitor.py")
        sampler = monitor.GpuSampler.__new__(monitor.GpuSampler)
        sampler.gpus, sampler.t_start, sampler.t_stop = [4], 1, 2
        sampler.samples = [(0, 4, 100, 30, 999, 50, 1000), (1.1, 4, 40, 4, 100, 40, 800),
                           (1.9, 4, 80, 5, 200, 45, 900), (3, 4, 100, 40, 999, 60, 1000)]
        result = sampler.summary()["4"]
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(result["power_mean_w"], 150)
        self.assertEqual(result["util_mean_pct"], 60)
        self.assertEqual(result["mem_used_peak_gib"], 40)
        self.assertEqual(result["mem_used_peak_timed_gib"], 5)

    def test_titan_real_sampler_uses_workload_not_sample_span(self):
        common = module_at("tested_common", "studies/titan-1080/bench/common.py")
        sampler = common.GpuSampler([0])
        sampler.t_start, sampler.t_stop = 0.25, 1.75
        gpu = {"index": 0, "power.draw": 100, "utilization.gpu": 50, "memory.used": 1,
               "clocks.sm": 900, "temperature.gpu": 40}
        sampler._rows = [(0, [gpu]), (1, [gpu]), (2, [gpu])]
        result = sampler.stats()
        self.assertEqual(result.duration_s, 1.5)
        self.assertEqual(result.energy_total_j, 150)
        self.assertEqual(result.power_mean_w[0], 100)
        self.assertEqual(result.integration[0]["status"], "ok")

    def test_titan_missing_gpu_reading_fails_closed(self):
        common = module_at("tested_common_missing", "studies/titan-1080/bench/common.py")
        sampler = common.GpuSampler([0])
        sampler.t_start, sampler.t_stop = 0.25, 1.75
        sampler._rows = [(0, []), (1, []), (2, [])]
        with self.assertRaisesRegex(RuntimeError, "invalid energy coverage"):
            sampler.stats()

    def test_titan_results_cannot_replace_existing_run(self):
        common = module_at("tested_common_write", "studies/titan-1080/bench/common.py")
        with tempfile.TemporaryDirectory() as folder:
            common.RESULTS_RAW = Path(folder)
            common.write_result({"value": 1}, "run")
            with self.assertRaises(FileExistsError):
                common.write_result({"value": 2}, "run")
            self.assertEqual(json.loads((Path(folder) / "run.json").read_text())["value"], 1)

    def test_optimizer_observer_counts_actual_calls_not_attempts(self):
        path = ROOT / "studies/a6000/src/mgbench/train/frameworks.py"
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "observe_optimizer")
        class Flag:
            def all(self): return self
            def item(self): return True
        parameter = types.SimpleNamespace(dtype="fp32", numel=lambda: 10, grad=Flag())
        class Optimizer:
            param_groups = [{"params": [parameter]}]
            def register_step_pre_hook(self, hook): self.before = hook
            def register_step_post_hook(self, hook): self.after = hook
            def step(self):
                self.before(self, (), {})
                self.after(self, (), {})
        scope = {"torch": types.SimpleNamespace(isfinite=lambda gradient: gradient)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
        optimizer = Optimizer()
        model = types.SimpleNamespace(parameters=lambda: [parameter])
        state, dtypes = scope["observe_optimizer"](optimizer, model, {"numerical_checks": True})
        optimizer.step()
        # A skipped AMP step never calls the underlying optimizer.
        optimizer.step()
        self.assertEqual(state["optimizer_calls"], 2)
        self.assertEqual(state["successful_updates"], 2)
        self.assertEqual(state["gradient_checks"], 2)
        self.assertEqual(state["nonfinite_gradient_steps"], 0)
        self.assertEqual(dtypes, {"fp32": 10})
        state["logical_updates_from_engine"] = True
        optimizer.step()
        optimizer.step()
        self.assertEqual(state["optimizer_calls"], 4)
        self.assertEqual(state["successful_updates"], 2)  # base subgroup calls are not extra logical updates

    def test_protocols_do_not_merge_qwen_precision_recipes(self):
        rows = [row for row in build_report.protocol_rows() if row["campaign"] == "a6000"
                and row["task"] == "train" and row["model"].startswith("qwen")]
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["precision"] for row in rows}), 3)

    def test_active_titan_output_names_do_not_claim_proxy_ttft(self):
        hf = (ROOT / "studies/titan-1080/bench/infer_hf.py").read_text()
        vllm = (ROOT / "studies/titan-1080/bench/infer_vllm.py").read_text()
        self.assertNotIn('"ttft_s":', hf)
        self.assertIn('"prefill_forward_proxy_s":', hf)
        self.assertNotIn('"latency_per_request_s":', vllm)
        self.assertIn('"batch_completion_duration_s":', vllm)

    def test_all_findings_have_evidence_without_false_gpu_closure(self):
        ledger = json.loads((ROOT / "validation/findings.json").read_text())
        self.assertFalse(ledger["gpu_validation_performed"])
        self.assertEqual({entry["id"] for entry in ledger["findings"]}, {f"A{i:02d}" for i in range(1, 19)})
        for entry in ledger["findings"]:
            self.assertTrue(entry["required_evidence"])
            for relative in entry["local_evidence"]:
                self.assertTrue((ROOT / relative).is_file(), (entry["id"], relative))
            self.assertNotEqual(entry["experimental_status"], "validated")

    def test_status_document_is_current(self):
        self.assertEqual((ROOT / "docs/SCIENTIFIC_STATUS.md").read_bytes(), scientific_status.document().encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
