"""Offline regression checks against recorded measurements and scientific identities."""
from __future__ import annotations

import ast
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

from tools import build_report as report
from tools import prepare_notebook, validate

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_manifest_is_intact(self):
        self.assertEqual(validate.manifest_errors(ROOT), [])

    def test_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "provenance").mkdir()
            (root / "record.json").write_text("modified", encoding="utf-8")
            manifest = {"schema_version": 1, "algorithm": "sha256", "files": [
                {"path": "record.json", "bytes": 8, "sha256": "0" * 64}]}
            (root / "provenance/measurements.sha256.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("Changed measured" in item for item in validate.manifest_errors(root)))

    def test_manifest_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "provenance").mkdir()
            manifest = {"schema_version": 1, "algorithm": "sha256", "files": [
                {"path": "../outside", "bytes": 0, "sha256": "0" * 64}]}
            (root / "provenance/measurements.sha256.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("escapes repository" in item for item in validate.manifest_errors(root)))

    def test_catalog_covers_all_canonical_artifacts(self):
        records = report.catalog(ROOT)
        counts = {campaign: sum(item["campaign"] == campaign for item in records)
                  for campaign in ("a6000", "a100", "titan-1080")}
        self.assertEqual(counts, {"a6000": 224, "a100": 40, "titan-1080": 64})
        identities = {(item["campaign"], item["experiment"], item["run_id"]) for item in records}
        self.assertEqual(len(identities), len(records))
        for item in records:
            self.assertTrue((ROOT / item["source"].split("#", 1)[0]).is_file())
        a6000 = [item for item in records if item["campaign"] == "a6000"]
        self.assertEqual(sum("rerun" in item["experiment"] for item in a6000), 16)

    def test_generated_artifacts_are_current(self):
        for relative, content in report.artifacts(ROOT).items():
            self.assertEqual((ROOT / relative).read_bytes(), content.encode("utf-8"), relative)

    def test_titan_scaling_uses_matching_single_gpu_baseline(self):
        for item in report.read_csv(ROOT / "studies/titan-1080/results/scaling_efficiency.csv"):
            throughput = float(item["tokens_per_s"])
            baseline = float(item["baseline_1gpu_tokens_per_s"])
            n = int(float(item["n_gpus"]))
            self.assertAlmostEqual(float(item["scaling_efficiency_pct"]), 100 * throughput / (baseline * n), delta=0.02)
            self.assertAlmostEqual(float(item["tokens_per_s_per_gpu"]), throughput / n, delta=0.11)

    def test_titan_training_throughput_and_energy_from_raw(self):
        for item in report.read_csv(ROOT / "studies/titan-1080/results/llm_runs.csv"):
            raw = report.load_json(ROOT / "studies/titan-1080/results/raw" / item["file"])
            self.assertEqual(item["status"], raw["status"])
            if raw["status"] != "ok":
                self.assertIsNone(report.number(item["tokens_per_s"]))
                continue
            config, perf = raw["config"], raw["perf"]
            total = config["steps"] * perf["tokens_per_step"]
            self.assertAlmostEqual(float(item["tokens_per_s"]), total / perf["wall_time_s"], delta=2)
            energy = raw.get("telemetry", {}).get("energy_total_j")
            if energy:
                self.assertAlmostEqual(float(item["energy_per_1m_tokens_j"]), energy / total * 1e6, delta=0.11)

    def test_titan_communication_units_and_ring_factor(self):
        for path in (ROOT / "studies/titan-1080/results/raw").glob("C_comm_*.json"):
            data = report.load_json(path)
            n = data["config"]["world_size"]
            for item in data["rows"]:
                expected = item["size_mb"] * 1024**2 / (item["time_ms"] / 1000) / 1e9
                self.assertAlmostEqual(item["algbw_gb_s"], expected, delta=0.02)
                self.assertAlmostEqual(item["busbw_gb_s"], expected * 2 * (n - 1) / n, delta=0.02)

    def test_a6000_main_summary_medians(self):
        raw = ROOT / "studies/a6000/results/raw/train_scaling"
        for item in report.read_csv(ROOT / "studies/a6000/results/summary_train.csv"):
            if item["status"] != "ok":
                continue
            samples = []
            metric = "throughput_samples_s" if item["model"] == "resnet50" else "throughput_tokens_s"
            for path in raw.glob("*/result.json"):
                data = report.load_json(path)
                spec = data["spec"]
                if data.get("status") == "ok" and spec["model"] == item["model"] and spec["framework"] == item["framework"] \
                        and spec["n_gpus"] == int(item["n_gpus"]) and spec.get("variant", "") == item["variant"]:
                    samples.append(data[metric])
            self.assertEqual(len(samples), int(item["repeats_ok"]))
            self.assertTrue(math.isclose(statistics.median(samples), float(item[metric]), rel_tol=1e-10))

    def test_a100_combined_csv_matches_jsonl(self):
        combined = {(item["benchmark"], item["key"]): item
                    for item in report.read_csv(ROOT / "studies/a100/all_results.csv")}
        count = 0
        for path in (ROOT / "studies/a100").glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                csv_row = combined[(path.stem, data["key"])]
                self.assertEqual(csv_row["status"], data["status"])
                for metric in ("tokens_per_s", "images_per_s", "output_tokens_per_s", "peak_alloc_gb"):
                    if report.number(data.get(metric)) is not None:
                        self.assertEqual(float(csv_row[metric]), float(data[metric]))
                count += 1
        self.assertEqual(count, len(combined))

    def test_clean_notebook_has_no_historical_side_effect_cells(self):
        data = json.loads((ROOT / "studies/a100/reproduce.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(data, prepare_notebook.notebook())
        for cell in data["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            self.assertNotIn("pkill", source)
            self.assertNotIn("authenticate_user", source)
            self.assertNotIn("drive.mount", source)
            lines = source.splitlines()
            if lines and lines[0].startswith("%%writefile"):
                lines = lines[1:]
            lines = [line if not line.lstrip().startswith(("!", "%")) else "" for line in lines]
            ast.parse("\n".join(lines))

    def test_active_local_links_exist(self):
        self.assertEqual(validate.local_link_errors(ROOT), [])

    def test_author_is_the_named_person(self):
        cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertEqual(cff.count("given-names:"), 1)
        self.assertIn("given-names: Mehran", cff)
        self.assertIn("family-names: Keivanimehr", cff)


if __name__ == "__main__":
    unittest.main()
