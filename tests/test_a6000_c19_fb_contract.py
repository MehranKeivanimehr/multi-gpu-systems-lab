"""CPU unit tests for Candidate 19: the first-boundary scope (G0, G1, G3, G4, X1 on GPU 4) under the Candidate 18
window contract. No GPU; every observation is a fixture. The integrated path is exercised by
studies/a6000/scripts/c19_fb_integration_selftest.py."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "studies/a6000"
SRC = A / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import yaml  # noqa: E402

from mgbench import smoke  # noqa: E402
from mgbench import window_contract as wc  # noqa: E402

CFG = yaml.safe_load((A / "configs/fb_qwen15b.yaml").read_text())
PAIR = yaml.safe_load((A / "configs/control_pair.yaml").read_text())
FEAS_PATH = ROOT.parents[1] / "35_c15_design/fb_memory_feasibility.json"
T4 = 50897289216
NV = 49140 * 2**20
U4 = "GPU-00000000-0000-0000-0000-000000000109"


class FirstBoundaryTiming(unittest.TestCase):
    def test_model_values(self):
        m = wc.timing_model(CFG["timing"], n_gpus_watched=1, n_cases=5)
        self.assertEqual(m["watch_overshoot_s"], 5 * 2 + 3 + 1)                       # 14
        self.assertEqual(m["stop_and_reap_bound_s"], 105)
        self.assertEqual(m["case_total_s"], 60 + 240 + 14 + 105 + 10)                # 429
        self.assertEqual(m["supervisor_timeout_stop_s"], 14 + 105 + 30 + 5)          # 154
        self.assertEqual(m["inclusive_window_s"], 120 + 0 + 5 * 429 + 30 + 154 + 10)  # 2459
        self.assertLessEqual(m["inclusive_window_s"], CFG["timing"]["window_s"])
        self.assertEqual((m["worker_runtime_max_s"], m["worker_timeout_stop_s"]), (242, 60))

    def test_generator_uses_the_scope_gpu_count(self):
        now = 1_900_000_000.0
        p = wc.supervisor_launch_parameters(CFG["timing"], now + 2700, now, n_cases=5, n_gpus_watched=1)
        self.assertEqual((p["RuntimeMaxSec"], p["TimeoutStopSec"], p["problems"]), (2700 - 164, 154, []))
        self.assertTrue(wc.supervisor_launch_parameters(CFG["timing"], now + 2200, now, n_cases=5, n_gpus_watched=1)["problems"])
        q = wc.supervisor_launch_parameters(PAIR["timing"], now + 1500, now)          # the pair default is unchanged
        self.assertEqual((q["RuntimeMaxSec"], q["TimeoutStopSec"]), (1331, 159))


@unittest.skipUnless(FEAS_PATH.exists(), "Candidate 15 design files are on the node only")
class FirstBoundaryCapacity(unittest.TestCase):
    def setUp(self):
        self.feas = json.loads(FEAS_PATH.read_text())

    def contract(self, case, expected=T4):
        f = wc.fb_feasibility_case(self.feas, case["id"])
        return wc.worker_capacity_contract(case, f, [4], {4: {"uuid": U4, "nvml_total_bytes": NV}},
                                           {"known_torch_total_bytes": {"4": expected} if expected else {}})

    def test_global_non_allocator_fallback(self):
        self.assertEqual(wc.fb_feasibility_case(self.feas, "G1_ddp_autocast"),
                         {"predicted_reserved_gib": 41.56, "non_allocator_gib": 0.8})

    def test_every_case_passes_on_gpu4_and_fails_below(self):
        for case in CFG["cases"]:
            self.assertTrue(wc.worker_capacity_check(self.contract(case), 0, T4)["pass"], case["id"])
            self.assertFalse(wc.worker_capacity_check(self.contract(case), 0, T4 - 2**20)["pass"], case["id"])
        g1 = CFG["cases"][1]
        self.assertFalse(wc.worker_capacity_check(self.contract(g1, expected=None), 0, int(47.0 * 2**30))["pass"])
        self.assertFalse(wc.worker_capacity_check(self.contract(g1), 0, NV + 1)["pass"])

    def test_nvml_upper_bound_refuses_a_small_device(self):
        f = wc.fb_feasibility_case(self.feas, "G0_fp32_control")
        r = wc.device_capacity_record(4, {"nvml_total_bytes": 46068 * 2**20}, CFG["cases"][0], f)
        self.assertFalse(r["pass"])

    def test_fractions_unchanged(self):
        self.assertEqual([c["mem_fraction"] for c in CFG["cases"]], [0.86, 0.9, 0.9, 0.85, 0.86])


@unittest.skipUnless(FEAS_PATH.exists(), "Candidate 15 design files are on the node only")
class FirstBoundaryConfig(unittest.TestCase):
    def setUp(self):
        self.feas = json.loads(FEAS_PATH.read_text())

    def test_frozen_config_and_refusals(self):
        self.assertEqual(smoke.fb_config_problems(CFG, self.feas), [])
        muts = {"no lease": lambda c: c.pop("lease_dir"), "relative lease": lambda c: c.update(lease_dir="leases"),
                "enrollment": lambda c: c["timing"].update(enroll_s=30), "no torch total": lambda c: c.pop("known_torch_total_bytes"),
                "torch total in MiB": lambda c: c.update(known_torch_total_bytes={4: 48540}),
                "post_case too long": lambda c: c["timing"].update(post_case_s=120),
                "window too short": lambda c: c["timing"].update(window_s=2400),
                "missing key": lambda c: c["timing"].pop("metadata_call_s")}
        for name, m in muts.items():
            c = copy.deepcopy(CFG)
            m(c)
            self.assertTrue(smoke.fb_config_problems(c, self.feas), name)


if __name__ == "__main__":
    unittest.main()
