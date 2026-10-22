"""CPU unit tests for Candidate 17's two-device identity and pair-path decisions (no GPU, no driver: the driver
records are fixtures). The integrated supervisor path is exercised separately by scripts/pair_integration_selftest.py."""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "studies/a6000/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from mgbench import device_identity as di  # noqa: E402
from mgbench import pair_controls as pc  # noqa: E402
from mgbench import smoke  # noqa: E402

U4, U5 = "GPU-00000000-0000-0000-0000-000000000109", "GPU-00000000-0000-0000-0000-000000000103"   # U5: CPU fixture only
P4, P5 = "00000000:81:00.0", "00000000:82:00.0"
DRV = {"count": 2, "devices": [{"ordinal": 0, "uuid": U4, "pci_bus_id": "0000:81:00.0"},
                               {"ordinal": 1, "uuid": U5, "pci_bus_id": "0000:82:00.0"}]}


class TwoDeviceIdentity(unittest.TestCase):
    def test_visibility_is_the_exact_ordered_set(self):
        self.assertEqual(di.multi_visibility_problems("4,5", [4, 5]), [])
        for bad in ("5,4", "4", "4,5,6", "4, 5", "4,4", "", None, f"{U4},{U5}"):
            self.assertTrue(di.multi_visibility_problems(bad, [4, 5]), bad)
        self.assertEqual(di.visibility_problems("4", 4), [])                       # single-device path unchanged
        self.assertTrue(di.visibility_problems("4,5", 4))

    def test_driver_set_requires_uuid_and_pci_per_ordinal(self):
        self.assertEqual(di.driver_set_problems(DRV, [(U4, P4), (U5, P5)]), [])
        swapped = {"count": 2, "devices": [dict(DRV["devices"][0], ordinal=1), dict(DRV["devices"][1], ordinal=0)]}
        dup = {"count": 2, "devices": [DRV["devices"][0], dict(DRV["devices"][0], ordinal=1)]}
        for bad in (swapped, dup, {"count": 1, "devices": DRV["devices"][:1]},
                    {"count": 3, "devices": DRV["devices"] + [dict(DRV["devices"][1], ordinal=2)]},
                    {"count": 2, "devices": [DRV["devices"][0], dict(DRV["devices"][1], pci_bus_id="0000:83:00.0")]},
                    {"count": 2, "devices": [DRV["devices"][0], dict(DRV["devices"][1], uuid=None)]}, None):
            self.assertTrue(di.driver_set_problems(bad, [(U4, P4), (U5, P5)]), bad)

    def test_prestep_main_multi_and_single(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for cvd, drv, idx, want in (("4,5", DRV, "4,5", 0), ("5,4", DRV, "4,5", 7), ("4,5", {"count": 1, "devices": DRV["devices"][:1]}, "4,5", 7)):
                out = Path(d) / f"p{cvd}{want}{len(drv['devices'])}.json"
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": cvd}), mock.patch.object(di, "driver_identity", lambda: drv):
                    rc = di.main([f"{U4},{U5}", f"{P4},{P5}", idx, str(out)])
                self.assertEqual(rc, want, cvd)
                self.assertEqual(json.loads(out.read_text())["problems"] == [], want == 0)

    def test_guarded_multi_refuses_bad_rank_or_identity(self):
        import json
        g = __import__("mgbench.train.guarded", fromlist=["x"])
        base = {"MGBENCH_EXPECTED_GPU_UUIDS": f"{U4},{U5}", "MGBENCH_EXPECTED_GPU_PCIS": f"{P4},{P5}",
                "MGBENCH_EXPECTED_GPU_INDICES": "4,5", "CUDA_VISIBLE_DEVICES": "4,5"}
        swapped = {"count": 2, "devices": [dict(DRV["devices"][0], ordinal=1), dict(DRV["devices"][1], ordinal=0)]}
        for env, drv, why in ((dict(base, LOCAL_RANK="2"), DRV, "LOCAL_RANK 2"),
                              (dict(base, LOCAL_RANK="1", CUDA_VISIBLE_DEVICES="5,4"), DRV, "not the approved ordered set"),
                              (dict(base, LOCAL_RANK="0"), swapped, "is not the approved")):
            with mock.patch.dict(os.environ, env), mock.patch("mgbench.device_identity.driver_identity", lambda d=drv: d), \
                    mock.patch.dict(sys.modules):
                sys.modules.pop("torch", None)                     # the check must run before any torch import
                self.assertEqual(g.main(), 7, why)
                probs = json.loads(os.environ["MGBENCH_DRIVER_IDENTITY"])["problems"]
                self.assertTrue(any(why in p for p in probs), probs)
                self.assertFalse(any("torch was imported" in p for p in probs))

    def test_worker_env_for_the_pair_and_single(self):
        e2 = smoke.env_for_worker({}, src=SRC, uuid=[U4, U5], index=[4, 5], pci=[P4, P5], threads=7, cache=Path("/c"), run_id="r")
        self.assertEqual((e2["CUDA_VISIBLE_DEVICES"], e2["MGBENCH_REQUIRE_DEVICE_COUNT"], e2["MGBENCH_EXPECTED_GPU_INDICES"]),
                         ("4,5", "2", "4,5"))
        self.assertNotIn("MGBENCH_REQUIRE_SINGLE_DEVICE", e2)
        self.assertNotIn("MGBENCH_EXPECTED_GPU_UUID", e2)
        e1 = smoke.env_for_worker({}, src=SRC, uuid=U4, index=4, pci=P4, threads=8, cache=Path("/c"), run_id="r")
        self.assertEqual((e1["CUDA_VISIBLE_DEVICES"], e1["MGBENCH_REQUIRE_SINGLE_DEVICE"]), ("4", "1"))
        for bad in (dict(uuid=[U4], index=[4, 5], pci=[P4, P5]), dict(uuid=[U4, U5], index=[4, 4], pci=[P4, P5])):
            with self.assertRaises(ValueError):
                smoke.env_for_worker({}, src=SRC, threads=1, cache=Path("/c"), run_id="r", **bad)


class PairDecisions(unittest.TestCase):
    def test_cleanup_is_decided_first(self):
        self.assertEqual(pc.case_outcome(False, None, True), ("cleanup_unconfirmed", True))   # never success
        self.assertEqual(pc.case_outcome(None, None, True), ("cleanup_unconfirmed", True))
        self.assertEqual(pc.case_outcome(True, "guard", True), ("stopped", True))
        self.assertEqual(pc.case_outcome(True, None, False), ("unexpected_result", True))
        self.assertEqual(pc.case_outcome(True, None, True), ("completed_valid", False))

    def test_supervisor_checks_cleanup_before_evaluation_and_uses_one_scope(self):
        sup = (ROOT / "studies/a6000/scripts/smoke_supervisor.py").read_text()
        self.assertLess(sup.index('if not cleanup["confirmed"]:'), sup.index("ev = pc.evaluate_pair_case("))
        self.assertIn('f"--nproc-per-node={len(cg)}"', sup)                       # ONE torchrun tree, one worker scope
        self.assertIn("scopes.start_scope(unit, cmd", sup)
        self.assertIn('reservation = out / ("RESERVATION_" + "_".join(f"gpu{g}" for g in window_gpus))', sup)
        self.assertIn("if all_cleanups_confirmed and reserved_by_us:", sup)

    def test_hook_cannot_activate_with_real_nvml_or_visible_cuda(self):
        fake = type(sys)("pynvml")
        with mock.patch.dict(sys.modules, {"pynvml": fake}), mock.patch.dict(os.environ, {"MGBENCH_SUPERVISOR_TEST_WORKER": "/x.py", "CUDA_VISIBLE_DEVICES": ""}):
            self.assertIsNone(pc.test_worker_hook(Path("/a"), 2))                   # not a declared stub
            fake.MGBENCH_TEST_STUB = True
            self.assertIsNotNone(pc.test_worker_hook(Path("/a"), 2))
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}):
                self.assertIsNone(pc.test_worker_hook(Path("/a"), 2))               # CUDA visible -> never
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MGBENCH_SUPERVISOR_TEST_WORKER", None)
            self.assertIsNone(pc.test_worker_hook(Path("/a"), 2))


if __name__ == "__main__":
    unittest.main()
