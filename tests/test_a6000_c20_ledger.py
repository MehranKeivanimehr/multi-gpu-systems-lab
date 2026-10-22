"""Candidate 20 regression: the DeepSpeed first-boundary ledger observation counts the extracted gradient TENSOR.
Found on hardware (window w19fb_20261021T224030Z, case G4): scan_units() reads `.grad` of parameters, so passing the
flattened gradient tensor counted 0 elements and the worker refused a complete extraction."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "studies/a6000/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ImportError:      # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TensorObservation(unittest.TestCase):
    def setUp(self):
        from mgbench.train import boundary_ledger as bl
        self.bl = bl

    def ledger(self, n):
        return self.bl.BoundaryLedger("zero2", True, "first-boundary extraction", {"group0": n}, 1, "test fixture")

    def observe(self, units):
        lg = self.ledger(8)
        lg.begin_step()
        lg.observe_units(units, "test")
        lg.end_step(False, "first-boundary")
        return lg.phases[lg.phase][-1]["coverage"]["state"]

    def test_tensor_scan_counts_elements_and_finiteness(self):
        self.assertEqual(self.bl.scan_tensors({"group0": torch.ones(8)}), {"group0": (8, True)})
        self.assertEqual(self.bl.scan_tensors({"group0": torch.tensor([1.0, float("nan")])}), {"group0": (2, False)})
        self.assertEqual(self.bl.scan_tensors({"group0": None}), {"group0": (0, None)})

    def test_complete_with_tensor_scan_and_the_old_defect_reproduced(self):
        g = torch.ones(8)
        self.assertEqual(self.observe(self.bl.scan_tensors({"group0": g})), self.bl.COMPLETE)
        self.assertEqual(self.observe(self.bl.scan_units({"group0": g})), self.bl.EMPTY)       # the hardware finding
        self.assertEqual(self.observe(self.bl.scan_tensors({"group0": torch.ones(7)})), self.bl.PARTIAL)
        self.assertEqual(self.observe(self.bl.scan_tensors({"group0": torch.tensor([float("inf")] * 8)})), self.bl.NONFINITE)

    def test_deepspeed_first_boundary_uses_the_tensor_scan(self):
        src = (SRC / "mgbench/train/frameworks.py").read_text()
        self.assertIn('scan_tensors({"group0": g32})', src)
        self.assertNotIn('scan_units({"group0": g32})', src)


if __name__ == "__main__":
    unittest.main()
