"""CPU tests for stage 20261016s2: device-identity chain, unique-identity coverage and strengthened sampled update
evidence. CUDA is never touched: the Driver API is a fake ctypes-like object, NVML is not used, torch runs on CPU.
These tests show refusal and accounting logic, not GPU behaviour."""
import ctypes
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "studies/a6000/src"
UUID4 = "GPU-00000000-0000-0000-0000-000000000109"
UUID5 = "GPU-00000000-0000-0000-0000-000000000112"
PCI4, PCI5 = "00000000:81:00.0", "00000000:A1:00.0"

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def identity_module():
    sys.path.insert(0, str(SRC))
    try:
        return importlib.import_module("mgbench.device_identity")
    finally:
        sys.path.remove(str(SRC))


class FakeCuda:
    """Stand-in for libcuda: devices = [(uuid string, pci string)], optional failing call names."""

    def __init__(self, devices, fail=(), v2=True):
        self.devices, self.fail = devices, set(fail)
        if not v2:
            self.cuDeviceGetUuid_v2 = None          # symbol absent in an older driver

    def _rc(self, name):
        return 999 if name in self.fail else 0

    def cuInit(self, flags):
        return self._rc("cuInit")

    def cuDeviceGetCount(self, ref):
        ref._obj.value = len(self.devices)
        return self._rc("cuDeviceGetCount")

    def cuDeviceGet(self, ref, ordinal):
        ref._obj.value = ordinal
        return self._rc("cuDeviceGet")

    def cuDeviceGetUuid_v2(self, buf, dev):
        raw = bytes.fromhex(self.devices[dev.value][0][4:].replace("-", ""))
        for i, b in enumerate(raw):
            buf[i] = b
        return self._rc("uuid")

    cuDeviceGetUuid = cuDeviceGetUuid_v2

    def cuDeviceGetPCIBusId(self, buf, n, dev):
        buf.value = self.devices[dev.value][1].encode()
        return self._rc("pci")

    def cuDeviceGetName(self, buf, n, dev):
        buf.value = b"NVIDIA RTX A6000"
        return 0


class VisibilityAndPci(unittest.TestCase):
    def setUp(self):
        self.d = identity_module()

    def test_only_the_single_approved_integer_index_is_accepted(self):
        self.assertEqual(self.d.visibility_problems("4", 4), [])
        for bad in ("4,5", UUID4, " 4", "4 ", "", None, "-1", "04", "5", "GPU-4", "4,"):
            self.assertTrue(self.d.visibility_problems(bad, 4), bad)

    def test_pci_normalisation_compares_nvml_and_cuda_forms(self):
        self.assertEqual(self.d.normalize_pci(PCI4), self.d.normalize_pci("0000:81:00.0"))
        self.assertNotEqual(self.d.normalize_pci(PCI4), self.d.normalize_pci(PCI5))
        for bad in (None, "", "81:00", "zz:81:00.0", "0000:81:00.9"):
            self.assertIsNone(self.d.normalize_pci(bad), bad)

    def test_uuid_bytes_format(self):
        raw = bytes.fromhex("5bf80cb6b51ffaf64539cbc85a51bf1a")
        self.assertEqual(self.d.format_uuid(raw), UUID4)


class DriverQueryAndChain(unittest.TestCase):
    def setUp(self):
        self.d = identity_module()

    def test_driver_query_reads_the_sole_device_without_context_calls(self):
        lib = FakeCuda([(UUID4, "0000:81:00.0")])
        rec = self.d.driver_identity(lib)
        self.assertEqual(rec["count"], 1)
        self.assertEqual((rec["devices"][0]["uuid"], rec["devices"][0]["ordinal"]), (UUID4, 0))
        src = (SRC / "mgbench/device_identity.py").read_text()
        for forbidden in ("cuCtxCreate", "cuDevicePrimaryCtxRetain", "cuMemAlloc", "cuLaunchKernel", "import torch"):
            self.assertNotIn(forbidden, src)
        lib_v1 = FakeCuda([(UUID4, "0000:81:00.0")], v2=False)
        self.assertEqual(self.d.driver_identity(lib_v1)["devices"][0]["uuid_api"], "cuDeviceGetUuid")

    def test_driver_error_raises(self):
        with self.assertRaises(RuntimeError):
            self.d.driver_identity(FakeCuda([(UUID4, "0000:81:00.0")], fail=("cuInit",)))

    def good(self):
        return dict(approved_uuid=UUID4, approved_pci=PCI4,
                    nvml={"index": 4, "uuid": UUID4, "pci_bus_id": PCI4},
                    driver={"count": 1, "devices": [{"ordinal": 0, "uuid": UUID4, "pci_bus_id": "0000:81:00.0"}]},
                    deepspeed={"logical": 0, "nvml_index": 4, "expected_index": 4, "uuid": UUID4, "pci_bus_id": PCI4},
                    torch_records=[{"stage": s, "device_count": 1, "current_device": 0, "uuid": UUID4,
                                    "pci_bus_id": "0000:81:00.0"} for s in ("after set_device", "after cap")])

    def test_consistent_chain_passes(self):
        self.assertEqual(self.d.chain_problems(**self.good()), [])

    def test_permutation_nvml_and_cuda_disagree(self):
        g = self.good()
        g["driver"]["devices"][0].update(uuid=UUID5, pci_bus_id="0000:a1:00.0")   # CUDA ordinal 0 is another card
        self.assertGreaterEqual(len(self.d.chain_problems(**g)), 2)

    def test_wrong_card_everywhere_is_still_refused(self):
        g = self.good()
        g["nvml"].update(uuid=UUID5, pci_bus_id=PCI5)
        g["driver"]["devices"][0].update(uuid=UUID5, pci_bus_id="0000:a1:00.0")
        g["deepspeed"].update(uuid=UUID5, pci_bus_id=PCI5)
        for r in g["torch_records"]:
            r.update(uuid=UUID5, pci_bus_id="0000:a1:00.0")
        self.assertGreaterEqual(len(self.d.chain_problems(**g)), 8)

    def test_deepspeed_mapping_to_another_index_refuses(self):
        g = self.good()
        g["deepspeed"].update(nvml_index=5)
        self.assertTrue(self.d.chain_problems(**g))

    def test_missing_identity_or_extra_device_refuses(self):
        for mutate in (lambda g: g["driver"].update(count=2),
                       lambda g: g["driver"]["devices"][0].pop("uuid"),
                       lambda g: g["driver"]["devices"][0].update(pci_bus_id=None),
                       lambda g: g["torch_records"][1].update(pci_bus_id=None),
                       lambda g: g["torch_records"][0].update(device_count=2),
                       lambda g: g["nvml"].update(uuid=None),
                       lambda g: g.update(approved_pci="not-a-pci")):
            g = self.good()
            mutate(g)
            self.assertTrue(self.d.chain_problems(**g))

    def test_prestep_writes_a_refusal_for_malformed_visibility(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": UUID4}), \
                mock.patch.object(self.d, "driver_identity", lambda: {"count": 1, "devices": [
                    {"ordinal": 0, "uuid": UUID4, "pci_bus_id": "0000:81:00.0"}]}), mock.patch("sys.stdout"):
            rc = self.d.main([UUID4, PCI4, "4", f"{d}/id.json"])
            rec = json.loads(Path(f"{d}/id.json").read_text())
        self.assertEqual(rc, 7)
        self.assertTrue(any("not a single non-negative integer" in p for p in rec["problems"]))

    def test_prestep_accepts_only_the_exact_chain(self):
        orig = self.d.driver_identity
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}), \
                mock.patch.object(self.d, "driver_identity", lambda: orig(FakeCuda([(UUID4, "0000:81:00.0")]))), \
                mock.patch("sys.stdout"):
            self.assertEqual(self.d.main([UUID4, PCI4, "4", f"{d}/ok.json"]), 0)
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}), \
                mock.patch.object(self.d, "driver_identity", lambda: orig(FakeCuda([(UUID5, "0000:a1:00.0")]))), \
                mock.patch("sys.stdout"):
            self.assertEqual(self.d.main([UUID4, PCI4, "4", f"{d}/perm.json"]), 7)


class IdentityOrdering(unittest.TestCase):
    def test_guarded_entry_checks_identity_before_importing_torch(self):
        src = (SRC / "mgbench/train/guarded.py").read_text()
        self.assertNotIn("\nimport torch", src)
        self.assertLess(src.index("driver_identity()"), src.index("from . import __main__"))
        self.assertIn('"torch" in sys.modules', src)

    def test_chain_is_checked_before_any_model_allocation_or_deepspeed_init(self):
        main = (SRC / "mgbench/train/__main__.py").read_text()
        run = main[main.index("def run("):]
        self.assertLess(run.index("identity_chain = _identity_chain(spec, device_records)"),
                        run.index('trainer = SETUP[spec["framework"]]'))
        self.assertIn("acc._get_nvml_gpu_id(torch.cuda.current_device())", main)

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_guarded_refuses_when_torch_was_imported_or_identity_missing(self):
        sys.path.insert(0, str(SRC))
        self.addCleanup(sys.path.remove, str(SRC))
        guarded = importlib.import_module("mgbench.train.guarded")
        env = {"MGBENCH_EXPECTED_GPU_UUID": UUID4, "MGBENCH_EXPECTED_GPU_PCI": PCI4,
               "MGBENCH_EXPECTED_GPU_INDEX": "4", "CUDA_VISIBLE_DEVICES": "4"}
        with mock.patch.dict(os.environ, env), mock.patch("sys.stdout"):
            self.assertEqual(guarded.main(), 7)                 # torch is already imported in this test process
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}, clear=True), mock.patch("sys.stdout"):
            self.assertEqual(guarded.main(), 7)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class UniqueIdentityCoverage(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(SRC))
        self.addCleanup(sys.path.remove, str(SRC))
        from mgbench.train import boundary_ledger, frameworks
        self.bl, self.fw = boundary_ledger, frameworks

    def test_omitted_tensor_masked_by_a_duplicate_with_equal_totals_is_not_complete(self):
        led = self.bl.BoundaryLedger("ddp", True, "in_step", {"a": 10, "b": 10}, 1, "native")
        led.begin_step()
        led.observe_units({"a": (10, True)}, "first look")
        led.observe_units({"a": (10, True)}, "duplicate look")          # 20 observed == 20 expected
        led.end_step(True, "hook")
        self.assertEqual(led.phases["warmup"][0]["coverage"]["state"], "repeated")
        led.begin_step()
        led.observe_units({"a": (10, True), "c": (10, True)}, "wrong identity")   # b omitted, c unexpected
        led.end_step(True, "hook")
        self.assertEqual(led.phases["warmup"][1]["coverage"]["state"], "partial")

    def tied_model(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(5, 4)
                self.head = torch.nn.Linear(4, 5, bias=False)
                self.head.weight = self.emb.weight                     # tied
                self.unused = torch.nn.Linear(4, 4)                    # trainable but never used in forward
                self.frozen = torch.nn.Linear(4, 4)
                self.frozen.requires_grad_(False)

            def forward(self, x):
                return self.head(self.emb(x)).sum()
        return M()

    def test_native_map_handles_tied_unused_frozen_and_rejects_foreign_or_duplicate_slots(self):
        m = self.tied_model()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
        u = self.fw.native_unit_map(m, opt)
        self.assertEqual(u["problems"], [])
        self.assertEqual(u["summary"]["unique_trainable"], 3)                  # emb/head weight, unused w + b
        self.assertEqual(u["tied_aliases"], {"emb.weight": ["head.weight"]})
        self.assertEqual(sorted(u["frozen"]), ["frozen.bias", "frozen.weight"])
        led = self.bl.BoundaryLedger("ddp", True, "in_step", {k: p.numel() for k, p in u["units"].items()}, 1, "n")
        m(torch.tensor([1, 2])).backward()
        led.begin_step()
        led.observe_units(self.bl.scan_units(u["units"]), "in-step")
        led.end_step(True, "hook")
        self.assertEqual(led.phases["warmup"][0]["coverage"]["state"], "partial")   # the unused layer has no grad
        foreign = torch.optim.AdamW(list(m.parameters()) + [torch.nn.Parameter(torch.zeros(3))], lr=1e-3)
        self.assertTrue(self.fw.native_unit_map(m, foreign)["problems"])
        dup = torch.optim.AdamW([{"params": [m.emb.weight]}, {"params": [m.unused.weight, m.unused.bias]}], lr=1e-3)
        dup.param_groups[1]["params"].append(m.emb.weight)                    # same tensor in two slots
        self.assertTrue(self.fw.native_unit_map(m, dup)["problems"])

    def fake_ds(self, pad=1, drop=False, stray=False):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2))
        params = list(model.parameters())
        n = sum(p.numel() for p in params)

        class DeepSpeedZeroOptimizer:
            pass
        zo = DeepSpeedZeroOptimizer()
        zo.bit16_groups = [params[:-1] if drop else params + ([torch.nn.Parameter(torch.zeros(2))] if stray else [])]
        zo.partition_size = [n + pad]
        zo.bit16_groups_flat = [torch.zeros(n + pad)]
        zo.single_partition_of_fp32_groups = [torch.zeros(n + pad)]
        zo.real_dp_process_group = [None]
        zo.nccl_start_alignment_factor = 2
        engine = types.SimpleNamespace(optimizer=zo, module=model)
        opt = types.SimpleNamespace(param_groups=[{"params": [zo.single_partition_of_fp32_groups[0]]}])
        return engine, opt

    def ds_layout(self, **kw):
        engine, opt = self.fake_ds(**kw)
        with mock.patch.object(self.fw.dist, "get_rank", lambda group=None: 0), \
                mock.patch.object(self.fw.dist, "get_world_size", lambda group=None: 1):
            return self.fw.deepspeed_zero12_layout(engine, opt)

    def test_deepspeed_partition_map_accounts_padding_and_rejects_missing_or_stray(self):
        lay = self.ds_layout(pad=1)
        self.assertIsNotNone(lay["accounting"])
        self.assertEqual(lay["model_to_partition_map"][0]["padding"], 1)
        self.assertEqual(lay["model_to_partition_map"][0]["params"], 4)
        self.assertIsNone(self.ds_layout(pad=2)["accounting"])                 # padding >= alignment
        self.assertIsNone(self.ds_layout(drop=True)["accounting"])             # a model parameter not in any group
        self.assertIsNone(self.ds_layout(stray=True)["accounting"])            # a group tensor that is not a parameter


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class StrengthenedUpdateEvidence(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(SRC))
        self.addCleanup(sys.path.remove, str(SRC))
        from mgbench.train import boundary_ledger, update_evidence
        self.ue, self.bl = update_evidence, boundary_ledger
        torch.manual_seed(1)

    def run_opt(self, cls=torch.optim.AdamW, steps=2, **kw):
        model = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.Linear(6, 3))
        opt = cls(model.parameters(), lr=1e-3, weight_decay=0.01, **kw)
        named = list(model.named_parameters())
        plan = self.ue.build_plan([named[0], named[-1]])
        record_before = self.ue.plan_record(plan)
        smp = self.ue.AdamWCallSampler(plan)
        opt.register_step_pre_hook(lambda o, a, k: smp.pre(o))
        opt.register_step_post_hook(lambda o, a, k: smp.post(o))
        for _ in range(steps):
            model(torch.randn(4, 8)).pow(2).mean().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        return smp, record_before

    def test_foreach_and_single_tensor_paths_pass_with_full_reporting(self):
        for foreach in (True, False):
            smp, plan = self.run_opt(foreach=foreach)
            for call in smp.calls:
                self.assertEqual(call["verdict"], "verified_sampled", foreach)
                self.assertEqual(call["selection"], "plan fixed before step 1")
                self.assertGreater(call["discriminating_samples"], 0)
                c = call["checks"][0]
                for key in ("max_abs_err_param", "max_rel_err_param", "max_abs_err_m", "max_abs_err_v",
                            "bias_correction1", "bias_correction2", "worst", "flags", "dtypes"):
                    self.assertIn(key, c)
                self.assertEqual(c["flags"]["foreach"], foreach)
                self.assertEqual(c["dtypes"]["param"], "torch.float32")
                self.assertEqual(c["worst"]["flat_index"] in c["idx"], True)
            second = smp.calls[1]["checks"][0]
            self.assertEqual((second["step_before"], second["step_after"]), (1.0, 2.0))   # post-warmup semantics
            self.assertAlmostEqual(second["bias_correction1"], 1 - 0.9 ** 2)
            self.assertEqual([c["name"] for c in smp.calls[0]["checks"]], [p["name"] for p in plan])

    def test_no_op_control_with_discriminating_samples_fails(self):
        class NoOp(torch.optim.AdamW):
            @torch.no_grad()
            def step(self, closure=None):
                return None
        smp, _ = self.run_opt(cls=NoOp, steps=1)
        call = smp.calls[0]
        self.assertGreater(call["discriminating_samples"], 0)       # the expected change exceeds the tolerance...
        self.assertEqual(call["verdict"], "failed")                 # ...and did not happen

    def test_no_op_after_a_real_step_with_existing_state_fails(self):
        model = torch.nn.Linear(8, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        model(torch.randn(3, 8)).pow(2).mean().backward()
        opt.step()                                                  # real warmup step creates state (step 1)
        plan = self.ue.build_plan(list(model.named_parameters()))
        smp = self.ue.AdamWCallSampler(plan)
        model(torch.randn(3, 8)).pow(2).mean().backward()
        smp.pre(opt)                                                # then a call that changes nothing
        smp.post(opt)
        call = smp.calls[0]
        self.assertGreater(call["discriminating_samples"], 0)
        self.assertEqual(call["verdict"], "failed")
        self.assertIn("step 1.0 -> 1.0", call["checks"][0]["reason"])

    def test_tolerance_formula_is_unchanged_and_documented(self):
        t = self.ue.FROZEN_TOLERANCES
        self.assertEqual((t["param_ulps"], t["param_rel_of_change"], t["moment_ulps"]), (8, 1e-4, 8))
        self.assertIn("max(|theta_before|,|theta_exp|)", t["formula"]["param"])
        self.assertEqual(self.ue.ulp32(1.0), 2.0 ** -23)
        self.assertEqual(self.ue.ulp32(0.75), 2.0 ** -24)

    def test_incomplete_plan_sampling_leaves_update_evidence_unverified(self):
        led = self.bl.BoundaryLedger("ddp", True, "in_step", {"a": 1}, 1, "n")
        led.layout = {"sample_plan": [{"name": "x"}, {"name": "y"}]}
        for phase in ("warmup", "measured"):
            if phase == "measured":
                led.start_measured()
            for _ in range(1 if phase == "warmup" else 3):
                led.begin_step()
                led.record_microbatch_losses([1.0, 1.0])
                led.observe_units({"a": (1, True)}, "s")
                led.base_call({"verdict": "verified_sampled", "checks": [{"name": "x"}], "sampled_elements": 1})
                led.end_step(True, "hook")
        s = led.summary(3, 1, 2)
        self.assertFalse(s["measured"]["sample_plan_complete_each_boundary"])
        a = self.bl.assess([s], checks_requested=True, update_evidence_requested=True, measurement_role="smoke",
                           diagnostics_in_timed_path=True)
        self.assertEqual(a["update_state"]["status"], "unverified")


if __name__ == "__main__":
    unittest.main()
