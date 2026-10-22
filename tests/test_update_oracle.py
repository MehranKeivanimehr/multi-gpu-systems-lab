"""Oracle: positives against the real CPU optimizers, plus the five reproduced
counterexamples and the full strict contract.

All fixtures here are SYNTHETIC and live outside any measurement-input tree.
CPU only; CUDA is hidden by the runner. The oracle is CPU-tested and NOT
integrated into live ZeRO validation.
"""
from __future__ import annotations

import importlib.util, math, pathlib, sys, unittest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("oracle_u", ROOT / "tools/update_oracle.py")
    m = importlib.util.module_from_spec(spec); sys.modules["oracle_u"] = m
    spec.loader.exec_module(m); return m


HP = {"lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
      "weight_decay": 0.0, "decoupled_weight_decay": True, "amsgrad": False}


def T(v, n=1):
    return torch.full((n,), float(v), dtype=torch.float64)


class RealOptimizerAgreement(unittest.TestCase):
    """Positive comparisons against the installed torch CPU implementations."""

    @classmethod
    def setUpClass(cls):
        cls.o = load(); torch.manual_seed(1234)

    def _drive(self, cls_, steps=5, n=129, **kw):
        p = torch.nn.Parameter(torch.randn(n, dtype=torch.float64))
        opt = cls_([p], **kw)
        g0 = opt.param_groups[0]
        hp = {"lr": g0["lr"], "beta1": g0["betas"][0], "beta2": g0["betas"][1],
              "eps": g0["eps"], "weight_decay": g0["weight_decay"],
              "decoupled_weight_decay": cls_ is torch.optim.AdamW,
              "amsgrad": g0.get("amsgrad", False)}
        out = []
        for t in range(1, steps + 1):
            g = torch.randn(n, dtype=torch.float64) * 0.05
            st = opt.state.get(p, {})
            m0 = st.get("exp_avg", torch.zeros_like(p)).clone()
            v0 = st.get("exp_avg_sq", torch.zeros_like(p)).clone()
            mx0 = st.get("max_exp_avg_sq")
            if hp["amsgrad"]:
                hp["max_exp_avg_sq"] = (mx0.clone() if mx0 is not None
                                        else torch.zeros_like(p))
            before = p.detach().clone(); p.grad = g.clone(); opt.step()
            st = opt.state[p]
            out.append(self.o.verify_update(
                param_before=before, param_after=p.detach().clone(), grad=g,
                exp_avg=m0, exp_avg_sq=v0, step=t, hparams=dict(hp),
                observed_exp_avg=st["exp_avg"], observed_exp_avg_sq=st["exp_avg_sq"],
                coverage=self.o.FULL, n_elements_checked=n, n_elements_total=n,
                model_map={"expected_elements": n}))
        return out

    def test_adamw(self):
        for r in self._drive(torch.optim.AdamW, lr=1e-3):
            self.assertTrue(r["verification_successful"], r["reasons"])
            self.assertTrue(r["discriminating"])
            self.assertTrue(r["whole_model_verified"])

    def test_adamw_weight_decay(self):
        for r in self._drive(torch.optim.AdamW, lr=1e-3, weight_decay=0.05):
            self.assertTrue(r["verification_successful"], r["reasons"])

    def test_adam_coupled(self):
        for r in self._drive(torch.optim.Adam, lr=1e-3, weight_decay=0.05):
            self.assertTrue(r["verification_successful"], r["reasons"])
            self.assertEqual(r["optimizer_mode"], "adam_coupled")

    def test_amsgrad_supported_explicitly(self):
        for r in self._drive(torch.optim.AdamW, lr=1e-3, amsgrad=True):
            self.assertTrue(r["verification_successful"], r["reasons"])

    def test_float32_is_supported_with_its_own_tolerance(self):
        n = 64
        p = torch.nn.Parameter(torch.randn(n, dtype=torch.float32))
        opt = torch.optim.AdamW([p], lr=1e-3)
        g0 = opt.param_groups[0]
        g = torch.randn(n, dtype=torch.float32) * 0.05
        before = p.detach().clone(); p.grad = g.clone(); opt.step()
        st = opt.state[p]
        r = self.o.verify_update(
            param_before=before, param_after=p.detach().clone(), grad=g,
            exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p), step=1,
            hparams={"lr": g0["lr"], "beta1": g0["betas"][0], "beta2": g0["betas"][1],
                     "eps": g0["eps"], "weight_decay": g0["weight_decay"],
                     "decoupled_weight_decay": True, "amsgrad": False},
            observed_exp_avg=st["exp_avg"], observed_exp_avg_sq=st["exp_avg_sq"],
            coverage=self.o.FULL, n_elements_checked=n, n_elements_total=n)
        self.assertTrue(r["verification_successful"], r["reasons"])
        self.assertGreater(r["rtol"], 1e-9, "float32 must get a looser rtol than float64")

    def test_assuming_default_weight_decay_is_rejected(self):
        n = 32
        p = torch.nn.Parameter(torch.randn(n, dtype=torch.float64))
        opt = torch.optim.AdamW([p], lr=1e-3)      # default weight_decay is 0.01
        g = torch.randn(n, dtype=torch.float64)
        before = p.detach().clone(); p.grad = g.clone(); opt.step()
        r = self.o.verify_update(
            param_before=before, param_after=p.detach().clone(), grad=g,
            exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p), step=1,
            hparams={**HP, "lr": 1e-3, "weight_decay": 0.0},
            coverage=self.o.FULL, n_elements_checked=n, n_elements_total=n)
        self.assertFalse(r["verification_successful"])


class ReproducedCounterexamples(unittest.TestCase):
    """The five cases the previous oracle got wrong, verbatim."""

    @classmethod
    def setUpClass(cls):
        cls.o = load()

    def _expected(self, before, grad):
        return self.o.expected_adam_update(
            param_before=before, grad=grad, exp_avg=T(0), exp_avg_sq=T(0), step=1,
            lr=HP["lr"], beta1=HP["beta1"], beta2=HP["beta2"], eps=HP["eps"],
            weight_decay=HP["weight_decay"],
            decoupled_weight_decay=HP["decoupled_weight_decay"], amsgrad=False)

    def test_case1_nan_observed_moment_is_refused(self):
        exp, _m, v = self._expected(T(1), T(1))
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_update(
                param_before=T(1), param_after=exp, grad=T(1),
                exp_avg=T(0), exp_avg_sq=T(0), step=1, hparams=HP,
                observed_exp_avg=T(float("nan")), observed_exp_avg_sq=v,
                atol=1e-12, rtol=1e-9, coverage=self.o.FULL,
                n_elements_checked=1, n_elements_total=1)

    def test_case2_missing_resolved_semantics_is_refused(self):
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_update(
                param_before=T(10), param_after=T(10), grad=T(0),
                exp_avg=T(0), exp_avg_sq=T(0), step=1,
                hparams={"lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
                coverage=self.o.FULL, n_elements_checked=1, n_elements_total=1)

    def test_case3_failed_check_cannot_claim_whole_model(self):
        exp, _m, _v = self._expected(T(1), T(1))
        r = self.o.verify_update(
            param_before=T(1), param_after=exp + 1.0, grad=T(1),
            exp_avg=T(0), exp_avg_sq=T(0), step=1, hparams=HP,
            atol=1e-12, rtol=1e-9, coverage=self.o.FULL,
            n_elements_checked=1, n_elements_total=1,
            model_map={"expected_elements": 1})
        self.assertFalse(r["verification_successful"])
        self.assertFalse(r["whole_model_verified"])

    def test_case4_duplicate_partition_identities_are_refused(self):
        rows = [{"verification_successful": True, "coverage": self.o.FULL,
                 "whole_model_verified": True, "discriminating": True, "partition": 0}
                for _ in range(2)]
        agg = self.o.verify_partition(rows, expected_partitions=[0, 1])
        self.assertFalse(agg["verification_successful"])
        self.assertFalse(agg["identities_reconciled"])
        self.assertEqual(agg["duplicate_identities"], [0])
        self.assertEqual(agg["missing_identities"], [1])
        self.assertFalse(agg["whole_model_verified"])

    def test_case5_with_principled_defaults_detects_the_absent_update(self):
        """The reproduced defect: the old oracle derived atol=1e-6, rtol=1e-5,
        giving 1e-6 + 1e-5*1000 = 0.010001, which swamped the ~1e-4 intended
        movement and passed a motionless parameter. The arithmetic-derived
        default is ~1.4e-11 at this scale, so no movement is now a genuine
        disagreement."""
        r = self.o.verify_update(
            param_before=T(1000), param_after=T(1000), grad=T(1),
            exp_avg=T(0), exp_avg_sq=T(0), step=1, hparams=HP,
            coverage=self.o.FULL, n_elements_checked=1, n_elements_total=1)
        self.assertFalse(r["verification_successful"])
        self.assertFalse(r["agrees_within_tolerance"])
        self.assertTrue(r["discriminating"],
                        "the principled tolerance must resolve a 1e-4 movement")
        self.assertLess(r["tolerance"], r["intended_movement"])
        self.assertLess(r["tolerance"], 1e-9)

    def test_a_coarse_tolerance_is_labelled_non_discriminating_not_passed(self):
        """Supplying the OLD coarse tolerance must yield the explicit
        non-discriminating label, never a pass and never a numerical failure."""
        r = self.o.verify_update(
            param_before=T(1000), param_after=T(1000), grad=T(1),
            exp_avg=T(0), exp_avg_sq=T(0), step=1, hparams=HP,
            atol=1e-6, rtol=1e-5,
            coverage=self.o.FULL, n_elements_checked=1, n_elements_total=1)
        self.assertAlmostEqual(r["tolerance"], 0.010001, places=6)
        self.assertTrue(r["agrees_within_tolerance"],
                        "at this tolerance the values do agree")
        self.assertFalse(r["discriminating"])
        self.assertFalse(r["verification_successful"],
                         "agreement that cannot distinguish a no-op is not success")
        self.assertTrue(any("NON-DISCRIMINATING" in x for x in r["reasons"]))

    def test_case5_default_tolerance_is_discriminating_at_sane_scale(self):
        """The same movement at |p|=1 is resolvable, so it must pass."""
        exp, _m, _v = self._expected(T(1), T(1))
        r = self.o.verify_update(
            param_before=T(1), param_after=exp, grad=T(1),
            exp_avg=T(0), exp_avg_sq=T(0), step=1, hparams=HP,
            coverage=self.o.FULL, n_elements_checked=1, n_elements_total=1)
        self.assertTrue(r["discriminating"])
        self.assertTrue(r["verification_successful"], r["reasons"])

    def test_a_genuinely_zero_intended_update_is_not_forced_to_fail(self):
        r = self.o.verify_update(
            param_before=T(5), param_after=T(5), grad=T(0),
            exp_avg=T(0), exp_avg_sq=T(0), step=1,
            hparams={**HP, "lr": 0.0},
            coverage=self.o.FULL, n_elements_checked=1, n_elements_total=1)
        self.assertTrue(r["zero_intended_update"])
        self.assertTrue(r["discriminating"])
        self.assertTrue(r["verification_successful"], r["reasons"])


class StrictContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = load()

    def _base(self, **over):
        kw = dict(param_before=T(1), param_after=T(1), grad=T(0), exp_avg=T(0),
                  exp_avg_sq=T(0), step=1, hparams=dict(HP), coverage=self.o.FULL,
                  n_elements_checked=1, n_elements_total=1)
        kw.update(over); return kw

    def test_nonfinite_inputs_refused(self):
        for f in ("param_before", "param_after", "grad", "exp_avg", "exp_avg_sq"):
            for bad in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=f, bad=bad), \
                     self.assertRaises(self.o.OracleRefusal):
                    self.o.verify_update(**self._base(**{f: T(bad)}))

    def test_nonfinite_hyperparameters_refused(self):
        for k in ("lr", "beta1", "beta2", "eps", "weight_decay"):
            with self.subTest(k=k), self.assertRaises(self.o.OracleRefusal):
                self.o.verify_update(**self._base(hparams={**HP, k: float("nan")}))

    def test_nonfinite_tolerances_refused(self):
        for k in ("atol", "rtol"):
            with self.subTest(k=k), self.assertRaises(self.o.OracleRefusal):
                self.o.verify_update(**self._base(**{k: float("inf")}))
            with self.subTest(k=k, neg=True), self.assertRaises(self.o.OracleRefusal):
                self.o.verify_update(**self._base(**{k: -1.0}))

    def test_booleans_refused_as_counts(self):
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_update(**self._base(n_elements_checked=True))
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_update(**self._base(n_elements_total=True))

    def test_checked_exceeding_total_refused(self):
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_update(**self._base(n_elements_checked=5, n_elements_total=1))

    def test_bad_step_refused(self):
        for s in (0, -1, 1.5, True, "1"):
            with self.subTest(s=s), self.assertRaises(self.o.OracleRefusal):
                self.o.verify_update(**self._base(step=s))

    def test_out_of_domain_betas_and_eps_refused(self):
        for k, v in (("beta1", 1.0), ("beta2", 1.0), ("beta1", -0.1), ("eps", 0.0)):
            with self.subTest(k=k, v=v), self.assertRaises(self.o.OracleRefusal):
                self.o.verify_update(**self._base(hparams={**HP, k: v}))

    def test_shape_mismatch_refused(self):
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_update(**self._base(grad=T(0, 4)))

    def test_bad_coverage_label_refused(self):
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_update(**self._base(coverage="partial"))

    def test_nonboolean_decay_mode_refused(self):
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_update(**self._base(
                hparams={**HP, "decoupled_weight_decay": "yes"}))

    def test_amsgrad_without_max_state_refused(self):
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_update(**self._base(hparams={**HP, "amsgrad": True}))

    def test_all_elements_checked_is_not_whole_model_without_a_map(self):
        r = self.o.verify_update(**self._base(hparams={**HP, "lr": 0.0}))
        self.assertTrue(r["all_supplied_elements_checked"])
        self.assertFalse(r["whole_model_verified"])
        self.assertIn("NOT established", r["whole_model_basis"])

    def test_sampled_coverage_never_whole_model(self):
        r = self.o.verify_update(**self._base(
            hparams={**HP, "lr": 0.0}, coverage=self.o.SAMPLED,
            n_elements_checked=1, n_elements_total=10,
            model_map={"expected_elements": 10}))
        self.assertFalse(r["whole_model_verified"])

    def test_partition_unexpected_identity_refused(self):
        rows = [{"verification_successful": True, "coverage": self.o.FULL,
                 "whole_model_verified": True, "discriminating": True, "partition": 7}]
        agg = self.o.verify_partition(rows, expected_partitions=[0])
        self.assertEqual(agg["unexpected_identities"], [7])
        self.assertFalse(agg["verification_successful"])

    def test_partition_missing_identity_refused(self):
        rows = [{"verification_successful": True, "coverage": self.o.FULL,
                 "whole_model_verified": True, "discriminating": True, "partition": 0}]
        agg = self.o.verify_partition(rows, expected_partitions=[0, 1, 2])
        self.assertEqual(agg["missing_identities"], [1, 2])
        self.assertFalse(agg["whole_model_verified"])

    def test_partition_without_identity_refused(self):
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_partition([{"verification_successful": True}],
                                    expected_partitions=[0])

    def test_partition_empty_or_bad_expectation_refused(self):
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_partition([], expected_partitions=[0])
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_partition(
                [{"partition": 0, "verification_successful": True}],
                expected_partitions=[])
        with self.assertRaises(self.o.OracleRefusal):
            self.o.verify_partition(
                [{"partition": 0, "verification_successful": True}],
                expected_partitions=[0, 0])

    def test_partition_non_discriminating_blocks_whole_model(self):
        rows = [{"verification_successful": True, "coverage": self.o.FULL,
                 "whole_model_verified": True, "discriminating": False, "partition": 0}]
        agg = self.o.verify_partition(rows, expected_partitions=[0])
        self.assertFalse(agg["whole_model_verified"])
        self.assertEqual(agg["non_discriminating_partitions"], [0])

    def test_integration_status_is_always_declared(self):
        r = self.o.verify_update(**self._base(hparams={**HP, "lr": 0.0}))
        self.assertIn("NOT integrated into live ZeRO", r["integration_status"])


if __name__ == "__main__":
    unittest.main()
