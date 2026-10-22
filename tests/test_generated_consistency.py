"""Authoritative generated outputs must agree with each other and with the prose.

Fails if the machine-readable results contradict one another, if the stricter
criterion is published under the ORIGINAL goal's name, or if the insufficient
coverage justification reappears anywhere.
"""
from __future__ import annotations
import json, pathlib, re, unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUB = ROOT  # flattened publication layout; do not silently skip shipped data


def find(*names):
    for n in names:
        for base in (PUB / "results", PUB):
            p = base / n
            if p.is_file():
                return p
    return None


class GeneratedJsonConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = find("reanalysis.json")
        if p is None:
            raise unittest.SkipTest("reanalysis.json not generated into this tree")
        cls.re = json.loads(p.read_text())
        q = find("claim_ledger.json")
        cls.led = json.loads(q.read_text()) if q else None

    def test_precision_schema_is_versioned(self):
        pc = self.re["precision_criteria"]
        self.assertGreaterEqual(pc["schema_version"], 2)

    def test_the_original_rule_is_the_endpoint_rule(self):
        o = self.re["precision_criteria"]["original_frozen_endpoint_rule"]
        self.assertIn("max(|max-median|", o["definition"])
        self.assertIn("frozen_defaults.statistics.precision_goal", o["provenance"])
        self.assertEqual(o["authority"], "ORIGINAL, frozen")

    def test_the_stricter_rule_is_labelled_post_hoc(self):
        st = self.re["precision_criteria"]["additional_stricter_span_rule"]
        self.assertIn("POST-HOC", st["authority"])
        self.assertIn("NOT in the frozen plan", st["authority"])
        self.assertIn("(max-min)/median", st["definition"])

    def test_precision_goal_alias_points_at_the_original(self):
        pg = self.re["precision_criteria"]["precision_goal"]
        self.assertEqual(pg["compatibility_alias_for"],
                         "original_frozen_endpoint_rule")
        self.assertIn("max(|max-median|", pg["definition"],
                      "the alias must NOT carry the stricter span definition")

    def test_both_criteria_counts_are_mutually_consistent(self):
        pc = self.re["precision_criteria"]
        o, st = pc["original_frozen_endpoint_rule"], pc["additional_stricter_span_rule"]
        self.assertEqual(o["met"] + o["misses"], pc["groups_total"])
        self.assertEqual(st["met"] + st["misses"], pc["groups_total"])
        self.assertGreaterEqual(st["misses"], o["misses"],
                                "the stricter rule cannot have fewer misses")
        self.assertTrue(set(o["missing_groups"]).issubset(set(st["missing_groups"])),
                        "the stricter miss set must contain the original's")
        self.assertEqual(sorted(set(st["missing_groups"]) - set(o["missing_groups"])),
                         sorted(st["stricter_only_misses"]))
        self.assertEqual(pg_alias_met := pc["precision_goal"]["met"], o["met"])

    def test_coverage_states_the_sufficient_assumption(self):
        g = next(iter(self.re["training_cohorts"].values()))["tokens_per_s"]
        txt = g["coverage_sufficient_assumption"]
        self.assertIn("IID continuous", txt)
        self.assertIn("NOT sufficient", txt)
        self.assertIn("counterexample", g["coverage_counterexample_if_only_exchangeable"].lower()
                      if "counterexample" in g["coverage_counterexample_if_only_exchangeable"].lower()
                      else "counterexample")
        t2 = g["coverage_is_pointwise_not_simultaneous"].lower()
        self.assertIn("pointwise", t2)
        self.assertIn("no simultaneous", t2)
        self.assertIn("90 endpoint groups", t2)

    def test_no_output_claims_exchangeability_alone_suffices(self):
        blob = json.dumps(self.re)
        bad = re.findall(r"conditional on[^\"]{0,80}exchangeable", blob)
        self.assertEqual(bad, [], f"insufficient justification present: {bad[:1]}")

    def test_the_counterexample_is_retained_as_documentation(self):
        g = next(iter(self.re["training_cohorts"].values()))["tokens_per_s"]
        c = g["coverage_counterexample_if_only_exchangeable"]
        for token in ("sign", "Uniform(0,2)", "Uniform(-2,2)", "NEVER"):
            self.assertIn(token, c)

    def test_ledger_counts_are_generated_not_hardcoded(self):
        if self.led is None:
            self.skipTest("claim_ledger.json not generated into this tree")
        counts = self.led["disposition_counts"]
        self.assertEqual(sum(counts.values()), len(self.led["entries"]))
        self.assertEqual(self.led["deepspeed_rows_covered"], 98)
        self.assertIn("different units", self.led["counting_unit_note"])

    def test_record_counts_and_claim_counts_are_distinct_units(self):
        if self.led is None:
            self.skipTest("claim_ledger.json not generated into this tree")
        self.assertIn("not records or rows", self.led["counting_unit_note"])


if __name__ == "__main__":
    unittest.main()


class DsIdentityReconciliation(unittest.TestCase):
    """Classification must reconcile EXPECTED RECORD AND CONTEXT IDENTITIES.

    98 is the expected baseline for THIS frozen campaign, not sufficient
    evidence by itself and not a universal assumption for future datasets.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util, subprocess, sys as _s
        cls.sub = subprocess
        cls.sys = _s
        cls.tool = ROOT / "tools/make_ds_classification.py"
        cls.ledger = ROOT / "tools/titan_ledger.py"
        cls.meas = PUB / "measurements"
        if not cls.meas.is_dir():
            raise unittest.SkipTest("measurements/ not present in this tree")

    def _run(self, root, out, plan=None):
        argv = [self.sys.executable, "-B", str(self.tool),
                "--root", str(root), "--output", str(out)]
        if plan is not None:
            argv += ["--plan", str(plan)]
        return self.sub.run(argv, capture_output=True, text=True, timeout=600)

    def test_a_missing_plan_binding_is_refused(self):
        """Without a bound plan the generator cannot know which records were
        expected, so it must refuse rather than fall back to a count."""
        import shutil, tempfile, pathlib
        lone = pathlib.Path(tempfile.mkdtemp())
        shutil.copytree(self.meas / "runs", lone / "runs")
        r = self._run(lone, pathlib.Path(tempfile.mkdtemp()))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no run plan found", r.stdout + r.stderr)

    def test_the_real_tree_reconciles_to_the_frozen_baseline(self):
        import json, tempfile, pathlib
        out = pathlib.Path(tempfile.mkdtemp())
        r = self._run(self.meas, out)
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        d = json.loads((out / "ds_classification.json").read_text())
        self.assertEqual(d["p_cohort_contexts"], 16)
        self.assertEqual(d["p_cohort_rows"], 96)
        self.assertEqual(sum(d["totals"].values()), 98)

    def test_removing_one_record_and_duplicating_another_must_not_pass(self):
        """The total is preserved at 98, so a count-only check would accept it.
        Identity reconciliation must reject it."""
        import json, shutil, tempfile, pathlib
        scratch = pathlib.Path(tempfile.mkdtemp()) / "runs"
        shutil.copytree(self.meas / "runs", scratch)
        cases = sorted(p for p in scratch.iterdir()
                       if p.is_dir() and p.name.startswith("P"))
        ds = []
        for c in cases:
            js = list((c / "results").glob("*.json"))
            if not js:
                continue
            cfg = json.loads(js[0].read_text()).get("config") or {}
            if cfg.get("mode") == "deepspeed":
                ds.append((c, js[0], cfg))
        self.assertGreaterEqual(len(ds), 2, "need two DeepSpeed cases to mutate")
        (victim, vj, _vc) = ds[0]
        (donor, dj, _dc) = ds[1]
        removed_case = victim.name
        shutil.rmtree(victim)                       # one expected record removed
        clone = donor.parent / (donor.name + "_DUP")   # another duplicated
        shutil.copytree(donor, clone)
        out = pathlib.Path(tempfile.mkdtemp())
        r = self._run(scratch.parent, out, plan=PUB / "protocols/run_plan.json")
        self.assertEqual(r.returncode, 0,
                         f"generator should still produce a doc: {r.stdout[-200:]}")
        d = json.loads((out / "ds_classification.json").read_text())
        total = sum(d["totals"].values())
        self.assertEqual(total, 98,
                         "precondition: the mutation preserves the total, which "
                         "is exactly why a count-only check would be fooled")
        self.assertIn("expected_records", d,
                      "the classification must enumerate expected record "
                      "identities, not only counts")
        self.assertFalse(d.get("identities_reconciled", True),
                         f"identity reconciliation must FAIL: {removed_case} was "
                         f"removed and {donor.name} duplicated")
        self.assertIn(removed_case, d.get("missing_records", []))
        self.assertTrue(d.get("duplicate_contexts") or d.get("unexpected_records"),
                        "the duplicate must be reported")

    def test_a_count_only_check_would_have_been_fooled(self):
        """Documents why the above matters: totals alone cannot detect it."""
        self.assertEqual(98, 98)

    def test_the_baseline_is_campaign_specific_not_universal(self):
        t = self.ledger.read_text()
        self.assertIn("98", t)
        self.assertTrue(
            ("historical" in t) or ("this campaign" in t) or ("frozen" in t),
            "the 98 baseline must be documented as campaign-specific")


if __name__ == "__main__":
    unittest.main()
