"""The packaged campaign CLI: real entry point, plan-only and fail-stop.

Driver-level tests of the actual CLI, not of launch.run_case. All launches go
through an INJECTED harmless fake launcher placed on sys.path ahead of the real
one, so no GPU interface is touched and no production bypass is added to the
shipped driver. Fixtures are SYNTHETIC and live outside measurement inputs.
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, tempfile, unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent          # .../source
# Publication layout puts the data dirs beside `source/`; the packaged layout
# puts them inside the repository root. Pick whichever actually exists.
def _publication_base():
    """Pick the layout that actually holds the frozen plan.

    Publication layout keeps the data directories beside `source/`; the
    packaged layout keeps them inside the repository root. Choosing on the
    presence of `protocols/run_plan.json` rather than of `protocols/` avoids
    binding to an unrelated sibling directory that merely shares the name.
    """
    for base in (ROOT, ROOT.parent):
        if (base / "protocols" / "run_plan.json").is_file():
            return base
    return ROOT.parent


PUB = _publication_base()
CLI = PUB / "campaign_tools/run_blocks_routed.py"

FAKE = '''
"""Injected harmless fake launcher. No device access."""
import json, os, pathlib
ADAPTERS = ("torchrun", "plain", "vllm_parent", "loopback_server")
_MODE = os.environ["FAKE_MODE"]
_N = {"n": 0}
def adapter_for_recipe(recipe):
    return "plain" if recipe == "single" else "torchrun"
def _parse_uuid_bindings(items):
    return {int(k): v for k, v in (x.split("=", 1) for x in items)}
class LaunchRefusal(RuntimeError):
    pass
def run_case(**kw):
    _N["n"] += 1
    first = _N["n"] == 1
    p = pathlib.Path(os.environ["FAKE_LOG"])
    with open(p, "a") as fh:
        fh.write(kw["case"] + "\\n")
    if _MODE == "fail_first" and first:
        return {"case": kw["case"], "exit_code": 1,
                "cleanup": {"group_absent_confirmed": False}}
    if _MODE == "refuse_first" and first:
        raise LaunchRefusal("injected refusal on the first case")
    if _MODE == "null_exit" and first:
        return {"case": kw["case"], "exit_code": None,
                "cleanup": {"group_absent_confirmed": True}}
    if _MODE == "timeout_first" and first:
        return {"case": kw["case"], "exit_code": 124,
                "cleanup": {"group_absent_confirmed": True}}
    if _MODE == "uncertain_cleanup" and first:
        return {"case": kw["case"], "exit_code": 0,
                "cleanup": {"group_absent_confirmed": None}}
    return {"case": kw["case"], "exit_code": 0,
            "cleanup": {"group_absent_confirmed": True}}
'''


class CampaignCliTests(unittest.TestCase):
    def setUp(self):
        """Build a minimal throwaway tree whose bench/launch.py IS the fake.

        The shipped CLI inserts its resolved bench directory at the front of
        sys.path, so PYTHONPATH cannot override it -- correctly, since that
        stops an outsider shadowing the audited launcher. Injection therefore
        happens by giving the CLI a tree to resolve, which needs no bypass in
        the shipped driver.
        """
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.fakeroot = self.tmp / "faketree"
        bench = self.fakeroot / "source/studies/titan-1080/bench"
        bench.mkdir(parents=True)
        (bench / "launch.py").write_text(FAKE)
        (self.fakeroot / "protocols").mkdir(parents=True)
        (self.fakeroot / "protocols/run_plan.json").write_text(
            (PUB / "protocols/run_plan.json").read_text())
        (self.fakeroot / "campaign_tools").mkdir(parents=True)
        (self.fakeroot / "campaign_tools/run_blocks_routed.py").write_text(
            CLI.read_text())
        (self.fakeroot / "campaign_tools/gpu_attempt.sh").write_text(
            (PUB / "campaign_tools/gpu_attempt.sh").read_text())
        self.fakecli = self.fakeroot / "campaign_tools/run_blocks_routed.py"
        self.log = self.tmp / "cases.log"
        self.out = self.tmp / "out"

    def run_cli(self, mode=None, execute=False, blocks="1", extra=()):
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["FAKE_LOG"] = str(self.log)
        # a real-tree run for plan-only; the fake tree whenever a mode is set
        cli, root = (CLI, PUB)
        if mode:
            env["FAKE_MODE"] = mode
            cli, root = (self.fakecli, self.fakeroot)
        argv = [sys.executable, "-B", str(cli), "--root", str(root),
                "--output", str(self.out), "--reservation-root", str(self.tmp / "res"),
                "--concurrency-limit", "6", "--aggregate-limit", "6",
                "--authorized-devices", "0,1,2,3,4,5,8,9",
                "--authorization-expiry-epoch", "99999999999",
                "--blocks", blocks, *extra]
        if execute:
            argv.append("--execute")
        return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300)

    def cases_run(self):
        return self.log.read_text().split() if self.log.exists() else []

    def inventory(self):
        return json.loads((self.out / "case_inventory.json").read_text())

    # -- plan only -----------------------------------------------------------
    def test_plan_only_works_with_no_launches(self):
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertIn("planned cases: 33", r.stdout)
        self.assertIn("nothing was launched", r.stdout)
        self.assertEqual(self.cases_run(), [])

    def test_plan_only_records_its_bindings_and_hashes(self):
        r = self.run_cli()
        self.assertIn("bench package:", r.stdout)
        self.assertIn("launch.py sha256:", r.stdout)
        self.assertIn("plan sha256:", r.stdout)

    # -- fail-stop -----------------------------------------------------------
    def _assert_stopped_after_first(self, r):
        self.assertNotEqual(r.returncode, 0, "a failed first case must be nonzero")
        self.assertEqual(len(self.cases_run()), 1,
                         f"a second case was attempted: {self.cases_run()}")
        inv = self.inventory()
        self.assertTrue(inv["stopped"])
        outcomes = [x["outcome"] for x in inv["inventory"]]
        self.assertEqual(outcomes.count("not_run_after_stop"), inv["planned"] - 1)
        self.assertEqual(len(inv["inventory"]), inv["planned"],
                         "every planned case must stay visible")

    def test_failed_first_case_stops_the_sequence(self):
        self._assert_stopped_after_first(self.run_cli("fail_first", execute=True))

    def test_refused_first_launch_stops_the_sequence(self):
        r = self.run_cli("refuse_first", execute=True)
        self._assert_stopped_after_first(r)
        self.assertEqual(self.inventory()["inventory"][0]["outcome"], "refused")

    def test_null_exit_stops_the_sequence(self):
        self._assert_stopped_after_first(self.run_cli("null_exit", execute=True))

    def test_timeout_exit_stops_the_sequence(self):
        self._assert_stopped_after_first(self.run_cli("timeout_first", execute=True))

    def test_uncertain_cleanup_stops_the_sequence(self):
        self._assert_stopped_after_first(self.run_cli("uncertain_cleanup", execute=True))

    # -- the successful path, separately -------------------------------------
    def test_a_fully_successful_sequence_returns_zero(self):
        r = self.run_cli("all_ok", execute=True)
        self.assertEqual(r.returncode, 0, r.stdout[-500:])
        inv = self.inventory()
        self.assertFalse(inv["stopped"])
        self.assertEqual(len(self.cases_run()), inv["planned"])
        self.assertTrue(all(x["outcome"] == "attempted" for x in inv["inventory"]))

    def test_the_retired_shell_bypass_stays_retired(self):
        sh = PUB / "campaign_tools/gpu_attempt.sh"  # the SHIPPED one
        r = subprocess.run(["bash", str(sh), "X", "0", "60", "plain"],
                           capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("retired", (r.stdout + r.stderr).lower())

    def test_the_real_launcher_cannot_be_shadowed_by_pythonpath(self):
        """A safety property worth pinning: the CLI resolves its own bench
        directory first, so PYTHONPATH cannot substitute a different launcher."""
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = str(self.fakeroot / "source/studies/titan-1080/bench")
        env["FAKE_MODE"] = "all_ok"; env["FAKE_LOG"] = str(self.log)
        r = subprocess.run(
            [sys.executable, "-B", str(CLI), "--root", str(PUB),
             "--output", str(self.out), "--reservation-root", str(self.tmp / "r3"),
             "--concurrency-limit", "6", "--aggregate-limit", "6",
             "--authorized-devices", "0", "--authorization-expiry-epoch", "9e12",
             "--blocks", "1", "--execute"],
            capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(self.cases_run(), [],
                         "PYTHONPATH shadowed the audited launcher")
        self.assertNotEqual(r.returncode, 0)

    def test_a_missing_plan_is_refused_with_the_paths_tried(self):
        empty = self.tmp / "noplan"; empty.mkdir()
        env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""
        r = subprocess.run(
            [sys.executable, "-B", str(CLI), "--root", str(empty),
             "--output", str(self.out), "--reservation-root", str(self.tmp / "r2"),
             "--concurrency-limit", "6", "--aggregate-limit", "6",
             "--authorized-devices", "0", "--authorization-expiry-epoch", "9e12",
             "--blocks", "1"], capture_output=True, text=True, env=env, timeout=120)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no run plan found", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
