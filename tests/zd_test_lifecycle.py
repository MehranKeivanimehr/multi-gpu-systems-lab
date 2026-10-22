"""Multi-rank ownership and cleanup, exercised with HARMLESS fixtures.

These tests drive the real production supervision mechanism
(`zerodiag.ownership.run_ranks`) — the same code path the model worker runs
under — with a fixture that imports no model worker and never touches CUDA.

Why harmless fixtures and not the real worker: executing the GPU-capable worker
on this unisolated host is what caused INCIDENT_001. Device access cannot be
revoked here without root, so the real worker is exercised only inside a
bubblewrap sandbox where the device nodes are absent (see
`tests/test_isolated_worker.py`), and otherwise deferred.

The defect these tests exist for, reproduced before the fix: the previous
launcher called `os.setsid` in rank 0 and then `os.setpgid(0, rank0_pid)` in
every later rank, which is EPERM across sessions and raised inside
`preexec_fn`, so only rank 0 ever launched.
"""
import json, os, pathlib, signal, subprocess, sys, tempfile, time, unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
from zerodiag.ownership import JobOwnership, run_ranks, JOB_TOKEN_ENV  # noqa: E402

FIXTURE = str(pathlib.Path(__file__).resolve().parent / "fixtures" / "harmless_rank.py")
GIB = 1073741824


def argv_builder(world_size, *, mode_for=None, sleep_for=None):
    def _argv(r, rd):
        argv = [sys.executable, "-I", FIXTURE, "--output", str(rd),
                "--rank", str(r), "--world-size", str(world_size),
                "--case-id", "LIFECYCLE"]
        if mode_for:
            argv += ["--mode", mode_for(r)]
        if sleep_for:
            argv += ["--sleep", str(sleep_for(r))]
        return argv
    return _argv


class Sentinel:
    """An unrelated process of mine that must survive every cleanup."""

    def __enter__(self):
        self.p = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(600)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self

    def alive(self):
        return self.p.poll() is None

    def __exit__(self, *exc):
        self.p.terminate()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()


class NormalCompletion(unittest.TestCase):

    def _run(self, ws):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"lc{ws}_"))
        with Sentinel() as s:
            rep = run_ranks(world_size=ws,
                            argv_for=argv_builder(ws, sleep_for=lambda r: 1.5),
                            out_dir=tmp, timeout_s=120, budget_bytes=8 * GIB,
                            interval_s=0.2)
            self.assertTrue(s.alive(), "an unrelated process was killed")
        return rep, tmp

    def test_world_size_two_four_and_six_all_ranks_launch(self):
        for ws in (2, 4, 6):
            rep, tmp = self._run(ws)
            self.assertEqual(rep["n_launched"], ws, f"ws={ws}")
            self.assertEqual(rep["ranks_launched"], list(range(ws)), f"ws={ws}")
            self.assertFalse(rep["partial_launch"], f"ws={ws}")
            self.assertEqual([rep["exit_codes"][str(r)] for r in range(ws)],
                             [0] * ws, f"ws={ws}")
            self.assertTrue(rep["owned_absent_confirmed"], f"ws={ws}")
            recs = sorted(tmp.glob("rank*/rank_record.json"))
            self.assertEqual(len(recs), ws, f"ws={ws}")

    def test_every_rank_got_its_own_session_and_the_job_token(self):
        rep, tmp = self._run(4)
        pgids, tokens = set(), set()
        for r in range(4):
            d = json.loads((tmp / f"rank{r}" / "rank_record.json").read_text())
            self.assertEqual(d["rank"], r)
            self.assertEqual(d["pid"], d["pgid"])     # own group leader
            self.assertEqual(d["pid"], d["sid"])      # own session leader
            pgids.add(d["pgid"]); tokens.add(d["job_token"])
        self.assertEqual(len(pgids), 4, "ranks must not share a process group")
        self.assertEqual(tokens, {rep["token"]})

    def test_the_ledger_is_persisted_for_cleanup_after_launcher_exit(self):
        rep, tmp = self._run(2)
        led = json.loads((tmp / "ownership_ledger.json").read_text())
        self.assertEqual(led["token"], rep["token"])
        self.assertEqual(len(led["ranks"]), 2)
        self.assertTrue(led["cleanable_after_launcher_exit"])
        for r in led["ranks"]:
            self.assertIn("pid", r); self.assertIn("pgid", r)

    def test_aggregate_accounting_is_whole_job_and_not_a_cap(self):
        rep, tmp = self._run(4)
        agg = rep["aggregate"]
        self.assertGreater(agg["peak_owned_rss_bytes"], 0)
        self.assertGreaterEqual(agg["samples"], 1)
        self.assertFalse(agg["enforced_cap"])
        self.assertIn("token sweep", agg["discovery"])
        self.assertIn("over-estimate", agg["shared_pages"])


class Failures(unittest.TestCase):

    def _run(self, ws, mode_for, **kw):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="lcf_"))
        with Sentinel() as s:
            rep = run_ranks(world_size=ws, argv_for=argv_builder(ws, mode_for=mode_for),
                            out_dir=tmp, timeout_s=kw.pop("timeout_s", 120),
                            budget_bytes=kw.pop("budget_bytes", 8 * GIB),
                            interval_s=0.2, **kw)
            self.assertTrue(s.alive(), "an unrelated process was killed")
        return rep, tmp

    def test_rank_zero_failure_is_reported_not_masked(self):
        rep, tmp = self._run(4, lambda r: "fail" if r == 0 else "ok")
        self.assertEqual(rep["exit_codes"]["0"], 7)
        self.assertEqual([rep["exit_codes"][str(r)] for r in (1, 2, 3)], [0, 0, 0])
        self.assertTrue(rep["owned_absent_confirmed"])
        err = (tmp / "rank0" / "worker.stderr").read_text()
        self.assertIn("rank 0 failing on purpose", err,
                      "stderr must be preserved, not suppressed")

    def test_nonzero_rank_failure_is_reported(self):
        rep, tmp = self._run(6, lambda r: "fail" if r == 4 else "ok")
        self.assertEqual(rep["exit_codes"]["4"], 7)
        self.assertEqual(rep["exit_codes"]["0"], 0)
        self.assertTrue(rep["owned_absent_confirmed"])
        self.assertIn("rank 4 failing on purpose",
                      (tmp / "rank4" / "worker.stderr").read_text())

    def test_partial_launch_is_a_failure_not_a_smaller_job(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="lcp_"))

        def _argv(r, rd):
            if r == 2:
                return [str(rd / "does-not-exist-executable")]
            return argv_builder(4)(r, rd)

        with Sentinel() as s:
            rep = run_ranks(world_size=4, argv_for=_argv, out_dir=tmp,
                            timeout_s=60, budget_bytes=8 * GIB, interval_s=0.2)
            self.assertTrue(s.alive())
        self.assertTrue(rep["partial_launch"])
        self.assertIn(2, rep["launch_errors"])
        self.assertLess(rep["n_launched"], 4)
        self.assertTrue(rep["owned_absent_confirmed"],
                        "ranks already started must be cleaned up")

    def test_timeout_terminates_the_whole_job(self):
        tmp2 = pathlib.Path(tempfile.mkdtemp(prefix="lct_"))
        with Sentinel() as s:
            rep2 = run_ranks(world_size=3,
                             argv_for=argv_builder(3, sleep_for=lambda r: 120),
                             out_dir=tmp2, timeout_s=3, budget_bytes=8 * GIB,
                             interval_s=0.2, grace_s=10)
            self.assertTrue(s.alive())
        self.assertTrue(rep2["timed_out"])
        self.assertTrue(rep2["owned_absent_confirmed"])
        self.assertEqual(len(list(tmp2.glob("rank*/done.flag"))), 0)

    def test_budget_breach_terminates_the_whole_job(self):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="lcb_"))
        with Sentinel() as s:
            rep = run_ranks(world_size=3,
                            argv_for=argv_builder(3, sleep_for=lambda r: 60),
                            out_dir=tmp, timeout_s=60,
                            budget_bytes=1,          # 1 byte: immediate breach
                            interval_s=0.2, grace_s=10)
            self.assertTrue(s.alive())
        self.assertTrue(rep["budget_breached"])
        self.assertTrue(rep["owned_absent_confirmed"])

    def test_no_surviving_owned_descendant_even_if_it_escaped_its_group(self):
        """A rank spawns a long-lived child; cleanup must still find it."""
        rep, tmp = self._run(3, lambda r: "child" if r == 1 else "ok")
        self.assertTrue((tmp / "rank1" / "spawned_child.flag").is_file())
        self.assertTrue(rep["owned_absent_confirmed"],
                        f"surviving: {rep['cleanup'].get('surviving_pids')}")
        self.assertEqual(rep["cleanup"]["foreign_processes_signalled"], 0)

    def test_cleanup_is_possible_after_the_leader_exits(self):
        """The ledger alone is enough to clean up a job the launcher lost."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="lcl_"))
        own = JobOwnership("zd-" + os.urandom(6).hex(),
                           ledger_path=tmp / "ownership_ledger.json")
        procs = []
        for r in range(3):
            rd = tmp / f"rank{r}"; rd.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen(
                [sys.executable, "-I", FIXTURE, "--output", str(rd),
                 "--rank", str(r), "--world-size", "3", "--case-id", "LEADEREXIT",
                 "--sleep", "120"],
                env=own.env_for_child(), start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            own.register(r, p.pid); procs.append(p)
        self.assertEqual(len(own.owned_pids()), 3)
        # a fresh supervisor, constructed only from the persisted ledger
        led = json.loads((tmp / "ownership_ledger.json").read_text())
        revived = JobOwnership(led["token"])
        for r in led["ranks"]:
            revived.ranks[r["rank"]] = r
        with Sentinel() as s:
            rep = revived.terminate(grace_s=10)
            self.assertTrue(s.alive())
        self.assertTrue(rep["owned_absent_confirmed"], rep["surviving_pids"])
        for p in procs:
            p.wait(timeout=10)

    def test_an_unrelated_sentinel_is_never_signalled(self):
        with Sentinel() as s:
            tmp = pathlib.Path(tempfile.mkdtemp(prefix="lcs_"))
            rep = run_ranks(world_size=2,
                            argv_for=argv_builder(2, sleep_for=lambda r: 30),
                            out_dir=tmp, timeout_s=2, budget_bytes=8 * GIB,
                            interval_s=0.2, grace_s=10)
            self.assertTrue(rep["timed_out"])
            self.assertTrue(rep["owned_absent_confirmed"])
            self.assertTrue(s.alive(), "cleanup killed an unrelated process")
            self.assertEqual(rep["cleanup"]["foreign_processes_signalled"], 0)
            self.assertEqual(rep["cleanup"]["scope"],
                             "owned ranks and their descendants only")


class ReproducedDefect(unittest.TestCase):
    """The original defect, kept as an executable record."""

    def test_cross_session_setpgid_is_eperm(self):
        """Why the old launcher could not launch rank 1 and beyond."""
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="lcr_"))
        first = subprocess.Popen(
            [sys.executable, "-I", FIXTURE, "--output", str(tmp / "r0"),
             "--rank", "0", "--world-size", "2", "--case-id", "REPRO",
             "--sleep", "5"],
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            with self.assertRaises(subprocess.SubprocessError):
                subprocess.Popen(
                    [sys.executable, "-I", FIXTURE, "--output", str(tmp / "r1"),
                     "--rank", "1", "--world-size", "2", "--case-id", "REPRO"],
                    preexec_fn=(lambda pg=first.pid: os.setpgid(0, pg)),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            first.terminate()
            try:
                first.wait(timeout=10)
            except subprocess.TimeoutExpired:
                first.kill()

    def test_the_production_launcher_no_longer_uses_preexec_fn(self):
        src = (ROOT / "zerodiag/ownership.py").read_text()
        self.assertNotIn("preexec_fn=", src,
                         "ownership must not pass preexec_fn to Popen")
        self.assertIn("start_new_session=True", src)
        cli = (ROOT / "zerodiag/cli.py").read_text()
        i = cli.index("def execute_case(")
        self.assertNotIn("preexec_fn=", cli[i:],
                         "execute_case must not pass preexec_fn to Popen")


if __name__ == "__main__":
    unittest.main(verbosity=2)
