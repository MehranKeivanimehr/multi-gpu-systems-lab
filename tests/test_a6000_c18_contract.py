"""CPU unit and fixture tests for Candidate 18's window execution contract (no GPU, no driver: every observation is a
fixture). The integrated supervisor path (enrollment, capacity, timing, cross-window) is exercised by
studies/a6000/scripts/c18_integration_selftest.py."""
import datetime as dt
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "studies/a6000"
SRC = A / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import yaml  # noqa: E402

from mgbench import pair_controls as pc  # noqa: E402
from mgbench import window_contract as wc  # noqa: E402

U4, U5 = "GPU-00000000-0000-0000-0000-000000000109", "GPU-00000000-0000-0000-0000-000000000103"   # U5: CPU fixture only
NV = 49140 * 2**20                    # NVML total of an A6000 as nvidia-smi shows it (49,140 MiB), in BYTES
T4 = 50897289216                      # GPU 4's measured Torch total (authorized hardware evidence)
CFG = yaml.safe_load((A / "configs/control_pair.yaml").read_text())
POLICY = json.loads((A / "configs/enrollment_policy_pair.json").read_text())
FEAS = {"cases": {"R5_fp32_world1_ga4": {"predicted_reserved_gib": 39.83, "non_allocator_gib": 0.8},
                  "C5_fp32_world2_ga2": {"predicted_reserved_gib": 40.83, "non_allocator_gib": 1.8}}}
R5, C5 = CFG["cases"]


def node(over=None):
    g = {i: {"uuid": f"GPU-0000000{i}-0000-0000-0000-00000000000{i}", "pci_bus_id": f"00000000:{0x10 + i:02x}:00.0"}
         for i in range(8)}
    g[4] = {"uuid": U4, "pci_bus_id": "00000000:81:00.0"}
    g[5] = {"uuid": U5, "pci_bus_id": "00000000:82:00.0"}
    for i in (4, 5):
        g[i].update(name="NVIDIA RTX A6000", nvml_total_bytes=NV, enforced_power_limit_w=300.0)
    for (i, k), v in (over or {}).items():
        g[i][k] = v
    return {"gpus": g}


class TimingModel(unittest.TestCase):
    def test_frozen_config_bounds(self):
        m = wc.timing_model(CFG["timing"])
        self.assertEqual(m["cleanup_bound_s"], 60 + 10 + 8 * 3 + 1)            # 95
        self.assertEqual(m["watch_overshoot_s"], 5 * 3 + 3 + 1)                # 19
        self.assertEqual(m["stop_and_reap_bound_s"], 105)
        self.assertEqual(m["case_total_s"], 60 + 300 + 19 + 105 + 10)          # 494
        self.assertEqual(m["supervisor_timeout_stop_s"], 19 + 105 + 30 + 5)    # 159 >= the whole cleanup path
        self.assertEqual(m["inclusive_window_s"], 120 + 30 + 2 * 494 + 30 + 159 + 10)   # 1337, not 1,100
        self.assertLessEqual(m["inclusive_window_s"], CFG["timing"]["window_s"])
        self.assertEqual((m["worker_runtime_max_s"], m["worker_timeout_stop_s"]), (302, 60))
        self.assertGreaterEqual(m["supervisor_timeout_stop_s"], m["stop_and_reap_bound_s"] + m["watch_overshoot_s"])

    def test_generated_launch_parameters_are_durations_on_one_clock(self):
        now = 1_800_000_000.0
        exp = now + 1500
        p = wc.supervisor_launch_parameters(CFG["timing"], exp, now)
        self.assertEqual(p["problems"], [])
        self.assertEqual(p["RuntimeMaxSec"], 1500 - 159 - 10)                 # a duration from launch, not a timestamp
        self.assertEqual(p["TimeoutStopSec"], 159)
        self.assertLessEqual(p["outer_end_epoch"], exp - p["kill_margin_s"])  # stop + verification before expiry
        later = wc.supervisor_launch_parameters(CFG["timing"], exp, now + 100)
        self.assertEqual(later["RuntimeMaxSec"], p["RuntimeMaxSec"] - 100)    # same expiry -> same end; no restart
        self.assertEqual(later["work_deadline_epoch"], p["work_deadline_epoch"])

    def test_near_expiry_refused(self):
        now = 1_800_000_000.0
        p = wc.supervisor_launch_parameters(CFG["timing"], now + 1200, now)
        self.assertTrue(p["problems"])
        m = wc.timing_model(CFG["timing"])
        w = wc.work_deadline_epoch(now + 1500, m)
        self.assertTrue(wc.case_admission(w - 524, w, m)["fits"])             # 494 + final 30
        self.assertFalse(wc.case_admission(w - 523, w, m)["fits"])

    def test_malformed_timing_refused(self):
        for bad in (None, -1, float("nan"), "60", True):
            t = dict(CFG["timing"], grace_s=bad)
            with self.assertRaises(ValueError):
                wc.timing_model(t)

    def test_cleanup_classes(self):
        m = wc.timing_model(CFG["timing"])
        self.assertEqual(wc.classify_cleanup({"confirmed": True}, 12.0, m), "confirmed")
        self.assertEqual(wc.classify_cleanup({"confirmed": True}, 105.0, m), "confirmed")
        self.assertEqual(wc.classify_cleanup({"confirmed": True}, 105.1, m), "confirmed_over_budget")
        self.assertEqual(wc.classify_cleanup({"confirmed": False}, 3.0, m), "unconfirmed")
        self.assertEqual(wc.classify_cleanup({"confirmed": True}, float("nan"), m), "unconfirmed")
        self.assertEqual(pc.case_outcome("confirmed_over_budget", None, True), ("cleanup_over_budget", True))
        self.assertEqual(pc.case_outcome("unconfirmed", None, True), ("cleanup_unconfirmed", True))
        self.assertEqual(pc.case_outcome("confirmed", None, True), ("completed_valid", False))
        self.assertEqual(pc.case_outcome(True, None, True), ("completed_valid", False))

    def test_own_scope_properties(self):
        m = wc.timing_model(CFG["timing"])
        now = 1_800_000_000.0
        w = now + 1000
        ok = {"TimeoutStopUSec": 159 * 10**6, "RuntimeMaxUSec": 1000 * 10**6, "ActiveEnterTimestampMonotonic": 5 * 10**6,
              "_monotonic_now_us": 5 * 10**6}
        self.assertEqual(wc.own_scope_timing_problems(ok, m, w, now), [])
        self.assertTrue(wc.own_scope_timing_problems(dict(ok, TimeoutStopUSec=30 * 10**6), m, w, now))
        self.assertTrue(wc.own_scope_timing_problems(dict(ok, RuntimeMaxUSec=1100 * 10**6), m, w, now))
        self.assertTrue(wc.own_scope_timing_problems({"error": "x"}, m, w, now))


class Enrollment(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_frozen_policy(self):
        self.assertEqual(wc.policy_problems(POLICY), [])
        for k, v in (("enroll_indices", [5, 6]), ("eligible_indices", [4, 5, 6]), ("cases_after_enrollment", ["X"])):
            self.assertTrue(wc.policy_problems(dict(POLICY, **{k: v})), k)
        self.assertTrue(wc.policy_problems(dict(POLICY, known={"4": {"uuid": U5}})))
        self.assertIsNone(CFG["gpu_uuids"][5])                  # GPU 5 stays unknown in the real records

    def test_observations(self):
        ok = node()
        self.assertEqual(wc.observation_problems({"reads": [ok, ok]}, POLICY), [])
        cases = {"unstable": {"reads": [ok, node({(5, "nvml_total_bytes"): NV - 2**20})]},
                 "one read": {"reads": [ok]},
                 "reorder": {"reads": [node({(4, "uuid"): U5, (5, "uuid"): U4})] * 2},
                 "class": {"reads": [node({(5, "name"): "NVIDIA RTX A5000"})] * 2},
                 "limit": {"reads": [node({(5, "enforced_power_limit_w"): 250.0})] * 2},
                 "dup uuid": {"reads": [node({(5, "uuid"): "GPU-00000000-0000-0000-0000-000000000102"})] * 2},
                 "wrong unit": {"reads": [node({(5, "nvml_total_bytes"): 49140})] * 2},
                 "nan": {"reads": [node({(5, "nvml_total_bytes"): float("nan")})] * 2},
                 "missing": {"reads": [node({(5, "nvml_total_bytes"): None})] * 2}}
        for name, obs in cases.items():
            self.assertTrue(wc.observation_problems(obs, POLICY), name)

    def _seal(self, obs=None):
        auth = {"id": "fixture-auth", "expires_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()}
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        rec = wc.build_record(auth=auth, auth_sha256="a" * 64, policy_sha256="p" * 64, config_sha256="c" * 64,
                              candidate_digest="d" * 64, window_dir=str(self.tmp), sup_unit="mgbench-pair-sup-x",
                              supervisor_start_utc=now, obs=obs or {"reads": [node(), node()]}, observed_utc=[now, now],
                              capacity={})
        b = wc.seal_record(self.tmp / "enrollment", rec)
        chk = dict(auth=auth, auth_sha256="a" * 64, policy_sha256="p" * 64, config_sha256="c" * 64,
                   window_dir=str(self.tmp), sup_unit="mgbench-pair-sup-x",
                   supervisor_start_epoch=dt.datetime.now(dt.timezone.utc).timestamp() - 5,
                   expires_epoch=dt.datetime.now(dt.timezone.utc).timestamp() + 3600)
        return b, chk

    def test_sealed_once_and_revalidated(self):
        b, chk = self._seal()
        rec, p = wc.record_problems(b["dir"], b, **chk)
        self.assertEqual(p, [])
        self.assertEqual(wc.enrolled_pins(rec)[0], {4: U4, 5: U5})
        self.assertEqual(oct(os.stat(Path(b["dir"]) / "ENROLLMENT_RECORD.json").st_mode & 0o777), "0o444")
        with self.assertRaises(FileExistsError):                          # never a second record
            wc.seal_record(self.tmp / "enrollment", {"authorization_sha256": "x"})
        for k, v in (("auth_sha256", "b" * 64), ("policy_sha256", "q" * 64), ("sup_unit", "mgbench-other"),
                     ("window_dir", "/elsewhere")):
            self.assertTrue(wc.record_problems(b["dir"], b, **dict(chk, **{k: v}))[1], k)   # stale / foreign
        self.assertTrue(wc.record_problems(b["dir"], b, **dict(chk, supervisor_start_epoch=chk["expires_epoch"] - 1))[1])

    def test_altered_duplicate_missing(self):
        b, chk = self._seal()
        rp = Path(b["dir"]) / "ENROLLMENT_RECORD.json"
        os.chmod(rp, 0o644)
        rp.write_text(rp.read_text().replace(U5, "GPU-00000000-0000-0000-0000-000000000105"))
        self.assertIn("altered", " ".join(wc.record_problems(b["dir"], b, **chk)[1]))
        shutil.rmtree(self.tmp / "enrollment")
        b, chk = self._seal()
        (Path(b["dir"]) / "ENROLLMENT_RECORD_2.json").write_text("{}")
        self.assertIn("duplicate", " ".join(wc.record_problems(b["dir"], b, **chk)[1]))
        shutil.rmtree(self.tmp / "enrollment")
        b, chk = self._seal()
        (Path(b["dir"]) / "ENROLLMENT_RECORD.json").unlink()
        self.assertTrue(wc.record_problems(b["dir"], b, **chk)[1])
        self.assertTrue(wc.record_problems(self.tmp / "absent", b, **chk)[1])

    def test_consistent_rewrite_of_record_and_seal_is_still_refused(self):
        b, chk = self._seal()
        d = Path(b["dir"])
        for n in ("ENROLLMENT_RECORD.json", "ENROLLMENT_RECORD.json.seal"):
            (d / n).unlink()
        shutil.rmtree(d)
        b_new, _ = self._seal(obs={"reads": [node({(5, "uuid"): "GPU-00000000-0000-0000-0000-000000000105"})] * 2})
        self.assertTrue(wc.record_problems(b_new["dir"], b, **chk)[1])   # the in-memory binding differs

    def test_identity_or_capacity_change_after_enrollment(self):
        b, chk = self._seal()
        rec, _ = wc.record_problems(b["dir"], b, **chk)
        snap = {"gpus": {i: {"uuid": node()["gpus"][i]["uuid"], "mem_total_mib": NV / 2**20, "power_limit_w": 300.0}
                         for i in range(8)}}
        self.assertEqual(wc.post_enrollment_problems(snap, rec), [])
        for k, v in (("uuid", "GPU-00000000-0000-0000-0000-000000000105"), ("mem_total_mib", 46068.0), ("power_limit_w", 250.0),
                     ("mem_total_mib", float("nan"))):
            s2 = json.loads(json.dumps(snap))
            s2 = {"gpus": {int(i): g for i, g in s2["gpus"].items()}}
            s2["gpus"][5][k] = v
            self.assertTrue(wc.post_enrollment_problems(s2, rec), (k, v))


class Capacity(unittest.TestCase):
    def test_units(self):
        self.assertEqual(NV / 2**20, 49140)
        self.assertAlmostEqual(T4 / 2**30, 47.402, places=3)
        self.assertLess(T4, NV)                             # Torch's usable total is BELOW NVML's total
        self.assertAlmostEqual((NV - T4) / 2**20, 600.56, places=1)

    def test_gpu4_sufficient_gpu5_insufficient_matching_names(self):
        obs = {4: node()["gpus"][4], 5: dict(node()["gpus"][5], nvml_total_bytes=46068 * 2**20)}
        recs, probs = wc.enrollment_capacity(obs, CFG, FEAS, POLICY)
        self.assertTrue(recs["R5_fp32_world1_ga4/gpu4"]["pass"] and recs["C5_fp32_world2_ga2/gpu4"]["pass"])
        self.assertFalse(recs["C5_fp32_world2_ga2/gpu5"]["pass"])
        self.assertTrue(probs and all("gpu5" in p for p in probs))

    def test_no_inference_from_gpu4(self):
        recs, probs = wc.enrollment_capacity({4: node()["gpus"][4], 5: node()["gpus"][5]}, CFG, FEAS, POLICY)
        self.assertEqual(probs, [])
        self.assertIsNone(recs["C5_fp32_world2_ga2/gpu5"]["expected_torch_total_bytes"])   # GPU 5: nothing assumed
        self.assertEqual(recs["C5_fp32_world2_ga2/gpu4"]["expected_torch_total_bytes"], T4)

    def test_bad_values(self):
        for v in (None, "51527024640", float("nan"), float("inf"), -1, 49140, 51527024640.5, True):
            r = wc.device_capacity_record(5, {"nvml_total_bytes": v}, C5, FEAS["cases"]["C5_fp32_world2_ga2"])
            self.assertFalse(r["pass"] if "pass" in r else False, v)
            self.assertTrue(r["problems"], v)

    def test_runtime_checks(self):
        contract = wc.worker_capacity_contract(C5, FEAS["cases"]["C5_fp32_world2_ga2"], [4, 5],
                                               {4: node()["gpus"][4], 5: node()["gpus"][5]}, POLICY)
        self.assertTrue(wc.worker_capacity_check(contract, 0, T4)["pass"])
        self.assertTrue(wc.worker_capacity_check(contract, 1, T4)["pass"])
        low = wc.worker_capacity_check(contract, 1, 47000 * 2**20)                 # below the requirement
        self.assertFalse(low["pass"])
        self.assertFalse(wc.worker_capacity_check(contract, 0, T4 - 2**20)["pass"])  # below GPU 4's admission assumption
        self.assertFalse(wc.worker_capacity_check(contract, 1, NV + 1)["pass"])      # above NVML: inconsistent
        self.assertFalse(wc.worker_capacity_check(contract, 1, None)["pass"])
        self.assertFalse(wc.worker_capacity_check(contract, 1, 47402)["pass"])       # wrong unit
        self.assertFalse(wc.worker_capacity_check({"ranks": []}, 0, T4)["pass"])
        r5 = wc.worker_capacity_contract(R5, FEAS["cases"]["R5_fp32_world1_ga4"], [4], {4: node()["gpus"][4]}, POLICY)
        self.assertTrue(wc.worker_capacity_check(r5, 0, T4)["pass"])

    def test_fractions_unchanged(self):
        self.assertEqual([c["mem_fraction"] for c in CFG["cases"]], [0.86, 0.88])


def _gloo_agree(rank, store, oks, out):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2)
    try:
        Path(out, f"r{rank}").write_text(json.dumps(wc.all_ranks_agree(oks[rank], dist, None)))
    finally:
        dist.destroy_process_group()


class CapacityAgreementGloo(unittest.TestCase):
    def test_one_failing_rank_stops_both(self):
        try:
            import torch.multiprocessing as mp
        except ImportError:
            self.skipTest("torch not installed")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        for n, oks in enumerate(([True, True], [True, False])):
            d = tmp / f"o{n}"
            d.mkdir()
            mp.spawn(_gloo_agree, args=(str(tmp / f"s{n}"), oks, str(d)), nprocs=2, join=True)
            got = [json.loads((d / f"r{r}").read_text()) for r in range(2)]
            self.assertEqual(got, [all(oks)] * 2)


def _lease_race(lease_dir, who, out):
    lease, p = wc.acquire_lease(Path(lease_dir), [4, 5], {"window_dir": who})
    Path(out, who).write_text(json.dumps({"won": lease is not None, "problems": p}))


class HostLease(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.ld = self.tmp / "leases"

    def test_joint_acquire_release_and_no_reclaim(self):
        a, p = wc.acquire_lease(self.ld, [4, 5], {"window_dir": "A"})
        self.assertEqual(p, [])
        b, p = wc.acquire_lease(self.ld, [4, 5], {"window_dir": "B"})
        self.assertIsNone(b)
        self.assertIn("never reclaimed", p[0])
        c, p = wc.acquire_lease(self.ld, [4], {"window_dir": "C"})                # a subset is also refused
        self.assertIsNone(c)
        self.assertEqual(wc.release_lease(self.ld, dict(a, token="not-ours"), "x"),
                         ["gpu4 lease is not ours; untouched", "gpu5 lease is not ours; untouched"])
        self.assertEqual(sorted(x.name for x in self.ld.glob("gpu*.lease")), ["gpu4.lease", "gpu5.lease"])
        self.assertEqual(wc.lease_held_problems(self.ld, a), [])
        self.assertEqual(wc.release_lease(self.ld, a, "done"), [])
        self.assertEqual(list(self.ld.glob("gpu*.lease")), [])
        self.assertEqual(len(list((self.ld / "released").glob("*.reason"))), 2)
        self.assertTrue(wc.lease_held_problems(self.ld, a))

    def test_dead_owner_is_not_reclaimed(self):
        a, _ = wc.acquire_lease(self.ld, [4, 5], {"window_dir": "A"})
        lp = self.ld / "gpu4.lease"
        os.chmod(lp, 0o644)
        lp.write_text(json.dumps(dict(a, pid=2**22 + 12345, created_utc="2020-01-01T00:00:00+00:00")))   # dead, old
        self.assertIsNone(wc.acquire_lease(self.ld, [4, 5], {"window_dir": "B"})[0])

    def test_partial_existing_lease_leaves_nothing(self):
        self.ld.mkdir(mode=0o700)
        (self.ld / "gpu5.lease").write_text("{}")
        b, p = wc.acquire_lease(self.ld, [4, 5], {"window_dir": "B"})
        self.assertIsNone(b)
        self.assertFalse((self.ld / "gpu4.lease").exists())

    def test_not_owner_only_directory_refused(self):
        self.ld.mkdir(mode=0o755)
        os.chmod(self.ld, 0o755)
        self.assertIsNone(wc.acquire_lease(self.ld, [4, 5], {"window_dir": "A"})[0])

    def test_simultaneous_processes_one_winner(self):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        out = self.tmp / "out"
        out.mkdir()
        procs = [ctx.Process(target=_lease_race, args=(str(self.ld), f"w{i}", str(out))) for i in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(30)
        won = [json.loads((out / f"w{i}").read_text())["won"] for i in range(4)]
        self.assertEqual(sum(won), 1)


class EnrollmentAuthorization(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def _auth(self, **over):
        from mgbench.launch_controls import candidate_digest
        cand = self.tmp / "cand"
        cand.mkdir(exist_ok=True)
        now = dt.datetime.now(dt.timezone.utc)
        a = {"id": "fixture", "scope": pc.PAIR_SCOPE, "candidate_root": str(cand), "candidate_digest": candidate_digest(cand),
             "gpu_indices": [4, 5], "gpu_uuids": {"4": U4, "5": None}, "case_gpus": pc.CASE_GPUS, "max_gpus": 2,
             "max_concurrent_jobs": 1, "max_cases": 2, "issued_utc": (now - dt.timedelta(seconds=5)).isoformat(),
             "expires_utc": (now + dt.timedelta(seconds=1500)).isoformat(), "nonrenewable": True,
             "enrollment": {"indices": [5], "sequence": pc.ENROLL_SEQUENCE, "policy_sha256": "p" * 64},
             "label": "SYNTHETIC CPU TEST FIXTURE"}
        a.update(over)
        p = self.tmp / f"auth{len(list(self.tmp.glob('auth*')))}.json"
        p.write_text(json.dumps(a))
        return p, cand

    def test_enrollment_must_be_covered_by_the_frozen_policy(self):
        p, cand = self._auth()
        kw = dict(candidate_root=cand, uuids={4: U4, 5: None})
        self.assertEqual(pc.pair_authorization_problems(p, enrollment_policy_sha256="p" * 64, **kw), [])
        self.assertTrue(pc.pair_authorization_problems(p, enrollment_policy_sha256="q" * 64, **kw))
        self.assertTrue(pc.pair_authorization_problems(p, enrollment_policy_sha256=None, **kw))
        for e in (None, {"indices": [5]}, {"indices": [5, 6], "sequence": pc.ENROLL_SEQUENCE, "policy_sha256": "p" * 64},
                  {"indices": [5], "sequence": ["enroll_gpu5", "C5_fp32_world2_ga2"], "policy_sha256": "p" * 64},
                  {"runtime_config": "any file"}):
            p2, _ = self._auth(enrollment=e)
            self.assertTrue(pc.pair_authorization_problems(p2, enrollment_policy_sha256="p" * 64, **kw), e)
        p3, _ = self._auth(gpu_uuids={"4": None, "5": None})
        self.assertTrue(pc.pair_authorization_problems(p3, candidate_root=cand, uuids={4: None, 5: None},
                                                       enrollment_policy_sha256="p" * 64))

    def test_pair_config_contract(self):
        feas = json.loads(Path(ROOT.parents[1] / "36_c16_design/pair_memory_feasibility.json").read_text()) \
            if (ROOT.parents[1] / "36_c16_design/pair_memory_feasibility.json").exists() else None
        if feas is None:
            self.skipTest("design evidence not present beside the candidate")
        self.assertEqual(pc.pair_config_problems(CFG, feas), [])
        self.assertTrue(pc.pair_config_problems(dict(CFG, gpu_uuids={4: U4, 5: U5}), feas))
        self.assertTrue(pc.pair_config_problems(dict(CFG, timing=dict(CFG["timing"], window_s=1100)), feas))


if __name__ == "__main__":
    unittest.main()
