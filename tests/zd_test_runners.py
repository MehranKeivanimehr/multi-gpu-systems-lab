"""Acceptance for the reference and streaming executing runners.

Both runners are GPU-capable, so every execution here happens inside the
verified device-isolated sandbox (see `_isolation.py`); without it the tests
are deferred rather than run unprotected. Inside the sandbox each runner gets
through its CPU-side work — affinity, monitor, protocol and corpus
verification, tokenizer checks where the environment allows — and then refuses
at its own device guard. That exercises the production parser, the runner
binding check, the real subprocess path, evidence writing and cleanup.

Passing these does not establish hardware validity.
"""
import json, os, pathlib, subprocess, sys, tempfile, time, unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = REPO / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from zerodiag import cli as CLI                    # noqa: E402
from zerodiag import matrix as MX                   # noqa: E402
from zerodiag import reference as RF                # noqa: E402
from zerodiag import streaming as ST                # noqa: E402
import _zd_isolation as ISO                         # noqa: E402

def _in_tree(rel: str, root=None):
    """Locate a file in either tree layout (candidate or published study).

    The candidate keeps `protocols/` and `out/` beside the package; the
    published repository keeps `protocols/` inside the study directory. Trying
    both removes the need for a build-time path rewrite, which is one less
    thing that can silently fail to match.
    """
    base = ROOT if root is None else root
    for b in (base, base.parent):
        c = b / rel
        if c.exists():
            return c
    return base / rel


CORPUS = _in_tree("protocols/prompt_corpus.json")


def fake_uuid(i):
    return f"GPU-{i:08d}-0000-0000-0000-000000000000"


SYNTH = {
    "source": "SYNTHETIC test fixture, not a real device inventory",
    "source_sha256": "0" * 64,
    "groups": {
        "p1080_1": {"device_indices": [901], "device_uuids": [fake_uuid(901)],
                    "device_names": ["SYNTHETIC GTX 1080 Ti"],
                    "family": "GTX 1080 Ti", "topology_worst_link": "N/A"},
        "p1080_2": {"device_indices": [901, 902],
                    "device_uuids": [fake_uuid(901), fake_uuid(902)],
                    "device_names": ["SYNTHETIC GTX 1080 Ti"] * 2,
                    "family": "GTX 1080 Ti", "topology_worst_link": "PIX"},
        "p1080_4": {"device_indices": [901, 902, 903, 904],
                    "device_uuids": [fake_uuid(i) for i in (901, 902, 903, 904)],
                    "device_names": ["SYNTHETIC GTX 1080 Ti"] * 4,
                    "family": "GTX 1080 Ti", "topology_worst_link": "PIX"},
        "p1080_6": {"device_indices": [901, 902, 903, 904, 905, 906],
                    "device_uuids": [fake_uuid(i) for i in range(901, 907)],
                    "device_names": ["SYNTHETIC GTX 1080 Ti"] * 6,
                    "family": "GTX 1080 Ti", "topology_worst_link": "PHB"},
        "rtx_1": {"device_indices": [911], "device_uuids": [fake_uuid(911)],
                  "device_names": ["SYNTHETIC TITAN RTX"],
                  "family": "TITAN RTX", "topology_worst_link": "N/A"},
        "rtx_2": {"device_indices": [911, 912],
                  "device_uuids": [fake_uuid(911), fake_uuid(912)],
                  "device_names": ["SYNTHETIC TITAN RTX"] * 2,
                  "family": "TITAN RTX", "topology_worst_link": "PHB"},
    },
}


class RunnerSourceContracts(unittest.TestCase):
    """Checks that need no execution at all."""

    def test_both_runners_are_bound_by_the_source_digest(self):
        b = CLI.source_tree_digest()
        for rel in ("zerodiag/ref_runner.py", "zerodiag/stream_runner.py"):
            self.assertIn(rel, b)
            self.assertTrue(b[rel])

    def test_the_launcher_selects_a_runner_per_kind(self):
        src = (ROOT / "zerodiag/cli.py").read_text()
        self.assertIn('"training": "zerodiag/worker.py"', src)
        self.assertIn('"reference": "zerodiag/ref_runner.py"', src)
        self.assertIn('"streaming": "zerodiag/stream_runner.py"', src)

    def test_a_chunk_is_never_called_a_token_in_the_runner(self):
        src = (ROOT / "zerodiag/stream_runner.py").read_text()
        self.assertIn("a CHUNK of text, not a token", src)
        self.assertIn("token_attribution", src)
        # token_ids are recorded only if the server actually supplies them
        self.assertIn("Only record token_ids if the server actually supplies", src)

    def test_the_streaming_runner_binds_one_fresh_loopback_server(self):
        # Assert the argv the production builder actually produces, not a
        # string in the file: the flag could be present in a comment.
        from zerodiag import stream_runner as SR
        argv = SR.build_server_argv("/py", model="M", revision="R",
                                    host="127.0.0.1", port=41234)
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--port") + 1], "41234")
        src = (ROOT / "zerodiag/stream_runner.py").read_text()
        self.assertIn('"fresh_for_this_session": True', src)
        # The listening address is MEASURED, never asserted: the installed
        # server binds all interfaces regardless of --host.
        self.assertIn("listening_bind_addresses", src)
        self.assertNotIn('"loopback_only": True', src)
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        try:
            rec = SR.listening_bind_addresses(s.getsockname()[1])
            self.assertEqual(rec["addresses"], ["127.0.0.1"])
            self.assertTrue(rec["loopback_only"])
        finally:
            s.close()

    def test_the_reference_runner_refuses_to_claim_dropout_matching(self):
        src = (ROOT / "zerodiag/ref_runner.py").read_text()
        self.assertIn("no claim is made about", src)
        self.assertIn("dropout", src)
        self.assertIn("does_not_establish", src)

    def test_reference_is_not_described_as_an_identical_recipe(self):
        c = MX.reference_cases()[0]
        self.assertIn("loss agreement is not gradient correctness",
                      c["does_not_establish"])
        self.assertIn("INFORMED", c["historical_status"])
        # the training recipe is FP32 GPT-2; the reference is a different model
        self.assertNotEqual(c["recipe"], "zero1")
        self.assertIn("hf_generate", c["recipe"])


@unittest.skipUnless(ISO.available(), ISO.skip_reason())
class RunnerExecution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.synth_dir = pathlib.Path(tempfile.mkdtemp(prefix="runsynth_"))
        cls.synth = cls.synth_dir / "device_groups_frozen.json"
        cls.synth.write_text(json.dumps(SYNTH))

    def _run(self, case_id, *, interpreter=None, extra=()):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="runlife_"))
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["ZERODIAG_GROUPS_PATH"] = str(self.synth)
        for v in ("CUDA_VISIBLE_DEVICES", "TITAN_CPU_ONLY_GUARD"):
            env.pop(v, None)
        probe = subprocess.run(
            [sys.executable, "-B", "-c",
             "import json,sys;from zerodiag import matrix as MX;"
             "print(json.dumps({c['case_id']:c for c in "
             "MX.full_matrix()['remaining_cases']}[sys.argv[1]]))", case_id],
            cwd=ROOT, capture_output=True, text=True, timeout=180, env=env)
        self.assertEqual(probe.returncode, 0, probe.stderr[-400:])
        case = json.loads(probe.stdout)
        for u in case["device_uuids"]:
            self.assertTrue(u.startswith("GPU-0000"),
                            f"SAFETY: case names a non-synthetic device {u!r}")
        sb = json.loads(subprocess.run(
            [sys.executable, "-B", "-c",
             "import json;from zerodiag.cli import source_tree_digest as D;"
             "print(json.dumps(D()))"], cwd=ROOT, capture_output=True,
            text=True, timeout=180, env=env).stdout)
        now = time.time()
        rec = {"authorization_id": f"TEST-RUN-{case_id}",
               "approved_by": "cpu fixture",
               "approval_text": "synthetic devices; no real device is named",
               "start_epoch": now - 5, "expiry_epoch": now + 1200,
               "case_id": case_id, "world_size": case["world_size"],
               "device_indices": list(case["device_indices"]),
               "device_uuids": list(case["device_uuids"]),
               "device_names": list(case["device_names"]),
               "source_digest": sb["__tree__"]}
        rec["digest"] = CLI._receipt_digest(rec)
        rp = tmp / "receipt.json"; rp.write_text(json.dumps(rec))
        out = tmp / "run"
        argv = [sys.executable, "-B", "-m", "zerodiag.cli", "--execute-case",
                "--case-id", case_id, "--output", str(out),
                "--receipt", str(rp), "--corpus", str(CORPUS),
                "--max-cpus", "4", "--memory-budget-gib", "8.0",
                "--monitor-interval-s", "0.5", *extra]
        if interpreter:
            argv += ["--interpreter", interpreter]
        p = subprocess.run(
            ISO.wrap(argv, extra_env={"ZERODIAG_GROUPS_PATH": str(self.synth)},
                     writable=[tmp, self.synth_dir]),
            cwd=ROOT, capture_output=True, text=True, timeout=1800, env=env)
        s = out / "case_summary.json"
        return p, (json.loads(s.read_text()) if s.is_file() else None), out

    def test_reference_runner_reaches_its_device_guard_and_refuses(self):
        p, summ, out = self._run("R_rtx_1_tinyllama_fp32")
        self.assertIsNotNone(summ, p.stdout[-900:] + p.stderr[-500:])
        self.assertEqual(summ["kind"], "reference")
        self.assertEqual(summ["runner"], "zerodiag/ref_runner.py")
        self.assertEqual(summ["ranks_expected"], 1)
        self.assertEqual(summ["ranks_with_evidence"], 1)
        self.assertEqual(summ["exit_codes"], [90])
        self.assertTrue(summ["group_absent_confirmed"])
        ev = json.loads((out / "rank0" / "reference_evidence.json").read_text())
        self.assertEqual(ev["stage"], "refused")
        self.assertIn("no CUDA device visible", ev["reason"])
        # the CPU-side work really ran before the guard
        self.assertTrue(ev["corpus_check"]["ok"])
        self.assertEqual(ev["protocol_digest"], RF.protocol_digest())
        self.assertEqual(ev["containment"]["affinity"]["n_permitted"], 4)
        self.assertEqual(ev["child_visible_devices"], fake_uuid(911))

    def test_streaming_runner_reaches_its_device_guard_and_refuses(self):
        p, summ, out = self._run("S_rtx_1_stream_256_64")
        self.assertIsNotNone(summ, p.stdout[-900:] + p.stderr[-500:])
        self.assertEqual(summ["kind"], "streaming")
        self.assertEqual(summ["runner"], "zerodiag/stream_runner.py")
        self.assertEqual(summ["exit_codes"], [90])
        self.assertTrue(summ["group_absent_confirmed"])
        ev = json.loads((out / "rank0" / "streaming_evidence.json").read_text())
        self.assertEqual(ev["stage"], "refused")
        self.assertIn("no CUDA device visible", ev["reason"])
        self.assertEqual(ev["protocol_digest"], ST.protocol_digest())
        self.assertEqual(ev["protocol"]["prompt_len_tokens"], 256)
        self.assertEqual(ev["protocol"]["output_tokens"], 64)
        self.assertEqual(ev["protocol"]["sessions"], 1)

    def test_a_non_matching_corpus_refuses_before_any_device_work(self):
        bad = self.synth_dir / "bad_corpus.json"
        bad.write_text(json.dumps({"prompts": ["x"], "n_prompts": 1,
                                   "corpus_sha256": "0" * 64}))
        p, summ, out = self._run("R_rtx_1_tinyllama_fp32",
                                 extra=("--corpus", str(bad)))
        # the launcher passed our corpus; the runner must reject it
        ev = json.loads((out / "rank0" / "reference_evidence.json").read_text())
        self.assertEqual(ev["stage"], "refused")
        self.assertFalse(ev["corpus_check"]["ok"])
        self.assertIn("frozen digest", ev["corpus_check"]["reason"])
        self.assertEqual(summ["exit_codes"], [90])


if __name__ == "__main__":
    unittest.main(verbosity=2)
