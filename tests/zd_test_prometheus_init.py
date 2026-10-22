"""Reproduce the streaming startup failure and its repair, on CPU, isolated.

The historical streaming attempt died during engine startup inside
prometheus_client. This test does not mock that away: it drives the INSTALLED
prometheus_client 0.26.0 in a child process, inside the device-denying
sandbox, under the two directory lifecycles that matter.

* `reclaimed` reproduces what vLLM 0.6.3.post1 does to itself at
  `api_server.py:154`: the multiprocess directory is a module-global
  `tempfile.TemporaryDirectory()`, so it can be removed while startup is still
  in progress. The installed client then raises on the missing path.
* `fresh_private` is the repair: a directory this project owns for the
  attempt, created and exported before the server process starts, which is
  exactly the case `api_server.py:149` defers to.

The child needs `prometheus_client`, which lives in the vLLM environment. If
that interpreter is not present the test FAILS: a skipped repair check is not
evidence that the repair works.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = REPO / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _zd_isolation as _isolation                            # noqa: E402
import _zd_vllm_env as _vllm_env                              # noqa: E402
from zerodiag import stream_runner as SR          # noqa: E402

VLLM_PY = _vllm_env.VLLM_PYTHON

CHILD = r'''
import json, os, pathlib, shutil, sys
mode, d = sys.argv[1], sys.argv[2]
os.environ["PROMETHEUS_MULTIPROC_DIR"] = d          # set BEFORE the import
if mode == "reclaimed":
    shutil.rmtree(d)                                # the TemporaryDirectory
out = {"mode": mode, "dir_exists_at_import": pathlib.Path(d).is_dir()}
try:
    import prometheus_client
    from prometheus_client import CollectorRegistry, Gauge, multiprocess
    import prometheus_client.values as V
    out["prometheus_client_file"] = V.__file__
    out["value_class"] = V.ValueClass.__name__
    reg = CollectorRegistry()
    g = Gauge("titan_startup_probe", "startup probe", registry=reg,
              multiprocess_mode="mostrecent")
    g.set(1.0)
    out["db_files"] = sorted(p.name for p in pathlib.Path(d).glob("*.db"))
    agg = CollectorRegistry()
    multiprocess.MultiProcessCollector(agg)
    out["collected"] = sorted({m.name for m in agg.collect()})
    out["ok"] = True
except BaseException as exc:
    out["ok"] = False
    out["error"] = type(exc).__name__
    out["message"] = str(exc)[:300]
print(json.dumps(out))
'''


def run_child(mode: str, d: pathlib.Path) -> dict:
    script = pathlib.Path(_isolation.SANDBOX_TMP) / "prom_child.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(CHILD)
    argv = _isolation.wrap([VLLM_PY, "-I", str(script), mode, str(d)],
                           writable=[str(d.parent), str(script.parent)])
    r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        raise AssertionError(f"child produced no JSON\n"
                             f"rc={r.returncode}\nout={r.stdout[-800:]}\n"
                             f"err={r.stderr[-800:]}")
    return json.loads(lines[-1])


class PrometheusStartup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _isolation.available():
            raise AssertionError(
                "device-denying isolation is unavailable, so this CPU "
                "reproduction cannot run safely: "
                + str(_isolation.probe().get("reason")))
        _vllm_env.require()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            dir=str(_isolation.SANDBOX_TMP))
        self.base = pathlib.Path(self.tmp.name) / "prometheus_multiproc"

    def tearDown(self):
        self.tmp.cleanup()

    def test_reclaimed_directory_reproduces_the_startup_failure(self):
        SR.prepare_metrics_dir(self.base)
        got = run_child("reclaimed", self.base)
        self.assertFalse(got["dir_exists_at_import"])
        self.assertFalse(got["ok"], got)
        self.assertIn(got["error"],
                      ("FileNotFoundError", "ValueError", "NotADirectoryError"),
                      got)

    def test_fresh_private_directory_lets_initialization_complete(self):
        rec = SR.prepare_metrics_dir(self.base)
        got = run_child("fresh_private", self.base)
        self.assertTrue(got["dir_exists_at_import"])
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["value_class"], "MmapedValue",
                         "the installed client must be in multiprocess mode")
        self.assertTrue(got["db_files"], got)
        self.assertTrue(all(f.endswith(".db") for f in got["db_files"]))
        self.assertIn("titan_startup_probe", got["collected"])
        # the directory this project owns is the one that was written to
        out = SR.metrics_dir_outcome(rec)
        self.assertTrue(out["used_by_server"])
        self.assertEqual(out["db_files_written"], got["db_files"])

    def test_the_repair_did_not_touch_anything_shared(self):
        rec = SR.prepare_metrics_dir(self.base)
        self.assertTrue(str(self.base).startswith(str(self.tmp.name)))
        self.assertFalse(rec["shared_metrics_files_deleted"])
        self.assertNotIn("PROMETHEUS_MULTIPROC_DIR", os.environ,
                         "the parent process environment is left alone")

    def test_installed_versions_are_the_ones_diagnosed(self):
        r = subprocess.run(
            [VLLM_PY, "-I", "-c",
             "import importlib.metadata as m, json;"
             "print(json.dumps({k: m.version(k) for k in "
             "('vllm','prometheus_client')}))"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        v = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(v["vllm"], "0.6.3.post1")
        self.assertEqual(v["prometheus_client"], "0.26.0")

    def test_installed_vllm_defers_to_a_preset_directory(self):
        # Locate the installed file WITHOUT importing vllm: importing it
        # enumerates CUDA devices, and no device is contacted during CPU
        # preparation. `sysconfig` imports nothing heavy.
        r = subprocess.run(
            [VLLM_PY, "-I", "-c",
             "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        site = pathlib.Path(r.stdout.strip().splitlines()[-1])
        src = site / "vllm/entrypoints/openai/api_server.py"
        self.assertTrue(src.is_file(), src)
        text = src.read_text().splitlines()
        window = "\n".join(text[140:170])
        self.assertIn('if "PROMETHEUS_MULTIPROC_DIR" not in os.environ', window)
        self.assertIn("prometheus_multiproc_dir = tempfile.TemporaryDirectory()",
                      window)
        self.assertIn("Found PROMETHEUS_MULTIPROC_DIR was set by user", window)


class GuidedDecodingImportChain(unittest.TestCase):
    """The second streaming defect: every request imported a broken `outlines`.

    Reproduced against the installed sources on CPU, inside the device-denying
    sandbox. The repair selects the installed backend; it does not patch a
    third-party library and does not change the generation policy.
    """

    PROBE = r'''
import json
out = {}
try:
    from vllm.entrypoints.openai.protocol import CompletionRequest
    r = CompletionRequest(model="m", prompt=[1, 2, 3], max_tokens=64,
                          temperature=0.0, stream=True)
    sp = r.to_sampling_params(default_max_tokens=64)
    gd = sp.guided_decoding
    out["guided_decoding_is_none"] = gd is None
    out["any_guide_set"] = bool(gd and (gd.json or gd.regex or gd.choice
                                        or gd.grammar or gd.json_object))
    out["temperature"] = sp.temperature
    out["max_tokens"] = sp.max_tokens
except BaseException as exc:
    out["request_probe"] = f"{type(exc).__name__}: {exc}"[:200]
for name, mod in (("outlines", "outlines"),
                  ("lmfe", "vllm.model_executor.guided_decoding."
                           "lm_format_enforcer_decoding")):
    try:
        __import__(mod)
        out[name] = "ok"
    except BaseException as exc:
        out[name] = f"{type(exc).__name__}: {exc}"[:200]
print(json.dumps(out))
'''

    @classmethod
    def setUpClass(cls):
        PrometheusStartup.setUpClass.__func__(cls)
        script = pathlib.Path(_isolation.SANDBOX_TMP) / "guided_probe.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(cls.PROBE)
        argv = _isolation.wrap(
            [VLLM_PY, "-I", str(script)], writable=[str(script.parent)],
            extra_env=_vllm_env.OFFLINE_ENV)
        r = subprocess.run(argv, capture_output=True, text=True, timeout=900)
        lines = [ln for ln in r.stdout.strip().splitlines()
                 if ln.startswith("{")]
        if not lines:
            raise AssertionError(f"probe produced no JSON: rc={r.returncode} "
                                 f"{r.stderr[-600:]}")
        cls.got = json.loads(lines[-1])

    def test_every_request_carries_guided_decoding_params(self):
        self.assertFalse(self.got["guided_decoding_is_none"],
                         "if this becomes None the repair is unnecessary")
        self.assertFalse(self.got["any_guide_set"],
                         "no guide is requested by this protocol")

    def test_the_default_backend_import_is_broken_here(self):
        self.assertNotEqual(self.got["outlines"], "ok", self.got["outlines"])
        self.assertIn("ImportError", self.got["outlines"])

    def test_the_selected_backend_imports(self):
        self.assertEqual(self.got["lmfe"], "ok", self.got["lmfe"])

    def test_the_generation_policy_is_untouched(self):
        self.assertEqual(self.got["temperature"], 0.0)
        self.assertEqual(self.got["max_tokens"], 64)

    def test_the_runner_asks_for_the_installed_backend(self):
        from zerodiag import stream_runner as SR
        argv = SR.build_server_argv("/py", model="M", revision="R",
                                    host="127.0.0.1", port=1)
        self.assertEqual(argv[argv.index("--guided-decoding-backend") + 1],
                         "lm-format-enforcer",
                         "the production argv itself must carry the backend")
        text = (ROOT / "zerodiag/stream_runner.py").read_text()
        self.assertIn("lm_format_enforcer_decoding.py:44", text)
        self.assertNotIn("huggingface_hub.errors.BucketNotFoundError =", text,
                         "the repair must not fabricate third-party API")


if __name__ == "__main__":
    unittest.main(verbosity=2)
