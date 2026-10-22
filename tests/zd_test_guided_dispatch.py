"""The repair must reach the REAL request path, not just a command string.

Everything here is asserted against the installed vLLM: its own argument
parser resolves the production server argv, its own request model consumes the
production payload, and its own
`build_guided_decoding_logits_processor_async` runs with a real tokenizer. The
probe runs inside the device-denying sandbox under the vLLM interpreter.

Two controls keep the test from being vacuous:

* the same call with `default_guided_backend="outlines"` must still raise the
  dependency ImportError -- if it stopped raising, this test would be passing
  for the wrong reason;
* a `guided_grammar` must still select Outlines even with the
  lm-format-enforcer backend, which is why the frozen request is asserted to
  carry no grammar.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = REPO / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _zd_isolation as _isolation                             # noqa: E402
import _zd_vllm_env as _vllm_env                               # noqa: E402
from zerodiag import stream_runner as SR            # noqa: E402
from zerodiag import streaming as ST                # noqa: E402

VLLM_PY = _vllm_env.VLLM_PYTHON
def _find(rel: str) -> str:
    """Locate a protocol file in either tree layout, without a build rule."""
    for base in (ROOT, ROOT.parent):
        c = base / rel
        if c.is_file():
            return str(c)
    return str(ROOT / rel)


CORPUS = os.environ.get("ZERODIAG_CORPUS",
                        _find("protocols/prompt_corpus.json"))
PROBE = pathlib.Path(__file__).resolve().parent / "probes/guided_dispatch_probe.py"
# The probe result is cached for the record, but NEVER inside the repository:
# a test that writes into the tree it is testing pollutes a published artifact.
# The on-node runner copies it into `out/regress/` explicitly.
CACHE = pathlib.Path(os.environ.get(
    "ZD_PROBE_CACHE",
    str(pathlib.Path(_isolation.SANDBOX_TMP) / "guided_dispatch_probe.json")))


def run_probe() -> dict:
    # The sandbox mounts a fresh tmpfs over /tmp, so a repository checked out
    # under /tmp would be invisible inside it. Stage the probe and the corpus
    # in the sandbox's own writable directory, and bind the repository root in,
    # so this contract works wherever the repository happens to live.
    stage = pathlib.Path(_isolation.SANDBOX_TMP) / "guided_dispatch"
    stage.mkdir(parents=True, exist_ok=True)
    probe = stage / PROBE.name
    probe.write_text(PROBE.read_text())
    corpus = stage / "prompt_corpus.json"
    corpus.write_text(pathlib.Path(CORPUS).read_text())
    argv = _isolation.wrap(
        [VLLM_PY, "-I", str(probe), str(ROOT), str(corpus)],
        writable=[str(_isolation.SANDBOX_TMP), str(ROOT)],
        extra_env=_vllm_env.OFFLINE_ENV)
    r = subprocess.run(argv, capture_output=True, text=True, timeout=2400)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise AssertionError(f"probe produced no JSON (rc={r.returncode})\n"
                             f"{r.stdout[-1200:]}\n{r.stderr[-1500:]}")
    got = json.loads(lines[-1])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(got, indent=2, sort_keys=True, default=str))
    return got


class RealGuidedDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _isolation.available():
            raise AssertionError(
                "device-denying isolation is unavailable, so this probe "
                "cannot run safely: " + str(_isolation.probe().get("reason")))
        _vllm_env.require()
        cls.d = run_probe()

    # -- the installed versions this was verified against ---------------
    def test_the_installed_versions_are_the_ones_verified(self):
        v = self.d["versions"]
        self.assertEqual(v["vllm"], "0.6.3.post1")
        self.assertEqual(v["lm-format-enforcer"], "0.10.6")
        self.assertEqual(v["prometheus_client"], "0.26.0")

    # -- 1. the server receives the intended configuration --------------
    def test_vllm_own_parser_resolves_the_backend_from_production_argv(self):
        a = self.d["server_args"]
        self.assertEqual(a["namespace_backend"], "lm-format-enforcer")
        self.assertEqual(a["engine_args_backend"], "lm-format-enforcer")
        self.assertEqual(self.d["decoding_config"]["configured_backend"],
                         "lm-format-enforcer")
        self.assertEqual(self.d["decoding_config"]["class_default_backend"],
                         "outlines",
                         "if the installed default were already "
                         "lm-format-enforcer the override would be doing "
                         "nothing and this test would be vacuous")
        self.assertEqual(self.d["production_server_argv"],
                         SR.build_server_argv(
                             self.d["production_server_argv"][0],
                             model=ST.PROTOCOL["model"],
                             revision=ST.PROTOCOL["model_revision"],
                             host="127.0.0.1", port=12345),
                         "the probe must parse the production argv itself")

    def test_the_frozen_study_parameters_survive_argument_parsing(self):
        a = self.d["server_args"]
        self.assertEqual(a["model"], ST.PROTOCOL["model"])
        self.assertEqual(a["revision"], ST.PROTOCOL["model_revision"])
        self.assertEqual(a["tokenizer_revision"], ST.PROTOCOL["model_revision"])
        self.assertEqual(a["dtype"], "float16")
        self.assertEqual(a["max_model_len"], 1024)
        self.assertEqual(a["host"], "127.0.0.1")

    # -- 2. the frozen request carries no constraint --------------------
    def test_the_production_payload_carries_no_constraint_field(self):
        self.assertEqual(self.d["payload_constraint_fields_present"], [])
        self.assertEqual(sorted(self.d["payload_keys"]),
                         ["max_tokens", "model", "prompt", "stream",
                          "stream_options", "temperature"])
        self.assertEqual(self.d["request"]["constraint_fields_set_on_request"],
                         {})
        self.assertIsNone(self.d["request"]["guided_decoding_backend_field"])

    def test_no_grammar_json_regex_or_choice_reaches_sampling_params(self):
        sp = self.d["sampling_params"]
        self.assertFalse(sp["guided_decoding_is_none"],
                         "vLLM 0.6.3.post1 always attaches a "
                         "GuidedDecodingParams; if that changed, the whole "
                         "repair would be unnecessary and this test must be "
                         "revisited")
        for k in ("guided_json", "guided_regex", "guided_choice",
                  "guided_grammar", "guided_json_object"):
            self.assertIsNone(sp[k], k)
        self.assertIsNone(sp["guided_backend_on_request"])

    def test_the_input_is_the_frozen_token_ids(self):
        self.assertTrue(self.d["payload_prompt_is_ids"])
        self.assertEqual(self.d["payload_prompt_len"],
                         ST.PROTOCOL["prompt_len_tokens"])
        self.assertEqual(self.d["request"]["prompt_len"],
                         ST.PROTOCOL["prompt_len_tokens"])

    # -- 3. the real dispatch does not take the failing path ------------
    def test_the_real_builder_returns_no_logits_processor(self):
        r = self.d["real_dispatch"]
        self.assertEqual(r["default_backend_supplied"], "lm-format-enforcer")
        self.assertFalse(r["branch_would_select_outlines"])
        self.assertTrue(r["resolved_processor_is_none"])
        self.assertIsNone(r["logits_processors_after"])
        self.assertIsNone(r["guided_decoding_after"])

    def test_the_outlines_import_chain_is_never_entered(self):
        self.assertFalse(self.d["outlines_in_sys_modules_before"])
        self.assertFalse(self.d["outlines_in_sys_modules_after"],
                         "the broken import chain was entered after all")
        self.assertFalse(self.d["datasets_in_sys_modules_after"])

    def test_generation_parameters_and_tokenizer_are_unchanged(self):
        r, sp = self.d["real_dispatch"], self.d["sampling_params"]
        self.assertEqual(sp["temperature"], 0.0)
        self.assertEqual(r["temperature_after"], 0.0)
        self.assertEqual(sp["max_tokens"], ST.PROTOCOL["output_tokens"])
        self.assertEqual(r["max_tokens_after"], ST.PROTOCOL["output_tokens"])
        self.assertEqual(sp["n"], 1)
        self.assertIn(sp["best_of"], (None, 1))
        self.assertEqual(sp["top_k"], -1)
        self.assertEqual(sp["top_p"], 1.0)
        self.assertEqual(r["tokenizer_vocab_size"], 32000)
        self.assertIn("Llama", r["tokenizer_class"])

    # -- the controls that stop this test being vacuous -----------------
    def test_the_outlines_backend_still_fails_here(self):
        c = self.d["outlines_backend_control"]
        self.assertIsInstance(c, str)
        self.assertIn("ImportError", c,
                      "if the outlines path no longer fails, the backend "
                      "selection is no longer what makes the session work, "
                      "and the finding must be re-stated")

    def test_a_grammar_would_select_outlines_regardless_of_the_backend(self):
        g = self.d["grammar_control"]
        self.assertEqual(g["backend"], "lm-format-enforcer")
        self.assertTrue(g["branch_selects_outlines"],
                        "the installed dispatch is "
                        "`backend == 'outlines' or guided_params.grammar`, so "
                        "a grammar overrides the backend; this is why the "
                        "frozen request must carry none")

    def test_the_asserted_branch_is_the_installed_branch(self):
        lines = self.d["dispatch_source"]["branch_lines"]
        self.assertIn("if guided_params.backend == 'outlines' or "
                      "guided_params.grammar:", lines)
        self.assertIn("if guided_params.backend == 'lm-format-enforcer':",
                      lines)

    # -- honest package health ------------------------------------------
    def test_the_dependency_conflict_is_reported_not_hidden(self):
        pc = self.d["pip_check"]
        self.assertNotEqual(
            pc["returncode"], 0,
            "if `pip check` is now clean, the environment conflict has been "
            "resolved by something and the report must say so instead")
        joined = " ".join(pc["stdout"])
        self.assertIn("datasets", joined)
        self.assertIn("huggingface-hub", joined)

    def test_the_installed_server_binds_all_interfaces_despite_host(self):
        """`--host` does not restrict the listener in this version."""
        lines = self.d["bind_source"]["lines"]
        joined = " | ".join(f"{n}:{t}" for n, t in lines)
        self.assertIn("socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
                      joined)
        self.assertTrue(any('.bind(("", args.port))' in t for _, t in lines),
                        f"expected an empty-host bind; saw {joined}")
        src = (ROOT / "zerodiag/stream_runner.py").read_text()
        self.assertIn("listening_bind_addresses", src,
                      "the runner must MEASURE the bind address")
        self.assertNotIn('"loopback_only": True', src,
                         "an unmeasured loopback claim must not return")

    def test_selecting_a_backend_is_not_called_a_dependency_repair(self):
        src = (ROOT / "zerodiag/stream_runner.py").read_text()
        self.assertIn("NOT a repair of the dependency conflict", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
