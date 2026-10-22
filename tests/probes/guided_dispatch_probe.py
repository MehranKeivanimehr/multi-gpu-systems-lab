"""Exercise the REAL vLLM request path for the frozen streaming request.

Runs inside the device-denying sandbox, under the vLLM interpreter. Prints one
JSON object. It does not check whether a flag appears in a string: it parses
the production server argv with vLLM's own parser, builds the production
request payload into vLLM's own request model, and calls vLLM's own
guided-decoding builder with the resulting sampling params and a real
tokenizer.

argv: <candidate_root> <corpus_path>
"""
import asyncio
import importlib.metadata as md
import json
import pathlib
import subprocess
import sys

out = {}
root, corpus_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)


def record(key, fn):
    try:
        out[key] = fn()
    except BaseException as exc:                      # noqa: BLE001
        out[key] = f"{type(exc).__name__}: {exc}"[:400]


# ---- 0. installed identities and package health --------------------------
def versions():
    v = {}
    for pkg in ("vllm", "prometheus_client", "outlines", "datasets",
                "huggingface_hub", "lm-format-enforcer", "transformers",
                "torch"):
        try:
            v[pkg] = md.version(pkg)
        except Exception as exc:                      # noqa: BLE001
            v[pkg] = f"?{type(exc).__name__}"
    return v


record("versions", versions)

# ---- 1. the production server argv, parsed by vLLM's own parser ----------
from zerodiag import stream_runner as SR            # noqa: E402
from zerodiag import streaming as ST                # noqa: E402

ARGV = SR.build_server_argv(sys.executable, model=ST.PROTOCOL["model"],
                            revision=ST.PROTOCOL["model_revision"],
                            host="127.0.0.1", port=12345)
out["production_server_argv"] = ARGV


def parse_server_args():
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils import FlexibleArgumentParser
    from vllm.engine.arg_utils import AsyncEngineArgs
    parser = make_arg_parser(FlexibleArgumentParser(description="probe"))
    # argv[0:3] are the interpreter, -m and the module name
    ns = parser.parse_args(ARGV[3:])
    ea = AsyncEngineArgs.from_cli_args(ns)
    return {"namespace_backend": ns.guided_decoding_backend,
            "engine_args_backend": ea.guided_decoding_backend,
            "model": ea.model, "revision": ea.revision,
            "tokenizer_revision": ea.tokenizer_revision,
            "dtype": str(ea.dtype), "max_model_len": ea.max_model_len,
            "gpu_memory_utilization": ea.gpu_memory_utilization,
            "disable_log_requests": getattr(ns, "disable_log_requests", None),
            "host": ns.host, "port": ns.port}


record("server_args", parse_server_args)


def decoding_config():
    from vllm.config import DecodingConfig
    eff = out["server_args"]["engine_args_backend"]
    cfg = DecodingConfig(guided_decoding_backend=eff)
    return {"configured_backend": cfg.guided_decoding_backend,
            "class_default_backend": DecodingConfig().guided_decoding_backend}


record("decoding_config", decoding_config)

# ---- 2. the production request payload, through vLLM's request model ------
PROMPT = json.loads(pathlib.Path(corpus_path).read_text())["prompts"][0]
PROMPT = PROMPT[:ST.PROTOCOL["prompt_len_tokens"]]
PAYLOAD = SR.build_request_payload(ST.PROTOCOL["model"], PROMPT,
                                   ST.PROTOCOL["output_tokens"])
out["payload_keys"] = sorted(PAYLOAD)
out["payload_prompt_len"] = len(PAYLOAD["prompt"])
out["payload_prompt_is_ids"] = all(isinstance(x, int) for x in PAYLOAD["prompt"])
out["payload_constraint_fields_present"] = [
    k for k in SR.CONSTRAINT_FIELDS if k in PAYLOAD]


def build_request():
    from vllm.entrypoints.openai.protocol import CompletionRequest
    r = CompletionRequest(**PAYLOAD)
    set_fields = {k: getattr(r, k, None) for k in SR.CONSTRAINT_FIELDS
                  if getattr(r, k, None) not in (None, [], {}, False)}
    return {"model": r.model, "max_tokens": r.max_tokens,
            "temperature": r.temperature, "stream": r.stream,
            "n": r.n, "best_of": r.best_of,
            "prompt_len": len(r.prompt),
            "constraint_fields_set_on_request": set_fields,
            "guided_decoding_backend_field": r.guided_decoding_backend}


record("request", build_request)


def sampling_params():
    from vllm.entrypoints.openai.protocol import CompletionRequest
    r = CompletionRequest(**PAYLOAD)
    sp = r.to_sampling_params(default_max_tokens=ST.PROTOCOL["output_tokens"])
    gd = sp.guided_decoding
    return {"temperature": sp.temperature, "max_tokens": sp.max_tokens,
            "n": sp.n, "best_of": sp.best_of, "top_p": sp.top_p,
            "top_k": sp.top_k, "seed": sp.seed,
            "logits_processors_before": (None if sp.logits_processors is None
                                         else len(sp.logits_processors)),
            "guided_decoding_is_none": gd is None,
            "guided_json": getattr(gd, "json", None),
            "guided_regex": getattr(gd, "regex", None),
            "guided_choice": getattr(gd, "choice", None),
            "guided_grammar": getattr(gd, "grammar", None),
            "guided_json_object": getattr(gd, "json_object", None),
            "guided_backend_on_request": getattr(gd, "backend", None)}


record("sampling_params", sampling_params)

# ---- 3. the REAL dispatch, with a real tokenizer -------------------------
out["outlines_in_sys_modules_before"] = "outlines" in sys.modules


def real_dispatch():
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.protocol import CompletionRequest
    from vllm.engine.async_llm_engine import (
        build_guided_decoding_logits_processor_async)
    tok = AutoTokenizer.from_pretrained(
        ST.PROTOCOL["model"], revision=ST.PROTOCOL["model_revision"],
        local_files_only=True)
    r = CompletionRequest(**PAYLOAD)
    sp = r.to_sampling_params(default_max_tokens=ST.PROTOCOL["output_tokens"])
    effective = out["decoding_config"]["configured_backend"]
    # exactly what the engine does: the request's backend or the server default
    before = getattr(sp.guided_decoding, "backend", None)
    res = asyncio.run(build_guided_decoding_logits_processor_async(
        sampling_params=sp, tokenizer=tok, default_guided_backend=effective))
    return {"tokenizer_class": type(tok).__name__,
            "tokenizer_vocab_size": int(tok.vocab_size),
            "request_backend_before": before,
            "default_backend_supplied": effective,
            "branch_would_select_outlines": (effective == "outlines"
                                             or bool(getattr(
                                                 sp.guided_decoding, "grammar",
                                                 None))),
            "logits_processors_after": (None if res.logits_processors is None
                                        else len(res.logits_processors)),
            "guided_decoding_after": res.guided_decoding,
            "resolved_processor_is_none": (res.logits_processors in (None, [])),
            "temperature_after": res.temperature,
            "max_tokens_after": res.max_tokens}


record("real_dispatch", real_dispatch)
out["outlines_in_sys_modules_after"] = "outlines" in sys.modules
out["datasets_in_sys_modules_after"] = "datasets" in sys.modules


# ---- 4. the failing path still fails, so the probe is discriminating ------
def outlines_backend_control():
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.protocol import CompletionRequest
    from vllm.engine.async_llm_engine import (
        build_guided_decoding_logits_processor_async)
    tok = AutoTokenizer.from_pretrained(
        ST.PROTOCOL["model"], revision=ST.PROTOCOL["model_revision"],
        local_files_only=True)
    r = CompletionRequest(**PAYLOAD)
    sp = r.to_sampling_params(default_max_tokens=ST.PROTOCOL["output_tokens"])
    try:
        asyncio.run(build_guided_decoding_logits_processor_async(
            sampling_params=sp, tokenizer=tok,
            default_guided_backend="outlines"))
        return "NO ERROR (the dependency conflict is gone)"
    except BaseException as exc:                       # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"[:200]


record("outlines_backend_control", outlines_backend_control)


def grammar_control():
    """A grammar selects Outlines regardless of the backend setting."""
    from vllm.sampling_params import GuidedDecodingParams
    gd = GuidedDecodingParams.from_optional(
        json=None, grammar="root ::= \"a\"", backend="lm-format-enforcer")
    return {"backend": gd.backend, "grammar": gd.grammar,
            "branch_selects_outlines": (gd.backend == "outlines"
                                        or bool(gd.grammar))}


record("grammar_control", grammar_control)


def dispatch_source():
    import vllm.model_executor.guided_decoding as g
    src = pathlib.Path(g.__file__).read_text()
    first = [ln.strip() for ln in src.splitlines()
             if ln.strip().startswith("if guided_params.backend")]
    return {"file": g.__file__, "branch_lines": first}


record("dispatch_source", dispatch_source)


def pip_check():
    r = subprocess.run([sys.executable, "-m", "pip", "check"],
                       capture_output=True, text=True, timeout=600)
    return {"returncode": r.returncode,
            "stdout": r.stdout.strip().splitlines()[:40],
            "stderr": r.stderr.strip()[-200:]}


def bind_source():
    """How the installed server actually creates its listening socket."""
    import vllm.entrypoints.openai.api_server as m
    src = pathlib.Path(m.__file__).read_text().splitlines()
    hits = [(i + 1, ln.strip()) for i, ln in enumerate(src)
            if "socket.socket(" in ln or ".bind((" in ln]
    return {"file": m.__file__, "lines": hits}


record("bind_source", bind_source)
record("pip_check", pip_check)
print(json.dumps(out, default=str))
