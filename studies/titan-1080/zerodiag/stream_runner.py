"""Executing runner for the prospective streaming session: 256 in / 64 out.

One fresh server for this session only, reached by this runner over loopback.
Its listening address is MEASURED at readiness rather than assumed: vLLM
0.6.3.post1 creates its own socket and binds `("", port)`
(`entrypoints/openai/api_server.py:544`), so `--host 127.0.0.1` does not make
the listener loopback-only. The server is started as a child of this runner and
inherits the job token, so the production launcher's ownership sweep reaches it;
it also leads its own session, so its whole process group can be signalled and
confirmed gone.

Naming discipline is enforced, not merely documented: a network chunk is not a
token. Token-level timing is emitted only when `streaming.token_attribution`
reports a genuine one-to-one timestamped token mapping; otherwise first-text and
chunk-level metrics are reported under those names.

Historical sessions are not rewritten and this one session is never combined
with them as if it were an identically controlled repeat.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from zerodiag import streaming as ST             # noqa: E402
from zerodiag.containment import apply_affinity, MonitorThread  # noqa: E402

MONITOR = None


def prepare_metrics_dir(base) -> dict:
    """Create a FRESH, private, per-attempt Prometheus multiprocess directory.

    Why this exists, from the installed sources rather than online examples:

    * `vllm/entrypoints/openai/api_server.py:149` -- when
      `PROMETHEUS_MULTIPROC_DIR` is NOT already in the environment, vLLM builds
      its own `tempfile.TemporaryDirectory()` and assigns it to the module
      global `prometheus_multiproc_dir` (:154). That global is rebound/collected
      while the engine is still starting, so the directory is removed before
      `prometheus_client/values.py:95` opens `<prefix>_<pid>.db` inside it, and
      startup dies on the missing path. That was the observed failure.
    * `api_server.py:157` shows the supported alternative: if the variable IS
      already set, vLLM defers to it and only warns that the directory "must be
      wiped between vLLM runs or you will find inaccurate metrics".
    * `prometheus_client/multiprocess.py:30` raises
      `ValueError('env PROMETHEUS_MULTIPROC_DIR is not set or not a directory')`,
      so the path must exist before the server starts.

    A fresh private directory per attempt satisfies both: the path survives for
    the whole attempt (we own it, not a garbage-collected temporary), and it is
    empty, so no earlier run's `.db` files are mixed in. Nothing shared is
    deleted and no unrelated check is disabled.
    """
    d = pathlib.Path(base)
    pre_existing = sorted(x.name for x in d.glob("*")) if d.is_dir() else []
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return {"dir": str(d), "created_fresh": True, "private_to_attempt": True,
            "pre_existing_entries_removed": pre_existing,
            "is_empty_at_start": not any(d.iterdir()),
            "exists_before_server_start": d.is_dir(),
            "shared_metrics_files_deleted": False,
            "env_var": "PROMETHEUS_MULTIPROC_DIR",
            "set_before_process_start": True,
            "installed_basis": [
                "vllm/entrypoints/openai/api_server.py:149 (skips its own "
                "TemporaryDirectory when the variable is already set)",
                "vllm/entrypoints/openai/api_server.py:154 (module-global "
                "TemporaryDirectory -- the reclaimed path, the original defect)",
                "prometheus_client/values.py:95 (opens <prefix>_<pid>.db in it)",
                "prometheus_client/multiprocess.py:30 (requires an existing "
                "directory)"]}


def metrics_dir_outcome(rec: dict) -> dict:
    """Post-attempt check: did the installed stack actually use our directory?"""
    d = pathlib.Path(rec["dir"])
    files = sorted(x.name for x in d.glob("*.db")) if d.is_dir() else []
    return {"db_files_written": files, "n_db_files": len(files),
            "directory_still_present": d.is_dir(),
            "used_by_server": bool(files)}


def listening_bind_addresses(port: int) -> dict:
    """The ACTUAL local address of the listening socket for `port`.

    vLLM 0.6.3.post1 creates its own listening socket at
    `entrypoints/openai/api_server.py:543-544` with
    `socket.socket(AF_INET, SOCK_STREAM)` and `sock.bind(("", args.port))`,
    then hands the file descriptor to uvicorn. An empty host means
    **0.0.0.0**, so `--host 127.0.0.1` does NOT restrict the listener in this
    version, and uvicorn logs `Uvicorn running on socket ('0.0.0.0', <port>)`.

    This function therefore MEASURES the bind address from `/proc/net/tcp`
    instead of asserting it. Claiming "loopback only" without measuring it was
    the defect this replaces.
    """
    rec = {"port": port, "listening": [], "loopback_only": None,
           "source": "/proc/net/tcp (and tcp6), state 0A = LISTEN",
           "installed_basis": ("vllm/entrypoints/openai/api_server.py:544 "
                               "binds ('', port), i.e. all interfaces, "
                               "regardless of --host")}
    for name, fam in (("tcp", "ipv4"), ("tcp6", "ipv6")):
        f = pathlib.Path("/proc/net") / name
        if not f.is_file():
            continue
        try:
            lines = f.read_text().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            cols = ln.split()
            if len(cols) < 4 or cols[3] != "0A":
                continue
            local = cols[1]
            addr_hex, port_hex = local.rsplit(":", 1)
            if int(port_hex, 16) != int(port):
                continue
            if fam == "ipv4":
                b = bytes.fromhex(addr_hex)
                addr = ".".join(str(x) for x in reversed(b))
            else:
                addr = "[" + addr_hex + "]"
            rec["listening"].append({"family": fam, "address": addr})
    addrs = {x["address"] for x in rec["listening"]}
    rec["addresses"] = sorted(addrs)
    rec["loopback_only"] = bool(addrs) and addrs <= {"127.0.0.1", "[" + "0" * 31 + "1" + "]"}
    if not addrs:
        rec["note"] = "no listening socket was found for this port"
    elif not rec["loopback_only"]:
        rec["note"] = ("the listener is NOT restricted to loopback: on a "
                       "shared node the port is reachable on every interface "
                       "while the server runs. This is the installed server's "
                       "behaviour, not a choice of this runner.")
    return rec


def process_group_members(pgid: int) -> list:
    """Live (non-zombie) processes in a process group, read from /proc."""
    live = []
    for d in pathlib.Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
        except OSError:
            continue
        try:
            after = stat.rsplit(") ", 1)[1].split()
            state, got_pgid = after[0], int(after[2])
        except (IndexError, ValueError):
            continue
        if got_pgid == pgid and state != "Z":
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", "replace").strip()
            except OSError:
                cmd = ""
            live.append({"pid": int(d.name), "state": state, "cmdline": cmd[:160]})
    return sorted(live, key=lambda x: x["pid"])


def shutdown_server(proc, *, grace_s: float = 45.0) -> dict:
    """Stop the whole server process GROUP and confirm it is gone.

    The server is started with `start_new_session=True`, so it leads its own
    session and process group. vLLM's multiprocessing engine runs as a child of
    it: terminating only the direct child would leave that engine holding
    device memory, which is exactly the "leaves descendants" failure this
    records. The group is signalled, escalated, and then confirmed absent.
    """
    rec = {"pid": proc.pid, "pgid": None, "members_before": [],
           "signals_sent": [], "members_after": [], "confirmed_absent": False,
           "exit_code": None, "reaped": False}
    try:
        rec["pgid"] = os.getpgid(proc.pid)
    except OSError:
        rec["pgid"] = proc.pid
    rec["members_before"] = process_group_members(rec["pgid"])
    rec["descendants_before"] = [m for m in rec["members_before"]
                                 if m["pid"] != proc.pid]
    deadline = time.monotonic() + grace_s
    for sig, label in ((signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")):
        if not process_group_members(rec["pgid"]):
            break
        try:
            os.killpg(rec["pgid"], sig)
            rec["signals_sent"].append(label)
        except OSError as exc:
            rec["signals_sent"].append(f"{label}:{type(exc).__name__}")
        while time.monotonic() < deadline:
            if not process_group_members(rec["pgid"]):
                break
            time.sleep(0.5)
        if not process_group_members(rec["pgid"]):
            break
        deadline = time.monotonic() + min(15.0, grace_s)
    try:                                   # reap our direct child, if it is one
        rec["exit_code"] = proc.wait(timeout=10)
        rec["reaped"] = True
    except Exception:                      # noqa: BLE001
        rec["exit_code"] = proc.poll()
    rec["members_after"] = process_group_members(rec["pgid"])
    rec["confirmed_absent"] = not rec["members_after"]
    rec["note"] = ("zombies are excluded from the live set and the direct "
                   "child is reaped before the final check, so a zombie is "
                   "never reported as a surviving process")
    return rec


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_health(port: int, *, timeout_s: float, proc) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2.0)
    return False


GUIDED_BACKEND = "lm-format-enforcer"


def build_request_payload(model: str, prompt_ids: list, max_tokens: int) -> dict:
    """THE production request payload. Nothing else builds one.

    The frozen workload is unconstrained greedy generation from explicit token
    ids. No `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`,
    `response_format` or `guided_decoding_backend` field is sent, because any
    of them would change what the engine does -- and `guided_grammar` in
    particular selects the Outlines backend regardless of the server's
    `--guided-decoding-backend` setting
    (`guided_decoding/__init__.py`: `if guided_params.backend == 'outlines' or
    guided_params.grammar:`). The acceptance tests assert the absence of all
    of them against this one definition.
    """
    return {"model": model, "prompt": list(prompt_ids),
            "max_tokens": int(max_tokens), "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}


CONSTRAINT_FIELDS = ("guided_json", "guided_regex", "guided_choice",
                     "guided_grammar", "guided_whitespace_pattern",
                     "guided_decoding_backend", "response_format", "tools",
                     "tool_choice", "logit_bias", "allowed_token_ids")


def build_server_argv(interpreter: str, *, model: str, revision: str,
                      host: str, port: int) -> list:
    """THE production server command line. Nothing else builds one."""
    return [interpreter, "-B", str(HERE / "stream_server.py"),
            "--model", model, "--revision", revision,
            "--tokenizer-revision", revision,
            "--host", host, "--port", str(port),
            "--dtype", "float16", "--max-model-len", "1024",
            "--gpu-memory-utilization", "0.85",
            # Every completion request in vLLM 0.6.3.post1 carries a non-None
            # GuidedDecodingParams, because
            # `entrypoints/openai/protocol.py:321` calls
            # `GuidedDecodingParams.from_optional(...)`, which always returns
            # an object (`sampling_params.py:63`) even when no guide is
            # requested. `engine/multiprocessing/client.py:545` therefore
            # always builds a guided-decoding processor, and
            # `guided_decoding/__init__.py:12` imports `outlines` whenever the
            # effective backend is 'outlines' -- BEFORE
            # `outlines_decoding.py:64` can return None for a guide-free
            # request. In this environment that import chain is broken:
            # outlines -> datasets 5.1.0 -> huggingface_hub 0.36.2, which no
            # longer exports the names datasets imports. The server then raises
            # inside the already-started SSE response, so the client sees
            # HTTP 200 with an empty body and zero events.
            #
            # Selecting the installed, importable backend routes the same
            # guide-free request through `lm_format_enforcer_decoding.py:44`,
            # which returns None. No logits processor is attached either way,
            # so the model, prompt, precision and greedy policy are unchanged;
            # only which optional library vLLM imports on the request path
            # changes. This is NOT a repair of the dependency conflict itself,
            # which is reported separately and remains present.
            "--guided-decoding-backend", GUIDED_BACKEND,
            "--disable-log-requests"]


def _stream_one(port: int, model: str, prompt_ids: list[int], max_tokens: int,
                rid: str, events: dict, status: dict, usage: dict,
                t_origin: float, request_timing: dict | None = None) -> None:
    body = json.dumps(
        build_request_payload(model, prompt_ids, max_tokens)).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    evs: list[dict] = []
    timing = {"request_start_s": round(time.monotonic() - t_origin, 6)}
    if request_timing is not None:
        request_timing[rid] = timing
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            timing["headers_received_s"] = round(time.monotonic() - t_origin, 6)
            timing["http_status"] = r.status
            if r.status != 200:
                raise ValueError(f"unexpected HTTP status {r.status}")
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                t = round(time.monotonic() - t_origin, 6)
                if payload == "[DONE]":
                    evs.append({"t": t, "done": True})
                    continue
                try:
                    obj = json.loads(payload)
                except ValueError:
                    evs.append({"t": t, "unparsed": payload[:200]})
                    continue
                ch = (obj.get("choices") or [{}])[0]
                # Raw event, preserved. `text` is a CHUNK of text, not a token.
                e = {"t": t, "text": ch.get("text") or "",
                     "finish_reason": ch.get("finish_reason")}
                e["payload"] = obj
                if obj.get("error"):
                    e["error"] = obj["error"]
                # Only record token_ids if the server actually supplies them.
                for key in ("token_ids", "logprobs"):
                    if ch.get(key) is not None:
                        e[key] = ch[key]
                if obj.get("usage"):
                    usage[rid] = obj["usage"]
                evs.append(e)
        status[rid] = "ok"
    except urllib.error.HTTPError as exc:
        status[rid] = f"http_{exc.code}"
        evs.append({"t": round(time.monotonic() - t_origin, 6),
                    "error": f"HTTPError {exc.code}"})
    except Exception as exc:
        status[rid] = f"error:{type(exc).__name__}"
        evs.append({"t": round(time.monotonic() - t_origin, 6),
                    "error": f"{type(exc).__name__}: {exc}"})
    events[rid] = evs
    timing["request_end_s"] = round(time.monotonic() - t_origin, 6)


def session_evidence_complete(ev: dict) -> dict:
    """A streaming case passes only under the FROZEN success rule.

    `protocols/STREAMING_SUCCESS_RULE.md`, implemented once in
    `zerodiag.streaming.classify_request` / `account_session` and shared with
    the published offline tool. A healthy server, an HTTP 200, or a complete
    set of empty bodies is explicitly not a completed case.
    """
    missing, detail = [], {}
    acc = ev.get("accounting") or {}
    counts = acc.get("counts") or {}
    detail["requests_expected"] = acc.get("requests_expected")
    detail["requests_recorded"] = acc.get("requests_recorded")
    detail["counts"] = counts
    detail["successful_without_length_evidence"] = acc.get(
        "successful_without_length_evidence")
    if not acc:
        missing.append("no session accounting was produced")
    if not acc.get("accounting_complete"):
        missing.append("request accounting is incomplete: recorded "
                       f"{acc.get('requests_recorded')} of expected "
                       f"{acc.get('requests_expected')}")
    if not counts.get("successful"):
        missing.append("no request satisfied the frozen success rule "
                       "(content event + completion marker, with HTTP 200 "
                       "alone insufficient)")
    if not ev.get("raw_events_file"):
        missing.append("raw events were not persisted")
    wl = ev.get("frozen_workload") or {}
    detail["frozen_workload"] = wl
    if not wl.get("issued_as_frozen"):
        missing.append("the issued workload does not match the frozen "
                       "protocol")
    if not wl.get("input_length_verified"):
        missing.append("the consumed input length was not verified against "
                       "the frozen protocol")
    detail["metrics_dir_used"] = (ev.get("metrics_outcome") or {}).get(
        "used_by_server")
    sd = ev.get("server_shutdown")
    detail["server_shutdown"] = sd
    if sd is not None and not sd.get("confirmed_absent"):
        missing.append("the server process group was not confirmed absent; a "
                       "session whose server leaves descendants is not a "
                       "completed observation")
    srv = ev.get("server") or {}
    detail["server_exit_before_session"] = srv.get("exited_before_session")
    if srv.get("exited_before_session"):
        missing.append("the server process exited before the session "
                       "completed")
    return {"ok": not missing, "missing": missing, "detail": detail,
            "rule": ST.SUCCESS_RULE["document"],
            "basis": ("evidence recorded by this runner under the frozen "
                      "rule, not server liveness and not a latency "
                      "distribution"),
            "scope": ("one session establishes only this session's "
                      "observations; its concurrent requests are not "
                      "independent repeated sessions")}


def completion_markers(events: dict, status: dict, usage: dict) -> dict:
    """Per-request completion evidence, read from the preserved raw events."""
    per = {}
    for rid, evs in sorted(events.items()):
        done = any(e.get("done") is True for e in evs)
        fin = [e.get("finish_reason") for e in evs if e.get("finish_reason")]
        text = "".join(e.get("text") or "" for e in evs)
        per[rid] = {"done_sentinel": done,
                    "finish_reason": (fin[-1] if fin else None),
                    "http_status": status.get(rid),
                    "output_chars": len(text),
                    "server_usage": usage.get(rid),
                    "token_identities_supplied": any(
                        e.get("token_ids") is not None for e in evs)}
    return {"per_request": per,
            "any": any(d["done_sentinel"] or d["finish_reason"]
                       for d in per.values()),
            "all": bool(per) and all(d["done_sentinel"] or d["finish_reason"]
                                     for d in per.values()),
            "n_with_finish_reason": sum(1 for d in per.values()
                                        if d["finish_reason"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-cpus", type=int, default=4)
    ap.add_argument("--memory-budget-gib", type=float, default=8.0)
    ap.add_argument("--monitor-interval-s", type=float, default=0.5)
    ap.add_argument("--server-timeout-s", type=float, default=600.0)
    ap.add_argument("--requests", type=int, default=None)
    ap.add_argument("--device-index", type=int, default=None,
                    help="physical index of the bound device, already verified "
                         "against its UUID by the launcher's preflight")
    a = ap.parse_args(argv)
    if a.world_size != 1 or a.rank != 0:
        print("REFUSED: the streaming session is world size one", file=sys.stderr)
        return 95

    t0 = time.perf_counter()
    aff = apply_affinity(a.max_cpus)
    from zerodiag.ownership import JobOwnership, JOB_TOKEN_ENV
    out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    # Keep the outer launcher's token so its ledger can also sweep the server
    # if this runner exits abruptly. The inner ledger excludes the client group
    # so normal server cleanup cannot signal the client or its monitor.
    server_ownership = JobOwnership(
        os.environ.get(JOB_TOKEN_ENV) or ("stream-" + os.urandom(16).hex()),
        ledger_path=out / "server_ownership_ledger.json",
        excluded_pgids=(os.getpgid(0),))
    global MONITOR
    MONITOR = MonitorThread(
        budget_bytes=int(a.memory_budget_gib * 1073741824),
        interval_s=a.monitor_interval_s,
        additional_pids=server_ownership.owned_pids,
        on_breach=lambda report: server_ownership.terminate(grace_s=5.0)).start()

    out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    ev = {"case_id": a.case_id, "kind": "streaming", "stage": "started",
          "rank": a.rank, "world_size": a.world_size,
          "containment": {"affinity": aff, "monitor": MONITOR.report()},
          "child_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
          "protocol": ST.PROTOCOL, "protocol_digest": ST.protocol_digest(),
          "device_contact": "pending"}
    server = None

    def dump(stage, **kw):
        ev["stage"] = stage
        ev["elapsed_s"] = round(time.perf_counter() - t0, 3)
        ev["memory_monitor"] = MONITOR.report() if MONITOR else None
        ev.update(kw)
        if stage == "completed":
            gate = session_evidence_complete(ev)
            ev["session_evidence"] = gate
            if not gate["ok"]:
                ev["stage"] = stage = "refused"
                ev["reason"] = ("the session lacks required evidence (" +
                                ", ".join(gate["missing"]) + "); a streaming "
                                "case is not complete without an actual "
                                "request, recorded events and a completion "
                                "marker")
            ev["streaming_case_complete"] = (stage == "completed")
        (out / "streaming_evidence.json").write_text(
            json.dumps(ev, indent=2, sort_keys=True, default=str))

    try:
        # The frozen workload is validated BEFORE any device is contacted: a
        # corpus or input length that is not the frozen one is refused without
        # touching a GPU, and the refusal is reachable on CPU, which is how it
        # is regression-tested.
        want = ST.PROTOCOL["prompt_len_tokens"]
        corpus_obj = json.loads(pathlib.Path(a.corpus).read_text())
        prompts = corpus_obj["prompts"]
        # The corpus holds PRE-TOKENIZED id lists. They are sent through the
        # backend's token-input interface (`prompt` accepts a list of token
        # ids); the tokenizer is never asked to tokenize them.
        ev["corpus_representation"] = {
            "prompts_are_pre_tokenized_id_lists": all(
                isinstance(x, list) and all(isinstance(i, int) for i in x)
                for x in prompts[:4]),
            "declared_prompt_len": corpus_obj.get("prompt_len"),
            "consumed_as": "token ids via the completions `prompt` field",
            "tokenizer_used_to_tokenize": False}
        ids_by_prompt, lengths = [], []
        for row in prompts:
            if not (isinstance(row, list) and all(isinstance(i, int) for i in row)):
                dump("refused", reason="corpus prompt is not a list of token ids")
                return 90
            lengths.append(len(row))
            if len(row) >= want:
                ids_by_prompt.append(row[:want])
        ev["consumed_token_lengths"] = lengths
        ev["length_deviations"] = [(i, n) for i, n in enumerate(lengths) if n != want]
        ev["prompt_length_check"] = ST.verify_prompt_lengths(
            [len(i) for i in ids_by_prompt])
        if not ids_by_prompt:
            dump("refused", reason=(f"no corpus prompt reaches {want} tokens; "
                                    f"observed lengths {sorted(set(lengths))}"))
            return 90
        # The ACTUAL consumed length must be the frozen one, exactly, for
        # every prompt that will be sent. A session run at the wrong input
        # length is not the frozen workload, so it is refused here rather than
        # measured and explained afterwards.
        consumed = sorted({len(i) for i in ids_by_prompt})
        if consumed != [want]:
            dump("refused", reason=(f"consumed prompt lengths {consumed} are "
                                    f"not exactly the frozen {want} token "
                                    f"ids; refusing to run a different "
                                    f"workload"))
            return 90

        import torch
        ev["torch_version"] = torch.__version__
        if not torch.cuda.is_available():
            dump("refused", reason="no CUDA device visible to the child")
            return 90
        props = torch.cuda.get_device_properties(0)
        ev["child_device_name"] = props.name
        ev["device_contact"] = "one device, read via torch properties"

        model = ST.PROTOCOL["model"]
        rev = ST.PROTOCOL["model_revision"]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, revision=rev,
                                            local_files_only=True)
        ev["revision_check"] = ST.verify_revision(rev, rev)

        port = _free_port()
        ev["server"] = {"host_requested": "127.0.0.1", "port": port,
                        "fresh_for_this_session": True,
                        "loopback_only": "unmeasured until readiness"}
        ev["guided_decoding_backend"] = {
            "selected": "lm-format-enforcer",
            "default_would_be": "outlines",
            "reason": ("vLLM 0.6.3.post1 builds a guided-decoding processor "
                       "for every request because "
                       "GuidedDecodingParams.from_optional never returns None; "
                       "the default 'outlines' backend's import chain is broken "
                       "in this environment (datasets 5.1.0 vs "
                       "huggingface_hub 0.36.2)"),
            "effect_on_sampling": ("none: no guide is requested, so both "
                                   "backends return no logits processor"),
            "generation_policy_unchanged": True}
        # vLLM resolves CUDA_VISIBLE_DEVICES with int() --
        # `vllm/platforms/cuda.py:98 device_id_to_physical_device_id` does
        # `int(physical_device_id)` -- so a UUID-valued visibility string raises
        # ValueError and the server never starts. This runner keeps its OWN
        # UUID-valued visibility for identity verification (torch accepts it,
        # and the check above already confirmed the device name), and gives the
        # server the physical index that the launcher's preflight already
        # verified against that same UUID. The binding chain is
        # receipt UUID -> preflight index/UUID match -> server index.
        server_env = dict(os.environ)
        if a.device_index is None:
            dump("refused", reason=("no --device-index supplied; the streaming "
                                    "server cannot be bound without the "
                                    "preflight-verified physical index"))
            return 90
        server_env["CUDA_VISIBLE_DEVICES"] = str(a.device_index)
        metrics = prepare_metrics_dir(out / "prometheus_multiproc")
        server_env["PROMETHEUS_MULTIPROC_DIR"] = metrics["dir"]
        metrics["propagated_in_child_env"] = (
            server_env.get("PROMETHEUS_MULTIPROC_DIR") == metrics["dir"])
        ev["metrics_initialization"] = metrics
        ev["server_device_binding"] = {
            "runner_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "server_visibility": str(a.device_index),
            "binding_chain": ("receipt UUID -> launcher preflight verified "
                              "index/UUID -> server physical index"),
            "reason": ("vllm/platforms/cuda.py:98 parses the visibility string "
                       "with int(); a UUID raises ValueError"),
            "identity_verified_by": "torch device properties against the receipt",
        }
        server = subprocess.Popen(
            build_server_argv(sys.executable, model=model, revision=rev,
                              host="127.0.0.1", port=port),
            env=server_ownership.env_for_child(server_env),
            # Its own session, so the whole group -- including vLLM's
            # multiprocessing engine child -- can be signalled and confirmed
            # gone. `start_new_session` is implemented in C, not a Python
            # `preexec_fn`, so it is safe in this threaded parent.
            start_new_session=True,
            stdout=open(out / "server.stdout", "w"),
            stderr=open(out / "server.stderr", "w"))
        server_ownership.register(0, server.pid)
        ev["server"]["pid"] = server.pid
        if not _wait_health(port, timeout_s=a.server_timeout_s, proc=server):
            dump("refused", reason=("the loopback server did not become "
                                    "healthy within the allowance"),
                 server_exit=server.poll())
            return 90
        ev["server"]["healthy"] = True
        # measured, not asserted: see listening_bind_addresses()
        ev["server"]["bind"] = listening_bind_addresses(port)
        ev["server"]["loopback_only"] = ev["server"]["bind"]["loopback_only"]
        if ev["server"]["loopback_only"] is not True:
            dump("refused", reason="API listener is not confirmed loopback-only; no requests issued")
            return 90
        try:
            ev["server"]["pgid"] = os.getpgid(server.pid)
            ev["server"]["group_members_at_readiness"] = process_group_members(
                ev["server"]["pgid"])
        except OSError as exc:
            ev["server"]["pgid_error"] = f"{type(exc).__name__}: {exc}"

        n_req = a.requests or ST.PROTOCOL["requests"]
        conc = ST.PROTOCOL["concurrency"]
        max_tok = ST.PROTOCOL["output_tokens"]
        events, status, usage, request_timing = {}, {}, {}, {}
        ev["request_parameters"] = {
            "endpoint": "/v1/completions", "stream": True,
            "stream_options": {"include_usage": True},
            "model": model, "revision": rev,
            "prompt_is_token_ids": True,
            "prompt_len_tokens": want,
            "max_tokens": max_tok, "temperature": 0.0,
            "greedy": True, "requests": n_req, "concurrency": conc,
            "client_timeout_s": 300}
        ev["clock_definitions"] = {
            "event_t": ("seconds from t_origin, from time.monotonic(); "
                        "monotonic, same process, not wall-clock and not "
                        "comparable across machines"),
            "t_origin_monotonic": None,
            "t_origin_utc": None,
            "elapsed_s": "time.perf_counter() since runner start",
            "token_times": ("NOT derived; a chunk arrival time is not a token "
                            "time and individual token timestamps are never "
                            "inferred from network chunks")}
        t_origin = time.monotonic()
        ev["clock_definitions"]["t_origin_monotonic"] = round(t_origin, 6)
        ev["clock_definitions"]["t_origin_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        work = [(f"r{i:03d}", ids_by_prompt[i % len(ids_by_prompt)])
                for i in range(n_req)]
        idx = 0
        lock = threading.Lock()

        def worker():
            nonlocal idx
            while True:
                with lock:
                    if idx >= len(work):
                        return
                    rid, ids = work[idx]; idx += 1
                _stream_one(port, model, ids, max_tok, rid, events, status,
                            usage, t_origin, request_timing)

        threads = [threading.Thread(target=worker) for _ in range(conc)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        srv_rc = server.poll()
        ev["server"]["exited_before_session"] = srv_rc is not None
        ev["server"]["exit_code_after_session"] = srv_rc
        (out / "raw_events.json").write_text(
            json.dumps({"events": events, "status": status, "usage": usage,
                        "request_timing": request_timing},
                       indent=2, sort_keys=True, default=str))
        completion = {rid: (usage.get(rid) or {}).get("completion_tokens")
                      for rid in events}
        ev["raw_events_file"] = "raw_events.json"
        ev["accounting"] = ST.account_session(
            events, status, usage, requests_expected=n_req,
            output_tokens=max_tok)
        ev["frozen_workload"] = {
            "requests_expected": ST.PROTOCOL["requests"],
            "requests_issued": len(work),
            "concurrency": conc,
            "prompt_len_tokens": ST.PROTOCOL["prompt_len_tokens"],
            "output_tokens": ST.PROTOCOL["output_tokens"],
            "consumed_prompt_lengths": sorted({len(i) for i in ids_by_prompt}),
            "input_length_verified": all(
                len(i) == ST.PROTOCOL["prompt_len_tokens"]
                for i in ids_by_prompt),
            "issued_as_frozen": (len(work) == ST.PROTOCOL["requests"]
                                 and conc == ST.PROTOCOL["concurrency"]
                                 and max_tok == ST.PROTOCOL["output_tokens"]),
            "requests_not_reduced": ("the full frozen request count was "
                                     "issued; no request was dropped from the "
                                     "accounting"),
            "request_ids": [rid for rid, _ in work],
        }
        ev["completion_markers"] = completion_markers(events, status, usage)
        ev["outputs_sample"] = {
            rid: "".join(e.get("text") or "" for e in evs)[:200]
            for rid, evs in sorted(events.items())[:3]}
        ev["summary"] = ST.summarize_session(events, completion_tokens=completion,
                                             status=status, request_timing=request_timing)
        ev["actual_completion_tokens"] = completion
        ev["requested_output_tokens"] = max_tok
        ev["completion_tokens_match_protocol"] = all(
            v == max_tok for v in completion.values() if v is not None)
        ev["metrics_outcome"] = metrics_dir_outcome(metrics)
        ev["cuda_peak_alloc_mib"] = round(
            torch.cuda.max_memory_allocated() / 1048576, 1)
        dump("completed")
        return 0 if ev.get("streaming_case_complete") else 90
    except Exception as exc:
        import traceback
        dump("error", reason=f"{type(exc).__name__}: {exc}",
             traceback=traceback.format_exc())
        return 91
    finally:
        if server is not None:
            try:
                sd = shutdown_server(server, grace_s=45.0)
                union_cleanup = server_ownership.terminate(grace_s=5.0)
                sd["cross_session_cleanup"] = union_cleanup
                sd["confirmed_absent"] = (sd.get("confirmed_absent") is True
                    and union_cleanup.get("owned_absent_confirmed") is True)
            except Exception as exc:                        # noqa: BLE001
                sd = {"confirmed_absent": False,
                      "error": f"{type(exc).__name__}: {exc}"}
            ev["server_shutdown"] = sd
            try:
                (out / "streaming_evidence.json").write_text(
                    json.dumps(ev, indent=2, sort_keys=True, default=str))
            except OSError:
                pass
        if MONITOR is not None:
            ev["memory_monitor"] = MONITOR.stop()
        cleanup_failed = server is not None and not (ev.get("server_shutdown") or {}).get("confirmed_absent")
        if cleanup_failed or (ev.get("memory_monitor") or {}).get("attempt_successful") is False:
            ev["stage"] = "cleanup_unverified" if cleanup_failed else "refused"
            ev["streaming_case_complete"] = False
            ev["reason"] = "owned cleanup or aggregate monitoring was not confirmed"
        (out / "streaming_evidence.json").write_text(
            json.dumps(ev, indent=2, sort_keys=True, default=str))
        if cleanup_failed:
            return 91


if __name__ == "__main__":
    sys.exit(main())
