"""The REAL one-boundary worker. Runs inside an owned child process.

Executable path: retained factories -> installed DeepSpeed ZeRO optimizer ->
engine_probe -> update_oracle -> verdict -> evidence JSON. Nothing here is a
planning stub; this is the code a device run executes.

It is bounded by construction: zero warmup, exactly ONE audited optimizer
boundary, bounded snapshot retention (only the partitions of this rank, before
and after, released as soon as the verdict is formed).
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, resource, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from zerodiag import engine_probe as EP      # noqa: E402
from zerodiag import factories as F          # noqa: E402
from zerodiag import update_oracle as OR     # noqa: E402
from zerodiag import verdict as V            # noqa: E402


MONITOR = None


def start_containment(max_cpus: int, budget_gib: float, interval_s: float):
    """R1 + R3: real affinity mask, then the memory monitor, BEFORE the model."""
    from zerodiag.containment import apply_affinity, MonitorThread
    aff = apply_affinity(max_cpus)
    mt = MonitorThread(budget_bytes=int(budget_gib * 1073741824),
                       interval_s=interval_s).start()
    global MONITOR
    MONITOR = mt
    return {"affinity": aff, "monitor": mt.report()}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--receipt", required=True, help="authorization receipt JSON")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--zero-stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--ds-config", required=True)
    ap.add_argument("--max-cpus", type=int, default=4)
    ap.add_argument("--memory-budget-gib", type=float, default=8.0)
    ap.add_argument("--monitor-interval-s", type=float, default=0.5)
    ap.add_argument("--chunk-elems", type=int, default=4_194_304,
                    help="maximum elements per snapshot/oracle chunk")
    # Multi-rank and precision are REQUIRED. There is no default that silently
    # means world size one, rank zero or FP32.
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--precision", required=True, choices=["fp32", "amp_fp16"])
    a = ap.parse_args(argv)
    if a.rank < 0 or a.world_size < 1 or a.rank >= a.world_size:
        print(f"REFUSED: rank {a.rank} is not valid for world size "
              f"{a.world_size}", file=sys.stderr)
        return 95
    t0 = time.perf_counter()
    # R1/R3 before anything heavy is constructed. A failure here is a refusal.
    containment = start_containment(a.max_cpus, a.memory_budget_gib,
                                    a.monitor_interval_s)
    out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    rec = json.loads(pathlib.Path(a.receipt).read_text())
    ev: dict = {"receipt": rec, "worker_pid": os.getpid(),
                "stage": "started", "device_contact": "pending",
                "containment": containment,
                # Rank identity is recorded BEFORE the device guard, so a rank
                # that refuses for lack of a device still reports which rank it
                # was. Previously these were set after the guard and a refused
                # rank carried no identity at all.
                "rank": a.rank, "world_size": a.world_size,
                "case_id": a.case_id, "precision": a.precision,
                "distributed_env_seen": {
                    k: os.environ.get(k) for k in
                    ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR",
                     "MASTER_PORT")},
                "child_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "")}

    def dump(status: str, **kw):
        ev["stage"] = status
        ev["elapsed_s"] = round(time.perf_counter() - t0, 3)
        ev["peak_rss_gib_self"] = round(rss_gib(), 3)
        # The self RUSAGE figure above is a post-hoc high-water mark for THIS
        # process only. The operational monitor below is the operational
        # evidence: sampled, owned-process-group attributed, with a stop path.
        if MONITOR is not None:
            ev["memory_monitor"] = MONITOR.report()
        else:
            ev["memory_monitor"] = None
            ev["verification_successful"] = False
            ev["refusal"] = ("no memory monitor was running; unmonitored "
                             "execution is not accepted as a successful attempt")
        ev.update(kw)
        (out / "integration_evidence.json").write_text(
            json.dumps(ev, indent=2, sort_keys=True, default=str))

    try:
        import torch
        ev["torch_version"] = torch.__version__
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ev["child_visible_devices"] = vis
        if not torch.cuda.is_available():
            dump("refused", reason="no CUDA device visible to the child")
            return 90
        # child-visible identity, cross-checked against the receipt
        props = torch.cuda.get_device_properties(0)
        ev["child_device_name"] = props.name
        ev["child_device_capability"] = f"{props.major}.{props.minor}"
        ev["child_device_total_mib"] = round(props.total_memory / 1048576, 1)
        if rec.get("device_name") and rec["device_name"] != props.name:
            dump("refused", reason=f"receipt names {rec['device_name']!r} but the "
                                   f"child sees {props.name!r}")
            return 90
        ev["device_contact"] = "one device, read via torch properties"

        # ---- retained factories, the real ones ----
        model, cfg = F.build_model(seed=1234, device="cuda")
        ev["model_identity"] = F.resolve_model_identity(model, cfg)
        # R4: the coverage inventory is reconciled against the ACTUAL unique
        # parameter inventory resolved by storage, not against counts or world
        # size. Tied aliases are deduplicated by distinct storage.
        _mi = ev["model_identity"]
        ev["param_inventory"] = {
            "n_unique_parameters": _mi["distinct_storages"],
            "n_unique_elements": _mi["unique_trainable_numel"],
            "named_parameter_entries": _mi["named_parameter_entries"],
            "tied_alias_groups": _mi["tied_alias_groups"],
            "deduplication": ("distinct untyped-storage data pointers; tied "
                              "lm_head/wte share one storage and are counted "
                              "once"),
        }
        ev["initial_state"] = F.state_digest(model)
        # The generator is rank-seeded, so each rank's realized input is
        # recorded separately and never assumed identical across ranks.
        gen = F.build_generator(seed=1234, rank=a.rank, device="cuda")
        ids = F.synthetic_batch(a.batch_size, a.seq_len, torch.device("cuda"), gen)
        ev["realized_input"] = F.input_digest(ids)
        base_opt = F.build_optimizer(model)
        ev["resolved_optimizer"] = F.resolve_optimizer_semantics(base_opt)

        # ---- the installed DeepSpeed engine, real ----
        import deepspeed, inspect
        ev["deepspeed_version"] = deepspeed.__version__
        import deepspeed.runtime.zero.stage_1_and_2 as s12
        ev["installed_source"] = {
            "stage_1_and_2.py": hashlib.sha256(
                pathlib.Path(inspect.getfile(s12)).read_bytes()).hexdigest()}
        ds_cfg = json.loads(pathlib.Path(a.ds_config).read_text())
        ds_cfg["train_micro_batch_size_per_gpu"] = a.batch_size
        ds_cfg["gradient_accumulation_steps"] = 1
        ds_cfg.setdefault("zero_optimization", {})["stage"] = a.zero_stage
        # AMP resolved from the retained benchmark's own configuration.
        ds_cfg, _amp_sem = F.resolve_amp_config(a.precision, ds_cfg)
        ev["amp_semantics"] = _amp_sem
        ev["resolved_ds_config"] = ds_cfg
        engine, _, _, _ = deepspeed.initialize(
            model=model, optimizer=base_opt, config=ds_cfg,
            model_parameters=model.parameters())
        zopt = engine.optimizer
        ev["wrapper_classes"] = {
            "engine": type(engine).__name__,
            "zero_optimizer": type(zopt).__name__,
            "zero_optimizer_module": type(zopt).__module__,
            "base_optimizer": type(getattr(zopt, "optimizer", None)).__name__,
        }
        dump("engine_initialised")

        # ---- where is the hook relative to scaling and clipping? ----
        ev["hook_position"] = {
            "reads": "single_partition_of_fp32_groups[i].grad immediately before "
                     "and after the base optimizer advances its step counter",
            "established_from": "installed stage_1_and_2 source: unscale_and_clip_grads "
                                "is applied to the partition gradient before "
                                "self.optimizer.step(), so master.grad at this point "
                                "is post-unscale and post-clip",
            "native_gradscaler_semantics_assumed": False,
            "note": "DeepSpeed performs its own loss scaling and clipping; torch "
                    "GradScaler semantics are NOT assumed to apply",
            "fp32_run": "loss scaling is disabled in this FP32 case, so only "
                        "clipping applies",
        }
        ev["layout"] = EP.describe_optimizer(zopt)
        ev["expected_partitions"] = EP.expected_partition_map(
            zopt, rank=a.rank, world_size=a.world_size)["expected_partitions"]
        ev["storage_correspondence"] = EP.verify_storage_correspondence(
            zopt, rank=a.rank)
        # Expected base-optimizer calls are DERIVED from the installed engine's
        # own partitioning, one per fp32 master group, not generalised from
        # Gate-0's single-group layout.
        _n_groups = len(EP._req(zopt, "single_partition_of_fp32_groups"))
        ev["expected_base_calls"] = _n_groups
        ev["expected_base_calls_basis"] = (
            "len(single_partition_of_fp32_groups): DeepSpeed calls "
            "_optimizer_step(i) once per fp32 master group per rank")
        dump("layout_bound")

        # ---- exactly one boundary: zero warmup ----
        outp = engine(input_ids=ids, labels=ids)
        loss = outp.loss
        ev["loss"] = float(loss.detach())
        engine.backward(loss)
        # The master partition's .grad is live only inside the ZeRO step, so the
        # boundary is captured at the base optimizer's own step() call.
        with EP.boundary_capture(zopt, rank=a.rank, world_size=a.world_size,
                                 chunk_elems=a.chunk_elems) as box:
            engine.step()
        ev["capture"] = {k: box[k] for k in
                         ("base_step_calls", "capture_point",
                          "observed_gradient",
                          "reconstructed_from_averaged_gradients")}
        if box["base_step_calls"] != _n_groups:
            dump("refused", reason=(
                f"the base optimizer was called {box['base_step_calls']} times "
                f"for one logical boundary; this engine layout has {_n_groups} "
                f"fp32 master group(s), so exactly {_n_groups} call(s) were "
                f"expected"))
            return 90
        before, after = box["before"], box["after"]
        if before is None or after is None:
            dump("refused", reason=("the base optimizer was never called, so no "
                                    "boundary was observed"))
            return 90
        # Working/master consistency against THIS rank's own partition view,
        # after engine.step() has fully returned. The previous whole-group
        # comparison refused at world size > 1 because a partition is not the
        # group; comparing a prefix would have been wrong.
        ev["partition_working_map"] = EP.partition_working_map(
            zopt, rank=a.rank, world_size=a.world_size)
        ev["module_reflects_masters"] = EP.module_reflects_masters_partitioned(
            zopt, rank=a.rank, world_size=a.world_size,
            chunk_elems=a.chunk_elems)
        hp = dict(ev["resolved_optimizer"])
        hp = {k: hp[k] for k in ("lr", "beta1", "beta2", "eps", "weight_decay",
                                 "decoupled_weight_decay", "amsgrad")}
        verd = V.verify_boundary(
            before=before, after=after,
            expected_map={"expected_partitions": ev["expected_partitions"]},
            resolved_hparams=hp, accumulation_steps=1, rank=a.rank,
            world_size=a.world_size,
            storage_check=ev["storage_correspondence"],
            module_check=ev["module_reflects_masters"], coverage=OR.FULL,
            chunk_elems=a.chunk_elems,
            param_inventory=ev.get("param_inventory"))
        # bounded retention: keep scalars and verdicts, drop the tensors
        for snap in (before, after):
            for g in snap["groups"]:
                for k in ("master", "grad", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    g[k] = None
        for p in verd["per_partition"]:
            p.pop("resolved_hparams", None)
        ev["verdict"] = verd
        # One rank writes one rank-local verdict. Reconciliation across the
        # exact expected rank set happens in the launcher, which is the only
        # process that sees every rank's evidence.
        ev["rank_verdict"] = verd
        ev["cuda_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
        dump("completed")
        return 0 if verd["rank_verification_successful"] else 1
    except EP.ProbeRefusal as exc:
        dump("refused", reason=f"ProbeRefusal: {exc}")
        return 90
    except OR.OracleRefusal as exc:
        dump("refused", reason=f"OracleRefusal: {exc}")
        return 90
    except Exception as exc:                      # recorded, never hidden
        import traceback
        dump("error", reason=f"{type(exc).__name__}: {exc}",
             traceback=traceback.format_exc()[-2000:])
        return 2


if __name__ == "__main__":
    sys.exit(main())
