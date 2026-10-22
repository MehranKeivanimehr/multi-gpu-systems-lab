"""The final diagnostic verdict. Refuses; never falls back to mechanics.

Combines, for one logical boundary:
  * the explicitly enumerated expected partition identity map,
  * storage correspondence between masters and the base optimizer,
  * the corrected oracle's conditional update check per partition,
  * partition reconciliation by identity,
  * accumulation normalisation, boundary/base-call/skip accounting,
  * module/master reflection where the layout exposes it.

Declared coverage: per-partition checks are FULL over the elements this rank
owns. Whole-model verification additionally requires every expected partition of
every rank to be present and reconciled; a single rank's full check is NOT
whole-model verification and is never labelled as such.
"""
from __future__ import annotations

from typing import Any

from . import engine_probe as EP
from . import update_oracle as OR


def coverage_inventory(*, expected_ids: list, results: list, rank: int,
                       world_size: int, param_inventory: dict | None) -> dict:
    """Inventory-based coverage: what was expected, what was observed.

    Coverage is a mapping claim about OBSERVED gradients. It is distinct from
    an independent recomputation of the gradient, which this diagnostic does
    not perform. Rank count establishes nothing: at world size one complete
    coverage MAY be possible, but only this inventory can show it.
    """
    expected_partitions = [(e["group"], e["rank"]) for e in expected_ids]
    expected_elements = sum(e["unpadded_numel"] for e in expected_ids)
    observed_partitions = [tuple(r["partition"]) for r in results]
    observed_ok = [tuple(r["partition"]) for r in results
                   if r.get("verification_successful")]
    observed_elements = sum(
        e["unpadded_numel"] for e, r in zip(expected_ids, results)
        if r.get("verification_successful"))
    missing = [p for p in expected_partitions if p not in observed_ok]
    unexpected = [p for p in observed_partitions if p not in expected_partitions]
    out = {
        "n_expected_partitions": len(expected_partitions),
        "expected_partitions": expected_partitions,
        "expected_elements_this_rank": expected_elements,
        "n_observed_partitions": len(observed_partitions),
        "observed_partitions_verified": observed_ok,
        "observed_elements_verified": observed_elements,
        "missing_partitions": missing,
        "unexpected_partitions": unexpected,
        "rank": rank, "world_size": world_size,
        "declared_per_partition_coverage": coverage_label(),
    }
    if param_inventory is None:
        out["model_parameter_inventory"] = None
        out["complete"] = False
        out["incomplete_because"] = (
            "no model parameter inventory was supplied, so observed gradient "
            "identities cannot be mapped to the expected unique parameters; "
            "partition-level accounting alone does not establish completeness")
        return out
    uniq = int(param_inventory["n_unique_parameters"])
    elems = int(param_inventory["n_unique_elements"])
    out["model_parameter_inventory"] = {
        "n_unique_parameters_expected": uniq,
        "n_unique_elements_expected": elems,
        "elements_accounted_for_by_verified_partitions": observed_elements,
        "elements_unaccounted": elems - observed_elements,
        "deduplication": param_inventory.get(
            "deduplication", "named_parameters() deduplicates tied tensors"),
    }
    out["complete"] = bool(
        not missing and not unexpected and observed_elements == elems)
    if not out["complete"]:
        out["incomplete_because"] = (
            f"{len(missing)} expected partition(s) unverified and "
            f"{elems - observed_elements} of {elems} expected unique elements "
            f"unaccounted for")
    return out


def coverage_label() -> str:
    return OR.FULL


def verify_boundary(*, before: dict, after: dict, expected_map: dict,
                    resolved_hparams: dict, accumulation_steps: int,
                    rank: int, world_size: int,
                    storage_check: dict, module_check: dict,
                    coverage: str = OR.FULL,
                    param_inventory: dict | None = None,
                    chunk_elems: int = OR.DEFAULT_CHUNK_ELEMS) -> dict:
    """Judge one boundary on one rank."""
    if before["rank"] != rank or after["rank"] != rank:
        raise EP.ProbeRefusal("snapshot rank does not match the declared rank")
    if len(before["groups"]) != len(after["groups"]):
        raise EP.ProbeRefusal("group count changed across the step")
    if accumulation_steps != 1:
        raise EP.ProbeRefusal(
            f"accumulation_steps={accumulation_steps}: the retained recipe uses "
            f"1, and a different normalisation changes the intended update; "
            f"refusing rather than guessing the divisor")
    if not storage_check.get("base_optimizer_sees_the_master_partitions"):
        raise EP.ProbeRefusal(
            f"master/base-optimizer storage correspondence failed: "
            f"{storage_check}")

    exp_ids = expected_map["expected_partitions"]
    if len(exp_ids) != len(before["groups"]):
        raise EP.ProbeRefusal(
            f"{len(exp_ids)} expected partitions but {len(before['groups'])} "
            f"observed groups")

    results, boundaries, base_calls, successes, skips = [], 0, 0, 0, 0
    for eid, b, a in zip(exp_ids, before["groups"], after["groups"]):
        boundaries += 1
        if b["numel"] != eid["partition_numel"]:
            raise EP.ProbeRefusal(
                f"group {eid['group']}: observed partition numel {b['numel']} "
                f"!= expected {eid['partition_numel']}")
        sb, sa = b["state_step"], a["state_step"]
        if sa is None:
            raise EP.ProbeRefusal(f"group {eid['group']}: no optimizer step state after the step")
        advanced = (sb is None and sa == 1) or (sb is not None and sa == sb + 1)
        if advanced:
            base_calls += 1
        else:
            skips += 1
            results.append({"partition": (eid["group"], rank),
                            "verification_successful": False,
                            "discriminating": True, "coverage": coverage,
                            "whole_model_verified": False,
                            "reason": f"base optimizer step did not advance: {sb} -> {sa}"})
            continue
        step_index = sa
        zeros = b["exp_avg"] if b["exp_avg"] is not None else None
        if zeros is None:
            import torch
            # First-step case: the base optimizer had no moments yet. These are
            # genuine zero tensors of partition size on the HOST (not device);
            # they are accounted for in the host budget as two partition-sized
            # fp32 buffers, and they are read chunk-wise like every other input.
            zeros = torch.zeros_like(b["master"])
            zeros_sq = torch.zeros_like(b["master"])
        else:
            zeros_sq = b["exp_avg_sq"]
        hp = dict(resolved_hparams)
        if hp.get("amsgrad"):
            hp["max_exp_avg_sq"] = (b["max_exp_avg_sq"] if b["max_exp_avg_sq"] is not None
                                    else zeros_sq)
        # Route 1: the production verdict path uses the CHUNKED oracle over the
        # independent CPU snapshots captured by capture_boundary. The unchunked
        # verify_update remains in the module as the reference implementation
        # that the allocation-bound test compares against; it is not called
        # here.
        r = OR.verify_update_chunked(
            param_before=b["master"], param_after=a["master"], grad=b["grad"],
            exp_avg=zeros, exp_avg_sq=zeros_sq, step=step_index, hparams=hp,
            observed_exp_avg=a["exp_avg"], observed_exp_avg_sq=a["exp_avg_sq"],
            coverage=coverage,
            n_elements_checked=eid["partition_numel"],
            n_elements_total=eid["partition_numel"],
            model_map={"expected_elements": eid["partition_numel"]},
            chunk_elems=chunk_elems)
        r["partition"] = (eid["group"], rank)
        r["expected_identity"] = eid
        results.append(r)
        if r["verification_successful"]:
            successes += 1

    agg = OR.verify_partition(
        results, expected_partitions=[(e["group"], rank) for e in exp_ids])
    nondisc = [r["partition"] for r in results if not r.get("discriminating", True)]
    return {
        "rank": rank, "world_size": world_size,
        "logical_boundaries": boundaries,
        "base_optimizer_calls": base_calls,
        "successful_updates": successes,
        "skipped_updates": skips,
        "accumulation_steps": accumulation_steps,
        "per_partition": results,
        "partition_reconciliation": agg,
        "storage_correspondence": storage_check,
        "module_reflects_masters": module_check,
        "non_discriminating_partitions": nondisc,
        "rank_verification_successful": bool(
            agg["verification_successful"] and not nondisc and skips == 0),
        "coverage_declared": coverage,
        "chunked_verification": {
            "chunk_elems": int(chunk_elems),
            "oracle_entry_point": "update_oracle.verify_update_chunked",
            "snapshot_placement": before.get("snapshot_placement"),
            "snapshot_chunk_elems": before.get("snapshot_chunk_elems"),
            "per_partition_chunks": [r.get("n_chunks") for r in results],
            "per_partition_final_chunk": [r.get("final_chunk_elems")
                                          for r in results],
            "arithmetic_device": "cpu",
            "cuda_bitwise_agreement": "NOT established on CPU",
        },
        # R4. Coverage is an INVENTORY claim: expected unique parameters and
        # elements, expected partitions, and the identities actually observed.
        # Rank count proves nothing by itself. At world size one complete
        # coverage may be possible; it is established by the inventory below
        # matching, never by there being a single rank.
        "coverage_inventory": coverage_inventory(
            expected_ids=exp_ids, results=results, rank=rank,
            world_size=world_size, param_inventory=param_inventory),
        "coverage_is_established_by": ("comparing observed gradient identities "
                                       "against the expected unique parameter "
                                       "and partition inventory"),
        "coverage_is_not_established_by": ("the number of ranks, or by any "
                                          "property of world size one"),
        "coverage_complete": bool(coverage_inventory(
            expected_ids=exp_ids, results=results, rank=rank,
            world_size=world_size, param_inventory=param_inventory)["complete"]),
        "gradient_identity_and_coverage_vs_recomputation": (
            "this maps OBSERVED gradients to the expected parameters and "
            "partitions; it is not an independent recomputation of the "
            "gradient. Active dropout does not remove the mapping "
            "requirement - it is why the observed gradient, and not a "
            "recomputed one, is the object of the comparison."),
        "establishes": ("optimizer-update agreement CONDITIONAL on the observed "
                        "post-clip gradient"),
        "does_not_establish": [
            "gradient correctness (the gradient is observed, not recomputed; "
            "dropout p=0.1 is active in the retained recipe)",
            "training quality or convergence",
            "validity of any historical run",
        ],
    }


def combine_ranks(rank_results: list[dict], *, world_size: int) -> dict:
    """Whole-model verdict across ranks, refusing on any missing rank."""
    if not rank_results:
        raise EP.ProbeRefusal("no rank results supplied")
    ranks = [r["rank"] for r in rank_results]
    dup = sorted({x for x in ranks if ranks.count(x) > 1})
    missing = sorted(set(range(world_size)) - set(ranks))
    unexpected = sorted(set(ranks) - set(range(world_size)))
    failing = [r["rank"] for r in rank_results if not r["rank_verification_successful"]]
    sampled = [r["rank"] for r in rank_results if r["coverage_declared"] != OR.FULL]
    ok = not (dup or missing or unexpected or failing or sampled)
    return {
        "world_size": world_size, "ranks_reported": sorted(ranks),
        "duplicate_ranks": dup, "missing_ranks": missing,
        "unexpected_ranks": unexpected, "failing_ranks": failing,
        "sampled_ranks": sampled,
        "whole_model_verified": bool(ok),
        "coverage_declaration": ("FULL across every expected partition of every "
                                 "rank" if ok else
                                 f"REFUSED: dup={dup} missing={missing} "
                                 f"unexpected={unexpected} failing={failing} "
                                 f"sampled={sampled}"),
        "total_logical_boundaries": sum(r["logical_boundaries"] for r in rank_results),
        "total_base_calls": sum(r["base_optimizer_calls"] for r in rank_results),
        "total_successful_updates": sum(r["successful_updates"] for r in rank_results),
        "total_skipped_updates": sum(r["skipped_updates"] for r in rank_results),
    }
