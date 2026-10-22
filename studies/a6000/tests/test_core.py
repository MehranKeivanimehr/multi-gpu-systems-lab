"""CPU-only tests: config expansion, GPU allowlist, metric math, memory estimator, parameter counts."""
import pytest
import torch

from mgbench.config import cluster_config, expand_experiment, load_yaml, model_registry, validate_gpus, ROOT
from mgbench.metrics import (bus_bandwidth_allreduce, lm_matmul_params, lm_total_params, scaling_efficiency,
                             step_stats)
from mgbench.models import build_causal_lm, load_hf_config_dict
from mgbench.safety import check_feasible, estimate_state_gib

CLUSTER = cluster_config()
REG = model_registry()


def test_allowlist_rejects_other_users_gpus():
    for bad in ([0], [3, 4], [4, 8], [4, 4]):
        with pytest.raises(ValueError):
            validate_gpus(bad, CLUSTER)
    validate_gpus([4, 5, 6, 7], CLUSTER)


def test_every_experiment_stays_on_allowed_gpus():
    for path in (ROOT / "configs/experiments").glob("*.yaml"):
        for spec in expand_experiment(load_yaml(path), CLUSTER):
            assert set(spec["gpus"]) <= set(CLUSTER["allowed_gpus"]), (path, spec["run_id"])
            assert len(spec["gpus"]) == spec["n_gpus"]


def test_train_order_is_small_to_large_and_ids_unique():
    specs = expand_experiment(load_yaml(ROOT / "configs/experiments/train_scaling.yaml"), CLUSTER)
    ns = [s["n_gpus"] for s in specs]
    assert ns == sorted(ns)
    assert len({s["run_id"] for s in specs}) == len(specs)


def test_step_stats():
    s = step_stats([0.1, 0.2, 0.3, 0.4])
    assert s["median_ms"] == pytest.approx(250)
    assert s["min_ms"] == pytest.approx(100) and s["max_ms"] == pytest.approx(400)
    assert s["p10_ms"] == pytest.approx(130)


def test_scaling_efficiency_and_busbw():
    assert scaling_efficiency(380, 100, 4) == pytest.approx(0.95)
    assert scaling_efficiency(190, 100, 4, n_base=2) == pytest.approx(0.95)
    alg, bus = bus_bandwidth_allreduce(1e9, 0.1, 4)
    assert alg == pytest.approx(10) and bus == pytest.approx(15)


@pytest.mark.parametrize("name", ["qwen2.5-0.5b", "qwen2.5-1.5b", "llama3.1-8b"])
def test_param_count_matches_real_model(name):
    hf = load_hf_config_dict(REG[name])
    model = build_causal_lm(REG[name], "meta", torch.float32)
    real = sum(p.numel() for p in model.parameters())  # parameters() de-duplicates tied weights
    assert lm_total_params(hf) == pytest.approx(real, rel=0.002)
    linear = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert lm_matmul_params(hf) == linear


def test_llama_8b_parameter_count_is_8b():
    assert lm_total_params(load_hf_config_dict(REG["llama3.1-8b"])) == pytest.approx(8.03e9, rel=0.01)


def test_estimator_feasibility_for_8b():
    hf = load_hf_config_dict(REG["llama3.1-8b"])
    spec = lambda fw, n: {"kind": "train", "framework": fw, "n_gpus": n}  # noqa: E731
    assert not check_feasible(spec("ddp", 4), hf, CLUSTER)[0]
    assert not check_feasible(spec("zero1", 4), hf, CLUSTER)[0]
    assert not check_feasible(spec("zero2", 4), hf, CLUSTER)[0]
    assert not check_feasible(spec("fsdp", 2), hf, CLUSTER)[0]
    assert check_feasible(spec("fsdp", 4), hf, CLUSTER)[0]
    assert check_feasible(spec("zero3", 4), hf, CLUSTER)[0]


def test_estimator_small_model_fits_everywhere():
    hf = load_hf_config_dict(REG["qwen2.5-1.5b"])
    for fw in ("ddp", "fsdp", "zero1", "zero2", "zero3"):
        assert check_feasible({"kind": "train", "framework": fw, "n_gpus": 1}, hf, CLUSTER)[0]
    assert estimate_state_gib(10**9, "fsdp", 4) == pytest.approx(16e9 / 4 / 2**30)


def test_framework_overrides_merge_without_touching_base():
    from mgbench.config import framework_config, resolved_framework_config

    spec = {"framework": "zero3", "fw_overrides": {"zero3": {"deepspeed": {"communication_data_type": "bf16"}}}}
    fw = resolved_framework_config(spec)
    assert fw["deepspeed"]["communication_data_type"] == "bf16"
    assert fw["deepspeed"]["zero_optimization"]["stage"] == 3
    assert framework_config("zero3")["deepspeed"]["communication_data_type"] == "fp32"
    assert resolved_framework_config({"framework": "fsdp"})["grad_reduce_dtype"] == "fp32"


def test_rerun_configs_expand_as_intended():
    llama = expand_experiment(load_yaml(ROOT / "configs/experiments/llama8b_memory_rerun.yaml"), CLUSTER)
    ids = [s["run_id"] for s in llama]
    assert len(ids) == 8 and sum("cap95__" in i for i in ids) == 4 and sum("mb1" in i for i in ids) == 2
    assert [s["env"] for s in llama if "expseg" in s["run_id"]] == [{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}] * 2
    assert all(s["gpus"] == [4, 5, 6, 7] and s["seq_len"] == 1024 for s in llama)
    assert all(s.get("mem_fraction") == 0.95 for s in llama if "cap95" in s["run_id"])
    assert sum("cap95-expseg" in i for i in ids) == 2
    assert all("mem_fraction" not in s and s["micro_batch"] == 1 for s in llama if "mb1" in s["run_id"])
    vl = expand_experiment(load_yaml(ROOT / "configs/experiments/vllm_tp_concurrency_rerun.yaml"), CLUSTER)
    assert len(vl) == 8 and all(s["max_num_seqs"] == 256 * s["tp"] and s["dp"] == 1 for s in vl)
    assert all(s["env"].get("NCCL_P2P_LEVEL") == "NVL" for s in vl if s["tp"] == 4)


def test_mem_fraction_ceiling_is_enforced():
    exp = load_yaml(ROOT / "configs/experiments/llama8b_memory_rerun.yaml")
    exp["workloads"][0]["mem_fraction"] = 0.99
    with pytest.raises(ValueError):
        expand_experiment(exp, CLUSTER)
