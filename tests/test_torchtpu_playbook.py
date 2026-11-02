"""Pytest suite for `torchtpu-playbook`."""

from __future__ import annotations

import importlib

exp01 = importlib.import_module("experiments.01_eager_then_compile")
exp02 = importlib.import_module("experiments.02_hunting_graph_breaks")
exp03 = importlib.import_module("experiments.03_three_traps_fixed")
exp04 = importlib.import_module("experiments.04_dtensor_mesh_sharding")
exp05 = importlib.import_module("experiments.05_torch_jax_dlpack_stablehlo")


def test_exp01_one_graph_and_parity() -> None:
    r = exp01.run_eager_then_compile(num_steps=6)
    assert r["graphs_compiled"] == 1
    assert r["calls_reusing_graph"] == 5
    assert r["eager_vs_compiled_max_abs_diff"] == 0
    assert r["loss_decreased"]


def test_exp02_finds_item_and_mask_breaks() -> None:
    r = exp02.audit_naive_step(num_steps=5)
    assert r["graphs_per_call"] == 3
    assert r["graph_breaks_per_call"] == 2
    assert "host sync: Tensor.item()" in r["break_reasons"]
    assert all(n > 0 for n in r["new_graphs_per_step"])
    assert r["eager_vs_compiled_abs_diff"] < 1e-6


def test_exp03_fixed_compiles_once_with_parity() -> None:
    r = exp03.compare_naive_vs_fixed()
    assert r["fixed"]["graph_breaks_per_call"] == 0
    assert r["fixed"]["graphs_compiled_over_batches"] == 1
    assert r["fixed"]["recompiles_after_warmup"] == 0
    assert r["naive"]["graphs_compiled_over_batches"] > 6
    assert r["numerical_parity"]["allclose"]
    assert r["numerical_parity"]["max_abs_diff"] < 1e-5


def test_exp04_dtensor_megatron_mlp() -> None:
    r = exp04.run_dtensor_sharding()
    assert r["mesh_shape"] == {"data": 2, "model": 2}
    assert r["per_rank_shapes"]["up_weight"] == [256, 128]
    assert r["per_rank_shapes"]["down_weight"] == [128, 256]
    assert r["forward_collectives"] == {"all_reduce": 1}
    assert r["all_ranks_allclose"]


def test_exp05_dlpack_bridge_and_stablehlo() -> None:
    r = exp05.run_pytorch_jax_bridge_demo()
    assert r["fwd_allclose"] and r["bwd_allclose"]
    assert all(r["zero_copy"].values())
    assert r["stablehlo"]["dialect_is_stablehlo"]
    assert r["stablehlo"]["dot_general"] == 2
    assert r["training_losses"][-1] < r["training_losses"][0]
