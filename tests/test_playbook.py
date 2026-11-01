"""Pytest suite for `pytorch-xla-tpu-playbook`."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import importlib

exp01 = importlib.import_module("experiments.01_lazy_tensor_5_lines")
exp02 = importlib.import_module("experiments.02_hunting_graph_breaks")
exp03 = importlib.import_module("experiments.03_three_xla_traps_fixed")
exp04 = importlib.import_module("experiments.04_spmd_sharding_tpu")
exp05 = importlib.import_module("experiments.05_call_jax_from_pytorch")


def test_exp01_lazy_tensor_single_compile() -> None:
    res = exp01.run_lazy_tensor_demo(num_steps=6)
    assert res["uncached_compiles"] == 1
    assert res["cached_compiles"] == 5
    assert res["host_stalls"] == 0
    assert res["final_loss"] < res["initial_loss"]


def test_exp02_hunting_graph_breaks_detects_all_traps() -> None:
    audit = exp02.audit_naive_training_loop()
    assert audit["uncached_compiles"] >= 5
    assert audit["host_barrier_stalls"] >= 5
    assert audit["aten_nonzero_fallbacks"] >= 5


def test_exp03_three_traps_fixed_zero_recompiles_and_parity() -> None:
    comp = exp03.compare_naive_vs_optimized()
    assert comp["xla_optimized"]["uncached_compiles"] == 1
    assert comp["xla_optimized"]["cached_compiles"] == 5
    assert comp["xla_optimized"]["recompiles_after_warmup"] == 0
    assert comp["xla_optimized"]["host_barrier_stalls"] == 0
    assert comp["xla_optimized"]["aten_nonzero_fallbacks"] == 0
    assert comp["numerical_parity"]["allclose"] is True
    assert comp["numerical_parity"]["max_abs_diff"] < 1e-5


def test_exp04_spmd_4_device_sharding() -> None:
    info = exp04.run_spmd_sharding_experiment()
    assert info["num_xla_devices"] == 4
    assert info["mesh_shape"] == {"data": 2, "model": 2}
    assert info["per_device_shard_shapes"]["x_shard"] == [8, 32, 128]
    assert info["per_device_shard_shapes"]["w_up_shard"] == [128, 256]
    assert info["per_device_shard_shapes"]["w_down_shard"] == [256, 128]
    assert info["spmd_parity_allclose"] is True


def test_exp05_pytorch_jax_dlpack_autograd_bridge() -> None:
    summary = exp05.run_pytorch_jax_bridge_demo()
    assert summary["fwd_allclose"] is True
    assert summary["bwd_allclose"] is True
    assert summary["fwd_max_abs_diff"] < 1e-5
    assert summary["grad_x_max_abs_diff"] < 1e-5
    assert summary["training_losses"][-1] < summary["training_losses"][0]
