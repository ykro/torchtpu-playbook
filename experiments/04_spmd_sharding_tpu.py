"""Experiment 04: OpenXLA GSPMD Sharding Across 4 Devices (`Mesh` + `mark_sharding`).

Demonstrates how a single-program multi-data (SPMD) Transformer MLP block shards
batches and weight matrices across 4 OpenXLA devices:
- Data axis (`data`) shards the batch dimension `[B, S, D]` across devices.
- Model axis (`model`) shards the MLP expansion matrix `[D, F]` column-wise (tensor-parallel)
  and the projection matrix `[F, D]` row-wise so the OpenXLA GSPMD partitioner inserts
  collective `all-reduce` / `all-gather` communication automatically in HLO.

Run with:
  XLA_FLAGS="--xla_force_host_platform_device_count=4" python experiments/04_spmd_sharding_tpu.py
"""

from __future__ import annotations

import os

# Ensure 4 OpenXLA host devices are visible if not already set in the environment
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

try:
    import torch_xla.core.xla_model as _xm  # type: ignore[import-untyped]
    import torch_xla.distributed.spmd as xs  # type: ignore[import-untyped]
    import torch_xla.runtime as xr  # type: ignore[import-untyped]

    HAS_TORCH_XLA_SPMD = True
except ImportError:
    _xm = None
    xs = None
    xr = None
    HAS_TORCH_XLA_SPMD = False


class PyTorchTransformerMLP(nn.Module):
    """Reference PyTorch Transformer MLP block used to verify SPMD numerical equivalence."""

    def __init__(self, d_model: int = 128, d_ff: int = 512) -> None:
        super().__init__()
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w_down(torch.nn.functional.gelu(self.w_up(x), approximate="tanh"))


def run_spmd_sharding_experiment() -> dict[str, Any]:
    """Shards a PyTorch Transformer MLP across a 2x2 OpenXLA device mesh (`data`, `model`).

    When running on Cloud TPU with `torch_xla.distributed.spmd` (`HAS_TORCH_XLA_SPMD`),
    executes native PyTorch/XLA `xs.Mesh` and `xs.mark_sharding`. On local CPU/macOS
    machines without `torch_xla`, uses the shared OpenXLA GSPMD partitioner via 4 host
    PJRT devices (`--xla_force_host_platform_device_count=4`) to verify identical shard
    geometry and numerical equivalence.
    """
    torch.manual_seed(99)
    bsz, seq_len, d_model, d_ff = 16, 32, 128, 512
    pt_model = PyTorchTransformerMLP(d_model=d_model, d_ff=d_ff)
    x_pt = torch.randn(bsz, seq_len, d_model)

    with torch.no_grad():
        ref_out_pt = pt_model(x_pt)

    if HAS_TORCH_XLA_SPMD and xr is not None and xs is not None and _xm is not None:
        xr.use_spmd()
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (2, 2) if num_devices >= 4 else (num_devices, 1)
        device_ids = np.arange(mesh_shape[0] * mesh_shape[1])
        tpu_mesh = xs.Mesh(device_ids, mesh_shape, ("data", "model"))

        device = _xm.xla_device()
        model_xla = PyTorchTransformerMLP(d_model=d_model, d_ff=d_ff).to(device)
        model_xla.load_state_dict(pt_model.state_dict())
        x_xla = x_pt.detach().clone().to(device)

        # Native PyTorch/XLA GSPMD sharding annotations
        xs.mark_sharding(x_xla, tpu_mesh, ("data", None, None))
        xs.mark_sharding(model_xla.w_up.weight, tpu_mesh, ("model", None))
        xs.mark_sharding(model_xla.w_down.weight, tpu_mesh, (None, "model"))

        with torch.no_grad():
            out_xla = model_xla(x_xla)
        _xm.mark_step()
        spmd_out_pt = out_xla.cpu()
        devices = [f"tpu:{i}" for i in range(num_devices)]
        mesh_devices = np.zeros(mesh_shape, dtype=int)
        x_shard_shape = (bsz // mesh_shape[0], seq_len, d_model)
        w_up_shard_shape = (d_model, d_ff // mesh_shape[1])
        w_down_shard_shape = (d_ff // mesh_shape[1], d_model)
    else:
        devices = jax.devices()
        num_devices = len(devices)
        if num_devices >= 4:
            mesh_devices = np.array(devices[:4]).reshape(2, 2)
        else:
            mesh_devices = np.array(devices).reshape(num_devices, 1)

        mesh = Mesh(mesh_devices, axis_names=("data", "model"))

        # Convert PyTorch parameters and activations to OpenXLA PJRT buffers with GSPMD specs
        x_np = x_pt.detach().numpy()
        w_up_np = pt_model.w_up.weight.detach().numpy().T  # [d_model, d_ff]
        w_down_np = pt_model.w_down.weight.detach().numpy().T  # [d_ff, d_model]

        x_sharding = NamedSharding(mesh, P("data", None, None))
        w_up_sharding = NamedSharding(mesh, P(None, "model"))
        w_down_sharding = NamedSharding(mesh, P("model", None))
        out_sharding = NamedSharding(mesh, P("data", None, None))

        x_sharded = jax.device_put(jnp.asarray(x_np), x_sharding)
        w_up_sharded = jax.device_put(jnp.asarray(w_up_np), w_up_sharding)
        w_down_sharded = jax.device_put(jnp.asarray(w_down_np), w_down_sharding)

        @jax.jit(out_shardings=out_sharding)
        def spmd_transformer_mlp(x: jax.Array, w_up: jax.Array, w_down: jax.Array) -> jax.Array:
            hidden = jax.nn.gelu(jnp.einsum("bsd,df->bsf", x, w_up), approximate=True)
            proj = jnp.einsum("bsf,fd->bsd", hidden, w_down)
            return x + proj

        spmd_out = spmd_transformer_mlp(x_sharded, w_up_sharded, w_down_sharded)
        spmd_out.block_until_ready()

        spmd_out_pt = torch.from_numpy(np.array(spmd_out, copy=True))
        x_shard_shape = tuple(x_sharded.addressable_shards[0].data.shape)
        w_up_shard_shape = tuple(w_up_sharded.addressable_shards[0].data.shape)
        w_down_shard_shape = tuple(w_down_sharded.addressable_shards[0].data.shape)

    max_diff = float(torch.max(torch.abs(ref_out_pt - spmd_out_pt)).item())
    allclose = bool(torch.allclose(ref_out_pt, spmd_out_pt, atol=1e-4, rtol=1e-4))

    info = {
        "num_xla_devices": num_devices,
        "mesh_shape": {"data": int(mesh_devices.shape[0]), "model": int(mesh_devices.shape[1])},
        "global_shapes": {
            "x": [bsz, seq_len, d_model],
            "w_up": [d_model, d_ff],
            "w_down": [d_ff, d_model],
        },
        "per_device_shard_shapes": {
            "x_shard": list(x_shard_shape),
            "w_up_shard": list(w_up_shard_shape),
            "w_down_shard": list(w_down_shard_shape),
        },
        "spmd_parity_allclose": allclose,
        "spmd_max_abs_diff": max_diff,
    }

    x_global = [bsz, seq_len, d_model]
    w_up_global = [d_model, d_ff]
    w_down_global = [d_ff, d_model]
    print(f"OpenXLA Devices Visible : {num_devices} ({[str(d) for d in devices]})")
    print(f"GSPMD Mesh Shape        : data={mesh_devices.shape[0]}, model={mesh_devices.shape[1]}")
    print(f"Input x Global -> Shard : {x_global} -> {list(x_shard_shape)} P('data', None, None)")
    print(f"w_up Global    -> Shard : {w_up_global} -> {list(w_up_shard_shape)} P(None, 'model')")
    print(
        f"w_down Global  -> Shard : {w_down_global} -> {list(w_down_shard_shape)} P('model', None)"
    )
    print(f"SPMD Parity allclose    : {allclose} (max_abs_diff={max_diff:.2e})")
    return info


if __name__ == "__main__":
    print("=== Experiment 04: OpenXLA GSPMD Sharding Across 4 Devices ===")
    run_spmd_sharding_experiment()
