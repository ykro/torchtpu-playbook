"""Experiment 04: Declared sharding with `DeviceMesh` + DTensor on a 2x2 mesh ('data', 'model').

TorchTPU supports `torch.distributed` as is: the process group is called `tpu_dist`, the mesh
is a regular `DeviceMesh("tpu", ...)` and DTensor/FSDP2 run on top (per the TorchTPU
distributed docs). The same program runs on a laptop with four CPU processes and the `gloo`
backend, which is what this experiment does:

1. Four processes form a `DeviceMesh` of shape (2, 2) named ('data', 'model').
2. The batch is split over 'data'. The MLP is split over 'model' with the Megatron pattern:
   the first linear by columns (`ColwiseParallel`) and the second by rows (`RowwiseParallel`).
3. `CommDebugMode` counts the collectives that DTensor inserts in the forward pass.
4. Every rank compares its output against the unsharded model on the same slice.

On a TPU VM the changes would be `backend="tpu_dist"` and `DeviceMesh("tpu", ...)`; that path
is in `gcp/run_torchtpu_v6e.sh` and was not run.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

WORLD_SIZE = 4
MESH_SHAPE = (2, 2)
MESH_DIMS = ("data", "model")
BATCH, SEQ, D_MODEL, D_FF = 16, 32, 128, 512


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(D_MODEL, D_FF, bias=False)
        self.down = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


def _worker(rank: int, init_file: str, out_dir: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.debug import CommDebugMode
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    use_tpu = os.environ.get("TORCHTPU_DEVICE") == "tpu"
    device_type = "tpu" if use_tpu else "cpu"
    backend = "tpu_dist" if use_tpu else "gloo"
    dist.init_process_group(
        backend, init_method=f"file://{init_file}", rank=rank, world_size=WORLD_SIZE
    )
    mesh = init_device_mesh(device_type, MESH_SHAPE, mesh_dim_names=MESH_DIMS)
    dp_rank = mesh["data"].get_local_rank()
    tp_rank = mesh["model"].get_local_rank()

    torch.manual_seed(2026)
    reference = MLP()
    sharded = MLP()
    sharded.load_state_dict(reference.state_dict())
    sharded = parallelize_module(
        sharded, mesh["model"], {"up": ColwiseParallel(), "down": RowwiseParallel()}
    )

    gen = torch.Generator().manual_seed(7)
    x_global = torch.randn(BATCH, SEQ, D_MODEL, generator=gen)
    per_dp = BATCH // MESH_SHAPE[0]
    x_local = x_global[dp_rank * per_dp : (dp_rank + 1) * per_dp].to(device_type)

    comm = CommDebugMode()
    with comm, torch.no_grad():
        out = sharded(x_local)
    fwd_counts = {str(k).split(".")[-1]: v for k, v in comm.get_comm_counts().items()}

    ref_out = reference(x_global[dp_rank * per_dp : (dp_rank + 1) * per_dp])
    out_cpu = out.detach().cpu()
    max_diff = float((out_cpu - ref_out.detach()).abs().max())

    record = {
        "rank": rank,
        "data_rank": dp_rank,
        "model_rank": tp_rank,
        "x_local_shape": list(x_local.shape),
        "up_weight_local_shape": list(sharded.up.weight.to_local().shape),
        "down_weight_local_shape": list(sharded.down.weight.to_local().shape),
        "up_weight_global_shape": list(sharded.up.weight.shape),
        "down_weight_global_shape": list(sharded.down.weight.shape),
        "forward_collectives": fwd_counts,
        "max_abs_diff_vs_unsharded": max_diff,
        "allclose": bool(torch.allclose(out_cpu, ref_out.detach(), atol=1e-5, rtol=1e-5)),
    }
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    dist.barrier()
    dist.destroy_process_group()


def run_dtensor_sharding() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        mp.spawn(_worker, args=(init_file, tmp), nprocs=WORLD_SIZE, join=True)
        ranks = []
        for r in range(WORLD_SIZE):
            with open(os.path.join(tmp, f"rank{r}.json"), encoding="utf-8") as fh:
                ranks.append(json.load(fh))

    r0 = ranks[0]
    result = {
        "process_group_backend": "tpu_dist"
        if os.environ.get("TORCHTPU_DEVICE") == "tpu"
        else "gloo (4 CPU processes)",
        "world_size": WORLD_SIZE,
        "mesh_shape": dict(zip(MESH_DIMS, MESH_SHAPE)),
        "global_shapes": {
            "x": [BATCH, SEQ, D_MODEL],
            "up_weight": r0["up_weight_global_shape"],
            "down_weight": r0["down_weight_global_shape"],
        },
        "per_rank_shapes": {
            "x": r0["x_local_shape"],
            "up_weight": r0["up_weight_local_shape"],
            "down_weight": r0["down_weight_local_shape"],
        },
        "forward_collectives": r0["forward_collectives"],
        "max_abs_diff_vs_unsharded": max(r["max_abs_diff_vs_unsharded"] for r in ranks),
        "all_ranks_allclose": all(r["allclose"] for r in ranks),
        "ranks": [
            {"rank": r["rank"], "data": r["data_rank"], "model": r["model_rank"]} for r in ranks
        ],
    }

    print(f"Process group : {result['process_group_backend']}")
    print(f"Mesh          : {result['mesh_shape']}")
    for r in result["ranks"]:
        print(f"  rank {r['rank']} -> data={r['data']}, model={r['model']}")
    g, p = result["global_shapes"], result["per_rank_shapes"]
    print(f"x           global {g['x']} -> per rank {p['x']} (split over 'data')")
    print(f"up.weight   global {g['up_weight']} -> per rank {p['up_weight']} (columns, 'model')")
    print(f"down.weight global {g['down_weight']} -> per rank {p['down_weight']} (rows, 'model')")
    print(f"Collectives in forward: {result['forward_collectives']}")
    print(
        f"Every rank matches the unsharded model: {result['all_ranks_allclose']} "
        f"(max |diff|={result['max_abs_diff_vs_unsharded']:.2e})"
    )
    return result


if __name__ == "__main__":
    print("=== Experiment 04: DeviceMesh + DTensor, Megatron-style MLP on a 2x2 mesh ===")
    run_dtensor_sharding()
