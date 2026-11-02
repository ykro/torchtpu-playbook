"""Experiment 03: Fixing the three traps, always in the same order.

1. `.item()` in the middle of the step -> keep the value as a tensor and decide on the device
   with `torch.where`; read losses on the host once, after the loop.
2. Boolean mask `h[h > 0]` -> `torch.where(h > 0, h, 0)` plus a count, so the shape never
   depends on the data.
3. Variable sequence lengths -> pad every batch to a static bucket of 64 positions and carry a
   mask. TorchTPU offers its own version of this idea (bounded dynamism with buckets,
   `torch._dynamo.mark_dynamic(..., min=..., max=...)`); the manual bucket shown here works
   with any graph compiler and runs on a laptop.

Both versions see the same six batches (lengths 20, 33, 41, 52, 60, 20) and must produce the
same loss. The judge counts graph breaks and graphs handed to the backend.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch._dynamo
import torch.nn as nn

from experiments.tpu_backend import (
    GraphRecorder,
    backend_label,
    compile_counted,
    quiet_dynamo,
    resolve_device,
)

D_MODEL = 32
D_HIDDEN = 64
STATIC_BUCKET_LEN = 64
SEQ_LENGTHS = [20, 33, 41, 52, 60, 20]
BATCH = 4


class SeqBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(D_MODEL, D_HIDDEN)
        self.down = nn.Linear(D_HIDDEN, 1)


def naive_loss(block: SeqBlock, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x: (B, L, D) with a different L per batch. Three traps, in order."""
    h = torch.relu(block.up(x))
    if h.mean().item() > 0.5:  # trap 1: host sync to decide in Python
        h = h * 0.5
    active = h[h > 0]  # trap 2: data-dependent shape
    penalty = active.mean() * 1e-3
    pred = block.down(h).squeeze(-1)  # trap 3: L changes from batch to batch
    return ((pred - y) ** 2).mean() + penalty


def fixed_loss(
    block: SeqBlock, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """x: (B, 64, D) padded, mask: (B, 64) with 1 on real positions. Same math."""
    m = mask.unsqueeze(-1)
    h = torch.relu(block.up(x)) * m
    valid = m.sum() * D_HIDDEN
    mean_h = h.sum() / valid
    h = torch.where(mean_h > 0.5, h * 0.5, h)  # fix 1: decide on the device
    positive = (h > 0) & (m > 0)
    penalty = torch.where(positive, h, torch.zeros_like(h)).sum() / positive.sum()  # fix 2
    pred = block.down(h).squeeze(-1)
    sq = ((pred - y) ** 2) * mask  # fix 3: padded positions contribute zero
    return sq.sum() / mask.sum() + penalty * 1e-3


def _make_batches(device: torch.device) -> list[dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(19)
    batches = []
    for length in SEQ_LENGTHS:
        x = torch.randn(BATCH, length, D_MODEL, generator=gen)
        y = torch.randn(BATCH, length, generator=gen)
        pad = STATIC_BUCKET_LEN - length
        x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad))
        y_pad = torch.nn.functional.pad(y, (0, pad))
        mask = torch.zeros(BATCH, STATIC_BUCKET_LEN)
        mask[:, :length] = 1.0
        batches.append(
            {
                "x": x.to(device),
                "y": y.to(device),
                "x_pad": x_pad.to(device),
                "y_pad": y_pad.to(device),
                "mask": mask.to(device),
            }
        )
    return batches


def compare_naive_vs_fixed() -> dict[str, Any]:
    quiet_dynamo()
    device = resolve_device()
    torch.manual_seed(5)
    block = SeqBlock().to(device)
    batches = _make_batches(device)

    torch._dynamo.reset()
    naive_rec = GraphRecorder()
    naive_c = compile_counted(naive_loss, naive_rec)
    naive_losses = [naive_c(block, b["x"], b["y"]).detach() for b in batches]

    torch._dynamo.reset()
    fixed_rec = GraphRecorder()
    fixed_c = compile_counted(fixed_loss, fixed_rec)
    fixed_new: list[int] = []
    fixed_losses = []
    for b in batches:
        before = fixed_rec.compiles
        fixed_losses.append(fixed_c(block, b["x_pad"], b["y_pad"], b["mask"]).detach())
        fixed_new.append(fixed_rec.compiles - before)

    # explain() runs after the counted runs: it shares Dynamo's caches with torch.compile.
    torch._dynamo.reset()
    naive_expl = torch._dynamo.explain(naive_loss)(block, batches[0]["x"], batches[0]["y"])
    torch._dynamo.reset()
    b0 = batches[0]
    fixed_expl = torch._dynamo.explain(fixed_loss)(block, b0["x_pad"], b0["y_pad"], b0["mask"])

    torch._dynamo.reset()

    naive_t = torch.stack(naive_losses).cpu()
    fixed_t = torch.stack(fixed_losses).cpu()
    max_diff = float((naive_t - fixed_t).abs().max())

    result = {
        "device": str(device),
        "backend": backend_label(),
        "seq_lengths": SEQ_LENGTHS,
        "static_bucket_len": STATIC_BUCKET_LEN,
        "naive": {
            "graphs_per_call": naive_expl.graph_count,
            "graph_breaks_per_call": naive_expl.graph_break_count,
            "graphs_compiled_over_batches": naive_rec.compiles,
        },
        "fixed": {
            "graphs_per_call": fixed_expl.graph_count,
            "graph_breaks_per_call": fixed_expl.graph_break_count,
            "graphs_compiled_over_batches": fixed_rec.compiles,
            "new_graphs_per_batch": fixed_new,
            "recompiles_after_warmup": sum(fixed_new[1:]),
        },
        "padding_overhead_shortest_batch": round(STATIC_BUCKET_LEN / min(SEQ_LENGTHS), 2),
        "numerical_parity": {
            "naive_losses": [round(float(v), 6) for v in naive_t],
            "fixed_losses": [round(float(v), 6) for v in fixed_t],
            "max_abs_diff": max_diff,
            "allclose": bool(torch.allclose(naive_t, fixed_t, atol=1e-5, rtol=1e-5)),
        },
    }

    print(f"Device  : {result['device']}")
    print(f"Backend : {result['backend']}")
    print(f"Sequence lengths per batch: {SEQ_LENGTHS} -> static bucket {STATIC_BUCKET_LEN}")
    print(
        f"[naive] {naive_expl.graph_count} graphs and {naive_expl.graph_break_count} breaks "
        f"per call; {naive_rec.compiles} graphs compiled over {len(batches)} batches"
    )
    print(
        f"[fixed] {fixed_expl.graph_count} graph and {fixed_expl.graph_break_count} breaks "
        f"per call; {fixed_rec.compiles} graph compiled over {len(batches)} batches "
        f"(new graphs per batch: {fixed_new})"
    )
    print(f"Recompiles after warmup (fixed): {result['fixed']['recompiles_after_warmup']}")
    print(
        f"Padding cost: the batch of length {min(SEQ_LENGTHS)} is processed with "
        f"{STATIC_BUCKET_LEN} positions ({result['padding_overhead_shortest_batch']}x)"
    )
    print(
        f"Loss parity naive vs fixed: allclose={result['numerical_parity']['allclose']} "
        f"(max |diff|={max_diff:.2e})"
    )
    return result


if __name__ == "__main__":
    print("=== Experiment 03: The three traps fixed (item -> mask -> lengths) ===")
    compare_naive_vs_fixed()
