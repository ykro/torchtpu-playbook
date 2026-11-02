"""Experiment 02: Hunting graph breaks in a step that looks harmless.

The naive step below runs fine in eager mode and the loss is correct. Under `torch.compile`
it contains two classic problems for a compiler that wants one static graph:
1. `.item()` in the middle of the step: the value has to travel to the host (the CPU that
   runs Python) before the next line can be decided, so Dynamo cuts the graph there.
2. A boolean mask, `h[h > 0]`: the output size depends on the data (`aten::nonzero`), so the
   shape is unknown at compile time and Dynamo cuts the graph again.

`torch._dynamo.explain` reports the graphs and the reasons for every break, and the
`GraphRecorder` backend counts how many graphs reach the backend over five steps. On TorchTPU
both problems are documented as forced synchronizations, and `torch.nonzero` is listed as
unsupported for dynamic shapes.

Measured on CPU: the graph after the mask is compiled again on every step, because the
number of active values changes with the data and the graph was specialized to that size.
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

D_MODEL = 64
D_HIDDEN = 128


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(D_MODEL, D_HIDDEN)
        self.down = nn.Linear(D_HIDDEN, 1)


def naive_step_loss(block: Block, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Correct in eager mode, hostile to a graph compiler."""
    h = torch.relu(block.up(x))
    # Trap 1: a Python decision that needs a value from the device.
    if h.mean().item() > 0.5:
        h = h * 0.5
    # Trap 2: a data-dependent shape.
    active = h[h > 0]
    sparsity_penalty = active.mean() * 1e-3
    pred = block.down(h)
    return ((pred - y) ** 2).mean() + sparsity_penalty


def _classify(reason: str) -> str:
    text = reason.lower()
    if "item()" in text or "_local_scalar_dense" in text:
        return "host sync: Tensor.item()"
    if "nonzero" in text or "dynamic shape" in text:
        return "data-dependent shape: aten::nonzero (boolean mask)"
    return "other"


def audit_naive_step(num_steps: int = 5) -> dict[str, Any]:
    quiet_dynamo()
    torch._dynamo.reset()
    device = resolve_device()
    torch.manual_seed(11)
    block = Block().to(device)
    gen = torch.Generator().manual_seed(3)
    batches = [
        (
            torch.randn(32, D_MODEL, generator=gen).to(device),
            torch.randn(32, 1, generator=gen).to(device),
        )
        for _ in range(num_steps)
    ]

    explanation = torch._dynamo.explain(naive_step_loss)(block, *batches[0])
    reasons = [_classify(str(b.reason)) for b in explanation.break_reasons]

    torch._dynamo.reset()
    recorder = GraphRecorder()
    compiled = compile_counted(naive_step_loss, recorder)
    new_graphs_per_step: list[int] = []
    for x, y in batches:
        before = recorder.compiles
        compiled(block, x, y)
        new_graphs_per_step.append(recorder.compiles - before)
    # Size of the masked tensor that feeds the last graph, step by step.
    masked_sizes = [int((torch.relu(block.up(x)) > 0).sum()) for x, _ in batches]

    eager_val = float(naive_step_loss(block, *batches[0]).detach().cpu())
    compiled_val = float(compiled(block, *batches[0]).detach().cpu())

    result = {
        "device": str(device),
        "backend": backend_label(),
        "graphs_per_call": explanation.graph_count,
        "graph_breaks_per_call": explanation.graph_break_count,
        "break_reasons": reasons,
        "steps": num_steps,
        "graphs_compiled_over_steps": recorder.compiles,
        "new_graphs_per_step": new_graphs_per_step,
        "masked_sizes_per_step": masked_sizes,
        "eager_vs_compiled_abs_diff": abs(eager_val - compiled_val),
    }

    print(f"Device  : {result['device']}")
    print(f"Backend : {result['backend']}")
    print(
        f"torch._dynamo.explain: {explanation.graph_count} graphs, "
        f"{explanation.graph_break_count} graph breaks per call"
    )
    for i, reason in enumerate(reasons, start=1):
        print(f"  break {i}: {reason}")
    print(f"Graphs handed to the backend over {num_steps} steps: {recorder.compiles}")
    print(f"  new graphs per step: {new_graphs_per_step}")
    print(f"  size of h[h > 0] per step: {masked_sizes} (a new shape every step)")
    print(
        f"Loss is still correct (eager vs compiled |diff|): "
        f"{result['eager_vs_compiled_abs_diff']:.2e}"
    )
    return result


if __name__ == "__main__":
    print("=== Experiment 02: Hunting graph breaks with torch._dynamo.explain ===")
    audit_naive_step()
