"""Experiment 01: The same PyTorch training step, eager first and then `torch.compile`.

TorchTPU keeps the PyTorch contract: the code is device-agnostic, runs eagerly by default
(`torch.device("tpu")`), and a hot function can be compiled with
`torch.compile(fn, backend="tpu")`. TorchDynamo captures the function as an FX graph, and the
TPU backend lowers that graph to StableHLO and compiles it with XLA.

This experiment measures, on whatever device is available:
1. Six eager training steps and six compiled steps from the same initial weights.
2. How many FX graphs reach the backend (1) and how many calls reuse it (5).
3. That eager and compiled losses match step by step.
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


class TinyRegressor(nn.Module):
    """A two-layer MLP: small enough for a laptop, same code path as a large model."""

    def __init__(self, d_in: int = 64, d_hidden: int = 128) -> None:
        super().__init__()
        self.up = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.down = nn.Linear(d_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


def _make_batches(steps: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(7)
    w_true = torch.randn(64, 1, generator=gen)
    batches = []
    for _ in range(steps):
        x = torch.randn(32, 64, generator=gen)
        y = x @ w_true * 0.1
        batches.append((x.to(device), y.to(device)))
    return batches


def _train(
    model: nn.Module, loss_fn: Any, batches: list[tuple[torch.Tensor, torch.Tensor]]
) -> list[float]:
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    losses: list[torch.Tensor] = []
    for x, y in batches:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    # One host transfer at the end, never inside the step.
    return [round(float(v), 6) for v in torch.stack(losses).cpu()]


def run_eager_then_compile(num_steps: int = 6) -> dict[str, Any]:
    quiet_dynamo()
    torch._dynamo.reset()
    device = resolve_device()
    batches = _make_batches(num_steps, device)

    torch.manual_seed(2026)
    eager_model = TinyRegressor().to(device)
    compiled_model = TinyRegressor().to(device)
    compiled_model.load_state_dict(eager_model.state_dict())

    def eager_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((eager_model(x) - y) ** 2).mean()

    def compiled_body(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((compiled_model(x) - y) ** 2).mean()

    recorder = GraphRecorder()
    compiled_loss = compile_counted(compiled_body, recorder)

    eager_losses = _train(eager_model, eager_loss, batches)
    compiled_losses = _train(compiled_model, compiled_loss, batches)

    max_diff = max(abs(a - b) for a, b in zip(eager_losses, compiled_losses))
    graph = recorder.graphs[0]
    result = {
        "device": str(device),
        "backend": backend_label(),
        "steps": num_steps,
        "graphs_compiled": recorder.compiles,
        "calls_reusing_graph": num_steps - recorder.compiles,
        "graph_nodes": graph.num_nodes,
        "graph_ops": graph.ops,
        "eager_losses": eager_losses,
        "compiled_losses": compiled_losses,
        "eager_vs_compiled_max_abs_diff": max_diff,
        "loss_decreased": compiled_losses[-1] < compiled_losses[0],
    }

    print(f"Device  : {result['device']}")
    print(f"Backend : {result['backend']}")
    for i, (e, c) in enumerate(zip(eager_losses, compiled_losses)):
        print(f"[step {i}] eager loss={e:.6f} | compiled loss={c:.6f}")
    print(
        f"Graphs handed to the backend: {recorder.compiles} "
        f"({graph.num_nodes} FX nodes: {', '.join(graph.ops)})"
    )
    print(f"Calls that reused the cached graph: {result['calls_reusing_graph']}")
    print(f"Eager vs compiled max |diff|: {max_diff:.2e}")
    return result


if __name__ == "__main__":
    print("=== Experiment 01: Eager first, then torch.compile on the same step ===")
    run_eager_then_compile()
