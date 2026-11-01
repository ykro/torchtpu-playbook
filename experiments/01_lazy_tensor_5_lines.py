"""Experiment 01: Moving PyTorch to OpenXLA/TPU in 5 Lines (`LazyTensor` & `mark_step`).

Demonstrates how PyTorch/XLA differs from eager CUDA execution:
1. Operations on `xla_device()` do not launch kernels immediately; they record a Lazy IR graph.
2. Calling `mark_step()` inserts an explicit barrier that cuts the IR graph, compiles it once
   into an OpenXLA HLO program, and executes the entire fused training step on the accelerator.
3. Subsequent steps with identical static shapes hit the OpenXLA cache (`CachedCompile`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import xla_runtime as xm


class ResidualMLP(nn.Module):
    """Compact residual MLP used to trace a realistic training step."""

    def __init__(self, d_model: int = 256, d_ff: int = 512) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.fc2(self.act(self.fc1(x))))


def run_lazy_tensor_demo(
    num_steps: int = 6, batch_size: int = 32, d_model: int = 256
) -> dict[str, int | float]:
    """Runs a 6-step training loop on the OpenXLA device with explicit `mark_step()`."""
    torch.manual_seed(42)
    xm.clear_metrics()

    # Line 1: Acquire the OpenXLA device
    device = xm.xla_device()

    # Line 2: Move model to the XLA device
    model = ResidualMLP(d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()

    # Static input & target tensors on the XLA device
    inputs = torch.randn(batch_size, d_model, device=device)
    targets = torch.zeros(batch_size, d_model, device=device)

    # Flush initial parameter & buffer placement before timing the training loop
    xm.mark_step()
    xm.clear_metrics()

    logged_losses: list[float] = []

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        # Line 3: Inspect pending Lazy IR ops BEFORE mark_step()
        pending_before_barrier = xm.get_metrics_snapshot()["PendingIROps"]

        # Line 4: Register an async host callback so logging never blocks mid-step
        xm.add_step_closure(
            lambda s, loss_t: logged_losses.append(float(loss_t.detach().item())),
            args=(step, loss),
        )

        # Line 5: Cut the Lazy IR graph and dispatch fused OpenXLA execution
        xm.mark_step()

        snap = xm.get_metrics_snapshot()
        print(
            f"[Step {step}] PendingIROps before barrier={pending_before_barrier:3d} | "
            f"Pending after={snap['PendingIROps']} | "
            f"UncachedCompile={snap['UncachedCompile']} | "
            f"CachedCompile={snap['CachedCompile']} | "
            f"Loss={logged_losses[-1]:.6f}"
        )

    final_snap = xm.get_metrics_snapshot()
    return {
        "steps": num_steps,
        "uncached_compiles": final_snap["UncachedCompile"],
        "cached_compiles": final_snap["CachedCompile"],
        "host_stalls": final_snap["HostBarrierStalls"],
        "initial_loss": round(logged_losses[0], 6),
        "final_loss": round(logged_losses[-1], 6),
    }


if __name__ == "__main__":
    print("=== Experiment 01: PyTorch on OpenXLA in 5 Lines (LazyTensor + mark_step) ===")
    summary = run_lazy_tensor_demo()
    print("\n--- OpenXLA Metrics Report ---")
    print(xm.metrics_report())
    print("\nSummary:", summary)
