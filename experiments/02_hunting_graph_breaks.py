"""Experiment 02: Hunting Graph Breaks & Silent Eager Fallbacks with XLA Metrics.

Instruments a naive PyTorch model that looks idiomatic on CUDA GPUs but triggers
three severe OpenXLA performance anti-patterns on TPU:
1. Mid-step `.item()` evaluation inside a conditional / print statement (`HostBarrierStalls`).
2. Dynamic boolean masking `x[x > 0]` triggering `aten::nonzero` CPU fallback.
3. Variable sequence lengths across steps triggering `UncachedCompile` every step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import xla_runtime as xm


class NaiveFeatureScorer(nn.Module):
    """A model with hidden XLA pitfalls commonly written by PyTorch GPU engineers."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        # Pitfall A: Data-dependent boolean indexing (`aten::nonzero`)
        positive_vals = h[h > 0.0]
        # Pitfall B: `.item()` inside forward forces immediate Device -> Host transfer
        if positive_vals.numel() > 0 and positive_vals.mean().item() > 100.0:
            h = h * 0.5
        h_relu = torch.relu(h)
        return self.head(h_relu).mean()


def audit_naive_training_loop() -> dict[str, Any]:
    """Runs 5 steps with variable sequence lengths and audits the XLA metrics report."""
    torch.manual_seed(7)
    xm.clear_metrics()
    device = xm.xla_device()

    model = NaiveFeatureScorer(d_model=128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Flush model init
    xm.mark_step()
    xm.clear_metrics()

    # Pitfall C: Unpadded variable sequence lengths per step
    variable_seq_lengths = [19, 27, 33, 41, 52]

    for step, seq_len in enumerate(variable_seq_lengths):
        optimizer.zero_grad(set_to_none=True)
        batch = torch.randn(8, seq_len, 128, device=device)
        loss = model(batch)
        loss.backward()
        optimizer.step()

        # Pitfall A again: printing `loss.item()` before `mark_step()`
        loss_val = loss.item()
        xm.mark_step()

        snap = xm.get_metrics_snapshot()
        print(
            f"[Naive Step {step} | seq_len={seq_len}] loss={loss_val:.5f} | "
            f"UncachedCompile={snap['UncachedCompile']} | "
            f"HostStalls={snap['HostBarrierStalls']} | "
            f"AtenNonzero={snap['AtenNonzeroFallbacks']}"
        )

    final_snap = xm.get_metrics_snapshot()
    return {
        "steps_executed": len(variable_seq_lengths),
        "uncached_compiles": final_snap["UncachedCompile"],
        "cached_compiles": final_snap["CachedCompile"],
        "host_barrier_stalls": final_snap["HostBarrierStalls"],
        "aten_nonzero_fallbacks": final_snap["AtenNonzeroFallbacks"],
    }


if __name__ == "__main__":
    print("=== Experiment 02: Hunting Graph Breaks with OpenXLA Metrics ===")
    audit = audit_naive_training_loop()
    print("\n--- Diagnostic OpenXLA Metrics Report (Naive Loop) ---")
    print(xm.metrics_report())
    print("\nDiagnostic Audit Summary:", audit)
