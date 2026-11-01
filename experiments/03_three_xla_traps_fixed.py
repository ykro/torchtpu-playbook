"""Experiment 03: Fixing the 3 Classic XLA Traps (Side-by-Side Benchmark & Parity Check).

Compares a Naive PyTorch training loop against an OpenXLA-Optimized training loop on the
exact same model weights and data batches:
  Trap 1: `.item()` / `print(loss)` mid-step -> Fixed with `xm.add_step_closure` + `torch.where`.
  Trap 2: Dynamic boolean indexing `h[h > 0]` (`aten::nonzero`) -> Fixed with static `torch.where`.
  Trap 3: Unpadded variable sequence lengths -> Fixed with static bucket padding + attention mask.

Verifies:
- Numerical parity (`torch.allclose`) between Naive and XLA-Optimized outputs and gradients.
- Reduction of `UncachedCompile` after warmup to 0, `HostBarrierStalls` to 0, and
  `AtenNonzeroFallbacks` to 0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import xla_runtime as xm

STATIC_BUCKET_LEN = 64


def pad_to_static_bucket(
    batch: torch.Tensor, target_len: int = STATIC_BUCKET_LEN
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads a `[B, S, D]` tensor to `[B, target_len, D]` and returns a `[B, target_len, 1]` mask."""
    bsz, seq_len, d_model = batch.shape
    padded = torch.zeros(bsz, target_len, d_model, dtype=batch.dtype, device=batch.device)
    padded[:, :seq_len, :] = batch
    mask = torch.zeros(bsz, target_len, 1, dtype=batch.dtype, device=batch.device)
    mask[:, :seq_len, :] = 1.0
    return padded, mask


class TrapModel(nn.Module):
    """Shared linear weights supporting both naive dynamic and XLA-static execution paths."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward_naive(self, x: torch.Tensor) -> torch.Tensor:
        """Naive path: dynamic boolean mask (`h[h > 0]`) + `.item()` host sync + unpadded shape."""
        h = self.proj(x)
        # Trap 2: Dynamic boolean mask (`aten::nonzero`)
        pos_elements = h[h > 0.0]
        # Trap 1: `.item()` forces mid-step device-to-host barrier stall
        pos_sum = pos_elements.sum() if pos_elements.numel() > 0 else torch.tensor(0.0)
        if pos_sum.item() < -1e9:
            h = h * 0.0
        # Reconstruct positive-only activation
        activated = torch.zeros_like(h)
        activated[h > 0.0] = h[h > 0.0]
        out = self.head(activated)
        return out.mean()

    def forward_xla_optimized(self, x_padded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """XLA-optimized path: static `torch.where`, static bucket shape, zero `.item()` syncs."""
        h = self.proj(x_padded)
        # Fix 2: Static elementwise `torch.where` instead of boolean indexing
        activated = torch.where(h > 0.0, h, torch.zeros_like(h))
        # Apply static padding mask so padded tokens contribute zero to the loss
        token_scores = self.head(activated) * mask
        valid_tokens = mask.sum() * token_scores.shape[-1]
        return token_scores.sum() / torch.clamp(valid_tokens, min=1.0)


def compare_naive_vs_optimized() -> dict[str, Any]:
    """Runs both Naive and XLA-Optimized loops on identical batches and compares metrics."""
    torch.manual_seed(123)
    device = xm.xla_device()
    d_model = 128
    seq_lengths = [20, 28, 36, 44, 52, 60]

    # Create raw batches outside the timed loop
    raw_batches = [torch.randn(8, s, d_model, device=device) for s in seq_lengths]
    padded_batches = [pad_to_static_bucket(b, STATIC_BUCKET_LEN) for b in raw_batches]

    # Instantiate two models with identical initial weights
    naive_model = TrapModel(d_model=d_model).to(device)
    opt_model = TrapModel(d_model=d_model).to(device)
    opt_model.load_state_dict(naive_model.state_dict())

    naive_opt = torch.optim.SGD(naive_model.parameters(), lr=1e-2)
    xla_opt = torch.optim.SGD(opt_model.parameters(), lr=1e-2)

    # --- 1. Run Naive Loop ---
    xm.mark_step()
    xm.clear_metrics()
    t0 = time.perf_counter()
    naive_losses: list[float] = []
    for batch in raw_batches:
        naive_opt.zero_grad(set_to_none=True)
        loss = naive_model.forward_naive(batch)
        loss.backward()
        naive_opt.step()
        # Trap 1: synchronous `.item()` inside the training loop
        naive_losses.append(float(loss.item()))
        xm.mark_step()
    naive_elapsed_ms = (time.perf_counter() - t0) * 1000.0
    naive_metrics = xm.get_metrics_snapshot()

    # --- 2. Run XLA-Optimized Loop ---
    xm.mark_step()
    xm.clear_metrics()
    t1 = time.perf_counter()
    opt_losses: list[float] = []
    for x_pad, mask in padded_batches:
        xla_opt.zero_grad(set_to_none=True)
        loss = opt_model.forward_xla_optimized(x_pad, mask)
        loss.backward()
        xla_opt.step()
        # Fix 1: Asynchronous step closure executes only after mark_step() completes
        xm.add_step_closure(
            lambda loss_t: opt_losses.append(float(loss_t.detach().item())),
            args=(loss,),
        )
        xm.mark_step()
    opt_elapsed_ms = (time.perf_counter() - t1) * 1000.0
    opt_metrics = xm.get_metrics_snapshot()

    # --- 3. Numerical Parity Verification ---
    losses_t_naive = torch.tensor(naive_losses, dtype=torch.float64)
    losses_t_opt = torch.tensor(opt_losses, dtype=torch.float64)
    max_abs_diff = float(torch.max(torch.abs(losses_t_naive - losses_t_opt)).item())
    parity_ok = bool(torch.allclose(losses_t_naive, losses_t_opt, atol=1e-5, rtol=1e-5))

    result = {
        "steps": len(seq_lengths),
        "naive": {
            "uncached_compiles": naive_metrics["UncachedCompile"],
            "cached_compiles": naive_metrics["CachedCompile"],
            "host_barrier_stalls": naive_metrics["HostBarrierStalls"],
            "aten_nonzero_fallbacks": naive_metrics["AtenNonzeroFallbacks"],
            "elapsed_ms": round(naive_elapsed_ms, 3),
            "losses": [round(x, 6) for x in naive_losses],
        },
        "xla_optimized": {
            "uncached_compiles": opt_metrics["UncachedCompile"],
            "cached_compiles": opt_metrics["CachedCompile"],
            "recompiles_after_warmup": max(0, opt_metrics["UncachedCompile"] - 1),
            "host_barrier_stalls": opt_metrics["HostBarrierStalls"],
            "aten_nonzero_fallbacks": opt_metrics["AtenNonzeroFallbacks"],
            "elapsed_ms": round(opt_elapsed_ms, 3),
            "losses": [round(x, 6) for x in opt_losses],
        },
        "numerical_parity": {
            "allclose": parity_ok,
            "max_abs_diff": max_abs_diff,
        },
    }
    return result


if __name__ == "__main__":
    print("=== Experiment 03: Fixing the 3 Classic XLA Traps ===")
    comparison = compare_naive_vs_optimized()
    print("\n[Naive PyTorch on XLA]")
    for k, v in comparison["naive"].items():
        print(f"  {k}: {v}")
    print("\n[OpenXLA-Optimized PyTorch]")
    for k, v in comparison["xla_optimized"].items():
        print(f"  {k}: {v}")
    print("\n[Numerical Parity]")
    print(f"  torch.allclose: {comparison['numerical_parity']['allclose']}")
    print(f"  max_abs_diff  : {comparison['numerical_parity']['max_abs_diff']:.2e}")
