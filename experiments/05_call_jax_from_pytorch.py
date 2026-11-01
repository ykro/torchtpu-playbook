"""Experiment 05: Calling Compiled `@jax.jit` Kernels Inside PyTorch Autograd via DLPack.

Demonstrates zero-copy interoperability between PyTorch and JAX (`torch_xla2` / DLPack pattern):
1. Converts `torch.Tensor` to `jax.Array` without copying memory via `torch.utils.dlpack.to_dlpack`
   and `jax.dlpack.from_dlpack`.
2. Executes a fused `@jax.jit` RMSNorm + SwiGLU kernel and captures its pullback with `jax.vjp`.
3. Wraps both forward and backward passes inside a custom `torch.autograd.Function` so a
   standard `torch.nn.Module` can train seamlessly with `loss.backward()` and `torch.optim.AdamW`.
4. Verifies exact forward and backward gradient parity against a pure PyTorch reference.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.dlpack
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.utils.dlpack


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Zero-copy conversion from detached contiguous PyTorch tensor to JAX Array via DLPack."""
    contig = t.detach().contiguous()
    return jax.dlpack.from_dlpack(contig)


def _jax_to_torch(a: jax.Array) -> torch.Tensor:
    """Zero-copy conversion from JAX Array to PyTorch tensor via DLPack."""
    return torch.utils.dlpack.from_dlpack(a)


@jax.jit
def _jax_fused_rmsnorm_swiglu(
    x: jax.Array,
    rms_weight: jax.Array,
    w_gate: jax.Array,
    w_val: jax.Array,
) -> jax.Array:
    """Fused OpenXLA kernel computing RMSNorm(x) followed by SwiGLU projection."""
    eps = 1e-6
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_norm = x * jax.lax.rsqrt(variance + eps) * rms_weight
    gate = jax.nn.silu(jnp.matmul(x_norm, w_gate))
    val = jnp.matmul(x_norm, w_val)
    return gate * val


@jax.jit
def _jax_fused_fwd_with_vjp(
    x: jax.Array,
    rms_weight: jax.Array,
    w_gate: jax.Array,
    w_val: jax.Array,
) -> tuple[jax.Array, Any]:
    """Executes the forward pass and returns the primal output plus the compiled VJP pullback."""
    return jax.vjp(_jax_fused_rmsnorm_swiglu, x, rms_weight, w_gate, w_val)


class JAXFusedSwiGLUFunction(torch.autograd.Function):
    """Bridges PyTorch autograd with a `@jax.jit` + `jax.vjp` OpenXLA kernel via DLPack."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        rms_weight: torch.Tensor,
        w_gate: torch.Tensor,
        w_val: torch.Tensor,
    ) -> torch.Tensor:
        out_jax, vjp_fn = _jax_fused_fwd_with_vjp(
            _torch_to_jax(x),
            _torch_to_jax(rms_weight),
            _torch_to_jax(w_gate),
            _torch_to_jax(w_val),
        )
        ctx.vjp_fn = jax.jit(vjp_fn)
        return _jax_to_torch(out_jax)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dx, d_rms, d_gate, d_val = ctx.vjp_fn(_torch_to_jax(grad_out))
        return (
            _jax_to_torch(dx),
            _jax_to_torch(d_rms),
            _jax_to_torch(d_gate),
            _jax_to_torch(d_val),
        )


class HybridPyTorchJAXBlock(nn.Module):
    """A standard `torch.nn.Module` whose hot inner kernel is executed by `@jax.jit`."""

    def __init__(self, d_model: int = 128, d_ff: int = 256) -> None:
        super().__init__()
        self.rms_weight = nn.Parameter(torch.ones(d_model))
        self.w_gate = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
        self.w_val = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
        self.out_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        fused = JAXFusedSwiGLUFunction.apply(x, self.rms_weight, self.w_gate, self.w_val)
        return x + self.out_proj(fused)

    def forward_pure_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps) * self.rms_weight
        gate = torch.nn.functional.silu(x_norm @ self.w_gate)
        val = x_norm @ self.w_val
        return x + self.out_proj(gate * val)


def run_pytorch_jax_bridge_demo() -> dict[str, Any]:
    """Verifies forward & backward gradient parity and runs a 5-step hybrid training loop."""
    torch.manual_seed(2026)
    d_model, d_ff = 128, 256

    hybrid_block = HybridPyTorchJAXBlock(d_model=d_model, d_ff=d_ff)
    ref_block = HybridPyTorchJAXBlock(d_model=d_model, d_ff=d_ff)
    ref_block.load_state_dict(hybrid_block.state_dict())

    x_hybrid = torch.randn(8, 32, d_model, requires_grad=True)
    x_ref = x_hybrid.detach().clone().requires_grad_(True)

    # Forward & Backward pass through JAX bridge vs Pure PyTorch
    out_hybrid = hybrid_block.forward_hybrid(x_hybrid)
    loss_hybrid = out_hybrid.pow(2).mean()
    loss_hybrid.backward()

    out_ref = ref_block.forward_pure_pytorch(x_ref)
    loss_ref = out_ref.pow(2).mean()
    loss_ref.backward()

    fwd_diff = float(torch.max(torch.abs(out_hybrid - out_ref)).item())
    grad_x_diff = float(torch.max(torch.abs(x_hybrid.grad - x_ref.grad)).item())  # type: ignore[arg-type]
    grad_w_gate_diff = float(
        torch.max(torch.abs(hybrid_block.w_gate.grad - ref_block.w_gate.grad)).item()  # type: ignore[arg-type]
    )

    fwd_allclose = bool(torch.allclose(out_hybrid, out_ref, atol=1e-5, rtol=1e-5))
    bwd_allclose = bool(
        torch.allclose(x_hybrid.grad, x_ref.grad, atol=1e-5, rtol=1e-5)  # type: ignore[arg-type]
        and torch.allclose(
            hybrid_block.w_gate.grad,
            ref_block.w_gate.grad,
            atol=1e-5,
            rtol=1e-5,  # type: ignore[arg-type]
        )
    )

    # Run a 5-step training loop using standard PyTorch AdamW over the hybrid module
    optimizer = torch.optim.AdamW(hybrid_block.parameters(), lr=5e-3)
    target = torch.zeros_like(out_hybrid.detach())
    step_losses: list[float] = []

    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        pred = hybrid_block.forward_hybrid(x_hybrid.detach())
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        step_losses.append(round(float(loss.item()), 6))
        print(f"[Hybrid PyTorch+JAX Step {step}] loss={step_losses[-1]:.6f}")

    summary = {
        "fwd_allclose": fwd_allclose,
        "bwd_allclose": bwd_allclose,
        "fwd_max_abs_diff": fwd_diff,
        "grad_x_max_abs_diff": grad_x_diff,
        "grad_w_gate_max_abs_diff": grad_w_gate_diff,
        "training_losses": step_losses,
    }
    print(f"\nForward Parity allclose : {fwd_allclose} (max_abs_diff={fwd_diff:.2e})")
    print(
        f"Backward Parity allclose: {bwd_allclose} "
        f"(grad_x_diff={grad_x_diff:.2e}, grad_w_gate_diff={grad_w_gate_diff:.2e})"
    )
    return summary


if __name__ == "__main__":
    print("=== Experiment 05: Zero-Copy PyTorch <-> JAX Autograd Bridge (DLPack + jax.vjp) ===")
    run_pytorch_jax_bridge_demo()
