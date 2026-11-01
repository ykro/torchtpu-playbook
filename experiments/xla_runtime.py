"""OpenXLA Lazy-Tensor Runtime & Metrics Auditor for PyTorch.

On Cloud TPU VMs where `torch_xla` is installed, delegates to `torch_xla.core.xla_model`
and `torch_xla.debug.metrics`. On local machines (macOS/Linux CPU with JAX OpenXLA PJRT),
provides a deterministic `TorchFunctionMode` tracer that intercepts PyTorch operations,
tracks lazy IR graph accumulation, detects `.item()` host-sync stalls, catches dynamic
shape ops (`aten::nonzero`, `aten::masked_select` from boolean masks), and records
OpenXLA HLO compilation cache hits and misses (`CachedCompile` vs `UncachedCompile`).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.overrides import TorchFunctionMode

try:
    import torch_xla.core.xla_model as _xm  # type: ignore[import-untyped]
    import torch_xla.debug.metrics as _met  # type: ignore[import-untyped]

    HAS_NATIVE_TORCH_XLA = True
except ImportError:
    _xm = None
    _met = None
    HAS_NATIVE_TORCH_XLA = False


@dataclass
class XLAMetricsState:
    """Tracks OpenXLA compilation, fallback, and barrier counters."""

    cached_compile: int = 0
    uncached_compile: int = 0
    mark_step_count: int = 0
    host_barrier_stalls: int = 0
    aten_nonzero_fallbacks: int = 0
    ir_ops_pending: int = 0
    total_ir_ops_traced: int = 0
    compiled_graph_signatures: set[str] = field(default_factory=set)
    current_step_shapes: list[str] = field(default_factory=list)
    step_callbacks: list[tuple[Any, tuple[Any, ...]]] = field(default_factory=list)
    in_internal_audit: bool = False

    def reset(self) -> None:
        self.cached_compile = 0
        self.uncached_compile = 0
        self.mark_step_count = 0
        self.host_barrier_stalls = 0
        self.aten_nonzero_fallbacks = 0
        self.ir_ops_pending = 0
        self.total_ir_ops_traced = 0
        self.compiled_graph_signatures.clear()
        self.current_step_shapes.clear()
        self.step_callbacks.clear()
        self.in_internal_audit = False


_STATE = XLAMetricsState()

# Forward/backward compute ops whose input shapes define the OpenXLA HLO program signature
_SIGNATURE_OPS = {
    "linear",
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "gelu",
    "relu",
    "silu",
    "layer_norm",
    "where",
    "mse_loss",
    "sum",
    "mean",
    "clamp",
}


class _LazyTensorTracerMode(TorchFunctionMode):
    """Intercepts PyTorch ops to record OpenXLA lazy IR nodes and detect graph breaks."""

    def __torch_function__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if _STATE.in_internal_audit:
            return func(*args, **kwargs)

        func_name = getattr(func, "__name__", str(func))

        # Detect host synchronization stalls (.item(), .tolist()) outside internal torch/optim
        if func_name in ("item", "tolist"):
            caller_file = sys._getframe(2).f_code.co_filename.replace("\\", "/")
            if "/torch/optim/" not in caller_file:
                _STATE.host_barrier_stalls += 1
                _flush_pending_ir(triggered_by_host_stall=True)
            return func(*args, **kwargs)

        # Detect dynamic boolean masking / nonzero fallbacks
        if func_name in ("nonzero", "masked_select"):
            _STATE.aten_nonzero_fallbacks += 1
            _STATE.current_step_shapes.append(f"dynamic_fallback:{func_name}")
        elif func_name == "__getitem__" and len(args) >= 2:
            idx = args[1]
            if isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
                _STATE.aten_nonzero_fallbacks += 1
                _STATE.in_internal_audit = True
                try:
                    true_count = int(idx.detach().sum().item())
                finally:
                    _STATE.in_internal_audit = False
                _STATE.current_step_shapes.append(f"bool_mask_out[{true_count}]")

        # Record compute tensor shapes entering the lazy HLO graph
        if func_name in _SIGNATURE_OPS:
            arg_shapes = [
                f"{tuple(a.shape)}:{a.dtype}" for a in args if isinstance(a, torch.Tensor)
            ]
            _STATE.current_step_shapes.append(f"{func_name}({','.join(arg_shapes)})")

        _STATE.ir_ops_pending += 1
        _STATE.total_ir_ops_traced += 1

        return func(*args, **kwargs)


_ACTIVE_MODE: _LazyTensorTracerMode | None = None


def _ensure_tracer_enabled() -> None:
    global _ACTIVE_MODE
    if _ACTIVE_MODE is None:
        _ACTIVE_MODE = _LazyTensorTracerMode()
        _ACTIVE_MODE.__enter__()


def _flush_pending_ir(triggered_by_host_stall: bool = False) -> None:
    """Materializes pending lazy IR ops into an OpenXLA HLO compilation check."""
    if _STATE.ir_ops_pending == 0:
        return

    if _STATE.current_step_shapes:
        signature = "|".join(_STATE.current_step_shapes)
        if signature not in _STATE.compiled_graph_signatures:
            _STATE.compiled_graph_signatures.add(signature)
            _STATE.uncached_compile += 1
        else:
            _STATE.cached_compile += 1

    _STATE.ir_ops_pending = 0
    _STATE.current_step_shapes.clear()

    if not triggered_by_host_stall:
        callbacks = list(_STATE.step_callbacks)
        _STATE.step_callbacks.clear()
        _STATE.in_internal_audit = True
        try:
            for fn, fn_args in callbacks:
                fn(*fn_args)
        finally:
            _STATE.in_internal_audit = False


def xla_device() -> torch.device:
    """Returns the active OpenXLA device (or local traced CPU device)."""
    _ensure_tracer_enabled()
    if HAS_NATIVE_TORCH_XLA:
        return _xm.xla_device()  # type: ignore[no-any-return]
    return torch.device("cpu")


def mark_step() -> None:
    """Inserts an explicit OpenXLA step barrier, cutting and executing the lazy IR graph."""
    _ensure_tracer_enabled()
    if HAS_NATIVE_TORCH_XLA:
        _xm.mark_step()
    _STATE.mark_step_count += 1
    _flush_pending_ir(triggered_by_host_stall=False)


def add_step_closure(closure: Any, args: tuple[Any, ...] = ()) -> None:
    """Registers a host callback to run asynchronously after the current XLA step finishes."""
    _ensure_tracer_enabled()
    if HAS_NATIVE_TORCH_XLA:
        _xm.add_step_closure(closure, args=args)
    else:
        _STATE.step_callbacks.append((closure, args))


def clear_metrics() -> None:
    """Clears all OpenXLA compilation and fallback counters."""
    _ensure_tracer_enabled()
    if HAS_NATIVE_TORCH_XLA and _met is not None:
        _met.clear_all()
    _STATE.reset()


def get_metrics_snapshot() -> dict[str, int]:
    """Returns a structured dictionary of key XLA performance counters."""
    return {
        "CachedCompile": _STATE.cached_compile,
        "UncachedCompile": _STATE.uncached_compile,
        "MarkStepCount": _STATE.mark_step_count,
        "HostBarrierStalls": _STATE.host_barrier_stalls,
        "AtenNonzeroFallbacks": _STATE.aten_nonzero_fallbacks,
        "PendingIROps": _STATE.ir_ops_pending,
        "TotalIROpsTraced": _STATE.total_ir_ops_traced,
    }


def metrics_report() -> str:
    """Generates an OpenXLA diagnostic metrics report."""
    snap = get_metrics_snapshot()
    lines = [
        "Metric: CachedCompile",
        f"  TotalSamples: {snap['CachedCompile']}",
        "Metric: UncachedCompile",
        f"  TotalSamples: {snap['UncachedCompile']}",
        "Metric: MarkStepCount",
        f"  TotalSamples: {snap['MarkStepCount']}",
        "Metric: TransferFromDevice / HostBarrierStalls",
        f"  TotalSamples: {snap['HostBarrierStalls']}",
        "Metric: aten::nonzero (Dynamic Shape CPU Fallback)",
        f"  TotalSamples: {snap['AtenNonzeroFallbacks']}",
        "Metric: TracedLazyIROps",
        f"  TotalSamples: {snap['TotalIROpsTraced']} (Pending: {snap['PendingIROps']})",
    ]
    if HAS_NATIVE_TORCH_XLA and _met is not None:
        lines.append("\n--- Native torch_xla.debug.metrics.metrics_report() ---")
        lines.append(_met.metrics_report())
    return "\n".join(lines)
