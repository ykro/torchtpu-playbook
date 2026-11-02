"""Shared helpers: pick the device and wrap the `torch.compile` backend so every graph is counted.

TorchTPU (https://github.com/google-pytorch/torch_tpu) registers the `tpu` device and the
`tpu` compile backend when the `torch_tpu` wheel is installed. That wheel only exists for
Linux x86_64 (TPU VMs), so on a laptop this module falls back to CPU and to an eager
backend that runs the captured FX graph with PyTorch itself.

What is measured is the same on both: TorchDynamo captures the Python function into FX
graphs and hands each one to the backend. `GraphRecorder` sits in that hand-off and records
every graph it receives. A new graph means a new compilation for the backend; a call that
reuses a cached graph does not reach the backend at all.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch._dynamo


def quiet_dynamo() -> None:
    """Silences Dynamo graph-break warnings (they print absolute file paths)."""
    torch._logging.set_logs(graph_breaks=False, recompiles=False)
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)


def torchtpu_available() -> bool:
    """True only where the `torch_tpu` wheel is installed (a Cloud TPU VM)."""
    return importlib.util.find_spec("torch_tpu") is not None


def resolve_device() -> torch.device:
    """`tpu` on a TPU VM with TorchTPU, `cpu` everywhere else."""
    return torch.device("tpu") if torchtpu_available() else torch.device("cpu")


def backend_label() -> str:
    """Human-readable name of the backend that executes the captured graphs."""
    if torchtpu_available():
        return "tpu (TorchTPU: FX -> StableHLO -> XLA)"
    return "eager on CPU (FX graph executed by PyTorch; torch_tpu not installed)"


@dataclass
class GraphRecord:
    """One FX graph handed to the backend by TorchDynamo."""

    num_nodes: int
    ops: list[str]
    input_shapes: list[list[int]]


@dataclass
class GraphRecorder:
    """A `torch.compile` backend that records each graph and delegates execution.

    On CPU it returns `gm.forward` (the FX graph runs eagerly). On a TPU VM it forwards the
    graph to the registered `tpu` backend, so the count is taken at the same point.
    """

    graphs: list[GraphRecord] = field(default_factory=list)

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: list[Any]) -> Callable:
        ops: list[str] = []
        for node in gm.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                target = node.target
                name = getattr(target, "__name__", None) or str(target)
                ops.append(name)
        shapes = [list(t.shape) for t in example_inputs if isinstance(t, torch.Tensor)]
        self.graphs.append(GraphRecord(len(gm.graph.nodes), ops, shapes))
        if torchtpu_available():
            return torch._dynamo.lookup_backend("tpu")(gm, example_inputs)
        return gm.forward

    @property
    def compiles(self) -> int:
        return len(self.graphs)


def compile_counted(fn: Callable, recorder: GraphRecorder) -> Callable:
    """`torch.compile` with static shapes, as XLA expects, routed through the recorder."""
    return torch.compile(fn, backend=recorder, dynamic=False)
