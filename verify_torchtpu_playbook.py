"""Outside Judge for `torchtpu-playbook`.

Runs the five experiments end to end, asserts their invariants and writes:
- `benchmarks/latest_results.json`
- `benchmarks/terminal_output.txt`

Every number comes from this run. On a laptop the device is CPU and the compile backend is
an eager recorder (see `experiments/tpu_backend.py`); the JSON says which one was used.
"""

from __future__ import annotations

import importlib
import io
import json
import platform
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import jax
import torch

from experiments.tpu_backend import backend_label, resolve_device, torchtpu_available

exp01 = importlib.import_module("experiments.01_eager_then_compile")
exp02 = importlib.import_module("experiments.02_hunting_graph_breaks")
exp03 = importlib.import_module("experiments.03_three_traps_fixed")
exp04 = importlib.import_module("experiments.04_dtensor_mesh_sharding")
exp05 = importlib.import_module("experiments.05_torch_jax_dlpack_stablehlo")

RULE = "=" * 74


def _check(checks: list[tuple[str, bool]], name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def run_all() -> tuple[dict[str, Any], str, bool]:
    buf = io.StringIO()
    checks: list[tuple[str, bool]] = []
    t0 = time.perf_counter()
    with redirect_stdout(buf):
        print(RULE)
        print(" OUTSIDE JUDGE: torchtpu-playbook (PyTorch on Cloud TPU with TorchTPU)")
        print(RULE)
        print(f"PyTorch        : {torch.__version__}")
        print(f"JAX            : {jax.__version__}")
        print(f"torch_tpu found: {torchtpu_available()}")
        print(f"Device         : {resolve_device()}")
        print(f"Compile backend: {backend_label()}")
        print(f"Platform       : {platform.system()} {platform.machine()}\n")

        print("--- [1/5] Eager first, then torch.compile ---")
        r01 = exp01.run_eager_then_compile()
        _check(checks, "exp01 one graph for six steps", r01["graphs_compiled"] == 1)
        _check(checks, "exp01 eager == compiled losses", r01["eager_vs_compiled_max_abs_diff"] == 0)
        _check(checks, "exp01 loss decreased", r01["loss_decreased"])
        print()

        print("--- [2/5] Hunting graph breaks ---")
        r02 = exp02.audit_naive_step()
        _check(checks, "exp02 two graph breaks per call", r02["graph_breaks_per_call"] == 2)
        _check(
            checks,
            "exp02 breaks are .item() and the boolean mask",
            r02["break_reasons"]
            == [
                "host sync: Tensor.item()",
                "data-dependent shape: aten::nonzero (boolean mask)",
            ],
        )
        _check(
            checks,
            "exp02 new graphs on every step",
            all(n > 0 for n in r02["new_graphs_per_step"]),
        )
        print()

        print("--- [3/5] The three traps fixed ---")
        r03 = exp03.compare_naive_vs_fixed()
        _check(
            checks,
            "exp03 fixed step has 0 graph breaks",
            r03["fixed"]["graph_breaks_per_call"] == 0,
        )
        _check(
            checks,
            "exp03 fixed step compiles once",
            r03["fixed"]["graphs_compiled_over_batches"] == 1,
        )
        _check(
            checks, "exp03 0 recompiles after warmup", r03["fixed"]["recompiles_after_warmup"] == 0
        )
        _check(
            checks,
            "exp03 naive compiles more graphs than batches",
            r03["naive"]["graphs_compiled_over_batches"] > len(r03["seq_lengths"]),
        )
        _check(checks, "exp03 naive and fixed losses match", r03["numerical_parity"]["allclose"])
        print()

        print("--- [4/5] DeviceMesh + DTensor on a 2x2 mesh ---")
        r04 = exp04.run_dtensor_sharding()
        _check(
            checks, "exp04 mesh is data=2, model=2", r04["mesh_shape"] == {"data": 2, "model": 2}
        )
        _check(
            checks,
            "exp04 per-rank shards are halves",
            r04["per_rank_shapes"]
            == {"x": [8, 32, 128], "up_weight": [256, 128], "down_weight": [128, 256]},
        )
        _check(
            checks,
            "exp04 one all_reduce in forward",
            r04["forward_collectives"] == {"all_reduce": 1},
        )
        _check(checks, "exp04 every rank matches unsharded", r04["all_ranks_allclose"])
        print()

        print("--- [5/5] PyTorch <-> JAX via DLPack, and StableHLO ---")
        r05 = exp05.run_pytorch_jax_bridge_demo()
        _check(checks, "exp05 forward parity", r05["fwd_allclose"])
        _check(checks, "exp05 gradient parity", r05["bwd_allclose"])
        _check(checks, "exp05 zero-copy both ways (CPU)", all(r05["zero_copy"].values()))
        _check(checks, "exp05 StableHLO with 2 dot_general", r05["stablehlo"]["dot_general"] == 2)
        _check(
            checks,
            "exp05 hybrid loss decreased",
            r05["training_losses"][-1] < r05["training_losses"][0],
        )

        passed = sum(ok for _, ok in checks)
        elapsed = time.perf_counter() - t0
        status = "PASS" if passed == len(checks) else "FAIL"
        print()
        print(RULE)
        print(f" VERIFICATION STATUS: {status} ({passed}/{len(checks)} checks in {elapsed:.1f}s)")
        print(RULE)

    results = {
        "environment": {
            "torch_version": torch.__version__,
            "jax_version": jax.__version__,
            "torch_tpu_installed": torchtpu_available(),
            "device": str(resolve_device()),
            "compile_backend": backend_label(),
            "platform": f"{platform.system()} {platform.machine()}",
        },
        "exp01_eager_then_compile": r01,
        "exp02_hunting_graph_breaks": r02,
        "exp03_three_traps_fixed": r03,
        "exp04_dtensor_mesh_sharding": r04,
        "exp05_torch_jax_dlpack_stablehlo": r05,
        "checks": {name: ok for name, ok in checks},
        "all_verifications_passed": passed == len(checks),
    }
    return results, buf.getvalue(), passed == len(checks)


def main() -> None:
    bench_dir = Path(__file__).resolve().parent / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    results, log, ok = run_all()
    print(log, end="")
    (bench_dir / "latest_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (bench_dir / "terminal_output.txt").write_text(log, encoding="utf-8")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
