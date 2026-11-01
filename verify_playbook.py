"""The Outside Judge (`verify_playbook.py`) for `pytorch-xla-tpu-playbook`.

Executes all 5 experiments end-to-end, enforces hard assertions on OpenXLA compilation,
host-sync barrier stalls, dynamic fallback elimination, 4-device GSPMD sharding, and
PyTorch-to-JAX zero-copy autograd parity, and writes verified artifacts to:
- `benchmarks/latest_results.json`
- `benchmarks/terminal_output.txt`
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import importlib
import io
import json
import platform
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import jax
import torch

exp01 = importlib.import_module("experiments.01_lazy_tensor_5_lines")
exp02 = importlib.import_module("experiments.02_hunting_graph_breaks")
exp03 = importlib.import_module("experiments.03_three_xla_traps_fixed")
exp04 = importlib.import_module("experiments.04_spmd_sharding_tpu")
exp05 = importlib.import_module("experiments.05_call_jax_from_pytorch")


def run_all_verifications() -> tuple[dict[str, Any], str]:
    """Runs all 5 playbook experiments, validates invariants, and captures terminal output."""
    buf = io.StringIO()

    with redirect_stdout(buf):
        print("======================================================================")
        print(" PyTorch/XLA & OpenXLA TPU Playbook — Outside Judge Verification Suite")
        print("======================================================================")
        print(f"PyTorch Version : {torch.__version__}")
        print(f"JAX Version     : {jax.__version__}")
        print(f"OpenXLA Devices : {[str(d) for d in jax.devices()]}")
        print(f"Platform        : {platform.platform()}\n")

        print("--- [1/5] Experiment 01: LazyTensor & mark_step() in 5 Lines ---")
        r01 = exp01.run_lazy_tensor_demo()
        print()

        print("--- [2/5] Experiment 02: Hunting Graph Breaks with XLA Metrics ---")
        r02 = exp02.audit_naive_training_loop()
        print()

        print("--- [3/5] Experiment 03: Fixing the 3 Classic XLA Traps ---")
        r03 = exp03.compare_naive_vs_optimized()
        print(
            f"[Trap Fix Summary] Naive UncachedCompile={r03['naive']['uncached_compiles']} -> "
            f"Optimized UncachedCompile={r03['xla_optimized']['uncached_compiles']} "
            f"(Recompiles after warmup={r03['xla_optimized']['recompiles_after_warmup']})"
        )
        print(
            f"[Trap Fix Summary] Naive HostStalls={r03['naive']['host_barrier_stalls']} -> "
            f"Optimized HostStalls={r03['xla_optimized']['host_barrier_stalls']}"
        )
        print(
            f"[Trap Fix Summary] Naive AtenNonzero={r03['naive']['aten_nonzero_fallbacks']} -> "
            f"Optimized AtenNonzero={r03['xla_optimized']['aten_nonzero_fallbacks']}"
        )
        print(
            f"[Trap Fix Summary] Numerical Parity allclose={r03['numerical_parity']['allclose']} "
            f"(max_abs_diff={r03['numerical_parity']['max_abs_diff']:.2e})\n"
        )

        print("--- [4/5] Experiment 04: OpenXLA GSPMD Sharding Across 4 Devices ---")
        r04 = exp04.run_spmd_sharding_experiment()
        print()

        print("--- [5/5] Experiment 05: Zero-Copy PyTorch <-> JAX Autograd Bridge ---")
        r05 = exp05.run_pytorch_jax_bridge_demo()
        print()

        # Hard Judge Assertions
        assert r01["uncached_compiles"] == 1, (
            f"Exp01 expected 1 compile, got {r01['uncached_compiles']}"
        )
        assert r01["cached_compiles"] == 5, (
            f"Exp01 expected 5 cached steps, got {r01['cached_compiles']}"
        )
        assert r01["host_stalls"] == 0, f"Exp01 expected 0 host stalls, got {r01['host_stalls']}"
        assert r01["final_loss"] < r01["initial_loss"], "Exp01 loss did not decrease"

        assert r02["uncached_compiles"] >= 5, "Exp02 naive loop should recompile every step"
        assert r02["host_barrier_stalls"] >= 5, "Exp02 naive loop should record host stalls"
        assert r02["aten_nonzero_fallbacks"] >= 5, (
            "Exp02 naive loop should record aten::nonzero fallbacks"
        )

        recompiles = r03["xla_optimized"]["recompiles_after_warmup"]
        assert recompiles == 0, f"Exp03 expected 0 recompiles after warmup, got {recompiles}"
        assert r03["xla_optimized"]["host_barrier_stalls"] == 0, (
            "Exp03 expected 0 host barrier stalls"
        )
        assert r03["xla_optimized"]["aten_nonzero_fallbacks"] == 0, (
            "Exp03 expected 0 aten::nonzero fallbacks"
        )
        assert r03["numerical_parity"]["allclose"] is True, "Exp03 numerical parity check failed"

        assert r04["num_xla_devices"] == 4, (
            f"Exp04 expected 4 XLA devices, got {r04['num_xla_devices']}"
        )
        assert r04["spmd_parity_allclose"] is True, "Exp04 SPMD parity check failed"

        assert r05["fwd_allclose"] is True, "Exp05 PyTorch->JAX forward parity failed"
        assert r05["bwd_allclose"] is True, "Exp05 PyTorch->JAX backward VJP parity failed"
        assert r05["training_losses"][-1] < r05["training_losses"][0], (
            "Exp05 hybrid training loss did not decrease"
        )

        print("======================================================================")
        print(" ALL OUTSIDE JUDGE VERIFICATIONS PASSED (0 RECOMPILES AFTER WARMUP)   ")
        print("======================================================================")

    terminal_log = buf.getvalue()
    results = {
        "environment": {
            "torch_version": torch.__version__,
            "jax_version": jax.__version__,
            "xla_devices": [str(d) for d in jax.devices()],
        },
        "exp01_lazy_tensor": r01,
        "exp02_hunting_graph_breaks": r02,
        "exp03_three_xla_traps_fixed": r03,
        "exp04_spmd_sharding_tpu": r04,
        "exp05_call_jax_from_pytorch": r05,
        "all_verifications_passed": True,
    }
    return results, terminal_log


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    bench_dir = repo_root / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    results, terminal_log = run_all_verifications()
    print(terminal_log, end="")

    (bench_dir / "latest_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (bench_dir / "terminal_output.txt").write_text(terminal_log, encoding="utf-8")


if __name__ == "__main__":
    main()
