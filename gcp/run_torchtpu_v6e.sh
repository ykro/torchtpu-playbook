#!/usr/bin/env bash
# On-demand Cloud TPU v6e runner for torchtpu-playbook, with auto-teardown.
#
# STATUS: written from the TorchTPU docs (torchtpu.readthedocs.io, 0.2.0 on PyPI) and NOT run
# yet. Every number in benchmarks/ comes from a laptop CPU run. If you run this, the judge
# writes a second JSON whose "environment.device" says "tpu".
#
# What changes on the TPU VM:
# - torch_tpu is installed, so experiments/tpu_backend.py picks device "tpu" and forwards each
#   captured graph to the registered "tpu" compile backend (still counted by GraphRecorder).
# - Experiment 04 uses backend "tpu_dist" and DeviceMesh("tpu", ...) (TORCHTPU_DEVICE=tpu).
# - Experiment 05 (DLPack + JAX) stays on CPU tensors: TorchTPU does not document DLPack
#   between torch_tpu tensors and JAX, so it is not part of the TPU run.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id}"
ZONE="${ZONE:-us-east5-b}"
TPU_NAME="${TPU_NAME:-torchtpu-playbook-$(date +%s)}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v6e-4}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv6e}"
ARTIFACT_DIR="${ARTIFACT_DIR:-./benchmarks/tpu_artifacts}"

cleanup() {
  echo "[cleanup] Deleting TPU VM ${TPU_NAME} in ${ZONE}..."
  gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" --quiet || true
}
trap cleanup EXIT INT TERM

mkdir -p "${ARTIFACT_DIR}"

echo "[1/4] Creating on-demand TPU VM ${TPU_NAME} (${ACCELERATOR_TYPE}, ${RUNTIME_VERSION})..."
gcloud compute tpus tpu-vm create "${TPU_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  --accelerator-type="${ACCELERATOR_TYPE}" --version="${RUNTIME_VERSION}"

echo "[2/4] Copying the repo..."
gcloud compute tpus tpu-vm scp --project="${PROJECT_ID}" --zone="${ZONE}" --recurse \
  ./experiments ./tests ./verify_torchtpu_playbook.py ./pyproject.toml \
  "${TPU_NAME}:~/torchtpu-playbook/"

echo "[3/4] Installing TorchTPU 0.2.0 and running experiments 01-04 on the TPU..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" \
  --command="set -e; cd ~/torchtpu-playbook && \
    sudo apt-get update -qq && sudo apt-get install -y -qq python3.12 python3.12-venv && \
    python3.12 -m venv .venv && . .venv/bin/activate && \
    pip install --quiet torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu && \
    python -c 'import torch; print(torch.__version__, torch.device(\"tpu\"))' && \
    python experiments/01_eager_then_compile.py && \
    python experiments/02_hunting_graph_breaks.py && \
    python experiments/03_three_traps_fixed.py && \
    TORCHTPU_DEVICE=tpu python experiments/04_dtensor_mesh_sharding.py" \
  | tee "${ARTIFACT_DIR}/tpu_terminal_output.txt"

echo "[4/4] Done. Output saved to ${ARTIFACT_DIR}/tpu_terminal_output.txt"
