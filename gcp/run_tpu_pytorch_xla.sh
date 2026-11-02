#!/usr/bin/env bash
# Ephemeral Cloud TPU v5e runner for pytorch-xla-tpu-playbook
# Provisions a v5litepod-4 TPU VM in Google Cloud, executes the PyTorch/XLA
# benchmark and verify_playbook.py suite, downloads the metrics_report() artifact,
# and tears down the TPU VM automatically on exit.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id}"
ZONE="${ZONE:-us-central1-a}"
TPU_NAME="${TPU_NAME:-pytorch-xla-playbook-$(date +%s)}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-4}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv5-lite}"
ARTIFACT_DIR="${ARTIFACT_DIR:-./benchmarks/tpu_artifacts}"

export CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity"

cleanup() {
  echo "[cleanup] Deleting ephemeral TPU VM ${TPU_NAME} in ${ZONE}..."
  gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --quiet || true
}
trap cleanup EXIT INT TERM

mkdir -p "${ARTIFACT_DIR}"

echo "[1/4] Creating Cloud TPU VM ${TPU_NAME} (${ACCELERATOR_TYPE}, ${RUNTIME_VERSION})..."
gcloud compute tpus tpu-vm create "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --accelerator-type="${ACCELERATOR_TYPE}" \
  --version="${RUNTIME_VERSION}" \
  --spot

echo "[2/4] Copying experiments and verifier to ${TPU_NAME}..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command="mkdir -p ~/pytorch-xla-tpu-playbook"
gcloud compute tpus tpu-vm scp \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --recurse \
  ./experiments ./verify_playbook.py "${TPU_NAME}:~/pytorch-xla-tpu-playbook/"

echo "[3/4] Running PyTorch/XLA experiments and metrics_report() verification on TPU..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command="cd ~/pytorch-xla-tpu-playbook && \
    pip install --quiet torch torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    PJRT_DEVICE=TPU python3 experiments/01_lazy_tensor_5_lines.py && \
    PJRT_DEVICE=TPU python3 experiments/02_hunting_graph_breaks.py && \
    PJRT_DEVICE=TPU python3 experiments/03_three_xla_traps_fixed.py && \
    PJRT_DEVICE=TPU python3 experiments/04_spmd_sharding_tpu.py && \
    PJRT_DEVICE=TPU python3 experiments/05_call_jax_from_pytorch.py && \
    PJRT_DEVICE=TPU python3 verify_playbook.py > ~/tpu_metrics_report.txt"

echo "[4/4] Downloading TPU metrics_report() log to ${ARTIFACT_DIR}..."
gcloud compute tpus tpu-vm scp \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  "${TPU_NAME}:~/tpu_metrics_report.txt" "${ARTIFACT_DIR}/tpu_metrics_report.txt"

echo "[done] TPU run complete. Metrics saved to ${ARTIFACT_DIR}/tpu_metrics_report.txt"
