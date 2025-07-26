#!/usr/bin/env bash
# Use to run built docker image
set -euo pipefail

IMAGE_TAG="${1:-sglang_sssd:latest}"

docker run --gpus all \
  --shm-size 32g \
  --ipc=host \
  -it "${IMAGE_TAG}" \
  -e HF_TOKEN="$HF_TOKEN" \
