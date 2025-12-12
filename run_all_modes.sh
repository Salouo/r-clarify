#!/usr/bin/env bash
set -euo pipefail

# Limit GPUs used by this script
export CUDA_VISIBLE_DEVICES=0,1,2

modes=(reflect)

for mode in "${modes[@]}"; do
  echo "Running mode: ${mode}"
  uv run python -m src.run --mode "${mode}"
done
