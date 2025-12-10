#!/usr/bin/env bash
set -euo pipefail

# Limit GPUs used by this script
export CUDA_VISIBLE_DEVICES=2,3

modes=(direct cot baseline)

for mode in "${modes[@]}"; do
  echo "Running mode: ${mode}"
  python -m src.run --mode "${mode}"
done
