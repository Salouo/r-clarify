#!/usr/bin/env bash
set -euo pipefail

# Limit GPUs used by this script
export CUDA_VISIBLE_DEVICES=3

modes=(reflect)
start=5
end=10

for mode in "${modes[@]}"; do
  echo "Running mode: ${mode}"
  uv run python -m src.run --mode "${mode}" --start "${start}" --end "${end}"
done
