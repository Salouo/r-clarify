#!/usr/bin/env bash
set -euo pipefail

# Limit GPUs used by this script
export CUDA_VISIBLE_DEVICES=2,3

modes=(baseline reflect)

for mode in "${modes[@]}"; do
  echo "Running mode: ${mode}"
  # 通过 --clarify-quota 设置每个样本允许的最大澄清次数；设为 2 可根据需要调整，None 表示不限
  uv run python -m src.run --mode "${mode}" --clarify-quota 2
done
