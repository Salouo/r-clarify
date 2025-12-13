#!/usr/bin/env bash
set -euo pipefail

# Limit GPUs used by this script
export CUDA_VISIBLE_DEVICES=3

# 默认最大 trial 数，便于集中调整（全模式共用）
MAX_TRIALS=5

modes=(reflect_k)

for mode in "${modes[@]}"; do
  echo "Running mode: ${mode}"
  extra_args=(--max-trials "${MAX_TRIALS}")
  if [[ "${mode}" == "reflect" ]]; then
    extra_args+=(--memo-window 1)
  elif [[ "${mode}" == "reflect_k" ]]; then
    extra_args+=(--memo-window 3)
  fi
  # 通过 --clarify-quota 设置每个样本允许的最大澄清次数；设为 2 可根据需要调整，None 表示不限
  uv run python -m src.run --mode "${mode}" --clarify-quota 2 "${extra_args[@]}"
done
