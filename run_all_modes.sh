#!/usr/bin/env bash
set -euo pipefail

# Run src/run.py across all modes with shared config.
# Usage:
#   ./run_all_modes.sh -g 0 -t 3 -w 2
# Options:
#   -g  GPU id/list to expose via CUDA_VISIBLE_DEVICES (default: env CUDA_VISIBLE_DEVICES, else 0)
#   -t  max trials per sample (default: 5)
#   -w  memo window length for reflect_k only (default reflect_k: 3; others fixed at 1)
# Script uses uv (required). Make sure uv is installed.

# 可以直接在这里设默认值，也可以在命令行传参覆盖
gpu_id="${CUDA_VISIBLE_DEVICES:-1}"   # 例: "0,1,2,3" 多卡
max_trials=5
memo_window=3      # only used for reflect_k unless overridden

while getopts ":g:t:w:h" opt; do
  case "$opt" in
    g) gpu_id="$OPTARG" ;;
    t) max_trials="$OPTARG" ;;
    w) memo_window="$OPTARG" ;;
    h)
      echo "Usage: $0 [-g <gpu_id_list>] [-t <max_trials>] [-w <memo_window_for_reflect_k>]"
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$gpu_id"
echo "Using GPU $CUDA_VISIBLE_DEVICES; max_trials=$max_trials; reflect_k memo_window=$memo_window"

# Require uv
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not found in PATH." >&2
  exit 1
fi
runner_cmd=(uv run python)

# 选择 modes
modes=(reflect_k)

for mode in "${modes[@]}"; do
  echo "----------------------------------------"
  echo "Running mode: $mode"
  cmd=("${runner_cmd[@]}" -m src.run --mode "$mode" --max-trials "$max_trials")

  # memo_window: reflect_k 可指定；其他模式固定 1（即便 run.py 不使用，也保持显式一致）
  if [[ "$mode" == "reflect_k" ]]; then
    if [[ -n "$memo_window" ]]; then
      cmd+=(--memo-window "$memo_window")
    fi
  else
    cmd+=(--memo-window 1)
  fi

  echo "Command: ${cmd[*]}"
  "${cmd[@]}"
done
