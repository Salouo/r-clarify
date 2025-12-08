#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-reflect_k}"   # 默认跑 reflect_k 模式
TOTAL_SAMPLES=545        # 数据集中样本总数（如有变化手动改这里）

RESULTS_FILE="results/${MODE}.json"

echo "[Runner] mode=${MODE}, target samples=${TOTAL_SAMPLES}"

while true; do
  echo "[Runner] 启动一轮: uv run python -m src.run \"${MODE}\""

  # 如果中途报错（退出码非 0），不要直接退出脚本，而是提示后重试
  if ! uv run python -m src.run "${MODE}"; then
    echo "[Runner] 本轮运行出错，将在 10 秒后重试……"
    sleep 10
  fi

  # 检查当前已经完成多少样本
  if [[ -f "${RESULTS_FILE}" ]]; then
    COMPLETED=$(python - <<PY
import json, sys
from pathlib import Path

path = Path("${RESULTS_FILE}")
data = json.loads(path.read_text(encoding="utf-8"))

results = data.get("results", [])
done = len({r.get("sample_index") for r in results if r.get("sample_index") is not None})
print(done)
PY
)
  else
    COMPLETED=0
  fi

  echo "[Runner] 当前已完成样本数: ${COMPLETED}/${TOTAL_SAMPLES}"

  if [[ "${COMPLETED}" -ge "${TOTAL_SAMPLES}" ]]; then
    echo "[Runner] 全部样本已完成，退出。"
    break
  fi

  echo "[Runner] 还有样本未完成，将在 10 秒后继续下一轮……"
  sleep 10
done

