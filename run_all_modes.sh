#!/usr/bin/env bash
set -euo pipefail

modes=(baseline reflect reflect_k cot direct)

for mode in "${modes[@]}"; do
  echo "Running mode: ${mode}"
  python -m src.run --mode "${mode}"
done
