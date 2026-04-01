#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"

echo "Creating result directory and collecting metrics..."
mkdir -p "$ROOT_DIR/result"
"$PYTHON_BIN" "$ROOT_DIR/scripts/collect_eval_results.py"

echo "Done. CSV at: $ROOT_DIR/result/results.csv"