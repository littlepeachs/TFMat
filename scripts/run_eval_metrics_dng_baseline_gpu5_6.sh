#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR/.."

PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
RUN_DIR="${RUN_DIR:-$PWD/hydra_jobs/singlerun/DNG-mp20-baseline-gpu2}"
DATASET="${DATASET:-mp_20}"
GT_FILE="${GT_FILE:-$PWD/data/mp_20/test.csv}"
GEN_GPU="${GEN_GPU:-5}"
METRIC_GPU="${METRIC_GPU:-6}"
ODE_INT_STEPS="${ODE_INT_STEPS:-100}"
ANNEAL_SLOPE="${ANNEAL_SLOPE:-5}"
N_JOBS="${N_JOBS:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
SEED="${SEED:-0}"
LABEL="${LABEL:-dng_mp20_baseline_gpu5_gen}"
LOG_FILE="${LOG_FILE:-$PWD/scripts/dng_mp20_baseline_eval_metrics_gpu56.log}"

mkdir -p "$(dirname "$LOG_FILE")"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Missing run dir: $RUN_DIR" >&2
  exit 1
fi

GEN_FILE="$RUN_DIR/eval_gen_${LABEL}.pt"
METRICS_FILE="$RUN_DIR/eval_metrics_${LABEL}.json"

{
  echo "===== DNG Baseline Generation + Metrics ====="
  echo "DATE=$(date '+%F %T')"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "RUN_DIR=$RUN_DIR"
  echo "DATASET=$DATASET"
  echo "GT_FILE=$GT_FILE"
  echo "GEN_GPU=$GEN_GPU"
  echo "METRIC_GPU=$METRIC_GPU"
  echo "ODE_INT_STEPS=$ODE_INT_STEPS"
  echo "ANNEAL_SLOPE=$ANNEAL_SLOPE"
  echo "N_JOBS=$N_JOBS"
  echo "NUM_SAMPLES=$NUM_SAMPLES"
  echo "SEED=$SEED"
  echo "LABEL=$LABEL"
  echo "GEN_FILE=$GEN_FILE"
  echo "METRICS_FILE=$METRICS_FILE"
  echo
  echo "[1/2] Generation"
  echo "CUDA_VISIBLE_DEVICES=$GEN_GPU HYDRA_FULL_ERROR=1 $PYTHON_BIN scripts/generation.py --model_path $RUN_DIR --dataset $DATASET --ode-int-steps $ODE_INT_STEPS --anneal_coords --anneal_slope $ANNEAL_SLOPE --num-samples $NUM_SAMPLES --seed $SEED --label $LABEL"
  echo
} | tee "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$GEN_GPU" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" scripts/generation.py \
  --model_path "$RUN_DIR" \
  --dataset "$DATASET" \
  --ode-int-steps "$ODE_INT_STEPS" \
  --anneal_coords \
  --anneal_slope "$ANNEAL_SLOPE" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --label "$LABEL" >> "$LOG_FILE" 2>&1

{
  echo
  echo "[2/2] Compute metrics"
  echo "CUDA_VISIBLE_DEVICES=$METRIC_GPU $PYTHON_BIN scripts/compute_metrics.py --root_path $RUN_DIR --tasks gen --gt_file $GT_FILE --label $LABEL -j $N_JOBS"
  echo
} >> "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$METRIC_GPU" "$PYTHON_BIN" scripts/compute_metrics.py \
  --root_path "$RUN_DIR" \
  --tasks gen \
  --gt_file "$GT_FILE" \
  --label "$LABEL" \
  -j "$N_JOBS" >> "$LOG_FILE" 2>&1

{
  echo "DONE=$(date '+%F %T')"
  echo "GEN_FILE=$GEN_FILE"
  echo "METRICS_FILE=$METRICS_FILE"
} >> "$LOG_FILE"

echo "Finished."
echo "Log: $LOG_FILE"
echo "Gen file: $GEN_FILE"
echo "Metrics file: $METRICS_FILE"