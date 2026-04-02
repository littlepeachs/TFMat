#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR/.."

PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
RUN_DIR="${RUN_DIR:-$PWD/hydra_jobs/singlerun/DNG-mp20-text-gpu3}"
DATASET="${DATASET:-mp_20}"
GT_FILE="${GT_FILE:-$PWD/data_text/mp_20/test.csv}"
TEXT_EMBED_PATH="${TEXT_EMBED_PATH:-$PWD/data_text/mp_20/precomputed_embeddings/test_text2_matscibert_mean.pt}"
GPU="${GPU:-6}"
ODE_INT_STEPS="${ODE_INT_STEPS:-100}"
ANNEAL_SLOPE="${ANNEAL_SLOPE:-5}"
N_JOBS="${N_JOBS:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
LABEL_PREFIX="${LABEL_PREFIX:-dng_mp20_text_gpu6_seed}"
LOG_DIR="${LOG_DIR:-$PWD/scripts/dng_mp20_text_multiseed_logs}"

mkdir -p "$LOG_DIR"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Missing run dir: $RUN_DIR" >&2
  exit 1
fi

if [[ ! -f "$TEXT_EMBED_PATH" ]]; then
  echo "Missing text embedding file: $TEXT_EMBED_PATH" >&2
  exit 1
fi

echo "===== DNG Text Multi-seed Generation + Metrics ====="
echo "DATE=$(date '+%F %T')"
echo "RUN_DIR=$RUN_DIR"
echo "GT_FILE=$GT_FILE"
echo "TEXT_EMBED_PATH=$TEXT_EMBED_PATH"
echo "GPU=$GPU"
echo "NUM_SAMPLES=$NUM_SAMPLES"
echo "SEEDS=$SEEDS"
echo "LOG_DIR=$LOG_DIR"
echo

for seed in $SEEDS; do
  label="${LABEL_PREFIX}${seed}"
  log_file="$LOG_DIR/${label}.log"
  {
    echo "===== seed=$seed ====="
    echo "DATE=$(date '+%F %T')"
    echo "LABEL=$label"
    echo "[1/2] Generation"
    echo "CUDA_VISIBLE_DEVICES=$GPU HYDRA_FULL_ERROR=1 $PYTHON_BIN scripts/generation.py --model_path $RUN_DIR --dataset $DATASET --ode-int-steps $ODE_INT_STEPS --anneal_coords --anneal_slope $ANNEAL_SLOPE --num-samples $NUM_SAMPLES --seed $seed --label $label --text-embedding-path $TEXT_EMBED_PATH"
    echo
  } > "$log_file"

  CUDA_VISIBLE_DEVICES="$GPU" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" scripts/generation.py \
    --model_path "$RUN_DIR" \
    --dataset "$DATASET" \
    --ode-int-steps "$ODE_INT_STEPS" \
    --anneal_coords \
    --anneal_slope "$ANNEAL_SLOPE" \
    --num-samples "$NUM_SAMPLES" \
    --seed "$seed" \
    --label "$label" \
    --text-embedding-path "$TEXT_EMBED_PATH" >> "$log_file" 2>&1

  {
    echo
    echo "[2/2] Compute metrics"
    echo "CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN scripts/compute_metrics.py --root_path $RUN_DIR --tasks gen --gt_file $GT_FILE --label $label -j $N_JOBS"
    echo
  } >> "$log_file"

  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/compute_metrics.py \
    --root_path "$RUN_DIR" \
    --tasks gen \
    --gt_file "$GT_FILE" \
    --label "$label" \
    -j "$N_JOBS" >> "$log_file" 2>&1

  echo "DONE=$(date '+%F %T')" >> "$log_file"
done

echo
echo "Finished."
  echo "Logs are under: $LOG_DIR"