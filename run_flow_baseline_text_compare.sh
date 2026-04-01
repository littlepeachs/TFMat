#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
export HYDRA_JOBS="${HYDRA_JOBS:-$ROOT_DIR/hydra_jobs}"
export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/log}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
GPU_BASELINE="${GPU_BASELINE:-0}"
GPU_TEXT="${GPU_TEXT:-0}"
EPOCHS="${EPOCHS:-3000}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-crystalflow-gridtest}"
BASELINE_EXP="${BASELINE_EXP:-CSP-mp20-baseline}"
TEXT_EXP="${TEXT_EXP:-CSP-mp20-text}"
TEXT_COLUMN="${TEXT_COLUMN:-text2}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"
COMPARE_LOG="${COMPARE_LOG:-$ROOT_DIR/compare_flow_text.log}"
BASELINE_LOG="${BASELINE_LOG:-$ROOT_DIR/${BASELINE_EXP}.log}"
TEXT_LOG="${TEXT_LOG:-$ROOT_DIR/${TEXT_EXP}.log}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-128}"
COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"
BASELINE_EXTRA_ARGS="${BASELINE_EXTRA_ARGS:-}"
TEXT_EXTRA_ARGS="${TEXT_EXTRA_ARGS:-}"

COMMON_EXTRA_ARR=()
BASELINE_EXTRA_ARR=()
TEXT_EXTRA_ARR=()
if [[ -n "$COMMON_EXTRA_ARGS" ]]; then
  read -r -a COMMON_EXTRA_ARR <<< "$COMMON_EXTRA_ARGS"
fi
if [[ -n "$BASELINE_EXTRA_ARGS" ]]; then
  read -r -a BASELINE_EXTRA_ARR <<< "$BASELINE_EXTRA_ARGS"
fi
if [[ -n "$TEXT_EXTRA_ARGS" ]]; then
  read -r -a TEXT_EXTRA_ARR <<< "$TEXT_EXTRA_ARGS"
fi

mkdir -p "$(dirname "$COMPARE_LOG")" "$HYDRA_JOBS" "$WANDB_DIR"

BASELINE_CMD=(
  "$PYTHON_BIN" diffcsp/run.py
  data=mp_20
  "data.train_max_epochs=$EPOCHS"
  "data.datamodule.batch_size.train=$TRAIN_BATCH_SIZE"
  "data.datamodule.batch_size.val=$VAL_BATCH_SIZE"
  "data.datamodule.batch_size.test=$TEST_BATCH_SIZE"
  model=flow_polar
  optim.optimizer.lr=1e-3
  optim.optimizer.weight_decay=0
  optim.lr_scheduler.factor=0.6
  +model.lattice_polar_sigma=0.1
  model.cost_coord=10
  model.cost_lattice=1
  model.decoder.num_freqs=256
  model.decoder.rec_emb=sin
  model.decoder.num_millers=8
  +model.decoder.na_emb=0
  model.decoder.hidden_dim=512
  model.decoder.num_layers=6
  "logging.wandb.mode=$WANDB_MODE"
  "logging.wandb.project=$WANDB_PROJECT"
  "expname=$BASELINE_EXP"
)

TEXT_CMD=(
  "$PYTHON_BIN" diffcsp/run.py
  data=mp_20_text
  "data.text_column=$TEXT_COLUMN"
  "data.train_max_epochs=$EPOCHS"
  "data.datamodule.batch_size.train=$TRAIN_BATCH_SIZE"
  "data.datamodule.batch_size.val=$VAL_BATCH_SIZE"
  "data.datamodule.batch_size.test=$TEST_BATCH_SIZE"
  model=flow_polar_text
  optim.optimizer.lr=1e-3
  optim.optimizer.weight_decay=0
  optim.lr_scheduler.factor=0.6
  +model.lattice_polar_sigma=0.1
  model.cost_coord=10
  model.cost_lattice=1
  model.decoder.num_freqs=256
  model.decoder.rec_emb=sin
  model.decoder.num_millers=8
  +model.decoder.na_emb=0
  model.decoder.hidden_dim=512
  model.decoder.num_layers=6
  "logging.wandb.mode=$WANDB_MODE"
  "logging.wandb.project=$WANDB_PROJECT"
  "expname=$TEXT_EXP"
)

if (( ${#COMMON_EXTRA_ARR[@]} > 0 )); then
  BASELINE_CMD+=("${COMMON_EXTRA_ARR[@]}")
  TEXT_CMD+=("${COMMON_EXTRA_ARR[@]}")
fi
if (( ${#BASELINE_EXTRA_ARR[@]} > 0 )); then
  BASELINE_CMD+=("${BASELINE_EXTRA_ARR[@]}")
fi
if (( ${#TEXT_EXTRA_ARR[@]} > 0 )); then
  TEXT_CMD+=("${TEXT_EXTRA_ARR[@]}")
fi

{
  echo "===== CrystalFlow Baseline vs Text Compare ====="
  echo "DATE=$(date '+%F %T')"
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "HYDRA_JOBS=$HYDRA_JOBS"
  echo "WANDB_DIR=$WANDB_DIR"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "EPOCHS=$EPOCHS"
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
  echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
  echo "TEST_BATCH_SIZE=$TEST_BATCH_SIZE"
  echo "WANDB_MODE=$WANDB_MODE"
  echo "WANDB_PROJECT=$WANDB_PROJECT"
  echo "GPU_BASELINE=$GPU_BASELINE"
  echo "GPU_TEXT=$GPU_TEXT"
  echo "BASELINE_EXP=$BASELINE_EXP"
  echo "TEXT_EXP=$TEXT_EXP"
  echo "TEXT_COLUMN=$TEXT_COLUMN"
  echo "COMMON_EXTRA_ARGS=$COMMON_EXTRA_ARGS"
  echo "BASELINE_EXTRA_ARGS=$BASELINE_EXTRA_ARGS"
  echo "TEXT_EXTRA_ARGS=$TEXT_EXTRA_ARGS"
  echo "BASELINE_LOG=$BASELINE_LOG"
  echo "TEXT_LOG=$TEXT_LOG"
  echo "BASELINE_COMMAND=${BASELINE_CMD[*]}"
  echo "TEXT_COMMAND=${TEXT_CMD[*]}"
  echo
} | tee "$COMPARE_LOG"

{
  echo "===== Baseline Run ====="
  echo "DATE=$(date '+%F %T')"
  echo "GPU=$GPU_BASELINE"
  echo "EXP_NAME=$BASELINE_EXP"
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
  echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
  echo "TEST_BATCH_SIZE=$TEST_BATCH_SIZE"
  echo "COMMAND=${BASELINE_CMD[*]}"
  echo
} | tee "$BASELINE_LOG"

{
  echo "===== Text Run ====="
  echo "DATE=$(date '+%F %T')"
  echo "GPU=$GPU_TEXT"
  echo "EXP_NAME=$TEXT_EXP"
  echo "TEXT_COLUMN=$TEXT_COLUMN"
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
  echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
  echo "TEST_BATCH_SIZE=$TEST_BATCH_SIZE"
  echo "COMMAND=${TEXT_CMD[*]}"
  echo
} | tee "$TEXT_LOG"

if [[ "$RUN_BACKGROUND" == "1" ]]; then
  nohup env CUDA_VISIBLE_DEVICES="$GPU_BASELINE" HYDRA_FULL_ERROR=1 "${BASELINE_CMD[@]}" >> "$BASELINE_LOG" 2>&1 &
  BASELINE_PID=$!
  nohup env CUDA_VISIBLE_DEVICES="$GPU_TEXT" HYDRA_FULL_ERROR=1 "${TEXT_CMD[@]}" >> "$TEXT_LOG" 2>&1 &
  TEXT_PID=$!
  {
    echo "BASELINE_PID=$BASELINE_PID"
    echo "TEXT_PID=$TEXT_PID"
  } | tee -a "$COMPARE_LOG" "$BASELINE_LOG" "$TEXT_LOG"
else
  env CUDA_VISIBLE_DEVICES="$GPU_BASELINE" HYDRA_FULL_ERROR=1 "${BASELINE_CMD[@]}" 2>&1 | tee -a "$BASELINE_LOG"
  env CUDA_VISIBLE_DEVICES="$GPU_TEXT" HYDRA_FULL_ERROR=1 "${TEXT_CMD[@]}" 2>&1 | tee -a "$TEXT_LOG"
fi