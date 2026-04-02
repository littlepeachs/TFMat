#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
export HYDRA_JOBS="${HYDRA_JOBS:-$ROOT_DIR/hydra_jobs}"
export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/log}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-crystalflow-gridtest}"
EPOCHS="${EPOCHS:-3000}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-128}"

BASELINE_GPU="${BASELINE_GPU:-2}"
TEXT_GPU="${TEXT_GPU:-3}"

BASELINE_EXP="${BASELINE_EXP:-DNG-mp20-baseline-gpu2}"
TEXT_EXP="${TEXT_EXP:-DNG-mp20-text-gpu3}"

MASTER_LOG="${MASTER_LOG:-$ROOT_DIR/train_dng_mp20_baseline_text.log}"
BASELINE_LOG="${BASELINE_LOG:-$ROOT_DIR/${BASELINE_EXP}.log}"
TEXT_LOG="${TEXT_LOG:-$ROOT_DIR/${TEXT_EXP}.log}"

LAST_PID=""

mkdir -p "$HYDRA_JOBS" "$WANDB_DIR"

start_baseline() {
  local gpu_id="$1"
  local exp_name="$2"
  local log_file="$3"

  {
    echo "===== DNG Baseline Training ====="
    echo "DATE=$(date '+%F %T')"
    echo "GPU_ID=$gpu_id"
    echo "EXP_NAME=$exp_name"
    echo "LOG_FILE=$log_file"
    echo
  } | tee "$log_file"

  nohup env CUDA_VISIBLE_DEVICES="$gpu_id" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" diffcsp/run.py \
    data=mp_20 \
    data.train_max_epochs="$EPOCHS" \
    data.datamodule.batch_size.train="$TRAIN_BATCH_SIZE" \
    data.datamodule.batch_size.val="$VAL_BATCH_SIZE" \
    data.datamodule.batch_size.test="$TEST_BATCH_SIZE" \
    model=flow_polar_w_type \
    +model.type_encoding=table \
    optim.optimizer.lr=1e-3 \
    optim.optimizer.weight_decay=0 \
    optim.lr_scheduler.factor=0.6 \
    +model.lattice_polar_sigma=0.1 \
    model.cost_type=10 \
    model.cost_coord=10 \
    model.cost_lattice=1 \
    model.decoder.num_freqs=256 \
    model.decoder.rec_emb=sin \
    model.decoder.num_millers=8 \
    +model.decoder.na_emb=0 \
    model.decoder.hidden_dim=512 \
    model.decoder.num_layers=6 \
    "logging.wandb.mode=$WANDB_MODE" \
    "logging.wandb.project=$WANDB_PROJECT" \
    "expname=$exp_name" >> "$log_file" 2>&1 &
  LAST_PID="$!"
}

start_text() {
  local gpu_id="$1"
  local exp_name="$2"
  local log_file="$3"

  {
    echo "===== DNG Text Training ====="
    echo "DATE=$(date '+%F %T')"
    echo "GPU_ID=$gpu_id"
    echo "EXP_NAME=$exp_name"
    echo "LOG_FILE=$log_file"
    echo
  } | tee "$log_file"

  nohup env CUDA_VISIBLE_DEVICES="$gpu_id" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" diffcsp/run.py \
    data=mp_20_text \
    data.train_max_epochs="$EPOCHS" \
    data.datamodule.batch_size.train="$TRAIN_BATCH_SIZE" \
    data.datamodule.batch_size.val="$VAL_BATCH_SIZE" \
    data.datamodule.batch_size.test="$TEST_BATCH_SIZE" \
    model=flow_polar_w_type \
    +model.type_encoding=table \
    +model.guide_threshold=-1 \
    optim.optimizer.lr=1e-3 \
    optim.optimizer.weight_decay=0 \
    optim.lr_scheduler.factor=0.6 \
    +model.lattice_polar_sigma=0.1 \
    model.cost_type=10 \
    model.cost_coord=10 \
    model.cost_lattice=1 \
    model.decoder.num_freqs=256 \
    model.decoder.rec_emb=sin \
    model.decoder.num_millers=8 \
    +model.decoder.na_emb=0 \
    model.decoder.hidden_dim=512 \
    model.decoder.num_layers=6 \
    "logging.wandb.mode=$WANDB_MODE" \
    "logging.wandb.project=$WANDB_PROJECT" \
    "expname=$exp_name" >> "$log_file" 2>&1 &
  LAST_PID="$!"
}

{
  echo "===== MP20 DNG Baseline + Text Launcher ====="
  echo "DATE=$(date '+%F %T')"
  echo "ROOT_DIR=$ROOT_DIR"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "WANDB_MODE=$WANDB_MODE"
  echo "WANDB_PROJECT=$WANDB_PROJECT"
  echo "EPOCHS=$EPOCHS"
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
  echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
  echo "TEST_BATCH_SIZE=$TEST_BATCH_SIZE"
  echo "BASELINE_GPU=$BASELINE_GPU"
  echo "TEXT_GPU=$TEXT_GPU"
  echo "BASELINE_EXP=$BASELINE_EXP"
  echo "TEXT_EXP=$TEXT_EXP"
  echo "MASTER_LOG=$MASTER_LOG"
  echo
} | tee "$MASTER_LOG"

if [[ "$RUN_BACKGROUND" == "1" ]]; then
  start_baseline "$BASELINE_GPU" "$BASELINE_EXP" "$BASELINE_LOG"
  BASELINE_PID="$LAST_PID"
  start_text "$TEXT_GPU" "$TEXT_EXP" "$TEXT_LOG"
  TEXT_PID="$LAST_PID"

  {
    echo "BASELINE_PID=$BASELINE_PID"
    echo "TEXT_PID=$TEXT_PID"
    echo "BASELINE_LOG=$BASELINE_LOG"
    echo "TEXT_LOG=$TEXT_LOG"
  } | tee -a "$MASTER_LOG"
else
  env CUDA_VISIBLE_DEVICES="$BASELINE_GPU" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" diffcsp/run.py \
    data=mp_20 \
    data.train_max_epochs="$EPOCHS" \
    data.datamodule.batch_size.train="$TRAIN_BATCH_SIZE" \
    data.datamodule.batch_size.val="$VAL_BATCH_SIZE" \
    data.datamodule.batch_size.test="$TEST_BATCH_SIZE" \
    model=flow_polar_w_type \
    +model.type_encoding=table \
    optim.optimizer.lr=1e-3 \
    optim.optimizer.weight_decay=0 \
    optim.lr_scheduler.factor=0.6 \
    +model.lattice_polar_sigma=0.1 \
    model.cost_type=10 \
    model.cost_coord=10 \
    model.cost_lattice=1 \
    model.decoder.num_freqs=256 \
    model.decoder.rec_emb=sin \
    model.decoder.num_millers=8 \
    +model.decoder.na_emb=0 \
    model.decoder.hidden_dim=512 \
    model.decoder.num_layers=6 \
    "logging.wandb.mode=$WANDB_MODE" \
    "logging.wandb.project=$WANDB_PROJECT" \
    "expname=$BASELINE_EXP" 2>&1 | tee -a "$BASELINE_LOG"

  env CUDA_VISIBLE_DEVICES="$TEXT_GPU" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" diffcsp/run.py \
    data=mp_20_text \
    data.train_max_epochs="$EPOCHS" \
    data.datamodule.batch_size.train="$TRAIN_BATCH_SIZE" \
    data.datamodule.batch_size.val="$VAL_BATCH_SIZE" \
    data.datamodule.batch_size.test="$TEST_BATCH_SIZE" \
    model=flow_polar_w_type \
    +model.type_encoding=table \
    +model.guide_threshold=-1 \
    optim.optimizer.lr=1e-3 \
    optim.optimizer.weight_decay=0 \
    optim.lr_scheduler.factor=0.6 \
    +model.lattice_polar_sigma=0.1 \
    model.cost_type=10 \
    model.cost_coord=10 \
    model.cost_lattice=1 \
    model.decoder.num_freqs=256 \
    model.decoder.rec_emb=sin \
    model.decoder.num_millers=8 \
    +model.decoder.na_emb=0 \
    model.decoder.hidden_dim=512 \
    model.decoder.num_layers=6 \
    "logging.wandb.mode=$WANDB_MODE" \
    "logging.wandb.project=$WANDB_PROJECT" \
    "expname=$TEXT_EXP" 2>&1 | tee -a "$TEXT_LOG"
fi
