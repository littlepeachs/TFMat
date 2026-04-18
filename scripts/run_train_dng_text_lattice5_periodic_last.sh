#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
export HYDRA_JOBS="${HYDRA_JOBS:-$ROOT_DIR/hydra_jobs}"
export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/log}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
GPU="${GPU:-5}"
EXP_NAME="${EXP_NAME:-DNG-mp20-text-lattice5-periodic-last-gpu5}"
EPOCHS="${EPOCHS:-3000}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-crystalflow-gridtest}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-128}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/${EXP_NAME}.log}"

mkdir -p "$HYDRA_JOBS" "$WANDB_DIR"

{
  echo "===== DNG Text Training (lattice5 + periodic + last) ====="
  echo "DATE=$(date '+%F %T')"
  echo "GPU=$GPU"
  echo "EXP_NAME=$EXP_NAME"
  echo "EPOCHS=$EPOCHS"
  echo "LOG_FILE=$LOG_FILE"
  echo
} | tee "$LOG_FILE"

nohup env CUDA_VISIBLE_DEVICES="$GPU" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" diffcsp/run.py \
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
  model.cost_lattice=5 \
  model.decoder.num_freqs=256 \
  model.decoder.rec_emb=sin \
  model.decoder.num_millers=8 \
  +model.decoder.na_emb=0 \
  model.decoder.hidden_dim=512 \
  model.decoder.num_layers=6 \
  train.model_checkpoints.save_last=True \
  train.periodic_checkpoints.every_n_epochs=100 \
  "logging.wandb.mode=$WANDB_MODE" \
  "logging.wandb.project=$WANDB_PROJECT" \
  "expname=$EXP_NAME" >> "$LOG_FILE" 2>&1 &

PID=$!
echo "Training started: PID=$PID on GPU $GPU"
echo "PID=$PID" >> "$LOG_FILE"