#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
export HYDRA_JOBS="${HYDRA_JOBS:-$ROOT_DIR/hydra_jobs}"
export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/log}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
GPU_ID="${GPU_ID:-0}"
EXP_NAME="${EXP_NAME:-CSP-mp20-text}"
TEXT_COLUMN="${TEXT_COLUMN:-text2}"
EPOCHS="${EPOCHS:-3000}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-crystalflow-gridtest}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/${EXP_NAME}.log}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-128}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

EXTRA_ARR=()
if [[ -n "$EXTRA_ARGS" ]]; then
	read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"
fi

mkdir -p "$(dirname "$LOG_FILE")" "$HYDRA_JOBS" "$WANDB_DIR"

CMD=(
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
	"expname=$EXP_NAME"
)

if (( ${#EXTRA_ARR[@]} > 0 )); then
	CMD+=("${EXTRA_ARR[@]}")
fi

{
	echo "===== CrystalFlow Text Training ====="
	echo "DATE=$(date '+%F %T')"
	echo "ROOT_DIR=$ROOT_DIR"
	echo "PROJECT_ROOT=$PROJECT_ROOT"
	echo "HYDRA_JOBS=$HYDRA_JOBS"
	echo "WANDB_DIR=$WANDB_DIR"
	echo "PYTHON_BIN=$PYTHON_BIN"
	echo "GPU_ID=$GPU_ID"
	echo "EXP_NAME=$EXP_NAME"
	echo "TEXT_COLUMN=$TEXT_COLUMN"
	echo "EPOCHS=$EPOCHS"
	echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE"
	echo "VAL_BATCH_SIZE=$VAL_BATCH_SIZE"
	echo "TEST_BATCH_SIZE=$TEST_BATCH_SIZE"
	echo "WANDB_MODE=$WANDB_MODE"
	echo "WANDB_PROJECT=$WANDB_PROJECT"
	echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
	echo "LOG_FILE=$LOG_FILE"
	echo "EXTRA_ARGS=$EXTRA_ARGS"
	echo "COMMAND=${CMD[*]}"
	echo
} | tee "$LOG_FILE"

if [[ "$RUN_BACKGROUND" == "1" ]]; then
	echo "Launching in background..." | tee -a "$LOG_FILE"
	nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" HYDRA_FULL_ERROR=1 "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
	echo "PID=$!" | tee -a "$LOG_FILE"
else
	env CUDA_VISIBLE_DEVICES="$GPU_ID" HYDRA_FULL_ERROR=1 "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
fi