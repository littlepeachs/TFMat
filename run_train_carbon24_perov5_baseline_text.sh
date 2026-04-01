#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
export HYDRA_JOBS="${HYDRA_JOBS:-$ROOT_DIR/hydra_jobs}"
export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/log}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
LAST_PID=""
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-crystalflow-gridtest}"
TEXT_COLUMN="${TEXT_COLUMN:-text2}"
TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-m3rg-iitd/matscibert}"
TEXT_POOLING="${TEXT_POOLING:-mean}"
TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-256}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-64}"

CARBON_BASELINE_GPU="${CARBON_BASELINE_GPU:-0}"
CARBON_TEXT_GPU="${CARBON_TEXT_GPU:-1}"
PEROV_BASELINE_GPU="${PEROV_BASELINE_GPU:-4}"
PEROV_TEXT_GPU="${PEROV_TEXT_GPU:-7}"

CARBON_BASELINE_EXP="${CARBON_BASELINE_EXP:-CSP-carbon24-baseline-gpu0}"
CARBON_TEXT_EXP="${CARBON_TEXT_EXP:-CSP-carbon24-text-gpu1-offline-emb}"
PEROV_BASELINE_EXP="${PEROV_BASELINE_EXP:-CSP-perov5-baseline-gpu4}"
PEROV_TEXT_EXP="${PEROV_TEXT_EXP:-CSP-perov5-text-gpu7-offline-emb}"

MASTER_LOG="${MASTER_LOG:-$ROOT_DIR/train_carbon24_perov5_baseline_text.log}"
CARBON_BASELINE_LOG="${CARBON_BASELINE_LOG:-$ROOT_DIR/${CARBON_BASELINE_EXP}.log}"
CARBON_TEXT_LOG="${CARBON_TEXT_LOG:-$ROOT_DIR/${CARBON_TEXT_EXP}.log}"
PEROV_BASELINE_LOG="${PEROV_BASELINE_LOG:-$ROOT_DIR/${PEROV_BASELINE_EXP}.log}"
PEROV_TEXT_LOG="${PEROV_TEXT_LOG:-$ROOT_DIR/${PEROV_TEXT_EXP}.log}"

mkdir -p "$HYDRA_JOBS" "$WANDB_DIR"

build_train_cmd() {
  local data_name="$1"
  local model_name="$2"
  local exp_name="$3"

  local -a cmd=(
    "$PYTHON_BIN" diffcsp/run.py
    "data=$data_name"
    "model=$model_name"
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
    "expname=$exp_name"
  )

  printf '%q ' "${cmd[@]}"
}

start_baseline() {
  local gpu_id="$1"
  local data_name="$2"
  local exp_name="$3"
  local log_file="$4"
  local cmd

  cmd="$(build_train_cmd "$data_name" flow_polar "$exp_name")"

  {
    echo "===== Baseline Training ====="
    echo "DATE=$(date '+%F %T')"
    echo "GPU_ID=$gpu_id"
    echo "DATA_NAME=$data_name"
    echo "EXP_NAME=$exp_name"
    echo "LOG_FILE=$log_file"
    echo "COMMAND=$cmd"
    echo
  } | tee "$log_file"

  nohup env CUDA_VISIBLE_DEVICES="$gpu_id" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" diffcsp/run.py \
    "data=$data_name" \
    model=flow_polar \
    optim.optimizer.lr=1e-3 \
    optim.optimizer.weight_decay=0 \
    optim.lr_scheduler.factor=0.6 \
    +model.lattice_polar_sigma=0.1 \
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

start_text_pipeline() {
  local gpu_id="$1"
  local dataset_name="$2"
  local data_name="$3"
  local exp_name="$4"
  local log_file="$5"
  local data_root="$ROOT_DIR/data_text/$dataset_name"
  local embed_dir="$data_root/precomputed_embeddings"

  mkdir -p "$embed_dir"

  {
    echo "===== Text Training Pipeline ====="
    echo "DATE=$(date '+%F %T')"
    echo "GPU_ID=$gpu_id"
    echo "DATASET_NAME=$dataset_name"
    echo "DATA_NAME=$data_name"
    echo "EXP_NAME=$exp_name"
    echo "TEXT_COLUMN=$TEXT_COLUMN"
    echo "TEXT_MODEL_NAME=$TEXT_MODEL_NAME"
    echo "TEXT_POOLING=$TEXT_POOLING"
    echo "TEXT_MAX_LENGTH=$TEXT_MAX_LENGTH"
    echo "TEXT_BATCH_SIZE=$TEXT_BATCH_SIZE"
    echo "EMBED_DIR=$embed_dir"
    echo "LOG_FILE=$log_file"
    echo
    echo "Embedding outputs:"
    echo "  $embed_dir/train_text2_matscibert_mean.pt"
    echo "  $embed_dir/val_text2_matscibert_mean.pt"
    echo "  $embed_dir/test_text2_matscibert_mean.pt"
    echo
  } | tee "$log_file"

  nohup bash -lc '
    set -euo pipefail
    cd '"'"$ROOT_DIR"'"'
    export CUDA_VISIBLE_DEVICES='"'"$gpu_id"'"'
    export HYDRA_FULL_ERROR=1
    export PROJECT_ROOT='"'"$PROJECT_ROOT"'"'
    export HYDRA_JOBS='"'"$HYDRA_JOBS"'"'
    export WANDB_DIR='"'"$WANDB_DIR"'"'
    export HF_HUB_DISABLE_XET='"'"$HF_HUB_DISABLE_XET"'"'

    '"'"$PYTHON_BIN"'"' scripts/precompute_text_embeddings.py '"'"$data_root/train.csv"'"' \
      --text-column '"'"$TEXT_COLUMN"'"' \
      --model-name '"'"$TEXT_MODEL_NAME"'"' \
      --pooling '"'"$TEXT_POOLING"'"' \
      --batch-size '"'"$TEXT_BATCH_SIZE"'"' \
      --max-length '"'"$TEXT_MAX_LENGTH"'"' \
      --device cuda \
      --output-path '"'"$embed_dir/train_text2_matscibert_mean.pt"'"'

    '"'"$PYTHON_BIN"'"' scripts/precompute_text_embeddings.py '"'"$data_root/val.csv"'"' \
      --text-column '"'"$TEXT_COLUMN"'"' \
      --model-name '"'"$TEXT_MODEL_NAME"'"' \
      --pooling '"'"$TEXT_POOLING"'"' \
      --batch-size '"'"$TEXT_BATCH_SIZE"'"' \
      --max-length '"'"$TEXT_MAX_LENGTH"'"' \
      --device cuda \
      --output-path '"'"$embed_dir/val_text2_matscibert_mean.pt"'"'

    '"'"$PYTHON_BIN"'"' scripts/precompute_text_embeddings.py '"'"$data_root/test.csv"'"' \
      --text-column '"'"$TEXT_COLUMN"'"' \
      --model-name '"'"$TEXT_MODEL_NAME"'"' \
      --pooling '"'"$TEXT_POOLING"'"' \
      --batch-size '"'"$TEXT_BATCH_SIZE"'"' \
      --max-length '"'"$TEXT_MAX_LENGTH"'"' \
      --device cuda \
      --output-path '"'"$embed_dir/test_text2_matscibert_mean.pt"'"'

    '"'"$PYTHON_BIN"'"' diffcsp/run.py \
      '"'"data=$data_name"'"' \
      model=flow_polar_text \
      optim.optimizer.lr=1e-3 \
      optim.optimizer.weight_decay=0 \
      optim.lr_scheduler.factor=0.6 \
      +model.lattice_polar_sigma=0.1 \
      model.cost_coord=10 \
      model.cost_lattice=1 \
      model.decoder.num_freqs=256 \
      model.decoder.rec_emb=sin \
      model.decoder.num_millers=8 \
      +model.decoder.na_emb=0 \
      model.decoder.hidden_dim=512 \
      model.decoder.num_layers=6 \
      '"'"logging.wandb.mode=$WANDB_MODE"'"' \
      '"'"logging.wandb.project=$WANDB_PROJECT"'"' \
      '"'"expname=$exp_name"'"'
  ' >> "$log_file" 2>&1 &
  LAST_PID="$!"
}

{
  echo "===== Carbon-24 / Perov-5 Baseline + Text Launcher ====="
  echo "DATE=$(date '+%F %T')"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "WANDB_MODE=$WANDB_MODE"
  echo "WANDB_PROJECT=$WANDB_PROJECT"
  echo "TEXT_COLUMN=$TEXT_COLUMN"
  echo "TEXT_MODEL_NAME=$TEXT_MODEL_NAME"
  echo "CARBON_BASELINE_GPU=$CARBON_BASELINE_GPU"
  echo "CARBON_TEXT_GPU=$CARBON_TEXT_GPU"
  echo "PEROV_BASELINE_GPU=$PEROV_BASELINE_GPU"
  echo "PEROV_TEXT_GPU=$PEROV_TEXT_GPU"
  echo "MASTER_LOG=$MASTER_LOG"
  echo
} | tee "$MASTER_LOG"

start_baseline "$CARBON_BASELINE_GPU" carbon_24 "$CARBON_BASELINE_EXP" "$CARBON_BASELINE_LOG"
CARBON_BASELINE_PID="$LAST_PID"
start_text_pipeline "$CARBON_TEXT_GPU" carbon_24 carbon_24_text "$CARBON_TEXT_EXP" "$CARBON_TEXT_LOG"
CARBON_TEXT_PID="$LAST_PID"
start_baseline "$PEROV_BASELINE_GPU" perov_5 "$PEROV_BASELINE_EXP" "$PEROV_BASELINE_LOG"
PEROV_BASELINE_PID="$LAST_PID"
start_text_pipeline "$PEROV_TEXT_GPU" perov_5 perov_5_text "$PEROV_TEXT_EXP" "$PEROV_TEXT_LOG"
PEROV_TEXT_PID="$LAST_PID"

{
  echo "CARBON_BASELINE_PID=$CARBON_BASELINE_PID"
  echo "CARBON_TEXT_PID=$CARBON_TEXT_PID"
  echo "PEROV_BASELINE_PID=$PEROV_BASELINE_PID"
  echo "PEROV_TEXT_PID=$PEROV_TEXT_PID"
} | tee -a "$MASTER_LOG"