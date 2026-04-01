#!/bin/bash
# Training script for CrystalFlow with RMFlow-style MeanFlow + NLL refinement.

cd /ssd/liwentao/GenAI/CrystalMF/CrystalFlow

PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-mp_20}"
EXP_NAME="${EXP_NAME:-crystalmf-rmflow-${DATASET}}"
EPOCHS="${EPOCHS:-3000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"
POST_LAMBDA_NLL="${POST_LAMBDA_NLL:-5e-2}"

if (( PRETRAIN_EPOCHS > EPOCHS )); then
  PRETRAIN_EPOCHS="$EPOCHS"
fi

run_stage() {
  local max_epochs="$1"
  local lambda_nll="$2"

  CUDA_VISIBLE_DEVICES="$GPU_ID" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" diffcsp/run.py \
    data="$DATASET" data.train_max_epochs="$max_epochs" \
    model=flow_polar_rm \
    model.lambda_nll="$lambda_nll" \
    optim.optimizer.lr=1e-3 \
    optim.optimizer.weight_decay=0 \
    optim.lr_scheduler.factor=0.6 \
    model.cost_coord=1 model.cost_lattice=1 \
    model.decoder.num_freqs=256 \
    model.decoder.rec_emb=sin model.decoder.num_millers=8 \
    +model.decoder.na_emb=0 \
    model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
    expname="$EXP_NAME" \
    logging.wandb.group="${DATASET}_rmflow" \
    train.pl_trainer.max_epochs="$max_epochs"
}

echo "[Phase 1/2] MeanFlow pretrain"
echo "  EXP_NAME=$EXP_NAME"
echo "  DATASET=$DATASET"
echo "  PRETRAIN_EPOCHS=$PRETRAIN_EPOCHS"
run_stage "$PRETRAIN_EPOCHS" 0.0

if (( PRETRAIN_EPOCHS < EPOCHS )); then
  echo "[Phase 2/2] RMFlow post-train"
  echo "  EPOCHS=$EPOCHS"
  echo "  POST_LAMBDA_NLL=$POST_LAMBDA_NLL"
  run_stage "$EPOCHS" "$POST_LAMBDA_NLL"
fi