#!/bin/bash
# Training script for CrystalMF (iMF adaptation of CrystalFlow)
# Usage: bash scripts/train_imf.sh

cd /ssd/liwentao/GenAI/CrystalMF/CrystalFlow

PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
GPU_ID="${GPU_ID:-0}"
EXP_NAME="${EXP_NAME:-crystalmf-imf-mp20-normp0}"
EPOCHS="${EPOCHS:-3000}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WARMUP_DATA_PROPORTION="${WARMUP_DATA_PROPORTION:-1.0}"
WARMUP_COST_COORD="${WARMUP_COST_COORD:-5}"
MIXED_DATA_PROPORTION="${MIXED_DATA_PROPORTION:-0.5}"
MIXED_COST_COORD="${MIXED_COST_COORD:-5}"

if (( WARMUP_EPOCHS > EPOCHS )); then
  WARMUP_EPOCHS="$EPOCHS"
fi

echo "[Phase 1/2] FM warmup"
echo "  EXP_NAME=$EXP_NAME"
echo "  EPOCHS=$EPOCHS"
echo "  WARMUP_EPOCHS=$WARMUP_EPOCHS"

CUDA_VISIBLE_DEVICES="$GPU_ID" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" diffcsp/run.py \
  data=mp_20 data.train_max_epochs="$WARMUP_EPOCHS" \
  model=flow_polar_imf \
  model.norm_p=0 \
  model.data_proportion="$WARMUP_DATA_PROPORTION" \
  optim.optimizer.lr=1e-3 \
  optim.optimizer.weight_decay=0 \
  optim.lr_scheduler.factor=0.6 \
  model.cost_coord="$WARMUP_COST_COORD" model.cost_lattice=1 \
  model.decoder.num_freqs=256 \
  model.decoder.rec_emb=sin model.decoder.num_millers=8 \
  +model.decoder.na_emb=0 \
  model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
  expname="$EXP_NAME" \
  logging.wandb.group=mp_20_imf \
  train.pl_trainer.max_epochs="$WARMUP_EPOCHS"

if (( WARMUP_EPOCHS < EPOCHS )); then
  echo "[Phase 2/2] mixed iMF"
  CUDA_VISIBLE_DEVICES="$GPU_ID" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" diffcsp/run.py \
    data=mp_20 data.train_max_epochs="$EPOCHS" \
    model=flow_polar_imf \
    model.norm_p=0 \
    model.data_proportion="$MIXED_DATA_PROPORTION" \
    optim.optimizer.lr=1e-3 \
    optim.optimizer.weight_decay=0 \
    optim.lr_scheduler.factor=0.6 \
    model.cost_coord="$MIXED_COST_COORD" model.cost_lattice=1 \
    model.decoder.num_freqs=256 \
    model.decoder.rec_emb=sin model.decoder.num_millers=8 \
    +model.decoder.na_emb=0 \
    model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
    expname="$EXP_NAME" \
    logging.wandb.group=mp_20_imf \
    train.pl_trainer.max_epochs="$EPOCHS"
fi
