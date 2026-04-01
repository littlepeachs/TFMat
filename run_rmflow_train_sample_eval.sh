#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/ssd/liwentao/GenAI/CrystalMF/CrystalFlow"
PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-mp_20}"
EPOCHS="${EPOCHS:-50}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
POST_LAMBDA_NLL="${POST_LAMBDA_NLL:-5e-2}"
SAMPLE_STEPS="${SAMPLE_STEPS:-1}"
NUM_EVALS="${NUM_EVALS:-1}"
N_JOBS="${N_JOBS:-32}"
TEST_BS="${TEST_BS:-}"
WANDB_GROUP="${WANDB_GROUP:-${DATASET}_rmflow_pipeline}"
EXP_PREFIX="${EXP_PREFIX:-crystalmf-rmflow}"
TIMESTAMP="$(date +%m%d-%H%M%S)"
EXP="${EXP:-${EXP_PREFIX}-${TIMESTAMP}}"
RUN_DIR="$ROOT_DIR/hydra_jobs/singlerun/$EXP"
LABEL="${LABEL:-rmflow_${SAMPLE_STEPS}step_e${EPOCHS}}"

if (( PRETRAIN_EPOCHS > EPOCHS )); then
  PRETRAIN_EPOCHS="$EPOCHS"
fi

cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HYDRA_FULL_ERROR=1

run_stage() {
  local stage_name="$1"
  local max_epochs="$2"
  local lambda_nll="$3"

  echo "$stage_name"
  echo "  max_epochs=$max_epochs"
  echo "  lambda_nll=$lambda_nll"

  "$PYTHON_BIN" diffcsp/run.py \
    data="$DATASET" \
    data.train_max_epochs="$max_epochs" \
    model=flow_polar_rm \
    model.lambda_nll="$lambda_nll" \
    expname="$EXP" \
    logging.wandb.group="$WANDB_GROUP" \
    optim.optimizer.lr=1e-3 \
    optim.optimizer.weight_decay=0 \
    optim.lr_scheduler.factor=0.6 \
    model.cost_coord=1 \
    model.cost_lattice=1 \
    model.decoder.num_freqs=256 \
    model.decoder.rec_emb=sin \
    model.decoder.num_millers=8 \
    +model.decoder.na_emb=0 \
    model.decoder.hidden_dim=512 \
    model.decoder.num_layers=6 \
    +train.pl_trainer.log_every_n_steps=20 \
    train.pl_trainer.num_sanity_val_steps=0 \
    logging.progress_bar_refresh_rate=10
}

echo "[1/3] Training started"
echo "  EXP=$EXP"
echo "  RUN_DIR=$RUN_DIR"
echo "  DATASET=$DATASET"
echo "  EPOCHS=$EPOCHS"
echo "  PRETRAIN_EPOCHS=$PRETRAIN_EPOCHS"
echo "  GPU_ID=$GPU_ID"

run_stage "[1a/3] MeanFlow pretrain" "$PRETRAIN_EPOCHS" 0.0

if (( PRETRAIN_EPOCHS < EPOCHS )); then
  run_stage "[1b/3] RMFlow post-train" "$EPOCHS" "$POST_LAMBDA_NLL"
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

BEST_CKPT="$(ls -1t "$RUN_DIR"/*.ckpt 2>/dev/null | head -n 1 || true)"
if [[ -z "$BEST_CKPT" ]]; then
  echo "No checkpoint found under $RUN_DIR" >&2
  exit 1
fi

echo "[2/3] Sampling started"
echo "  CHECKPOINT=$BEST_CKPT"
echo "  SAMPLE_STEPS=$SAMPLE_STEPS"
echo "  NUM_EVALS=$NUM_EVALS"
echo "  LABEL=$LABEL"

EVAL_ARGS=(
  --model_path "$RUN_DIR"
  --dataset "$DATASET"
  --ode-int-steps "$SAMPLE_STEPS"
  --num_evals "$NUM_EVALS"
  --label "$LABEL"
)

if [[ -n "$TEST_BS" ]]; then
  EVAL_ARGS+=(--test_bs "$TEST_BS")
fi

"$PYTHON_BIN" scripts/evaluate.py "${EVAL_ARGS[@]}"

echo "[3/3] Structure quality check started"
"$PYTHON_BIN" scripts/compute_metrics.py \
  --root_path "$RUN_DIR" \
  --tasks csp \
  --gt_file "data/${DATASET}/test.csv" \
  --label "$LABEL" \
  -j "$N_JOBS"

DIFF_FILE="$RUN_DIR/eval_diff_${LABEL}.pt"
METRICS_FILE="$RUN_DIR/eval_metrics_${LABEL}.json"

echo "Pipeline finished"
echo "  RUN_DIR=$RUN_DIR"
echo "  CHECKPOINT=$BEST_CKPT"
echo "  SAMPLE_FILE=$DIFF_FILE"
echo "  METRICS_FILE=$METRICS_FILE"

if [[ -f "$METRICS_FILE" ]]; then
  echo "===== Metrics JSON ====="
  cat "$METRICS_FILE"
  echo
fi