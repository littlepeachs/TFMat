#!/bin/bash
# Evaluation script for CrystalMF with different step counts
# Usage: bash scripts/eval_imf.sh <model_checkpoint_path>

# if [ -z "$1" ]; then
#     echo "Usage: bash scripts/eval_imf.sh <model_checkpoint_path>"
#     exit 1
# fi
/ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun/crystalmf-imf-mp20-baseline
MODEL_PATH=$1
# cd /ssd/liwentao/GenAI/CrystalMF/CrystalFlow

echo "=== Evaluating with 1-step (one-step iMF) ==="
python scripts/evaluate.py \
  --model_path /ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun/crystalmf-imf-mp20-baseline \
  --dataset mp_20 \
  --ode-int-steps 1 \
  --label imf_1step \
  --num_evals 1

echo "=== Evaluating with 4-step (few-step iMF) ==="
python scripts/evaluate.py \
  -m $MODEL_PATH \
  --dataset mp_20 \
  --ode-int-steps 4 \
  --label imf_4step \
  --num_evals 1

echo "=== Evaluating with 8-step (few-step iMF) ==="
python scripts/evaluate.py \
  -m $MODEL_PATH \
  --dataset mp_20 \
  --ode-int-steps 8 \
  --label imf_8step \
  --num_evals 1

echo "=== Evaluating with 100-step (baseline comparison) ==="
python scripts/evaluate.py \
  -m $MODEL_PATH \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --label imf_100step \
  --num_evals 1

echo "=== Evaluation complete. Results saved in $MODEL_PATH ==="
