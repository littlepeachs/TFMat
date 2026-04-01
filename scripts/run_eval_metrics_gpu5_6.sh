#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR/.."

WRAPPER="$ROOT_DIR/../run_eval_metrics_baseline_offline.sh"
PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
N_JOBS="${N_JOBS:-32}"
ODE_INT_STEPS="${ODE_INT_STEPS:-100}"
NUM_EVALS="${NUM_EVALS:-20}"
ANNEAL_SLOPE="${ANNEAL_SLOPE:-5}"
TEST_BS="${TEST_BS:-}"

# GPU assignment
BASELINE_GPU=5
OFFLINE_GPU=6

# Carbon-24 round
DATASET=carbon
BASELINE_RUN_DIR="$ROOT_DIR/../hydra_jobs/singlerun/CSP-carbon24-baseline-gpu0"
OFFLINE_RUN_DIR="$ROOT_DIR/../hydra_jobs/singlerun/CSP-carbon24-text-gpu1-offline-emb"
BASELINE_LABEL=carbon24_baseline_gpu5_texttest
OFFLINE_LABEL=carbon24_text_gpu6_texttest
BASELINE_GT_FILE="$ROOT_DIR/../data_text/carbon_24/test.csv"
OFFLINE_GT_FILE="$ROOT_DIR/../data_text/carbon_24/test.csv"
BASELINE_TEST_DATASET_PATH="$ROOT_DIR/../data_text/carbon_24/test.csv"
OFFLINE_TEST_DATASET_PATH="$ROOT_DIR/../data_text/carbon_24/test.csv"
BASELINE_TEST_SAVE_PATH="$ROOT_DIR/../data_text/carbon_24/test_baseline_eval.pt"
OFFLINE_TEST_SAVE_PATH="$ROOT_DIR/../data_text/carbon_24/test_text_eval.pt"
BASELINE_LOG="$ROOT_DIR/carbon24_baseline_eval_metrics_gpu5.log"
OFFLINE_LOG="$ROOT_DIR/carbon24_text_eval_metrics_gpu6.log"
SUMMARY_LOG="$ROOT_DIR/eval_metrics_carbon24_gpu56.log"

echo "=== Running carbon_24 eval+metrics on GPUs ${BASELINE_GPU}/${OFFLINE_GPU} ==="
RUN_BACKGROUND=0 \
  DATASET="$DATASET" \
  BASELINE_GPU="$BASELINE_GPU" \
  OFFLINE_GPU="$OFFLINE_GPU" \
  BASELINE_RUN_DIR="$BASELINE_RUN_DIR" \
  OFFLINE_RUN_DIR="$OFFLINE_RUN_DIR" \
  BASELINE_LABEL="$BASELINE_LABEL" \
  OFFLINE_LABEL="$OFFLINE_LABEL" \
  BASELINE_GT_FILE="$BASELINE_GT_FILE" \
  OFFLINE_GT_FILE="$OFFLINE_GT_FILE" \
  BASELINE_TEST_DATASET_PATH="$BASELINE_TEST_DATASET_PATH" \
  OFFLINE_TEST_DATASET_PATH="$OFFLINE_TEST_DATASET_PATH" \
  BASELINE_TEST_SAVE_PATH="$BASELINE_TEST_SAVE_PATH" \
  OFFLINE_TEST_SAVE_PATH="$OFFLINE_TEST_SAVE_PATH" \
  BASELINE_LOG="$BASELINE_LOG" \
  OFFLINE_LOG="$OFFLINE_LOG" \
  SUMMARY_LOG="$SUMMARY_LOG" \
  N_JOBS="$N_JOBS" \
  ODE_INT_STEPS="$ODE_INT_STEPS" \
  NUM_EVALS="$NUM_EVALS" \
  ANNEAL_SLOPE="$ANNEAL_SLOPE" \
  TEST_BS="$TEST_BS" \
  bash "$WRAPPER" all

# Perov-5 round
DATASET=perovskite
BASELINE_RUN_DIR="$ROOT_DIR/../hydra_jobs/singlerun/CSP-perov5-baseline-gpu4"
OFFLINE_RUN_DIR="$ROOT_DIR/../hydra_jobs/singlerun/CSP-perov5-text-gpu7-offline-emb"
BASELINE_LABEL=perov5_baseline_gpu5_texttest
OFFLINE_LABEL=perov5_text_gpu6_texttest
BASELINE_GT_FILE="$ROOT_DIR/../data_text/perov_5/test.csv"
OFFLINE_GT_FILE="$ROOT_DIR/../data_text/perov_5/test.csv"
BASELINE_TEST_DATASET_PATH="$ROOT_DIR/../data_text/perov_5/test.csv"
OFFLINE_TEST_DATASET_PATH="$ROOT_DIR/../data_text/perov_5/test.csv"
BASELINE_TEST_SAVE_PATH="$ROOT_DIR/../data_text/perov_5/test_baseline_eval.pt"
OFFLINE_TEST_SAVE_PATH="$ROOT_DIR/../data_text/perov_5/test_text_eval.pt"
BASELINE_LOG="$ROOT_DIR/perov5_baseline_eval_metrics_gpu5.log"
OFFLINE_LOG="$ROOT_DIR/perov5_text_eval_metrics_gpu6.log"
SUMMARY_LOG="$ROOT_DIR/eval_metrics_perov5_gpu56.log"

echo "=== Running perov_5 eval+metrics on GPUs ${BASELINE_GPU}/${OFFLINE_GPU} ==="
RUN_BACKGROUND=0 \
  DATASET="$DATASET" \
  BASELINE_GPU="$BASELINE_GPU" \
  OFFLINE_GPU="$OFFLINE_GPU" \
  BASELINE_RUN_DIR="$BASELINE_RUN_DIR" \
  OFFLINE_RUN_DIR="$OFFLINE_RUN_DIR" \
  BASELINE_LABEL="$BASELINE_LABEL" \
  OFFLINE_LABEL="$OFFLINE_LABEL" \
  BASELINE_GT_FILE="$BASELINE_GT_FILE" \
  OFFLINE_GT_FILE="$OFFLINE_GT_FILE" \
  BASELINE_TEST_DATASET_PATH="$BASELINE_TEST_DATASET_PATH" \
  OFFLINE_TEST_DATASET_PATH="$OFFLINE_TEST_DATASET_PATH" \
  BASELINE_TEST_SAVE_PATH="$BASELINE_TEST_SAVE_PATH" \
  OFFLINE_TEST_SAVE_PATH="$OFFLINE_TEST_SAVE_PATH" \
  BASELINE_LOG="$BASELINE_LOG" \
  OFFLINE_LOG="$OFFLINE_LOG" \
  SUMMARY_LOG="$SUMMARY_LOG" \
  N_JOBS="$N_JOBS" \
  ODE_INT_STEPS="$ODE_INT_STEPS" \
  NUM_EVALS="$NUM_EVALS" \
  ANNEAL_SLOPE="$ANNEAL_SLOPE" \
  TEST_BS="$TEST_BS" \
  bash "$WRAPPER" all

# MP20 round
DATASET=mp_20
BASELINE_RUN_DIR="$ROOT_DIR/../hydra_jobs/singlerun/CSP-mp20-baseline-gpu2"
OFFLINE_RUN_DIR="$ROOT_DIR/../hydra_jobs/singlerun/CSP-mp20-text-gpu3-offline-emb"
BASELINE_LABEL=mp20_baseline_gpu5_texttest_csp1
OFFLINE_LABEL=mp20_text_gpu6_texttest_csp1
BASELINE_GT_FILE="$ROOT_DIR/../data_text/mp_20/test.csv"
OFFLINE_GT_FILE="$ROOT_DIR/../data_text/mp_20/test.csv"
BASELINE_TEST_DATASET_PATH="$ROOT_DIR/../data_text/mp_20/test.csv"
OFFLINE_TEST_DATASET_PATH="$ROOT_DIR/../data_text/mp_20/test.csv"
BASELINE_TEST_SAVE_PATH="$ROOT_DIR/../data_text/mp_20/test_baseline_eval.pt"
OFFLINE_TEST_SAVE_PATH="$ROOT_DIR/../data_text/mp_20/test_text_eval.pt"
BASELINE_LOG="$ROOT_DIR/mp20_baseline_eval_metrics_gpu5.log"
OFFLINE_LOG="$ROOT_DIR/mp20_text_eval_metrics_gpu6.log"
SUMMARY_LOG="$ROOT_DIR/eval_metrics_mp20_gpu56.log"

echo "=== Running mp_20 eval+metrics on GPUs ${BASELINE_GPU}/${OFFLINE_GPU} ==="
RUN_BACKGROUND=0 \
  DATASET="$DATASET" \
  BASELINE_GPU="$BASELINE_GPU" \
  OFFLINE_GPU="$OFFLINE_GPU" \
  BASELINE_RUN_DIR="$BASELINE_RUN_DIR" \
  OFFLINE_RUN_DIR="$OFFLINE_RUN_DIR" \
  BASELINE_LABEL="$BASELINE_LABEL" \
  OFFLINE_LABEL="$OFFLINE_LABEL" \
  BASELINE_GT_FILE="$BASELINE_GT_FILE" \
  OFFLINE_GT_FILE="$OFFLINE_GT_FILE" \
  BASELINE_TEST_DATASET_PATH="$BASELINE_TEST_DATASET_PATH" \
  OFFLINE_TEST_DATASET_PATH="$OFFLINE_TEST_DATASET_PATH" \
  BASELINE_TEST_SAVE_PATH="$BASELINE_TEST_SAVE_PATH" \
  OFFLINE_TEST_SAVE_PATH="$OFFLINE_TEST_SAVE_PATH" \
  BASELINE_LOG="$BASELINE_LOG" \
  OFFLINE_LOG="$OFFLINE_LOG" \
  SUMMARY_LOG="$SUMMARY_LOG" \
  N_JOBS="$N_JOBS" \
  ODE_INT_STEPS="$ODE_INT_STEPS" \
  NUM_EVALS="$NUM_EVALS" \
  ANNEAL_SLOPE="$ANNEAL_SLOPE" \
  TEST_BS="$TEST_BS" \
  bash "$WRAPPER" all

echo "All done. Logs saved under scripts/ and root logs."