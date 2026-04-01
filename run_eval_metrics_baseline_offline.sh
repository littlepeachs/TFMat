#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/ssd/liwentao/miniconda/envs/adsorbdiff/bin/python}"
DATASET="${DATASET:-mp_20}"
GT_FILE="${GT_FILE:-data/mp_20/test.csv}"
BASELINE_GT_FILE="${BASELINE_GT_FILE:-$GT_FILE}"
OFFLINE_GT_FILE="${OFFLINE_GT_FILE:-data_text/mp_20/test.csv}"
BASELINE_TEST_DATASET_PATH="${BASELINE_TEST_DATASET_PATH:-$ROOT_DIR/data_text/mp_20/test.csv}"
OFFLINE_TEST_DATASET_PATH="${OFFLINE_TEST_DATASET_PATH:-$ROOT_DIR/data_text/mp_20/test.csv}"
BASELINE_TEST_SAVE_PATH="${BASELINE_TEST_SAVE_PATH:-$ROOT_DIR/data_text/mp_20/test_baseline_eval.pt}"
OFFLINE_TEST_SAVE_PATH="${OFFLINE_TEST_SAVE_PATH:-$ROOT_DIR/data_text/mp_20/test_text_eval.pt}"
ODE_INT_STEPS="${ODE_INT_STEPS:-100}"
NUM_EVALS="${NUM_EVALS:-1}"
ANNEAL_SLOPE="${ANNEAL_SLOPE:-5}"
TEST_BS="${TEST_BS:-}"
N_JOBS="${N_JOBS:-32}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"

BASELINE_GPU="${BASELINE_GPU:-5}"
OFFLINE_GPU="${OFFLINE_GPU:-6}"

BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$ROOT_DIR/hydra_jobs/singlerun/CSP-mp20-baseline-gpu2}"
OFFLINE_RUN_DIR="${OFFLINE_RUN_DIR:-$ROOT_DIR/hydra_jobs/singlerun/CSP-mp20-text-gpu3-offline-emb}"

BASELINE_LABEL="${BASELINE_LABEL:-baseline_gpu5_csp1}"
OFFLINE_LABEL="${OFFLINE_LABEL:-offline_gpu6_csp1}"

SUMMARY_LOG="${SUMMARY_LOG:-$ROOT_DIR/eval_metrics_baseline_offline_gpu56.log}"
BASELINE_LOG="${BASELINE_LOG:-$ROOT_DIR/baseline_eval_metrics_gpu5.log}"
OFFLINE_LOG="${OFFLINE_LOG:-$ROOT_DIR/offline_eval_metrics_gpu6.log}"

mkdir -p "$(dirname "$SUMMARY_LOG")"

run_pipeline() {
  local run_name="$1"
  local gpu_id="$2"
  local run_dir="$3"
  local label="$4"
  local log_file="$5"
  local gt_file="$6"
  local test_dataset_path="$7"
  local test_dataset_save_path="$8"

  if [[ ! -d "$run_dir" ]]; then
    echo "[$run_name] Missing run dir: $run_dir" >&2
    return 1
  fi

  local eval_args=(
    --model_path "$run_dir"
    --dataset "$DATASET"
    --ode-int-steps "$ODE_INT_STEPS"
    --num_evals "$NUM_EVALS"
    --anneal_coords
    --anneal_slope "$ANNEAL_SLOPE"
    --test_dataset_path "$test_dataset_path"
    --test_dataset_save_path "$test_dataset_save_path"
    --label "$label"
  )

  if [[ -n "$TEST_BS" ]]; then
    eval_args+=(--test_bs "$TEST_BS")
  fi

  {
    echo "===== $run_name ====="
    echo "DATE=$(date '+%F %T')"
    echo "GPU=$gpu_id"
    echo "RUN_DIR=$run_dir"
    echo "LABEL=$label"
    echo "DATASET=$DATASET"
    echo "GT_FILE=$gt_file"
    echo "TEST_DATASET_PATH=$test_dataset_path"
    echo "TEST_DATASET_SAVE_PATH=$test_dataset_save_path"
    echo "ODE_INT_STEPS=$ODE_INT_STEPS"
    echo "NUM_EVALS=$NUM_EVALS"
    echo "ANNEAL_SLOPE=$ANNEAL_SLOPE"
    if [[ -n "$TEST_BS" ]]; then
      echo "TEST_BS=$TEST_BS"
    fi
    echo "N_JOBS=$N_JOBS"
    echo
    echo "[1/2] Evaluate"
    echo "$PYTHON_BIN scripts/evaluate.py ${eval_args[*]}"
    echo
  } > "$log_file"

  CUDA_VISIBLE_DEVICES="$gpu_id" HYDRA_FULL_ERROR=1 "$PYTHON_BIN" scripts/evaluate.py "${eval_args[@]}" >> "$log_file" 2>&1

  {
    echo
    echo "[2/2] Compute metrics"
    echo "$PYTHON_BIN scripts/compute_metrics.py --root_path $run_dir --tasks csp --gt_file $GT_FILE --label $label -j $N_JOBS"
    echo
  } >> "$log_file"

  "$PYTHON_BIN" scripts/compute_metrics.py \
    --root_path "$run_dir" \
    --tasks csp \
    --gt_file "$gt_file" \
    --label "$label" \
    -j "$N_JOBS" >> "$log_file" 2>&1

  # If multiple evaluations were run (NUM_EVALS>1), tell compute_metrics to
  # treat outputs as multi-eval batches so it can aggregate correctly.
  # NUM_EVALS is exported into the environment by the wrapper script.
  cm_cmd=("$PYTHON_BIN" scripts/compute_metrics.py --root_path "$run_dir" --tasks csp --gt_file "$gt_file" --label "$label" -j "$N_JOBS")
  if [[ -n "${NUM_EVALS:-}" && ${NUM_EVALS} -gt 1 ]]; then
    cm_cmd+=(--multi_eval)
  fi
  "${cm_cmd[@]}" >> "$log_file" 2>&1

  {
    echo
    echo "DONE=$(date '+%F %T')"
    echo "DIFF_FILE=$run_dir/eval_diff_${label}.pt"
    echo "METRICS_FILE=$run_dir/eval_metrics_${label}.json"
  } >> "$log_file"
}

{
  echo "===== Baseline vs Offline Eval+Metrics ====="
  echo "DATE=$(date '+%F %T')"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "DATASET=$DATASET"
  echo "GT_FILE=$GT_FILE"
  echo "BASELINE_GT_FILE=$BASELINE_GT_FILE"
  echo "OFFLINE_GT_FILE=$OFFLINE_GT_FILE"
  echo "BASELINE_TEST_DATASET_PATH=$BASELINE_TEST_DATASET_PATH"
  echo "OFFLINE_TEST_DATASET_PATH=$OFFLINE_TEST_DATASET_PATH"
  echo "BASELINE_TEST_SAVE_PATH=$BASELINE_TEST_SAVE_PATH"
  echo "OFFLINE_TEST_SAVE_PATH=$OFFLINE_TEST_SAVE_PATH"
  echo "ODE_INT_STEPS=$ODE_INT_STEPS"
  echo "NUM_EVALS=$NUM_EVALS"
  echo "ANNEAL_SLOPE=$ANNEAL_SLOPE"
  echo "N_JOBS=$N_JOBS"
  echo "BASELINE_GPU=$BASELINE_GPU"
  echo "OFFLINE_GPU=$OFFLINE_GPU"
  echo "BASELINE_RUN_DIR=$BASELINE_RUN_DIR"
  echo "OFFLINE_RUN_DIR=$OFFLINE_RUN_DIR"
  echo "BASELINE_LABEL=$BASELINE_LABEL"
  echo "OFFLINE_LABEL=$OFFLINE_LABEL"
  echo "BASELINE_LOG=$BASELINE_LOG"
  echo "OFFLINE_LOG=$OFFLINE_LOG"
  echo
} | tee "$SUMMARY_LOG"

if [[ "$RUN_BACKGROUND" == "1" ]]; then
  nohup env \
    RUN_BACKGROUND=0 \
    BASELINE_GPU="$BASELINE_GPU" \
    OFFLINE_GPU="$OFFLINE_GPU" \
    BASELINE_RUN_DIR="$BASELINE_RUN_DIR" \
    OFFLINE_RUN_DIR="$OFFLINE_RUN_DIR" \
    BASELINE_LABEL="$BASELINE_LABEL" \
    OFFLINE_LABEL="$OFFLINE_LABEL" \
    BASELINE_LOG="$BASELINE_LOG" \
    OFFLINE_LOG="$OFFLINE_LOG" \
    SUMMARY_LOG="$SUMMARY_LOG" \
    PYTHON_BIN="$PYTHON_BIN" \
    DATASET="$DATASET" \
    GT_FILE="$GT_FILE" \
    ODE_INT_STEPS="$ODE_INT_STEPS" \
    NUM_EVALS="$NUM_EVALS" \
    ANNEAL_SLOPE="$ANNEAL_SLOPE" \
    TEST_BS="$TEST_BS" \
    N_JOBS="$N_JOBS" \
    bash "$0" baseline_only >/dev/null 2>&1 &
  BASELINE_PID=$!
  nohup env \
    RUN_BACKGROUND=0 \
    BASELINE_GPU="$BASELINE_GPU" \
    OFFLINE_GPU="$OFFLINE_GPU" \
    BASELINE_RUN_DIR="$BASELINE_RUN_DIR" \
    OFFLINE_RUN_DIR="$OFFLINE_RUN_DIR" \
    BASELINE_LABEL="$BASELINE_LABEL" \
    OFFLINE_LABEL="$OFFLINE_LABEL" \
    BASELINE_LOG="$BASELINE_LOG" \
    OFFLINE_LOG="$OFFLINE_LOG" \
    SUMMARY_LOG="$SUMMARY_LOG" \
    PYTHON_BIN="$PYTHON_BIN" \
    DATASET="$DATASET" \
    GT_FILE="$GT_FILE" \
    ODE_INT_STEPS="$ODE_INT_STEPS" \
    NUM_EVALS="$NUM_EVALS" \
    ANNEAL_SLOPE="$ANNEAL_SLOPE" \
    TEST_BS="$TEST_BS" \
    N_JOBS="$N_JOBS" \
    bash "$0" offline_only >/dev/null 2>&1 &
  OFFLINE_PID=$!
  {
    echo "BASELINE_PID=$BASELINE_PID"
    echo "OFFLINE_PID=$OFFLINE_PID"
  } | tee -a "$SUMMARY_LOG"
  exit 0
fi

MODE="${1:-all}"

case "$MODE" in
  baseline_only)
    run_pipeline "baseline" "$BASELINE_GPU" "$BASELINE_RUN_DIR" "$BASELINE_LABEL" "$BASELINE_LOG" "$BASELINE_GT_FILE" "$BASELINE_TEST_DATASET_PATH" "$BASELINE_TEST_SAVE_PATH"
    ;;
  offline_only)
    run_pipeline "offline_text" "$OFFLINE_GPU" "$OFFLINE_RUN_DIR" "$OFFLINE_LABEL" "$OFFLINE_LOG" "$OFFLINE_GT_FILE" "$OFFLINE_TEST_DATASET_PATH" "$OFFLINE_TEST_SAVE_PATH"
    ;;
  all)
    run_pipeline "baseline" "$BASELINE_GPU" "$BASELINE_RUN_DIR" "$BASELINE_LABEL" "$BASELINE_LOG" "$BASELINE_GT_FILE" "$BASELINE_TEST_DATASET_PATH" "$BASELINE_TEST_SAVE_PATH" &
    BASELINE_PID=$!
    run_pipeline "offline_text" "$OFFLINE_GPU" "$OFFLINE_RUN_DIR" "$OFFLINE_LABEL" "$OFFLINE_LOG" "$OFFLINE_GT_FILE" "$OFFLINE_TEST_DATASET_PATH" "$OFFLINE_TEST_SAVE_PATH" &
    OFFLINE_PID=$!
    wait "$BASELINE_PID"
    wait "$OFFLINE_PID"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac

echo "Finished. Logs:"
echo "  $BASELINE_LOG"
echo "  $OFFLINE_LOG"