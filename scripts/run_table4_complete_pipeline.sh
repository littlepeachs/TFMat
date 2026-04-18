#!/bin/bash
# Complete pipeline for Table 4 experiment
# This script runs all steps: CGCNN training, material generation, and evaluation

set -e  # Exit on error

MODEL_PATH="/ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun/DNG-mp20-text-lattice5-periodic-last-gpu5"
TRAIN_CSV="data_text/mp_20/train.csv"
VAL_CSV="data_text/mp_20/val.csv"
TEST_CSV="data_text/mp_20/test.csv"
CGCNN_DIR="./cgcnn_models"

echo "=========================================="
echo "Table 4 Experiment - Complete Pipeline"
echo "=========================================="
echo ""

# Step 1: Train CGCNN for formation energy
echo "Step 1/4: Training CGCNN for formation energy prediction..."
echo "------------------------------------------"
if [ -f "${CGCNN_DIR}/cgcnn_formation_energy_per_atom_best.pth" ]; then
    echo "Formation energy model already exists. Skipping training."
else
    python scripts/train_cgcnn_for_properties.py \
        --train-csv "${TRAIN_CSV}" \
        --val-csv "${VAL_CSV}" \
        --target-property formation_energy_per_atom \
        --batch-size 32 \
        --epochs 100 \
        --lr 0.001 \
        --atom-fea-len 64 \
        --h-fea-len 128 \
        --n-conv 3 \
        --n-h 1 \
        --output-dir "${CGCNN_DIR}"
fi
echo ""

# Step 2: Train CGCNN for band gap
echo "Step 2/4: Training CGCNN for band gap prediction..."
echo "------------------------------------------"
if [ -f "${CGCNN_DIR}/cgcnn_band_gap_best.pth" ]; then
    echo "Band gap model already exists. Skipping training."
else
    python scripts/train_cgcnn_for_properties.py \
        --train-csv "${TRAIN_CSV}" \
        --val-csv "${VAL_CSV}" \
        --target-property band_gap \
        --batch-size 32 \
        --epochs 100 \
        --lr 0.001 \
        --atom-fea-len 64 \
        --h-fea-len 128 \
        --n-conv 3 \
        --n-h 1 \
        --output-dir "${CGCNN_DIR}"
fi
echo ""

# Step 3: Generate materials
echo "Step 3/4: Generating materials with DNG model..."
echo "------------------------------------------"
if [ -f "${MODEL_PATH}/eval_diff_mp20_table4.pt" ]; then
    echo "Generated materials already exist. Skipping generation."
else
    python scripts/evaluate.py \
        -m "${MODEL_PATH}" \
        --dataset mp_20 \
        --num_evals 1 \
        --test_dataset_path "${TEST_CSV}" \
        --label mp20_table4
fi
echo ""

# Step 4: Compute matching statistics
echo "Step 4/4: Computing matching statistics..."
echo "------------------------------------------"

# 4a: Structure properties only
echo "Computing structure properties (Formula, Space Group, Crystal System)..."
python scripts/compute_text_prompt_match_with_cgcnn.py \
    "${MODEL_PATH}/eval_diff_mp20_table4.pt" \
    --symprec 0.1 \
    --output results_table4_mp20_structure_only.pt

echo ""

# 4b: All properties with CGCNN
echo "Computing all properties (including Formation Energy and Band Gap with CGCNN)..."
python scripts/compute_text_prompt_match_with_cgcnn.py \
    "${MODEL_PATH}/eval_diff_mp20_table4.pt" \
    --fe-model-path "${CGCNN_DIR}/cgcnn_formation_energy_per_atom_best.pth" \
    --bg-model-path "${CGCNN_DIR}/cgcnn_band_gap_best.pth" \
    --symprec 0.1 \
    --output results_table4_mp20_full.pt

echo ""
echo "=========================================="
echo "Table 4 Experiment Completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results_table4_mp20_structure_only.pt (structure properties only)"
echo "  - results_table4_mp20_full.pt (all properties with CGCNN)"
echo ""
echo "Compare with TGDMat paper Table 4 (MP-20):"
echo "  - Formula: 70.54%"
echo "  - Space Group: 67.88%"
echo "  - Crystal System: 73.54%"
echo "  - Formation Energy: 92.88%"
echo "  - Band Gap: 96.73%"
