#!/bin/bash
# Generate materials for Table 4 experiment using the trained DNG model

MODEL_PATH="/ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun/DNG-mp20-text-lattice5-periodic-last-gpu5"
CHECKPOINT="epoch=1834-step=194510.ckpt"

# Test dataset
TEST_CSV="data_text/mp_20/test.csv"

# Output label
LABEL="mp20_table4"

echo "=========================================="
echo "Generating materials for Table 4 experiment"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Test dataset: ${TEST_CSV}"
echo "Output label: ${LABEL}"
echo ""

# Run evaluation
python scripts/evaluate.py \
    -m "${MODEL_PATH}" \
    --num_evals 1 \
    --test_dataset_path "${TEST_CSV}" \
    --label "${LABEL}" \
    --dataset mp_20

echo ""
echo "=========================================="
echo "Generation completed!"
echo "Output file: ${MODEL_PATH}/eval_diff_${LABEL}.pt"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Train CGCNN models (see docs/table4_experiment_guide.md)"
echo "2. Run: python scripts/compute_text_prompt_match_with_cgcnn.py \\"
echo "        ${MODEL_PATH}/eval_diff_${LABEL}.pt \\"
echo "        --fe-model-path ./cgcnn_models/cgcnn_formation_energy_per_atom_best.pth \\"
echo "        --bg-model-path ./cgcnn_models/cgcnn_band_gap_best.pth \\"
echo "        --output results_table4_mp20.pt"
