#!/bin/bash
# Train CGCNN for formation energy on GPU 2

export CUDA_VISIBLE_DEVICES=2

echo "=========================================="
echo "Training CGCNN for Formation Energy"
echo "GPU: 2"
echo "=========================================="

python scripts/train_cgcnn_for_properties.py \
    --train-csv data_text/mp_20/train.csv \
    --val-csv data_text/mp_20/val.csv \
    --target-property formation_energy_per_atom \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --atom-fea-len 64 \
    --h-fea-len 128 \
    --n-conv 3 \
    --n-h 1 \
    --output-dir ./cgcnn_models \
    2>&1 | tee logs/train_cgcnn_fe_gpu2.log

echo "Formation energy model training completed!"
