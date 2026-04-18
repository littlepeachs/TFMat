#!/bin/bash
# Train CGCNN for band gap on GPU 3

export CUDA_VISIBLE_DEVICES=3

echo "=========================================="
echo "Training CGCNN for Band Gap"
echo "GPU: 3"
echo "=========================================="

python scripts/train_cgcnn_for_properties.py \
    --train-csv data_text/mp_20/train.csv \
    --val-csv data_text/mp_20/val.csv \
    --target-property band_gap \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --atom-fea-len 64 \
    --h-fea-len 128 \
    --n-conv 3 \
    --n-h 1 \
    --output-dir ./cgcnn_models \
    2>&1 | tee logs/train_cgcnn_bg_gpu3.log

echo "Band gap model training completed!"
