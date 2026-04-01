#!/bin/bash

# Clean up new folders in hydra_jobs/singlerun
echo "Cleaning up new folders..."
find ./hydra_jobs/singlerun -maxdepth 1 -type d -name "*new" -exec rm -rf {} +
echo "Cleanup complete."

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 nohup python diffcsp/run.py \
data=mp_20 data.train_max_epochs=3000 \
model=flow_polar \
optim.optimizer.lr=1e-3 \
optim.optimizer.weight_decay=0 \
optim.lr_scheduler.factor=0.6 \
+model.lattice_polar_sigma=0.1 \
model.cost_coord=10 model.cost_lattice=1 \
model.decoder.num_freqs=256 \
model.decoder.rec_emb=sin model.decoder.num_millers=8 \
+model.decoder.na_emb=0 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
logging.wandb.mode=online \
logging.wandb.project=crystalflow-gridtest \
expname=CSP-mp20-new &> CSP-mp20-new.log &  