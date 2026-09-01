# TFMat Project Structure

## Overview

This document describes the organization of the TFMat codebase and explains the role of each major component.

## Directory Structure

```
CrystalFlow/
├── conf/                           # Hydra configuration files
│   ├── data/
│   │   ├── mp_20.yaml             # MP-20 dataset config (baseline)
│   │   ├── mp_20_text.yaml        # MP-20 with text conditioning
│   │   ├── perov_5.yaml           # Perov-5 dataset config
│   │   └── carbon_24.yaml         # Carbon-24 dataset config
│   ├── model/
│   │   ├── flow_polar.yaml        # CSP flow model (baseline)
│   │   ├── flow_polar_text.yaml   # CSP with text conditioning
│   │   └── flow_polar_w_type.yaml # DNG model; text conditioning comes from data=*_text
│   ├── optim/
│   │   └── default.yaml           # Optimizer and scheduler settings
│   └── train/
│       └── default.yaml           # Training configuration
│
├── diffcsp/                        # Core implementation
│   ├── pl_modules/
│   │   ├── flow_model.py          # Flow matching model
│   │   ├── text_encoder.py        # MatSciBERT text encoder
│   │   └── decoder.py             # Structure decoder
│   ├── pl_data/
│   │   ├── datamodule.py          # PyTorch Lightning data module
│   │   └── dataset.py             # Dataset with text loading
│   ├── common/
│   │   ├── data_utils.py          # Data processing utilities
│   │   └── constants.py           # Physical constants
│   └── run.py                     # Main training entry point
│
├── scripts/                        # Evaluation and analysis
│   ├── evaluate.py                # CSP evaluation script
│   ├── generation.py              # DNG evaluation script
│   ├── compute_metrics.py         # Metric computation
│   ├── eval_utils.py              # Evaluation utilities
│   │
│   ├── compute_text_prompt_match_with_cgcnn.py  # Property prediction
│   ├── compute_element_composition_match.py     # Element matching
│   ├── analyze_prediction_errors.py             # Error analysis
│   ├── analyze_element_match_distribution.py    # Match distribution
│   │
│   ├── plot_property_distributions.py           # Property plots
│   ├── plot_prediction_scatter_and_save.py      # Scatter plots
│   ├── plot_text_embedding_tsne.py              # t-SNE visualization
│   └── plot_dng_generation_compare_tsne.py      # Generation comparison
│
├── data_text/                      # Text-annotated datasets (not in repo)
│   ├── perov_5/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── precomputed_embeddings/
│   ├── carbon_24/
│   └── mp_20/
│
├── docs/                           # Documentation
│   ├── main.tex                   # Paper manuscript
│   ├── PROJECT_STRUCTURE.md       # This file
│   ├── TABLE4_EXPERIMENT_RESULTS.md  # Experiment results
│   └── table4_experiment_guide.md    # Experiment guide
│
├── result/                         # Evaluation results
│   ├── results-1.csv              # Single-sample CSP results
│   └── results-20.csv             # Multi-sample CSP results
│
├── README.md                       # Original CrystalFlow README
├── README_TFMAT.md                 # TFMat-specific README
├── .gitignore                      # Git ignore rules
├── .env.template                   # Environment template
└── setup.py                        # Package installation
```

## Key Components

### 1. Configuration System (conf/)

TFMat uses Hydra for configuration management. Key config groups:

- **data**: Dataset specifications, batch sizes, data paths
- **model**: Model architecture, text encoder settings, conditioning parameters
- **optim**: Learning rate, weight decay, scheduler settings
- **train**: Training epochs, devices, logging

### 2. Core Model (diffcsp/)

#### Flow Matching Model (`pl_modules/flow_model.py`)

- Implements continuous normalizing flow
- Predicts velocity fields for lattice and coordinates
- Supports classifier-free guidance
- Handles both CSP and DNG tasks

#### Text Encoder (`pl_modules/text_encoder.py`)

- Wraps MatSciBERT for semantic encoding
- Mean pooling over token embeddings
- Projects to conditioning space
- Frozen during training (optional)

#### Decoder (`pl_modules/decoder.py`)

- Processes noisy crystal states
- Incorporates text conditions via cross-attention or concatenation
- Outputs velocity predictions

### 3. Data Pipeline (diffcsp/pl_data/)

#### Dataset (`dataset.py`)

- Loads crystal structures from CSV
- Parses text descriptions
- Optionally loads precomputed embeddings
- Handles composition and property annotations

#### DataModule (`datamodule.py`)

- PyTorch Lightning data module
- Manages train/val/test splits
- Configures batch sizes and workers

### 4. Evaluation Scripts (scripts/)

#### Core Evaluation

- `evaluate.py`: CSP evaluation with guided sampling
- `generation.py`: DNG evaluation
- `compute_metrics.py`: Match rate, RMSE, validity metrics

#### Analysis Scripts

- `compute_text_prompt_match_with_cgcnn.py`: Train CGCNN surrogates and predict properties
- `compute_element_composition_match.py`: Compute element-level matching scores
- `analyze_prediction_errors.py`: Detailed error analysis
- `analyze_element_match_distribution.py`: Match distribution statistics

#### Visualization Scripts

- `plot_property_distributions.py`: Formation energy and band gap distributions
- `plot_prediction_scatter_and_save.py`: Ground truth vs predicted scatter plots
- `plot_text_embedding_tsne.py`: t-SNE of text embeddings
- `plot_dng_generation_compare_tsne.py`: Compare baseline vs text-conditioned

### 5. Documentation (docs/)

- `main.tex`: Full paper manuscript with results
- `TABLE4_EXPERIMENT_RESULTS.md`: Detailed experimental results
- `table4_experiment_guide.md`: Guide to reproducing experiments
- `PROJECT_STRUCTURE.md`: This file

## Data Format

### CSV Structure

Each dataset CSV contains:

```csv
material_id,pretty_formula,cif,spacegroup,formation_energy_per_atom,band_gap,text_description
mp-1234,NaCl,"CIF string...",225,-2.34,5.6,"Prototype: rocksalt, Space Group: 225, ..."
```

### Text Description Format

```
Prototype: <name>, Space Group: <number>, Formation Energy: <value> eV/atom, Band Gap: <value> eV
```

### Precomputed Embeddings

Optional `.pt` files containing:
```python
{
    'embeddings': torch.Tensor,  # [N, 768] MatSciBERT embeddings
    'material_ids': List[str]     # Corresponding material IDs
}
```

## Model Checkpoints

Checkpoints are saved in `hydra/singlerun/<expname>/` with structure:

```
<expname>/
├── checkpoints/
│   ├── epoch=1999-step=100000.ckpt
│   └── last.ckpt
├── config.yaml                    # Full Hydra config
├── train.log                      # Training logs
└── wandb/                         # W&B logs
```

## Evaluation Outputs

Evaluation scripts produce:

```
<checkpoint_dir>/
├── eval_gen_<label>.pt           # Generated structures
├── eval_metrics_<label>.json     # Computed metrics
├── eval_gen_<label>.dir/         # Extracted CIF files
└── eval_log_<label>.txt          # Evaluation log
```

## Key Hyperparameters

### Text Conditioning

- `model.text_encoder`: "matscibert" (default)
- `model.text_condition_dim`: 768 (MatSciBERT hidden size)
- `model.condition_dropout`: 0.1 (for classifier-free guidance)

### Flow Matching

- `model.lattice_polar_sigma`: 0.1 (noise scale for lattice)
- `model.cost_coord`: 10 (coordinate loss weight)
- `model.cost_lattice`: 1 (lattice loss weight)
- `model.cost_type`: 10 (type loss weight, DNG only)

### Sampling

- `ode_int_steps`: 100 (number of integration steps)
- `anneal_slope`: 10 (coordinate annealing strength)
- `guidance_factor`: "auto" (adaptive guidance)

### Training

- `optim.optimizer.lr`: 1e-3
- `optim.lr_scheduler.factor`: 0.6 (reduce on plateau)
- `data.train_max_epochs`: 2000

## Reproducing Paper Results

### Table 1: Single-Sample CSP

```bash
# Perov-5
python scripts/evaluate.py --model_path <ckpt> --dataset perov_5 --ode-int-steps 100 --anneal_slope 10 --label perov5_s100_a10

# Carbon-24
python scripts/evaluate.py --model_path <ckpt> --dataset carbon_24 --ode-int-steps 100 --anneal_slope 10 --label carbon24_s100_a10

# MP-20
python scripts/evaluate.py --model_path <ckpt> --dataset mp_20 --ode-int-steps 100 --anneal_slope 10 --label mp20_s100_a10
```

### Table 2: MP-20 with num_eval=20

```bash
python scripts/evaluate.py --model_path <ckpt> --dataset mp_20 --ode-int-steps 100 --num_evals 20 --anneal_slope 10 --label mp20_s100_a10_n20
```

### Table 3: MP-20 DNG

```bash
python scripts/generation.py --model_path <ckpt> --dataset mp_20 --ode-int-steps 100 --anneal_slope 10 --label mp20_dng_s100_a10
```

### Figure 3: t-SNE Visualization

```bash
python scripts/plot_text_embedding_tsne.py --baseline_file <baseline.pt> --text_file <tfmat.pt> --test_file data_text/mp_20/test.csv
```

### Figure 4: Property Validation

```bash
python scripts/compute_text_prompt_match_with_cgcnn.py --generated_file <tfmat.pt> --test_file data_text/mp_20/test.csv
python scripts/plot_property_distributions.py --generated_file <tfmat.pt> --test_file data_text/mp_20/test.csv
```

## Common Issues

### Out of Memory

- Reduce `data.datamodule.batch_size.train`
- Use gradient accumulation: `train.pl_trainer.accumulate_grad_batches=2`
- Enable mixed precision: `train.pl_trainer.precision=16`

### Slow Training

- Precompute text embeddings (see `scripts/convert_eval_gen_to_cached_text.py`)
- Increase `data.datamodule.num_workers`
- Use multiple GPUs with DDP

### Poor Match Rate

- Increase `anneal_slope` (try 15 or 20)
- Increase `ode_int_steps` (try 200)
- Adjust `guidance_factor` (try 1.0 or 1.2)

## Development Workflow

1. **Modify model**: Edit `diffcsp/pl_modules/`
2. **Update config**: Edit `conf/model/`
3. **Train**: Run `diffcsp/run.py` with Hydra overrides
4. **Evaluate**: Use `scripts/evaluate.py` or `scripts/generation.py`
5. **Analyze**: Run analysis scripts in `scripts/`
6. **Visualize**: Generate plots with `scripts/plot_*.py`

## Testing

```bash
# Quick smoke test (1 epoch, small batch)
python diffcsp/run.py data=mp_20_text model=flow_polar_text data.train_max_epochs=1 data.datamodule.batch_size.train=4 expname=smoke_test

# Evaluate on small subset
python scripts/evaluate.py --model_path <ckpt> --dataset mp_20 --num_batches_to_samples 2 --label smoke_eval
```

## Contributing

When adding new features:

1. Update relevant config files in `conf/`
2. Add documentation to this file
3. Include example usage in `README_TFMAT.md`
4. Add tests if applicable
5. Update `.gitignore` for new output files
