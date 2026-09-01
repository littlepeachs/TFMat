# TFMat: Text-Conditioned Flow Matching for Crystal Structure Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TFMat extends CrystalFlow by introducing text-conditioned flow matching for crystal structure generation. The model maps structured material descriptions to semantic condition vectors and uses them to guide continuous flow from noise to target crystal structures.

## Key Features

- **Text-conditioned generation**: Uses MatSciBERT to encode structural prototypes, symmetry information, and scalar properties
- **Flow matching backbone**: Predicts lattice and fractional-coordinate velocities along continuous trajectories
- **Classifier-free guidance**: Enables flexible control over generation through guidance factors
- **Dual-mode operation**: Supports both crystal structure prediction (CSP) and de novo generation (DNG)

## Performance Highlights

### Crystal Structure Prediction (Single-Sample)

| Dataset | TFMat Match Rate | CrystalFlow Match Rate | Improvement |
|---------|------------------|------------------------|-------------|
| Perov-5 | **91.33%** | 53.69% | +37.64% |
| Carbon-24 | **46.40%** | 15.02% | +31.38% |
| MP-20 | **77.80%** | 67.65% | +10.15% |

### MP-20 with num_eval=20

- **Match Rate**: 92.04% (highest among all compared methods)
- Outperforms TGDMat (Long) at 82.02% and CrystalFlow at 85.38%

### De Novo Generation on MP-20

- **Number of Elements EMD**: 0.1990 (best)
- **Density EMD**: 0.0794 (best)
- **Structural Validity**: 99.06%
- **Coverage Precision**: 99.67%

## Important Notes

### Analysis Scripts

Some analysis scripts require experimental results to be generated first:
- `plot_property_distributions.py` - Requires top2000 results file
- `compute_element_composition_match.py` - Requires generated CIF files
- `plot_text_embedding_tsne.py` - Requires generated structures

**Workflow**: Run evaluation scripts first to generate these files, then run analysis scripts.

### Training with Hydra

Always specify the data configuration when running training:

```bash
# ✓ Correct
python diffcsp/run.py data=mp_20_text model=flow_polar_text

# ✗ Incorrect (will fail)
python diffcsp/run.py --help
```

### PyTorch Geometric Warnings

You may see warnings about `pyg-lib` and `torch-sparse`. These do not affect core functionality but may impact performance of certain graph operations. To resolve (optional):

```bash
# Reinstall PyG extensions matching your PyTorch version
pip uninstall pyg-lib torch-sparse torch-scatter torch-cluster -y
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-<version>+<cuda>.html
```

## Installation

### Dependencies

```bash
conda create -n tfmat python=3.11.9
conda activate tfmat

# PyTorch and PyG
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Core dependencies
pip install lightning==2.3.2 finetuning_scheduler
pip install hydra-core omegaconf python-dotenv wandb swanlab rich
pip install p_tqdm pymatgen pyxtal smact matminer einops chemparse torchdyn

# Transformers for text encoding
pip install transformers

# Install package
pip install -e .

# Create directories
mkdir -p log hydra
```

### Environment Configuration

Create a `.env` file from the template:

```bash
cp .env.template .env
```

Edit `.env` to specify:

```bash
export PROJECT_ROOT="/path/to/CrystalFlow"
export HYDRA_JOBS="/path/to/CrystalFlow/hydra"
export WANDB_DIR="/path/to/CrystalFlow/log"
```

## Datasets

The model requires text-annotated datasets in `data_text/` directory:

- `data_text/perov_5/`: Perovskite structures
- `data_text/carbon_24/`: Carbon allotropes
- `data_text/mp_20/`: Materials Project subset

Each dataset should contain:
- `train.csv`, `val.csv`, `test.csv`: Structure data with text descriptions
- `precomputed_embeddings/`: Pre-computed MatSciBERT embeddings (optional, for faster training)

## Training

### CSP Task on MP-20 with Text Conditioning

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python diffcsp/run.py \
  data=mp_20_text \
  data.train_max_epochs=2000 \
  model=flow_polar_text \
  model.text_encoder=matscibert \
  model.text_condition_dim=768 \
  model.condition_dropout=0.1 \
  optim.optimizer.lr=1e-3 \
  optim.optimizer.weight_decay=0 \
  optim.lr_scheduler.factor=0.6 \
  +model.lattice_polar_sigma=0.1 \
  model.cost_coord=10 \
  model.cost_lattice=1 \
  model.decoder.num_freqs=256 \
  model.decoder.hidden_dim=512 \
  model.decoder.num_layers=6 \
  logging.wandb.mode=online \
  logging.wandb.project=tfmat \
  expname=CSP-mp20-text \
  > CSP-mp20-text.log 2>&1 &
```

### DNG Task on MP-20 with Text Conditioning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HYDRA_FULL_ERROR=1 python diffcsp/run.py \
  train.pl_trainer.devices=4 \
  +train.pl_trainer.strategy=ddp_find_unused_parameters_true \
  data=mp_20_text \
  data.train_max_epochs=2000 \
  model=flow_polar_w_type \
  model.text_encoder=matscibert \
  model.text_condition_dim=768 \
  model.condition_dropout=0.1 \
  +model.type_encoding=table \
  +model.guide_threshold=-1 \
  optim.optimizer.lr=1e-3 \
  optim.optimizer.weight_decay=0 \
  optim.lr_scheduler.factor=0.6 \
  +model.lattice_polar_sigma=0.1 \
  model.cost_type=10 \
  model.cost_coord=10 \
  model.cost_lattice=1 \
  model.decoder.num_freqs=256 \
  model.decoder.hidden_dim=512 \
  model.decoder.num_layers=6 \
  logging.wandb.mode=online \
  logging.wandb.project=tfmat \
  expname=DNG-mp20-text \
  > DNG-mp20-text.log 2>&1 &
```

## Evaluation

### CSP Evaluation (Single Sample)

```bash
python scripts/evaluate.py \
  --model_path <checkpoint_path> \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --anneal_coords \
  --anneal_slope 10 \
  --guidance_factor auto \
  --label tfmat_s100_a10

python scripts/compute_metrics.py \
  --root_path <checkpoint_dir> \
  --tasks csp \
  --gt_file data_text/mp_20/test.csv \
  --label tfmat_s100_a10
```

### CSP Evaluation (Multiple Samples)

```bash
python scripts/evaluate.py \
  --model_path <checkpoint_path> \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --num_evals 20 \
  --anneal_coords \
  --anneal_slope 10 \
  --guidance_factor auto \
  --label tfmat_s100_a10_n20

python scripts/compute_metrics.py \
  --root_path <checkpoint_dir> \
  --tasks csp \
  --gt_file data_text/mp_20/test.csv \
  --multi_eval \
  --label tfmat_s100_a10_n20
```

### DNG Evaluation

```bash
python scripts/generation.py \
  --model_path <checkpoint_path> \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --anneal_coords \
  --anneal_slope 10 \
  --guidance_factor auto \
  --label tfmat_dng_s100_a10

python scripts/compute_metrics.py \
  --root_path <checkpoint_dir> \
  --tasks gen \
  --gt_file data_text/mp_20/test.csv \
  --label tfmat_dng_s100_a10
```

## Text Conditioning Format

Text descriptions should follow this structure:

```
Prototype: <prototype_name>, Space Group: <space_group_number>, 
Formation Energy: <value> eV/atom, Band Gap: <value> eV
```

Example:
```
Prototype: rocksalt, Space Group: 225, Formation Energy: -2.34 eV/atom, Band Gap: 2.1 eV
```

## Key Hyperparameters

### Sampling Configuration

- **ODE steps**: 100 (balance between quality and speed)
- **Annealing slope**: 10 (coordinate refinement strength)
- **Guidance factor**: auto (adaptive classifier-free guidance)

### Training Configuration

- **Condition dropout**: 0.1 (enables classifier-free guidance)
- **Text encoder**: MatSciBERT (frozen)
- **Condition dimension**: 768 (MatSciBERT hidden size)

## Analysis Scripts

### Property Prediction Validation

```bash
# Train CGCNN surrogate models
python scripts/compute_text_prompt_match_with_cgcnn.py \
  --generated_file <generated_structures.pt> \
  --test_file data_text/mp_20/test.csv

# Analyze prediction errors
python scripts/analyze_prediction_errors.py \
  --results_file <prediction_results.csv>

# Plot property distributions
python scripts/plot_property_distributions.py \
  --generated_file <generated_structures.pt> \
  --test_file data_text/mp_20/test.csv
```

### Element Composition Analysis

```bash
python scripts/compute_element_composition_match.py \
  --generated_file <generated_structures.pt> \
  --test_file data_text/mp_20/test.csv \
  --top_k 2000

python scripts/analyze_element_match_distribution.py \
  --match_file element_composition_match_results_5000.txt
```

### t-SNE Visualization

```bash
python scripts/plot_text_embedding_tsne.py \
  --baseline_file <crystalflow_generated.pt> \
  --text_file <tfmat_generated.pt> \
  --test_file data_text/mp_20/test.csv
```

## Project Structure

```
CrystalFlow/
├── conf/                      # Hydra configuration files
│   ├── data/                  # Dataset configs
│   ├── model/                 # Model configs (flow_polar_text, etc.)
│   ├── optim/                 # Optimizer configs
│   └── train/                 # Training configs
├── diffcsp/                   # Core model implementation
│   ├── pl_modules/            # PyTorch Lightning modules
│   ├── pl_data/               # Data modules
│   ├── common/                # Shared utilities
│   └── run.py                 # Main training script
├── scripts/                   # Evaluation and analysis scripts
│   ├── evaluate.py            # CSP evaluation
│   ├── generation.py          # DNG evaluation
│   ├── compute_metrics.py     # Metric computation
│   └── *.py                   # Analysis scripts
├── data_text/                 # Text-annotated datasets (not in repo)
├── docs/                      # Documentation and paper
│   ├── main.tex               # Paper manuscript
│   └── *.pdf                  # Figures
└── result/                    # Evaluation results
    ├── results-1.csv          # Single-sample results
    └── results-20.csv         # Multi-sample results
```

## Citation

If you use TFMat in your research, please cite:

```bibtex
@article{tfmat2024,
  title={TFMat: Text-Conditioned Flow Matching for Crystal Structure Generation},
  author={Anonymous},
  journal={arXiv preprint},
  year={2024}
}
```

Also cite the original CrystalFlow paper:

```bibtex
@article{luo2024crystalflow,
  title={CrystalFlow: A Flow-Based Generative Model for Crystalline Materials},
  author={Luo, Xiaoshan and others},
  journal={arXiv preprint arXiv:2412.11693},
  year={2024}
}
```

## Acknowledgments

This work builds upon:
- [CrystalFlow](https://github.com/ixsluo/CrystalFlow): Base flow matching framework
- [DiffCSP](https://github.com/txie-93/cdvae): Training infrastructure
- [CDVAE](https://github.com/txie-93/cdvae): Benchmark datasets
- [MatSciBERT](https://huggingface.co/m3rg-iitd/matscibert): Text encoder

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
