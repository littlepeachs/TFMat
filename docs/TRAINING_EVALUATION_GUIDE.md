# TFMat 训练和评估流程详解

## 概述

TFMat 通过对比 **baseline (CrystalFlow)** 和 **text-conditioned (TFMat)** 两个版本来验证文本条件的有效性。

---

## 🔑 关键区别

### Baseline vs Text-Conditioned

| 组件 | Baseline | Text-Conditioned |
|------|----------|------------------|
| **数据配置** | `data=mp_20` | `data=mp_20_text` |
| **模型配置** | `model=flow_polar` | `model=flow_polar_text` |
| **数据路径** | `data/mp_20/` | `data_text/mp_20/` |
| **条件信号** | `formation_energy_per_atom` | `text_embedding` |
| **文本编码器** | 无 | MatSciBERT |
| **预计算嵌入** | 无 | `precomputed_embeddings/*.pt` |

---

## 📁 数据准备

### 1. Baseline 数据 (`data/mp_20/`)

```
data/mp_20/
├── train.csv          # 训练集（标准格式）
├── val.csv            # 验证集
├── test.csv           # 测试集
├── train_ori.pt       # 预处理缓存
├── val_ori.pt
└── test_ori.pt
```

**CSV 格式**:
```csv
material_id,pretty_formula,cif,spacegroup,formation_energy_per_atom,band_gap
mp-1234,NaCl,"CIF string...",225,-2.34,5.6
```

### 2. Text-Conditioned 数据 (`data_text/mp_20/`)

```
data_text/mp_20/
├── train.csv          # 包含 text2 列
├── val.csv
├── test.csv
├── train_text.pt      # 预处理缓存（含文本）
├── val_text.pt
├── test_text.pt
└── precomputed_embeddings/
    ├── train_text2_matscibert_mean.pt    # 预计算的 MatSciBERT 嵌入
    ├── val_text2_matscibert_mean.pt
    └── test_text2_matscibert_mean.pt
```

**CSV 格式（额外的 text2 列）**:
```csv
material_id,pretty_formula,cif,spacegroup,formation_energy_per_atom,band_gap,text2
mp-1234,NaCl,"CIF string...",225,-2.34,5.6,"Prototype: rocksalt, Space Group: 225, Formation Energy: -2.34 eV/atom, Band Gap: 5.6 eV"
```

---

## 🚀 训练流程

### 方式 1：单独训练 Baseline

```bash
# CSP Baseline (MP-20)
CUDA_VISIBLE_DEVICES=0 python diffcsp/run.py \
  data=mp_20 \
  data.train_max_epochs=2000 \
  model=flow_polar \
  optim.optimizer.lr=1e-3 \
  optim.optimizer.weight_decay=0 \
  optim.lr_scheduler.factor=0.6 \
  +model.lattice_polar_sigma=0.1 \
  model.cost_coord=10 \
  model.cost_lattice=1 \
  model.decoder.num_freqs=256 \
  model.decoder.rec_emb=sin \
  model.decoder.num_millers=8 \
  +model.decoder.na_emb=0 \
  model.decoder.hidden_dim=512 \
  model.decoder.num_layers=6 \
  logging.wandb.mode=online \
  logging.wandb.project=tfmat \
  expname=CSP-mp20-baseline
```

**关键参数**:
- `data=mp_20` - 使用标准数据（无文本）
- `model=flow_polar` - 标准 flow matching 模型
- 条件信号：`formation_energy_per_atom`

### 方式 2：单独训练 Text-Conditioned

```bash
# CSP Text-Conditioned (MP-20)
CUDA_VISIBLE_DEVICES=1 python diffcsp/run.py \
  data=mp_20_text \
  data.text_column=text2 \
  data.train_max_epochs=2000 \
  model=flow_polar_text \
  optim.optimizer.lr=1e-3 \
  optim.optimizer.weight_decay=0 \
  optim.lr_scheduler.factor=0.6 \
  +model.lattice_polar_sigma=0.1 \
  model.cost_coord=10 \
  model.cost_lattice=1 \
  model.decoder.num_freqs=256 \
  model.decoder.rec_emb=sin \
  model.decoder.num_millers=8 \
  +model.decoder.na_emb=0 \
  model.decoder.hidden_dim=512 \
  model.decoder.num_layers=6 \
  logging.wandb.mode=online \
  logging.wandb.project=tfmat \
  expname=CSP-mp20-text
```

**关键参数**:
- `data=mp_20_text` - 使用文本标注数据
- `data.text_column=text2` - 指定文本列名
- `model=flow_polar_text` - 文本条件模型
- 条件信号：`text_embedding` (768维 MatSciBERT)

### 方式 3：并行对比训练（推荐）

使用提供的脚本同时训练两个版本：

```bash
# 使用对比训练脚本
GPU_BASELINE=0 GPU_TEXT=1 \
  EPOCHS=2000 \
  BASELINE_EXP=CSP-mp20-baseline \
  TEXT_EXP=CSP-mp20-text \
  bash all_results/old_scripts/run_flow_baseline_text_compare.sh
```

这会在两个 GPU 上并行训练：
- GPU 0: Baseline
- GPU 1: Text-Conditioned

---

## 📊 评估流程

### 1. CSP 评估（单样本）

#### Baseline
```bash
python scripts/evaluate.py \
  --model_path hydra_jobs/singlerun/CSP-mp20-baseline/epoch=2421-step=256732.ckpt \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --anneal_coords \
  --anneal_slope 10 \
  --label baseline_s100_a10

python scripts/compute_metrics.py \
  --root_path hydra_jobs/singlerun/CSP-mp20-baseline \
  --tasks csp \
  --gt_file data/mp_20/test.csv \
  --label baseline_s100_a10
```

#### Text-Conditioned
```bash
python scripts/evaluate.py \
  --model_path hydra_jobs/singlerun/CSP-mp20-text/epoch=1385-step=146916.ckpt \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --anneal_coords \
  --anneal_slope 10 \
  --guidance_factor auto \
  --label text_s100_a10_gfauto

python scripts/compute_metrics.py \
  --root_path hydra_jobs/singlerun/CSP-mp20-text \
  --tasks csp \
  --gt_file data_text/mp_20/test.csv \
  --label text_s100_a10_gfauto
```

**关键区别**:
- Text 版本使用 `--guidance_factor auto` 启用 classifier-free guidance
- Text 版本的 gt_file 使用 `data_text/mp_20/test.csv`

### 2. CSP 评估（多样本，num_eval=20）

```bash
# Baseline
python scripts/evaluate.py \
  --model_path <baseline_ckpt> \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --num_evals 20 \
  --anneal_coords \
  --anneal_slope 10 \
  --label baseline_s100_a10_n20

# Text-Conditioned
python scripts/evaluate.py \
  --model_path <text_ckpt> \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --num_evals 20 \
  --anneal_coords \
  --anneal_slope 10 \
  --guidance_factor auto \
  --label text_s100_a10_gfauto_n20
```

### 3. DNG 评估

#### Baseline
```bash
python scripts/generation.py \
  --model_path hydra_jobs/singlerun/DNG-mp20-baseline/epoch=2699-step=286200.ckpt \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --anneal_coords \
  --anneal_slope 10 \
  --label baseline_dng_s100_a10

python scripts/compute_metrics.py \
  --root_path hydra_jobs/singlerun/DNG-mp20-baseline \
  --tasks gen \
  --gt_file data/mp_20/test.csv \
  --label baseline_dng_s100_a10
```

#### Text-Conditioned
```bash
python scripts/generation.py \
  --model_path hydra_jobs/singlerun/DNG-mp20-text-lattice5-periodic-last-gpu5/epoch=1834-step=194510.ckpt \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --anneal_coords \
  --anneal_slope 10 \
  --guidance_factor auto \
  --label text_dng_s100_a10_gfauto

python scripts/compute_metrics.py \
  --root_path hydra_jobs/singlerun/DNG-mp20-text-lattice5-periodic-last-gpu5 \
  --tasks gen \
  --gt_file data_text/mp_20/test.csv \
  --label text_dng_s100_a10_gfauto
```

---

## 🔬 超参数扫描

论文中报告的最佳配置来自超参数扫描：

```bash
# 扫描 ODE steps
for steps in 100 200; do
  # 扫描 anneal slope
  for slope in 5 10; do
    # 扫描 guidance factor
    for gf in 0.8 1.2 auto; do
      python scripts/generation.py \
        --model_path <ckpt> \
        --ode-int-steps $steps \
        --anneal_slope $slope \
        --guidance_factor $gf \
        --label s${steps}_a${slope}_gf${gf}
    done
  done
done
```

**最佳配置（论文 Table 4）**:
- ODE steps: 100
- Anneal slope: 10
- Guidance factor: auto

---

## 📈 结果对比

### 论文报告的结果

| Dataset | Baseline | Text-Conditioned | 提升 |
|---------|----------|------------------|------|
| **Perov-5** | 53.69% | **91.33%** | +37.64% |
| **Carbon-24** | 15.02% | **46.40%** | +31.38% |
| **MP-20** | 67.65% | **77.80%** | +10.15% |
| **MP-20 (×20)** | 85.38% | **92.04%** | +6.66% |

### 结果文件位置

```
hydra_jobs/singlerun/
├── CSP-mp20-baseline/
│   ├── eval_diff_baseline_gpu5_csp1.pt
│   └── eval_metrics_baseline_s100_a10.json
└── CSP-mp20-text/
    ├── eval_diff_offline_gpu6_csp1.pt
    └── eval_metrics_text_s100_a10_gfauto.json
```

---

## 🎯 关键技术细节

### 1. Text Embedding

**预计算方式**:
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("m3rg-iitd/matscibert")
model = AutoModel.from_pretrained("m3rg-iitd/matscibert")

# 编码文本
inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
outputs = model(**inputs)

# Mean pooling
embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
```

**文本格式**:
```
Prototype: <name>, Space Group: <number>, Formation Energy: <value> eV/atom, Band Gap: <value> eV
```

### 2. Classifier-Free Guidance

在推理时，模型同时预测：
- 无条件速度场：`v_uncond = f(x_t, t, c=0)`
- 条件速度场：`v_cond = f(x_t, t, c=text_emb)`

然后线性插值：
```python
v = (1 - gamma) * v_uncond + gamma * v_cond
```

其中 `gamma` 是 guidance factor：
- `gamma=0`: 纯无条件生成
- `gamma=1`: 纯条件生成
- `gamma=auto`: 自适应（论文中最佳）

### 3. Condition Dropout

训练时随机丢弃条件（`p=0.1`）：
```python
if random.random() < 0.1:
    text_emb = torch.zeros_like(text_emb)
```

这使得模型学会同时处理有条件和无条件的情况。

---

## 🛠️ 实用脚本

### 快速训练脚本

```bash
# scripts/train_csp_text.sh
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python diffcsp/run.py \
  data=mp_20_text \
  model=flow_polar_text \
  data.train_max_epochs=2000 \
  expname=CSP-mp20-text-$(date +%Y%m%d)
```

### 快速评估脚本

```bash
# scripts/eval_csp_text.sh
#!/bin/bash
python scripts/evaluate.py \
  --model_path $1 \
  --dataset mp_20 \
  --ode-int-steps 100 \
  --anneal_coords \
  --anneal_slope 10 \
  --guidance_factor auto \
  --label $(basename $(dirname $1))
```

---

## 📝 常见问题

### Q1: 为什么需要预计算文本嵌入？

**A**: 加速训练。MatSciBERT 编码很慢，预计算后可以直接加载。

### Q2: 如何生成文本描述？

**A**: 从数据库字段组合：
```python
text = f"Prototype: {prototype}, Space Group: {spacegroup}, " \
       f"Formation Energy: {formation_energy:.2f} eV/atom, " \
       f"Band Gap: {band_gap:.2f} eV"
```

### Q3: Baseline 和 Text 版本可以用相同的超参数吗？

**A**: 是的！论文中两者使用相同的架构和超参数，只是条件信号不同。

### Q4: 如何选择 guidance factor？

**A**: 
- 从 `auto` 开始（自适应）
- 如果需要更强的文本控制，增加到 1.2-1.5
- 如果生成多样性不足，降低到 0.8

---

## 🎓 引用

如果使用此流程，请引用：

```bibtex
@article{tfmat2024,
  title={TFMat: Text-Conditioned Flow Matching for Crystal Structure Generation},
  author={Anonymous},
  journal={arXiv preprint},
  year={2024}
}
```

---

**文档版本**: 1.0  
**最后更新**: 2026-04-18  
**作者**: TFMat Team
