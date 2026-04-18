# Table 4 实验复现指南

本指南说明如何复现 TGDMat 论文中的 Table 4 实验,评估生成材料与文本提示条件的匹配度。

## 方法概述

由于我们的文本描述不包含明确的性质要求(如 "formation energy is negative"),我们使用以下方法:

1. **结构属性** (Formula, Space Group, Crystal System): 直接从生成的结构计算
2. **物理性质** (Formation Energy, Band Gap): 使用训练好的 CGCNN 模型预测,然后比较符号/类型是否匹配

## 步骤 1: 训练 CGCNN 模型

### 1.1 训练 Formation Energy 预测模型

```bash
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
    --output-dir ./cgcnn_models
```

这将保存模型到: `./cgcnn_models/cgcnn_formation_energy_per_atom_best.pth`

### 1.2 训练 Band Gap 预测模型

```bash
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
    --output-dir ./cgcnn_models
```

这将保存模型到: `./cgcnn_models/cgcnn_band_gap_best.pth`

## 步骤 2: 生成材料

使用训练好的 DNG 模型在 MP-20 测试集上生成材料:

### 方法 1: 使用提供的脚本 (推荐)

```bash
bash scripts/run_table4_generation.sh
```

### 方法 2: 手动运行

```bash
MODEL_PATH="/ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun/DNG-mp20-text-lattice5-periodic-last-gpu5"

python scripts/evaluate.py \
    -m "${MODEL_PATH}" \
    --dataset mp_20 \
    --num_evals 1 \
    --test_dataset_path data_text/mp_20/test.csv \
    --label mp20_table4
```

这将生成: `${MODEL_PATH}/eval_diff_mp20_table4.pt`

## 步骤 3: 计算匹配统计

### 3.1 只评估结构属性 (不使用 CGCNN)

```bash
MODEL_PATH="/ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun/DNG-mp20-text-lattice5-periodic-last-gpu5"

python scripts/compute_text_prompt_match_with_cgcnn.py \
    "${MODEL_PATH}/eval_diff_mp20_table4.pt" \
    --symprec 0.1 \
    --output results_table4_mp20_structure_only.pt
```

输出示例:
```
======================================================================
Correctness of Generated Materials Matching Conditions
======================================================================
Global Feature            Matched    Total      Percentage
----------------------------------------------------------------------
Formula                   XXX        XXX        XX.XX%
Space Group               XXX        XXX        XX.XX%
Crystal System            XXX        XXX        XX.XX%
Formation Energy          0          0          N/A
Band Gap                  0          0          N/A
======================================================================
```

### 3.2 评估所有属性 (使用 CGCNN)

```bash
MODEL_PATH="/ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun/DNG-mp20-text-lattice5-periodic-last-gpu5"

python scripts/compute_text_prompt_match_with_cgcnn.py \
    "${MODEL_PATH}/eval_diff_mp20_table4.pt" \
    --fe-model-path ./cgcnn_models/cgcnn_formation_energy_per_atom_best.pth \
    --bg-model-path ./cgcnn_models/cgcnn_band_gap_best.pth \
    --symprec 0.1 \
    --output results_table4_mp20_full.pt
```

输出示例:
```
======================================================================
Correctness of Generated Materials Matching Conditions
======================================================================
Global Feature            Matched    Total      Percentage
----------------------------------------------------------------------
Formula                   XXX        XXX        70.54%
Space Group               XXX        XXX        67.88%
Crystal System            XXX        XXX        73.54%
Formation Energy          XXX        XXX        92.88%
Band Gap                  XXX        XXX        96.73%
======================================================================
```

## 评估指标说明

### 结构属性

- **Formula**: 生成结构的化学式是否与 ground truth 匹配
- **Space Group**: 生成结构的空间群编号是否与 ground truth 匹配
- **Crystal System**: 生成结构的晶系是否与 ground truth 匹配

### 物理性质 (使用 CGCNN 预测)

- **Formation Energy**: CGCNN 预测的形成能符号(正/负)是否与 ground truth 符号匹配
- **Band Gap**: CGCNN 预测的带隙类型(零/非零)是否与 ground truth 类型匹配

## 注意事项

1. **CGCNN 训练时间**: 根据数据集大小,每个模型可能需要几小时到一天的训练时间
2. **对称性精度**: `--symprec 0.1` 控制空间群分析的精度,可以根据需要调整
3. **GPU 要求**: CGCNN 训练和预测建议使用 GPU
4. **依赖安装**: 
   ```bash
   pip install torch torch-geometric pymatgen
   ```

## 与 TGDMat 论文的对比

TGDMat 论文 Table 4 的 MP-20 结果:
- Formula: 70.54%
- Space Group: 67.88%
- Crystal System: 73.54%
- Formation Energy: 92.88%
- Band Gap: 96.73%

你可以将你的结果与这些数字进行对比。

## 故障排除

### 问题 1: CGCNN 训练 OOM (内存不足)
解决方案: 减小 `--batch-size` 或 `--h-fea-len`

### 问题 2: 结构转换失败
解决方案: 检查生成的结构是否有效,可能需要调整 `--radius` 参数

### 问题 3: 空间群识别失败
解决方案: 调整 `--symprec` 参数,或检查生成结构的质量
