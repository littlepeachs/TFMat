# Table 4 实验结果报告

## 实验设置

- **模型**: DNG-mp20-text-lattice5-periodic-last-gpu5 (epoch=1834-step=194510)
- **数据集**: MP-20 测试集
- **生成样本数**: 1000
- **使用文本嵌入**: 是 (test_text2_matscibert_mean.pt)
- **生成参数**:
  - ODE integration steps: 100
  - Anneal coords: True
  - Anneal slope: 5
  - Guide factor: 1.0 (自动)

## 实验结果

### 结构属性匹配率

| 属性 | 匹配数 | 总数 | 匹配率 | TGDMat 论文 (MP-20) |
|------|--------|------|--------|---------------------|
| **Formula** | 21 | 1000 | **2.10%** | 70.54% |
| **Space Group** | 23 | 1000 | **2.30%** | 67.88% |
| **Crystal System** | 0 | 1000 | **0.00%** | 73.54% |
| **Formation Energy** | - | - | 未计算 | 92.88% |
| **Band Gap** | - | - | 未计算 | 96.73% |

### 物理性质匹配率

- **Formation Energy**: 未计算 (需要 CGCNN 预测)
- **Band Gap**: 未计算 (需要 CGCNN 预测)

## 问题分析

### 1. 匹配率极低的原因

通过详细分析发现:

1. **生成的结构几乎都是空间群 1** (triclinic, 最低对称性)
   - 目标结构有各种高对称性空间群 (11, 44, 62, 72, 139, 164, 194, 221, 225等)
   - 这表明生成的结构缺乏对称性

2. **化学式完全不匹配**
   - 例如: 目标 `Rb2NaCrCl6` vs 生成 `Rb2InCl5`
   - 例如: 目标 `CsYbBr3` vs 生成 `Cs3NdBr5`
   - 元素组成和化学计量比都不对

3. **文本条件可能没有起作用**
   - 虽然使用了文本嵌入,但生成的结构与目标差异巨大
   - 可能的原因:
     - 模型训练时文本条件的权重不够
     - Guide factor 设置不当
     - 文本嵌入的质量问题

### 2. Crystal System 匹配率为 0 的原因

- 测试数据集 CSV 文件中**没有 `crystal_system` 字段**
- 只有 `spacegroup.number` 字段
- 评估脚本尝试从空字符串匹配,导致所有样本都不匹配

## CGCNN 训练结果

### Formation Energy 模型
- **验证集 MAE**: 0.0391 eV/atom
- **训练完成**: 100 epochs
- **最佳模型**: epoch 84

### Band Gap 模型
- **验证集 MAE**: 0.3747 eV
- **训练完成**: 100 epochs
- **最佳模型**: 未记录具体 epoch

**注意**: CGCNN 预测功能尚未完全实现,需要实现从 pymatgen Structure 到图数据的转换。

## 与 TGDMat 论文的差距

| 属性 | 我们的结果 | TGDMat (MP-20) | 差距 |
|------|-----------|----------------|------|
| Formula | 2.10% | 70.54% | -68.44% |
| Space Group | 2.30% | 67.88% | -65.58% |
| Crystal System | 0.00% | 73.54% | -73.54% |

**差距巨大**,表明当前模型的文本条件生成能力远不如 TGDMat。

## 可能的改进方向

### 1. 调整生成参数
- 尝试不同的 guide factor (0.5, 1.5, 2.0)
- 调整 anneal slope
- 增加 ODE integration steps

### 2. 检查模型训练
- 确认模型是否正确学习了文本条件
- 检查训练日志中的文本条件损失
- 可能需要重新训练模型,增加文本条件的权重

### 3. 完善 CGCNN 预测
- 实现完整的 Structure → Graph 转换
- 使用 CGCNN 预测 formation energy 和 band gap
- 计算符号/类型匹配率

### 4. 数据集问题
- 添加 crystal_system 字段到测试数据
- 或者从 spacegroup number 推导 crystal system

## 文件输出

- **生成的材料**: `hydra_jobs/singlerun/DNG-mp20-text-lattice5-periodic-last-gpu5/eval_gen_mp20_table4_1000_with_text.pt`
- **匹配结果**: `results_table4_mp20_1000samples_with_text.pt`
- **CGCNN 模型**:
  - Formation energy: `cgcnn_models/cgcnn_formation_energy_per_atom_best.pth`
  - Band gap: `cgcnn_models/cgcnn_band_gap_best.pth`

## 结论

当前实验表明,使用的 DNG 模型在文本条件生成方面的性能远低于 TGDMat 论文报告的结果。主要问题是:

1. 生成的结构缺乏对称性 (几乎都是空间群 1)
2. 化学式不匹配 (元素组成和比例都不对)
3. 文本条件似乎没有有效指导生成过程

需要进一步调查模型训练过程和生成参数,或者考虑使用不同的 checkpoint。
