# Table 4 实验 - 快速开始

复现 TGDMat 论文 Table 4: 评估生成材料与文本提示条件的匹配度

## 🎯 目标

评估生成材料在以下属性上的匹配率:
- **Formula** (化学式)
- **Space Group** (空间群)
- **Crystal System** (晶系)
- **Formation Energy** (形成能符号: 正/负)
- **Band Gap** (带隙类型: 零/非零)

## 🚀 一键运行

```bash
bash scripts/run_table4_complete_pipeline.sh
```

这个脚本会自动完成:
1. ✅ 训练 CGCNN (formation energy)
2. ✅ 训练 CGCNN (band gap)
3. ✅ 生成材料 (使用 DNG 模型)
4. ✅ 计算匹配统计

## 📊 预期结果

TGDMat 论文 Table 4 的 MP-20 结果:

| Global Feature | Percentage |
|----------------|------------|
| Formula | 70.54% |
| Space Group | 67.88% |
| Crystal System | 73.54% |
| Formation Energy | 92.88% |
| Band Gap | 96.73% |

## 📁 输出文件

- `results_table4_mp20_structure_only.pt` - 仅结构属性
- `results_table4_mp20_full.pt` - 所有属性(含 CGCNN 预测)

## 📖 详细文档

查看 `docs/table4_experiment_guide.md` 了解:
- 分步执行说明
- 参数调整
- 故障排除

## ⚙️ 核心脚本

1. **train_cgcnn_for_properties.py** - 训练性质预测模型
2. **compute_text_prompt_match_with_cgcnn.py** - 计算匹配率
3. **run_table4_generation.sh** - 仅生成材料
4. **run_table4_complete_pipeline.sh** - 完整流程

## 💡 方法说明

由于文本描述不包含明确的性质要求,我们使用:

- **结构属性**: 直接从生成结构用 pymatgen 计算
- **物理性质**: 用 CGCNN 预测 → 比较符号/类型是否匹配

这样避免了昂贵的 DFT 计算!

## 🔧 依赖安装

```bash
pip install torch torch-geometric pymatgen
```

## 📝 模型路径

- DNG 模型: `/ssd/liwentao/GenAI/CrystalMF/CrystalFlow/hydra_jobs/singlerun/DNG-mp20-text-lattice5-periodic-last-gpu5`
- Checkpoint: `epoch=1834-step=194510.ckpt`
- 数据集: `data_text/mp_20/`
