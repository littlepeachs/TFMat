# CrystalFlow 在线 MatSciBERT 文本控制增量改造方案

## 目标

在不破坏原始 CrystalFlow 训练与复现实验的前提下，新增一条可选的在线文本条件分支：

1. 训练时在线运行 MatSciBERT
2. 默认冻结 MatSciBERT 参数
3. 使用 pooled text embedding 作为图级条件
4. 通过新配置开启，原始配置与脚本保持可用

## 增量修改原则

1. 原始数据配置不变
2. 原始模型配置不变
3. 原始训练脚本不变
4. 新增逻辑默认关闭
5. 只有文本专用配置才会加载 tokenizer 和 MatSciBERT

## 第一阶段实现边界

本阶段只实现全局文本条件，不引入 cross-attention。

### 包含内容

1. 数据集可选读取文本列
2. 数据集可选在线 tokenizer
3. 新增冻结的 MatSciBERT 编码器模块
4. 在 Flow 中拼接数值条件和文本条件
5. 新增文本训练配置与脚本

### 暂不包含

1. token-level cross-attention
2. 联合微调 MatSciBERT
3. 文本采样脚本改造
4. 文本条件评估脚本

## 文件级修改

### 兼容扩展

1. diffcsp/common/data_utils.py
   - 预处理时可选保留一个原始文本列

2. diffcsp/pl_data/dataset.py
   - 新增可选文本字段配置
   - 仅在文本配置启用时做 tokenizer
   - 输出 text_input_ids 和 text_attention_mask

3. diffcsp/pl_modules/flow.py
   - 新增可选 text_encoder 分支
   - 在 guidance 条件下拼接数值条件和文本条件
   - 默认未配置 text_encoder 时行为不变

4. setup.py
   - 增加文本功能的可选依赖 extra

### 新增文件

1. diffcsp/pl_modules/text_encoder.py
   - 在线 MatSciBERT 编码器
   - 默认冻结参数
   - 支持 cls 或 mean pooling

2. conf/data/mp_20_text.yaml
   - 文本版 MP-20 数据配置
   - 指向 data_text/mp_20
   - 使用独立缓存文件，避免污染原缓存

3. conf/model/flow_polar_text.yaml
   - 文本版 Flow 配置
   - 启用 guide_threshold
   - 配置 text_encoder

4. train_flow_text.sh
   - 文本控制训练脚本

## 数据流

1. data_text 中读取 cif 与 text 列
2. 预处理阶段缓存晶体图与原始文本
3. Dataset 在 __getitem__ 中对文本做 tokenizer
4. Batch 携带 text_input_ids 和 text_attention_mask
5. Flow 调用 text_encoder 得到 text_cemb
6. 若同时存在数值条件，则与数值 cemb 直接拼接
7. decoder 使用现有 cemb 注入路径控制生成

## 兼容性保证

1. 原始 train_flow.sh 无需改动
2. 原始 model=flow_polar 无需改动
3. 原始 data=mp_20 无需改动
4. 未配置 text_column 时 dataset 不会加载 tokenizer
5. 未配置 text_encoder 时 model 不会加载 transformers
6. guide_threshold 未启用时，不会构造任何条件 embedding

## 后续第二阶段

若第一阶段验证文本有效，再进入第二阶段：

1. 在 cspnet 中加入 token-level cross-attention
2. 同时保留 pooled global text embedding
3. 将 cross-attention 做成配置开关，继续保持默认关闭

## 对照实验建议

1. baseline: 原始 CrystalFlow
2. text-global: frozen MatSciBERT + pooled text embedding
3. shuffled-text: 打乱文本对照组

该分组可用于判断收益是否真正来自文本语义，而非额外噪声或模型容量。