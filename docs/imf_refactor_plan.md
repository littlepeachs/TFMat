# CrystalFlow iMF 正统改造方案

> 目标：按 Improved Mean Flows (iMF) 的论文与官方实现思路，将 CrystalFlow 的一步/少步晶体生成改造成真正的平均流框架，而不是仅做双时间条件的近似替换。

## 一、当前实现的主要问题

### 1. 现有 `flow.py` 不是 Mean Flow
- `diffcsp/pl_modules/flow.py` 学的是瞬时速度场。
- 其采样依赖 Euler 多步积分。
- 直接把它改成一步采样，本质是拿大步长外推局部速度，理论上就不适合 one-step。

### 2. 现有 `flow_imf.py` 不是论文里的真正 iMF
当前 `diffcsp/pl_modules/flow_imf.py` 的问题：
- 没有真正使用 `torch.func.jvp`。
- 没有显式 `u-head` / `v-head` 双头网络。
- 没有 auxiliary `v-loss`。
- `data_proportion`、`norm_p`、`norm_eps` 只定义未完整生效。
- 缺少 `fm_mask`（即 `r=t` 的 flow-matching 子样本）。
- 训练时几乎看不到 `(r=0, t=1)` 的一步端点样本。
- 训练配置未对齐 CrystalFlow baseline。

### 3. 现有 one-step 修复只是数值稳定补丁
- 当前 one-step 已通过切换到 `v(z,t,t)` 分支避免爆炸。
- 这个修改是工程补丁，不是论文中严格的 one-step mean-flow 公式。

---

## 二、实施目标（方案 2：最正统）

### 阶段 A：最小正确版 iMF
1. 新增双头 decoder：`CSPNetIMF`
2. 使用真正的 `torch.func.jvp`（必要时提供回退）
3. 训练时同时优化 `u-loss` 与 `v-loss`
4. 让 one-step / few-step 的采样接口与训练定义严格一致

### 阶段 B：补齐 iMF 稳定器
1. 引入 adaptive weighting
2. 实现 `sample_tr()`
3. 使用 `data_proportion` 构造 `fm_mask`
4. 加入 endpoint oversampling（显式采样 `(r=0, t=1)`）

### 阶段 C：配置对齐 baseline
1. 对齐 `lattice_polar_sigma`
2. 对齐 `cost_coord`
3. 对齐 `decoder.num_freqs/rec_emb/num_millers`
4. 对齐优化器与 scheduler

---

## 三、具体文件级修改方案

### 1. 新增双头 decoder
新增文件：
- `diffcsp/pl_modules/cspnet_imf.py`

设计原则：
- 保留 CrystalFlow 的图结构、晶格表示、边构造、对称性归纳偏置。
- 将网络分成：
  - shared trunk
  - `u` head
  - `v` head
- 输出：
  - `(u_l, u_f)`
  - `(v_l, v_f)`
- 若以后需要 type 生成，再扩展到 `(u_t, v_t)`。

建议结构：
- 前 `num_layers - head_num_layers` 层作为 shared trunk
- 后 `head_num_layers` 层复制两份，分别作为 `u` 与 `v` heads
- 每个 head 独立拥有：
  - 图层
  - lattice 输出层
  - coord 输出层
  - （可选）type 输出层

### 2. 重写 `flow_imf.py`
文件：
- `diffcsp/pl_modules/flow_imf.py`

训练逻辑改为：
1. 采样 `(t, r, fm_mask)`
2. 构造 `z_t`
3. 由双头 decoder 输出 `(u, v)`
4. 用 `torch.func.jvp` 对 `u_fn` 做 JVP
5. 构造
   \[
   V = u + (t-r) \cdot \mathrm{stopgrad}(du/dt)
   \]
6. 同时优化：
   - `loss_u = mse(V, target_v)`
   - `loss_v = mse(v, target_v)`
7. 对 lattice / coords 分别统计损失并加权

### 3. 引入真正的 `sample_tr()`
在 `flow_imf.py` 中新增：
- `logit_normal_dist()`
- `sample_tr()`

逻辑：
- 采样 `t`
- 采样 `r`
- 保证 `r <= t`
- 使用 `data_proportion` 将部分样本强制设为 `r=t`

### 4. 引入 adaptive weighting
在 `flow_imf.py` 中增加：
```python
adp_wt = (loss + norm_eps) ** norm_p
loss = loss / adp_wt.detach()
```
对 sample-wise 损失做加权。

### 5. 引入 endpoint oversampling
在 `flow_imf.py` 中增加超参：
- `endpoint_proportion`

逻辑：
- 训练时随机抽一部分样本设为：
  - `r = 0`
  - `t = 1`
- 让网络显式见到真正 one-step 端点。

### 6. 调整采样逻辑
目标：
- `N=1` 与 `N>1` 都与训练定义保持一致。
- one-step 不再依赖 heuristic 特判，而是依赖正确训练出来的 `u`。

### 7. 配置与脚本修改
涉及文件：
- `conf/model/flow_polar_imf.yaml`
- `train_imf.sh`

增加/对齐：
- `+model.lattice_polar_sigma=0.1`
- `model.cost_coord=10`
- `model.decoder.num_freqs=256`
- `model.decoder.rec_emb=sin`
- `model.decoder.num_millers=8`
- `+model.decoder.na_emb=0`
- `optim.optimizer.lr=1e-3`
- `optim.lr_scheduler.factor=0.6`

新增建议超参：
- `head_num_layers`
- `aux_v_loss_weight`
- `u_loss_weight`
- `endpoint_proportion`
- `use_true_jvp`
- `data_proportion`

---

## 四、建议实施顺序

### Step 1
新增 `CSPNetIMF` 双头 decoder，保证网络接口正确。

### Step 2
重写 `flow_imf.py`：
- 用双头输出
- 用 `torch.func.jvp`
- 加 `loss_v`

### Step 3
加入：
- `fm_mask`
- adaptive weighting
- endpoint oversampling

### Step 4
对齐训练配置并重新训练。

### Step 5
做 sanity check：
- `N=1` 的 lattice 长度/角度范围
- `N=4/8` 的稳定性
- `loss_u` / `loss_v` 是否同步下降

---

## 五、验收标准

### 训练期
- `loss_u_lattice`、`loss_u_coord` 持续下降
- `loss_v_lattice`、`loss_v_coord` 持续下降
- `loss_v` 不应明显高于 `loss_u`

### 采样期
- `N=1` 输出 lattice 不再出现几百 Å / 0° / 180° 崩坏
- `N=1` 的长度均值应与 `N=4/8` 在同一量级
- few-step 结果不应因重构而退化

### 对比实验
至少汇报：
- CrystalFlow baseline (`N=100`)
- iMF (`N=1`)
- iMF (`N=4`)
- iMF (`N=8`)

---

## 六、实现过程注意事项

1. `torch.func.jvp` 若遇到 forward-mode AD 不支持的算子，需要提供 fallback。
2. 不要破坏 CrystalFlow 原有 `flow.py` 的 baseline 逻辑。
3. 先实现 lattice + coords，不扩展 type，先保证 CSP 路径正确。
4. 清理调试断点 `pdb.set_trace()`，避免训练脚本中断。

---

## 七、最终目标

将当前“数值上能跑通但理论上不完整”的 `flow_imf.py`，改造成：
- 结构上接近官方 iMF
- 训练目标上严格对齐论文
- 能真实支撑 one-step / few-step 晶体生成实验
