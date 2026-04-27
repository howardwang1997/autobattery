# Plan B: Neural Operator for Full PDE Field Prediction

## Hardware: 2× H20 (96GB each)

---

## 一句话总结

用 Fourier Neural Operator (FNO) 学习完整的 P2D PDE 解算子，实现从电化学参数到**全时空场**（浓度、电位、电流密度等）的瞬时预测，比 PyBaMM 快 100-1000 倍。

---

## 与当前项目的关系

```
当前 PINN:            参数 → 端电压 V(t)            [标量预测]
Neural Operator:      参数 → 全场 c_e(x,t), φ(x,t)... [场预测]
Foundation Model:     多体系参数 → V(t)               [跨体系预测]
```

- **代码复用率 ~70%**：simulation/, data/, configs/ 完全复用
- **新增模块**：`src/operator/` — FNO 架构、训练器、数据管线
- **独立发论文**，和 Foundation Model 不冲突
- **2卡完全够用**：FNO ~10M参数，40MB，单卡都绰绰有余

---

## 2卡分配策略

```
GPU 0: 训练主进程 (FNO forward + backward)
GPU 1: 备用 — 可用于同时跑验证/推理/数据预处理
       或跑第二组实验（超参搜索时并行）
```

训练时实际上单卡就能跑，第二卡用于：
- 同时跑验证集评估（不阻塞训练）
- 同时跑第二组超参配置
- Ensemble 训练（2个模型同时）

---

## 为什么 FNO 而不是 DeepONet？

| 特性 | FNO | DeepONet |
|------|-----|----------|
| 全局感受野 | 天然（Fourier层） | 需要足够深的网络 |
| 分辨率无关 | 是（spectral） | 否 |
| 训练速度 | 快 | 较慢 |
| 适合PDE | **非常适合** | 适合 |
| PyTorch生态 | neuralop 库成熟 | DeepXDE 支持 |

结论：选 **FNO**。用 `neuralop` 库（MIT开源，PyTorch原生）。

---

## 架构设计

### 输入输出

```
输入:
  - 参数向量 p: (7,) — D_e, D_s, k0, k_sei, t_plus, sigma_e, R_sei_0
  - 运行条件 c: (2,) — C-rate, temperature
  - 查询坐标网格 (x_i, t_j): (Nx, Nt)

输出（全场预测）:
  - c_e(x, t):      电解液浓度    (Nx, Nt)
  - phi_e(x, t):    电解液电位    (Nx, Nt)
  - phi_s(x, t):    固相电位      (Nx, Nt)
  - c_s(r, x, t):   正极粒子浓度  (Nr, Nx, Nt)
  - j(x, t):        局部反应速率  (Nx, Nt)
  - V(t):           端电压        (Nt,)
  - L_sei(t):       SEI厚度       (Nt,)
```

### 模型结构

```
Parameter Encoder (MLP):
  (p, c) → z_param  [参数 → 隐编码]

FNO Backbone:
  Input:  [z_param broadcast] + [c_e, phi_e, phi_s initial fields]
  Lift:   Conv 1x1 → hidden channels (64)
  FNO Block ×4:
    - SpectralConv2d (Fourier层，在 x-t 平面做卷积)
    - Skip connection + GeLU
    - 参数条件注入 (FiLM conditioning: gamma*z + beta)
  Project: Conv 1x1 → output channels (7 fields)
  Output: c_e, phi_e, phi_s, c_s, j, V, L_sei

Physics-Informed Loss:
  - 数据拟合 loss（和 PyBaMM 全场对比）
  - PDE residual loss（复用 MetalBatteryPDE）
  - BC/IC loss
```

### 模型参数量估算

```
FNO 4层, 64 channels, mode 16×16:
  每层: 2 × 64 × 64 × (16×16) ≈ 2M params
  总计: ~10M params → 40MB
  单卡显存占用: ~200MB (含激活值和梯度)
  → 2卡绰绰有余，batch size 可以开到 128+
```

---

## 数据管线

### 1. 扩展 PyBaMM 数据生成器（保存全场）

当前 `data_generator.py` 只保存 V(t)。需要扩展：

```python
solver.solve_full_field(params, c_rate, temp, Nx=50, Nr=20, Nt=200)
# 返回:
#   c_e:      (Nt, Nx)     电解液浓度空间分布
#   phi_e:    (Nt, Nx)     电解液电位
#   phi_s:    (Nt, Nx)     固相电位
#   c_s:      (Nt, Nr, Nx) 正极粒子浓度（仅正极区域）
#   j:        (Nt, Nx)     局部反应速率
#   V:        (Nt,)        端电压
#   L_sei:    (Nt,)        SEI厚度
```

**PyBaMM 全场提取代码（关键）：**
```python
solution = pybamm_solver.solve(params, c_rate, temp)
c_e = solution["Electrolyte concentration [mol.m-3]"]
c_s = solution["Positive particle concentration [mol.m-3]"]
phi_e = solution["Electrolyte potential [V]"]
phi_s = solution["Positive electrode potential [V]"]
j = solution["Positive electrode interfacial current density [A.m-2]"]
```

### 2. 数据量估算

```
单次仿真输出: ~300KB
10,000 样本:  ~3GB（压缩后 ~1GB），完全放内存
```

### 3. 生成时间

```
PyBaMM 单次全场仿真: ~2-5秒（CPU）
10,000 样本:
  - 1核: ~6-14小时
  - 8核并行: ~1-2小时
  - 32核并行: ~20-40分钟
```

---

## 训练计划

### Phase 1: 数据生成（Day 1-3）

```
目标: 生成 10,000 组全场仿真数据

参数扫描范围（和现有 configs/lmb.yaml 一致）:
  D_e:     [1e-11, 1e-9]    m²/s
  D_s:     [1e-14, 1e-11]   m²/s
  k0:      [1e-11, 1e-9]    m/(s·√(mol/m³))
  k_sei:   [1e-12, 1e-8]    m/s
  t_plus:  [0.2, 0.5]
  sigma_e: [0.1, 2.0]       S/m
  R_sei_0: [1e-3, 1e-1]     Ω·m²

运行条件:
  C-rates: [0.2, 0.5, 1.0, 2.0]
  Temperature: [25°C]
  
空间网格: Nx=50, Nr=20, Nt=200

脚本: scripts/11_generate_fullfield.py
存储: data/fullfield/fullfield_lmb.h5
```

### Phase 2: FNO 训练（Day 4-7）

```
目标: 训练 FNO，全场预测误差 < 2%

模型配置:
  - FNO 4层, 64 channels, modes (16, 16)
  - FiLM conditioning 注入参数
  - 参数编码器: 3层 MLP, dim=64

训练配置 (2卡):
  - 单卡训练 (GPU 0), GPU 1 做验证
  - Batch size: 64
  - Adam, lr=1e-3, cosine decay
  - 500 epochs
  - Loss: MSE(field) + 0.1 × PDE_residual
  - BF16 mixed precision

训练时间估算 (单卡 H20):
  - 每个epoch: ~60秒 (10K samples, batch=64, 156 steps)
  - 500 epochs: ~8小时
  - 含PDE loss: ~10-12小时
  - → 过夜跑一晚搞定

2卡并行策略:
  - 方案A: GPU 0 训练, GPU 1 同时跑超参搜索的第二组配置
  - 方案B: GPU 0/1 各跑一个模型 → Ensemble 2个模型同时训

脚本: scripts/12_train_fno.py
输出: outputs/checkpoints/fno_lmb_final.pt
```

### Phase 3: 验证与对比（Day 8-9）

```
验证内容:
  1. 全场精度: 7个场的相对L2误差
  2. 速度测试: FNO vs PyBaMM 单次推理时间
  3. 参数敏感性: 不同参数范围的泛化能力
  4. 外推测试: 训练集外的 C-rate/温度

对比基线:
  - PyBaMM DFN (ground truth)
  - 标准 PINN
  - FNO (this work)

预期结果:
  - 速度: FNO ~1ms vs PyBaMM ~3s → 3000x 加速
  - 精度: 全场误差 < 2%, 电压误差 < 0.5%

脚本: scripts/13_validate_fno.py
图表: outputs/figures/fno_*.png
```

### Phase 4: 应用演示（Day 10-12）

```
1. 实时数字孪生演示
   - 输入实测 V(t) → FNO 反推内部状态
   - 可视化: 动画展示 c_e(x,t), phi_e(x,t) 演化

2. 参数辨识
   - 用 FNO 作为正向模型，优化参数拟合实验数据
   - 对比 InversePINN 和 FNO-based inversion 的精度/速度

3. 内部状态可视化
   - 生成论文的"电池透视"图
   - 展示 Li 沉积分布、SEI 生长、电解液耗尽等

脚本: scripts/14_demo_fno.py
```

---

## 新增代码结构

```
src/operator/
├── __init__.py
├── fno.py              # FNO 模型定义 (SpectralConv, FNOBlock, FNO2d)
├── condition.py        # FiLM conditioning, 参数编码器
├── dataset.py          # FullFieldDataset (从 HDF5 加载)
├── trainer.py          # FNOTrainer (单卡/双卡训练)
├── physics_loss.py     # 场级别的 PDE residual loss
└── inference.py        # 快速推理接口

scripts/
├── 11_generate_fullfield.py   # 生成全场训练数据
├── 12_train_fno.py            # 训练 FNO (2卡)
├── 13_validate_fno.py         # 验证 + 速度 benchmark
└── 14_demo_fno.py             # 应用演示 + 可视化

configs/
└── fno.yaml            # FNO 模型和训练配置
```

---

## 时间线总览

```
Day 1-3:    数据管线 + 全场数据生成 (CPU并行)
Day 4-5:    实现 FNO 架构 (src/operator/)
Day 6-7:    训练 FNO (单卡 ~10小时, 过夜跑)
Day 8-9:    验证 + 速度 benchmark + 调参
Day 10-12:  应用演示 + 内部状态可视化
Day 13-14:  整理图表, paper draft

总计: ~2周 (14天)
```

---

## 论文定位

**标题候选:**
- "Fourier Neural Operator as a Fast Surrogate for Metal Battery P2D Electrochemical Modeling"
- "Real-Time Digital Twin of Lithium Metal Batteries via Physics-Informed Neural Operators"

**目标期刊:** Joule, Applied Energy, Energy & Environmental Science

**核心卖点:**
1. **全场预测** — 不仅是端电压，而是电池内部所有物理量的时空分布
2. **3000x 加速** — 比传统 FEM 求解器快 3 个数量级
3. **物理约束** — PDE residual loss 保证预测的物理一致性
4. **实用性** — 可用于实时数字孪生、BMS 算法、快速参数辨识

**主要对比实验:**
| 方法 | 预测内容 | 速度 | 精度(V) |
|------|---------|------|---------|
| PyBaMM DFN | 全场 | ~3s | Ground truth |
| 标准 PINN | V(t) | ~10ms | <1% |
| **FNO (this)** | **全场** | **~1ms** | **<2%** |

---

## 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| FNO 在参数空间外推差 | 中 | 增加参数扫描范围，加 PDE loss 约束 |
| PyBaMM 全场提取接口不稳定 | 中 | 先在单个模型上验证提取代码 |
| H20 无 FP64 | 低 | FNO 用 FP16/BF16 即可，不影响精度 |
| 训练时间超出预期 | 低 | 减少到 200 epochs 先看趋势，再决定是否继续 |

---

## 依赖

```bash
pip install neuralop h5py wandb  # neuralop: MIT FNO库
# 其余依赖已在环境中
```
