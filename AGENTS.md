# AGENTS.md — Project Memory

## 项目约束（每次执行必须遵守）

### 1. 数据安全
- **大量历史实验数据尚未接入**，数据链路打通之前不能依赖实验数据
- **处理实验数据时，必须调用 HKRI Agent 的 minimax-m2.7 模型**（配置在 `~/.config/opencode/opencode.json` 的 HKRI provider 中），不得使用其他模型处理实验数据
- 仿真数据（PyBaMM 生成）无此限制

### 2. 研究定位
- 目标是**方法创新且 solid、结果好、有影响力**的工作
- 不追求凑数据发 paper，而是做出**开创性贡献**
- 当前可用资源：2×H20 GPU (root@10.239.68.24)、PyBaMM 仿真、3 个实验电池数据

### 3. 远程执行
- 计算任务在 `root@10.239.68.24` 上执行
- conda 环境：`source /root/miniconda3/bin/activate autobattery`
- 杀 GPU 占用进程：`ps aux | grep gpu_load | grep -v grep | awk '{print $2}' | xargs kill`
- PyTorch 安装方式：`pip install torch==2.5.1 -i https://mirrors.aliyun.com/pypi/simple/`（conda 版本有符号冲突）

### 4. 项目状态
- **Phase 0-3 已完成**：FNO surrogate (1772x, 8.4mV)、参数可辨识性分析、降解诊断
- **Foundation Model 已起步**：BatteryTransformer 8.3M params, 5 化学体系, best 3.6mV
- **当前分支**：`feature/foundation-model`
- **待做**：锂金属增强模型、更多物理（热/锂沉积）、scale up Foundation Model

### 5. 核心结论
- 电压签名有效秩 = 4/7，参数可辨识性是物理本质决定的
- V(t) 是降解分解最优空间（dQ/dV 更差）
- Group NNLS 是最佳分解方法（62% pass）
- Foundation Model 跨化学体系 zero-shot 可达 ~5mV

### 6. 已识别的学术短板（待改进）

#### P0 — 不做可能被拒

1. **实验数据量不足**
   - 主 pipeline 仅 3 个实验电池，Bayesian 论文声称 134 个但 raw data 未完全接入
   - **改进**：接入公开数据集做外部验证——Stanford/SLAC 124 cells (Severson 2019)、CALCE、Toyota TRI（上千个电池）
   - Foundation Model 的 zero-shot 泛化必须有实验验证，不能只做仿真到仿真
   - 注意：数据链路打通前不能依赖内部实验数据（见 §1 数据安全）

2. **缺少竞争 baseline 对比**
   - 论文 draft 几乎没有和现有方法的定量对比
   - **必须对比**：Dubarry ICA 分解（dQ/dV degradation mode analysis）、Severson/Attia early-cycle prediction、传统 PyBaMM 参数拟合、纯数据驱动方法（XGBoost/RF）
   - Foundation Model 方向需对比 DeepONet、Neural ODE 等 operator learning baseline

#### P1 — 影响理论深度和说服力

3. **理论证明不够严格**
   - rank ≤ 4 是构造性证明（逐个排除），缺少正式的数学框架
   - **改进**：用 Fisher Information Matrix 谱分析给出 Cramér-Rao 下界；做 profile likelihood analysis；给出紧致性分析（实验中实际 rank 是 3 还是 4）；推广到 DFN 模型的 rank 上界

4. **负面结果缺少信息论叙事**
   - multi-C-rate 不改善辨识、dQ/dV 更差、PDE residual 无效——目前只是陈述事实
   - **改进**：用数据处理不等式（DPI）证明 dQ/dV 变换不增加互信息；用信息论语言重述"V(t) 是最优空间"；将负面结果提升为核心理论贡献

5. **缺少不确定性量化 (UQ)**
   - FNO 代理模型只给点估计，无置信区间
   - **改进**：加 MC Dropout 或 Deep Ensemble；参数辨识结果必须带置信区间；这直接呼应 identifiability 理论（不可辨识参数自然有巨大 CI）

#### P2 — 提升 novelty 和影响力

6. **Foundation Model 创新性不足**
   - 8.3M params 标准 Transformer decoder + chemistry embedding，架构无创新
   - **改进**：引入物理归纳偏置（Butler-Volmer 参数化输出层）；等变性设计；scale 到 50M+ 并展示 scaling law；改名 "Cross-Chemistry Surrogate" 避免被质疑

7. **代码工程影响可重复性**
   - 79 个探索性脚本，缺乏统一入口
   - **改进**：提供一键复现 pipeline（`reproduce_all.py`）；清理脚本命名；提供预训练 checkpoint 下载；考虑 Colab demo

### 7. 新方向：三阶段 Pretrain-Finetune-LoRA 框架

#### 核心思路
仿真预训练 → 实验批量微调 → 在线 LoRA 逐电池适配，将现有的 identifiability 理论、FNO surrogate、Foundation Model 三个 track 统一为一个闭环故事。

#### 三阶段设计

| 阶段 | 数据 | 目标 | 理论支撑 |
|------|------|------|---------|
| Stage 1: 预训练 | 海量 PyBaMM 仿真（多化学体系） | 学习物理流形（V(t) 对参数的响应结构） | 跨化学体系泛化 |
| Stage 2: 实验微调 | 大量公开实验数据（Stanford/SLAC 124 cells, Toyota TRI, CALCE） | 桥接 sim-to-real gap（噪声、未建模物理） | 真实世界校准 |
| Stage 3: 在线 LoRA | 单个电池的实时 cycle 数据 | 适配个体差异（制造变异、特定降解路径） | 在 4 维可辨识子空间内适配 |

#### 核心创新点：LoRA rank = Identifiability rank

- **理论**：电压灵敏度 Jacobian 有效秩 = 4 → LoRA 的 rank 应设为 4
- 这不是经验调参，而是**物理理论决定的设计选择**
- 消融预期：r=4 最优，r>4 过拟合，r<4 信息不足
- 这是第一个用可辨识性理论 justify LoRA rank 的工作
- 每个 LoRA 层仅增加 256×4×2 = 2048 参数（dim=256 时），8 层 Transformer 总共 ~16K 参数在线更新，BMS 可部署

#### 与现有三个 track 的关系

| 现有工作 | 在新框架中的角色 |
|---------|----------------|
| 可辨识性分析（rank ≤ 4） | 指导 LoRA rank 选择 + 指导只微调可辨识参数 |
| FNO surrogate（1772x 加速） | 预训练阶段的模型架构候选 |
| Foundation Model（跨化学体系） | Stage 1 预训练模型 |

#### 目标期刊
- **Joule** / **Nature Machine Intelligence**（最匹配，理论+应用兼备）
- **Nature Energy**（如果实验规模够大：100+ 电池 × 多化学体系）
- **ICLR/NeurIPS**（如果强调 LoRA rank = identifiability rank 的方法论贡献）

#### 待解决风险
1. **实验数据依赖**：Stage 2 效果取决于公开数据集的量/质量，至少需要 Stanford 124 + CALCE + TRI
2. **在线 LoRA 稳定性**：单 cycle 噪声可能导致适配偏移，需要 trusted region 或正则化
3. **消融实验量大**：三阶段 × 多化学体系 × 多 rank，需仔细设计
4. **数据安全约束**：Stage 2 的实验数据处理必须用 HKRI Agent 的 minimax-m2.7 模型（见 §1）

#### 与竞品的差异化

| 方法 | 实验数据需求 | 物理一致性 | 实时性 | 跨体系 | 理论指导 |
|------|------------|-----------|--------|--------|---------|
| Severson/Attia | 100+ cycles | 无 | 快 | 差 | 无 |
| PyBaMM 拟合 | 10+ curves | 强 | 慢 | 每种重来 | 无 |
| PINN from scratch | 10+ curves | 强 | 慢 | 每种重训 | 无 |
| DeepONet/Neural ODE | 中等 | 中 | 快 | 差 | 无 |
| **本框架** | **1-5 cycles** | **强** | **快** | **好** | **rank 理论** |
