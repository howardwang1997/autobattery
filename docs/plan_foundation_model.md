# Plan A: Battery Electrochemistry Foundation Model

## Hardware: 8× H20 (768GB total, NVLink interconnect)

---

## 一句话总结

构建电池领域的"基础模型"——在海量多化学体系仿真数据上预训练一个大模型，使其能够零样本或少样本适应任意电池化学体系的参数辨识、性能预测和寿命估计。

---

## 为什么这是"基础模型"而不是"大PINN"？

```
传统 PINN:   针对单一化学体系，从零训练
基础模型:    预训练阶段见过所有化学体系，学到了"电池的通用规律"
             ↓ fine-tune
             只需少量实验数据即可适配新电池
```

类比：
- NLP: GPT 在海量文本上预训练 → few-shot 适配任何任务
- CV: SAM 在海量图像上预训练 → zero-shot 分割任何物体
- **电池**: BFOM 在海量仿真上预训练 → few-shot 适配任何电池体系

**学术界现状：** 截至2026年4月，电池领域**没有**电化学建模的基础模型。

---

## 8卡分配策略

```
日常训练:
  8卡 DDP 数据并行训练 200M 模型
  每卡 batch=64 → 全局 batch=512
  H20 的 96GB 显存利用率: ~10%（模型小，可以大幅增加 batch）

并行实验模式:
  分成 2×4卡 或 4×2卡 同时跑不同实验:
  - 4卡 × 2组: 跑 2 组超参配置
  - 2卡 × 4组: 跑 4 组 fine-tune 实验
  - 1卡 × 8组: 跑 8 组小规模消融实验

消融实验 (发顶刊必须):
  - 同时跑 8 组不同化学体系组合的 leave-one-out 实验
  - 同时跑 8 组不同模型规模的对比
  - 每组实验 ~2-4小时 → 一个下午出全部结果
```

---

## 架构设计

### 核心思路：电池响应 = 条件序列生成

```
电池的电压曲线 V(t) 可以看作一个"序列"，
由参数和运行条件"条件化"生成。

这和语言模型生成文本序列是同构的！
```

### 模型架构：Battery Transformer

```
Input Encoder:
  - 化学体系编码: chemistry_embed ∈ {Li-ion, Li-metal, Na-ion, solid-state, ...}
  - 参数向量 p: (N_params,) → MLP → z_param (dim=256)
  - 运行条件 c: (C-rate, T, protocol) → MLP → z_cond (dim=64)
  - [CLS] token + z_param + z_cond → prefix tokens

Transformer Decoder (Causal):
  - 输入: prefix tokens + 已生成的 V(t_i) tokens
  - 输出: V(t_{i+1}) 预测
  - 24层, 16 heads, dim=768, FFN=3072
  - ~200M 参数

Output:
  - V(t): 逐点自回归生成电压曲线
  - (扩展) 内部状态 tokens → 可选的场预测头

Physics Regularization (可选):
  - 在 transformer 输出上加 PDE residual loss
  - 用 FiLM 层注入参数到 transformer 中间层
```

### 模型规模（8卡选定方案）

```
方案: 200M 参数, 24层, dim=768, 16 heads

显存分析 (单卡):
  模型参数: ~800MB (BF16)
  优化器状态 (AdamW): ~1.6GB (FP32)
  梯度: ~800MB
  激活值 (batch=64, seq=200): ~2GB
  总计: ~5.2GB/卡
  
H20 每卡 96GB → 利用率 ~5%
  → 可以大幅增加 batch size 或序列长度
  → batch=512/卡 也没问题

8卡 DDP 优势:
  - 全局 batch = 8×64 = 512 → 训练更稳定
  - 或 全局 batch = 8×256 = 2048 → 大 batch 训练
  - 梱积梯度同步开销极小（模型只有200M，通信量 ~1.6GB/step）
```

---

## 数据管线

### 1. 多化学体系仿真数据

```
覆盖的化学体系:
  1. Li-ion (Chen2020):     石墨/NMC811, 主流锂电
  2. Li-ion (Marquis2019):  石墨/NCA, Tesla
  3. Li-metal (LMB):        金属Li/NMC, 下一代锂电
  4. Na-ion (Chayambuka2022): 硬碳/层状氧化物
  5. Na-metal (NMB):        金属Na/正极
  6. Li-ion (Ai2020):       石墨/LFP, 储能/电动车
  7. Solid-state (可选):    如果PyBaMM支持

每种化学体系的参数扫描:
  - 核心参数: 5-8个（D_e, D_s, k0, t_plus, sigma_e, ...）
  - 每个参数: 10个采样点 (Latin Hypercube)
  - 运行条件: 4 C-rates × 3 温度 = 12种
  - 每种化学体系: ~10K 组仿真

总计: 7 种化学体系 × 10K = 70,000 组仿真
```

### 2. 数据表示

```
每组仿真的数据格式:
{
  "chemistry_id": 2,                    # 化学体系编号
  "params": [D_e, D_s, k0, ...],        # 归一化参数
  "conditions": [C_rate, T],            # 运行条件
  "V(t)": [4.2, 4.15, 4.10, ..., 2.5], # 电压曲线 (200点)
  "capacity": 3.0,                       # 额定容量
  "energy": 11.0,                        # 能量
}

归一化:
  - 每种化学体系的参数 → 全局归一化到 [0, 1]
  - 电压 → 对应化学体系的 [V_min, V_max] 归一化
  - 时间 → [0, 1]
```

### 3. 生成时间估算

```
PyBaMM 单次仿真: ~2-5秒 (CPU)
70,000 组仿真:
  - 1核: ~40-100小时
  - 8核: ~5-12小时
  - 32核: ~2-4小时

存储:
  - 70K × 200点 × 8bytes ≈ 112MB
  - 加上参数和元数据: ~200MB
```

---

## 训练计划

### Phase 1: 数据生成（Day 1-5）

```
Day 1-2: 搭建多化学体系数据生成管线
  - 新建 src/foundation/data/multichem_generator.py
  - 支持自动检测 PyBaMM 内置参数集
  - 每种化学体系的可扫描参数列表（需手动定义）
  - 统一输出格式
  - 验证: 每种化学体系跑 10 个样本确认输出正确

Day 3-5: 运行数据生成
  - 7种化学体系 × 10K 样本 = 70K
  - 多核 CPU 并行 (和 GPU 训练不冲突)
  - 保存到 data/foundation/train.h5

脚本: scripts/21_generate_multichem.py
```

### Phase 2: 模型实现（Day 6-9）

```
Day 6-7: Battery Transformer 架构
  - src/foundation/model/
    ├── chemistry_encoder.py  # 化学体系 embedding
    ├── param_encoder.py      # 参数 → 隐编码
    ├── battery_transformer.py # Transformer decoder
    └── output_heads.py       # 电压/容量/场 预测头

Day 8: 训练管线 (DDP)
  - src/foundation/trainer.py
    - 8卡 DDP (torchrun --nproc_per_node=8)
    - BF16 mixed precision
    - Gradient accumulation (如果需要更大的等效batch)
    - Wandb logging
  - 测试: 用 1K 样本跑通 DDP，验证 loss 下降

Day 9: Fine-tune 管线
  - LoRA / Adapter 实现
  - Few-shot 数据加载器
  - 评估框架 (zero-shot, few-shot metrics)

脚本: scripts/22_pretrain.py, scripts/23_finetune.py
```

### Phase 3: 预训练（Day 10-12）

```
训练配置 (200M参数, 8卡 DDP):
  - torchrun --nproc_per_node=8
  - 每卡 batch=64 → 全局 batch=512
  - AdamW, lr=5e-4, warmup 2000 steps
  - Cosine decay to 1e-5
  - BF16 mixed precision
  - 200 epochs over 70K samples

训练时间估算 (8卡):
  - 每个epoch: 70K/512 ≈ 137 steps
  - 每step: ~0.3秒 (200M模型, 8卡DDP, BF16)
  - 每epoch: ~41秒
  - 200 epochs: ~2.3小时
  - 含验证、logging、checkpoint: ~3-4小时
  → 上午启动，下午出结果

8卡优势体现:
  - 8卡 DDP: ~3-4小时完成预训练
  - 2卡做同样的事需要 ~12-16小时
  - 节省的 10 小时 = 多跑 3 轮实验

脚本: scripts/22_pretrain.py
输出: outputs/checkpoints/bfom_pretrained.pt (~800MB)
```

### Phase 4: 评估与消融（Day 13-17）

```
Day 13-14: 核心评估实验

实验1: Few-shot 参数辨识 (LMB)
  - 给定 1/5/10/50 条实验电压曲线
  - Fine-tune 后预测该电池的参数
  - 对比: 从零训练 vs 预训练+fine-tune

实验2: Zero-shot 跨化学体系泛化
  - 预训练中留出一种化学体系
  - 测试 zero-shot 预测该体系的电压曲线
  - 展示模型学到了"通用电池规律"

实验3: 迁移学习
  - 在 Li-ion 上预训练 → 在 LMB 上 fine-tune

实验4: 不确定性估计
  - Monte Carlo Dropout / Ensemble (8卡同时训8个模型)

Day 15-17: 消融实验 (8卡并行优势)

实验5: 化学体系数量消融 (8组同时跑)
  - GPU 0-1: 用 2 种化学体系预训练
  - GPU 2-3: 用 3 种化学体系预训练
  - GPU 4-5: 用 5 种化学体系预训练
  - GPU 6-7: 用 7 种化学体系预训练 (full)
  → 一次出 4 个数据点，不用排队等

实验6: 模型规模消融 (4组同时跑)
  - GPU 0-1: 50M 模型
  - GPU 2-3: 100M 模型
  - GPU 4-5: 200M 模型
  - GPU 6-7: 400M 模型
  → 一次出 scaling curve

实验7: 数据量消融 (8组同时跑)
  - 每组用不同比例的训练数据 (1K, 5K, 10K, 20K, 30K, 50K, 70K, 100K)

脚本: scripts/24_evaluate.py, scripts/25_ablation.py
```

### Phase 5: 论文（Day 18-30）

```
Day 18-20: 整理全部实验结果
  - 统一评估指标
  - 生成所有图表

Day 21-30: 论文撰写

论文图表 (7图2表):
  - Fig 1: BFOM 架构图
  - Fig 2: 多化学体系训练数据分布 (7种化学体系的 V(t) 曲线)
  - Fig 3: 预训练 loss + 收敛曲线
  - Fig 4: Few-shot fine-tune 性能 (shot 数 vs 误差)
  - Fig 5: Zero-shot 泛化热图 (化学体系 × 指标)
  - Fig 6: 消融实验 (体系数/模型大小/数据量 scaling curves)
  - Fig 7: 参数辨识结果对比 (BFOM vs PINN vs 传统拟合)
  - Table 1: 方法总对比
  - Table 2: 跨化学体系 fine-tune 结果
```

---

## 新增代码结构

```
src/foundation/
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── chemistry_encoder.py    # 化学体系 embedding (7种)
│   ├── param_encoder.py        # 参数 → 隐编码
│   ├── battery_transformer.py  # Transformer decoder
│   ├── output_heads.py         # 电压/容量/场 预测头
│   └── bfom.py                 # BatteryFoundationModel (整合)
├── data/
│   ├── __init__.py
│   ├── multichem_generator.py  # 多化学体系数据生成
│   ├── chemistry_registry.py   # 化学体系参数注册表
│   └── dataset.py              # MultiChemDataset
├── trainer.py                  # DDP pre-trainer + fine-tuner
├── lora.py                     # LoRA fine-tune 实现
├── evaluator.py                # Few-shot / zero-shot 评估
└── configs.py                  # 配置管理

scripts/
├── 21_generate_multichem.py    # 生成多化学体系数据
├── 22_pretrain.py              # 预训练 (8-GPU DDP: torchrun --nproc_per_node=8)
├── 23_finetune.py              # Fine-tune on LMB (LoRA)
├── 24_evaluate.py              # 评估 (few-shot, zero-shot)
└── 25_ablation.py              # 消融实验 (8卡并行)

configs/
├── bfom_pretrain.yaml          # 预训练配置
└── bfom_finetune.yaml          # Fine-tune 配置
```

---

## 时间线总览

```
Week 1 (Day 1-5):     数据管线 + 多化学体系数据生成
Week 2 (Day 6-9):     模型实现 + DDP 训练管线
Week 3 (Day 10-12):   预训练 (~3-4小时/轮, 可多轮迭代)
Week 3 (Day 13-17):   评估 + 消融 (8卡并行, 一天出多组结果)
Week 4-5 (Day 18-30): 论文撰写

总计: 3周实验 + 2周论文 = 5周

关键路径:
  Day 10 是第一个里程碑 — 预训练完成，看 loss 是否收敛
  Day 15 是第二个里程碑 — 消融实验出齐，确认论文故事
  Day 20 是第三个里程碑 — 全部图表完成，开始写
```

---

## 8卡 vs 2卡的差别（Plan A 为什么需要8卡）

```
                          2卡              8卡
预训练 (200M):            ~12-16小时        ~3-4小时
超参搜索 (4组并行):       串行 ~4天         1天
消融实验 (8组并行):       串行 ~1-2周       1-2天
Fine-tune (4组同时):      串行 ~2天         半天
-----------------------------------------------
总实验周期:               ~8-10周           ~3-5周

核心优势不是"能不能做"，而是"迭代速度":
  8卡 → 每天都能看结果、调方向
  2卡 → 每 2-3 天才能看一轮结果
```

---

## 论文定位

**标题候选:**
- "BFOM: Battery Foundation Model for Universal Electrochemical Modeling"
- "A Pretrained Transformer for Few-Shot Battery Parameter Identification Across Chemistries"

**目标期刊:** Nature Machine Intelligence, Joule, Nature Energy

**核心卖点:**
1. **首个电池电化学基础模型** — 填补空白
2. **跨化学体系泛化** — 一次预训练，适配多种电池
3. **Few-shot 能力** — 少量实验数据即可 fine-tune
4. **物理可解释** — 参数有明确的物理意义

**Narrative:**
> 电池研发中，每种新化学体系都需要重新建模和标定参数。我们提出 BFOM，在海量多体系仿真数据上预训练，使模型掌握了电池电化学的通用规律。面对新电池体系，只需几条实验曲线即可完成参数辨识——将传统数周的工作缩短到几分钟。

**主要对比实验:**
| 方法 | 所需实验数据 | 参数辨识误差 | 跨体系泛化 |
|------|------------|------------|-----------|
| 传统拟合 (PyBaMM) | 50+ 条 | <1% | 每种重新来 |
| PINN (from scratch) | 10+ 条 | <2% | 每种重新训练 |
| 纯 ML (XGBoost) | 100+ 条 | <3% | 不能 |
| **BFOM (this)** | **1-5 条** | **<3%** | **Zero-shot 可用** |

---

## 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 不同化学体系参数维度不同 | 高 | 用 padding + mask 处理变长参数 |
| PyBaMM 某些化学体系不支持 | 中 | 优先保证5种主流体系 |
| Transformer 拟合平滑曲线"杀鸡用牛刀" | 中 | 先用 50M 小模型验证，再 scale up |
| 跨体系 zero-shot 效果差 | 中 | 预期结果，重点展示 few-shot |
| Fine-tune 过拟合 | 中 | LoRA + 小 lr + early stopping |
| DDP 多卡通信问题 | 低 | 先单卡跑通，再切 DDP |
| 论文 novelty 受质疑 | 低 | 强调"首个基础模型" + 消融实验 |

---

## 依赖

```bash
pip install wandb      # 实验追踪
pip install timm       # Transformer 组件 (可选)
# 不需要额外大依赖，纯 PyTorch DDP 即可
```

---

## 与 Plan B (Neural Operator) 的关系

```
Plan B (Neural Operator, 2卡):  侧重 "深度" — 全时空场预测, 先出成果
Plan A (Foundation Model, 8卡): 侧重 "广度" — 跨化学体系, 冲顶刊

时间安排:
  Plan B 先做 (2卡, 2周) → 产出 Paper 1
  Plan A 接着做 (8卡, 5周) → 产出 Paper 2

可以组合:
  Phase 1: BFOM 预训练获得通用参数编码
  Phase 2: FNO 在特定体系上做全场预测
  → Paper 3: "Foundation Model + Neural Operator"

发论文策略:
  - Paper 1: FNO for Metal Battery (Joule) — 2周出成果
  - Paper 2: BFOM Foundation Model (Nature MI) — 5周
  - Paper 3 (可选): BFOM + FNO 组合 (Nature Energy) — 额外 3周
```
