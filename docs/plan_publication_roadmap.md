# autobattery 项目评审与 NE/Joule 发表路线（LMB 1000–5000 循环）

## Context

你目前的项目（autobattery）想用 PINN + FNO 给 Li 金属电池（LMB）做电化学建模与退化诊断。已完成 Phase 0：12k 条 PyBaMM 合成 V(t)、训了一个 200K MLP（不是真 PINN），per-sim 归一化下 33 mV RMSE；逆向辨识在 NEWAREA 上 31.6 mV——但你自己 work-log 已承认"参数无物理意义"。FNO 与退化诊断（scripts 11–28）多为脚手架，未跑通。

你新提的资产是关键：**1000–5000 循环的 LMB 实验数据 + cryo-EM/XPS/SEM 合作组 + 4×H20**，目标 Nature Energy / Joule。这条信息决定了一切：现有代码方向（Severson 风格快速 surrogate）发不到 NE/Joule，必须把项目重定位为 **"长循环 LMB 退化的机理可解释 AI 诊断 + 多模态后验证"**。本计划是这个重定位的科研路线 + 第一阶段代码改造清单。

---

## 1. 现有工作的价值、问题、弱点

### 价值（保留并放大的部分）
- **多域 PINN 架构**（`src/pinn/network.py:49-214`）：domain embedding + 三个 head，处理 neg/sep/pos 分区思路是对的，可移植到 LMB
- **差分电压签名法**（`scripts/28_degradation_modes.py:53-94`）：用 ΔV(t) = V_cycle(t) − V_ref(t) 与参数线性回归构造签名库，再用 NNLS 拟合实验，回避 sim-to-real gap，把 RMSE 从直接拟合的 178 mV 降到 12.7 mV——**这是整个仓库最有论文潜力的 idea**
- **两阶段 Adam→L-BFGS 反演**（`src/pinn/inverse.py:107-171`）+ log-space 参数变换：标准但扎实
- **PyBaMM + 多进程合成数据管线**（`scripts/01_generate_synthetic.py`）：12k 条曲线已有，可直接复用做 LMB-DFN
- 单元测试 15 个全过 → 工程基础在线

### 问题（必须修，否则连 JPS 都过不了 reviewer）

**P1. "PINN" 名不副实**
- `02_train_forward.py` 训的是 VoltageMLP（`src/pinn/network.py` 末尾），**完全没有用 PDE 残差**，是个纯数据驱动 MLP（work-log:185-188 自己写明）
- MultiDomainPINN 试过但 RMSE 卡在 350 mV，被放弃——意味着 PDE loss 的训练动力学没调通（典型 PINN 失败模式：multi-scale loss、no NTK 加权、未做 hard BC）
- **若投 NE/Joule 还自称 PINN，会被立刻 desk reject**

**P2. per-sim 归一化是数据泄露**
- `v_norm = (V - V_mean_sim) / V_std_sim`（work-log:192-195 自己承认）：推断时需要目标曲线本身的 mean/std，是 cheating
- 33 mV RMSE 是高估的，实际泛化到新单体后会显著退化

**P3. 用 Chen2020 石墨参数冒充 Li 金属**
- work-log:47–48 明确 "当前用的是标准 DFN + Chen2020 参数（石墨负极），不是真正的 Li 金属负极"
- 没有 Li 金属特异物理：无枝晶、无 dead Li、无 plating CE、电解液还原项缺失
- PDE 里 `pdes.py` 虽有 plating Butler-Volmer 和 SEI，但 hardcoded E_a=5e4 J/mol、R_p=1μm、t⁺=0.38（`pdes.py:62,142,208`），既无引用也无敏感性

**P4. 1D 模型 + 平面假设漏掉 Li 金属最关键的退化机理**
- Li 金属的核心退化是非均匀沉积 → 枝晶 → dead Li → 多孔化 → 电解液耗尽，**全是 2D/3D 形貌驱动的**
- 1D P2D 在物理上无法表达这些 → 即使 PINN 训得完美，机理也是错的

**P5. 没有 baseline、没有 ablation、没有 UQ**
- 没有 PyBaMM 直接拟合做 ground truth（work-log:50 标 TODO）
- 没有"无 PDE loss 的 NN"对照
- 没有不确定度（Bayesian PINN / ensemble / conformal）→ 顶刊 reviewer 必问"如何知道你识别的 SEI 厚度是 5 nm 还是 50 nm？"

**P6. 实验数据未真正集成**
- `/root/data/raw/...` 是容器路径（`scripts/24_multi_cell_degradation.py:40-68`），本地不可用
- 未做交叉单体验证（A/B 训练，C 测试）
- 未做温度、化学体系、N/P 比的分层验证

### 弱点（结构性，决定上限）

**W1. 故事还停留在 surrogate / parameter ID（已成红海）**
- 顶刊门槛：**机理新颖性** 或 **决策闭环**，单纯快/准已经不够
- PINN/FNO 给电池做 surrogate 在 2023-2025 已有十余篇（综述见 §2），独立做这个最多发 JPS

**W2. 用合成数据训练 → 实验数据反演的 sim-to-real gap 没真正解决**
- 差分电压签名法回避了，但本质上仍依赖签名库的物理保真度，而签名库来自有问题的 P2D（见 P3）

**W3. 没有"形貌－电学"的桥梁**
- 顶刊时代 LMB 工作几乎都要求把电学测量与 cryo-EM/SEM 形貌定量挂钩——这正是你合作组能补的

---

## 2. 相关工作与新颖性评估（2022–2026）

| 方向 | 代表工作 | 你 vs 它 |
|------|---------|---------|
| PINN+电池退化 | Wang et al., *Nat. Commun.* 2024（387 cells, 0.87% MAPE） | 他们做 Li-ion SOH 黑箱+物理正则；你做 LMB 机理分解 → **不同战场** |
| Param-embedded FNO | PE-FNO arXiv 2024（<1.7 mV，<1% conc 误差） | 已经把 Li-ion P2D FNO 做透；你直接做 LMB-FNO 仅是"换数据集"，**新颖性不足** |
| LMB cycle life ML | *Adv. Sci.* 2024、*Batt. Supercaps* 2025、*Discov AI* 2025 | XGBoost/LSTM 黑箱预测，**全无机理**——这是你的真正空档 |
| PINO (PINN+FNO) | Caltech JMLR 2021 | 框架已成熟；你的工作必须靠 LMB 物理 + cryo-EM 验证去差异化 |
| 差分电池模型 | DiffLiB 2026 | 给 Li-ion DFN 做了可微反演；LMB + 差分仍空白 |
| ML 力场看枝晶 | *Nat. Commun.* 2025（cryo + EQ + MLFF） | 原子尺度；与你电芯尺度的电学诊断**互补**，可引用作机理依据 |
| Severson MIT | *Nat. Energy* 2019 | LFP，不是 LMB；早期预测范式但无机理 |

**新颖性诊断**：当前设计——"LMB PINN + FNO surrogate + 退化诊断"——核心组件**单独都已有人做过**。要发 NE/Joule，必须重新组织为"**只有 LMB、只有长循环、只有 cryo-EM 验证**"才能有的工作。

**实用性诊断**：当前路线的实用价值集中在"诊断电芯当前状态"——但从 V/I/CE 到 D_e/k_SEI 的反演，对工业 BMS 而言**信息量低于直接预测剩余寿命**。要补"指导设计/控制"的下游闭环。

---

## 3. 战略重定位：可投 Nature Energy / Joule 的故事

### 论文标题（草拟）
**"Mechanistic decomposition of lithium-metal battery capacity fade across thousands of cycles via cryo-EM-anchored neural operator"**

### 一段话 abstract（投稿语境）
> Lithium metal batteries (LMBs) lose capacity through a tangled mix of dead lithium, SEI growth, electrolyte depletion, plating-stripping inefficiency, and cathode degradation, but state-of-the-art ML models predict cycle life as a black box, leaving the underlying mechanism opaque. We present a neural-operator framework that, given only voltage-current cycling data, decomposes per-cycle capacity fade into five physically grounded modes, validated end-to-end against cryo-EM, XPS, and SEM at landmark cycles. Across N cells (1000–5000 cycles), our model attains <X% mode-attribution error against post-mortem ground truth and predicts 80%-of-life cycle from the first 100 cycles within Y%. Mechanistic outputs—dead-Li mass, SEI thickness, active-area loss—directly suggest electrolyte- and protocol-design changes, two of which we experimentally validate.

### 三个新颖性支柱（**全部缺一不可**——这是 NE/Joule 的门槛）

**Pillar A — 长循环 LMB 数据集本身**
- 1000–5000 循环 LMB 数据极为罕见（公开 LMB 数据普遍 < 500 循环）
- 即使方法学一般，数据集稀缺性本身就是论文价值（可发布为 dataset paper 或主图之一）
- **行动**：先把数据 metadata 摸清——化学体系、N/P 比、电解液、温度、循环条件、测试机型号

**Pillar B — 形貌锚定的机理分解**（最关键差异化）
- cryo-EM/XPS/SEM 合作组是杀手锏：在 100/500/1000/2000/N_eol 循环各拆 1–2 个电芯
- 模型输出（dead-Li 体积分数、SEI 厚度、孔隙率）直接 vs cryo-EM 测量值 → **量化的 mechanistic accuracy**
- 这个验证链条目前全文献无人做——*Nat. Commun.* 2025 的 cryo-EM ML 工作是原子尺度，没有电芯尺度回路

**Pillar C — 设计/控制闭环**
- 从机理诊断推断"如果加 X% additive 或者改充电协议，能延寿多少"
- 至少做 1–2 个新实验验证模型的设计建议（这是 Joule reviewer 必问的"so what"）
- 4×H20 + 几周可做：用诊断模型做贝叶斯优化，找最佳充电协议（Attia 2020 Nature 范式，但是 LMB）

### 为什么 V/I/CE 单独不足以发 NE/Joule，但加上 cryo-EM 就够
- 反问：V/I 反演 → 任何机理参数总有多解性（identifiability problem）
- cryo-EM 在 3–5 个时间点给出 absolute ground truth → 把多解约束成单解，**让"V → 机理"可信**
- 这是把"工程加速器"工作升级为"科学发现"工作的唯一路径

---

## 4. 分阶段路线（6–12 个月）

### Phase A — 基础修缮（4 周，必做）
**目标**：把 Phase 0 的硬伤补上，让物理与代码站得住脚

A1. 修复 PINN 实质性
- 重写 `src/pinn/forward.py` 真正加 PDE 残差到 loss；用 NTK 加权（Wang et al. 2022 SciML） 或 SoftAdapt
- 输出端 hard BC（让网络结构本身满足边界，而非 soft penalty）
- 目标：移除 per-sim 归一化后，新单体上 RMSE < 30 mV（带 PDE loss）
- 文件：`src/pinn/losses.py`、`src/pinn/network.py`、新增 `src/pinn/ntk_weighting.py`

A2. 建 PyBaMM 真 Li-metal anode 模型（做 baseline）
- PyBaMM 已有 `lithium_plating_option`（plating + stripping + SEI 共耦合，O'Kane 2022 模型）
- `src/simulation/model.py` 切到 lithium_plating="reversible"+SEI 项；参数从 Hu et al. 2024 LMB 文献找
- 这同时是 baseline（直接 PyBaMM 拟合）和 surrogate 的训练数据来源
- 文件：`src/simulation/model.py`、`src/simulation/parameters.py`、`configs/lmb.yaml`

A3. 把差分电压签名法系统化（你的最强 idea）
- 当前在 `scripts/28_degradation_modes.py` 是 prototype，搬到 `src/diagnosis/dv_signature.py` 模块化
- 加 NNLS → 改 ridge / elastic net 比较，加 bootstrap 置信区间
- 加 ablation：每去掉一个签名退化多少 → 证明每个机理项都不可省

A4. 建立交叉验证 protocol
- 至少 3 cells × 1000+ 循环，做 leave-one-cell-out
- 严格 train/val/test split 不能再泄露
- 文件：`src/data/cv_split.py`

### Phase B — 形貌锚定（6–10 周，论文核心）
**目标**：把 V→机理 的反演结果与 cryo-EM/XPS 实测对齐

B1. 选 landmark 循环 + 设计拆解实验（与合作组确定）
- 建议：循环 1（pristine）、100、500、1000、N_eol，每个点 2 个平行电芯
- 测量：cryo-EM（dead Li 与 SEI 形貌+厚度）、XPS（SEI 化学组分）、SEM（裂纹/孔隙率）

B2. 模型输出 ↔ 测量物理量的桥接
- 模型输出 dead-Li 体积分数 ↔ cryo-EM 量化（用图像分析得 μm³/cm²）
- 模型输出 SEI 厚度 ↔ XPS depth profile（带 Ar 刻蚀）
- 模型输出 LAM_pos ↔ SEM 二次颗粒裂纹率
- 这是论文 Fig. 3-4 的素材

B3. 把形貌测量喂回模型作为正则
- soft constraint：在已知 cryo-EM 测量的循环上，模型输出 dead-Li 必须吻合
- 用观测到的 SEI 化学组分修正 PDE 中 SEI 反应常数 k_SEI 的先验
- 这直接对应 Bayesian PINN 思路（Yang et al. 2021，*JCP*）

B4. 不确定度
- Deep ensemble（5–10 个 seed）或 MC dropout
- conformal prediction 给参数估计 calibrated 置信区间
- 文件：新增 `src/pinn/uq.py`

### Phase C — 闭环验证 + 论文（4–6 周）

C1. 早期预测 benchmark
- 输入前 100 循环 → 预测 80%-of-life 循环数；vs Severson features baseline、vs LSTM baseline
- 在 LMB 上做这个对比目前文献空白

C2. 设计建议生成与验证（Joule 必需）
- 用机理模型做 "what-if"：变 N/P 比、变电解液 LiFSI 浓度、变充电 CC/CCCV 协议
- 至少做 1–2 个新实验验证最具反差的预测
- 这是把论文从 npj 推到 Joule 的关键

C3. 论文图谱
- F1：故事图 + 数据集 overview（cells, cycles, chemistry, conditions）
- F2：方法学（PINN+FNO+签名法 + cryo-EM 锚定）
- F3：机理分解结果（5 模式随循环演化 + cryo-EM 验证散点图）
- F4：早期预测 vs baselines + UQ
- F5：设计建议闭环（实验验证）
- F6：泛化到不同化学体系/温度

### 可选 Phase D — Foundation Model（若 NE 投失败转 Nat. MI）
- 当前 work-log 提的 200M Battery Transformer + 70K 多化学合成数据
- **建议暂缓**：先把 Phase A-C 做完。foundation model 在没有形貌锚定时做不出顶刊，沦为"又一个 surrogate"

---

## 5. 关键文件改动清单

| 文件 | 改动 | 优先级 |
|------|------|--------|
| `src/pinn/losses.py` | 加 NTK / SoftAdapt 加权；修 BC/IC 实现 | P1 |
| `src/pinn/network.py` | 移除 per-sim norm；加 hard BC output transform | P1 |
| `src/pinn/forward.py` | 真正在 loss 里启用 PDE 残差项 | P1 |
| `src/simulation/model.py` | 切到 PyBaMM lithium_plating + SEI 完整模型 | P1 |
| `src/simulation/parameters.py` | LMB-specific 参数（plating CE、SEI Tafel、电解液耗尽） | P1 |
| `configs/lmb.yaml` | 新增 plating 选项；扩参数扫描 | P1 |
| `src/pinn/pdes.py` | 把 hardcoded E_a / R_p / t⁺ 改为可学习或 swept | P2 |
| 新增 `src/diagnosis/dv_signature.py` | 把 `scripts/28` 模块化 | P1 |
| 新增 `src/diagnosis/uq.py` | Deep ensemble + conformal | P2 |
| 新增 `src/diagnosis/morphology_constraint.py` | cryo-EM 测量作为 soft constraint | P1 |
| 新增 `src/data/landmark_cells.py` | 管理拆解电芯的 metadata（循环号、测量类型） | P1 |
| `src/data/cv_split.py` | leave-one-cell-out CV，严格 split | P1 |
| 删除 / 重写 `scripts/24_multi_cell_degradation.py` 里的 hardcoded 路径 | 移到 config | P2 |
| 新增 `scripts/30_baseline_pybamm_fit.py` | PyBaMM 直接拟合作 baseline | P1 |
| 新增 `scripts/31_baseline_severson_features.py` | Severson 特征 + RF/XGB 基线 | P1 |
| 新增 `scripts/40_design_optimization.py` | 用诊断模型做 BO 找最佳充电协议 | P2 |

---

## 6. 验证 / 成功标准

执行端怎么知道每一阶段成功：

**Phase A 完成判据**
- [ ] `pytest tests/` 全过；新增 PDE loss 的单元测试
- [ ] 在 hold-out cell 上无 per-sim norm，V RMSE < 30 mV
- [ ] PyBaMM lithium_plating 模型能复现合作组任一电芯的前 100 循环 capacity fade
- [ ] 差分签名法在 simulated→simulated 上 R² > 0.95，参数恢复误差 < 10%

**Phase B 完成判据**
- [ ] cryo-EM landmark 测量数据 ≥ 3 个循环点已录入 `data/landmark/`
- [ ] 模型预测的 dead-Li 体积分数 vs cryo-EM 测量：MAE < 20%（顶刊门槛）
- [ ] SEI 厚度预测 vs XPS：MAE < 30%
- [ ] UQ：90% 置信区间覆盖率 ≥ 85%

**Phase C 完成判据**
- [ ] 早期预测：first 100 cycles → 80%-of-life，RMSE < Severson baseline 的 70%
- [ ] 至少 1 个设计预测在新实验上验证（capacity retention 提升 ≥ 5%）

**端到端验证 commands**
```bash
conda run -n autobattery pytest tests/ -v
conda run -n autobattery python scripts/30_baseline_pybamm_fit.py --cell A
conda run -n autobattery python scripts/02_train_forward.py --config configs/lmb.yaml --no-per-sim-norm
conda run -n autobattery python scripts/22_degradation_diagnosis.py --cells A,B,C --cv leave-one-out
```

---

## 7. 必须先解决的开放问题（写论文前必须答）

这些用户已标记"待确认"，但是项目能不能立项的前置条件：

1. **数据 metadata**——化学体系（cathode、电解液、N/P 比、Li 厚度、电流密度、温度）
   - 不知道化学体系等于不知道写哪个故事，最优先确认
   - 建议：先与合作方/数据来源方开 1 次会，把每个 cell 的元数据填表
2. **数据规模**——具体 cell 数量、每 cell 多少循环、是否有重复条件
   - N≥5 cells × ≥1000 循环是发 NE/Joule 的最低数据规模门槛
3. **post-mortem 可行性确认**
   - 与 cryo-EM/XPS 合作组确认：能否拿到/还有没有 landmark 循环的电芯（已循环完毕的电芯还是要新做？）
   - 若已无完成循环的电芯，需要重启一批新实验（建议 ≥ 6 个 cell × 至少 3 个 landmark 循环）→ 这决定时间线是 6 月还是 12 月
4. **是否能补 EIS/dQ-dV**——现有数据机型若支持，建议每 50-100 循环加 EIS 与 1 次 GITT
   - 这极大增强反演 identifiability，可把 cryo-EM 验证标准从 MAE 30% 提到 < 15%

---

## 8. 一句话总结
**当前代码方向（surrogate + 黑箱反演）发不了 NE/Joule；但你的"长循环 LMB 数据 + cryo-EM 合作组"组合能发——只要把项目重定位为"形貌锚定的机理分解"，把现有的差分电压签名法做扎实，把 PINN 真的写成 PINN，并且至少做 1 个设计闭环验证。**
