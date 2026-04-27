# 改进计划：降解诊断方法提升

> 最后更新：2026-04-27
> 目标：将当前工作提升至 Joule / Nat. Energy 级别发表

---

## 一、当前状态

### 已完成（可靠）
- FNO 代理模型：1772× 加速，8.4mV (LMB) / 5.8mV (LIB)，c_e 0.87%
- 参数可辨识性三层结构（化学体系无关）
- Fisher 信息分析：c_e 信息量 >> V
- LFP 降解数据：1200 sims，7 参数，3 C-rates
- LFP VoltageFNO：28K params，27.8mV
- 差分电压签名分解：NEWAREA 12.7mV 拟合
- 多电池对比：NEWAREA / NEWAREB / TVC
- 早期寿命预测：0.88% RMSE

### 已完成（发现问题）
- 合成真实验证：**单模式识别率极低（0-32% top-match）**
- SVD 分析：signature 有效秩 = 4（7 个模式中只有 4 个可独立辨识）
- 6 种分解方法对比：**Group NNLS 最好（62% pass），其余全部 ≤38%**
- R_mult ↔ D_n 相关性 r=0.86，**无法从电压曲线分离**
- 多 C-rate 堆叠、Rate-feature 分解、ElasticNet：全部 0% pass
- 时序约束分解（script 34）：初步结果比 baseline 更差（1/8 vs 3/8）

### 根本性发现
1. 电压曲线的有效秩为 4，任何分解方法最多只能恢复 4 个独立方向
2. R_mult、D_n、t+ 的电压签名高度相似（都是过电位效应），无法唯一区分
3. SEI、LAM_pos、LAM_neg 可以被可靠检测（bootstrap 100%）
4. 方法在**多模式同时活跃**的场景下表现好（80-100%），单模式场景差

---

## 二、关键差距（距顶刊）

| 差距 | 影响 | 难度 |
|------|------|------|
| 分解方法精度不够 | 核心贡献的可信度受质疑 | **高** |
| 只有 1 个电池验证 | 统计显著性不足 | 中 |
| 无与已有方法对比 | 创新性论证不足 | 低 |
| 无 ground truth 验证 | 不知道分解结果对不对 | **高** |
| 无不确定性量化 | 结果缺乏置信度 | 低 |
| 时序约束分解尚未成功 | 理论上应该能提升 | 中 |

---

## 三、改进方案

### 方案 A：PyBaMM 多周期降解仿真（Ground Truth）

**核心思路**：用 PyBaMM 运行真正的多周期降解实验（非单次放电参数化），产生已知降解轨迹的电压数据。

**方法**：
1. 运行 PyBaMM SPM + SEI growth model，500+ cycles
2. 记录每 cycle 的电压曲线 + 已知的 SEI 厚度、LLI、LAM
3. 对这些合成数据运行分解方法
4. 将分解结果与已知 ground truth 对比

**优点**：
- 真正的 ground truth（知道 SEI 确实增长了多少）
- 可以测试不同降解机制主导的场景
- 论文中最强的验证

**风险**：
- PyBaMM 多周期仿真很慢（可能需要几小时）
- 需要确保模型包含所有降解机制

**预计时间**：1-2 天

---

### 方案 B：ICA (dQ/dV) 特征分解

**核心思路**：在 dQ/dV（增量容量分析）空间而非原始电压空间做分解。

**方法**：
1. 将电压曲线转为 dQ/dV 曲线
2. 在 dQ/dV 空间计算 Ridge signatures
3. 检查 dQ/dV signatures 的相关性矩阵
4. 如果相关性降低 → 分解精度提升
5. 如果相关性不变 → 至少可以作为对比方法

**物理论证**：
- LAM 在 dQ/dV 中表现为**峰值位置移动**
- R_mult 在 dQ/dV 中表现为**峰值高度降低**（过电位不改变热力学）
- SEI 在 dQ/dV 中表现为**峰值面积减小**（活性锂损失）
- 三者的 dQ/dV 签名**物理上就应该不同**

**预计时间**：0.5-1 天

---

### 方案 C：时序约束分解（修正）

**核心思路**：当前时序约束实现有 bug（比 baseline 差），需要修正。

**问题诊断**：
- L-BFGS-B 2000 次迭代未收敛
- 单调性惩罚可能太强（λ_mono=1.0），将错误模式也强制单调增长
- 应该：先做 per-cycle NNLS 得到初始解，然后只对**检测到的活跃模式**施加单调约束

**修正方案**：
1. 阶段一：per-cycle NNLS（baseline）
2. 阶段二：识别哪些模式在某些周期中为正
3. 阶段三：仅对活跃模式施加单调 + 平滑约束
4. 阶段四：重新优化

**预计时间**：0.5 天

---

### 方案 D：多电池验证 + 方法对比

**核心思路**：对所有可用电池运行完整分析，并与 ICA 基线对比。

**方法**：
1. NEWAREA（434 cycles，重降解）→ 已有结果
2. NEWAREB（325 cycles，中等降解）→ 需要提取电压曲线
3. TVC（221 cycles，轻微降解，45°C）→ 需要提取电压曲线
4. 对每个电池运行：group decomposition + ICA baseline
5. 展示我们的方法提供了 ICA 无法提供的定量信息

**ICA baseline**：
- 计算 dQ/dV 曲线
- 追踪峰值高度、位置、面积随 cycle 变化
- 与分解方法的结果定性对比

**预计时间**：1 天

---

### 方案 E：重新定位论文

**核心思路**：不强求"精确分解 7 种模式"，而是诚实地报告：

1. **可辨识性分析本身就是贡献**：
   - SVD 有效秩分析 → 证明电压曲线最多辨识 4 个方向
   - 签名相关性矩阵 → 定量给出哪些模式可分离
   - 这对电池诊断领域有指导意义

2. **Group decomposition 是合理的框架**：
   - 4 个物理意义明确的组（Resistance / LAM / SEI / Diffusion）
   - 62% pass，100% 检测 SEI 和 LAM
   - 比 ICA 更定量，比等效电路更可解释

3. **FNO + 差分签名方法是方法论贡献**：
   - 差分匹配避免 sim-to-real gap
   - FNO 提供 1772× 加速的快速签名计算

**目标期刊调整**：
- 如果分解精度能提升到 75%+：Joule
- 如果保持现状但验证充分：Energy & Environmental Science / Adv. Energy Materials
- 如果只做可辨识性分析：Electrochimica Acta / J. Power Sources

---

## 四、执行优先级

```
优先级 1（核心突破）：
  [A] PyBaMM 多周期 ground truth 仿真
  [B] ICA dQ/dV 特征分解

优先级 2（补充验证）：
  [C] 时序约束分解修正
  [D] 多电池验证 + ICA 对比

优先级 3（论文组织）：
  [E] 重新定位 + 论文写作
```

---

## 五、风险与应急

| 风险 | 概率 | 应急方案 |
|------|------|----------|
| dQ/dV signatures 仍然高度相关 | 30% | 只作为对比方法，不作为核心改进 |
| PyBaMM 多周期仿真太慢 | 20% | 用 SPM（非 DFN）加速，减少 cycle 数 |
| 所有改进都无法突破 75% | 40% | 转方案 E，将可辨识性分析作为主要贡献 |
| 时序约束仍然更差 | 30% | 放弃该方法，只报告 NNLS + Group 结果 |

---

## 六、文件结构

```
scripts/
  25_gen_lfp_degradation.py      # LFP 降解数据生成
  26_train_lfp_fno.py             # VoltageFNO 训练
  28_degradation_modes.py         # 差分签名分解（原始方法）
  29_synthetic_validation.py      # 合成验证
  30_methods_comparison.py        # 方法对比
  31_rate_decomposition.py        # 倍率分解（失败）
  32_hierarchical_validation.py   # 分层分解 + bootstrap
  33_neware_group_decomposition.py # NEWAREA group 分解
  34_temporal_decomposition.py    # 时序约束分解（待修正）
  (待添加)
  35_ground_truth_pysims.py       # PyBaMM 多周期 ground truth
  36_ica_decomposition.py         # ICA dQ/dV 分解
  37_multi_cell_analysis.py       # 多电池综合分析
  38_paper_figures.py             # 论文级图表

outputs/
  degradation/
    validation/                    # 合成验证结果
    methods_comparison/            # 方法对比
    hierarchical/                  # 分层分解
    neware_group/                  # NEWAREA group 结果
    temporal/                      # 时序约束结果
```
