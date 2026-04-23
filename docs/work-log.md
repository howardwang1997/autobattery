# Work Log

## Project: autobattery — PINN for Metal Battery Electrochemical Modeling

---

## TODO

### Priority: High

- [ ] **确认LMB实验数据格式并加载到 `data/raw/`**
  - 需要CSV格式，至少包含 time/voltage/current 列
  - 理想：多C-rate (0.1C, 0.5C, 1C, 2C) + 多温度 (25°C, 45°C) + 循环寿命数据
  - 使用 `src/data/loader.py` 的 `ExperimentalDataLoader` 加载
  - 在 notebook 中做探索性分析

- [ ] **生成LMB合成训练数据**
  - 运行 `python scripts/01_generate_synthetic.py --config configs/lmb.yaml`
  - 目标：10,000组参数扫描仿真
  - 参数范围：D_e, D_s, k0, k_sei, t_plus, sigma_e, R_sei_0
  - 注意：需要先确认PyBaMM的Li金属负极配置是否正确

- [ ] **训练正向PINN（快速P2D求解器）**
  - 运行 `python scripts/02_train_forward.py --config configs/lmb.yaml`
  - 目标：在参数空间上学习 params → V(t) 映射
  - 验证：与PyBaMM精确解对比，误差 < 1%
  - 可视化：保存电压对比图到 `outputs/figures/`

- [ ] **训练逆向PINN（参数辨识）**
  - 运行 `python scripts/03_train_inverse.py --config configs/lmb.yaml --data-dir data/raw`
  - 从实验循环数据中反推：D_e, D_s, k0, k_sei, t_plus, R_sei_0
  - Phase 1: Adam (5000 epochs) → Phase 2: L-BFGS (5000 epochs)
  - 验证：反推参数是否物理合理（与文献值对比）

- [ ] **验证与论文准备**
  - 运行 `python scripts/04_validate.py`
  - 交叉验证：80%训练，20%预测
  - 外推验证：用低C-rate训练，预测高C-rate
  - 生成论文所需的全部图表

### Priority: Medium

- [ ] **适配PyBaMM Li金属负极模型**
  - 当前用的是标准DFN + Chen2020参数（石墨负极），不是真正的Li金属负极
  - 需要确认PyBaMM是否原生支持Li metal anode，或需要自定义子模型
  - 关键差异：去掉负极粒子扩散，加入金属沉积/溶解动力学

- [ ] **扩展到NMB（Na金属电池）**
  - 基于LMB pipeline改编
  - Na金属负极参数化（Na沉积过电位、Na SEI成分不同）
  - 需要Na金属电池的实验数据

- [ ] **热耦合模型**
  - 如果实验数据中有显著温度变化，需要加入热PDE
  - PyBaMM支持热耦合DFN模型

- [ ] **SEI生长动力学深入建模**
  - 当前使用反应限制模型，可扩展为溶剂扩散限制模型
  - 从循环寿命数据中提取SEI演化规律

- [ ] **ML力场参数计算（可选）**
  - 用UMA/MACE计算Na+扩散系数作为PINN初始参数
  - 需要 `pip install fairchem-core mace-torch ase pymatgen`

### Priority: Low

- [ ] **超参数扫描**
  - 运行 `python scripts/05_sweep_params.py` 或使用 wandb sweep
  - 扫描 hidden_dim, num_layers, lambda_data, lambda_pde, learning_rate

- [ ] **DeepONet/算子学习升级**
  - 当前是标准PINN，可升级为Physics-Informed DeepONet
  - 学习完整的参数→解算子映射，而非单点预测

- [ ] **开源发布准备**
  - 代码清理、文档完善
  - 添加Colab notebook演示
  - 创建conda环境一键安装脚本

---

## Week 1 (2026-04-23): Project Setup & Architecture

### 2026-04-23

**Done:**
- [x] Research survey completed (see `RESEARCH_SURVEY.md`)
- [x] Na-ion digital twin technical research (see `NA_ION_DIGITAL_TWIN_RESEARCH.md`)
- [x] Decided on research direction: PINN electrochemical modeling for LMB (primary) + NMB (extension)
- [x] Project architecture designed and implemented
- [x] All core modules written:
  - `src/simulation/` — PyBaMM model, parameter management, solver, data generator
  - `src/data/` — Experimental/synthetic data loaders, preprocessor, PyTorch datasets
  - `src/pinn/` — MultiDomainPINN, InversePINN, PDE residuals, losses, forward/inverse trainers
  - `src/utils/` — Physics constants, visualization
  - `src/mlff/` — ML force field integration (optional)
- [x] Training scripts: 01-05
- [x] Unit tests: 15 tests, all passing
- [x] Config files: base.yaml, lmb.yaml, nmb.yaml
- [x] Architecture documentation written

**Environment Setup:**
- [x] conda environment `autobattery` created (Python 3.11)
- [x] Dependencies installed: torch 2.11, pybamm 26.3, numpy, scipy, pandas, matplotlib, pytest
- [x] Package installed in editable mode

**Baseline Verification:**
- [x] PyBaMM Li-ion DFN (Chen2020): works at 0.2C, 0.5C, 1.0C, 2.0C
- [x] PyBaMM Na-ion DFN (Chayambuka2022): works at 0.1C, 0.2C, 0.5C (fails at ≥1.0C due to interpolation bounds)
- [x] PyBaMM baseline plot saved to `outputs/pybamm_baseline.png`
- [x] All 15 unit tests passing

**Bug Fixes Applied:**
- Fixed missing `import numpy as np` in `network.py`
- Fixed `autograd.grad` calls to use `allow_unused=True` with None fallback in `pdes.py`
- Fixed `torch.exp()` requiring Tensor input in SEI growth model
- Fixed PyBaMM solver: temperature format, current function parameter, extrapolation tolerance

---

## Week 2: TBD

### Notes / Ideas
- PyBaMM Li-ion DFN with Chen2020参数目前模拟的是**石墨负极**，不是Li金属负极。需要调研PyBaMM是否原生支持Li metal anode，或者需要自定义子模型。这是后续实验的关键前提。
- Chayambuka2022 Na-ion参数集在高C-rate (≥1.0C) 时不稳定，是PyBaMM的已知限制。生成合成数据时需注意C-rate范围。
- 确认实验数据格式后，需要在 `src/data/loader.py` 中适配具体的列名映射。

---

## Week 3: TBD

---

## Experiment Log

### Experiment: PyBaMM Baseline Verification
- Date: 2026-04-23
- Config: N/A (direct PyBaMM calls)
- Goal: Verify PyBaMM DFN models work for both Li-ion and Na-ion
- Setup: CasADi solver, default parameters
- Results:
  - Li-ion (Chen2020): V=2.500-4.038V at 1.0C, 170 pts
  - Na-ion (Chayambuka2022): V=2.000-3.821V at 0.5C, 438 pts
  - LMB wrapper: V=2.500-4.036V at 1.0C, 196 pts
  - NMB wrapper: V=3.302-3.894V at 0.5C, 200 pts
- Observations:
  - Li-ion DFN works reliably across all C-rates
  - Na-ion DFN fails at ≥1.0C due to interpolation bounds (k_n, sigma_e)
  - Solver needs `max_step_decrease_count=5` for stability
  - Default current function is 1C CC, need to set via `params["Current function [A]"]`
- Next steps:
  - Generate parameter sweep data at stable C-rates
  - Adapt model for true metal anode configuration

---

### Experiment Template

```
## Experiment: [name]
- Date: YYYY-MM-DD
- Config: configs/xxx.yaml
- Data: data/xxx.npz
- Goal: [what we want to learn]
- Setup:
  - Model: [architecture details]
  - Training: [epochs, lr, etc.]
- Results:
  - [key metrics]
- Observations:
  - [what we learned]
- Next steps:
  - [what to try next]
```
