# H20 实验执行 Runbook（Phase A）

> 配套：`docs/plan_publication_roadmap.md`、`scripts/h20/`
>
> 适用场景：4×H20（96 GB HBM3 each）、Linux、conda 已安装、ssh 可用。
> 你（数据/实验作者）在 H20 上 `git pull` 后照本文件执行即可。

---

## 0. 准备

### 0.1 同步代码

```bash
cd ~/code/autobattery        # 或者你 H20 上的克隆位置
git pull origin main
```

### 0.2 同步实验数据

把本地 LMB 长循环 xlsx/csv 同步到 H20：

```bash
# 本地（macOS/Linux）→ H20
rsync -avh --progress \
  /path/to/local/lmb_long_cycle/ \
  H20_HOST:~/code/autobattery/data/raw/lmb_long_cycle/
```

> 文件命名建议：`<cell_id>.xlsx` 或 `<cell_id>.csv`（cell_id 与 metadata 表 / landmark manifest 中的 cell_id 一致）。`data/` 目录已被 `.gitignore`。

如果你之后还要跑形貌锚定（Phase B），把 cryo-EM/XPS/SEM 的 manifest 与原始测量同步到 `data/landmark/`，schema 见 `src/data/landmark_cells.py`。

### 0.3 一次性环境装机

```bash
bash scripts/h20/00_setup_env.sh
```

执行内容：

1. 创建 conda 环境 `autobattery`（Python 3.11）；
2. 安装 PyTorch 2.5.1（与 H20 上 cuDNN 9.1 匹配，见 work-log:166–167）；
3. 安装 PyBaMM ≥ 24.5、numpy/scipy/pandas/sklearn/xgboost、openpyxl、h5py；
4. `pip install -e .` 把仓库装成可编辑包；
5. **冒烟测试**：跑一次 30 分钟的 LMB plating 仿真，验证 `OKane2022` 参数集 + lithium plating 模型能积分；
6. 打印 GPU 列表（应看到 4 块 H20）；
7. `pytest -q tests/` 跑全部单元测试。

预期输出：

```
smoke-test OK; voltage range 3.012 .. 4.183 V
torch 2.5.1
cuda available: True
device count: 4
  [0] NVIDIA H20  96 GB
  [1] NVIDIA H20  96 GB
  [2] NVIDIA H20  96 GB
  [3] NVIDIA H20  96 GB
... 18 passed (其中 5 个 PyBaMM 集成测试)
```

如果第 5 步冒烟测试 fail：
- 大概率是 PyBaMM 版本太旧，没有 `OKane2022`。`pip install -U pybamm`，再跑一次。
- 如果是 CasADi 求解器在 0.5C 跑不通：把 `c_rate=0.5` 改成 `0.2`，或者增加 `max_step_decrease_count=10`（看 `src/simulation/solver.py:42`）。

---

## 1. 生成 LMB 合成训练数据（Step 01）

```bash
NUM_WORKERS=32 bash scripts/h20/01_generate_synthetic_lmb.sh
```

| 参数 | 默认 | 含义 |
|------|------|------|
| `NUM_WORKERS` | 32 | PyBaMM 多进程 worker 数。H20 是 96 核以上，可以拉到 64 |
| `CONFIG` | `configs/lmb.yaml` | 改成 `configs/nmb.yaml` 跑钠金属 |
| `OUTPUT` | `data/synthetic` | |
| `SEED` | 42 | 复现性 |

预期：约 6–10 小时（CPU bound），生成 ~25k 条 V(t) 曲线（5000 sample × 5 C-rate），写到 `data/synthetic/synthetic_lmb.npz`。

中间会每 50 条保存一次（容错）。如果中途中断，重跑会从头开始——这是已知问题，后续可改增量。

健康监测：
- `tail -f logs/h20/01_*.log`
- 失败率 > 30% 通常说明 plating 参数扫描范围太极端，求解器收敛不了；编辑 `configs/lmb.yaml > simulation.parameter_sweep` 收紧上限再跑。

---

## 2. 构造差分电压签名库（Step 02）

```bash
BOOTSTRAP=200 bash scripts/h20/02_signature_library.sh
```

预期：< 5 分钟。产物：

```
outputs/diagnosis/signature_library_lmb.npz
```

字段（npz key）：`signatures`、`param_names`、`log_scale_mask`、`param_ref`、`param_scale`、`time_grid`、`bootstrap_signatures`、`meta`。

健康监测：日志最后一段会列出每个参数的 signature L2 范数。范数 < 1e-3 的参数说明在数据里没体现出灵敏度（要么扫描范围太窄、要么对 V(t) 的影响真的弱）。在 `configs/lmb.yaml` 里调 `parameter_sweep` 范围或者干脆把这个参数从签名库里去掉。

---

## 3. PyBaMM 直接拟合 baseline（Step 03）

为多 cell × 多 cycle 跑直接拟合 baseline。每个 cell × cycle 一个 scipy DE 优化，5–15 分钟。

```bash
DATA_DIR=data/raw/lmb_long_cycle CYCLES="10 100 500 1000" \
  bash scripts/h20/03_pybamm_baseline.sh
```

H20 上为了跑得快，可以手动开多个并行 shell 各处理一部分 cell（脚本本身是串行的，但你可以分批 dispatch）：

```bash
# Shell A
DATA_DIR=data/raw/lmb_long_cycle/batch_a bash scripts/h20/03_pybamm_baseline.sh
# Shell B
DATA_DIR=data/raw/lmb_long_cycle/batch_b bash scripts/h20/03_pybamm_baseline.sh
```

或者直接 wrap 成 GNU parallel：

```bash
ls data/raw/lmb_long_cycle/*.xlsx | parallel -j 8 \
  "python scripts/30_baseline_pybamm_fit.py \
     --data {} --cycle 100 --c-rate 1.0 \
     --output outputs/baselines/pybamm_fit/{/.}/"
```

产物：`outputs/baselines/pybamm_fit/<cell>/cycle_<N>/{fit_params.json, fit_summary.json, fit_curve.npz}`。

判通：
- `fit_summary.json.rmse_mV` 应该在 5–30 mV 之间。> 100 mV 说明物理模型与该 cell 偏差过大（很可能 N/P 比、电解液、温度与 OKane2022 默认差太多）。
- `fit_params.json` 里参数会被推到 bound 边界——边界本身就是嫌疑点，按需放宽。

---

## 4. Severson 早期预测 baseline（Step 04）

```bash
DATA_DIR=data/raw/lmb_long_cycle FEATURE_CYCLES="10 100" MODEL=rf \
  bash scripts/h20/04_severson_baseline.sh
```

跑 leave-one-cell-out 随机森林（默认）回归 EOL（80% 容量保持率对应的循环）。

产物：`outputs/baselines/severson/{features.csv, predictions.csv, cv_metrics.json}`。

判通：
- 至少 3 个 cell 能通过特征提取，否则脚本会直接报错。
- LOO RMSE < 100 cycle 对短链（500–1000）算合格；长链（5000）要求 RMSE/EOL < 15%。
- 切换 `MODEL=xgb` 或 `MODEL=ridge` 做 ablation。

---

## 5. 端到端机理诊断（Step 05）

```bash
LIBRARY=outputs/diagnosis/signature_library_lmb.npz \
  DATA_DIR=data/raw/lmb_long_cycle \
  BOOTSTRAP=100 ABLATION=1 \
  bash scripts/h20/05_diagnose_cells.sh
```

每个 cell：
- `outputs/diagnosis/cells/<cell>/trajectory.npz`：所有循环的 mode 系数轨迹；
- `outputs/diagnosis/cells/<cell>/diagnosis.json`：人读版结果；
- `outputs/diagnosis/cells/<cell>/ablation.json`：去掉每个 mode 后 RMSE 的增加量（用来给论文 § ablation）。

判通：
- ΔV 拟合 RMSE 应 < 30 mV（大多数 cycle）。> 80 mV 说明签名库覆盖不全 → 回到 Step 01-02 加更多扫描参数。
- Ablation 表里至少 3 个 mode 的 ΔRMSE > 5 mV，否则签名之间共线性太强、需要把扫描范围拉开。

---

## 6. Forward PINN 训练（Step 06，待 Phase A1 完成）

```bash
GPU=0 USE_PDE=0 EPOCHS=1000 \
  bash scripts/h20/06_train_forward_pinn.sh
```

⚠ 当前的 `src/pinn/forward.py` 仍是 VoltageMLP（数据驱动 MLP），并且用 per-sim 归一化（数据泄露）。**这一步只适合在 Phase A1 PINN 重写完成后再跑**。在那之前用 `MODEL=mlp USE_PDE=0` 跑只是为了 sanity-check 训练流程不崩。

Phase A1 需要在代码侧：
1. 在 `src/pinn/network.py` 增加 hard BC output transform（output = base + (network output) × mask）；
2. 在 `src/pinn/losses.py` 加 NTK 自适应权重；
3. 在 `src/pinn/forward.py` 移除 `_v_per_sim_mean / _v_per_sim_std`，改 global normalization（或者 hard BC 后干脆不做归一化）；
4. PDE collocation 点采样要按 c_rate 与参数协同采样（当前 collocation_params 全 0，物理上没意义）。

---

## 7. FNO surrogate 训练（Step 07，可选）

```bash
GPU=1 EPOCHS=200 BATCH=16 bash scripts/h20/07_train_fno.sh
```

只有当目标论文要 surrogate-speedup 卖点时才跑。Phase B 形貌锚定路线**不依赖** FNO。

---

## 8. 监控与调试

通用：
- 所有 launcher 写日志到 `logs/h20/<step>_<UTC>.log`，`tee -a` 同时打到屏幕。
- 训练用 `nvidia-smi -l 5` 看 GPU 利用率；要求 ≥ 80%。
- PyBaMM 求解失败的话先看 `logs/h20/01_*.log` 里 "Solver failed" 的频率，再去 `src/simulation/solver.py` 调求解器参数。

故障速查：

| 现象 | 可能原因 | 处理 |
|------|---------|------|
| `OKane2022 not in pybamm.parameter_sets` | PyBaMM < 24.5 | `pip install -U "pybamm>=24.5"` |
| `lithium plating option not recognised` | PyBaMM 用了旧 API | 同上升级 |
| 训练 loss = NaN | per-sim norm 数据泄露被去掉后 batch normalization 失效 | 改回 BatchNorm 或加 LayerNorm |
| 反演 RMSE > 100 mV | 签名库不覆盖该 cell 的退化模式 | 在 `configs/lmb.yaml` 加新参数到 `parameter_sweep`，重跑 Step 01-02 |
| Severson 脚本报 < 3 cells | 大部分 cell 没到 cycle 100 | 把 `FEATURE_CYCLES` 改成 `5 50` |
| `landmark` 测试在 PyBaMM 不可用时被 skip | 正常 | — |

---

## 9. 把结果同步回本地

H20 → 本地（用于做图、写论文）：

```bash
rsync -avh --progress \
  H20_HOST:~/code/autobattery/outputs/ \
  ./outputs/

# 仅同步诊断结果（避免 checkpoint 太大）：
rsync -avh --progress --include='*/' --include='*.json' --include='*.npz' --exclude='*' \
  H20_HOST:~/code/autobattery/outputs/diagnosis/ \
  ./outputs/diagnosis/
```

`outputs/` 同样在 `.gitignore` 里。把要进论文的 figures / metrics 单独 commit 到一个 `paper/` 子目录，或者放到 supplementary 仓库。

---

## 10. 当前阻塞 / 待办

跟 `docs/plan_publication_roadmap.md § 7` 一致，但执行视角：

- [ ] **数据 metadata 回填**：每个 cell 的化学体系/N:P/电解液/温度填到 `data/raw/lmb_long_cycle/metadata.csv`（schema 待定，建议字段：cell_id, cathode, electrolyte, NP_ratio, Li_thickness_um, current_density_mAcm2, temp_C, cycler_model, source）。**没有这个填表，论文写不出**。
- [ ] **Phase A1 PINN 重写**（见 § 6）；产物：可信的 forward PINN with PDE residual + global normalization。
- [ ] **landmark cell manifest**：与 cryo-EM 合作组确认能拿到/还需要再循环哪些 cell，按 `src/data/landmark_cells.py.LandmarkMeasurement` 填 `data/landmark/manifest.json`。
- [ ] **Phase B 桥接代码**：`src/diagnosis/morphology_constraint.py`（待 landmark 数据到位再写）。
- [ ] **Phase C 设计闭环**：`scripts/40_design_optimization.py`（贝叶斯优化充电协议），等 Phase A 数据稳定后再启动。
