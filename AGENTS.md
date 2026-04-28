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
