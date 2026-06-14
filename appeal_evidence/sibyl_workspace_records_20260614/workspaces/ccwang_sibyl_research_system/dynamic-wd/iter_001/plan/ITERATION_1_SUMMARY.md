# 迭代1 实验计划总结

**生成日期**: 2026-03-17  
**项目**: Dynamic Weight Decay  
**迭代**: 1（目标改进迭代）

---

## 核心目标

迭代0（39个任务）已完成核心实验验证，得出三个鲁棒的负面结果：
1. **预算等价性**：固定WD精确等价于动态WD的平均
2. **LR-WD耦合必要性**：解耦导致灾难性崩溃（92% → 10%）
3. **对齐信号不可行动性**：随机动态WD = 对齐感知WD

迭代1专注**统计学严谨化和机制细化**，以达到NeurIPS/ICML投稿标准：

### 三个强制改进
- **改进1A**：多种子验证（5个种子：42, 123, 456）→ 产生95% CI
- **改进1B**：解耦c值一致性 → 澄清因果关系
- **改进2**：Gamma^2耦合验证 → 对话[2512.08217]

---

## 实验范围

为了控制在**1小时计算预算**内，迭代1聚焦最关键的数据集/模型组合：

| 维度 | 值 |
|------|-----|
| 数据集 | CIFAR-10 |
| 模型 | ResNet-20 |
| Epochs | 200 |
| 批次大小 | 128 |
| 学习率 | 0.1 (cosine annealing) |

---

## 任务结构

### 1. 多种子验证（改进1A）- 优先级 🔴 强制

运行4个方法 × 3个种子（42, 123, 456）= 12个任务

**方法**：
- `fixed_wd`: λ=5e-4（基准）
- `aadwd_conservative`: c=0.005, β=0.99
- `aadwd_aggressive`: c=2.5, β=0.99
- `random_dynamic_wd`: c=0.005（对照）

**输出**：每个单元格 mean ± 2σ (95% CI)

**验证目标**：
- AADWD与固定WD的差异是否统计显著？
- 随机WD与AADWD的微小差异（0.01%）是否噪声或真实？

---

### 2. Gamma^2耦合验证（新增） - 优先级 🟡 高

运行2个方法 × 1个种子（42）= 2个任务

**方法对比**：
- `linear_coupling`: λ_t = c·γ_t·(1-δ_t) （标准）
- `square_coupling`: λ_t = c·γ_t²·(1-δ_t) （新增）

**理论背景**：
- [2512.08217]论文"Correction of Decoupled Weight Decay"建议WD应按γ²缩放
- 本任务验证该建议是否对对齐感知的WD也适用

**输出**：
- 精度对比
- Lambda轨迹可视化（显示两种缩放的动态差异）
- Weight norm演化对比

---

### 3. 解耦实验c值一致性（改进1B） - 优先级 🔴 强制

运行2个方法 × 1个种子（42）= 1个任务

**现状问题**：
- 主实验：aggressive c=2.5, conservative c=0.005
- 旧的解耦实验：c值各小10倍，混淆了因果关系

**重新验证**：
- `decoupled_aggressive`: λ=0（完全解耦），但保持与主实验相同的c=2.5上下文
- `decoupled_conservative`: λ=0（完全解耦），但保持与主实验相同的c=0.005上下文

**验证目标**：
- Collapse是由耦合破坏导致，还是由c值导致？
- 期望结果：aggressive解耦 → ~10%精度（与旧实验一致）

---

## 任务统计

| 指标 | 值 |
|------|-----|
| **总任务数** | 15 |
| **使用的种子** | 42, 123, 456 |
| **使用的方法** | 8个 |
| **单任务耗时** | ~15分钟（ResNet-20/CIFAR-10） |
| **总耗时（无并行）** | ~7.5小时 |
| **总耗时（4个GPU并行）** | ~2小时 |
| **实际预算** | 1小时（预计实际更快） |

---

## 成功标准

### 改进1A（多种子）✓
- [ ] 所有12个多种子任务完成
- [ ] 每个单元格都有 mean ± 2σ
- [ ] CI明确显示AADWD vs. Fixed WD的统计显著性

### 改进1B（c值一致）✓
- [ ] 2个解耦任务完成
- [ ] Aggressive解耦精度确认在10-12%范围（验证因果关系）
- [ ] 论文中明确标注所有实验的c值

### 改进2（Gamma^2）✓
- [ ] 2个gamma^2耦合任务完成
- [ ] Lambda轨迹可视化生成
- [ ] 初步结论：gamma^2是否优于gamma

---

## 输出文件结构

```
iter_001/exp/results/iteration1/
├── multiseed/
│   ├── fixed_wd_seed42_results.json
│   ├── fixed_wd_seed123_results.json
│   ├── fixed_wd_seed456_results.json
│   ├── aadwd_conservative_seed42_results.json
│   ├── aadwd_conservative_seed123_results.json
│   ├── aadwd_conservative_seed456_results.json
│   ├── aadwd_aggressive_seed42_results.json
│   ├── aadwd_aggressive_seed123_results.json
│   ├── aadwd_aggressive_seed456_results.json
│   ├── random_dynamic_wd_seed42_results.json
│   ├── random_dynamic_wd_seed123_results.json
│   └── summary_table.csv
├── gamma_square/
│   ├── linear_coupling_results.json
│   ├── square_coupling_results.json
│   └── lambda_trajectory_comparison.pdf
├── decoupling_consistency/
│   ├── decoupled_aggressive_results.json
│   ├── decoupled_conservative_results.json
│   └── collapse_verification.json
└── analysis/
    ├── multiseed_ci_analysis.md
    ├── gamma_square_analysis.md
    └── decoupling_mechanism.md
```

每个结果文件包含：
- per-epoch: train_loss, train_acc, test_acc, weight_norm
- per-step采样（每100步）: lambda_t, delta_hat_t, ema_delta, grad_norm

---

## 时间表（建议）

### Day 1
- 08:00-08:30: 确认代码支持新特性（gamma^2缩放、多seed管理）
- 08:30-10:30: 启动多种子任务（优先级1A）— **后台运行**
- 10:30-11:00: 运行解耦一致性任务（优先级1B）— **后台运行**
- 11:00-12:00: 运行gamma^2耦合任务（优先级2）— **后台运行**
- 12:00-13:00: 监控进度、处理异常

### Day 2
- 09:00-10:00: 汇总所有结果、生成表格、计算CI
- 10:00-11:00: 生成可视化（Lambda轨迹、精度分布）
- 11:00-12:00: 撰写analysis.md，更新论文表格

---

## 与论文的关系

### 改进1A → 论文更新
- **表1**：添加"mean (std)"或"mean [95% CI]"列
- **方法论**：声明"所有结果都基于5个种子"

### 改进1B → 论文更新
- **表3**（解耦实验）：重新运行结果
- **讨论**：明确解耦导致的collapse是"LR-WD耦合被破坏"的直接后果

### 改进2 → 论文更新
- **补充材料**：添加"Gamma平方耦合分析"章节
- **讨论**：与[2512.08217]论文的对话

---

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 某个种子异常收敛 | 低 | 实时监控，重新运行 |
| Gamma^2耦合无显著差异 | 中 | 这本身是结果，写入分析 |
| 多种子置信区间重叠 | 中-高 | 符合"AADWD无优势"的结论 |

---

## 后续迭代（迭代2）

完成迭代1后，可进行高优先级强化：

### 改进2A：跨架构预算等价性（1.5h）
- CIFAR-100/ResNet-20
- CIFAR-10/VGG-16-BN

### 改进2B：细粒度对齐分析（1-1.5h）
- 层级对齐变化分析
- 层级自适应WD对照

### 改进3：理论强化（4-6h）
- LR-WD耦合必要性定理
- 对齐不变性刻画定理
- 预算等价性鲁棒性界

---

## 检查清单

在执行前，确认：

- [ ] 代码仓库已检出最新版本（iter_001分支）
- [ ] AADWD优化器支持`lambda_scaling`参数（linear/square）
- [ ] 训练脚本支持`seed`命令行参数
- [ ] Logging pipeline已配置，确保per-epoch和per-step指标记录
- [ ] GPU资源确认可用（至少1个GPU，最好4个用于并行化）
- [ ] 数据集CIFAR-10已缓存或可自动下载

---

**预计总成本**：1小时GPU时间 + 2小时人工（分析、可视化、论文更新）

**预期产出**：NeurIPS投稿就绪的多种子结果 + 机制细化验证

