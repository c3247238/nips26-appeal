# Full 200-Epoch 实验设计建议

**基于 Pilot 阶段结果**
生成时间：2026-03-17
适用实验：ResNet20/CIFAR-10（主实验）+ VGG16-BN/CIFAR-100（跨架构验证）

---

## 1. 总体方针

Pilot 阶段结论：**GO — 进行 200 epoch full 实验**。

核心目标从 pilot 的"可行性验证"升级到：
1. **证明 AADWD-Aggressive 在 200 epoch 下相对 Fixed-WD 的最终精度优势**
2. **区分 alignment-aware 调度 vs 单纯动态调度的贡献**（通过 norm_matched_wd 对照）
3. **量化 alignment proxy 可靠性**（r 值在 200 epoch 下是否达到 >=0.85）

---

## 2. 推荐实验组（ResNet20 / CIFAR-10）

### 2.1 必须包含的方法（核心对比）

| 优先级 | 方法 | 配置 | 目的 |
|--------|------|------|------|
| P0 | **Fixed-WD (5e-4)** | wd=5e-4 | 主 baseline（pilot 最佳固定 WD） |
| P0 | **Fixed-WD (1e-3)** | wd=1e-3 | 次 baseline（gen gap 略优） |
| P0 | **AADWD-Aggressive** | c=0.01, beta=0.999, λ∈[1e-6, 0.01] | 核心方法 |
| P0 | **Norm-Matched-WD** | 匹配 AADWD 的 weight norm 轨迹 | 关键消融（区分 alignment vs norm 轨迹） |
| P1 | **No-WD** | wd=0 | 无正则对照 |
| P1 | **Stagewise-WD** | milestones=[100,150], gamma=0.2 | 简单动态基线（milestone 在 200 epoch 内充分生效） |

### 2.2 可选方法（资源充裕时加入）

| 优先级 | 方法 | 配置 | 目的 |
|--------|------|------|------|
| P2 | **AADWD-Square** | c=0.01, beta=0.999 | 低动态性 AADWD 变体 |
| P2 | **AADWD-Conservative (调参后)** | c=0.001, lambda_max=0.005 | 修复过正则问题后重测 |
| P2 | **Random-Dynamic-WD** | 相同 lambda 范围随机采样 | 纯随机对照 |
| P3 | **CWD (sign-based)** | 仅在资源充裕时考虑 | 计算代价极高（预估 ~2h/run），性价比低 |

---

## 3. 关键超参配置（200 Epoch 专用）

### 3.1 学习率 Milestones 调整（重要！）

Pilot 使用 `lr_milestones=[30, 60, 90]`，**在 200 epoch 实验中不合适**（相当于在 15%/30%/45% 处降 LR，过早且密集）。

**推荐 200-epoch LR schedule：**
```yaml
lr_milestones: [100, 150]   # 50% / 75% 处降 LR，更符合 ResNet 训练习惯
lr_gamma: 0.1               # 比 pilot 的 0.2 更激进（标准做法）
# 等效：LR: 0.1 → 0.01 (epoch 100) → 0.001 (epoch 150)
```

### 3.2 AADWD-Aggressive 推荐配置

```yaml
method: aadwd_aggressive
arch: resnet20
dataset: cifar10
epochs: 200
batch_size: 128
lr: 0.1
momentum: 0.9
lr_milestones: [100, 150]
lr_gamma: 0.1
beta: 0.999             # 已修正（pilot 建议）
c: 0.01                 # 默认配置，鲁棒性验证显示 c 在 [0.001, 0.05] 均可
lambda_min: 1e-06
lambda_max: 0.01
seed: [42, 123, 456]    # 建议 3 个 seed 以量化方差
```

### 3.3 AADWD-Conservative（修复版）配置

```yaml
method: aadwd_conservative
c: 0.001                # 降低 10x（原 0.01 导致 lambda 过快趋向 max）
lambda_max: 0.005       # 适当降低上限（原 0.01 过高）
beta: 0.999
# 其余同 aggressive
```

### 3.4 Norm-Matched-WD 配置说明

该方法核心：**使用 AADWD-Aggressive 的 weight norm 轨迹作为目标，设计能产生相同 norm 衰减的 fixed schedule**。
具体步骤：
1. 先运行一次 AADWD-Aggressive（200 epoch），记录每个 epoch 的 mean weight norm
2. 拟合一个 WD schedule 函数，使 fixed schedule 模型的 weight norm 轨迹与之匹配
3. 用该 schedule 运行 norm_matched_wd

---

## 4. 跨架构实验（VGG16-BN / CIFAR-100）

### 4.1 Pilot 发现的问题

**重要警示**：Pilot 中 VGG16/CIFAR-100 的 Fixed-WD 基线（wd=5e-4）只达 37.15%，而 AADWD 达 48.7%。差距主要来源于 **fixed_wd 在 20 epoch 严重欠拟合**（gen_gap=-0.43%），而非 AADWD 的真实优势。

### 4.2 Full 实验修正方案

```yaml
# 1. 首先运行 WD 网格搜索，为 VGG16/CIFAR-100 找到最优固定 WD
method: fixed_wd_grid
arch: vgg16_bn
dataset: cifar100
epochs: 200
wd_values: [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]   # CIFAR-100 通常需要更小 WD
lr_milestones: [100, 150]
lr_gamma: 0.1

# 2. 用最优 WD 作为 baseline 与 AADWD-Aggressive 对比
method: aadwd_aggressive
arch: vgg16_bn
dataset: cifar100
# 其余配置与 ResNet20 相同
```

---

## 5. 评估指标与监控计划

### 5.1 每次 checkpoint 记录（每 10 epoch）

- `test_acc`, `train_acc`, `gen_gap`
- `weight_norm` (per-layer 和 global)
- `lambda_t`（AADWD 方法）
- `EMA_delta_hat_t`（AADWD 方法）
- `train_loss`, `test_loss`

### 5.2 Alignment Proxy 专项记录（验证 H3）

```yaml
# 每 10 epoch 运行一次 proxy 诊断
proxy_eval_interval: 10
num_mini_batches: 50        # 与 tier0 一致
large_batch_size: 4096
beta_proxy: 0.999           # 修正后的 beta
# 目标：Pearson r >= 0.85 在 200 epoch 内的某个阶段成立
```

### 5.3 关键对比指标（论文主表格）

| 指标 | 说明 |
|------|------|
| `best_test_acc` | 主要精度指标 |
| `final_test_acc` | 最终 epoch 精度 |
| `gen_gap` | 泛化间隙 |
| `weight_norm_final` | 最终权重范数 |
| `lambda_trajectory` | AADWD 的 λ_t 动态（图） |
| `proxy_r_timeline` | 对齐代理 r 随 epoch 变化（图） |

---

## 6. 计算资源预估

| 方法 | 预估时间/run | 备注 |
|------|-------------|------|
| Fixed-WD variants | ~8 min/run | 53s × 10 = ~9 min（估算 200/20 = 10x） |
| AADWD-Aggressive | ~90 min/run | 572s × 10 ≈ 96 min |
| Norm-Matched-WD | ~100 min/run | 592s × 10 ≈ 99 min |
| Stagewise-WD | ~17 min/run | 100s × 10 ≈ 17 min |
| CWD | ~120 min/run | 698s × 10 ≈ 116 min（不推荐） |

**推荐最小实验套件（P0 方法 × 3 seeds）**：
- 6 个 fixed_wd grid + 3 个 aadwd_aggressive + 3 个 norm_matched_wd
= ~12 runs，总时间约 8h（1 张 GPU）

**推荐完整实验套件（P0+P1 方法 × 3 seeds）**：
- 加入 no_wd + stagewise_wd
= ~18 runs，总时间约 10h（1 张 GPU）或 3-4h（2-3 张并行）

---

## 7. 风险管理与应急预案

### 7.1 H2 风险（alignment 边际增益不显著）

**风险**：200 epoch 下 AADWD 与 norm_matched_wd 仍然接近。

**应对**：
- 分析 alignment proxy 轨迹，识别 AADWD λ_t 的主动调整时机
- 设计更细粒度的消融：仅在 LR 下降阶段（epoch 100-150）启用 alignment-aware 调整
- 关注 test_acc standard deviation across seeds：alignment-aware 是否能减小方差

### 7.2 H3 风险（proxy r 仍低于 0.85）

**风险**：即使用 beta=0.999，200 epoch 下 r 仍不达标。

**应对**：
1. 增大 beta 到 0.9995（进一步平滑）
2. 尝试 cand_empirical 候选策略（基于历史梯度统计而非 EMA）
3. 评估是否可以用 weight norm 变化率作为 proxy 替代

### 7.3 Conservative 变体（修复版）失败

**应对**：调整 c=0.0001 或彻底关闭 conservative 路线，聚焦 aggressive 变体

---

## 8. 论文写作建议（基于 Pilot 洞察）

### 8.1 核心叙事主线

**推荐叙事框架**（基于 pilot 结果）：

> "传统固定 WD 忽略了训练动态——在训练早期，梯度对齐信号强，适合强正则；后期信号弱，过强正则化反而有害。AADWD 通过跟踪 alignment proxy 自适应调整 WD，在 ResNet20/CIFAR-10 上实现 XX% 精度提升。"

### 8.2 关键数字（Pilot 版本，供参考）

- 最优固定 WD vs AADWD（在 20 epoch pilot）：89.35% vs 85.09% — **注意：pilot 中固定 WD 领先，200 epoch 结果可能逆转**
- AADWD vs Random-Dynamic-WD：85.09% vs 80.34%（+4.75%）— 支持 alignment-aware 的必要性
- 超参鲁棒性：c 跨越 2 个数量级变化仅 0.65% — 实用性佐证
- 跨架构（初步）：VGG16/CIFAR-100 中 AADWD vs Fixed-WD = 48.7% vs 37.15%（+11.55%）

### 8.3 需要 200 Epoch 才能写的论点

1. "AADWD 在训练后期（LR decay 后）展现更明显的 WD 调整效果"（当前 milestones 未在 20 epoch 内触发）
2. "对齐代理 r 随训练进展显著提升"（20 epoch 仅 0.849，预期 200 epoch 后改善）
3. "最终 test accuracy 超越最优固定 WD"（pilot 中固定 WD 反而更好，需 200 epoch 翻转这一结果）

---

## 9. 执行优先级总结

```
立即执行（必须）:
  1. Fixed-WD (5e-4) × 3 seeds     [8 min × 3]
  2. Fixed-WD (1e-3) × 3 seeds     [8 min × 3]
  3. AADWD-Aggressive × 3 seeds    [90 min × 3] ← 关键
  4. Norm-Matched-WD × 3 seeds     [100 min × 3] ← 消融关键

执行（建议）:
  5. No-WD × 1 seed
  6. Stagewise-WD × 3 seeds  (需修正 milestones=[100,150])

推迟（资源充裕时）:
  7. AADWD-Square × 2 seeds
  8. AADWD-Conservative (修复版) × 2 seeds
  9. VGG16/CIFAR-100 WD 网格搜索 → AADWD 对比

不建议（除非有特殊理由）:
  10. CWD (sign-based)  — 12.9x 计算代价，20 epoch 效果差
```
