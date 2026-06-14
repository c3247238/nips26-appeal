# Iteration 5 实验计划

**日期**: 2026-03-18
**规划者**: Sibyl Planner Agent (Opus 4.6)
**核心目标**: 实验执行优先 — P0 实验全部完成前禁止进入写作阶段
**质量门聚焦**: ImageNet/ResNet-50 实验（4 种方法 × 3 种子 × 90 epochs）

---

## 一、已有数据盘点

| 来源 | 内容 | 状态 |
|------|------|------|
| `iter_003/exp/results/full/cifar10/resnet20/` | ResNet-20/CIFAR-10/AdamW, 7 methods × 3 seeds, 200 epochs | 完成 |
| `iter_003/exp/results/full/cifar100/resnet20/` | ResNet-20/CIFAR-100/AdamW, 7 methods × 3 seeds, 200 epochs | 完成 |
| `iter_003/exp/results/sgd_baseline/` | ResNet-20/SGD, 7 methods × 3 seeds × 2 datasets = 42 runs | 完成（存在 ρ 混淆） |
| `iter_004/exp/results/` (pilot) | VGG-16-BN pilot: 3 methods × 1 seed × 10 epochs | 仅 pilot，无统计效力 |

**关键缺口**：ImageNet 零数据、NoBN 零数据、ρ sweep 零数据、VGG 完整实验零数据、matched-ρ SGD 零数据。

---

## 二、计算资源

- **本地 GPU**: 8× NVIDIA RTX PRO 6000 Blackwell (98GB VRAM each)
- **compute_backend**: local
- **当前占用**: GPU 0/2/3/4/5/6 有不同程度占用，GPU 1 空闲，GPU 7 低占用
- **调度策略**: 单卡任务优先使用空闲 GPU，ImageNet 任务使用 2-GPU DataParallel
- **ImageNet 数据**: 本地未找到，需通过 HuggingFace datasets 或 torchvision 下载（~150GB）

---

## 三、实验矩阵总览

### P0 实验（阻断性，必须全部完成）

| ID | 实验 | 架构 | 数据集 | 方法数 | 种子 | Epochs | 总 runs | GPU 需求 | 预计时间 |
|----|------|------|--------|--------|------|--------|---------|----------|----------|
| P0-1 | NoBN 消融 | ResNet-20-NoBN | CIFAR-10 | 3 | 42,123,456 | 200 | 9 | 1/run | ~1.5h |
| P0-2a | ρ=0.05 sweep | ResNet-20 | CIFAR-10 | 4 | 42,123,456 | 200 | 12 | 1/run | ~1h |
| P0-2b | ρ=5.0 sweep | ResNet-20 | CIFAR-10 | 4 | 42,123,456 | 200 | 12 | 1/run | ~1.5h |
| P0-3a | Matched-ρ SGD | ResNet-20 | CIFAR-10 | 3 | 42,123,456 | 200 | 9 | 1/run | ~1h |
| P0-3b | Matched-ρ SGD | ResNet-20 | CIFAR-100 | 3 | 42,123,456 | 200 | 9 | 1/run | ~1h |
| P0-4 | VGG-16-BN 完整 | VGG-16-BN | CIFAR-10 | 7 | 42,123,456 | 200 | 21 | 1/run | ~5-6h |
| P0-α1 | ImageNet seed=42 | ResNet-50 | ImageNet-1K | 4 | 42 | 90 | 4 | 2/run | ~2-3h |
| P0-α2 | ImageNet seed=123 | ResNet-50 | ImageNet-1K | 4 | 123 | 90 | 4 | 2/run | ~2-3h |
| P0-α3 | ImageNet seed=456 | ResNet-50 | ImageNet-1K | 4 | 456 | 90 | 4 | 2/run | ~2-3h |

**总计**: ~84 runs, ~60-80 GPU-hours, ~15-18h wall-clock（8 GPU 并行调度）

---

## 四、实验参数详细规格

### P0-1: NoBN 消融实验

```yaml
architecture: resnet20_nobn  # 所有 BatchNorm → Identity
dataset: cifar10
optimizer: AdamW
lr: 5e-4  # 降低 LR 提升 NoBN 稳定性
wd: 5e-4
wd_methods: [constant, cwd_hard, no_wd]
seeds: [42, 123, 456]
epochs: 200
batch_size: 128
lr_schedule: cosine
gradient_clipping: max_norm=1.0
warmup_epochs: 10
```

**代码变更**: 在 `models.py` 中新增 `resnet20_nobn()` 函数，将 `BasicBlock` 和 `ResNet` 中所有 `nn.BatchNorm2d` 替换为 `nn.Identity()`。

**可证伪预测**:
- 若 NoBN 展布 < 0.5% → ℓ∞ 约束是充分机制（强结论）
- 若 NoBN 展布 > 1% → BN 是 Phi 不变性的必要条件

**稳定性缓解措施**: 降低 lr 到 5e-4、添加 gradient clipping (max_norm=1.0)、10 epoch warmup。若仍然发散，进一步降低 lr 到 1e-4。

### P0-2: ρ 扫描实验

```yaml
architecture: resnet20
dataset: cifar10
optimizer: AdamW
lr: 1e-3

# ρ=0.05 配置（Regime I 深处）
rho_low:
  wd: 5e-5  # wd/lr = 0.05
  methods: [constant, cwd_hard, half_lambda, no_wd]

# ρ=5.0 配置（Regime I-II 边界）
rho_high:
  wd: 5e-3  # wd/lr = 5.0
  methods: [constant, cwd_hard, half_lambda, no_wd]
  gradient_clipping: max_norm=5.0
  early_stopping: train_loss > 5.0 after 20 epochs
  fallback: wd=2e-3 (ρ=2.0)

# 已有数据: ρ=0.5 (wd=5e-4) from iter_003
seeds: [42, 123, 456]
epochs: 200
batch_size: 128
```

**理论预测**:
- ρ=0.05: 展布 < 0.1%（深 Regime I，WD 方法几乎无区别）
- ρ=0.5: 展布 ~0.05-0.1%（已有数据）
- ρ=5.0: 展布 1-3%（Regime I-II 边界，方法差异放大）

### P0-3: Matched-ρ SGD 对照

```yaml
architecture: resnet20
datasets: [cifar10, cifar100]
optimizer: SGDW (decoupled)
lr: 0.01  # 注意：与原始 SGD lr=0.1 不同
momentum: 0.9
wd: 5e-3  # wd/lr = 0.5，匹配 AdamW ρ
wd_methods: [constant, cwd_hard, no_wd]
seeds: [42, 123, 456]
epochs: 200
batch_size: 128
lr_schedule: cosine
```

**目的**: 消除 18.3× 效应量比中的 ρ 混淆（原始 SGD ρ=0.005 vs AdamW ρ=0.5，相差 100 倍）。

**代码变更**: `train_unified.py` 需增加 `--optimizer sgd` 参数，调用 `create_sgd_optimizer()`。

### P0-4: VGG-16-BN 完整实验

```yaml
architecture: vgg16_bn
dataset: cifar10
optimizer: AdamW
lr: 1e-3
wd: 5e-4
wd_methods: [constant, cwd_hard, half_lambda, cosine_schedule, swd, random_mask, no_wd]
seeds: [42, 123, 456]
epochs: 200
batch_size: 128
lr_schedule: cosine
```

**已知**: cwd_hard 2.3× 慢于 constant（pilot 确认）。调度策略：优先派发非 cwd_hard 方法，cwd_hard 最后。

### P0-ALPHA: ImageNet/ResNet-50 核心实验 【最高优先级】

```yaml
architecture: resnet50  # torchvision.models.resnet50 或自定义
dataset: imagenet  # ImageNet-1K (ILSVRC2012)
optimizer: AdamW
lr: 1e-3
betas: [0.9, 0.999]
wd: 1e-4  # ρ=0.1，进入 Regime I 更深处
wd_methods: [constant, cwd_hard, cosine_schedule, no_wd]
seeds: [42, 123, 456]
epochs: 90
batch_size: 256  # per GPU
total_batch_size: 512  # 2 GPU DataParallel
lr_schedule: cosine with 5-epoch linear warmup
precision: bf16 mixed precision (torch.amp)
num_workers: 8  # per GPU
```

**新代码需求**:
1. `models.py` 添加 ResNet-50 工厂函数（使用 torchvision 预定义架构）
2. `data.py` 添加 ImageNet 数据加载（标准 torchvision.datasets.ImageNet + 预处理）
3. `train_unified.py` 扩展：
   - bf16 mixed precision (GradScaler + autocast)
   - DataParallel 多 GPU 支持
   - top-5 accuracy 记录
   - 逐层 ρ 热力图数据记录（`per_layer_rho` dict per epoch）
   - LR warmup 调度器

**诊断指标（每 epoch 零额外 GPU 成本）**:
- `test_top1_acc`, `test_top5_acc`, `train_loss`
- 逐层: `weight_norm`, `gradient_norm`, `rho_per_layer` (‖g‖/‖w‖)
- 逐层: `alignment_cosine` cos(g, w)
- BEM, CSI, AIS
- `sign_flip_rate`（CWD 方法专用）

**分阶段执行**:
```
Phase 1: seed=42, 4 methods, 4 GPU (2 per run × 2 并行) → ~2-3h → 验证基础设施
Phase 2: seed=123, 4 methods → ~2-3h
Phase 3: seed=456, 4 methods → ~2-3h
```

---

## 五、Pilot 验证策略

每个新实验类型先运行 pilot（samples=100%数据集，seed=42，5 epochs），确认：
1. 训练不发散（loss < 5.0）
2. 无 OOM
3. 诊断指标正确记录
4. 计时准确（估算完整实验时间）

| Pilot | Pass 标准 | 超时 |
|-------|----------|------|
| NoBN pilot | Loss < 5.0 @ epoch 5, no OOM | 900s |
| ρ sweep pilot | ρ=0.05 稳定；ρ=5.0 稳定或 fallback | 600s |
| ImageNet infra pilot | 1 epoch 完成，top-1 > 10%，诊断指标正常 | 1800s |

---

## 六、GPU 调度时间表

```
=== 8× RTX PRO 6000 并行调度 ===

[Wave 0: Pilot 验证] T+0h ~ T+0.5h
  GPU 1: pilot_nobn (5 epochs, ~5min)
  GPU 7: pilot_rho_sweep (5 epochs ×2, ~10min)

[Wave 1: CIFAR 消融] T+0.5h ~ T+2.5h（pilot 通过后立即启动）
  GPU 1: NoBN full (9 runs sequential, ~1.5h)
  GPU 7: ρ=0.05 full (12 runs, ~1h)
  GPU 1→空闲后: ρ=5.0 full (12 runs, ~1.5h)
  空闲 GPU: matched-ρ SGD CIFAR-10 (9 runs, ~1h)
  空闲 GPU: matched-ρ SGD CIFAR-100 (9 runs, ~1h)

[Wave 2: VGG-16-BN] T+2h ~ T+8h（与 Wave 1 可部分重叠）
  GPUs 空闲后: VGG-16-BN 21 runs 分配到可用 GPU
  非 cwd_hard 方法优先（每 run ~1.2h）
  cwd_hard 方法最后（每 run ~2.8h）

[Wave 3: ImageNet 基础设施] T+0h ~ T+0.5h（与 Wave 0 并行）
  GPU 空闲 2 块: pilot_imagenet (1 epoch, ~20min)

[Wave 4: ImageNet Full Phase 1-2] pilot 通过后
  4 GPU (2 per run × 2 并行): seed=42 4 methods (~2-3h)
  4 GPU: seed=123 4 methods (~2-3h)

[Wave 5: ImageNet Phase 3 + 分析]
  4 GPU: seed=456 4 methods (~2-3h)
  CPU: 数据分析和图表生成

总计: ~15-18h wall-clock，~60-80 GPU-hours
```

**注意**: 由于多数 GPU 当前被占用，实际调度取决于 GPU 释放时间。系统使用 `gpu_poll` 机制持续等待空闲 GPU。

---

## 七、Gate 决策逻辑

### Gate 1 — CIFAR 消融完成后（~T+2.5h）

**触发条件**: P0-1 (NoBN) + P0-2 (ρ sweep) + P0-3a (matched-ρ SGD CIFAR-10) 全部完成

**决策**:
1. NoBN 展布判断 → 决定 ℓ∞ 路径 vs BN 旋转平衡路径的声明强度
2. ρ sweep 展布-ρ 曲线 → 决定三元体系声明（Conjecture vs Empirical Finding）
3. Matched-ρ SGD 效应量比 → 确认 ρ 是否为 18.3× 的混淆变量

### Gate 2 — VGG 完成后（~T+8h）

**触发条件**: P0-4 (VGG-16-BN) 完成

**决策**:
1. VGG Phi spread < 0.5% → 跨架构泛化确认
2. VGG 效应量与 ResNet-20 对比 → 参数量从 270K→15M 的尺度效应

### Gate 3 — ImageNet Phase 1 完成后（~T+6h+）

**触发条件**: P0-α1 (ImageNet seed=42) 完成

**决策**:
1. ImageNet 4 方法准确率差异 → 决定论文声明范围（场景 A/B/C/D）
2. 逐层 ρ 热力图质量 → 确认核心贡献图的新颖性
3. 是否继续 Phase 2-3 → 如果 Phase 1 显示方向正确则继续

### 写作门（硬约束）

**最低条件**: Gate 1 + Gate 2 通过
**推荐条件**: Gate 1 + Gate 2 + Gate 3 通过
**绝对禁止**: Gate 1 未通过时进入 writing 阶段

---

## 八、论文声明范围（基于实验结果动态调整）

| 场景 | 完成范围 | 声明强度 | 预期分数 |
|------|---------|---------|---------|
| A (理想) | 全部 P0 + ImageNet 3 seeds | 统一动态 WD 框架，跨 3 架构 × 2 尺度验证 | 7.5-8.0 |
| B | P0 + ImageNet 1-2 seeds | 框架保留，ImageNet "方向一致性" | 7.0-7.5 |
| C | P0-1/2/3/4 完成，无 ImageNet | 降级为 CIFAR-scale + 消融分析 | 6.5-7.0 |
| D | P0 仍未完成 | 彻底重新定位为实证观察报告 | 5.5-6.0 |

---

## 九、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| ImageNet 数据下载 ~150GB | 高 | 延迟 ImageNet 启动 | 优先用 HuggingFace streaming；并行下载不阻塞 CIFAR 实验 |
| NoBN 训练不稳定 | 高 | P0-1 无法完成 | lr=5e-4, grad clip=1.0, warmup=10; fallback lr=1e-4 |
| ρ=5.0 训练发散 | 高 | P0-2b 部分失败 | grad clip=5.0, early stop, fallback ρ=2.0 |
| GPU 被长期占用 | 中 | 延迟所有实验 | gpu_poll 持续轮询，aggressive mode 接管低占用 GPU |
| cwd_hard VGG 过慢 | 已确认 | P0-4 时间延长 | 接受 2.3× 开销，最后调度 |
| ImageNet 90 epochs 超时 | 中 | 减少种子数 | bf16 加速; 接受 N=2 seeds (场景 B) |

---

## 十、新代码需求清单

### models.py 扩展

1. **`resnet20_nobn(num_classes)`**: ResNet-20 变体，所有 BatchNorm2d → Identity
2. **`resnet50(num_classes=1000)`**: 使用 `torchvision.models.resnet50(weights=None)` 或自定义实现

### data.py 扩展

3. **`get_imagenet_dataloaders()`**: ImageNet-1K 标准数据加载
   - Train: RandomResizedCrop(224) + RandomHorizontalFlip + Normalize
   - Val: Resize(256) + CenterCrop(224) + Normalize
   - 优先检查本地路径，fallback 到 HuggingFace datasets

### train_unified.py 扩展

4. **`--optimizer` 参数**: 支持 `adamw`（默认）和 `sgdw`
5. **bf16 mixed precision**: `torch.amp.autocast` + `GradScaler`
6. **DataParallel**: `torch.nn.DataParallel` 支持多 GPU
7. **top-5 accuracy**: ImageNet 评估时记录 top-1 和 top-5
8. **逐层 ρ 数据**: 每 epoch 记录 `per_layer_rho` dict 到 epoch_metrics.jsonl
9. **LR warmup**: 线性 warmup 调度器（前 N epoch）
10. **gradient clipping**: `--grad_clip_norm` 参数

---

## 十一、成功标准

| 层级 | 完成范围 | 预期分数 |
|------|---------|---------|
| 最低及格 | P0-1 + P0-2 + P2 修复 | 6.5-7.0 |
| 目标标准 | 全部 CIFAR P0 + VGG | 7.0-7.5 |
| 理想标准 | 全部 P0 + ImageNet 3 seeds | 7.5-8.0 |
| 提交就绪 | 以上全部 + 完整图表 + 自动验证 | 8.0+ |

**核心信条**: ImageNet 实验是最高优先级，CIFAR P0 消融是基础保障，写作和理论工作必须等待实验全部完成。
