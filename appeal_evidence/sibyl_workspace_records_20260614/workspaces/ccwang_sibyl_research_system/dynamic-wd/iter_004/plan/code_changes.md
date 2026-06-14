# Iteration 4 代码修改计划

**日期**: 2026-03-18
**涉及文件**: `exp/code/` 目录下的训练代码

---

## 一、Bug 修复（P0）

### 1.1 修复 BEM 计算 — `optimizers.py` + `train_unified.py`

**问题根因**: `HalfLambdaPhi.get_per_param_wd()` 返回 `self.base_wd * 0.5`，但 `get_metrics()` 中的 `effective_wd` 分支逻辑无法捕获这个值。HalfLambdaPhi 没有设置任何诊断键（`wd_schedule`、`wd_multiplier`、`mask_ratio` 都不存在），所以 fallback 到 `self.phi.base_wd`，即完整的 5e-4 而非实际的 2.5e-4。

**修改 1**: `optimizers.py` — HalfLambdaPhi 类

```python
# 修改前
class HalfLambdaPhi(PhiModulator):
    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        return self.base_wd * 0.5

# 修改后
class HalfLambdaPhi(PhiModulator):
    def get_per_param_wd(self, param, grad, state, step, total_steps, lr):
        effective = self.base_wd * 0.5
        self._diagnostics['effective_wd_scalar'] = effective
        return effective
```

**修改 2**: `optimizers.py` — `UnifiedAdamW.get_metrics()` 和 `UnifiedSGDW.get_metrics()`

在 `get_metrics()` 的 effective_wd 计算中，添加对 `effective_wd_scalar` 的检测：

```python
# 在 get_metrics() 中添加优先级最高的分支
diag = self.phi.get_diagnostics()
if 'effective_wd_scalar' in diag:
    metrics['effective_wd'] = diag['effective_wd_scalar']
elif 'wd_schedule' in diag:
    metrics['effective_wd'] = diag['wd_schedule']
# ... 其余不变
```

**修改 3**: `train_unified.py` — `compute_bem()`

```python
# 修改前
def compute_bem(mean_wd_schedule, constant_wd, tolerance=0.01):
    if constant_wd < 1e-10:
        return 0.0
    return abs(mean_wd_schedule - constant_wd) / constant_wd

# 修改后：有向 BEM（去绝对值）
def compute_bem(mean_wd_schedule, constant_wd):
    """Budget Equivalence Metric (signed).

    BEM = (mean_wd - constant_wd) / constant_wd
    BEM > 0: over-budget, BEM < 0: under-budget, BEM = 0: exactly equivalent.
    """
    if constant_wd < 1e-10:
        return 0.0
    return (mean_wd_schedule - constant_wd) / constant_wd
```

### 1.2 修复 AIS 定义 — `train_unified.py`

**当前状态**: 实现实际上是正确的（取 abs 后做 entropy 归一化，范围 [0,1]）。问题在论文描述而非代码。

**代码改进**（增加文档清晰度）:

```python
def compute_ais(per_layer_alignments):
    """Alignment Informativeness Score.

    Measures diversity of per-layer |cos(g, w)| via normalized entropy.
    Range: [0, 1]. High AIS = alignment varies meaningfully across layers.

    Input: list of |cos(g_l, w_l)| values (already abs'd).
    Formula: AIS = H({|cos(g_l, w_l)|}_{l=1}^L) / log(K)
    where K is the number of bins.
    """
    # ... 实现不变，仅更新 docstring
```

### 1.3 修复 CSI 定义 — `train_unified.py`

```python
# 修改前：变异系数定义
def compute_csi(weight_norms_history, window=10):
    # std(deltas) / mean(|deltas|)

# 修改后：保留实现，添加归一化接口
def compute_csi(weight_norms_history, window=10):
    """Coupling Stability Index: CV of weight norm changes.

    CSI = std(Δ‖w‖) / (mean(|Δ‖w‖|) + eps)
    Lower = more stable. Normalize to constant baseline: CSI_rel = CSI / CSI_constant.
    """
    # 实现不变

def compute_csi_relative(csi_method, csi_constant):
    """Relative CSI normalized to constant baseline.
    CSI_rel = CSI_method / CSI_constant. Constant baseline = 1.0.
    """
    if csi_constant < 1e-10:
        return 1.0
    return csi_method / csi_constant
```

---

## 二、功能新增

### 2.1 SGD 训练脚本统一 — `train_unified.py`

**当前状态**: `train_sgd.py` 是独立脚本，与 `train_unified.py` 有部分重复。

**方案**: 在 `train_unified.py` 中添加 `--optimizer` 参数：

```python
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'sgd'])
```

在 `run_training()` 中根据 optimizer 选择：

```python
if config['optimizer'] == 'sgd':
    from optimizers import create_sgd_optimizer
    optimizer = create_sgd_optimizer(
        model, config['wd_method'],
        lr=config['lr'], wd=config['wd'],
        epochs=config['epochs'], batch_size=config['batch_size'],
        dataset_size=dataset_size,
        momentum=config.get('momentum', 0.9),
        **kwargs
    )
else:
    optimizer = create_unified_optimizer(...)
```

### 2.2 VGG-16-BN 模型支持验证 — `models.py`

**当前状态**: VGG-16-BN 已在 `models.py` 中实现（`vgg16_bn()` 函数），`create_model()` 工厂函数已支持 `vgg16_bn`。

**无需修改**。仅需在 Pilot 中验证 CIFAR-10/100 上的训练正确性。

### 2.3 TOST 统计检验 — 新文件 `exp/code/statistical_analysis.py`

```python
"""
Statistical analysis module for Iteration 4.

Provides:
- TOST (Two One-Sided Tests) equivalence testing
- Power analysis
- Bonferroni-Holm multiple comparison correction
- Effect size computation (Cohen's d)
"""

from scipy import stats
import numpy as np

def tost_paired(x, y, delta, alpha=0.05):
    """TOST paired equivalence test.

    H0: |μ_x - μ_y| >= delta (not equivalent)
    H1: |μ_x - μ_y| < delta (equivalent)

    Returns: (equivalent: bool, p_value: float, ci_90: tuple)
    """
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean_d = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    df = n - 1

    # Upper test: H0: μ_d >= delta
    t_upper = (mean_d - delta) / se
    p_upper = stats.t.cdf(t_upper, df)

    # Lower test: H0: μ_d <= -delta
    t_lower = (mean_d + delta) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)

    p_tost = max(p_upper, p_lower)

    # 90% CI for equivalence
    t_crit = stats.t.ppf(1 - alpha, df)
    ci_90 = (mean_d - t_crit * se, mean_d + t_crit * se)

    return p_tost < alpha, p_tost, ci_90

def power_analysis_paired(n, std, delta, alpha=0.05):
    """Compute power for paired t-test.

    Returns: power for detecting effect of size delta.
    """
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, n-1)
    ncp = delta / se  # non-centrality parameter
    power = 1 - stats.t.cdf(t_crit, n-1, loc=ncp) + stats.t.cdf(-t_crit, n-1, loc=ncp)
    return power

def minimum_detectable_effect(n, std, alpha=0.05, power=0.80):
    """Find MDE via binary search."""
    lo, hi = 0.0, 5 * std
    for _ in range(100):
        mid = (lo + hi) / 2
        if power_analysis_paired(n, std, mid, alpha) > power:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

def cohens_d_paired(x, y):
    """Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    return diff.mean() / (diff.std(ddof=1) + 1e-10)

def bonferroni_holm(p_values):
    """Bonferroni-Holm step-down correction.

    Returns: list of (original_idx, p_corrected, reject).
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    results = []
    for rank, idx in enumerate(sorted_indices):
        p_adj = min(p_values[idx] * (n - rank), 1.0)
        results.append((idx, p_adj))
    # Enforce monotonicity
    for i in range(1, len(results)):
        if results[i][1] < results[i-1][1]:
            results[i] = (results[i][0], results[i-1][1])
    return [(idx, p, p < 0.05) for idx, p in results]
```

### 2.4 批量实验运行脚本 — 新文件 `exp/code/run_iter4_experiments.sh`

```bash
#!/bin/bash
# Iteration 4 experiment runner
# Phase 2A: VGG-16-BN AdamW (70 runs)
# Phase 2B: ResNet-20 extra seeds (28 runs)

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="${CODE_DIR}/../results"

METHODS="constant cosine_schedule cwd_hard cwd_soft swd half_lambda no_wd"
DATASETS="cifar10 cifar100"
SEEDS_FULL="42 123 456 789 999"
SEEDS_EXTRA="789 999"

GPU_COUNT=8
gpu_idx=0

# Phase 2A: VGG-16-BN AdamW full
for dataset in $DATASETS; do
  for method in $METHODS; do
    for seed in $SEEDS_FULL; do
      output_dir="${RESULTS_BASE}/adamw/${dataset}/vgg16_bn/${method}/seed_${seed}"
      if [ -f "${output_dir}/_DONE" ]; then
        echo "SKIP: ${output_dir} (already done)"
        continue
      fi
      CUDA_VISIBLE_DEVICES=$gpu_idx python3 "${CODE_DIR}/train_unified.py" \
        --arch vgg16_bn --dataset $dataset --wd_method $method \
        --epochs 200 --lr 1e-3 --wd 5e-4 --seed $seed \
        --output_dir "$output_dir" --gpu_id 0 &
      gpu_idx=$(( (gpu_idx + 1) % GPU_COUNT ))
      # Wait when all GPUs busy
      if [ $gpu_idx -eq 0 ]; then
        wait
      fi
    done
  done
done

# Phase 2B: ResNet-20 extra seeds
for dataset in $DATASETS; do
  for method in $METHODS; do
    for seed in $SEEDS_EXTRA; do
      output_dir="${RESULTS_BASE}/adamw/${dataset}/resnet20/${method}/seed_${seed}"
      if [ -f "${output_dir}/_DONE" ]; then
        echo "SKIP: ${output_dir} (already done)"
        continue
      fi
      CUDA_VISIBLE_DEVICES=$gpu_idx python3 "${CODE_DIR}/train_unified.py" \
        --arch resnet20 --dataset $dataset --wd_method $method \
        --epochs 200 --lr 1e-3 --wd 5e-4 --seed $seed \
        --output_dir "$output_dir" --gpu_id 0 &
      gpu_idx=$(( (gpu_idx + 1) % GPU_COUNT ))
      if [ $gpu_idx -eq 0 ]; then
        wait
      fi
    done
  done
done

wait
echo "All experiments complete."
```

### 2.5 SGD VGG-16-BN 运行脚本 — 新文件 `exp/code/run_iter4_sgd.sh`

```bash
#!/bin/bash
# Phase 2C: VGG-16-BN SGD control experiments (24 runs)

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="${CODE_DIR}/../results"

SGD_METHODS="constant cosine_schedule cwd_hard no_wd"
DATASETS="cifar10 cifar100"
SEEDS="42 123 456"
GPU_COUNT=4
gpu_idx=0

for dataset in $DATASETS; do
  for method in $SGD_METHODS; do
    for seed in $SEEDS; do
      output_dir="${RESULTS_BASE}/sgd_baseline/${dataset}/vgg16_bn/${method}/seed_${seed}"
      if [ -f "${output_dir}/_DONE" ]; then
        echo "SKIP: ${output_dir} (already done)"
        continue
      fi
      CUDA_VISIBLE_DEVICES=$gpu_idx python3 "${CODE_DIR}/train_unified.py" \
        --arch vgg16_bn --dataset $dataset --wd_method $method \
        --optimizer sgd \
        --epochs 200 --lr 0.1 --wd 5e-3 --seed $seed \
        --lr_schedule multistep \
        --output_dir "$output_dir" --gpu_id 0 &
      gpu_idx=$(( (gpu_idx + 1) % GPU_COUNT ))
      if [ $gpu_idx -eq 0 ]; then
        wait
      fi
    done
  done
done

wait
echo "SGD experiments complete."
```

### 2.6 可视化脚本 — 新文件 `exp/code/visualize_iter4.py`

需要新建可视化脚本，生成：
1. Training curves（test_acc vs epoch, mean±std）
2. Weight norm trajectories
3. SGD vs AdamW effect size 对比条形图
4. TOST equivalence interval 图
5. BEM/AIS/CSI heatmap
6. Cross-architecture comparison

依赖: `matplotlib`, `seaborn`, `numpy`, `scipy`

---

## 三、现有结果目录结构调整

### 3.1 Iter 3 结果兼容

Iter 3 的 AdamW 结果在非标准路径 `iter_003/exp/results/sgd_baseline/cifar10/resnet20/constant/` 中。

**方案**: Iter 4 分析脚本支持读取 Iter 3 旧路径和 Iter 4 新路径：

```
# Iter 3 路径（AdamW，现有 3 seeds）
iter_003/exp/results/adamw/{dataset}/resnet20/{method}/seed_{seed}/

# Iter 3 路径（SGD baseline）
iter_003/exp/results/sgd_baseline/{dataset}/resnet20/{method}/seed_{seed}/

# Iter 4 路径（新实验）
iter_004/exp/results/adamw/{dataset}/{arch}/{method}/seed_{seed}/
iter_004/exp/results/sgd_baseline/{dataset}/{arch}/{method}/seed_{seed}/
```

### 3.2 合并分析

5-seed 分析需要合并 Iter 3 的 3 seeds + Iter 4 的 2 seeds。分析脚本需要支持跨迭代读取。

---

## 四、修改优先级总结

| 优先级 | 文件 | 修改类型 | 预计时间 |
|--------|------|----------|----------|
| P0 | `optimizers.py` | Bug fix: HalfLambdaPhi 诊断 + get_metrics effective_wd | 30 分钟 |
| P0 | `train_unified.py` | Bug fix: compute_bem 去绝对值 | 15 分钟 |
| P0 | `train_unified.py` | Enhancement: compute_csi_relative | 15 分钟 |
| P0 | `train_unified.py` | Feature: --optimizer sgd 参数 | 30 分钟 |
| P1 | `statistical_analysis.py` | 新文件: TOST + power analysis | 1 小时 |
| P1 | `run_iter4_experiments.sh` | 新文件: 批量运行脚本 | 30 分钟 |
| P1 | `run_iter4_sgd.sh` | 新文件: SGD 批量运行 | 20 分钟 |
| P1 | `visualize_iter4.py` | 新文件: 可视化脚本 | 2 小时 |
| P2 | `models.py` | Feature: ResNet-50 (for ImageNet) | 30 分钟 |
| P2 | `data.py` | Feature: ImageNet dataloader | 1 小时 |

**总计代码修改时间**: 约 6-7 小时
