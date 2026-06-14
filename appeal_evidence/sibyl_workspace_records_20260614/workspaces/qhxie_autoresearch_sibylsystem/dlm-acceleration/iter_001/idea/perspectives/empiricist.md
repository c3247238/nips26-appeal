# 实验主义者视角：DLM 推理加速实验设计

## 核心实验矩阵

本研究聚焦于 LLaDA-8B-Instruct 上四种 training-free 加速方法的系统性比较。实验矩阵为 **方法组合 × 任务类型**，共覆盖 15 种配置（4 单方法 + 6 两两组合 + 4 三方法组合 + 1 全组合）× 3 个 benchmark，加上 1 个无加速基线。

### 方法编号约定

| 编号 | 方法 | 代表实现 | 核心机制 |
|------|------|---------|---------|
| M1 | KV Caching | Fast-dLLM / dKV-Cache / EntropyCache | 在相邻扩散步间复用 KV 对，减少冗余注意力计算 |
| M2 | Adaptive Step Scheduling | Saber / 自定义实现 | 根据 token 置信度动态分配扩散步数，高置信 token 提前退出 |
| M3 | AR-guided Unmasking | FlashDLM Guided Diffusion | 轻量 AR 模型指导 unmask 顺序或候选 token 生成 |
| M4 | Speculative Decoding | DualDiffusion 思路 / 自实现 | 小模型草稿 + 大模型并行验证，利用 DLM 天然并行验证能力 |

### 任务 × 方法完整矩阵

| 配置 | GSM8K | MATH500 | HumanEval |
|------|-------|---------|-----------|
| Baseline（无加速，64 步） | B-G | B-M | B-H |
| M1 单独 | 1-G | 1-M | 1-H |
| M2 单独 | 2-G | 2-M | 2-H |
| M3 单独 | 3-G | 3-M | 3-H |
| M4 单独 | 4-G | 4-M | 4-H |
| M1+M2 | 12-G | 12-M | 12-H |
| M1+M3 | 13-G | 13-M | 13-H |
| M1+M4 | 14-G | 14-M | 14-H |
| M2+M3 | 23-G | 23-M | 23-H |
| M2+M4 | 24-G | 24-M | 24-H |
| M3+M4 | 34-G | 34-M | 34-H |
| M1+M2+M3 | 123-G | 123-M | 123-H |
| M1+M2+M4 | 124-G | 124-M | 124-H |
| M1+M3+M4 | 134-G | 134-M | 134-H |
| M2+M3+M4 | 234-G | 234-M | 234-H |
| M1+M2+M3+M4 | 1234-G | 1234-M | 1234-H |

**总实验数**: 16 配置 × 3 benchmark = **48 个实验单元**（每个单元含 3 次随机种子运行）。

---

## 实验方案 1：单方法基线评估（Phase 1）

### 目标
建立每种方法的独立 speed-accuracy Pareto 曲线，作为后续正交性和可组合性分析的基础。

### 实验设计

#### 1.1 Vanilla Baseline
- **模型**: LLaDA-8B-Instruct（HuggingFace: `GSAI-ML/LLaDA-8B-Instruct`）
- **扩散步数**: T = 64（官方默认）
- **解码策略**: 均匀 masking schedule，random order unmasking
- **硬件**: 单张 A100 80GB（或 H100 80GB，需在论文中明确标注）
- **温度**: 按 benchmark 标准设定（GSM8K: greedy, HumanEval: temperature=0.8 for pass@10, temperature=0 for pass@1）
- **生成长度**: GSM8K/MATH 最大 1024 tokens, HumanEval 最大 512 tokens
- **重复次数**: 3 次随机种子（seed=42, 123, 456），报告均值 ± 标准差

#### 1.2 M1: KV Caching 变体扫描
- **变体 1**: dKV-Cache-Decode（保守，near-lossless）
- **变体 2**: dKV-Cache-Greedy（激进，速度优先）
- **变体 3**: EntropyCache（基于 decoded token entropy 决定是否刷新）
- **变体 4**: Fast-dLLM block-wise KV cache
- **超参扫描**: cache refresh interval ∈ {1, 2, 4, 8, 16} 步（或各方法等价超参）
- **选择策略**: 在 GSM8K-mini（前 200 样本）上选择 Pareto 最优变体，后续实验固定该配置
- **控制变量**: 扩散总步数 T=64 不变，仅改变 KV 复用策略

#### 1.3 M2: Adaptive Step Scheduling 变体扫描
- **基础方案**: Saber 的 adaptive acceleration（随上下文增长，每步 unmask 更多 token）
- **置信度阈值扫描**: confidence threshold ∈ {0.7, 0.8, 0.9, 0.95, 0.99}
- **退出策略**: token 置信度超过阈值后不再参与后续扩散步
- **最小步数约束**: 每个 token 至少经历 min_steps ∈ {4, 8, 16} 步
- **控制变量**: 不使用 KV cache，不改变 unmask 顺序

#### 1.4 M3: AR-guided Unmasking
- **AR 指导模型选择**:
  - 方案 A: 外部小 AR 模型（如 Qwen2.5-0.5B 或 TinyLlama-1.1B）指导 unmask 顺序
  - 方案 B: FlashDLM 的 Guided Diffusion 策略（AR 模型提供 token 候选排名）
- **指导强度**: guidance weight ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
- **指导频率**: 每 k 步使用 AR 指导，k ∈ {1, 2, 4, 8}
- **控制变量**: 扩散步数 T=64，无 KV cache，无 adaptive scheduling

#### 1.5 M4: Speculative Decoding
- **草稿模型**: LLaDA-8B 的量化版本（INT4/INT8）或 early-exit variant（如只用前 16 层）
- **草稿步数**: draft 一次生成 k ∈ {2, 4, 8, 16} 个 token 的候选
- **验证策略**: 全模型单步验证（利用 DLM 天然的并行验证能力）
- **接受准则**: 
  - 方案 A: token-level 概率比较（类似 AR speculative decoding）
  - 方案 B: sequence-level 分数比较（diffusion score matching）
- **控制变量**: 验证模型始终为完整 LLaDA-8B-Instruct, T=64

### Phase 1 输出
- 每种方法的 speed-accuracy Pareto 曲线（x 轴: tokens/sec, y 轴: benchmark score）
- 每种方法的最优超参配置（后续实验使用）
- 各方法的计算 profile（FLOPs breakdown、内存占用、GPU utilization）

---

## 实验方案 2：正交性验证（Phase 2）

### 目标
系统验证哪些方法对可以"无冲突"地叠加——即组合后不会引入额外的质量降级（超过各自单独使用的降级之和）。

### 正交性的操作化定义

两种方法 Mi 和 Mj 被定义为**正交**，当且仅当：

$$\text{Acc}(Mi \cup Mj) \geq \text{Acc}(Mi) + \text{Acc}(Mj) - \text{Acc}(\text{Baseline}) - \epsilon$$

其中 epsilon 为统计容差（设为 baseline accuracy 的 2%）。

等价地，定义**正交性分数**：

$$\text{Ortho}(Mi, Mj) = \frac{\text{Acc}(Mi \cup Mj) - \text{Acc}(\text{Baseline})}{\text{Acc}(Mi) - \text{Acc}(\text{Baseline}) + \text{Acc}(Mj) - \text{Acc}(\text{Baseline})}$$

- Ortho ≈ 1.0: 完全正交（accuracy drop 可加性）
- Ortho > 1.0: 协同增效（组合比预期更好）
- Ortho < 1.0: 存在冲突/干扰

### 实验设计

#### 2.1 两两组合实验
对 C(4,2) = 6 种两两组合，使用 Phase 1 确定的最优超参：

| 组合 | 预期正交性 | 理论依据 |
|------|-----------|---------|
| M1+M2 (KV Cache + Adaptive Steps) | 高 | KV cache 减少单步计算，adaptive steps 减少总步数，两者作用于不同维度 |
| M1+M3 (KV Cache + AR-guided) | 中-高 | KV cache 加速前向计算，AR guidance 改变 unmask 顺序，但 AR guidance 可能改变 KV 分布使 cache 失效 |
| M1+M4 (KV Cache + Speculative) | 高 | KV cache 加速验证步骤的前向计算，speculative 减少验证次数 |
| M2+M3 (Adaptive Steps + AR-guided) | 中 | 两者都影响"哪些 token 在哪些步骤被处理"，可能存在控制权冲突 |
| M2+M4 (Adaptive Steps + Speculative) | 低-中 | Speculative decoding 的草稿长度与 adaptive scheduling 的动态步数可能互相干扰 |
| M3+M4 (AR-guided + Speculative) | 低 | 两者都引入外部模型（AR guide vs. draft model），内存开销叠加，可能竞争 GPU 资源 |

#### 2.2 统计检验
- 每个组合运行 3 次，收集 accuracy 和 throughput
- 使用 paired t-test 或 Wilcoxon signed-rank test 检验正交性分数是否显著偏离 1.0
- 显著性水平 alpha = 0.05，并报告 effect size (Cohen's d)

#### 2.3 干扰分析
对 Ortho < 0.8 的组合，进一步分析干扰机制：
- **KV 分布漂移**: 测量组合前后 KV cache hit rate 的变化
- **Step 分配冲突**: 可视化 token-level 的步数分配差异
- **内存竞争**: Profile 峰值 GPU 内存占用，对比单方法之和

### Phase 2 输出
- 6 × 3 的正交性分数矩阵（6 组合 × 3 benchmark）
- 正交/非正交的分类（热力图可视化）
- 干扰机制分析报告

---

## 实验方案 3：可组合性分析（Phase 3）

### 目标
对 Phase 2 确认为正交的方法组合，验证加速比是否近似相乘。

### 可组合性的操作化定义

定义**可组合性分数**：

$$\text{Compose}(Mi, Mj) = \frac{\text{Speedup}(Mi \cup Mj)}{\text{Speedup}(Mi) \times \text{Speedup}(Mj)}$$

- Compose ≈ 1.0: 完全可组合（加速比相乘）
- Compose < 1.0: 存在瓶颈（如内存带宽争用、调度开销）
- Compose > 1.0: 超线性加速（罕见，可能由 cache-friendly 行为导致）

### 实验设计

#### 3.1 加速比测量协议
- **Throughput 测量**: 端到端 tokens/sec（从第一个 token 生成到最后一个 token 完成）
- **Latency 测量**: 首 token 延迟 (TTFT) + 总生成延迟 (TGT)
- **预热**: 丢弃前 5 个样本的结果（GPU 预热 + JIT 编译）
- **测量样本量**: 至少 100 个样本的稳态 throughput
- **控制**: 固定 batch size = 1（单序列推理），后续扩展至 batch size ∈ {1, 4, 8, 16}

#### 3.2 三方法和四方法组合
仅对 Phase 2 中正交性分数 > 0.8 的方法对进行更高阶组合：
- 三方法组合: 最多 C(4,3) = 4 种
- 四方法组合: 1 种（如果三方法组合表现良好）
- 对每种组合，计算链式可组合性分数：
  $$\text{Compose}(Mi, Mj, Mk) = \frac{\text{Speedup}(Mi \cup Mj \cup Mk)}{\text{Speedup}(Mi) \times \text{Speedup}(Mj) \times \text{Speedup}(Mk)}$$

#### 3.3 瓶颈分析
- **Roofline 分析**: 为每种配置绘制 arithmetic intensity vs. throughput，定位是 compute-bound 还是 memory-bound
- **Profile 分解**: 使用 PyTorch Profiler / nsys 分解耗时：attention (%), FFN (%), KV management (%), scheduling overhead (%), draft model (%), verification (%)
- **内存带宽**: 测量 HBM bandwidth utilization，识别带宽瓶颈

### Phase 3 输出
- 可组合性分数表格（所有正交组合）
- 加速比分解图（stacked bar chart: 各方法贡献 vs. 开销）
- 瓶颈 roofline 图
- 最优组合方案及其 speed-accuracy trade-off

---

## 实验方案 4：Ablation Studies

### 4.1 扩散步数敏感性
- 固定最优方法组合，扫描 T ∈ {8, 16, 32, 64, 128}
- 测量 accuracy 和 throughput 随 T 变化的曲线
- 确定每个 benchmark 的最小可接受步数

### 4.2 序列长度敏感性
- 使用不同生成长度的任务子集：短 (<128 tokens)、中 (128-512)、长 (512-1024)
- 评估各方法的加速比是否随序列长度变化
- KV caching 方法预期在长序列上优势更大

### 4.3 模型规模影响（可选扩展）
- 如果资源允许，在 Dream-7B-Instruct 上重复 Phase 1 的关键实验
- 验证结论的跨模型泛化性

### 4.4 Cache 刷新频率消融
- 固定 M1 为最优 KV caching 变体
- 系统扫描 cache refresh interval 对 accuracy 的影响
- 绘制 refresh interval vs. accuracy 的降级曲线

---

## 评估指标

### 主要指标

| 指标 | 定义 | 测量方法 |
|------|------|---------|
| **Throughput** (tokens/sec) | 稳态生成速率 | 总生成 token 数 / 总生成时间（排除预热） |
| **Speedup** | 相对基线的加速倍数 | Throughput(method) / Throughput(baseline) |
| **Accuracy** | 任务正确率 | GSM8K: exact match; MATH: exact match; HumanEval: pass@1 和 pass@10 |
| **Accuracy Retention** | 相对基线的准确率保持率 | Acc(method) / Acc(baseline) × 100% |

### 辅助指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **TTFT** (Time to First Token) | 首 token 延迟 | 衡量交互响应速度 |
| **Peak Memory** (GB) | GPU 显存峰值 | 评估部署可行性 |
| **FLOPs/token** | 每生成 token 的浮点运算量 | 理论效率分析 |
| **GPU Utilization** (%) | SM 活跃率 | 识别计算/内存瓶颈 |
| **Cache Hit Rate** (%) | KV cache 命中率 | 仅 M1 相关 |
| **Effective Steps** | 实际使用的平均扩散步数 | 仅 M2 相关 |
| **Draft Accept Rate** (%) | 草稿被接受的比例 | 仅 M4 相关 |

### 质量-速度 Tradeoff 综合指标

定义 **Quality-Adjusted Speedup (QAS)**:

$$\text{QAS} = \text{Speedup} \times \text{Accuracy Retention}$$

QAS 将速度和质量统一到单一标量，便于方法排名。QAS > 1.0 表示净正收益。

---

## 基线对比

### 内部基线
1. **LLaDA-8B-Instruct Vanilla**: 64 步，无任何加速，作为所有实验的锚点
2. **LLaDA-8B-Instruct 减步**: 简单地减少步数至 32/16/8，作为"朴素加速"基线

### 外部基线（文献复现）
3. **Fast-dLLM**: 复现论文报告的最优配置（block-wise KV cache + confidence parallel decoding）
4. **FlashDLM**: 复现 FreeCache + Guided Diffusion 配置
5. **EntropyCache**: 复现 entropy-guided KV refresh
6. **Saber**: 复现 adaptive acceleration + backtracking

### AR 参考线（非直接对比，仅作速度参考）
7. **LLaMA3-8B-Instruct**: 同等规模 AR 模型的推理速度上界（使用 vLLM 优化）

---

## 实验时间线与资源估算

### Phase 1: 单方法基线（约 3-4 天）
- Baseline + 4 种方法的超参扫描
- 每种方法约 10-15 个超参配置 × 200 样本（GSM8K-mini）× 3 seeds = ~150 次推理
- 估算: 每次推理 5-10 分钟 → 每方法 ~12-25 小时
- **可并行**: 不同方法/seed 可在不同 GPU 上并行

### Phase 2: 正交性验证（约 2-3 天）
- 6 种两两组合 × 3 benchmarks × 3 seeds = 54 次推理
- 使用 Phase 1 最优超参，无需额外调参

### Phase 3: 可组合性分析（约 2-3 天）
- 最多 11 种高阶组合 × 3 benchmarks × 3 seeds = ~99 次推理
- 加 profiling 分析

### Phase 4: Ablation（约 1-2 天）
- 步数/长度/刷新频率消融实验

### 总计: ~8-12 天（单 GPU），~3-5 天（4 GPU 并行）

---

## 预期结论形式

### 结论 1: 单方法效果排名
"在 LLaDA-8B-Instruct 上，KV caching 在保持 >98% accuracy retention 的前提下提供最高的单方法加速比（X倍），adaptive step scheduling 次之（Y倍），而 speculative decoding 在当前 training-free 条件下加速有限（Z倍），主要受限于草稿模型质量。"

### 结论 2: 正交性图谱
"四种方法中，{M_i, M_j} 具有高正交性（Ortho > 0.9），可安全组合；而 {M_k, M_l} 因 [具体干扰机制] 存在显著冲突（Ortho < 0.7），不推荐直接叠加。"

### 结论 3: 可组合性上界
"最优正交组合 {Mi + Mj + ...} 在 GSM8K 上实现 X倍加速（保持 Y% accuracy），逼近理论相乘上界的 Z%。主要瓶颈为 [内存带宽 / 调度开销 / ...]。"

### 结论 4: 任务依赖性
"加速方法的有效性在推理任务和代码任务上表现出 [一致/分化] 的模式。具体而言，[方法X] 在代码生成上效果显著优于推理任务，可能因为 [代码的低 entropy 特性 / 结构化输出的 cache-friendly 特性 / ...]。"

### 结论 5: 实践建议
"对于部署 LLaDA-8B-Instruct 的实际场景，推荐 [最优组合方案] 作为默认加速配置，可在单 A100 上实现 [X] tokens/sec（原始 [Y] tokens/sec），accuracy 损失控制在 [Z]% 以内。"

---

## 潜在风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| M4 (speculative decoding) 在 training-free 条件下效果差 | 组合矩阵不完整 | 降级为 2-3 方法比较，M4 作为"探索性"方法报告 |
| 部分方法对的实现相互冲突（代码层面难以组合） | 实验矩阵缺失 | 提前在小规模数据上验证代码兼容性；不兼容的组合标记为"架构不兼容"并分析原因 |
| 不同 KV caching 变体之间差异过大 | Phase 1 选择困难 | 报告 Pareto 前沿上的 2-3 个代表点，而非单一最优点 |
| 评估方差过大 | 统计结论不可靠 | 增加 seeds 数量（从 3 增至 5）；使用 bootstrap confidence interval |
| GPU 资源不足导致实验超时 | 无法完成全矩阵 | 优先级排序：Phase 1 全量 > Phase 2 两两组合 > Phase 3 高阶组合 > Phase 4 ablation |

---

## 实现路线图

### Step 1: 环境搭建（Day 1）
- 部署 LLaDA-8B-Instruct
- 安装 Fast-dLLM、EntropyCache 等依赖
- 验证 baseline 推理结果与官方一致
- 搭建自动化评测 pipeline（脚本化的 benchmark 评估 + 结果记录）

### Step 2: Phase 1 执行（Day 2-4）
- 运行 baseline
- 依次实现并评估 M1-M4
- 超参选择，锁定最优配置

### Step 3: Phase 2 执行（Day 5-7）
- 实现方法两两组合的代码
- 运行正交性实验
- 分析干扰机制

### Step 4: Phase 3 + Phase 4（Day 8-12）
- 高阶组合实验
- Profiling 与 ablation
- 结果整理与可视化

### Step 5: 论文写作准备
- 核心表格：单方法 Pareto 曲线、正交性热力图、可组合性分解图
- 核心图表：speed-accuracy scatter plot、roofline diagram、ablation curves
