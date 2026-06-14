# Comparativist Analysis: DTA 实验结果的文献定位

**分析时间**: 2026-03-09
**分析范围**: Countdown-500 (4/7 methods complete), GSM8K (in progress), pilot 全方法对比

---

## 1. SOTA 对比：当前结果在已有工作中的位置

### 1.1 Countdown Benchmark

我们的 Dream-7B vanilla baseline 在 Countdown-500 上的准确率为 **4.7%**（3 seeds 平均）。这与 Dream 原论文报告的 Countdown 16.0%（8-shot）存在显著差距。差距可能来源于：
- 我们使用的是 500 个自生成 Countdown 问题（而非 Dream 论文的原始测试集）
- 评估设置差异（gen_len, temperature, shot 数量）
- Countdown 任务难度随 target 数值和操作数变化很大

**竞争方法在 Countdown 上的已知结果**：

| 方法 | 模型 | Countdown 准确率 | 来源 |
|------|------|-----------------|------|
| Dream vanilla (8-shot) | Dream-7B | 16.0% | Dream 原论文 |
| AGRPO | LLaDA-8B | +59.4%（相对提升） | arXiv 2510.04019 |
| MDPO | Dream-7B | +54.2%（相对提升） | arXiv 2508.13148 |
| d1 (diffu-GRPO) | LLaDA-8B | 显著提升 | arXiv 2504.12216 |
| Loopholing (LDDM) | 从头训练 | "改善" | arXiv 2510.19304 |
| LookUM | LLaDA-8B | +4pp (GSM8K) | arXiv 2511.05563 |
| **Ours: DMI** | **Dream-7B** | **9.3%** | **本实验** |
| **Ours: RCR** | **Dream-7B** | **5.7%** | **本实验** |
| **Ours: Vanilla** | **Dream-7B** | **4.7%** | **本实验** |
| **Ours: ReMDM-conf** | **Dream-7B** | **4.4%** | **本实验** |

**关键问题**: AGRPO 和 MDPO 在 Countdown 上报告了 50%+ 的相对提升，但它们是 **RL 后训练方法**（需要额外训练），与我们的 **training-free** 方法定位不同。在 training-free 方法中，我们的 DMI（9.3%）相对于 vanilla（4.7%）提供了约 **2x 改善**，但绝对数值仍然很低。

### 1.2 GSM8K Benchmark

我们的 Dream-7B vanilla 在 GSM8K 上约 **29.6%**（1300/1319 样本进度），这与以下已知结果对比：

| 方法 | 模型 | GSM8K 准确率 | 类型 |
|------|------|-------------|------|
| LLaDA-8B-Instruct | LLaDA | ~67% (SFT) | 训练后 |
| wd1 | LLaDA | 84.5% | RL 训练 |
| DCoLT | LLaDA | +9.8pp | RL 训练 |
| MetaState | Dream-7B | +1.2pp (vs Instruct baseline) | 需训练 |
| MetaState | LLaDA-8B | +1.5pp (vs Base) | 需训练 |
| CORE | LLaDA-8B-Base | 不详 (GSM8K) | Training-free |
| LookUM | LLaDA-8B | +4pp | Training-free |
| ReMDM | MDLM-170M | 未测 | Training-free |
| **Ours: Vanilla** | **Dream-7B** | **~29.6%** | **Baseline** |
| **Ours: ReMDM-conf** | **Dream-7B** | **~25.1% (350/1319)** | **Training-free** |

Dream-7B 在 GSM8K 上 ~30% 的表现合理（Dream 并非为数学推理优化，且我们使用的是 vanilla origin 模式而非 instruct）。

### 1.3 MBPP / Code Benchmarks

Pilot 阶段的 Dream-7B vanilla Pass@1 = 25.0%（16 样本），仅供参考。

| 方法 | 模型 | MBPP | 来源 |
|------|------|------|------|
| CORE | LLaDA-8B-Base | +9.2pp | arXiv 2602.04096 |
| DCoLT | LLaDA-8B | +11.4pp | arXiv 2505.10446 |
| LookUM | LLaDA-8B | +8pp | arXiv 2511.05563 |
| Soft-Masked | Dream-7B | 一致改善 | arXiv 2510.17206 |

---

## 2. 贡献边际（Contribution Margin）评估

### 2.1 DMI 的实际贡献

DMI（Diffusion Memory Injection）是目前我们最成功的方法：
- Countdown: **9.3% vs 4.7% vanilla**（+4.6pp, ~2x 改善）
- 计算开销极小（~1.05x）
- 跨 3 seeds 一致

**坦率评估**: +4.6pp 的绝对改善在统计上可能显著（需 McNemar 检验确认），但 **贡献边际较小**。与 MetaState（arXiv 2603.01331）相比，DMI 的核心思想（利用前步 logits 的软信息注入当前步）在概念上更简单，但 MetaState 使用了可训练的 GRU+CrossAttn 模块且需要 K-step unrolling 训练。DMI 的 **training-free** 特性是其关键优势。

然而，DMI 的原理——将上一步 logits 的 softmax 加权 embedding 注入当前步——可能被视为 Soft-Masked Diffusion（arXiv 2510.17206, NeurIPS 2025）的简化版本。Soft-Masked 也是"动态混合 mask embedding 与 top-k 预测 embedding"，思路高度相似。这对 DMI 的**新颖性构成直接威胁**。

### 2.2 ReMDM-conf 和 RCR 的结果

- **ReMDM-conf**: 4.4%（略低于 vanilla 4.7%），这与 ReMDM 原论文（主要在 MDLM-170M 的无条件文本生成上验证）的发现一致——remasking 在推理任务上的效果有限
- **RCR**: 5.7%（略高于 vanilla），但方差较大（std=0.006 与 vanilla 相同），改善可能不显著

这些结果**支持了 proposal 的核心论点**：纯 remasking 方法在推理任务上效果有限，需要跨步信息传递。

### 2.3 DTA 的状态（待完成）

DTA 是论文的核心方法，但目前 full-scale 结果尚未出。根据 pilot 数据：
- Countdown 16 样本: DTA 6.2% vs vanilla 12.5%（负面）
- LLaDA-8B GSM8K 16 样本: DTA 18.8% vs vanilla 43.8%（显著负面）
- DTA+ReMDM GSM8K (LLaDA): 31.2%（部分恢复但仍低于 vanilla）

**这是高度危险的信号**。如果 full-scale 结果确认 DTA 无法超越 vanilla，则论文的核心贡献（参数级推理时适应）将无法成立。

---

## 3. 并发工作检查（Concurrent Work）

### 3.1 直接竞争者

| 工作 | 发表时间 | 与我们的重叠 | 威胁等级 |
|------|---------|-------------|---------|
| **MetaState** (arXiv 2603.01331) | 2026-03-02 | 直接解决相同问题（Information Island），但用可训练模块 | **高** |
| **Soft-Masked DLMs** (arXiv 2510.17206, NeurIPS 2025) | 2025-10 | 与 DMI 核心思想高度重叠（前步预测信息传递） | **高** |
| **CORE** (arXiv 2602.04096) | 2026-02 | Training-free 推理时修正，定位相似 | **中** |
| **Loopholing** (arXiv 2510.19304) | 2025-10 | 解决 sampling wall / 信息丢失，但需从头训练 | **中** |
| **LookUM** (arXiv 2511.05563) | 2025-11 | Training-free 推理时改善，但方法完全不同（路径选择 vs 参数适应） | **低** |
| **EntRGi** (arXiv 2602.05000) | 2026-02 | 奖励引导的 test-time adaptation，但需奖励模型 | **低** |

### 3.2 最严重的竞争：MetaState

MetaState（2026-03-02，仅比我们早 1 周）直接使用了"Information Island"这一术语，且提出了更完整的解决方案：
- **训练式跨步记忆**：GRU Updater + CrossAttn Mixer/Injector
- **在 LLaDA-8B 和 Dream-7B 上验证**
- GSM8K: +1.2pp (Dream), +1.5pp (LLaDA)
- **骨干冻结**，仅训练记忆模块

**与 DTA 的关键区别**：
- MetaState 需要 K-step unrolling 训练，DTA 是 training-free
- MetaState 使用固定大小的外部记忆，DTA 更新模型参数
- MetaState 的改善幅度较小（+1-2pp），但更稳定

**如果 DTA 在 full-scale 上有效（>+3pp）**，我们可以定位为"MetaState 的 training-free 替代方案"，且改善幅度更大。**如果 DTA 无效**，这个定位不成立。

### 3.3 Soft-Masked DLMs vs DMI

Soft-Masked（NeurIPS 2025 已接收）的核心是：
> "动态混合 mask embedding 与 top-k 预测 embedding，保留前步计算信息"

DMI 的核心是：
> "上一步 logits 的 softmax 加权 embedding 注入当前步输入"

**两者在概念层面几乎等价**。区别在于：
- Soft-Masked 需要 continued pretraining（修改训练）
- DMI 是 training-free（直接在推理时注入）
- Soft-Masked 在代码基准上验证，DMI 在 Countdown 上验证

如果我们声称 DMI 是新贡献，**必须明确引用 Soft-Masked 并说明 training-free 的差异化**。

---

## 4. 新颖性评估

### 4.1 DTA 的新颖性

**如果有效（>+3pp）**：
- **高度新颖**：首个在 DLM 去噪过程中执行 training-free LoRA 参数更新的方法
- 独特定位：training-free + 参数级记忆 + 理论保证（ELBO 单调性）
- 与 MetaState 互补：MetaState=训练式外部记忆，DTA=推理时参数适应

**如果无效**：
- 新颖性仍在（方法本身是新的），但**实验贡献为零**
- 需要转向理论/分析论文：解释为什么参数级适应在 DLM 上不起作用

### 4.2 DMI 的新颖性

- **中等**：核心思想与 Soft-Masked DLMs 重叠
- **差异化点**: training-free（无需重新训练模型），且我们量化了效果（+4.6pp on Countdown）
- **风险**: 审稿人可能认为 DMI 是 Soft-Masked 的 trivial training-free 版本

### 4.3 整体论文的新颖性

论文的最强贡献不在单个方法，而在**信息增强谱系**（DMI → SCP → DTA）的系统性消融：
- 这种渐进式分析框架本身有价值
- 但如果 DTA（顶层方法）不工作，谱系的 climax 缺失

---

## 5. 发表可行性评估

### 5.1 最佳情况（DTA + DTA+ReMDM 在 full-scale 上显著有效）

- **DTA Countdown > 10%**（>+5pp over vanilla）: 可投 **NeurIPS/ICML 主会**
- 结合变分理论框架 + 信息增强谱系 + 跨模型验证
- 需要 GSM8K 和 MBPP 上也有正面结果

### 5.2 中等情况（仅 DMI 有效，DTA 效果平平）

- DMI +4.6pp + 理论分析 + 负面结果的诊断: 可投 **EMNLP/ICLR Workshop**
- 论文角度转为: "Why Cross-Step Information Matters (and Gradient Updates Don't Help) in Masked Diffusion"
- 与 MetaState 做详细对比（training-free vs training-based）

### 5.3 降级情况（DTA 无效，仅 DMI 有小幅改善）

- 分析论文: 结合 18 轮迭代的负面结果 + DMI/SCP/DTA 的系统消融
- 可投 **ACL Findings / EMNLP**
- 核心价值在于诊断性分析而非方法贡献

### 5.4 当前数据支持的投稿等级

以目前已完成的数据（4/7 methods on Countdown, GSM8K in progress），**尚不足以支持任何投稿**。必须等待：
1. DTA 和 DTA+ReMDM 的 full-scale Countdown 结果
2. GSM8K 全方法 x 3 seeds 结果
3. MBPP 结果
4. 统计检验（McNemar + Bootstrap CI）

---

## 6. 需要补充的基线和对比

### 6.1 缺失的关键基线

| 基线 | 重要性 | 原因 |
|------|--------|------|
| **Soft-Masked DLM** | **必须** | 与 DMI 直接竞争，需在同一设置下对比 |
| **CORE** | 高 | Training-free SOTA，且已在 LLaDA-8B 上验证 |
| **Best-of-N (vanilla)** | 高 | 最简单的推理时扩展基线，用于标定 DTA 的 compute-matched 效果 |
| **LookUM** (2-3 paths) | 中 | Training-free，compute-matched 对比 |
| **Prism** (HTS + SVF) | 中 | Training-free 搜索方法 |

### 6.2 缺失的分析

1. **Compute-matched 对比**: DTA 成本 ~2.5x vanilla，应对比 Best-of-N (2-3 samples) 在相同计算预算下的效果
2. **Token 级诊断**: Correction Precision/Recall（proposal 中承诺的诊断框架）尚未实现
3. **推理时扩展曲线**: Pilot 的 T={64,128,256,512} 扫描因样本量太小无法得出结论，需要 full-scale 验证
4. **LLaDA-8B full-scale**: Pilot 显示 DTA 在 LLaDA 上也无效，但需要更大样本量确认

---

## 7. 总结与建议

### 7.1 当前结果的诚实定位

**正面**:
- DMI（~2x 改善，training-free，几乎零额外开销）是一个**有实用价值的贡献**
- 系统性消融证明跨步信息传递比纯 remasking 更有效
- ReMDM-conf 和 RCR 在 Countdown 上的负面结果支持 Information Island 假说

**负面**:
- DTA（核心方法）的 pilot 信号持续为负
- DMI 与 Soft-Masked DLMs (NeurIPS 2025) 在概念上高度重叠
- 绝对准确率很低（最好的 DMI 也仅 9.3%），限制了实际影响力
- MetaState 几乎同时解决相同问题，且有更完整的方案

### 7.2 风险缓解建议

1. **最优先**: 等待 DTA full-scale 结果。如果 DTA 在 500 样本上显著优于 vanilla（p<0.05），一切论点成立
2. **对冲**: 立即实现 Soft-Masked DLM 作为对比基线，确保 DMI 的差异化可被量化
3. **Compute-matched**: 实现 Best-of-N baseline，证明 DTA 的改善不只是"用更多计算换来的"
4. **转向准备**: 如果 DTA 失败，准备"Why Parameter Adaptation Fails for DLM Inference"的分析论文框架，这本身有学术价值（理解 DLM 的结构性限制）

### 7.3 对 DTA 前景的坦率判断

基于所有 pilot 数据（Dream Countdown: 6.2% vs 12.5%, LLaDA Countdown: 0% vs 12.5%, LLaDA GSM8K: 18.8% vs 43.8%），**DTA 在 full-scale 上超越 vanilla 的概率约为 25-30%**。主要风险来源：
- Masked re-prediction 的自监督信号可能根本不包含"推理正确性"的信息
- LoRA 更新可能在 128 步的短去噪过程中无法学到足够的模式
- 参数更新可能干扰预训练权重的精细结构

**DMI 更可能成为论文的核心贡献**，建议在论文叙事中将 DTA 定位为"理论上更优雅但实践中受限"的方法，而 DMI 定位为"简单有效的 training-free 跨步信息传递机制"。

---

*本分析基于 arXiv 搜索（2603.01331 MetaState, 2510.17206 Soft-Masked, 2602.04096 CORE, 2511.05563 LookUM, 2510.19304 Loopholing, 2602.05000 EntRGi）、Google Scholar 搜索（AGRPO, MDPO, Dream benchmark results）和 Web 搜索（Countdown SOTA, inference-time scaling leaderboards）综合整理。*
