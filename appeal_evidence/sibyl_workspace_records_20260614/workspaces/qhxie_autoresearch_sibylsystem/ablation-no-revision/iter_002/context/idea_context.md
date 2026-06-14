

## Project Spec
# 项目: sae-absorption

## 研究主题
研究稀疏自编码器（SAE）中的特征吸收（feature absorption）现象：系统分析和量化其成因、规律及对可解释性的影响。

## 背景与动机
稀疏自编码器（SAE）是机械可解释性研究的核心工具，用于从语言模型激活中提取人类可解释的特征。然而，SAE 存在"特征吸收"现象：某些低频但语义独立的特征会被高频特征"吸收"，导致 SAE 特征不完整，影响特征质量和下游解释性研究的可靠性。

目前，该现象的普遍程度、成因机制和定量影响尚不清楚，需要系统化的分析和量化。

## 初始想法
- 设计系统化的分析框架，量化特征吸收现象的普遍程度（跨不同模型层、不同 SAE 配置）
- 探索特征吸收的成因：与特征共现频率、稀疏惩罚强度、SAE 字典大小等因素的关系
- 开发可复现的评估指标，衡量吸收现象对下游解释性任务（如电路发现、概念探测）的影响
- 研究方法以 training-free 分析为主，基于现有的预训练 SAE（如 SAELens 库中的 SAE）进行分析

## 关键参考文献
- SAELens 库及其预训练 SAE（GemmaScope, GPT2-small SAE 等）
- Feature absorption 相关文献（待 Sibyl 文献调研补全）

## 可用资源
- GPU: 本地 GPU（有 SSH 访问）
- 服务器: default（SSH MCP 连接）
- 远程路径: /home/qhxie/sibyl_system

## 实验约束
- 实验类型: training-free（分析现有预训练 SAE，不重新训练）
- 模型规模: 小到中等（GPT-2, Gemma-2B 等）
- 时间预算: 单实验 ≤ 1 小时，pilot 10-15 分钟

## 目标产出
- 学术论文（NeurIPS/ICLR 级别）
- 包含：特征吸收的量化分析、成因分析、对可解释性影响的实验

## 特殊需求
- 以 training-free 分析为主，充分利用 SAELens 现有预训练模型
- 论文应包含可复现的评估框架，方便社区后续研究


## 文献调研报告（请仔细阅读，避免重复已有工作）
# Literature Survey Report

**Research Topic**: Ablation-Not-Revision in LLM Reasoning
**Survey Date**: 2026-04-28
**arXiv Search Keywords**: LLM reasoning self-correction, answer consistency early stopping, process reward model, single-pass reasoning, DPO training reasoning
**Web Search Keywords**: LLM self-correction benchmark 2025, reasoning model accuracy MATH GSM8K, DeepSeek-R1 GRPO, Qwen2.5-Math benchmark

## 1. Field Overview

The field of LLM reasoning has evolved from simple chain-of-thought prompting to sophisticated self-correction and sampling strategies. Current research focuses on several key areas:

### 1.1 Chain-of-Thought and Process Supervision
- **Chain-of-Thought (CoT)** (Wei et al., 2022) established that intermediate reasoning steps significantly improve LLM reasoning
- **Step-DPO** (Lai et al., 2024) treats individual reasoning steps as units for preference optimization, achieving state-of-the-art on MATH and GSM8K
- **Process Reward Models (PRM)** provide step-level supervision signals for reasoning

### 1.2 Self-Correction and Error Analysis
- **Self-Refine** (Madaan et al., 2023) demonstrated iterative self-correction without training
- **Accuracy-Correction Paradox** (Li, 2026): stronger models paradoxically show lower intrinsic correction rates due to deeper errors
- **Error Depth Hypothesis**: stronger models make "deeper" errors that resist self-correction

### 1.3 Multi-Sample Reasoning and Early Stopping
- **Self-Consistency** (Wang et al., 2022): majority voting across multiple samples
- **RASC** (Wan et al., 2024): Reasoning-Aware Self-Consistency with early stopping and weighted majority voting
- **CGES** (Aghazadeh et al., 2025): Confidence-Guided Early Stopping with Bayesian framework
- **CoDE-Stop**: Confidence-based early stopping via answer dynamics
- **TRACES** (2026): Real-time step tagging for token reduction
- **ESTAR** (2026): Early-Stopping Token-Aware Reasoning with SFT+RL

### 1.4 Training for Reasoning
- **Step-DPO** (2406.18629): Process-level preference optimization for long-chain reasoning
- **GRPO/RLVR**: Reinforcement learning with verifiable rewards
- **DeepSeek-R1**: Modern reasoning model trained with GRPO

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | Chain-of-Thought Prompting Elicits Reasoning in LLMs | arXiv:2201.11903 | 2022 | Demonstrated intermediate reasoning steps improve LLM reasoning | Requires few-shot examples |
| 2 | Self-Consistency Improves Chain of Thought Reasoning | arXiv:2203.11171 | 2022 | Majority voting across samples improves accuracy | Computational cost scales with sample count |
| 3 | Self-Refine: Iterative Refinement with Self-Feedback | arXiv:2303.17651 | 2023 | Iterative self-correction without training | Limited for strong models |
| 4 | Step-DPO: Step-wise Preference Optimization | arXiv:2406.18629 | 2024 | Process-level DPO for reasoning | Requires step-level preference data |
| 5 | Reasoning-Aware Self-Consistency (RASC) | arXiv:2408.17017 | 2024 | Early stopping + weighted majority voting, ~70% sample reduction | Combined approach may not generalize |
| 6 | CGES: Confidence-Guided Early Stopping | arXiv:2511.02603 | 2025 | Bayesian framework for early stopping | Uses confidence signals, not answer consistency |
| 7 | CoDE-Stop: Early Stopping via Confidence Dynamics | arXiv:2604.04930 | 2025 | Confidence-based early stopping, 25-50% token reduction | Different signal from answer consistency |
| 8 | TRACES: Step Tagging for Early Stopping | arXiv:2604.21057 | 2026 | Real-time step tagging, 20-50% token reduction | Very recent, limited validation |
| 9 | ESTAR: Early-Stopping Token-Aware Reasoning | arXiv:2602.10004 | 2026 | SFT+RL for early stopping, 3.7x length reduction | Training-based approach |
| 10 | LEASH: Logit-Entropy Adaptive Stopping | arXiv:2511.04654 | 2025 | Entropy-based stopping, 30-35% token reduction | Different signal modality |
| 11 | Accuracy-Correction Paradox | arXiv:2601.00828 | 2026 | Error depth hypothesis: stronger models correct less | Limited to GSM8K-Complex |
| 12 | Probabilistic Inference Scaling Theory | arXiv:2508.16456 | 2025 | Exponential saturation in multi-sample accuracy | CRITICAL: Same formula as Activation Energy Theory |
| 13 | Beyond Majority Voting | arXiv:2510.01499 | 2025 | Optimal Weight (OW) and ISP for LLM aggregation | Partial overlap with voting methods |
| 14 | AgentAuditor | arXiv:2602.09341 | 2026 | Reasoning tree auditing | Related work for reasoning verification |
| 15 | IntAttn-Edit | arXiv:2502.XXXXX | 2025 | Joint MLP and attention editing | Still parameter-based |

## 3. SOTA Methods and Benchmarks

### Benchmarks
- **MATH**: High-school mathematics competition problems (SOTA: ~94% with Qwen2-72B + Step-DPO)
- **GSM8K**: Middle-school math word problems
- **GSM8K-Complex**: Extended GSM8K for error analysis
- **Game of 24**: Multi-step reasoning with constraints
- **CountDown/Sudoku**: Search-based reasoning tasks

### State-of-the-Art Methods

| Category | Methods | Performance |
|----------|---------|-------------|
| Training-free | Self-Consistency, RASC, CGES, CoDE-Stop | 5-15% improvement over single sample |
| Training-based | Step-DPO, GRPO, RLVR | 3-5% improvement on MATH |
| Early stopping | TRACES, ESTAR, LEASH | 20-70% token/sample reduction |

### Key Findings from Literature

1. **Exponential Saturation**: Multi-sample accuracy follows exponential convergence (Yang et al., 2025)
   - Formula: `P_k(correct) = P_∞ * (1 - exp(-k/k_0))`
   - Validated across 8 models and 8 datasets

2. **Accuracy-Correction Paradox**: Stronger models self-correct less (Li, 2026)
   - Deeper errors resist correction
   - Weaker models have shallower, more fixable errors

3. **Early Stopping Effectiveness**: Answer consistency signals enable efficient early stopping
   - RASC: ~70% sample reduction
   - CGES: ~69% sample reduction
   - TRACES: 20-50% token reduction

## 4. Identified Research Gaps

### 4.1 Answer Consistency vs Confidence Signals
- Most early stopping methods (CGES, CoDE-Stop) use **confidence/probability** signals
- Limited research on **answer persistence** (when answer first stabilizes across steps)
- **Gap**: Can answer consistency signals outperform confidence-based methods?

### 4.2 Activation Energy Theory
- Yang et al. (2025) provides exponential saturation formula
- **Gap**: Physical analogy framing (chemical kinetics) may provide novel theoretical lens
- **CRITICAL**: Cannot claim novel mathematical framework - formula already published

### 4.3 Single-Pass Reasoning
- Most improvements come from iterative refinement
- **Gap**: Training models to reason correctly in one pass could reduce inference costs
- **Problem**: Small models (7B) achieve only 10-26% on MATH - too weak for meaningful experiments

### 4.4 Model Capability Thresholds
- DeepSeek-Math-7B: 10-26% accuracy on MATH (too weak)
- Qwen2.5-Math-7B: Expected >50% accuracy (needs validation)
- **Gap**: What capability threshold is needed for meaningful self-correction research?

### 4.5 Training vs Stability
- Qin et al. (2025) suggests DPO improves stability, not reasoning
- **Gap**: Systematic analysis of what DPO actually improves
- **Implication**: Training may increase pre-exponential factor A, not reduce problem difficulty

## 5. Prior Art Collision Analysis

### CRITICAL COLLISION: Yang et al. (2025) arXiv:2508.16456

**Paper**: "A Probabilistic Inference Scaling Theory for LLM Self-Correction"

| Our Proposal | Yang et al. |
|--------------|-------------|
| `P_k(correct) = P_∞ * (1 - exp(-k/k_0))` | `Acc_t = Upp - α^t(Upp - Acc_0)` |
| P_∞ = asymptotic accuracy ceiling | Upp = upper bound |
| k_0 = characteristic sampling count | α = convergence rate |
| Ea = P_∞ * k_0 (activation energy) | Related to CL and CS |

**Mathematical Equivalence**:
- `1 - exp(-k/k_0) = 1 - (exp(-1/k_0))^k`
- When α = exp(-1/k_0), forms are equivalent
- **Conclusion**: Core formula already published

### RASC (2408.17017) - Most Similar Work
- Combines early stopping with weighted majority voting
- Uses "reasoning quality" + answer consistency
- **Difference**: We propose answer persistence signal + Borda count
- **Overlap**: Cannot claim "first combination" of early stopping + voting

### CoDE-Stop (2604.04930)
- Uses **confidence** from token probabilities
- **DIFFERENT SIGNAL** from answer consistency

### CGES (2511.02603)
- Bayesian framework for posteriors over answers
- Uses confidence-weighted aggregation
- **DIFFERENT SIGNAL** from answer persistence

## 6. Available Resources

### Open-source Code
| Repo | URL | Description |
|------|-----|-------------|
| Self-Refine | https://github.com/madaan/self-refine | Iterative refinement framework |
| Step-DPO | https://github.com/dvlab-research/Step-DPO | Step-wise preference optimization |
| RASC | Available on arXiv | Reasoning-aware self-consistency |
| DeepSeek | HuggingFace | DeepSeek-R1 models |

### Datasets
| Dataset | Source | Notes |
|---------|--------|-------|
| MATH | Competition datasets | 5 difficulty levels |
| GSM8K | Open-source | ~8,500 problems |
| GSM8K-Complex | Extended GSM8K | For error analysis |

### Pretrained Models
| Model | Size | Expected MATH Accuracy |
|-------|------|----------------------|
| DeepSeek-Math-7B | 7B | 10-26% (too weak) |
| Qwen2.5-Math-7B | 7B | Expected >50% (needs validation) |
| Qwen2.5-Math-72B | 72B | Expected >80% |
| DeepSeek-R1-Distill | 7B-32B | GRPO-trained reasoning |

## 7. Implications for Research Direction

### Directions Worth Exploring
1. **Model Capability Gating**: Need >50% baseline model for meaningful experiments
2. **Answer Persistence Signal**: Different from confidence-based early stopping
3. **Borda Count Aggregation**: Less explored than weighted majority voting
4. **Activation Energy Physical Framing**: Pedagogical value, not novel math

### Directions to Avoid
1. Claiming novel mathematical framework (Yang et al. collision)
2. Claiming "first combination" of early stopping + voting (RASC collision)
3. Using weak baseline models (DeepSeek 7B at 10-26%)

### Saturated Directions
- Generic iterative self-refinement
- Generic self-consistency without early stopping
- Confidence-based early stopping (extensive exploration)

## 8. Implementation Strategy

| Implementation | Match | License | Strategy | Rationale |
|---------------|-------|---------|----------|-----------|
| Step-DPO | High | MIT | Adopt | SOTA for reasoning fine-tuning, directly applicable |
| RASC | Medium | - | Extend | Early stopping + voting framework, adapt signal type |
| Self-Refine | Medium | Apache 2.0 | Reference | Base iterative refinement |
| CGES | Medium | - | Reference | Bayesian early stopping framework |

### Key Insights

1. **Model Selection Critical**: Need >50% baseline for meaningful experiments
2. **Yang et al. Collision**: Cannot claim novel mathematical framework
3. **Signal Type Matters**: Answer persistence vs confidence vs entropy
4. **Training May Not Reduce Difficulty**: DPO improves stability, not reasoning ability
5. **Hardware Available**: PyTorch 2.11.0 + Blackwell GPU confirmed working

## 9. Updated Project Status

### Current Round: Round 4 Pivot

| Aspect | Status | Notes |
|--------|--------|-------|
| Hardware | RESOLVED | PyTorch 2.11.0 supports RTX PRO 6000 Blackwell |
| API Key | BLOCKED | No DeepSeek/OpenAI API available |
| Baseline Model | NEEDS VALIDATION | Qwen2.5-Math-7B expected >50% |
| Activation Energy Theory | COLLISION | Yang et al. (2508.16456) same formula |
| Self-Consistency | EXPLORED | RASC, CGES, CoDE-Stop all exist |

### Recommended Next Steps

1. **Validate Qwen2.5-Math-7B** baseline on MATH (must achieve >50%)
2. If Qwen2.5 succeeds: Proceed with Step-DPO training or self-consistency experiments
3. If Qwen2.5 fails: Consider API-based evaluation or stronger local model
4. **Do not claim**: Novel exponential saturation formula, first combination of early stopping + voting
5. **Can claim**: Novel answer persistence signal, Borda aggregation, Activation Energy physical framing

## 10. References

### Key Papers

1. Yang et al. (2025). A Probabilistic Inference Scaling Theory for LLM Self-Correction. arXiv:2508.16456
2. Li (2026). Decomposing LLM Self-Correction: The Accuracy-Correction Paradox. arXiv:2601.00828
3. Wan et al. (2024). Reasoning-Aware Self-Consistency. arXiv:2408.17017
4. Lai et al. (2024). Step-DPO: Step-wise Preference Optimization. arXiv:2406.18629
5. Wang et al. (2022). Self-Consistency Improves CoT Reasoning. arXiv:2203.11171
6. Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning. arXiv:2201.11903
7. Madaan et al. (2023). Self-Refine: Iterative Refinement. arXiv:2303.17651
8. Aghazadeh et al. (2025). CGES: Confidence-Guided Early Stopping. arXiv:2511.02603
9. Qin et al. (2025). To Backtrack or Not to Backtrack. arXiv:2502.XXXXX

### Recent Papers (2026)

10. TRACES (2026). Step Tagging for Early Stopping. arXiv:2604.21057
11. ESTAR (2026). Early-Stopping Token-Aware Reasoning. arXiv:2602.10004
12. CoDE-Stop (2025). Confidence Dynamics for Early Stopping. arXiv:2604.04930
13. LEASH (2025). Logit-Entropy Adaptive Stopping. arXiv:2511.04654
14. AgentAuditor (2026). Reasoning Tree Auditing. arXiv:2602.09341
15. Beyond Majority Voting (2025). arXiv:2510.01499
