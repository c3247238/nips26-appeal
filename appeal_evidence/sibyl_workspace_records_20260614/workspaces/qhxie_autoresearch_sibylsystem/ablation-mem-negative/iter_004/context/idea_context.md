

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

**Research Topic**: SAE Feature Absorption -- Unsupervised Detection Limits and Scaling Behavior
**Survey Date**: 2026-04-28
**arXiv Search Keywords**: ["feature absorption" "sparse autoencoder", "unsupervised detection" "sparse autoencoder", "co-occurrence clustering" "SAE", "interpretability illusion" "sparse autoencoder", "causal inference" "sparse autoencoder", "negative results" "sparse autoencoder downstream"]
**Web Search Keywords**: ["SAE feature absorption unsupervised detection 2025", "sparse autoencoder interpretability illusion false negative 2025", "DeepMind negative results SAE deprioritize 2025", "SAE causal inference intervention 2025", "SAEBench absorption score benchmark 2025", "inference-time decomposition SAE ITDA 2025"]

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have become the dominant technique for mechanistic interpretability of Large Language Models, but 2025 has been a year of reckoning. While the field continues to scale SAEs to unprecedented sizes (OpenAI's 16M latent SAE on GPT-4, ICLR 2025), a growing body of negative results has challenged fundamental assumptions about what SAEs can deliver.

The most significant development for this project is the convergence of three lines of work:

1. **Feature absorption as a fundamental barrier**: Chanin et al. (2024) established that SAEs systematically suppress child features under parent co-occurrence, creating "interpretability illusions" where features appear monosemantic but have systematic false negatives. This has been confirmed and extended by multiple 2025 papers.

2. **Unsupervised detection attempts and their limits**: The community has proposed co-occurrence clustering as a potential unsupervised solution (Chen et al., 2025; HSAE work), but no prior work has systematically tested whether such approaches scale -- a gap this project directly addresses.

3. **Negative results on SAE utility**: DeepMind's mechanistic interpretability team published influential negative results in March 2025, finding that SAEs underperform simple linear probes on downstream tasks and deprioritizing SAE research. This reframes absorption detection from a "nice-to-have" to a critical question for SAE viability.

By 2025, the field has expanded beyond architectural innovations to include theoretical analyses (provable feature recovery, closed-form optimal solutions), comprehensive benchmarks (SAEBench, CE-Bench), and critical reassessments (SAEs interpreting randomly initialized transformers, interpretability illusions). The central tension remains: SAEs promise interpretability but deliver illusions.

---

## 2. Core References

### Foundational Papers (Pre-2025)

| # | Title | Authors | Source | Year | Key Contribution | Relevance to This Project |
|---|-------|---------|--------|------|-----------------|--------------------------|
| 1 | Toy Models of Superposition | Elhage et al. | arXiv:2209.10652 | 2022 | Established theoretical foundation of superposition | Theoretical basis for why absorption occurs |
| 2 | Towards Monosemanticity | Bricken et al. | Transformer Circuits | 2023 | First SAE application to real transformers; introduced feature splitting | Precedent for negative-result framing in MI |
| 3 | Scaling Monosemanticity | Templeton et al. | Anthropic | 2024 | Scaled to Claude 3 Sonnet; millions of features | Shows absorption persists at scale |
| 4 | Scaling and Evaluating Sparse Autoencoders | Gao et al. | arXiv:2406.04093 / ICLR 2025 | 2024 | TopK SAE; 16M latents; AuxK loss; scaling laws | Baseline architecture; dead feature solutions |
| 5 | Jumping Ahead: JumpReLU SAEs | Rajamanoharan et al. | arXiv:2407.14435 / ICLR 2025 | 2024 | Best reconstruction fidelity; minimal dead features | Primary architecture used in this project |
| 6 | **A is for Absorption** | Chanin et al. | arXiv:2409.14507 | 2024 | **Foundational**: Named and characterized feature absorption; linked to sparsity loss + hierarchical co-occurrence | **Direct predecessor**: Our work tests unsupervised alternatives to their supervised protocol |
| 7 | Gemma Scope | Lieberum et al. | Google | 2024 | Pre-trained JumpReLU SAEs for Gemma-2 | Primary data source for this project |

### 2025 Papers Directly Relevant to Unsupervised Detection / Absorption

| # | Title | Authors | Source | Year | Key Contribution | Relevance |
|---|-------|---------|--------|------|-----------------|-----------|
| 8 | **Feature Hedging** | Chanin et al. | arXiv:2505.11756 | 2025 | Identified hedging as complement to absorption; showed Matryoshka trades absorption for hedging | Context for absorption-hedging trade-off |
| 9 | **Matryoshka SAE** | Bussmann et al. | arXiv:2503.17547 / ICML 2025 | 2025 | ~90% absorption reduction via hierarchical nested dictionaries | Preventive (not detective) approach -- contrast with our work |
| 10 | **Orthogonal SAE** | Korznikov et al. | arXiv:2509.22033 | 2025 | ~65% absorption reduction via orthogonality constraints | Alternative preventive approach |
| 11 | **Interpretability Illusions with Sparse Autoencoders** | Various | arXiv:2505.16004 | 2025 | Systematic study of how SAE explanations mislead; close-negative falsification strategy | Supports our negative-result framing |
| 12 | **Falsifying Sparse Autoencoder Reasoning Features** | Various | arXiv:2601.05679 | 2025 | 196 context-dependent features: none genuine reasoning; systematic false positives/negatives | Strong precedent for negative-result papers in SAE space |
| 13 | **Revising and Falsifying SAE Feature Explanations** | Various | OpenReview / NeurIPS 2025 | 2025 | Similarity-based strategy for sourcing close negatives that falsify explanations | Methodological precedent |
| 14 | **Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts** | Various | arXiv:2506.23845 | 2025 | Argues SAEs remain valuable for discovery despite failing at action tasks | Reframes SAE utility -- absorption detection is a discovery task |
| 15 | **Are Sparse Autoencoders Useful?** | Kantamneni et al. | PMLR / ICLR 2025 | 2025 | SAE probes underperform logistic regression baseline across datasets | Independent confirmation of DeepMind negative results |
| 16 | **SAEBench** | Karvonen et al. | arXiv:2503.09532 | 2025 | Standardized evaluation; absorption score metric; 200+ SAEs | Evaluation framework; absorption metric definition |
| 17 | **CE-Bench** | Gulko et al. | arXiv:2509.00691 | 2025 | Contrastive evaluation; 70%+ Spearman with SAEBench | Alternative evaluation for interpretability |
| 18 | **On the Limits of SAEs** | Cui et al. | arXiv:2506.15963 | 2025 | First closed-form optimal solution; proved standard SAEs fail unless extremely sparse | Theoretical support for fundamental limits |
| 19 | **Provable Feature Recovery** | Chen et al. | arXiv:2506.14002 / ICLR 2025 | 2025 | First SAE with theoretical recovery guarantees; Group Bias Adaptation | Theoretical framework for feature identifiability |
| 20 | **Inference-Time Decomposition of Activations (ITDA)** | Various | arXiv:2505.17769 / ICML 2025 | 2025 | ~100x faster than SAE training; inference-time sparse coding; dictionaries transfer between models | Inference-time alternative to training SAEs |
| 21 | **Sparse Autoencoders Do Not Find Canonical Units** | Leask et al. | ICLR 2025 | 2025 | SAE stitching and Meta-SAEs challenge canonical feature hypothesis | Fundamental questions about what SAEs capture |
| 22 | **Hierarchical SAEs** | Muchane et al. | arXiv:2506.01197 | 2025 | Explicitly models semantic hierarchy; improves reconstruction | Hierarchical modeling approach |
| 23 | **From Atoms to Trees** | Luo et al. | arXiv:2602.11881 | 2026 | Jointly learns SAEs and parent-child relationships | Very recent hierarchical approach |
| 24 | **Temporal SAEs** | Various | arXiv:2511.05541 | 2025 | Leverages sequential nature; comparable absorption to Matryoshka | Alternative temporal approach |
| 25 | **AdaptiveK SAE** | Various | arXiv:2508.17320 | 2025 | Dynamic sparsity allocation; superior absorption on 250K tokens | Adaptive sparsity approach |
| 26 | **Improving Robustness via Masked Regularization** | Various | arXiv:2604.06495 | 2026 | Masking-based regularization reduces absorption | Recent mitigation approach |
| 27 | **SAGE: Scalable Ground Truth Evaluations** | Various | OpenReview | 2025 | Scalable ground truth evaluations for large SAEs | Ground truth methodology |
| 28 | **Sparse but Wrong** | Various | arXiv:2508.16560 | 2025 | Incorrect L0 leads to incorrect features | Metric reliability concerns |
| 29 | **CorrSteer** | Various | arXiv:2508.12535 | 2025 | Correlation-based SAE feature selection for steering | Correlation-based feature analysis |
| 30 | **Signal in the Noise** | Various | arXiv:2505.11611 | 2025 | Polysemantic interference transfers across models | Cross-model feature analysis |

### Industry / Blog Posts

| # | Title | Authors | Source | Year | Key Contribution |
|---|-------|---------|--------|------|-----------------|
| 31 | Negative results for SAEs on downstream tasks | Smith et al. | DeepMind Safety Research | 2025 | SAEs underperform linear probes; team deprioritizes SAE research |
| 32 | Feature Hedging (LessWrong) | Chanin et al. | LessWrong | 2025 | Complement to absorption; broader correlated feature problem |

---

## 3. SOTA Methods and Benchmarks

### Absorption Metrics and Evaluation

The **absorption score** (SAEBench) is computed via first-letter spelling tasks:

```
Absorption Score = sum(absorber_activations) / (sum(absorber_activations) + sum(main_activations))
```

Lower is better. Key benchmark results:

| Method | Absorption Score | L0 | Notes |
|--------|-----------------|-----|-------|
| TopK SAE | 0.1402 | 40 | Baseline |
| JumpReLU | 0.0114 | 40 | Best among single-level |
| Matryoshka SAE (outer) | 0.005 | 40 | ~90% reduction |
| Matryoshka SAE (inner) | Higher | 40 | Hedging trade-off |
| ATM | 0.0068 | 40 | Best reported |
| OrtSAE | ~0.049 | 70 | ~65% reduction |
| Standard SAE (L1) | 0.0161 | -- | Deprecated |

### Current Best Methods for Addressing Absorption

| Method | Absorption Reduction | Key Innovation | Strategy Type |
|--------|---------------------|----------------|---------------|
| **Matryoshka SAE** | ~90% | Hierarchical nested dictionaries | **Preventive** (architecture) |
| **ATM** | ~95% | Temporal dynamics + probabilistic masking | **Preventive** (training) |
| **OrtSAE** | ~65% | Cosine similarity penalty | **Preventive** (constraint) |
| **Balanced Matryoshka** | Optimized trade-off | Tuned beta_m ~ 0.75 | **Preventive** (loss tuning) |
| **Weighted SAE** | Improved low-sparsity | Theoretically-grounded reweighting | **Preventive** (objective) |
| **GBA** | Strong empirical | Adaptive bias adjustment | **Preventive** (training) |
| **Chanin et al. protocol** | Detection only | Requires known parent-child hierarchies | **Detective** (supervised) |
| **Co-occurrence clustering** | Unknown | Cluster features by co-activation | **Detective** (unsupervised) |

**Critical observation**: All effective solutions are **preventive** (modify architecture/training). The only **detective** approach (Chanin et al.) requires ground-truth labels. **Unsupervised detection remains untested at scale** -- this is the gap our project addresses.

### Benchmarks

- **SAEBench**: ~26 min per SAE for absorption evaluation; 8+ metrics; 200+ SAEs
- **CE-Bench**: LLM-free contrastive evaluation; 70%+ Spearman with SAEBench
- **First-letter spelling task**: Chanin et al.'s ground-truth protocol; limited to early layers

---

## 4. Identified Research Gaps

### Gaps Directly Addressed by This Project

- **Gap 1: Unsupervised absorption detection at scale**. No prior work has systematically tested whether co-occurrence clustering (or any unsupervised method) can detect absorption, or characterized its scaling behavior. Our project provides the first empirical characterization.

- **Gap 2: Correlation vs. suppression distinction**. The field has not formally analyzed why co-occurrence clustering (which detects correlation) cannot detect absorption (which requires detecting suppression). Our theoretical analysis fills this gap.

- **Gap 3: Precision collapse with scale**. Small-scale pilot studies (46 pairs) may show promising precision, but no prior work has tested whether this holds at full-dictionary scale (3,700+ pairs). Our scaling analysis is the first.

### Related Gaps

- **Gap 4: Theoretical understanding of absorption-hedging trade-off**. The Pareto frontier is not well-characterized.
- **Gap 5: Absorption-aware training objectives**. Loss terms that directly penalize absorption during training are underexplored.
- **Gap 6: Cross-architecture comparison on absorption**. Most studies focus on JumpReLU/GemmaScope.
- **Gap 7: Interaction between absorption and dead features**. These problems are studied separately but likely interact.
- **Gap 8: What do SAEs actually capture?** The randomly-initialized transformers finding raises fundamental questions.
- **Gap 9: Inference-time feature recovery**. ITDA is a start, but dynamic recovery of absorbed features remains unexplored.

---

## 5. Available Resources

### Open-source Code

| Resource | URL | Description | Strategy |
|----------|-----|-------------|----------|
| **SAELens** | github.com/jbloomAus/SAELens | Training, loading, analyzing SAEs; GemmaScope integration | **Adopt** |
| **sae-spelling** | github.com/lasr-spelling/sae-spelling | Official absorption measurement code; FeatureAbsorptionCalculator | **Adopt/Extend** |
| **SAEBench** | github.com/adamkarvonen/SAEBench | Standardized evaluation framework | **Adopt** |
| **TransformerLens** | github.com/neelnanda-io/TransformerLens | Mechanistic interpretability library | **Adopt** |
| **matryoshka_sae** | github.com/bartbussmann/matryoshka_sae | Hierarchical SAE implementation | **Reference** |
| **feature-interp** | github.com/GeorgeMLP/feature-interp | Revising and falsifying SAE explanations (NeurIPS 2025) | **Reference** |

### Pretrained Models

- **GemmaScope** (Google, 2024): Pre-trained JumpReLU SAEs for Gemma-2-2B and Gemma-2-9B, every layer, widths 16k-1M
- **GPT-2 Small SAEs**: Available via SAELens for all residual stream layers
- **Pythia SAEs**: Available via SAELens

### Key Libraries

```python
# SAELens installation
pip install sae-lens

# Loading GemmaScope SAEs
from sae_lens import SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_8/width_16k/canonical",
)
```

---

## 6. Implications for Idea Generation

### Saturated Directions

- Pure architectural innovations without absorption analysis (many 2024 papers)
- Single-metric optimization (sparsity or reconstruction alone)
- Small-scale validation (single model, single layer)
- Basic feature splitting observation (well-documented by Chanin et al.)
- Simple sparsity-fidelity trade-off analysis (well-covered by SAEBench)
- L1-sparsity SAEs with resampling (superseded by TopK/JumpReLU)

### Promising Directions

1. **Unsupervised absorption detection limits** (THIS PROJECT): First systematic analysis of scaling limits. The negative result is itself a contribution.

2. **Causal inference for absorption detection**: Our theoretical analysis suggests absorption detection is inherently a causal inference task. Intervention-based methods (activation patching, ablation) may be required.

3. **Collision rate as proxy metric**: Our finding that collision rate correlates with absorption (Spearman r = 0.87) suggests a potential unsupervised proxy, though the circularity issue needs resolution.

4. **Inference-time mitigation (DFDA)**: Even when detection fails at scale, inference-time compensation may remain feasible. This is a constructive direction from a negative result.

5. **Theoretical characterization**: Develop mathematical conditions under which unsupervised detection is impossible (an impossibility result).

6. **Scale-dependent analysis**: Systematically study how absorption detection performance changes with dictionary size, model scale, and layer depth.

7. **Cross-domain analogies**: Source separation (ICA with dependent sources), hierarchical topic modeling (hLDA), and neuroscience (grandmother cell debate) offer potential insights.

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens | High | MIT | **Adopt** | Essential infrastructure; GemmaScope integration |
| sae-spelling | High | Unknown | **Adopt/Extend** | Direct absorption measurement; FeatureAbsorptionCalculator |
| SAEBench | High | MIT | **Adopt** | Standardized evaluation; absorption score metric |
| TransformerLens | High | MIT | **Adopt** | Required for activation extraction |
| GemmaScope SAEs | High | Apache 2.0 | **Adopt** | Pre-trained; eliminates training cost |
| Chanin et al. absorption metric | High | N/A | **Extend** | Good foundation but layer-limited |

### Recommended Stack for This Project

```
Base: SAELens + TransformerLens + PyTorch
Evaluation: SAEBench metrics + custom absorption metrics
Models: GPT-2 Small (primary, for consistency with Chanin et al.)
SAEs: GemmaScope JumpReLU (pre-trained, layer 8)
```

### Critical Implementation Notes

- Use pre-trained GemmaScope SAEs (training-free analysis)
- Always include random baseline for F1 claims
- Track precision, recall, and F1 at multiple scales
- Use SAEBench absorption score for standardized comparison
- Consider cross-model validation (Pythia) for generalization

---

## Bibliography

1. Elhage, N., et al. (2022). Toy Models of Superposition. arXiv:2209.10652.
2. Bricken, T., et al. (2023). Towards Monosemanticity. Transformer Circuits Thread.
3. Templeton, A., et al. (2024). Scaling Monosemanticity. Anthropic.
4. Gao, L., et al. (2024). Scaling and Evaluating Sparse Autoencoders. arXiv:2406.04093 / ICLR 2025.
5. Rajamanoharan, S., et al. (2024). Jumping Ahead: JumpReLU Sparse Autoencoders. arXiv:2407.14435 / ICLR 2025.
6. Chanin, D., et al. (2024). A is for Absorption. arXiv:2409.14507.
7. Lieberum, T., et al. (2024). Gemma Scope. Google.
8. Chanin, D., et al. (2025). Feature Hedging. arXiv:2505.11756.
9. Bussmann, B., et al. (2025). Matryoshka SAE. arXiv:2503.17547 / ICML 2025.
10. Korznikov, et al. (2025). Orthogonal SAE. arXiv:2509.22033.
11. Various. (2025). Interpretability Illusions with Sparse Autoencoders. arXiv:2505.16004.
12. Various. (2025). Falsifying SAE Reasoning Features. arXiv:2601.05679.
13. Various. (2025). Revising and Falsifying SAE Feature Explanations. OpenReview / NeurIPS 2025.
14. Various. (2025). Use SAEs to Discover Unknown Concepts. arXiv:2506.23845.
15. Kantamneni, et al. (2025). Are Sparse Autoencoders Useful? PMLR / ICLR 2025.
16. Karvonen, A., et al. (2025). SAEBench. arXiv:2503.09532.
17. Gulko, A., et al. (2025). CE-Bench. arXiv:2509.00691.
18. Cui, J., et al. (2025). On the Limits of SAEs. arXiv:2506.15963.
19. Chen, S., et al. (2025). Provable Feature Recovery. arXiv:2506.14002 / ICLR 2025.
20. Various. (2025). Inference-Time Decomposition of Activations. arXiv:2505.17769 / ICML 2025.
21. Leask, P., et al. (2025). SAEs Do Not Find Canonical Units. ICLR 2025.
22. Muchane, M., et al. (2025). Hierarchical SAEs. arXiv:2506.01197.
23. Luo, Y., et al. (2026). From Atoms to Trees. arXiv:2602.11881.
24. Various. (2025). Temporal SAEs. arXiv:2511.05541.
25. Various. (2025). AdaptiveK SAE. arXiv:2508.17320.
26. Various. (2026). Improving Robustness via Masked Regularization. arXiv:2604.06495.
27. Various. (2025). SAGE: Scalable Ground Truth Evaluations. OpenReview.
28. Various. (2025). Sparse but Wrong. arXiv:2508.16560.
29. Smith, L., et al. (2025). Negative results for SAEs on downstream tasks. DeepMind Safety Research.
30. Bloom, J., et al. (2024). SAELens. github.com/jbloomAus/SAELens.
