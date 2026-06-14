

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

**Research Topic**: SAE Absorption, Dead Features, Feature Resuscitation, and Training Stability in Sparse Autoencoders
**Survey Date**: 2026-04-28
**arXiv Search Keywords**: ["feature absorption" "sparse autoencoder", "feature splitting" "sparse autoencoder", "SAE benchmark" "sparse autoencoder evaluation", "JumpReLU" "Gated SAE" "TopK SAE", "Matryoshka SAE" "hierarchical sparse autoencoder", "provable feature recovery" "sparse autoencoder", "dead features" "sparse autoencoder", "feature resuscitation" "sparse autoencoder", "training stability" "sparse autoencoder"]
**Web Search Keywords**: ["sparse autoencoder feature absorption SAE mechanistic interpretability 2024 2025", "SAEBench sparse autoencoder benchmark evaluation", "SAELens sparse autoencoder library GemmaScope pretrained SAEs", "Matryoshka SAE hierarchical absorption solution 2025", "transcoder vs sparse autoencoder interpretability 2025", "provable feature recovery sparse autoencoder theoretical guarantee 2025", "ghost gradients dead neurons sparse autoencoder Anthropic 2024", "TopK SAE JumpReLU training stability 2025", "adaptive temporal masking stable SAE training", "fundamental limits neural network sparsification dead neurons"]

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant technique for mechanistic interpretability of Large Language Models (LLMs), enabling researchers to decompose dense model activations into sparse, human-interpretable features. The theoretical foundation was laid by Elhage et al.'s "Toy Models of Superposition" (2022), which established that neural networks represent more features than they have dimensions by encoding features as approximately orthogonal directions in activation space.

The seminal application of SAEs to language models came with Bricken et al.'s "Towards Monosemanticity" (2023) from Anthropic's Transformer Circuits Thread, which demonstrated that dictionary learning via sparse autoencoders could decompose MLP activations into monosemantic features. Cunningham et al. (2023) showed SAEs outperform PCA/ICA on automated interpretability scores. This was followed by Templeton et al.'s scaling work (2024) on Claude 3 Sonnet, extracting millions of interpretable features.

However, critical limitations were identified in 2024-2025 across three interconnected dimensions:

1. **Feature Absorption** --- A phenomenon where broad, interpretable features get "absorbed" into more specific, token-aligned latents, creating interpretability illusions with arbitrary false negatives. Discovered by Chanin et al. (2024) in "A is for Absorption", this has sparked vigorous research on architectural solutions.

2. **Dead Features/Neurons** --- Latent dimensions that permanently stop activating during training, wasting capacity and reducing effective dictionary size. Anthropic's ghost gradients (2024) and OpenAI's AuxK loss (2024) were early responses, but fundamental limits exist.

3. **Training Instability** --- Sensitivity to initialization, learning rates, and hyperparameters causing inconsistent feature discovery across runs. The field has moved from L1-sparsity to TopK and JumpReLU architectures to decouple sparsity control from feature learning.

By 2025, the field has expanded to include architectural solutions (Matryoshka SAEs, Orthogonal SAEs, ATM), theoretical analyses (provable feature recovery guarantees, closed-form optimal solutions), comprehensive benchmarks (SAEBench, CE-Bench), and critical reassessments (SAEs interpreting randomly initialized transformers). The tension between reconstruction fidelity, sparsity, feature quality, and training stability remains the central challenge.

## 2. Core References

| # | Title | Authors | Source | Year | Key Contribution | Limitations |
|---|-------|---------|--------|------|-----------------|-------------|
| 1 | Toy Models of Superposition | Elhage et al. | Transformer Circuits Thread / arXiv:2209.10652 | 2022 | Established theoretical foundation of superposition; showed networks learn to represent sparse features in superposition | Toy models only; not applied to real LLMs |
| 2 | Towards Monosemanticity: Decomposing Language Models With Dictionary Learning | Bricken et al. | Transformer Circuits Thread | 2023 | First application of SAEs to real transformers; demonstrated monosemantic feature recovery; introduced feature splitting | Single-layer model only; limited scale |
| 3 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | Cunningham et al. | arXiv:2309.08600 | 2023 | Showed SAEs outperform PCA/ICA on automated interpretability scores | Did not identify absorption or dead features as failure modes |
| 4 | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | Templeton et al. | Anthropic / arXiv | 2024 | Scaled SAE methods to production LLM; millions of features; safety-relevant feature discovery | Did not systematically study absorption or dead features at scale |
| 5 | Scaling and evaluating sparse autoencoders | Gao et al. | arXiv:2406.04093 / ICLR 2025 | 2024 | Introduced k-sparse autoencoders (TopK), scaling laws, new quality metrics; trained 16M latent SAE; AuxK loss eliminates almost all dead latents | Does not address absorption specifically |
| 6 | Improving Dictionary Learning with Gated Sparse Autoencoders | Rajamanoharan et al. | arXiv:2404.16014 | 2024 | Gated SAE: separates gating from magnitude estimation; solves shrinkage; Pareto improvement | Absorption and dead features not studied |
| 7 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders | Rajamanoharan et al. | arXiv:2407.14435 / ICLR 2025 | 2024 | JumpReLU activation: SOTA reconstruction fidelity at given sparsity; trains L0 directly; minimal dead features via threshold-only gradients | Later shown to have absorption issues |
| 8 | **A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders** | Chanin, Wilken-Smith, Dulka, Bhatnagar, Bloom | arXiv:2409.14507 / OpenReview | 2024 | **Foundational paper**: Identified and named feature absorption; linked cause to sparsity loss + hierarchical co-occurrence; showed varying size/sparsity insufficient | Metric relies on ablation (limited to early layers); conservative underestimate; focused on GPT-2 |
| 9 | Ghost Grads: An improvement on resampling | Jermyn, Templeton | Transformer Circuits Thread (Jan 2024) | 2024 | Introduced auxiliary loss using exponential activation on dead neurons; often achieves zero dead neurons | ~2x training cost; abandoned by Anthropic due to loss spikes; bug in original implementation |
| 10 | Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders | Makkuva et al. | arXiv:2411.13117 | 2024 | Theoretical analysis of SAE inference optimality; identified amortization gap | Not directly about absorption or dead features |
| 11 | BatchTopK Sparse Autoencoders | Bussmann et al. | arXiv:2412.06410 | 2024 | Batch-level top-k: adaptive latent allocation; outperforms TopK, comparable to JumpReLU | Absorption not evaluated |
| 12 | **SAEBench: A Comprehensive Benchmark for Sparse Autoencoders** | Karvonen et al. | arXiv:2503.09532 | 2025 | **Standardized evaluation**: 8+ metrics across interpretability, disentanglement, applications; 200+ SAEs evaluated; moved beyond sparsity-fidelity | Proxy metrics may not fully capture absorption |
| 13 | A Survey on Sparse Autoencoders: Interpreting the Internal Representations of LLMs | Various | arXiv:2503.05613 / EMNLP 2025 | 2025 | Comprehensive survey covering SAE architectures, training, evaluation, and applications | Survey paper; limited novel contributions |
| 14 | **Learning Multi-Level Features with Matryoshka Sparse Autoencoders** | Bussmann et al. | arXiv:2503.17547 / ICML 2025 | 2025 | Proposed hierarchical nested SAEs; achieved ~90% reduction in absorption (0.49 -> 0.05 at L0=40) | Introduces feature hedging in inner levels; higher computational cost |
| 15 | **Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders** | Chanin et al. | arXiv:2505.11756 | 2025 | Identified hedging as the complement to absorption; showed Matryoshka trades absorption for hedging; proposed balanced loss coefficients (beta_m ~ 0.75) | Limited to empirical analysis; no theoretical characterization of trade-off |
| 16 | **Orthogonal Sparse Autoencoders Uncover Atomic Features** | Korznikov et al. | arXiv:2509.22033 | 2025 | Alternative solution via orthogonality constraints (cosine similarity penalty); ~65% absorption reduction; ~50% less compute than Matryoshka | Slightly lower explained variance |
| 17 | **Provable Feature Recovery via Sparse Autoencoders** | Chen et al. | arXiv:2506.14002 / ICLR 2025 | 2025 | **First SAE algorithm with theoretical recovery guarantees**; introduced Group Bias Adaptation (GBA); validated on 2B parameter LLMs | Guarantees require specific statistical model assumptions |
| 18 | **On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy** | Cui et al. | arXiv:2506.15963 | 2025 | **First closed-form optimal solution analysis**; proved standard SAEs fail unless features are extremely sparse; identified feature shrinking/vanishing; proposed Weighted SAE (WSAE) | Theoretical analysis; limited empirical validation |
| 19 | **Fundamental Limits of Neural Network Sparsification** | Dip Roy et al. | arXiv:2603.18056 | 2025 | Showed dead neuron recovery is severely limited under extreme sparsification; intrinsic to compression process; zero recovery on dSprites after 100 epochs | Extreme sparsification regime; may not generalize to moderate sparsity |
| 20 | **Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training** | T. Ed Li et al. | arXiv:2510.08855 / ICLR 2025 | 2025 | ATM achieves absorption score 0.0068 vs TopK's 0.1402 via dynamic probabilistic masking with EMA tracking | Newer method; less community validation |
| 21 | **Stable and Steerable Sparse Autoencoders with Weight Regularization** | Oliver Crook et al. | arXiv:2603.04198 | 2026 | L2 regularization doubles steering success rates; increases cross-seed consistency; tied init + unit-norm constraints | Aggressively kills many features, creating bimodal structure |
| 22 | Transcoders Beat Sparse Autoencoders for Interpretability | Paulo, Shabalin, Belrose | arXiv:2501.18823 | 2025 | Proposed transcoders as superior alternative; skip transcoders Pareto-dominate SAEs; higher automated interpretability scores across Pythia, Llama, Gemma | Different objective (input-output vs. self-reconstruction); not directly comparable for all use cases |
| 23 | Sparse Autoencoders Can Interpret Randomly Initialized Transformers | Various | arXiv:2501.17727 | 2025 | Showed SAEs find similar features in trained and untrained models; raises questions about what SAEs actually capture | Challenges the assumption that SAE features reflect learned computations |
| 24 | Measuring Sparse Autoencoder Feature Sensitivity | Tian et al. | arXiv:2509.23717 | 2025 | Introduces feature sensitivity: reliability of feature activation on semantically similar texts | Complementary to absorption; not causal |
| 25 | Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures | Muchane et al. | arXiv:2506.01197 | 2025 | HSAE: explicitly models semantic hierarchy; improves reconstruction and interpretability | Requires hierarchical structure assumption |
| 26 | From Atoms to Trees: Building a Structured Feature Forest with Hierarchical SAEs | Luo et al. | arXiv:2602.11881 | 2026 | HSAE jointly learns SAEs and parent-child relationships; recovers meaningful hierarchies | Very recent; limited validation |
| 27 | Data Whitening Improves Sparse Autoencoder Learning | Saraswatula & Klindt | arXiv:2511.13981 | 2025 | PCA whitening improves interpretability metrics on SAEBench; challenges sparsity-fidelity tradeoff | Minor reconstruction drops |
| 28 | Resurrecting the Salmon: Domain-Specific Sparse Autoencoders | O'Neill et al. | arXiv:2508.09363 | 2025 | Domain confinement mitigates fragmentation/absorption; 20% more variance explained | Limited to medical domain |
| 29 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | Song et al. | arXiv:2505.20254 | 2025 | Argues for feature consistency (PW-MCC metric); high consistency achievable with right architectures | Consistency != absence of absorption |
| 30 | Kronecker Factorization Improves Efficiency and Interpretability of SAEs | Various | arXiv:2505.22255 | 2025 | KronSAE: Kronecker factorization + mAND gating; reduces absorption via AND-like behavior | Architectural complexity |
| 31 | Distribution-Aware Feature Selection for SAEs | Various | arXiv:2508.21324 | 2025 | L2-norm/Squared-l scoring reduces absorption vs BatchTopK; no single config dominates all metrics | Trade-offs remain |
| 32 | CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark | Gulko et al. | arXiv:2509.00691 | 2025 | LLM-free contrastive evaluation using story pairs; 70%+ Spearman correlation with SAEBench | Limited metric coverage |
| 33 | Taming Polysemanticity in LLMs: Theory-Grounded Feature Recovery | Chen et al. | ICLR 2025 | 2025 | "Neuron resonance" phenomenon; neurons learn monosemantic features when activation frequency matches feature occurrence | Theoretical; limited scope |
| 34 | Train One Sparse Autoencoder Across Multiple Sparsity Levels | Various | EMNLP 2025 | 2025 | HierarchicalTopK enables post-hoc sparsity selection without retraining; BatchTopK mixes k values within batch | Limited sparsity range (k <= 128) |
| 35 | Interpreting CLIP with Hierarchical Sparse Autoencoders | Zaigrajew et al. | arXiv:2502.20578 / ICML 2025 | 2025 | Soft-capping reduces dead neurons (6 vs 491 at scale); multi-granularity TopK for vision-language | Vision-language specific; may not transfer to pure LLMs |
| 36 | Deep sparse autoencoders yield interpretable features too | Various | Alignment Forum | 2025 | Activation decay as alternative to dead neuron resampling for deep SAEs | Deep SAE specific |
| 37 | Sparse Autoencoders Do Not Find Canonical Units of Analysis | Leask et al. | ICLR 2025 / arXiv | 2025 | Challenged canonical feature hypothesis via SAE stitching and Meta-SAEs; introduced BatchTopK | Shows SAEs are incomplete/non-atomic, raising fundamental concerns |
| 38 | Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models | Various | ICLR 2025 | 2025 | Investigates feature universality across different LLMs via activation correlation and SVCCA | Focus on universality, not absorption |
| 39 | Binary Sparse Coding for Interpretability | Various | arXiv:2509.25596 | 2025 | Binarizes activations to prevent information smuggling through continuous activation strengths | Binary constraints may limit expressiveness |
| 40 | Self-Ablating Transformers: More Interpretability, Less Sparsity | Various | arXiv:2505.00509 | 2025 | Self-ablating mechanism for improved interpretability with reduced sparsity requirements | Alternative to SAEs entirely |

## 3. SOTA Methods and Benchmarks

### Current Best Methods for Addressing Absorption

| Method | Absorption Reduction | Key Innovation | Computational Overhead | Trade-off |
|--------|---------------------|----------------|----------------------|-----------|
| **Matryoshka SAE** | ~90% (0.49 -> 0.05 at L0=40) | Hierarchical nested dictionaries with independent reconstruction constraints | High (multiple dictionary sizes) | Introduces hedging in inner levels |
| **Balanced Matryoshka SAE** | Optimized trade-off | Tuned relative loss coefficients (beta_m ~ 0.75) | High | Balances absorption vs. hedging |
| **Orthogonal SAE (OrtSAE)** | ~65% | Cosine similarity penalty between latents | Low (~50% less than Matryoshka) | Slightly lower explained variance; 4-11% slower inference |
| **ATM (Adaptive Temporal Masking)** | ~95% (0.1402 -> 0.0068) | Temporal dynamics + probabilistic masking with dual EMAs | Medium | Very recent; limited validation |
| **Weighted SAE (WSAE)** | Improved low-sparsity recovery | Theoretically-grounded reweighting strategy | Low | Limited empirical validation |
| **Group Bias Adaptation (GBA)** | Strong empirical results | Adaptive bias adjustment for feature identifiability; theoretical guarantees | Medium | Requires specific statistical assumptions; tested up to 2B params |
| **KronSAE** | Lower absorption | Kronecker factorization + mAND gating | Medium | Architectural complexity |
| **JumpReLU** | ~92% (0.1402 -> 0.0114) | Threshold-only sparsity gradients; minimal dead features | Low | Requires careful gradient routing |
| **Binary Sparse Coding** | Moderate | Binarized activations prevent information smuggling | Low | Binary constraints may limit expressiveness |

### Current Best Methods for Dead Feature Resuscitation/Prevention

| Method | Mechanism | Effectiveness | Trade-off |
|--------|-----------|---------------|-----------|
| **AuxK Loss** (OpenAI) | Auxiliary reconstruction using dead latents; L_aux = ||e - e_hat||^2 with top-k_aux dead latents | "Eliminates almost all dead latents by end of training" | Low overhead; combined with tied initialization |
| **Ghost Gradients** (Anthropic) | Second forward pass with exponential activation on dead neurons; gradient pushes dead neurons toward explaining residual | Often achieves zero dead neurons | ~2x training cost; abandoned due to loss spikes; bug in original impl |
| **Soft-capping** | Bounds activation magnitudes | 6 vs 491 dead neurons at scale (latent size 12288) | Minimal fidelity impact |
| **L2 Weight Regularization** | Penalizes large weights; creates core of highly aligned features | Roughly doubles steering success rates; better cross-seed consistency | Aggressively kills many features; bimodal structure |
| **Tied Initialization** (W_enc = W_dec^T) | Better gradient flow at start | "Important for preventing dead latents" (OpenAI) | Standard practice |
| **Geometric Median Initialization** | Initialize decoder bias at geometric median of activation distribution | Avoids both dense and dead features; reduces hyperparameter sensitivity | Good initialization practice |
| **Bias Adaptation (GBA)** | Adaptively adjusts bias parameters; "neuron resonance" principle | Outperforms benchmarks on LLMs up to 2B; theoretical guarantees | Moderate overhead |
| **Activation Decay** | Penalize mean of squared sparse feature activations | Alternative to resampling for deep SAEs | Deep SAE specific |
| **JumpReLU (threshold-only gradients)** | Sparsity gradient only to threshold theta, not encoder/decoder | Hardly any dead features | Best reconstruction fidelity |
| **Lower Learning Rates** | Slower, more stable optimization | Anecdotally helps decrease dead latents | Longer training |
| **LR Warmup** | Gradual LR increase at start | Keeps features alive before ghost gradients activate | Minimal overhead |

### The Absorption-Hedging Trade-off (Key 2025 Insight)

| Problem | Cause | Affected Architecture | Solution Approach |
|---------|-------|----------------------|-------------------|
| **Feature Absorption** | Sparsity loss + hierarchical co-occurrence | Wide SAEs | Matryoshka hierarchy; orthogonality constraints |
| **Feature Hedging** | Reconstruction (MSE) loss + correlations | Narrow SAEs | Balanced loss coefficients; wider SAEs |

Chanin et al. (2025) showed these are complementary problems: Matryoshka SAEs trade absorption for hedging in inner (narrow) levels.

### Training Stability: Architecture Comparison

| Architecture | Stability Strength | Stability Weakness | Dead Features | Absorption Score |
|-------------|-------------------|-------------------|---------------|------------------|
| **Standard SAE (L1)** | Simple | Worst feature absorption; unstable training | High | 0.0161 |
| **TopK SAE** | Fast, efficient; fixed sparsity | Feature absorption (0.1402); dead features without AuxK | High (without AuxK) | 0.1402 |
| **JumpReLU** | Best reconstruction fidelity; minimal dead features | Requires careful gradient routing (only to theta) | Very low | 0.0114 |
| **BatchTopK** | Adaptive per-sample sparsity | Moderate absorption | Moderate | Moderate |
| **ATM (2025)** | Best feature absorption (0.0068); adaptive masking | Newer, less validated | Low | 0.0068 |
| **L2-Regularized TopK** | Excellent cross-seed consistency; better steering | Aggressive feature death (many dead latents) | High (bimodal) | Improved |
| **Matryoshka SAE** | Hierarchical feature organization | Higher compute; hedging in inner levels | Low | 0.005 (outer) |

### Mainstream Datasets and Evaluation Standards

- **Training data**: OpenWebText (most common), The Pile
- **Models**: GPT-2 Small (most common for research), Pythia series (70M-2.8B), Llama 3.2, Gemma 2, Claude 3 Sonnet
- **Standard evaluation metrics**:
  - **L0 Sparsity**: Average number of active features per input
  - **MSE / Normalized MSE / FVU**: Reconstruction error
  - **Loss Recovered**: Cross-entropy preservation when substituting SAE reconstructions; formula: (H* - H0) / (H_orig - H0)
  - **R^2 (Explained Variance)**: Proportion of variance captured
  - **Feature Absorption**: Fraction of cases where correct latents fail to activate (mean and full absorption)
  - **Dead Latent Percentage**: Fraction of features that never activate
  - **Cross-seed Consistency**: Feature alignment across different random initializations
  - **Steering Success Rate**: Ability to modify model behavior via feature intervention
  - **Sparse Probing**: k=1,2,5 probing accuracy for concept detection
  - **Auto-Interp**: LLM-as-judge automated interpretability scores

### SAEBench (2025) --- The Emerging Standard

SAEBench organizes evaluation around four capability dimensions:

| Capability | Metrics Used |
|------------|-------------|
| **Concept Detection** | Sparse Probing, Feature Absorption |
| **Interpretability** | LLM-as-judge automated interpretability |
| **Reconstruction** | Loss Recovered, KL Divergence, MSE |
| **Feature Disentanglement** | Unlearning, SCR, TPP, RAVEL |

Novel metrics introduced:
- **SCR (Spurious Correlation Removal)**: Tests if zero-ablating small latent sets removes unwanted correlations
- **TPP (Targeted Probe Perturbation)**: Measures completeness and isolation of concepts in small latent groups
- **RAVEL**: Evaluates clean separation of related attributes via interventions

Critical insight: "Small gains in sparsity-fidelity trade-off do not necessarily translate into qualitatively better representations."

### Transcoders as Alternative Paradigm (2025)

Paulo et al. (2025) proposed transcoders as a superior alternative to SAEs:

| Aspect | SAEs | Transcoders |
|--------|------|-------------|
| Objective | Self-reconstruction | Input-output function approximation |
| Training target | x_r = x_p (same layer) | x_p = NN_l(s), x_r = NN_{l+1}(s) |
| What they capture | Polysemantic activations | Layer-to-layer transformations |
| Key advantage | Simple | More interpretable; better generalization |

Skip transcoders (with affine skip connection) Pareto-dominate SAEs on reconstruction vs. interpretability. The authors recommend shifting focus from SAEs to (skip) transcoders.

## 4. Identified Research Gaps

### Absorption-Specific Gaps

- **Gap 1: Theoretical understanding of absorption-hedging trade-off**. While Matryoshka SAEs reduce absorption, they introduce hedging in inner levels. The fundamental Pareto frontier of this trade-off is not well-characterized theoretically.
- **Gap 2: Absorption-aware training objectives**. Current solutions modify architecture; absorption-aware loss terms that directly penalize absorption during training are underexplored.
- **Gap 3: Unsupervised absorption detection**. Current metrics require knowing the "parent" feature a priori. Detecting absorption without ground truth is unsolved.
- **Gap 4: Cross-architecture comparison on absorption**. Most absorption studies focus on JumpReLU/GemmaScope. Systematic comparison across all major architectures using standardized absorption metrics is lacking.
- **Gap 5: Absorption in non-text modalities**. SAEs are being applied to vision-language models. Whether absorption manifests differently in cross-modal features is unknown.

### Dead Feature-Specific Gaps

- **Gap 6: Theoretical understanding of feature death**. While empirical methods (AuxK, ghost gradients) work well, the theoretical conditions under which features die and can be recovered remain poorly understood. The "neuron resonance" principle from GBA is a start but needs extension.
- **Gap 7: Fundamental limits of recovery**. Dip Roy et al. (2025) showed zero recovery on dSprites even after 100 epochs. Whether this holds for LLM SAEs at moderate sparsity is unknown.
- **Gap 8: Dynamic feature resuscitation during inference**. Current resuscitation methods operate during training. Real-time revival of dead features during model deployment is unexplored.
- **Gap 9: Interaction between absorption and dead features**. These two problems are often studied separately, but they likely interact --- absorbed features may be more prone to dying, and dead features may contribute to absorption in surviving features.

### Training Stability Gaps

- **Gap 10: Unified stability metric**. The field uses many metrics (absorption score, dead latent %, cross-seed consistency) but lacks a unified framework that captures the full stability landscape.
- **Gap 11: Scaling to larger models**. Most dead feature/absorption research is validated on models <= 2B parameters. Behavior at 7B+ and especially 70B+ remains largely unstudied.
- **Gap 12: Training stability under data distribution shift**. SAE stability when trained on different data distributions or when deployed on out-of-distribution inputs is underexplored.
- **Gap 13: What do SAEs actually capture?** The finding that SAEs interpret randomly initialized transformers raises fundamental questions about whether absorption and dead features stem from data statistics and architecture rather than learned computations.

## 5. Available Resources

### Open-source Code

| Resource | URL | Description | Relevance |
|----------|-----|-------------|-----------|
| **SAELens** | https://github.com/jbloomAus/SAELens | Training, loading, analyzing SAEs; integrates GemmaScope; standard tool in MI community; supports ReLU, Gated, TopK, JumpReLU | **High** --- foundational library |
| **sae-spelling** | https://github.com/lasr-spelling/sae-spelling | Official code for "A is for Absorption"; FeatureAbsorptionCalculator, k-sparse probing | **High** --- direct absorption tooling |
| **matryoshka_sae** | https://github.com/bartbussmann/matryoshka_sae | Matryoshka SAE implementation; hierarchical dictionary learning | **High** --- primary absorption solution |
| **SAEBench** | https://github.com/adamkarvonen/SAEBench | Comprehensive benchmark framework for SAE evaluation | **High** --- standard evaluation suite |
| **TransformerLens** | https://github.com/neelnanda-io/TransformerLens | Mechanistic interpretability library for transformers | **High** --- required for interventions |
| **Sparsify** | https://github.com/EleutherAI/sparsify | EleutherAI's lean alternative focused on TopK SAEs | **Medium** --- alternative implementation |
| **SAETrainer** | https://github.com/sionic-ai/SAETrainer | Another training framework with different architectural choices | **Medium** --- alternative |
| **SAELens-V** | https://github.com/PKU-Alignment/SAELens-V | Extension for multimodal (vision + language) SAEs | **Medium** --- for cross-modal research |
| **MSAE** | https://github.com/WolodjaZ/MSAE | Hierarchical SAE for CLIP/SigLIP | **Medium** --- vision-language specific |
| **HSAE** | https://github.com/nqgl/HSAE | Hierarchical Sparse Autoencoder (early work) | **Medium** --- precursor to Matryoshka |

### Datasets and Pretrained Models

- **GemmaScope** (Google, 2024): Pre-trained JumpReLU SAEs for Gemma-2-2B and Gemma-2-9B, every layer, widths 16k-1M
- **GPT-2 Small SAEs**: Available via SAELens for all residual stream layers
- **Pythia SAEs**: Available via SAELens
- **OpenWebText / The Pile**: Standard training corpora for SAEs
- **SAEBench evaluation suite**: Standardized datasets for all 8+ metrics
- **CE-Bench dataset**: Contrastive evaluation pairs for SAE interpretability assessment

### Key Libraries

```python
# SAELens installation
pip install sae-lens

# Loading GemmaScope SAEs
from sae_lens import SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_0/width_16k/canonical",
)
```

## 6. Implications for Idea Generation

### Saturated Directions

- Pure architectural innovations without absorption analysis (many 2024 papers)
- Single-metric optimization (sparsity or reconstruction alone)
- Small-scale validation (single model, single layer)
- Basic feature splitting observation (well-documented)
- Simple sparsity-fidelity trade-off analysis (well-covered by SAEBench)
- L1-sparsity SAEs with resampling --- superseded by TopK/JumpReLU
- Pure empirical scaling studies without theoretical insight

### Promising Directions

1. **Joint absorption-death modeling**: Design a unified objective that simultaneously addresses feature absorption and dead features, exploiting their likely interaction. Current methods treat these as separate problems.

2. **Absorption-aware training objectives**: Rather than architectural changes, design loss terms that directly penalize absorption during training. This could combine with any SAE architecture (TopK, JumpReLU, etc.).

3. **Adaptive resuscitation schedules**: Instead of fixed thresholds (e.g., 2M tokens for ghost gradients, k_aux=512 for AuxK), learn dynamic resuscitation triggers based on feature importance and training dynamics.

4. **Hierarchical resuscitation**: Extend Matryoshka's hierarchical structure with explicit resuscitation pathways between levels --- when a feature dies at one level, attempt to revive it using information from adjacent levels.

5. **Stability-aware architecture search**: Automatically discover SAE architectures that optimize the stability-reconstruction-sparsity Pareto frontier, considering absorption, dead features, and cross-seed consistency jointly.

6. **Theoretical characterization of absorption**: Develop mathematical conditions under which absorption occurs, analogous to identifiability theory for feature recovery. The closed-form analysis by Cui et al. (2025) provides a starting framework.

7. **Unsupervised absorption detection**: Methods to identify absorption without ground-truth parent features. This would enable absorption auditing at scale across all SAE features.

8. **Scale-dependent analysis**: Systematically study how absorption and dead features change with model scale (100M -> 1B -> 10B+ parameters). Most research stops at 2B.

9. **Training-free mitigation**: Post-hoc methods to recover absorbed features from trained SAEs without retraining.

10. **Cross-modal absorption**: Study whether absorption manifests differently in vision-language models. Cross-modal hierarchical features may exhibit novel patterns.

11. **What do SAEs actually capture?**: Follow-up to the randomly-initialized transformers finding; distinguish architecture/data artifacts from learned computations.

### Cross-Domain Analogies with Potential

- **Signal processing**: Source separation problems (e.g., ICA with dependent sources) face similar "absorption" issues where correlated signals merge. Techniques from blind source separation may transfer.
- **Topic modeling**: Hierarchical topic models (e.g., hLDA) handle nested concepts without absorption-like failures. Their probabilistic approach to hierarchy may inspire SAE variants.
- **Neuroscience**: The "grandmother cell" debate (whether neurons encode specific vs. distributed representations) parallels the monosemanticity debate. Insights from sparse coding in visual cortex may apply.
- **Mixture of Experts**: Load balancing techniques from MoE training could address feature utilization imbalance (dead features).
- **Dictionary learning in signal processing**: OMP and CoSaMP algorithms have sophisticated atom selection criteria that could inspire SAE feature selection.

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens | High | MIT | **Adopt** | Mature library with GemmaScope integration; essential infrastructure for all SAE experiments; supports all major variants |
| sae-spelling | High | Unknown | **Adopt/Extend** | Official absorption measurement code; directly implements FeatureAbsorptionCalculator and k-sparse probing |
| matryoshka_sae | High | Unknown | **Adopt/Extend** | Primary solution to absorption; based on SAELens; can be extended with custom loss coefficients |
| SAEBench | High | MIT | **Adopt** | Standardized evaluation; necessary for rigorous benchmarking beyond sparsity-fidelity |
| TransformerLens | High | MIT | **Adopt** | Required for activation extraction and interventions |
| GemmaScope SAEs | High | Apache 2.0 | **Adopt** | Pre-trained SAEs eliminate training cost; established absorption test cases |
| Sparsify | Medium | MIT | **Extend** | Lean TopK-focused; good for experiments needing minimal dependencies |
| Chanin et al. absorption metric | High | N/A (paper) | **Extend** | Good foundation but layer-limited; extend to all layers |
| HSAE (Muchane et al.) | Medium | N/A | **Reference** | Hierarchical modeling approach informs design |
| KronSAE | Medium | N/A | **Reference** | mAND gating mechanism may inspire solutions |

### Recommended Stack

```
Base: SAELens + TransformerLens + PyTorch
Evaluation: SAEBench metrics + custom absorption metrics + CE-Bench
Models: Gemma-2-2B (primary), Pythia-70M-deduped (fast iteration), GPT-2 Small (historical baseline)
SAEs: GemmaScope (pre-trained), train custom variants if needed
```

### Key Reusable Components

1. **SAELens SAE loading**: Direct access to 1000+ pre-trained SAEs
2. **SAEBench evaluation pipeline**: Standardized metrics computation
3. **GemmaScope first-letter features**: Chanin et al. established these as absorption test cases
4. **k-sparse probing**: Feature splitting detection from absorption paper
5. **Integrated gradients ablation**: Causal verification from absorption paper
6. **FeatureAbsorptionCalculator**: Direct measurement tool from sae-spelling repo
7. **AuxK loss implementation**: From OpenAI's scaling paper; standard in SAELens
8. **Ghost gradients reference**: From Anthropic's Transformer Circuits Thread (for historical comparison)

### Critical Implementation Notes

- **Use TopK or JumpReLU instead of L1 sparsity** for new experiments --- L1 is now considered deprecated for SAEs.
- **Always include AuxK or equivalent dead feature mitigation** --- training without it yields unreliable results.
- **Use tied initialization (W_enc = W_dec^T)** --- shown to be important for preventing dead latents.
- **Track cross-seed consistency** --- this is becoming a standard evaluation criterion alongside reconstruction and sparsity.
- **Monitor absorption score explicitly** --- many papers now report this alongside traditional metrics.
- **Consider geometric median initialization for decoder bias** --- avoids both dense and dead features.
- **For JumpReLU**: Ensure sparsity gradient flows ONLY to threshold theta, not encoder/decoder --- this is critical for minimizing dead features.
- **For Matryoshka SAEs**: Tune beta_m ~ 0.75 for balanced absorption-hedging trade-off.

---

## Bibliography

1. Elhage, N., et al. (2022). Toy Models of Superposition. Transformer Circuits Thread. arXiv:2209.10652.
2. Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.
3. Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600.
4. Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Anthropic.
5. Gao, L., et al. (2024). Scaling and evaluating sparse autoencoders. arXiv:2406.04093.
6. Rajamanoharan, S., et al. (2024). Improving Dictionary Learning with Gated Sparse Autoencoders. arXiv:2404.16014.
7. Rajamanoharan, S., et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. arXiv:2407.14435.
8. Chanin, D., et al. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507.
9. Jermyn, A., & Templeton, A. (2024). Ghost Grads: An improvement on resampling. Transformer Circuits Thread (January 2024 Update).
10. Makkuva, A., et al. (2024). Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders. arXiv:2411.13117.
11. Bussmann, B., et al. (2024). BatchTopK Sparse Autoencoders. arXiv:2412.06410.
12. Karvonen, A., et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. arXiv:2503.09532.
13. Bussmann, B., et al. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. arXiv:2503.17547.
14. Chanin, D., et al. (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756.
15. Korznikov, et al. (2025). Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033.
16. Chen, S., et al. (2025). Provable Feature Recovery via Sparse Autoencoders. arXiv:2506.14002.
17. Cui, J., et al. (2025). On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963.
18. Dip Roy, et al. (2025). Fundamental Limits of Neural Network Sparsification. arXiv:2603.18056.
19. Li, T. E., et al. (2025). Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training. arXiv:2510.08855.
20. Crook, O., et al. (2026). Stable and Steerable Sparse Autoencoders with Weight Regularization. arXiv:2603.04198.
21. Paulo, G., Shabalin, S., & Belrose, N. (2025). Transcoders Beat Sparse Autoencoders for Interpretability. arXiv:2501.18823.
22. Various. (2025). Sparse Autoencoders Can Interpret Randomly Initialized Transformers. arXiv:2501.17727.
23. Tian, C., et al. (2025). Measuring Sparse Autoencoder Feature Sensitivity. arXiv:2509.23717.
24. Muchane, M., et al. (2025). Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures. arXiv:2506.01197.
25. Luo, Y., et al. (2026). From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders. arXiv:2602.11881.
26. Saraswatula, A., & Klindt, D. (2025). Data Whitening Improves Sparse Autoencoder Learning. arXiv:2511.13981.
27. O'Neill, C., et al. (2025). Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders. arXiv:2508.09363.
28. Song, X., et al. (2025). Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs. arXiv:2505.20254.
29. Various. (2025). Kronecker Factorization Improves Efficiency and Interpretability of SAEs. arXiv:2505.22255.
30. Various. (2025). Distribution-Aware Feature Selection for SAEs. arXiv:2508.21324.
31. Gulko, A., et al. (2025). CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark. arXiv:2509.00691.
32. Chen, S., et al. (2025). Taming Polysemanticity in LLMs: Theory-Grounded Feature Recovery. ICLR 2025.
33. Various. (2025). Train One Sparse Autoencoder Across Multiple Sparsity Levels. EMNLP 2025.
34. Zaigrajew, V., et al. (2025). Interpreting CLIP with Hierarchical Sparse Autoencoders. arXiv:2502.20578.
35. Various. (2025). Deep sparse autoencoders yield interpretable features too. Alignment Forum.
36. Leask, P., et al. (2025). Sparse Autoencoders Do Not Find Canonical Units of Analysis. ICLR 2025.
37. Various. (2025). Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models. ICLR 2025.
38. Various. (2025). Binary Sparse Coding for Interpretability. arXiv:2509.25596.
39. Various. (2025). Self-Ablating Transformers: More Interpretability, Less Sparsity. arXiv:2505.00509.
40. Lieberum, T., et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.
41. Bloom, J., et al. (2024). SAELens: Training Sparse Autoencoders on Language Models. https://github.com/jbloomAus/SAELens


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# 最终研究提案：无监督吸收检测（UAD）——聚焦核心贡献的务实路线

> 综合者：sibyl-synthesizer
> 日期：2026-04-28
> 基础：6 视角提案 + 上一轮 lessons learned

---

## 一、综合决策：选择最强想法

### 1.1 六视角评估矩阵

| 视角 | 核心主张 | 可行性 | 与项目契合度 | 采纳程度 |
|------|---------|--------|-------------|---------|
| **实用主义** | 聚焦 UAD，放弃训练-SAE，降级架构比较 | ★★★★★ | ★★★★★ | **完全采纳** |
| **实证主义** | 方法论验证优先，代理指标验证是金标准 | ★★★★★ | ★★★★★ | **完全采纳** |
| **反对者** | 质疑吸收本身是否真实问题，碰撞率合法性危机 | ★★★★ | ★★★★ | **部分采纳** |
| **理论** | 吸收的不可避免性定理，UAD 的理论基础 | ★★★ | ★★★★ | **部分采纳** |
| **跨学科** | 信息论/统计物理/认知科学框架 | ★★★ | ★★★ | **融入讨论** |
| **创新者** | 信道容量、相变、概念层级等跨领域框架 | ★★★ | ★★★ | **融入讨论** |

### 1.2 核心判断

**最强想法 = UAD（无监督吸收检测）作为唯一核心贡献**

理由：
1. **唯一被充分验证**：Pilot（d_SAE=3,072, F1=0.522）和 Full（d_SAE=16,384, F1=0.704）均成功运行
2. **真正新颖**：社区首个无需 ground truth 的吸收检测方法
3. **完全 training-free**：符合项目 spec 约束，无需训练 SAE
4. **工程可行**：单次分析 <30 分钟，在当前时间预算内完全可控
5. **可扩展**：可应用于任何预训练 SAE（GemmaScope、SAELens 等）

### 1.3 明确排除的方向

| 方向 | 排除理由 |
|------|---------|
| **训练-SAE 实验** | 违反 spec "training-free" 约束；上一轮 89-99% 死特征证明在当前预算下不可行 |
| **架构比较（因果推断）** | 混杂变量不可控（数据、字典大小、预训练 vs 从头训练）；干净比较需要数小时 GPU 时间 |
| **跨学科理论框架（主贡献）** | 信息论/相变/量子力学等框架新颖但缺乏实证支撑；适合 Discussion，不适合作为核心实验 |
| **"吸收是伪问题"论题** | 过于激进；虽然反对者提出有力质疑，但完全否定吸收会使 UAD 失去研究对象 |

---

## 二、纳入反馈意见

### 2.1 实用主义反馈 → 实验设计

**采纳**：
- 全部使用预训练 SAE（GemmaScope、SAELens GPT-2 Small SAEs）
- 碰撞率-吸收率相关性验证作为 P0 必须实验
- UAD 消融实验（DFDA、共现分析、特异性过滤的独立贡献）
- 放弃传统 p 值假设检验，改用效应量 + 置信区间

### 2.2 实证主义反馈 → 方法论底线

**采纳**：
- **P0：代理指标验证**——在至少 3 个层次结构上验证碰撞率与真实吸收率的相关性
- **事前功效分析**——所有实验设计前计算最小可检测效应
- **可复现性检查清单**——每个实验任务完成后自动验证
- **死特征诊断**——区分"真死"与"假死"，解释 89-99% 死特征率

### 2.3 反对者反馈 → 诚实披露

**采纳**：
- 论文显著位置声明：碰撞率是代理指标，与 Chanin et al. 吸收率的对应关系需验证
- 承认 F1=0.704 是中等分数，54% 误报率在实际审计中有限制
- 承认 UAD 在无 ground truth 场景下无法被验证——这是无监督方法的内在悖论
- 承认 GPT-2 Small 的代表性局限
- 承认统计功效不足对"证伪"声明的威胁

**不采纳**：
- 不将论文重新定位为"碰撞率不是吸收率的合法代理"——这会否定 UAD 的研究基础
- 不将吸收完全视为"伪问题"——虽然质疑有力，但 Chanin et al. 的监督检测已证明吸收在 ground truth 场景下真实存在

### 2.4 理论反馈 → 论文深度

**采纳**：
- 在 Discussion 中引入"吸收-死亡耦合"猜想：高死特征率不是训练失败，而是优化约束的必然结果
- 将 UAD 框架为字典学习可识别性问题的自然推论
- 提出"架构无关的归一化吸收度量"概念（作为未来工作）

**不采纳**：
- 不将理论证明作为核心实验——toy model 证明需要 2-3 天，超出时间预算
- 不声称"吸收不可避免性定理已证明"——仅作为猜想和讨论方向

### 2.5 跨学科/创新反馈 → Discussion 提升

**采纳**：
- Discussion 中加入"语义信道容量"概念（信息论）
- Introduction 引入"祖母细胞辩论"类比（认知科学）
- 用相变框架解释 pilot-full 8 倍差异（统计物理）
- "特征生态学"概念统一死特征和吸收（生态学）

**不采纳**：
- 不将跨学科框架作为核心实验——缺乏直接实证路径
- 不引入量子力学不确定性原理——类比过于牵强，可能损害论文可信度

---

## 三、最终实验计划

### 3.1 实验优先级

| 优先级 | 实验 | 预计时间 | 必要性 | 输出 |
|--------|------|----------|--------|------|
| **P0** | 碰撞率-吸收率相关性验证 | 1-2 小时 | **必须** | 相关系数 r，决定指标命运 |
| **P0** | UAD 在 GemmaScope 多 SAE 上验证 | 2-3 小时 | **必须** | 跨模型/跨层 F1 分数 |
| **P1** | UAD 消融实验 | 1-2 小时 | 重要 | 各组件独立贡献 |
| **P1** | DFDA 扩展（更多层次结构） | 1 小时 | 辅助 | 死特征-吸收关联稳健性 |
| **P2** | 跨层/跨宽度碰撞率观察 | 1-2 小时 | 可选 | 补充材料 |

### 3.2 技术栈

```
Base: SAELens + TransformerLens + PyTorch
模型: GPT-2 Small（主要）、Gemma-2-2B（如果 API 可用）
SAE: GemmaScope（预训练）、SAELens 提供的 GPT-2 Small SAEs
评估: 自定义 UAD 实现
```

**关键决策**：
1. 全部使用预训练 SAE，绝不自己训练
2. 优先使用 GemmaScope 多宽度 SAE（16k-1M）
3. UAD 实现封装为可复用模块
4. 中间结果缓存到磁盘

---

## 四、论文定位与结构

### 4.1 论文标题建议

**主标题**：Unsupervised Absorption Detection: A Training-Free Approach to Auditing SAE Feature Completeness

**备选（如果跨学科框架融入较多）**：Beyond Ground Truth: Unsupervised Detection of Feature Absorption in Sparse Autoencoders

### 4.2 核心叙事

```
问题：SAE 特征吸收检测需要 ground truth，无法大规模审计
→ 贡献：提出 UAD，首个无监督吸收检测方法（F1=0.704，完美召回）
→ 验证：在 GemmaScope 多 SAE 上验证泛化性
→ 诚实：讨论局限性（代理指标假设、GPT-2 局限、误报率）
→ 展望：理论框架（信息论、相变、可识别性）
```

### 4.3 论文结构

```
1. Introduction
   - 问题：吸收检测需要 ground truth，无法大规模审计
   - 贡献：提出 UAD，首个无监督吸收检测方法
   - 诚实声明：本文聚焦 UAD；碰撞率作为代理指标需验证

2. Background & Related Work
   - 特征吸收定义（Chanin et al.）
   - 现有检测方法的局限性（全部需要 ground truth）
   - 跨学科视角简述（祖母细胞、信道容量——1 段）

3. Method: Unsupervised Absorption Detection (UAD)
   - 3.1 死特征分布异常（DFDA）
   - 3.2 层次共现分析
   - 3.3 特异性过滤
   - 3.4 整体算法流程
   - 3.5 理论直觉（可识别性框架——1 段）

4. Experiments
   - 4.1 碰撞率-吸收率相关性验证（P0）
   - 4.2 UAD 在 GemmaScope 上的验证（P0）
   - 4.3 UAD 消融实验（P1）
   - 4.4 DFDA 扩展分析（P1）

5. Discussion
   - 5.1 吸收-死亡耦合猜想（理论视角）
   - 5.2 语义信道容量（信息论视角）
   - 5.3 相变解释（统计物理视角）
   - 5.4 局限性与未来工作

6. Conclusion
```

### 4.4 必须包含的诚实声明

> "本研究使用'碰撞率'作为特征吸收的代理指标。该代理指标与 Chanin et al. (2024) 定义的吸收率之间的对应关系已在第 4.1 节系统验证。在验证之前，所有基于碰撞率的结论应被理解为对'跨架构激活重叠模式'的描述。此外，UAD 作为无监督方法，其在无 ground truth 场景下的性能无法被直接验证——这是无监督检测的内在限制。"

---

## 五、风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| 碰撞率-吸收率相关性弱（r < 0.5） | 中 | 高 | 提前在 pilot 中验证；若弱则重命名指标为 CAAC |
| GemmaScope API 不可用 | 中 | 中 | 准备 GPT-2 Small SAE 作为备选 |
| UAD 在 GemmaScope 上性能下降 | 低 | 高 | 增加超参数调优；报告不同配置结果 |
| 审稿人认为贡献单一 | 中 | 中 | 强调 UAD 填补的是最关键的研究空白；跨学科 Discussion 提升深度 |
| 统计功效仍不足 | 中 | 高 | 事前功效分析；明确声明探索性分析 vs 假设检验 |

---

## 六、与上一轮 lessons learned 的对应

| Lessons Learned 问题 | 本提案解决方案 |
|---------------------|--------------|
| 死特征 89-99% | 完全放弃训练-SAE，使用预训练 SAE |
| 架构比较混杂 | 降级为观察性分析或补充材料 |
| 统计功效不足 | 事前功效分析；放弃 p 值，改用效应量+置信区间 |
| 代理指标未验证 | P0 实验：碰撞率-吸收率相关性验证 |
| 论文范围过度扩张 | 聚焦 UAD + DFDA，H1-H4 降级 |
| Abstract 过度宣称 | 严格匹配正文，所有声明有实验支持 |
| 术语漂移 | 全文统一：碰撞率=代理指标，吸收=Chanin 定义 |
| Gemma-2-2B 回退 | 明确所有结果仅基于 GPT-2 Small，Gemma 作为未来工作 |

---

## 七、总结

本提案的核心策略是**聚焦与诚实**：

1. **聚焦 UAD**：这是唯一被充分验证、在当前约束下完全可行的核心贡献
2. **验证代理指标**：碰撞率-吸收率相关性验证是论文可信度的底线
3. **诚实披露局限**：代理指标假设、GPT-2 局限、误报率、无 ground truth 验证悖论
4. **跨学科提升深度**：信息论/相变/认知科学框架融入 Discussion，提升论文理论层次
5. **放弃不可行方向**：训练-SAE、因果架构比较、纯理论证明

**最终定位**：一篇以 UAD 为核心方法贡献、以诚实的方法论讨论和局限性披露为特色、以跨学科理论框架为深度提升的论文。这不是"我们尝试了这些方法，大部分没成功"的负面结果报告，而是"我们开发了一个真正有用的工具，并诚实地讨论了它的边界"的建设性论文。


## 小型实验真实反馈（必须基于这些证据修正 idea，不能忽略负结果）
# Pilot Summary - Iteration 3

> Date: 2026-04-28
> Task: p1_collision_proxy_validation
> Status: COMPLETED

## P1: Collision Rate - Absorption Rate Proxy Validation

### Results

| Metric | Value | Pass Threshold | Status |
|--------|-------|----------------|--------|
| Spearman r | 0.711 | >= 0.3 | PASS |
| Pearson r | 0.733 | - | - |
| Bootstrap 95% CI | [0.219, 0.887] | Does not include 0 | PASS |
| GT pairs detected | 10/10 | >= 5 | PASS |
| Runtime | 25.3s | < 15 min | PASS |

### Key Findings

1. **Strong positive correlation**: Collision rate (cosine similarity of activation profiles) correlates with true absorption rate (Jaccard similarity of top-K feature sets) at r=0.71 (Spearman) and r=0.73 (Pearson).

2. **All 10 vowel pairs detected**: All possible vowel pairs (a-e, a-i, a-o, a-u, e-i, e-o, e-u, i-o, i-u, o-u) share at least one top-10 SAE feature, confirming absorption clustering.

3. **Bootstrap CI excludes zero**: The 95% bootstrap confidence interval [0.219, 0.887] does not include zero, supporting the directional relationship.

4. **Important caveat**: The top-5 features are nearly identical across all 26 letters (feature 11746 dominates), suggesting this SAE has a strong "first-letter" super-feature. This means the ground truth definition (shared top features = absorption) may be too permissive. However, the correlation between collision rate and absorption rate is still meaningful because collision rates vary even when all pairs are "absorbed" (ranging from 0.824 to 0.964).

### Methodology Notes

- **v2 redesign**: Used distribution-based approach (cosine similarity of full activation profiles) instead of single top feature per letter, which avoided the degenerate case where all letters mapped to the same feature.
- **Ground truth**: Defined as Jaccard overlap of top-10 features per letter. All vowel pairs share features, giving a non-degenerate absorption rate distribution (0.538 to 0.667).
- **Collision rate**: Cosine similarity of mean activation profiles across all 24,576 SAE features.

### Implications for Full Experiments

- **GO for full experiments**: P1 passes all criteria. Collision rate shows promise as a proxy for absorption rate.
- **Caution**: The dominance of feature 11746 suggests the SAE may have a single "first-letter" super-feature rather than distributed representations. This should be noted in the paper.
- **Recommendation**: Proceed with P2 (UAD reproducibility) and P3 (random baseline).

---

## P2: UAD Reproducibility Validation

### Result: FAIL

### Key Metrics

| Metric | Value | Pass Threshold | Status |
|--------|-------|----------------|--------|
| Precision | 0.0% | - | - |
| Recall | 14.3% | >= 80% | **FAIL** |
| F1 | 0.0 | >= 50% | **FAIL** |
| Detected Pairs | 4155 | - | - |
| Ground Truth Pairs | 7 | - | - |
| True Positives | 1 | - | - |
| False Positives | 4154 | - | - |
| Runtime | 149.3s | < 15 min | PASS |

### Ground Truth

Using number word hierarchy (one, two, three, four, five, six, seven, eight) on GPT-2 Small layer 8 with gpt2-small-res-jb SAE:
- Number features detected: 8 primary features
- Absorption features identified: 7 features that activate on multiple numbers
- Feature absorption pairs: 7 pairs

### Root Cause Analysis

UAD fails because of a **fundamental mismatch** between its detection mechanism and the nature of absorption:

1. **UAD detects co-occurrence**: It clusters features that fire on the SAME tokens (co-occur in the corpus).

2. **Absorption features are mutually exclusive**: They fire on DIFFERENT tokens representing different child concepts. For example:
   - Feature 11513 fires only on "three"
   - Feature 24189 fires only on "four", "five", "six", "seven", "eight"
   - These features never activate on the same token

3. **Co-occurrence is near zero**: Phi coefficients between absorption features on general text (OpenWebText) are near zero or negative.

4. **Clustering separates them**: With 50 clusters on 504 features, GT absorption features are placed in different clusters.

### Evidence

Token-level activation for number sequence "one two three four five six seven eight":
```
Token        F11513 F12413 F22971 F24189
one             0.0   15.3    0.0    0.0
two             0.0    0.0   24.2    0.0
three          29.4    0.0    0.0    0.0
four            0.0    0.0    0.0   18.9
five            0.0    0.0    0.0   14.9
six             0.0    0.0    0.0   16.6
seven           0.0    0.0    0.0   14.3
eight           0.0    0.0    0.0   15.9
```

Features are completely mutually exclusive at the token level.

### Implications

1. **UAD detects features that fire TOGETHER** (co-occurring), not features that fire on mutually exclusive instances of a parent concept.

2. **This is a fundamental limitation** of co-occurrence-based approaches for hierarchical absorption detection.

3. **UAD may work for**: detecting synonym features or contextually related features that co-occur frequently.

4. **For hierarchical absorption**: A different approach is needed - one that measures semantic similarity or causal dependence rather than co-occurrence.

### Honest Assessment

This is a **valid negative result**. The UAD method as implemented cannot detect the type of absorption defined by Chanin et al. (2024) in the gpt2-small-res-jb SAE. The method's reliance on co-occurrence clustering is fundamentally incompatible with the mutually exclusive nature of absorption features.

### Recommendations

1. **Report this as a negative result** in the paper
2. **Discuss the limitation** of co-occurrence-based approaches
3. **Consider alternative approaches** for absorption detection:
   - Semantic similarity between feature directions (cosine similarity of decoder weights)
   - Causal intervention (zeroing child features and measuring parent recovery)
   - Activation patching between related concepts
4. **Pivot the paper focus** to "Why co-occurrence clustering cannot detect feature absorption"

---

## P3: Random Baseline Validation

### Result: FAIL

### Key Metrics

| Metric | Value | Pass Threshold | Status |
|--------|-------|----------------|--------|
| UAD F1 | 0.00048 | - | - |
| Global Random F1 | 0.00011 | < 0.05 | **PASS** |
| Same-Cluster Random F1 | 0.00048 | - | - |
| UAD - Global Random | 0.00037 | >= 0.3 | **FAIL** |
| UAD == Same-Cluster Random | Yes | No | **FAIL** |
| Runtime | 151.3s | < 10 min | PASS |

### Baseline Comparisons

| Baseline Type | Mean F1 | Std F1 | Mean Precision | Mean Recall | Mean TP |
|---------------|---------|--------|----------------|-------------|---------|
| UAD (actual) | 0.00048 | - | 0.00024 | 0.143 | 1.0 |
| Global Random (analytical) | 0.00011 | ~0 | 0.00006 | 0.033 | ~0.3 |
| Global Random (empirical) | 0.00012 | 0.00021 | 0.00006 | 0.034 | 0.24 |
| **Same-Cluster Random** | **0.00048** | **0** | **0.00024** | **0.143** | **1.0** |
| Frequency-Weighted Random | 0.0 | 0 | 0.0 | 0.0 | 0.0 |

### Critical Finding: UAD == Same-Cluster Random

**UAD F1 (0.00048) is IDENTICAL to same-cluster random F1 (0.00048).** This means:

1. **The clustering step provides ZERO value**: Randomly sampling pairs from within the same clusters achieves exactly the same performance as UAD's full pipeline (co-occurrence matrix + hierarchical clustering + phi filtering + dead feature filtering).

2. **All UAD's complexity is irrelevant**: The phi coefficient filtering, dead feature filtering, and specificity checks do not improve over random sampling within clusters.

3. **UAD has no actual contribution**: For hierarchical absorption detection, UAD is statistically indistinguishable from a trivial random baseline.

### Why Same-Cluster Random Equals UAD

The UAD pipeline detects 4155 candidate pairs across 50 clusters. When all 4155 cluster-internal pairs are considered, randomly selecting 4155 pairs from the same cluster structure yields the exact same 1 true positive out of 7 ground truth pairs. The clustering happens to place 1 GT pair in the same cluster by chance, and neither UAD's filtering nor random sampling can distinguish it from the other 4154 false positive pairs.

### Pass Criteria Assessment

| Criterion | Required | Actual | Status |
|-----------|----------|--------|--------|
| Global random F1 < 0.05 | < 0.05 | 0.00011 | **PASS** |
| UAD F1 - random F1 >= 0.3 | >= 0.3 | 0.00037 | **FAIL** |
| **OVERALL** | Both pass | - | **FAIL** |

### Implications

1. **UAD provides no meaningful advantage over random**: The tiny F1 difference (0.00037) is 3 orders of magnitude below the required threshold (0.3).

2. **The method is not viable**: Even if UAD were slightly better than random, an F1 of ~0.0005 is practically useless for any real application.

3. **Confirms P2 finding**: P2 showed UAD fails due to fundamental co-occurrence mismatch. P3 confirms this failure is not salvageable - the method has no discriminative power.

---

## Overall Pilot Assessment

### Pass/Fail Summary

| Pilot | Status | Key Finding |
|-------|--------|-------------|
| P1 (Proxy Validation) | **PASS** | Collision rate correlates with absorption (r=0.71) |
| P2 (UAD Reproducibility) | **FAIL** | UAD cannot detect hierarchical absorption |
| P3 (Random Baseline) | **FAIL** | UAD indistinguishable from random |

### Decision: PIVOT

**All three pilots together tell a clear story:**

1. **P1 validates the proxy metric**: Collision rate is a reasonable proxy for absorption rate. This is a genuine empirical finding.

2. **P2 reveals UAD's fundamental flaw**: Co-occurrence clustering cannot detect features that are mutually exclusive at the token level.

3. **P3 confirms UAD has no value**: The method is statistically indistinguishable from random sampling within clusters.

**The paper should pivot to:**

> "Why Co-occurrence Clustering Cannot Detect Feature Absorption: A Negative Result and Conceptual Analysis"

**Core contributions of the pivoted paper:**
1. Validation that collision rate is a proxy for absorption rate (positive result)
2. Demonstration that co-occurrence-based methods fail for hierarchical absorption (negative result)
3. Conceptual analysis of why absorption features are mutually exclusive (theoretical insight)
4. Discussion of alternative approaches (semantic similarity, causal intervention)

**This is an honest, valuable negative result.** The SAE community needs to know that co-occurrence-based approaches are not suitable for absorption detection, so future work can focus on more promising directions.


## 小型实验结构化信号（供你提炼 go/no-go / confidence / hypothesis status）
{
  "overall_recommendation": "PIVOT",
  "selected_candidate_id": null,
  "candidates": [
    {
      "candidate_id": "uad_cooccurrence",
      "go_no_go": "NO_GO",
      "confidence": 0.95,
      "supported_hypotheses": [],
      "failed_assumptions": [
        "H1: UAD can detect absorption via co-occurrence clustering",
        "H2: Absorption features co-occur in natural text",
        "H3: UAD significantly outperforms random baseline"
      ],
      "key_metrics": {
        "precision": 0.0,
        "recall": 0.143,
        "f1": 0.0,
        "true_positives": 1,
        "false_positives": 4154,
        "ground_truth_pairs": 7
      },
      "notes": "UAD fails because absorption features are mutually exclusive at the token level. They fire on different tokens representing different child concepts and never co-occur. P3 confirms UAD is statistically indistinguishable from same-cluster random baseline (F1=0.00048 vs 0.00048). This is a fundamental limitation of co-occurrence-based approaches for hierarchical absorption detection."
    }
  ],
  "pilot_results": {
    "p1_collision_proxy": {
      "status": "PASS",
      "spearman_r": 0.711,
      "pearson_r": 0.733,
      "bootstrap_ci": [0.219, 0.887],
      "gt_pairs_detected": "10/10"
    },
    "p2_uad_reproduce": {
      "status": "FAIL",
      "precision": 0.0,
      "recall": 0.143,
      "f1": 0.0,
      "true_positives": 1,
      "false_positives": 4154,
      "false_negatives": 6,
      "ground_truth_pairs": 7
    },
    "p3_random_baseline": {
      "status": "FAIL",
      "uad_f1": 0.00048,
      "global_random_f1": 0.00011,
      "same_cluster_random_f1": 0.00048,
      "uad_minus_global_random": 0.00037,
      "uad_better_than_global": true,
      "uad_better_than_same_cluster": false,
      "pass_criteria_global_f1_lt_0.05": true,
      "pass_criteria_uad_minus_random_ge_0.3": false,
      "notes": "UAD F1 is identical to same-cluster random baseline. The clustering step provides zero value. UAD has no actual contribution for hierarchical absorption detection."
    }
  },
  "analysis": {
    "finding": "UAD fails to detect hierarchical absorption in pre-trained SAE",
    "root_cause": "Absorption features are mutually exclusive at the token level - they fire on different tokens representing different child concepts. UAD's co-occurrence clustering cannot detect features that never co-occur.",
    "evidence": {
      "token_level_mutual_exclusivity": "Feature 11513 fires only on 'three', feature 24189 only on 'four'-'eight'. They never activate on the same token.",
      "cooccurrence_phi": "Phi coefficient between absorption features is near zero or negative on general text (OpenWebText).",
      "uad_clustering": "With 50 clusters on 504 features, GT absorption features are placed in different clusters.",
      "true_positives": 1,
      "false_positives": 4154,
      "same_cluster_random_equivalence": "UAD F1 (0.00048) == Same-cluster random F1 (0.00048)"
    },
    "implications": [
      "UAD detects features that fire TOGETHER (co-occurring), not features that fire on mutually exclusive instances of a parent concept.",
      "This is a fundamental limitation of co-occurrence-based approaches for hierarchical absorption detection.",
      "UAD may work for detecting synonym features or contextually related features that co-occur frequently.",
      "For hierarchical absorption, a different approach is needed - one that measures semantic similarity or causal dependence rather than co-occurrence."
    ]
  },
  "recommendations": [
    "Report this as a negative result in the paper",
    "Discuss the limitation of co-occurrence-based approaches",
    "Consider alternative approaches: semantic similarity of decoder weights, causal intervention, activation patching",
    "Pivot paper focus to 'Why co-occurrence clustering cannot detect feature absorption'"
  ]
}


## 上一轮 validation 决策意见
# Idea Validation Decision

## Pilot Evidence Summary

Only one candidate was tested in this pilot round: **UAD with co-occurrence clustering** (`uad_cooccurrence`).

### P1: Collision Rate — Absorption Rate Proxy Validation (PASS)
- Spearman r = 0.711, Pearson r = 0.733
- Bootstrap 95% CI = [0.219, 0.887] (excludes zero)
- All 10 vowel pairs detected (10/10)
- **Verdict**: Collision rate is a valid proxy for absorption rate. This is a genuine positive finding.

### P2: UAD Reproducibility Validation (FAIL)
- Precision = 0.0%, Recall = 14.3%, F1 = 0.0
- True Positives = 1, False Positives = 4154, Ground Truth = 7 pairs
- **Verdict**: UAD completely fails to detect hierarchical absorption. Recall below 80% threshold by a massive margin.

### P3: Random Baseline Validation (FAIL)
- UAD F1 = 0.00048
- Same-cluster random F1 = 0.00048 (IDENTICAL)
- UAD minus global random = 0.00037 (required >= 0.3)
- **Verdict**: UAD is statistically indistinguishable from random sampling within clusters. The clustering step provides zero value.

### Root Cause
Absorption features are **mutually exclusive at the token level** — they fire on different tokens representing different child concepts. UAD's co-occurrence clustering detects features that fire TOGETHER, not features that fire on mutually exclusive instances of a parent concept. This is a **fundamental mismatch**, not a tunable parameter issue.

---

## Decision Matrix

### Candidate: `uad_cooccurrence`

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | F1=0.0, 4154 false positives, 1 true positive out of 7 ground truth pairs. UAD F1 identical to same-cluster random baseline (0.00048). |
| Hypothesis survival | 0.25 | 1 | All three hypotheses falsified: H1 (UAD detects absorption via co-occurrence) — FAIL; H2 (absorption features co-occur) — FAIL (phi ~ 0); H3 (UAD outperforms random) — FAIL (UAD == same-cluster random). |
| Path to full result | 0.20 | 1 | No credible path. The co-occurrence mechanism is fundamentally incompatible with token-level mutual exclusivity of absorption features. No parameter tuning can fix this. |
| Novelty | 0.15 | 2 | The negative result itself has novelty value: demonstrating why co-occurrence clustering cannot detect absorption is a useful contribution to the SAE community. |
| Resource efficiency | 0.10 | 1 | Any additional GPU budget on this candidate is wasted. The method has been shown to have zero discriminative power. |

**Weighted Score = 1.15**

---

## Decision Rationale

The decision is **PIVOT**. The evidence is overwhelming and conclusive:

1. **UAD F1 = Same-Cluster Random F1** (0.00048). This is the smoking gun. The entire UAD pipeline — co-occurrence matrix, hierarchical clustering, phi filtering, dead feature filtering — provides exactly zero value over randomly sampling pairs from within the same clusters. This is not a "needs more tuning" situation; it is a fundamental architectural mismatch.

2. **Token-level mutual exclusivity** is the root cause. Absorption features fire on DIFFERENT tokens (e.g., feature 11513 only on "three", feature 24189 only on "four"-"eight"). They never co-occur. UAD clusters features that co-occur. The method literally cannot detect the phenomenon it targets.

3. **P1's positive result is separable**. The collision proxy validation (r=0.71) is a genuine finding, but it does not rescue UAD. It merely validates that collision rate correlates with absorption rate — a useful observation, but not a detection method.

4. **The proposal.md is now obsolete**. It was written before pilot execution and assumed UAD would work (citing iteration 1's F1=0.704 on a different, likely flawed setup). The pilot has falsified that assumption.

5. **No backup candidates exist** in `candidates.json` (file not present). A fresh ideation round is needed.

---

## Next Actions

1. **Do NOT proceed with any full experiments** (E1-E6 from task_plan.json). All are predicated on UAD working, which it does not.

2. **Report the negative result honestly**. The pilot has produced a valuable finding: co-occurrence clustering cannot detect hierarchical feature absorption in SAEs. This should be documented as a negative result.

3. **Preserve P1's positive finding**. The collision rate proxy validation (r=0.71) is a genuine empirical contribution and should be retained.

4. **Pivot paper focus** to: "Why Co-occurrence Clustering Cannot Detect Feature Absorption: A Negative Result and Conceptual Analysis"

5. **Start fresh ideation** for a new candidate approach that does NOT rely on co-occurrence. Potential directions (from pilot recommendations):
   - Semantic similarity between feature decoder weights
   - Causal intervention (zeroing child features, measuring parent recovery)
   - Activation patching between related concepts
   - Direct analysis of feature geometry in activation space

6. **Consider whether this project should continue** or if the negative result should be written up and the project concluded. The spec's goal was to detect absorption; if the main approach is fundamentally flawed, the project may need a broader pivot.

SELECTED_CANDIDATE: none
CONFIDENCE: 0.95
DECISION: PIVOT


## 上一轮 validation 结构化决策
{
  "decision": "PIVOT",
  "selected_candidate_id": null,
  "confidence": 0.95,
  "candidate_scores": {
    "uad_cooccurrence": {
      "weighted_score": 1.15,
      "verdict": "PIVOT",
      "component_scores": {
        "pilot_signal_strength": {"weight": 0.30, "score": 1, "raw": "F1=0.0, UAD==random"},
        "hypothesis_survival": {"weight": 0.25, "score": 1, "raw": "H1-H3 all falsified"},
        "path_to_full_result": {"weight": 0.20, "score": 1, "raw": "Fundamental mismatch, no tuning can fix"},
        "novelty": {"weight": 0.15, "score": 2, "raw": "Negative result has some community value"},
        "resource_efficiency": {"weight": 0.10, "score": 1, "raw": "Any further GPU spend is wasted"}
      }
    }
  },
  "reasons": [
    "UAD F1 (0.00048) is identical to same-cluster random F1 (0.00048) — the method has zero discriminative power",
    "Absorption features are mutually exclusive at the token level; UAD's co-occurrence clustering detects features that fire together, not features that fire on mutually exclusive instances",
    "All three hypotheses (H1-H3) were falsified by the pilot evidence",
    "The failure is not salvageable through parameter tuning — it is a fundamental architectural mismatch",
    "P1's collision proxy validation (r=0.71) is a separable positive finding but does not rescue UAD as a detection method"
  ],
  "next_actions": [
    "Halt all full experiments (E1-E6) — they depend on UAD working",
    "Document the negative result: co-occurrence clustering cannot detect hierarchical absorption",
    "Preserve P1's positive finding (collision rate proxy validation, r=0.71)",
    "Pivot paper focus to negative result + conceptual analysis",
    "Start fresh ideation for non-co-occurrence approaches (semantic similarity, causal intervention, activation patching)",
    "Evaluate whether to continue project or write up negative result and conclude"
  ],
  "dropped_candidates": ["uad_cooccurrence"],
  "pivot_trigger": "UAD F1 identical to random baseline — fundamental co-occurrence mismatch",
  "salvageable_findings": [
    {
      "finding": "Collision rate correlates with absorption rate (Spearman r=0.711)",
      "value": "Genuine empirical contribution — validates proxy metric"
    },
    {
      "finding": "Absorption features are mutually exclusive at token level",
      "value": "Conceptual insight — explains why co-occurrence methods fail"
    }
  ],
  "obsolete_artifacts": [
    "current/idea/proposal.md — written before pilot, assumes UAD works"
  ]
}
