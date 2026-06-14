

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

**Research Topic**: 研究稀疏自编码器（SAE）中的特征吸收（feature absorption）现象：系统分析和量化其成因、规律及对可解释性的影响
**Survey Date**: 2026-04-13 (updated)
**arXiv Search Keywords**: "feature absorption" "feature splitting" sparse autoencoders; "sparse autoencoder" "mechanistic interpretability" features superposition; SAEBench evaluation sparse autoencoders; Gated SAE JumpReLU architecture; Matryoshka SAE hierarchical; feature hedging correlated; "survey sparse autoencoders" LLM; theoretical sparse autoencoders identifiable; SAE dark matter reconstruction; "feature manifolds" scaling; KronSAE features correlation; unified theory sparse dictionary learning; transcoder interpretability absorption; SynthSAEBench synthetic hierarchy; adaptive temporal masking SAE absorption; CE-Bench contrastive evaluation SAE; feature sensitivity sparse autoencoder
**Web Search Keywords**: sparse autoencoder feature absorption interpretability 2025; SAE feature absorption Chanin 2024 arxiv 2409.14507; Matryoshka SAE hierarchical feature absorption; SAE feature absorption solutions mitigation OrtSAE masked regularization; SAE superposition toy model Anthropic survey; sparse autoencoder absorption sparsity penalty L1 L0 hierarchical theoretical; feature anchoring unified theory SDL absorption dead neurons; KronSAE Kronecker factorization absorption metrics; transcoders beat SAEs interpretability; DeepMind negative results SAE safety

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant unsupervised tool in mechanistic interpretability for decomposing the polysemantic, superposed activations of large language models (LLMs) into approximately monosemantic, human-interpretable feature directions. The foundational motivation comes from Anthropic's "Toy Models of Superposition" (2022) and the subsequent "Towards Monosemanticity" (2023), which established that neural networks encode far more features than they have neurons by representing features as overlapping linear directions—a phenomenon called superposition. SAEs address this by projecting activations into an overcomplete sparse basis via a sparsity-inducing objective (L1 penalty or TopK activation), with the hope of recovering the underlying "true" monosemantic features.

The field advanced rapidly through 2024–2025: Anthropic and OpenAI scaled SAEs to frontier models (Claude 3 Sonnet, GPT-4), the community developed several improved architectures (Gated SAE, JumpReLU SAE, TopK SAE, Matryoshka SAE, OrtSAE), and comprehensive benchmarks (SAEBench, CE-Bench) emerged to standardize evaluation. However, alongside these successes came a growing body of negative results that challenge whether SAEs robustly recover the mechanistically meaningful features they promise. Three failure modes have attracted particular attention: (1) **feature absorption**—where a general feature silently fails to fire because a more specific co-occurring feature encodes the same information; (2) **feature hedging**—where correlated features are merged due to insufficient SAE width; and (3) **feature inconsistency**—where independent training runs converge to different feature sets. Feature absorption is arguably the most fundamental of these, as it creates a false sense of interpretability: the SAE appears to have learned a clean monosemantic latent, but that latent has systematic "holes" in its recall.

The current state of the field (early 2026) is one of productive tension: SAEs remain the most tractable approach to large-scale mechanistic interpretability, but their theoretical guarantees are weak, and several recent papers show that even well-trained SAEs may only marginally outperform random baselines on downstream tasks. Notably, DeepMind's safety research team publicly deprioritized SAE research in 2025 after finding that dense linear probes dramatically outperform SAE probes on safety-relevant downstream tasks (e.g., harmful intent detection), with feature absorption being a key culprit. The community is actively developing architectural solutions (Matryoshka SAEs, OrtSAE, KronSAE, masked regularization), alternative paradigms (transcoders, crosscoders), and theoretical frameworks (identifiability analysis, feature anchoring, MDL-based training) to address these shortcomings. A unified theoretical framework (arXiv:2512.05534) now casts all major sparse dictionary learning methods as a single piecewise biconvex optimization problem and provides principled explanations for absorption—though feature anchoring, its proposed remedy, has only been validated on synthetic benchmarks. Feature absorption specifically sits at the intersection of theory (caused by sparsity optimization under feature hierarchy) and practice (undermines causal circuit analysis and safety-relevant feature detection).

---

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | arXiv:2409.14507 (NeurIPS 2025) | 2024 | Defines and characterizes feature absorption; toy model proof that hierarchical features cause absorption; proposes absorption rate metric; validates on hundreds of SAEs (Gemma Scope, Llama 3.2, Qwen2); finds absorption in every tested SAE | Focuses on first-letter spelling task; absorption metric requires known probe directions; does not propose a solution |
| 2 | Toy Models of Superposition | transformer-circuits.pub (2022) | 2022 | Foundational: demonstrates neural nets encode more features than neurons via superposition; introduces the linear representation hypothesis as motivation for SAEs | Synthetic toy setting; does not address feature absorption |
| 3 | Towards Monosemanticity: Decomposing Language Models With Dictionary Learning | transformer-circuits.pub (2023) | 2023 | First large-scale demonstration that SAEs decompose MLP activations into interpretable monosemantic features on a 1-layer transformer | Small model; does not address absorption or failure modes |
| 4 | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | transformer-circuits.pub (2024) | 2024 | Shows SAEs scale to frontier models; finds safety-relevant features (deception, sycophancy, bias); introduces concept of feature families | No systematic absorption analysis; limited evaluation methodology |
| 5 | Scaling and Evaluating Sparse Autoencoders (TopK SAE) | arXiv:2406.04093 (OpenAI, ICLR 2025) | 2024 | Proposes k-sparse SAEs with clean scaling laws; introduces new evaluation metrics; scales to 16M latent SAE on GPT-4 | Does not specifically address feature absorption; evaluation metrics focus on proxy measures |
| 6 | Improving Dictionary Learning with Gated Sparse Autoencoders | arXiv:2404.16014 | 2024 | Proposes Gated SAE that decouples feature detection from magnitude estimation; reduces L1-induced shrinkage bias; Pareto improvement | Does not directly address feature absorption; still susceptible to hierarchical feature pathology |
| 7 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders | arXiv:2407.14435 | 2024 | Introduces JumpReLU activation for SOTA reconstruction at given sparsity; trains L0 directly via STE; achieves Pareto improvement on Gemma 2 9B | Does not specifically mitigate absorption |
| 8 | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | arXiv:2503.17547 (ICML 2025) | 2025 | Trains nested SAE dictionaries of increasing size simultaneously; organizes features hierarchically with smaller dicts learning general concepts; reduces absorption; superior on sparse probing and concept erasure | Minor reconstruction tradeoff; inner levels suffer from feature hedging |
| 9 | Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders | arXiv:2505.11756 | 2025 | Identifies "feature hedging" as a complementary failure mode to absorption; theoretically and empirically shows narrow SAEs merge correlated features; wider SAEs suffer less; proposes balanced Matryoshka SAE | Focuses on hedging rather than absorption; shows Matryoshka SAEs trade absorption for hedging |
| 10 | OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features | arXiv:2509.22033 | 2025 | Enforces orthogonality via pairwise cosine similarity penalty; reduces absorption by 65%, composition by 15%; discovers 9% more distinct features; linear computational overhead | Does not fully eliminate absorption; focuses on structural orthogonality rather than hierarchy |
| 11 | Improving Robustness In Sparse Autoencoders via Masked Regularization | arXiv:2604.06495 | 2026 | Proposes masking-based regularization to disrupt co-occurrence patterns during training; reduces absorption and improves OOD robustness across SAE architectures | Very recent (April 2026); limited to token masking intervention |
| 12 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability | arXiv:2503.09532 | 2025 | Establishes 8-metric evaluation suite including absorption task; 200+ open-source SAEs across 8 architectures; reveals proxy metrics do not reliably predict practical performance | Absorption metric is one of 8 and may not fully capture the phenomenon |
| 13 | Decomposing The Dark Matter of Sparse Autoencoders | arXiv:2410.14670 | 2024 | Investigates unexplained SAE reconstruction error ("dark matter"); shows ~50% of error vector is linearly predictable from input; links to scaling behavior | Does not directly address absorption; provides context for SAE failure modes |
| 14 | On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy | arXiv:2506.15963 | 2025 | First closed-form theoretical analysis showing SAEs generally fail to recover ground truth monosemantic features unless features are extremely sparse; proposes reweighted SAE (WSAE) | Theoretical bound may be loose; WSAE not evaluated on absorption specifically |
| 15 | Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders | arXiv:2506.14002 | 2025 | Proposes bias-adaptation training with theoretical recovery guarantees under a statistical model; Group Bias Adaptation (GBA) variant for LLMs | Guarantees rely on restrictive generative assumptions; does not model feature hierarchy |
| 16 | Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? | arXiv:2602.14111 | 2026 | Shows SAEs recover only 9% of true features in synthetic settings; random baselines match fully-trained SAEs on interpretability, sparse probing, and causal editing tasks | Extremely critical; may underestimate SAE utility in practice; does not address absorption |
| 17 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | arXiv:2505.20254 | 2025 | Shows SAE features are inconsistent across training runs; proposes PW-MCC metric; TopK SAEs achieve 0.80 consistency | Does not address absorption specifically; focuses on inter-run consistency |
| 18 | Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures | arXiv:2506.01197 | 2025 | Explicitly models semantic hierarchy in SAE; improves reconstruction and interpretability; computationally efficient | Recent; absorption evaluation not primary focus |
| 19 | From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders (HSAE) | arXiv:2602.11881 | 2026 | Jointly learns SAE series and parent-child relationships; structural constraint loss and random feature perturbation; consistently recovers semantically meaningful hierarchies | Very recent; relationship to absorption not fully characterized |
| 20 | Mathematical Models of Computation in Superposition | arXiv:2408.05451 | 2024 | Provides theoretical models for computation in superposition; shows networks can emulate circuits of width O(d^1.5) with width d using superposition | Foundational theory; does not address absorption directly |
| 21 | A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima | arXiv:2512.05534 | 2024 | First unified theoretical framework casting all SDL variants (SAEs, transcoders, crosscoders) as a piecewise biconvex optimization problem; provides principled explanations for feature absorption and dead neurons; proposes feature anchoring to restore identifiability | Theoretical framework; feature anchoring validated on synthetic benchmarks, real LLM validation limited |
| 22 | Transcoders Beat Sparse Autoencoders for Interpretability | arXiv:2501.18823 | 2025 | Shows transcoder features are significantly more interpretable than SAE features; proposes skip transcoders with affine skip connection; achieves Pareto improvement over SAEs on reconstruction and interpretability; reduces absorption as a side effect of input-output mapping | Does not directly study absorption; transcoder paradigm incompatible with existing SAE evaluation ecosystem |
| 23 | KronSAE: Kronecker Factorization Improves Efficiency and Interpretability of Sparse Autoencoders | arXiv:2505.22255 | 2025 | Proposes Kronecker product factorization of SAE latents; introduces mAND activation for logical AND-like gating; reduces mean absorption fraction and full-absorption score across all sparsity levels; competitive with TopK at lower compute | Architecture-specific; absorption reduction mechanism tied to factorization structure; not evaluated on SAEBench |
| 24 | SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data | arXiv:2602.14687 | 2026 | Provides large-scale synthetic data model with realistic feature characteristics including correlation, hierarchy, superposition, and Zipfian firing distributions; logistic probes achieve 0.974 F1 while best SAEs substantially underperform; enables controlled study of feature hierarchy and absorption | Very recent (Feb 2026); synthetic setting may not capture all real LLM characteristics |
| 25 | Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training (ATM) | arXiv:2510.08855 | 2025 | Introduces Adaptive Temporal Masking (ATM) that dynamically adjusts feature selection via importance scoring of activation magnitudes, frequencies, and reconstruction contributions; achieves mean absorption score 0.0068 vs. TopK 0.1402 and JumpReLU 0.0114 on Gemma-2-2B; strong performance on bias detection and sentiment analysis | Per-latent importance tracking adds training overhead; evaluated only on Gemma-2-2B |
| 26 | CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark of General Interpretability of Sparse Autoencoders | arXiv:2509.00691 (BlackboxNLP 2025) | 2025 | Fully LLM-free contrastive benchmark built on structured story pairs differing in targeted semantic attribute; >70% Spearman correlation with SAEBench; interpretability scores increase with latent width; robust across layer types (attention, MLP, residual) | Does not directly measure feature absorption; benchmarks overall interpretability not absorption specifically |
| 27 | Measuring Sparse Autoencoder Feature Sensitivity | arXiv:2509.23717 | 2025 | Frames feature absorption as a special case of poor feature sensitivity; develops scalable sensitivity evaluation methods; finds many interpretable features have poor sensitivity even when activation examples appear monosemantic; sensitivity measures a dimension of quality missed by existing evaluations | Does not propose new mitigation; sensitivity metric distinct from but related to absorption metric |

---

## 3. SOTA Methods and Benchmarks

### Current Best SAE Architectures (sparsity-fidelity frontier)
- **TopK SAE** (Gao et al., OpenAI 2024): Direct control of L0; achieves clean scaling laws; reference architecture; NOTE: significantly worsens feature absorption vs. L1 SAEs at low L0 (SAEBench finding)
- **JumpReLU SAE** (Rajamanoharan et al., DeepMind 2024): State-of-the-art reconstruction on Gemma 2 9B; trains L0 directly via STE; worst on absorption (trained longer → more absorption)
- **Gated SAE** (Rajamanoharan et al., DeepMind 2024): Decouples detection from magnitude; reduces shrinkage
- **Matryoshka SAE** (Bussmann et al., 2025): Best on SAEBench absorption, RAVEL, sparse probing, and spurious correlation removal; minor reconstruction penalty
- **OrtSAE** (Korznikov et al., 2025): Reduces absorption 65% by orthogonality penalty; best for disentanglement; linear computational overhead over vanilla SAEs
- **ATM SAE** (Li et al., 2025): Adaptive Temporal Masking with per-latent importance scoring; achieves absorption score 0.0068 vs. TopK 0.1402 and JumpReLU 0.0114 on Gemma-2-2B; best reported absorption scores
- **KronSAE** (2025): Kronecker product factorization of latents; reduces absorption fraction via structural correlation exploitation; lower parameter count than TopK
- **BatchTopK SAE**: Competitive on sparsity-fidelity frontier; baseline in SAEBench
- **Transcoders / Skip Transcoders** (Paulo et al., 2025): Maps MLP inputs to outputs; features more interpretable than SAEs; Pareto improvement on reconstruction + interpretability; reduces absorption as side effect of input-output paradigm

### Absorption Metric (Chanin et al., 2024)
The canonical absorption measurement procedure:
1. Find k feature splits for a first-letter feature using k-sparse probing
2. Identify false-negative tokens: all k split latents fail to activate, but LR probe correctly classifies
3. Run integrated-gradients ablation on false-negative tokens
4. Absorption detected if the highest-ablation-effect latent has cosine similarity > 0.025 with probe AND is ≥1.0 larger than the second-highest
5. Absorption rate = (absorbed letters) / (total letters with at least one false negative)

**Key empirical finding**: Absorption rate 15–35% across Gemma Scope 16k and 65k SAEs; wider and more sparse SAEs show higher absorption; no clear layer-wise pattern; present in Llama 3.2 1B, Qwen2 0.5B, and all architectures tested (L1 and TopK).

### Key Benchmarks
- **SAEBench** (Karvonen et al., 2025): 8 metrics, 200+ SAEs, interactive at neuronpedia.org/sae-bench; metrics include Absorption, RAVEL, Sparse Probing, Spurious Correlation Removal (SCR), Unlearning; key finding: proxy metrics (CE loss, sparsity) do not reliably predict practical performance
- **SynthSAEBench** (2026): Large-scale synthetic benchmark with realistic feature characteristics (hierarchy, correlation, Zipfian); ground-truth features known; reveals SAEs substantially underperform direct probes on feature recovery
- **Linear Representation Bench** (unified SDL theory paper, 2024): Synthetic benchmark for exposing SDL pathologies under full ground-truth access
- **Gemma Scope** (DeepMind): Open-source SAE suite on Gemma 2 2B / 9B, widths 1k–1M, all layers; primary evaluation target in absorption literature
- **CE-Bench** (Gulko et al., arXiv:2509.00691, BlackboxNLP 2025): Lightweight LLM-free contrastive benchmark on structured story pairs; >70% Spearman correlation with SAEBench; interpretability scores increase with latent width; code: https://github.com/Yusen-Peng/CE-Bench
- **Feature Sensitivity Metric** (Tian et al., arXiv:2509.23717, 2025): Measures how reliably a feature activates on texts similar to its activating examples; many interpretable features have poor sensitivity; frames absorption as special case of low sensitivity
- **SAELens** (Bloom et al.): Training and evaluation library; standard in the community

### Evaluation Models
- Primary: Gemma 2 2B, Gemma 2 9B (via Gemma Scope SAEs)
- Also evaluated: GPT-2, Llama 3.2 1B, Qwen2 0.5B, Pythia series

---

## 4. Identified Research Gaps

**Gap 1: No quantitative causal theory of absorption magnitude.**
Chanin et al. prove absorption occurs in the toy setting and provide an informal argument (sparsity gain from absorbing parent into child), but there is no closed-form prediction of how severe absorption will be as a function of SAE width, L0, feature hierarchy depth, or feature co-occurrence statistics. Such a theory would enable principled hyperparameter selection to minimize absorption.

**Gap 2: Absorption has only been studied on the first-letter spelling task.**
The canonical absorption evaluation (Chanin et al.) uses a narrow, controlled proxy task where the feature hierarchy (letter membership ⊃ specific token) is known in advance. It is unknown whether absorption rates generalize to semantically richer hierarchies (e.g., entity type ⊃ specific entity, sentiment ⊃ topic), to safety-relevant features (deception, bias), or to non-English settings. A systematic cross-domain absorption characterization does not exist.

**Gap 3: The relationship between absorption and other SAE failure modes is under-explored.**
Feature absorption, feature hedging (Chanin et al., 2025), feature inconsistency (Song et al., 2025), and "dark matter" (Engels et al., 2024) are described as distinct phenomena but may share underlying causes. No unified theoretical framework characterizes their joint occurrence or trade-offs.

**Gap 4: Mitigation methods have not been comprehensively compared under controlled conditions.**
Matryoshka SAE, OrtSAE, masked regularization, and EWG-SAE all claim to reduce absorption, but they were evaluated with different models, layers, and metrics. No head-to-head comparison exists on a standardized benchmark (e.g., SAEBench absorption task) to determine which approach achieves the best absorption–reconstruction–interpretability trade-off.

**Gap 5: The theoretical conditions under which absorption is avoidable are unclear.**
Chanin et al. note that any solution eliminating absorption will have worse L0 vs. variance-explained tradeoff (since absorption saves one L0 per parent-child pair). On the Limits of SAEs (Cui et al., 2025) shows SAEs generally fail to recover ground truth features unless features are extremely sparse. Together these suggest absorption may be inherent to sparsity optimization under feature hierarchy—but neither paper provides precise sufficient conditions for absorption-free recovery.

**Gap 6: Absorption in hierarchically rich domains (knowledge, reasoning) is unstudied.**
All absorption studies focus on syntactic-level features (first letter, token identity). Feature hierarchies in knowledge representation (country ⊃ city, species ⊃ individual animal) or reasoning (logical implication chains) may exhibit qualitatively different absorption patterns. Understanding absorption in these domains is important for the safety applications that motivate SAE research.

**Gap 7: No metric for absorption that does not require known probe directions.**
The current absorption metric requires training LR probes to identify the "should-have-fired" feature. This is only possible when the researcher knows which features to look for in advance. An unsupervised absorption detection method would dramatically expand the scope of absorption analysis.

**Gap 8: Feature anchoring's effectiveness specifically against absorption is uncharacterized.**
The unified SDL theory paper (arXiv:2512.05534) proposes feature anchoring to restore identifiability and improve feature recovery, but does not specifically quantify how it reduces absorption vs. other failure modes. Whether feature anchoring can be combined with existing architectures (Matryoshka, OrtSAE) for compounded benefit is unexplored.

**Gap 8b: The safety implications of absorption are poorly quantified.**
DeepMind's 2025 internal study found that 1-sparse SAE probes fail catastrophically at detecting harmful intent (while dense linear probes achieve near-perfect accuracy even OOD). However, no systematic analysis quantifies how absorption specifically drives this gap—i.e., how often safety-relevant features are absorbed into co-occurring context features. This link between absorption rate and downstream safety task performance is a critical missing piece for motivating absorption mitigation research.

---

## 5. Available Resources

### Open-Source Code
- **sae-spelling** (canonical absorption study): https://github.com/lasr-spelling/sae-spelling — Chanin et al. absorption experiments on first-letter task; absorption rate metric implementation
- **SAELens**: https://github.com/jbloomAus/SAELens — Standard SAE training and evaluation library used across all major papers
- **Gemma Scope**: https://huggingface.co/google/gemma-scope — 400+ pre-trained SAEs on Gemma 2 2B / 9B, primary evaluation target
- **SAEBench**: https://github.com/adamkarvonen/SAEBench + interactive at https://www.neuronpedia.org/sae-bench
- **OrtSAE**: Available from Korznikov et al. (2025) — implementation of orthogonality penalty
- **Neuronpedia**: https://www.neuronpedia.org — Interactive feature explorer for many Gemma Scope / EleutherAI SAEs

### Datasets
- **OpenWebText**: Standard SAE pre-training corpus (used in Chanin et al., Gao et al.)
- **Pile / RedPajama**: Alternative corpora for SAE training
- **Gemma Scope activation caches**: Pre-computed for efficient SAE training and evaluation
- **First-letter spelling task** (constructed): In-context learning prompts using template `{token} has the first letter: {letter}`; test set from alphabetical token lists

### Pre-trained Models and SAEs
- **Gemma Scope SAEs**: 400+ SAEs on Gemma 2 (1B, 2B, 9B, 27B), all layers, widths 1k–1M — HuggingFace: https://huggingface.co/google/gemma-scope
- **GPT-2 SAEs** (EleutherAI): Standard evaluation target for reproducibility
- **Llama 3.2 1B / 3B SAEs**: Available via SAELens
- **Claude 3 Sonnet SAEs** (Anthropic, internal)

---

## 6. Implications for Idea Generation

**Most promising directions for novel contributions:**

1. **Quantitative theory of absorption**: A mathematical model predicting absorption rate as a function of SAE configuration (width, L0, feature hierarchy statistics) would be highly impactful. The toy model in Chanin et al. is a starting point; extending to multi-level hierarchies and probabilistic feature co-occurrence patterns could yield clean scaling laws.

2. **Cross-domain absorption characterization**: Systematically measure absorption on semantically richer hierarchies (entity type hierarchies, knowledge taxonomies, safety-relevant features) using the existing sae-spelling metric framework. This would establish whether first-letter results generalize and identify which feature hierarchy properties predict absorption severity.

3. **Unified analysis of absorption + hedging**: Feature hedging (Chanin & Dulka, 2025) and absorption are both caused by SAE optimization under feature correlation but in opposite capacity regimes (too narrow → hedging; hierarchy → absorption). A unified theory and empirical characterization would close a significant conceptual gap.

4. **Unsupervised absorption detection**: Developing a method to detect absorbed features without requiring pre-specified probe directions is both theoretically interesting and practically important. Candidate approaches: meta-SAE decomposition of latents, leveraging decoder cosine similarity structure, or information-theoretic divergence metrics.

5. **Systematic mitigation comparison**: A controlled SAEBench-style comparison of Matryoshka SAE, OrtSAE, masked regularization, and EWG-SAE on the absorption task (and other SAEBench metrics) with matched compute and model settings would be a valuable empirical contribution.

**Additional promising directions (from newly found papers):**

6. **Feature anchoring for absorption mitigation**: The unified SDL theory paper proposes feature anchoring as a principled remedy for identifiability failure. Applying and evaluating feature anchoring specifically for absorption reduction on real LLM SAEs (vs. only synthetic benchmarks) is a concrete research opportunity.

7. **Linking absorption to safety task performance**: DeepMind found SAE probes fail badly at detecting harmful intent while dense probes succeed. Quantifying how absorption specifically drives this gap—by measuring absorption rate of safety-relevant features and correlating with downstream task performance—would provide the most compelling empirical case for absorption mitigation.

8. **KronSAE absorption metrics as a richer evaluation toolkit**: The three metrics from KronSAE (mean absorption fraction, full-absorption score, feature splits) are complementary to the Chanin et al. metric. Adopting all three metrics in a comprehensive absorption study would give a more complete picture of the phenomenon.

**Directions that appear saturated or risky:**
- Pure architecture comparisons without addressing absorption theoretically add limited insight
- Improving the sparsity-fidelity frontier (already well-studied: TopK, JumpReLU, Gated) without addressing interpretability failure modes
- Purely synthetic toy model studies without validation on real LLM SAEs
- Transcoder/crosscoder work as a primary focus (different paradigm that sidesteps rather than resolves absorption in SAEs)

**Cross-domain analogies with potential:**
- Hierarchical dictionary learning in signal processing / compressed sensing (multi-scale wavelet decompositions) may provide theoretical tools for analyzing feature hierarchy in SAEs
- Multi-label classification with label implication (where absorbing a specific label "absorbs" the general label) has been studied in the NLP literature and may provide relevant algorithmic solutions

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| sae-spelling (Chanin et al.) | High | MIT | Adopt | Canonical absorption metric implementation; directly reusable for measuring absorption rate on first-letter task and extensions |
| SAELens (Bloom et al.) | High | MIT | Adopt | Standard training + evaluation library; all major SAE architectures supported; large community |
| Gemma Scope SAEs (Google DeepMind) | High | Gemma ToS (non-commercial research) | Adopt | Pre-trained SAEs on 400+ configs; eliminates training cost for evaluation experiments |
| SAEBench (Karvonen et al.) | High | Apache 2.0 | Adopt | Standardized 8-metric evaluation; enables fair comparison with existing architectures |
| OrtSAE (Korznikov et al.) | Medium | check paper | Extend | Orthogonality regularization is architecturally simple to implement in SAELens; extend to study interaction with absorption |
| Matryoshka SAE (Bussmann et al.) | Medium | MIT | Extend | Nested training objective well-specified; extend with different hierarchy structures or combine with orthogonality penalty |
| Masked Regularization (Narayanaswamy et al., 2026) | Medium | check paper | Extend | Very recent; token masking during training is simple to implement; extend to study optimal masking strategies for absorption reduction |
| KronSAE (2025) | Medium | check paper | Extend | Kronecker factorization encoder is simple to implement on top of SAELens; provides three absorption metrics (mean absorption fraction, full-absorption score, feature splits) as additional evaluation toolkit |
| Unified SDL Theory + Linear Representation Bench (2512.05534) | Medium | check paper | Extend | Feature anchoring is a novel identifiability restoration technique; could be combined with existing architectures; synthetic benchmark useful for controlled absorption studies |
| SynthSAEBench (2602.14687) | High | check paper | Adopt | Provides controlled synthetic environment with known ground-truth feature hierarchies for studying absorption without reliance on first-letter proxy task |

Highlight: The **sae-spelling + SAELens + Gemma Scope** combination gives a complete pipeline from SAE training to absorption evaluation with minimal implementation overhead. All three are MIT-licensed (or equivalent research licenses) and have active community support. The absorption metric code in sae-spelling is directly reusable; the main extension needed is adapting it to feature hierarchies beyond first-letter spelling.

---

## References (Full Citations)

1. Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., & Bloom, J. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. *NeurIPS 2025*. arXiv:2409.14507. Code: https://github.com/lasr-spelling/sae-spelling

2. Elhage, N., Hume, T., Olsson, C., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*. https://transformer-circuits.pub/2022/toy_model/index.html

3. Bricken, T., Templeton, A., Batson, J., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/monosemantic-features

4. Templeton, A., Conerly, T., Marcus, J., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread*. https://transformer-circuits.pub/2024/scaling-monosemanticity/

5. Gao, L., Dupré la Tour, T., Tillman, H., et al. (2024). Scaling and evaluating sparse autoencoders. *ICLR 2025*. arXiv:2406.04093.

6. Rajamanoharan, S., Conmy, A., Smith, L., et al. (2024). Improving Dictionary Learning with Gated Sparse Autoencoders. arXiv:2404.16014.

7. Rajamanoharan, S., Lieberum, T., Sonnerat, N., et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. arXiv:2407.14435.

8. Bussmann, B., Nabeshima, N., Karvonen, A., & Nanda, N. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. *ICML 2025*. arXiv:2503.17547.

9. Chanin, D., Dulka, T., & Garriga-Alonso, A. (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756.

10. Korznikov, A., Galichin, A., Dontsov, A., Rogov, O., Tutubalina, E., & Oseledets, I. (2025). OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033.

11. Narayanaswamy, V., Thopalli, K., Kailkhura, B., & Sakla, W. (2026). Improving Robustness In Sparse Autoencoders via Masked Regularization. arXiv:2604.06495.

12. Karvonen, A., Rager, C., Lin, J., et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. arXiv:2503.09532.

13. Engels, J., Riggs, L., & Tegmark, M. (2024). Decomposing The Dark Matter of Sparse Autoencoders. arXiv:2410.14670.

14. Cui, J., Zhang, Q., Wang, Y., & Wang, Y. (2025). On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963.

15. Chen, S., Sheen, H., Xiong, X., Wang, T., & Yang, Z. (2025). Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders. arXiv:2506.14002.

16. Korznikov, A., Galichin, A., Dontsov, A., Rogov, O., Oseledets, I., & Tutubalina, E. (2026). Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? arXiv:2602.14111.

17. Song, X., Muhamed, A., Zheng, Y., et al. (2025). Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs. arXiv:2505.20254.

18. Muchane, M., Richardson, S., Park, K., & Veitch, V. (2025). Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures. arXiv:2506.01197.

19. Luo, Y., Zhan, Y., Jiang, J., et al. (2026). From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders. arXiv:2602.11881.

20. Hänni, K., Mendel, J., Vaintrob, D., & Chan, L. (2024). Mathematical Models of Computation in Superposition. arXiv:2408.05451.

21. Shu, D., Wu, X., Zhao, H., et al. (2025). A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models. arXiv:2503.05613.

22. Ayonrinde, K., Pearce, M.T., & Sharkey, L. (2024). Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs. arXiv:2410.11179.

23. Martin-Linares, C.P., & Ling, J.P. (2025). Attribution-Guided Distillation of Matryoshka Sparse Autoencoders. arXiv:2512.24975.

24. Bereska, L., Tzifa-Kratira, Z., Samavi, R., & Gavves, E. (2025). Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability. arXiv:2512.13568.

25. [Author TBD]. (2024). A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima. arXiv:2512.05534. Code: https://github.com/[see paper]/The_Linear_Representation_Bench

26. Paulo, G., Shabalin, S., & Belrose, N. (2025). Transcoders Beat Sparse Autoencoders for Interpretability. arXiv:2501.18823.

27. [Author TBD]. (2025). Kronecker Factorization Improves Efficiency and Interpretability of Sparse Autoencoders (KronSAE). arXiv:2505.22255. URL: https://arxiv.org/abs/2505.22255

28. [Author TBD]. (2026). SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data. arXiv:2602.14687.

29. DeepMind Safety Research Team. (2025). Negative Results for Sparse Autoencoders On Downstream Tasks and Deprioritising SAE Research. Medium blog post: https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9

30. Li, X., et al. (2025). Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training. arXiv:2510.08855. URL: https://arxiv.org/abs/2510.08855

31. Gulko, A., Peng, Y., & Kumar, S. (2025). CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark of General Interpretability of Sparse Autoencoders. BlackboxNLP 2025 @ EMNLP. arXiv:2509.00691. Code: https://github.com/Yusen-Peng/CE-Bench

32. Tian, C., et al. (2025). Measuring Sparse Autoencoder Feature Sensitivity. arXiv:2509.23717. URL: https://arxiv.org/abs/2509.23717


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# Research Proposal: When Sparsity Eats Its Young — Feature Absorption as Rate-Distortion Optimal Behavior in Sparse Autoencoders

## Title

**When Sparsity Eats Its Young: Feature Absorption as Rate-Distortion Optimal Behavior, Cross-Domain Characterization, and Probe-Free Detection in Sparse Autoencoders**

---

## Abstract

Feature absorption — the systematic failure of SAE latents to fire on hierarchically related inputs — has been identified as the most fundamental reliability failure in sparse autoencoders. Yet despite its recognized importance, three critical gaps remain: (1) no theory explains *why* absorption is as severe as it is for any given SAE configuration; (2) absorption has been measured only on a single artificial proxy task (first-letter spelling); and (3) no method detects absorbed features without pre-specified probe directions. We address all three gaps in a unified, training-free research program. Our central contribution is a formal proof that feature absorption is the **rate-distortion optimal behavior** under flat sparsity penalties when features are hierarchically structured: the SAE is not failing at optimization — it is succeeding at optimizing the wrong objective. This reframing yields a quantitative absorption threshold (a function of decoder cosine similarity, co-occurrence frequency, and sparsity penalty strength) that makes falsifiable, testable predictions. We validate this theory empirically across multiple semantic hierarchy types (first-letter spelling, entity type, geographic, grammatical) on Gemma 2 2B using Gemma Scope pre-trained SAEs, establishing the first systematic cross-domain characterization of absorption. In parallel, we develop and validate an **Absorption Susceptibility Index (ASI)** — a probe-free, training-free metric computed from SAE weights and activation statistics that predicts which features are at risk of absorption without requiring known probe directions. Finally, we demonstrate that absorption exhibits **phase-transition dynamics**: there exists a critical sparsity threshold per feature pair above which absorption becomes energetically inevitable, and — crucially — the absorbed state may be metastable (hysteretic), implying that post-hoc sparsity reduction cannot reliably reverse absorption. These contributions collectively explain, predict, detect, and characterize feature absorption at a depth not previously achieved.

---

## Motivation

Feature absorption is arguably the most serious failure mode of SAEs for mechanistic interpretability. It creates a dangerous false confidence: an SAE appears to have learned a clean monosemantic latent for a concept, but that latent systematically fails to fire on 15–35% of inputs where the concept is present (Chanin et al., 2024). DeepMind's safety team found that this failure extends to safety-relevant applications: SAE probes fail catastrophically at detecting harmful intent while dense linear probes succeed even out-of-distribution. Despite this importance, the current state of the field has three critical blind spots.

**Blind spot 1: No causal theory of absorption severity.** Chanin et al. (2024) showed that absorption exists in every tested SAE and provided an informal argument: absorption "saves one L0 per parent-child pair." This is actually a rate-distortion statement in disguise, but it has never been formalized. Without a theory, practitioners cannot predict how severe absorption will be for a new SAE configuration or which architectural choices will reduce it.

**Blind spot 2: Single-task empiricism.** The entire quantitative understanding of absorption rests on one measurement task: whether SAE features for first-letter membership ("starts with E") absorb specific token features ("elephant", "eight", etc.). This is an artificial, syntactic hierarchy. We have no idea whether absorption is worse or better for the semantically rich hierarchies (city → country, entity type → specific entity) that actually appear in the model's knowledge representation and matter for safety applications.

**Blind spot 3: Supervised-probe dependency.** Every absorption metric requires the researcher to specify in advance which features to look for. This is acceptable for studying the first-letter hierarchy (which is known) but prohibits any broad-spectrum absorption survey. If a model has absorbed a safety-relevant feature that the researcher did not know to look for, no current method would detect it.

The rate-distortion framework provides the conceptual key to unlocking all three blind spots simultaneously. By formalizing "absorption saves L0" as a rate-distortion optimization, we derive (1) a closed-form absorption threshold that makes quantitative predictions; (2) a probe-free Absorption Susceptibility Index computable from decoder geometry; and (3) a phase-transition account of absorption onset that explains why simply reducing sparsity may not reverse absorption that has already occurred.

---

## Research Questions

1. **RQ1 (Theory):** Is feature absorption the unique rate-distortion optimal behavior under flat sparsity penalties for hierarchically structured features? What is the closed-form absorption threshold as a function of measurable SAE properties?

2. **RQ2 (Cross-domain):** Does absorption occur at comparable rates across semantically distinct feature hierarchies (spelling, entity type, geographic, grammatical)? What hierarchy properties (co-occurrence frequency, cosine similarity between feature directions, specificity ratio) predict absorption severity?

3. **RQ3 (Probe-free detection):** Does the Absorption Susceptibility Index (ASI) — derived from decoder geometry and activation statistics alone — predict which feature pairs will exhibit absorption with sufficient precision and recall to be practically useful (AUROC ≥ 0.70)?

4. **RQ4 (Dynamics):** Is absorption onset a phase transition in the sparsity parameter? Is it reversible (continuous crossover) or does it exhibit hysteresis (first-order transition with metastable absorbed states)?

---

## Hypotheses

**H1 (Rate-Distortion Optimality):** For any parent feature p and child feature c with co-occurrence probability p_co, the SAE loss landscape has a solution where c's decoder absorbs p's information that achieves strictly lower loss than the non-absorption solution when:
```
lambda * p_co > ||d_p - proj_{d_c} d_p||^2 * p_co
```
i.e., when `lambda > sin^2(theta_{p,c})` where theta is the angle between parent and child decoder directions. This gives a closed-form absorption threshold as a function of three measurable quantities: lambda (sparsity penalty), theta (decoder angle), and p_co (co-occurrence frequency).

*Falsification criterion:* If the theoretical threshold fails to predict which feature pairs exhibit absorption with AUROC < 0.70 on held-out SAE configurations, H1 is rejected.

**H2 (Cross-domain generality):** Absorption rates on entity-type hierarchies (e.g., "city" → "Paris") and grammatical hierarchies (e.g., "noun" → "proper noun") will fall within one standard deviation of absorption rates on the first-letter spelling hierarchy, controlling for SAE configuration. Absorption severity is predicted by cos^2(theta_{p,c}) × (freq_p / freq_c) across all hierarchy types.

*Falsification criterion:* If absorption rates on non-spelling hierarchies are statistically indistinguishable from shuffled-label controls (permutation test p > 0.05 after Bonferroni correction across 4 hierarchy types), H2 is rejected.

**H3 (Probe-free detection):** The Absorption Susceptibility Index ASI(p,c) = cos^2(theta_{p,c}) × (freq_p / freq_c), where theta is the angle between decoder columns and frequencies are measured from a reference corpus, achieves AUROC ≥ 0.70 against the Chanin et al. ground-truth absorption labels on the first-letter spelling task.

*Falsification criterion:* AUROC < 0.65 on the first-letter task constitutes rejection.

**H4 (Phase transition):** Absorption rate as a function of the effective sparsity penalty (1/L0 for TopK SAEs) exhibits a rapid transition (identifiable as a crossover or discontinuous jump) at a critical L0_c that varies predictably with SAE width and feature hierarchy properties. If first-order, the absorbed state persists when sparsity is subsequently reduced (hysteresis detectable within 60-minute fine-tuning budget).

*Falsification criterion:* If absorption rate increases monotonically and smoothly with 1/L0 with no detectable transition (linear fit explains ≥ 90% of variance), H4's phase-transition framing adds no explanatory value.

---

## Expected Contributions

1. **Formal proof that absorption is rate-distortion optimal** under flat L0/L1 sparsity penalties for hierarchically structured features. This resolves the field's conceptual confusion: absorption is not a failure of SAE training but the success of optimizing an objective that implicitly rewards hierarchical feature merging.

2. **Closed-form absorption threshold** (function of decoder cosine similarity, co-occurrence frequency, and sparsity penalty) that makes quantitative predictions across SAE configurations — the first principled guide for hyperparameter selection to minimize absorption.

3. **Absorption Impossibility Theorem:** Proof that for any SAE with flat sparsity penalty, there exists a critical hierarchy depth h* = O(1/sqrt(lambda)) such that all hierarchies of depth ≥ h* will exhibit absorption regardless of SAE width or training duration.

4. **First cross-domain absorption characterization:** Systematic measurement of absorption across first-letter spelling, entity-type (RAVEL), geographic (city→country→continent), and grammatical (POS) hierarchies on Gemma 2 2B using Gemma Scope SAEs. This establishes whether the 15–35% absorption rate found by Chanin et al. generalizes.

5. **Absorption Susceptibility Index (ASI):** A training-free, probe-free metric for identifying feature pairs at risk of absorption, validated against the Chanin et al. ground truth.

6. **Phase transition characterization:** First empirical evidence for whether absorption onset is smooth (crossover) or sharp (first-order), and whether it is reversible. If hysteresis is confirmed, this fundamentally changes the practical approach to absorption mitigation: architectural changes during training are necessary; post-hoc sparsity tuning cannot fix established absorption.

7. **Unified account of architectural mitigations:** The rate-distortion framework provides a single theoretical explanation for why Matryoshka SAEs (hierarchical codebook), OrtSAE (increased decoder angle between features), and ATM SAE (non-uniform sparsity) all reduce absorption through different mechanisms that each modify the absorption threshold.

---

## Method

All experiments are training-free analysis on pre-trained SAEs. Primary models: Gemma 2 2B (via Gemma Scope SAEs, multiple widths and L0 settings) and GPT-2 Small (via SAELens pre-trained SAEs, as fallback and for reproducibility). Every individual experiment targets ≤ 1 hour wall-clock time.

### Phase A: Theoretical Framework and Absorption Threshold Derivation

**A.1 — Formal rate-distortion analysis (no GPU required):**
Extend the piecewise biconvex framework from arXiv:2512.05534 with an explicit rate term. The SAE loss is a Lagrangian: L = E[||x - Df(Ex + b)||^2] + lambda × E[||f(Ex + b)||_0]. For a hierarchically related pair (parent p, child c) with co-occurrence probability p_co:

- Compare two solutions: (i) both features active (L0 cost 2, distortion minimal), vs. (ii) only c active with c's decoder absorbing p's direction (L0 cost 1, distortion = ||d_p - proj_{d_c} d_p||^2 × p_co = sin^2(theta_{p,c}) × p_co).
- Absorption is preferred when: lambda × p_co > sin^2(theta_{p,c}) × p_co, i.e., lambda > sin^2(theta_{p,c}).
- **Key insight:** The co-occurrence frequency p_co cancels, so the absorption threshold depends only on the decoder angle and sparsity penalty — not on how often the hierarchy fires. This is a falsifiable prediction.

**A.2 — Absorption Impossibility Theorem:**
For a complete b-ary feature hierarchy of depth h where each parent-child pair has angle theta = arcsin(sqrt(lambda)), all pairs are exactly at the absorption threshold. For hierarchies with theta < arcsin(sqrt(lambda)) (small angles, which arise when the model represents graded specificity), absorption is guaranteed. The critical depth h* where absorption becomes unavoidable for any fixed lambda scales as h* = O(1/sqrt(lambda)).

### Phase B: Empirical Validation of the Absorption Threshold

**B.1 — Sparsity-absorption scaling curve (30–45 min):**
Load Gemma Scope SAEs for Gemma 2 2B, layer 12, canonical widths (16k, 65k) with multiple L0 settings. Measure absorption rate on the first-letter task using the Chanin et al. sae-spelling metric. Plot absorption rate vs. 1/L0. Fit the predicted functional form (sigmoid function of lambda - sin^2(theta_{p,c})). Compare to linear/power-law baselines.

**B.2 — Decoder geometry analysis (30 min):**
For absorbed feature pairs identified by the Chanin et al. metric, compute the decoder cosine similarity between the absorber and absorbee. Test H1: absorbed pairs should have significantly smaller decoder angle (higher cosine similarity) than non-absorbed pairs with similar co-occurrence rates. Use Wilcoxon rank-sum test with effect size reporting.

**B.3 — Cross-architecture validation (45 min):**
Compare absorption rates across Gemma Scope architectures (TopK, JumpReLU) at matched L0 values. The theory predicts that JumpReLU's higher effective lambda (trained longer, produces more absorption) should systematically raise absorption rates — testable by comparing architectures at identical L0.

### Phase C: Cross-Domain Absorption Characterization

**C.1 — Probe training (45 min):**
Train logistic regression probes on Gemma 2 2B residual stream activations (layer 12) for four hierarchy types:
- First-letter spelling (baseline, from sae-spelling repo)
- Entity type: city→country→continent using RAVEL dataset (Huang et al., 2024; 3000+ cities, already in SAEBench)
- Geographic: use RAVEL's country-continent pairs
- Grammatical: noun/verb POS → specific subtypes (proper noun, transitive verb) using Penn Treebank-tagged data

Require F1 ≥ 0.80 before computing absorption; exclude any hierarchy failing this gate. Run shuffled-label null controls (100 permutations per hierarchy) to establish the noise floor.

**C.2 — Cross-domain absorption measurement (45–60 min):**
Adapt the sae-spelling absorption calculator to use entity-attribute probes instead of letter probes. The core logic (probe + SAE latent + false-negative detection) is hierarchy-agnostic; only the probe target changes. Measure absorption rate per hierarchy type, per SAE width (16k, 65k), reporting ratio-to-shuffled-null and 95% bootstrap confidence intervals.

**C.3 — Hierarchy property analysis (30 min):**
For each hierarchy type, compute: (a) mean decoder cosine similarity between parent and child latents; (b) mean co-occurrence rate (parent freq / child freq). Test H2: hierarchies with higher decoder cosine similarity and larger frequency ratios should exhibit higher absorption. Report Spearman correlation across hierarchy types.

### Phase D: Absorption Susceptibility Index (Probe-Free Detection)

**D.1 — ASI computation (15 min):**
For all feature pairs in one SAE (Gemma Scope 16k, layer 12), compute:
```
ASI(p, c) = cos^2(theta_{p,c}) × (freq_p / freq_c)
```
where theta_{p,c} is the angle between decoder columns i and j, and frequencies are measured from a reference corpus (OpenWebText subset). Pre-filter to pairs with co-activation frequency > 0.01 to reduce the O(d_sae^2) comparison to tractable scale (~10k candidate pairs for a 16k SAE).

**D.2 — Validation against ground truth (30 min):**
Compare ASI-ranked pairs against the Chanin et al. absorption labels on the first-letter task. Compute AUROC, AUPRC, precision@k (for k=100, 500). Test H3: ASI should achieve AUROC ≥ 0.70. Report precision-recall curves rather than single thresholds.

**D.3 — Cross-domain ASI validation (30 min):**
For hierarchies where ground truth absorption is measurable (entity-type, geographic), test whether ASI computed from decoder geometry alone predicts cross-domain absorption rates. If ASI rank-correlates with cross-domain absorption rates (Spearman rho > 0.4), it has genuine predictive value without any probe training.

### Phase E: Phase Transition and Dynamics

**E.1 — Sparsity sweep phase detection (45 min):**
Using the full suite of Gemma Scope SAEs for layer 12 (widths 16k, 65k; multiple L0 values in the Gemma Scope canonical set), measure absorption rate as a function of effective sparsity. Test whether the relationship is smooth (linear/power-law in 1/L0) or exhibits a detectable inflection point consistent with a phase transition. Use a likelihood ratio test comparing linear vs. sigmoid functional forms.

**E.2 — Hysteresis test (60 min):**
Starting from a Gemma Scope JumpReLU SAE at high absorption (L0=25 or similar), fine-tune for 500 steps with reduced sparsity (effectively L0=75 or similar) using SAELens. Measure: (a) absorption rate before fine-tuning; (b) absorption rate after fine-tuning; (c) reconstruction quality. If absorption rate does not decrease to match the rate of a SAE trained from scratch at the lower L0, hysteresis is confirmed. Note: fine-tuning requires GPU; limit to one SAE (GPT-2 small for speed) if Gemma fine-tuning exceeds 1 hour. *This experiment requires training, but on existing pre-trained SAEs as initialization — minimal overhead.*

**E.3 — Phase diagram (30 min):**
Using SAEBench's collection of SAEs at multiple widths and L0 values, map absorption rate across the (L0, width) space. Identify whether the critical L0 shifts with width as predicted by the theory (wider SAEs tolerate more sparsity before absorption becomes dominant — analogous to a larger ecosystem supporting more similar species).

### Phase F: Unified Account of Architectural Mitigations

**F.1 — Theoretical analysis of mitigations (no GPU required):**
Analyze each leading mitigation through the absorption threshold lens:
- **Matryoshka SAE:** Inner dictionaries give parent features a dedicated "slot" not competing with child features, effectively increasing sin^2(theta) for the inner levels.
- **OrtSAE:** Orthogonality penalty directly increases the decoder angle between similar features, pushing more pairs above the absorption threshold.
- **ATM SAE:** Per-latent importance scoring creates non-uniform sparsity costs, effectively giving parent features a lower lambda, reducing absorption without changing architecture.
- **Masked regularization:** Disrupts co-occurrence patterns, which per the threshold formula does not affect absorption (p_co cancels) — but in practice, masking during training prevents the absorbed solution from forming, consistent with hysteresis.

**F.2 — Empirical verification (30 min):**
For Matryoshka vs. TopK on the same model/layer (available via SAEBench), compute: (a) mean decoder cosine similarity between parent-child pairs; (b) absorption rate; (c) ASI distribution. Test whether Matryoshka's lower absorption is accompanied by larger decoder angles between hierarchically related features, consistent with the theoretical account.

---

## Experimental Plan

| Experiment | Phase | Tool | Target | Falsification Criterion | Estimated Time |
|---|---|---|---|---|---|
| Rate-distortion theory validation | B | sae-spelling, SAELens | Gemma 2 2B, L12, 16k & 65k | AUROC < 0.70 for threshold prediction | 30 min |
| Decoder geometry analysis | B | SAELens | Gemma Scope SAEs | Non-significant angle difference for absorbed pairs | 30 min |
| Cross-architecture validation | B | SAEBench | Gemma Scope TopK vs. JumpReLU | Architecture absorption ranking contradicts theory | 45 min |
| Probe training (4 hierarchy types) | C | sae-spelling, RAVEL | Gemma 2 2B layer 12 | F1 < 0.80 for ≥ 3 hierarchies | 45 min |
| Cross-domain absorption measurement | C | sae-spelling | Gemma Scope 16k & 65k | Shuffled null indistinguishable from real (p > 0.05) | 60 min |
| Hierarchy property correlation | C | SAELens | Gemma Scope | Spearman rho < 0.3 for ASI vs. absorption rate | 30 min |
| ASI computation and validation | D | SAELens | Gemma Scope 16k, L12 | AUROC < 0.65 on first-letter task | 45 min |
| ASI cross-domain validation | D | SAELens | Gemma Scope | ASI cross-domain rho < 0.3 | 30 min |
| Phase transition detection | E | SAEBench | Gemma Scope suite | Linear model R² ≥ 0.90 vs. sigmoid (no phase transition) | 45 min |
| Hysteresis test | E | SAELens | GPT-2 small | Absorption rate drops to match scratch-trained baseline | 60 min |
| Phase diagram mapping | E | SAEBench | Gemma Scope | No systematic shift of critical L0 with width | 30 min |
| Mitigation theoretical analysis | F | Theory | N/A | Mitigations unexplainable via threshold formula | — |
| Mitigation empirical verification | F | SAEBench | Matryoshka vs. TopK | No decoder angle difference for hierarchical pairs | 30 min |

**Total estimated compute:** ~9–11 GPU-hours (single A100 or equivalent). All experiments except Phase E.2 (hysteresis test) are analysis-only on pre-trained SAEs. Phase E.2 adds ~1 hour of fine-tuning on GPT-2 small.

**Pilot (15 min):** Load GPT-2 small SAE (layer 6, width 24576 from SAELens). Compute ASI for all feature pairs (filter to co-activation > 0.01 → ~10k pairs). Identify top-100 pairs by ASI. Cross-reference against Neuronpedia labels to manually verify these pairs look like absorption candidates. Run the first-letter absorption rate calculation. This pilot validates the core computational pipeline and gives early signal on whether ASI has any predictive value.

---

## Resource Estimate

- **GPU:** Single A100 (40GB). Gemma 2 2B fits in ~10GB; SAEs add ~200MB each. GPT-2 Small for pilot and hysteresis.
- **GPU-hours:** ~9–11 hours total (pilot: 15 min; Phase B: 1.5 hr; Phase C: 2 hr; Phase D: 1 hr; Phase E: 2 hr; Phase F: 1 hr).
- **Models:** Gemma 2 2B (pre-trained, HuggingFace), GPT-2 Small (pre-trained via TransformerLens).
- **SAEs:** Gemma Scope pre-trained SAEs (16k, 65k widths, layer 12 primary, other layers for scaling analysis). GPT-2 Small SAEs from SAELens.
- **Datasets:** RAVEL (hij/ravel on HuggingFace), sae-spelling first-letter task data, OpenWebText subset for frequency computation.
- **Code:** sae-spelling (MIT), SAELens (MIT), SAEBench (Apache 2.0), Gemma Scope (Gemma ToS).
- **Storage:** ~20GB for cached model activations and SAE weights.

---

## Risk Assessment and Mitigations

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Rate-distortion threshold has AUROC < 0.70 (too weak for prediction) | High | Low-Medium | Theory remains valid as existence proof even if quantitative fit is imprecise; extend with additional predictors (feature norm, activation kurtosis); the theory's qualitative predictions (absorbed pairs have higher cosine similarity) are still publishable |
| Entity-attribute probes fail quality gate (F1 < 0.80) | Medium | Medium | Start with 8 candidate hierarchies and report all that pass; if fewer than 3 pass, narrow scope to first-letter + grammatical hierarchy; pilot in 15 min to screen |
| Cross-domain absorption rates floor out (indistinguishable from shuffled null) | Medium | Medium | Report as a quantitative finding: "absorption is specific to first-letter/syntactic hierarchies"; the null result is informative and refines the scope of the phenomenon |
| ASI AUROC < 0.65 (probe-free detection fails) | High | Medium | Drop as primary contribution; retain as supplementary analysis; strengthen cross-domain and theory contributions |
| Hysteresis experiment inconclusive (fine-tuning reverses absorption, no hysteresis) | Medium | Medium | Report as absence of hysteresis — consistent with a continuous crossover rather than first-order transition; still characterizes phase dynamics; update theoretical framing accordingly |
| Gemma 2 2B access issues | Low | Low | GPT-2 Small is primary fallback for all experiments; SAELens SAEs for GPT-2 are unrestricted |
| Multiple testing inflation of cross-domain results | Low | Low | Pre-register all comparisons; apply Bonferroni correction; report effect sizes with bootstrap CIs |

---

## Novelty Assessment

Searches conducted (April 2026) for: "feature absorption SAE rate distortion optimal," "feature absorption cross domain entity hierarchy," "sparse autoencoder absorption phase transition hysteresis," "SAE absorption probe-free unsupervised detection encoder decoder." Key findings:

**No collision found for core contributions:**

1. **Rate-distortion optimality of absorption:** The existing literature acknowledges that "absorption saves one L0" (Chanin et al., 2024) and that this makes absorption "an effective strategy for reducing L0 at fixed FVU" (Alignment Forum, 2025). The Tilde Research blog informally discusses rate-distortion framing. The MDL-SAEs paper (Ayonrinde et al., 2024) uses compression language. **However, no paper has formalized a closed-form absorption threshold or proven optimality rigorously.** The novelty is the formal derivation, not the informal observation.

2. **Cross-domain characterization:** Search results confirm this gap explicitly: "SAEBench evaluates feature absorption by using features for 'word starts with X', which is not useful for evaluating domain-specific feature absorption." No paper has systematically measured absorption across multiple semantic hierarchy types.

3. **Probe-free ASI:** All current detection methods (Chanin et al., SAEBench) require probe directions. The feature sensitivity metric (Tian et al., 2025) is related but distinct (measures reliability, not absorption specifically). No probe-free absorption predictor exists.

4. **Phase transition and hysteresis:** The search returned zero results for "SAE absorption phase transition hysteresis" or related queries. The statistical physics framing is entirely novel in this application domain.

**Potential collision flags:**
- The rate-distortion framing is partially anticipated by the Tilde Research blog — we must cite this and clearly articulate the formal extension.
- The Hierarchical SAE papers (arXiv:2506.01197, arXiv:2602.11881) address hierarchy in SAE architecture but not absorption measurement or its rate-distortion characterization.
- The feature sensitivity metric (Tian et al., 2025) overlaps with some aspects of the ASI but does not use decoder geometry and does not provide unsupervised absorption detection.

All four core contributions are novel relative to the literature as of April 2026.

---

## What Changed From Prior Rounds

This is the initial synthesis from 6 perspectives (no prior proposal or pilot evidence exists). Key decisions made in this synthesis:

1. **Selected rate-distortion theory as the organizing framework** because it provides the tightest theoretical novelty (9/10 from both Innovator and Theoretical perspectives), directly motivates the empirical program, and explains why existing mitigations work.

2. **Retained cross-domain characterization as the primary empirical contribution** because it directly addresses the most impactful practical gap (all downstream safety applications use semantic hierarchies, not letter-membership hierarchies), is strongly feasible via RAVEL + sae-spelling, and is the most clearly publishable regardless of theory outcome.

3. **Added phase transition dynamics (from Interdisciplinary perspective)** as an empirical secondary contribution because the hysteresis prediction is uniquely novel, the experiment is feasible, and if confirmed it has immediate practical implications.

4. **Integrated the Contrarian's reframing** by including absorption-as-diagnostic as an interpretive lens: the cross-domain absorption profile reveals the geometry of model representations, which is a richer story than pure mitigation.

5. **Adopted the Empiricist's rigorous controls** (shuffled-label null, Bonferroni correction, bootstrap CIs, probe quality gate, pre-registered comparisons) as non-negotiable methodological standards.

---

## Backup Ideas for Pivot

See `alternatives.md` for full descriptions of backup candidates.

---

## Appendix: Key Equations

**Absorption Threshold (from rate-distortion analysis):**
Absorption of parent p into child c is energetically preferred when:
```
lambda > sin^2(theta_{p,c})
```
where lambda is the sparsity penalty (or 1/L0 for TopK SAEs) and theta_{p,c} is the angle between the decoder directions of parent and child features.

**Absorption Susceptibility Index:**
```
ASI(p, c) = cos^2(theta_{p,c}) × (freq_p / freq_c)
```
High ASI indicates high absorption risk. Computable from SAE weights and activation frequency counts without any probe training.

**Critical Hierarchy Depth for Inevitable Absorption:**
For a feature hierarchy with branching factor b and a sparsity penalty lambda:
```
h* = O(1 / sqrt(lambda))
```
All hierarchies with depth ≥ h* will have at least one absorbed feature pair, regardless of SAE width or training duration.


## 当前可检验假设
# Testable Hypotheses

## Primary Hypotheses

### H1: Rate-Distortion Absorption Threshold

**Statement:** For a parent feature p and child feature c in a trained SAE, absorption of p into c occurs if and only if the sparsity penalty lambda exceeds sin^2(theta_{p,c}), where theta_{p,c} is the angle between their decoder directions:

```
Absorption(p, c) = 1  iff  lambda > sin^2(theta_{p,c})
```

**Expected outcome:** AUROC ≥ 0.70 when using the threshold to predict which feature pairs are absorbed (from Chanin et al. ground truth on first-letter task).

**Mechanism:** When lambda > sin^2(theta), the L0 savings from absorbing one activation (cost = lambda) exceed the reconstruction cost from the decoder misalignment (cost = sin^2(theta) × p_co, and p_co cancels). Hence absorption is the unique energetically preferred solution.

**Key measurement:** For each absorbed and non-absorbed feature pair identified by the Chanin et al. metric, compute cos^2(theta_{p,c}). Absorbed pairs should have systematically higher cos^2(theta) (smaller angle). Compare against prediction: all pairs with cos^2(theta) > 1 - lambda should be absorbed.

**Falsification:** AUROC < 0.65 after fitting the threshold to one SAE and predicting absorption in held-out SAE configurations.

---

### H2: Cross-Domain Generality of Absorption

**Statement:** Absorption occurs at statistically detectable rates (ratio-to-shuffled-null ≥ 3.0, p < 0.01 after Bonferroni correction) across all four hierarchy types tested:
- First-letter spelling (baseline)
- Entity-type (city→country from RAVEL)
- Geographic (city→continent from RAVEL)
- Grammatical (POS→subtype, e.g., noun→proper noun)

**Expected outcome:** Absorption rates for entity-type and geographic hierarchies fall within [10%, 45%] — within the range of the first-letter task's 15–35%, possibly higher due to richer semantic overlap between parent and child features. Shuffled-label controls show absorption rates < 5% for all hierarchy types.

**Key measurement:** For each hierarchy type, compute mean absorption rate across the 16k and 65k Gemma Scope SAEs at layer 12. Report ratio-to-shuffled-null with 95% bootstrap CI.

**Falsification:** If any of the 4 hierarchy types shows ratio-to-shuffled-null < 1.5 (indistinguishable from noise), absorption is not universal across hierarchy types — it is specific to certain hierarchy structures. This would refine the scope of the phenomenon.

---

### H3: Absorption Susceptibility Index (ASI) Validity

**Statement:** The Absorption Susceptibility Index:
```
ASI(p, c) = cos^2(theta_{p,c}) × (freq_p / freq_c)
```
computed from SAE decoder weights and activation frequencies alone (no probe training required), predicts absorbed feature pairs with AUROC ≥ 0.70 against the Chanin et al. ground truth on the first-letter task.

**Expected outcome:** Among the top-k pairs ranked by ASI, precision ≥ 0.40 and recall ≥ 0.40 for the absorbed-pair class. The precision-recall curve achieves AUPRC > 0.30 (well above the base rate of absorbed pairs ≈ 0.05–0.15 of all high-cosine pairs).

**Key measurement:** Compute pairwise ASI for all feature pairs with co-activation frequency > 0.01. Compare ASI ranking against the Chanin et al. absorption labels. AUROC against labeled absorbed/non-absorbed pairs.

**Falsification:** AUROC < 0.60 (at or near chance level given the imbalanced class distribution) means decoder geometry has no predictive value for absorption.

---

### H4: Absorption Onset Phase Dynamics

**H4a (Phase transition):** Absorption rate, plotted as a function of effective sparsity pressure (1/L0 for TopK SAEs), exhibits a rapid transition at a critical L0_c — identifiable as a sigmoid-like crossover rather than a linear increase. Specifically: a sigmoid functional form fits absorption-vs-sparsity data significantly better than a linear model (likelihood ratio test, p < 0.05).

**H4b (Hysteresis):** When a SAE trained at high sparsity (absorption present) is fine-tuned with reduced sparsity, the absorption rate decreases by at most 50% of the gap between the original rate and the rate of a SAE trained from scratch at the reduced sparsity. If the absorbed state is fully metastable, the absorption rate may not decrease at all.

**Expected outcome for H4a:** Sigmoid model R² > 0.85 vs. linear model R² < 0.70 on the absorption-vs-sparsity data across Gemma Scope SAE configurations.

**Expected outcome for H4b:** Fine-tuned SAE absorption rate remains ≥ 70% of the original high-sparsity absorption rate, while a from-scratch SAE at the same L0 achieves absorption rate ≤ 30% of the high-sparsity rate.

**Falsification for H4a:** Linear fit R² ≥ 0.90 (absorption increases uniformly with sparsity, no phase transition).

**Falsification for H4b:** Fine-tuned absorption rate drops below 50% of the original rate within the fine-tuning budget (absorption is fully reversible, no hysteresis).

---

## Secondary Hypotheses

### H5: Architectural Mitigations Increase Effective Decoder Angles

**Statement:** Matryoshka SAE and OrtSAE reduce absorption by mechanisms that are interpretable through the rate-distortion threshold: Matryoshka increases the effective decoder angle between hierarchically related features (by constraining inner-level features to cover general concepts), and OrtSAE does so directly via the orthogonality penalty.

**Expected outcome:** Matryoshka and OrtSAE SAEs have significantly larger mean cos^2(theta_{p,c}) for known absorbed feature pairs compared to TopK SAEs — i.e., the mitigations push feature pairs below the absorption threshold.

**Key measurement:** For the first-letter hierarchy, compute decoder cosine similarity between known absorbed pairs across Matryoshka, OrtSAE, and TopK SAEs at matched width.

**Falsification:** No significant difference in decoder angles between architectures despite significant difference in absorption rates — would imply the mitigation works via a different mechanism than predicted by the rate-distortion theory.

### H6: ASI Predicts Cross-Domain Absorption Rates

**Statement:** The cross-domain absorption rate of a given hierarchy type is predictable from the mean ASI of feature pairs in that hierarchy (higher mean ASI → higher absorption rate).

**Expected outcome:** Spearman correlation between mean ASI and absorption rate across the 4 hierarchy types is rho ≥ 0.7 (4 data points, so this is a directional claim rather than a statistical test).

**Key measurement:** Compute mean ASI for each hierarchy type's probe-identified feature pairs. Correlate with measured absorption rate.

**Falsification:** Mean ASI does not rank hierarchy types in the same order as absorption rates — would suggest ASI is hierarchy-specific and not a universal predictor.

---

## Summary Table

| Hypothesis | Prediction | Success Metric | Falsification |
|---|---|---|---|
| H1: RD threshold | Absorbed pairs have cos^2(theta) > 1 - lambda | AUROC ≥ 0.70 | AUROC < 0.65 |
| H2: Cross-domain | Absorption detectable in all 4 hierarchy types | Ratio-to-null ≥ 3.0, p < 0.01 | Any hierarchy indistinguishable from null |
| H3: ASI probe-free | Decoder geometry predicts absorption | AUROC ≥ 0.70 | AUROC < 0.60 |
| H4a: Phase transition | Sigmoid fit better than linear | LRT p < 0.05 | Linear R² ≥ 0.90 |
| H4b: Hysteresis | Fine-tuning does not reverse absorption | Rate > 70% of original after FT | Rate drops below 50% |
| H5: Mitigation mechanism | Mitigations increase decoder angles | Significant angle increase | No angle difference |
| H6: ASI cross-domain | Mean ASI predicts cross-domain absorption | Spearman rho ≥ 0.7 | rho < 0.3 |


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "candidates": [
    {
      "candidate_id": "cand_a",
      "title": "Rate-Distortion Theory + Cross-Domain Characterization + ASI + Phase Transition",
      "status": "front_runner",
      "summary": "A unified research program treating absorption as rate-distortion optimal behavior under flat sparsity penalties. Core contributions: (1) formal proof of absorption optimality with closed-form threshold lambda > sin^2(theta_{p,c}); (2) first cross-domain absorption characterization on 4 hierarchy types (spelling, entity-type, geographic, grammatical); (3) probe-free Absorption Susceptibility Index computed from decoder geometry; (4) phase transition characterization with hysteresis test. All components are training-free analysis on pre-trained Gemma Scope SAEs.",
      "hypotheses": [
        "Feature absorption is rate-distortion optimal when lambda > sin^2(theta_{p,c}), where theta is the decoder angle between parent and child features",
        "Absorption occurs at statistically detectable rates across all 4 tested hierarchy types (spelling, entity-type, geographic, grammatical)",
        "ASI = cos^2(theta_{p,c}) × (freq_p / freq_c) predicts absorbed feature pairs with AUROC >= 0.70 without probe training",
        "Absorption onset is a phase transition in the sparsity parameter, potentially with hysteresis (metastable absorbed state)"
      ],
      "pilot_focus": "15-minute pilot on GPT-2 Small: (1) load SAELens SAE, (2) compute ASI for all co-active pairs, (3) cross-reference top-100 pairs against Neuronpedia labels, (4) measure first-letter absorption rate. Validates the computational pipeline and gives early signal on ASI predictive value.",
      "estimated_gpu_hours": 9,
      "target_venue": "NeurIPS 2026 or ICLR 2027",
      "primary_gaps_addressed": ["Gap 1 (no quantitative theory)", "Gap 2 (absorption only on spelling)", "Gap 6 (semantic hierarchies unstudied)", "Gap 7 (no probe-free detection)"],
      "key_risk": "ASI has low predictive power (AUROC < 0.65) OR cross-domain absorption rates are indistinguishable from null controls. If both fail, fall back to Cand B."
    },
    {
      "candidate_id": "cand_b",
      "title": "Absorption-as-Representational-Diagnostic",
      "status": "backup",
      "summary": "Reframe absorption from a failure mode to a diagnostic signal. The absorption graph (absorber->absorbee relationships) encodes the model's internal feature hierarchy. Cross-layer absorption profiles reveal computational structure: early layers show less absorption (context-independent features); middle layers show more (contextually entangled). Mitigation methods reduce absorption rates by explicitly encoding hierarchy rather than by genuinely recovering previously absorbed features.",
      "hypotheses": [
        "Absorption rate correlates positively with hierarchical entanglement (measured by hierarchical mutual information) across layers and domains",
        "The absorption graph has non-trivial clustering structure reflecting semantic relationships and is consistent across SAE widths",
        "Cross-layer absorption profile has systematic structure (peaks at middle layers) consistent with known properties of layer-wise representations",
        "Matryoshka's absorption reduction is partially due to explicit hierarchy encoding rather than genuine recovery"
      ],
      "pilot_focus": "30-minute pilot: measure absorption rate on first-letter task across 6 Gemma Scope SAEs at layers 3, 6, 12, 19 for 16k and 65k. Test whether cross-layer variation has systematic structure.",
      "estimated_gpu_hours": 5,
      "target_venue": "NeurIPS 2026 or workshop",
      "primary_gaps_addressed": ["Gap 2", "Gap 3 (relationship between failure modes)"],
      "key_risk": "Absorption graph is inconsistent across SAE runs (Song et al. 2025 shows features are only ~0.80 consistent); cross-layer profile shows no systematic structure."
    },
    {
      "candidate_id": "cand_c",
      "title": "Systematic Mitigation Benchmark",
      "status": "backup",
      "summary": "Controlled head-to-head comparison of absorption mitigation methods (Matryoshka, OrtSAE, ATM SAE, masked regularization) on the same model, layer, and metric suite. First paper to compare all major methods fairly, revealing trade-offs that are currently invisible due to inconsistent evaluation protocols.",
      "hypotheses": [
        "No single mitigation method dominates across all absorption metrics AND downstream tasks (sparse probing, RAVEL disentanglement, reconstruction loss)",
        "Absorption reduction from Matryoshka does not fully generalize to entity-type hierarchies (reduction is architecture-specific)",
        "OrtSAE provides the best absorption-reconstruction balance without the reconstruction penalty of Matryoshka"
      ],
      "pilot_focus": "15-minute pilot: load Matryoshka vs. TopK Gemma Scope SAEs (both pre-trained and available via SAEBench), compare absorption rate on first-letter task, reconstruction CE delta. Verifies the benchmark pipeline works.",
      "estimated_gpu_hours": 4,
      "target_venue": "Workshop or findings paper",
      "primary_gaps_addressed": ["Gap 4 (no controlled mitigation comparison)"],
      "key_risk": "Purely empirical without conceptual insight — low ceiling for top-venue acceptance. Must be combined with theoretical framing from Cand A or B."
    },
    {
      "candidate_id": "cand_d",
      "title": "Feature Anchoring for Real LLM Absorption Reduction",
      "status": "backup",
      "summary": "First empirical evaluation of feature anchoring (proposed in arXiv:2512.05534 unified SDL theory) for absorption reduction on real LLMs. Currently only validated on synthetic benchmarks. Tests whether anchoring can be combined with existing architectures (Matryoshka, OrtSAE) for compounded benefit.",
      "hypotheses": [
        "Feature anchoring reduces absorption rate by at least 30% relative to unanchored TopK baseline on Gemma 2 2B",
        "Absorption reduction is largest for feature pairs with high decoder cosine similarity (near the absorption threshold from Cand A theory)",
        "Combining anchoring with OrtSAE achieves sub-additive absorption reduction (each addresses absorption through a different mechanism)"
      ],
      "pilot_focus": "30-minute pilot: implement feature anchoring on top of SAELens TopK SAE, train for 1000 steps on GPT-2 small layer 6, measure absorption rate vs. unanchored baseline.",
      "estimated_gpu_hours": 8,
      "target_venue": "NeurIPS 2026 or applied ML workshop",
      "primary_gaps_addressed": ["Gap 8 (feature anchoring effectiveness uncharacterized)"],
      "key_risk": "Requires SAE training (violates project's training-free preference); feature anchoring hyperparameters may require extensive tuning; limited to cases where anchoring directions can be specified."
    }
  ],
  "synthesis_notes": {
    "perspective_weights": {
      "innovator": "High — rate-distortion framing and ASI concept originated here. Highest novelty estimates (8-9/10) for the theoretical contributions.",
      "theoretical": "High — provided rigorous proof sketches for absorption optimality (Theorem 1), HiRIP certificate (Theorem 2), and impossibility result (Theorem 3). Core mathematical structure adopted in front-runner.",
      "pragmatist": "Medium-High — provided feasibility validation and reusable code infrastructure. Cross-domain characterization plan largely based on this perspective's sae-spelling + RAVEL combination.",
      "empiricist": "High — provided rigorous evaluation design (shuffled controls, Bonferroni correction, probe quality gates, bootstrap CIs). All controls adopted in front-runner.",
      "contrarian": "Medium — challenged assumptions productively. The reframing of mitigation analysis (do mitigations genuinely recover features or re-encode hierarchy?) is incorporated as Phase F. The scale of the paradigm challenge was walked back given ATM SAE's 95% absorption reduction.",
      "interdisciplinary": "Medium — phase transition framework and immunodominance index contributed. Phase transition dynamics incorporated as Phase E. Hysteresis test retained. Immunodominance index reabsorbed into ASI (same formula, different framing)."
    },
    "key_conflicts_resolved": [
      "Innovator proposed training-free analysis only; Interdisciplinary proposed hysteresis test requiring fine-tuning. Resolution: hysteresis test allowed as one 60-minute experiment on GPT-2 Small (small training cost, high scientific value).",
      "Pragmatist proposed cross-domain as primary contribution; Innovator proposed rate-distortion theory as primary. Resolution: they are complementary, not competing — theory motivates empirical design, cross-domain validates and extends theory.",
      "Contrarian suggested the entire SAE paradigm may be flawed. Resolution: acknowledged but scoped appropriately — we study absorption within the SAE paradigm (where the community needs answers), and the contrarian's insight that 'absorption reveals representation geometry' is incorporated as a secondary framing.",
      "Theoretical perspective proposed HiRIP as a certification tool; synthesis simplified this to the probe-free ASI which captures the same intuition (decoder geometry predicts absorption) in a more computationally tractable way."
    ],
    "dropped_ideas": [
      "Competitive exclusion / Lotka-Volterra as the primary framework (Ecological Coexistence Candidate B from Interdisciplinary) — retained as interpretive framing but not as the core mathematical structure; rate-distortion is cleaner.",
      "Standalone metric artifact study (Contrarian Candidate B) — addressed by the shuffled-label controls in the empirical design rather than as a separate paper.",
      "Reconstruction-only mitigation benchmark as primary contribution (Pragmatist Candidate C) — demoted to backup status."
    ]
  }
}


## 上一轮 validation 决策意见
# Idea Validation Decision

**Date:** 2026-04-13  
**Iteration:** iter_001  
**Evaluator:** sibyl-idea-validation-decision (sibyl-heavy)

---

## Pilot Evidence Summary

All pilot and full-phase experiments were completed on GPT-2 Small (gpt2-small-res-jb, layer 6 primary, 24576-width SAE). Eleven SAE configurations were tested for the scaling curve (layers 2–12, widths 24576–46080).

### Per-Hypothesis Status from Pilots

| Hypothesis | Target Metric | Observed Value | Pass? |
|---|---|---|---|
| H1: RD threshold predicts absorbed pairs | AUROC ≥ 0.70 | AUROC = 0.410 (direction reversed: cos^2 LOWER for absorbed features) | FALSIFIED |
| H2: Cross-domain absorption detectable | ratio-to-null ≥ 1.5, p<0.01 | All hierarchies: absorption_rate = 0.0, ratio = 0.0 (concept latents actively fire) | NOT SUPPORTED |
| H3: ASI (probe-free) predicts absorption | AUROC ≥ 0.70 | AUROC = 0.4215 (below null mean 0.497; anti-correlated with labels) | FALSIFIED |
| H4a: Phase transition in sparsity | sigmoid BIC better than linear, LRT p < 0.05 | BIC diff = -3.22, LRT p = 0.456; linear is better fit | NOT SUPPORTED |
| H4b: Hysteresis | absorption persists after sparsity reduction | All 11 L0 configs show absorption_rate ~95-97% (saturation, not testable) | NOT TESTABLE |

### What DID Work

- **EDA detector** (1 - cos(encoder_j, decoder_j)) achieves AUROC = 0.681 on GPT-2 L6, meaningfully above null (null mean = 0.502). Consistent with iter_001 reference (0.629). This is the only metric with genuine predictive value.
- **Probe training (C1):** first_letter F1=0.820, noun_proper F1=0.987, animate_inanimate F1=1.0 — robust probes exist. These are valid infrastructure for any future cross-domain work.
- **Theoretical derivation (F1):** Complete algebraic derivation of Theorem 1 (rate-distortion threshold) is non-circular and mathematically sound. The formal proof is complete.
- **Scaling observations:** All tested SAE configs show absorption_rate 77%–99%, high throughout. No SAE configuration escapes high absorption.
- **Figure 1 PDF** generated and verified.

### Critical Negative Findings (Scientifically Meaningful)

1. **H1 direction reversal:** Absorbed letter features have SIGNIFICANTLY LOWER cos^2(theta) with candidate parent features (pos_mean=0.052) than non-absorbed features (neg_mean=0.127), Cohen's d = -0.57, Wilcoxon p=1.3e-6. This is the OPPOSITE of H1's prediction. The mechanism of absorption is NOT decoder cosine alignment between parent and child — it is encoder-decoder misalignment (EDA).
2. **ASI anti-prediction:** The frequency-ratio component of ASI (AUROC = 0.612 alone) actually has signal, but the cos^2 component (AUROC = 0.206) actively hurts the composite. The combined ASI (AUROC = 0.422) is worse than freq_ratio alone.
3. **C2 NO_GO:** Cross-domain concept latents fire reliably (mean_max_activation = 16–22) for their respective concept words. The hypothesis that semantic parent-concept latents absorb child features is not supported in GPT-2 Small L6. The "absorption" that Chanin et al. measured on first-letter spelling appears to be a specific mechanism not shared by semantic hierarchies in the same model/layer.
4. **Phase transition absent:** Absorption rates are uniformly high (77–99%) across all tested L0 values. The system appears to be in the absorbed phase for all practical SAE configurations tested, making phase transition detection impossible.

---

## Decision Matrix

### cand_a: Rate-Distortion Theory + Cross-Domain + ASI + Phase Transition

| Criterion | Weight | Score (1-5) | Evidence |
|---|---|---|---|
| Pilot signal strength | 0.30 | 1 | H1 FALSIFIED (direction reversed). H3 FALSIFIED (AUROC=0.422 < null). H4a NOT SUPPORTED (LRT p=0.456). Only EDA shows positive signal (0.681). 4 out of 5 primary hypotheses fail. |
| Hypothesis survival | 0.25 | 1 | H1 falsified by direction reversal (not just weak signal — the effect is in the wrong direction). H3 explicitly falsified (below null). H2 not supported. H4b not testable. None of the four primary hypotheses survive. |
| Path to full result | 0.20 | 1 | The four proposed contributions (RD threshold, cross-domain, ASI, phase transition) have each been independently falsified or not supported. The paper cannot be written as proposed. No incremental fix bridges a falsified direction reversal or a null cross-domain finding. |
| Novelty (from report) | 0.15 | 4 | Novelty_score=8 from novelty_report.json. Cross-domain and phase transition sub-contributions are 9/10 novelty. High novelty survives, but it is novelty for contributions that cannot be empirically demonstrated. |
| Resource efficiency | 0.10 | 2 | ~9 GPU hours estimated; minimal cost. But GPT-2 Small was chosen as pilot — scaling to Gemma 2 2B would be required for the original program, with no guarantee results will improve. The probability-adjusted expected return is very low. |

**Weighted score: (0.30×1) + (0.25×1) + (0.20×1) + (0.15×4) + (0.10×2) = 0.30 + 0.25 + 0.20 + 0.60 + 0.20 = 1.55**

Verdict for cand_a: PIVOT (score 1.55 << 2.5 threshold)

---

### cand_b: Absorption-as-Representational-Diagnostic (Backup 1)

The pivot trigger for cand_b was: "Theory fails empirically (AUROC < 0.65) AND cross-domain absorption rates are indistinguishable from null controls." Both triggers are now satisfied.

| Criterion | Weight | Score (1-5) | Evidence |
|---|---|---|---|
| Pilot signal strength | 0.30 | 3 | The negative C2 finding IS a signal: concept latents actively fire, no absorption. This means the absorption graph for first-letter is qualitatively different from semantic hierarchies — the diagnostic value of the absorption graph's structure is still an open question. EDA (AUROC=0.681) is a strong encoder-decoder signal worth characterizing. |
| Hypothesis survival | 0.25 | 3 | cand_b's hypothesis ("absorption rate correlates with hierarchical entanglement across layers") has not been tested. The cross-layer B2/E1 data (11 SAE configs, all high absorption) provides a starting point. The key sub-hypothesis — "the absorption graph has non-trivial clustering structure" — is genuinely untested. |
| Path to full result | 0.20 | 3 | Clear path: (1) cross-layer absorption profile from existing B2/E1 data (all layers already measured), (2) construct absorption graph using EDA-validated approach, (3) test cross-run consistency, (4) compare Matryoshka vs. TopK absorption graphs. Moderate complexity, plausible execution. Risk: Paulo and Belrose (2025) consistency concern (~0.80 across seeds). |
| Novelty (from report) | 0.15 | 3.5 | Novelty_score=7 with "modify" recommendation. Novel diagnostic framing not in literature. But must address feature consistency risk explicitly. |
| Resource efficiency | 0.10 | 4 | Low incremental compute. Much of the infrastructure (SAE loading, absorption measurement, probe training) is already working. Estimated 5 GPU hours for the diagnostic framing variant. |

**Weighted score: (0.30×3) + (0.25×3) + (0.20×3) + (0.15×3.5) + (0.10×4) = 0.90 + 0.75 + 0.60 + 0.525 + 0.40 = 3.175**

Verdict for cand_b: REFINE (score 3.175, in the 2.5–3.5 range)

---

### cand_c: Systematic Mitigation Benchmark (Backup 2)

| Criterion | Weight | Score (1-5) | Evidence |
|---|---|---|---|
| Pilot signal strength | 0.30 | 2 | No pilot for cand_c was run. F2 (Matryoshka vs. TopK) exists and shows infrastructure works. But the core contribution (head-to-head benchmark) has no signal yet. |
| Hypothesis survival | 0.25 | 2 | "No single method dominates" is a reasonable prior, but has no pilot support. SAEBench collision (novelty=5) is significant. |
| Path to full result | 0.20 | 3 | Technically feasible using pre-trained SAEs from SAEBench. But SAEBench already covers most of the ground. Low differentiation risk. |
| Novelty (from report) | 0.15 | 2 | Novelty_score=5; SAEBench collision is "partial_overlap" but substantial. Recommended as "supplement" not standalone. |
| Resource efficiency | 0.10 | 3 | Low compute. But expected publication ceiling is workshop/findings paper. |

**Weighted score: (0.30×2) + (0.25×2) + (0.20×3) + (0.15×2) + (0.10×3) = 0.60 + 0.50 + 0.60 + 0.30 + 0.30 = 2.30**

Verdict for cand_c: PIVOT (score 2.30 < 2.5)

---

### cand_d: Feature Anchoring (Backup 3)

**DROPPED.** novelty_report.json explicitly states: "Core contribution already published in arXiv:2512.05534 — EXACT MATCH." Not evaluated.

---

## Decision Rationale

The primary candidate (cand_a) must be abandoned. This is not a matter of weak signal — three of four primary hypotheses are actively falsified (direction reversed, anti-correlated, null):

1. **H1 is falsified, not merely weak.** Absorbed features show LOWER decoder cosine similarity to parents (pos_mean=0.052) than non-absorbed features (neg_mean=0.127), p=1.3e-6. This is statistically robust movement in the wrong direction. The rate-distortion threshold lambda > sin^2(theta) predicts that absorbed pairs should have HIGHER cos^2(theta). The data says the opposite. The theory's core geometric prediction is wrong about which direction the effect runs.

2. **H3 is falsified, not merely weak.** ASI AUROC=0.422 is BELOW the null mean of 0.497. The cos^2 component of ASI actively destroys the signal that freq_ratio alone carries (0.612). There is no path to reformulating ASI that fixes a component that reliably anti-predicts.

3. **C2 is a design-confirmed negative result.** The C2 measurement pipeline was validated (v3 empirical activation discovery works; mean_max_activation=16–22 for concept latents). The finding is not a methodological artifact: concept latents in GPT-2 Small L6 reliably fire for their concept words. Cross-domain absorption in the form originally hypothesized (parent latent suppressing child features) is not present in this model/layer.

The sunk cost in cand_a is zero weight in this decision.

The correct decision is PIVOT to cand_b, subject to the following redesign:

### Why cand_b over cand_c

- cand_b has a genuine open question (cross-layer absorption diagnostic, absorption graph structure, EDA geometric interpretation) with no published answer.
- cand_b has existing infrastructure advantage: EDA is already validated (AUROC=0.681), probes exist for 3+ hierarchy types, 11-config cross-layer data from B2/E1 is in hand.
- cand_c's core question (which mitigation is best?) overlaps substantially with SAEBench (2025). Without theoretical framing, it is incremental.
- The ONLY valuable cand_c addition is OrtSAE and masked regularization — too narrow for a standalone contribution.

### Redesign Required for cand_b

The original cand_b framing assumed cross-layer absorption profile would have systematic structure. The existing E1 data (11 configs) shows absorption_rate 77–99% — extremely high and relatively flat. This means the "middle layers show more absorption" hypothesis needs revision. However, this IS a finding: absorption is saturated at near-100% across all tested configs, suggesting the SAE training always converges to the absorbed solution in this model.

New research questions worth pursuing:

1. **Why does EDA predict absorbed features (AUROC=0.681) but cos^2(theta) anti-predicts (AUROC=0.318)?** This is a concrete mechanistic puzzle: encoder-decoder misalignment is the marker of absorption, NOT decoder-decoder similarity. This inverts the naive geometric intuition and is a publishable discovery.

2. **Is the absorption graph structurally consistent across SAE training seeds?** Paulo and Belrose (2025) show feature inconsistency (~0.80) but did not study absorption specifically. Testing absorption graph consistency would directly address whether absorption is a reliable property of the model or an artifact of SAE training variance.

3. **Does the Matryoshka absorption reduction actually recover decoder geometry (increase cos^2 back to non-absorbed range) or merely push encoder-decoder alignment back (reduce EDA)?** The F2 experiment infrastructure exists. This question has not been answered.

4. **Can EDA (AUROC=0.681) be used for unsupervised broad-spectrum absorption detection?** The original H3 (ASI) failed, but EDA succeeds. The practical utility of probe-free detection could pivot to EDA-based scanning.

---

## Next Actions

The pivot is to cand_b with a focused redesign. Priority experiments:

1. **IMMEDIATE (reuse existing data):** Compute EDA-based absorption graph for GPT-2 Small L6 using existing B1/D2 results. Characterize structural properties (in-degree distribution, clustering, letter-specific absorber concentration). Estimated time: 30 minutes.

2. **SHORT TERM (1 hour):** Test cross-seed absorption graph consistency: train/load 2 different SAE seeds for GPT-2 Small L6 (if available in SAELens), compute EDA-based absorption graph for each, measure graph-level Jaccard similarity. This tests the Paulo-Belrose concern directly.

3. **SHORT TERM (1 hour):** Use existing F2 results (Matryoshka vs. TopK) to test whether Matryoshka reduces EDA (encoder-decoder misalignment) or just cos^2. This distinguishes "genuine recovery" from "re-encoding hierarchy."

4. **MEDIUM TERM (2 hours):** Cross-layer EDA absorption profile: for the 11 SAE configs already measured, compute EDA score distribution per layer. Does EDA magnitude vary systematically by layer? Does the L10 EDA AUROC=0.337 vs. L6 AUROC=0.681 represent a real phenomenon or sampling variance?

5. **Frame paper around the INVERSE DIRECTION FINDING:** The most defensible contribution is: "We sought to validate the rate-distortion theory of absorption; the evidence falsified the geometric prediction (cos^2 anti-correlates). Instead, encoder-decoder misalignment (EDA) is the empirical marker of absorption. We characterize when EDA works, what the absorption graph looks like, and whether architectural mitigations target the correct geometric property."

**What NOT to pursue:**
- Any further work on the RD threshold (H1) or ASI (H3) in their current formulations. Both are falsified.
- Hysteresis experiment (H4b): not testable because the system is already in the saturated absorbed phase.
- Feature anchoring (cand_d): already published.

SELECTED_CANDIDATE: cand_b
CONFIDENCE: 0.62
DECISION: PIVOT


## 上一轮 validation 结构化决策
{
  "decision": "PIVOT",
  "selected_candidate_id": "cand_b",
  "confidence": 0.62,
  "timestamp": "2026-04-13",
  "evaluator": "sibyl-idea-validation-decision",
  "candidate_scores": {
    "cand_a": {
      "weighted_score": 1.55,
      "verdict": "PIVOT",
      "score_breakdown": {
        "pilot_signal_strength": {"weight": 0.30, "score": 1, "contribution": 0.30},
        "hypothesis_survival": {"weight": 0.25, "score": 1, "contribution": 0.25},
        "path_to_full_result": {"weight": 0.20, "score": 1, "contribution": 0.20},
        "novelty": {"weight": 0.15, "score": 4, "contribution": 0.60},
        "resource_efficiency": {"weight": 0.10, "score": 2, "contribution": 0.20}
      },
      "falsified_hypotheses": ["H1", "H3"],
      "not_supported_hypotheses": ["H2", "H4a"],
      "not_testable_hypotheses": ["H4b"],
      "summary": "4/5 primary hypotheses falsified or not supported. H1 direction reversal (absorbed features have LOWER not HIGHER cos^2, p=1.3e-6). H3 AUROC=0.422 below null mean 0.497. H2: absorption_rate=0.0 for all non-spelling hierarchies. H4a: LRT p=0.456. No credible path to paper."
    },
    "cand_b": {
      "weighted_score": 3.175,
      "verdict": "PIVOT_TO_REFINE",
      "score_breakdown": {
        "pilot_signal_strength": {"weight": 0.30, "score": 3, "contribution": 0.90},
        "hypothesis_survival": {"weight": 0.25, "score": 3, "contribution": 0.75},
        "path_to_full_result": {"weight": 0.20, "score": 3, "contribution": 0.60},
        "novelty": {"weight": 0.15, "score": 3.5, "contribution": 0.525},
        "resource_efficiency": {"weight": 0.10, "score": 4, "contribution": 0.40}
      },
      "active_signals": [
        "EDA AUROC=0.681 (above null, consistent with iter_001 reference=0.629)",
        "Inverse direction finding: absorbed features have LOWER cos^2 — inverts the naive theory",
        "C2 negative result: semantic concept latents DO fire reliably — absorption appears domain-specific",
        "C1 probes validated: first_letter F1=0.820, noun_proper F1=0.987, animate_inanimate F1=1.0",
        "11-config cross-layer data already collected (layers 2-12)"
      ],
      "risks": [
        "Paulo and Belrose (2025): ~0.80 feature consistency across SAE seeds may limit absorption graph stability",
        "Cross-layer absorption rates are uniformly very high (77-99%) leaving limited variation to characterize"
      ],
      "redesign_required": "Shift from 'absorption is universal and characterized by cos^2' to 'encoder-decoder misalignment (EDA) is the marker of absorption; we invert the theoretical prediction and characterize when/why this holds'"
    },
    "cand_c": {
      "weighted_score": 2.30,
      "verdict": "PIVOT",
      "summary": "SAEBench already covers most of the ground (novelty=5, collision=partial_overlap). Incremental without theoretical framing. Lower expected venue than cand_b."
    },
    "cand_d": {
      "weighted_score": 0,
      "verdict": "DROPPED",
      "summary": "Core contribution already published in arXiv:2512.05534 (exact match). Feature anchoring validated on real LLMs in that paper. Not viable."
    }
  },
  "pivot_trigger": {
    "condition_1": "Theory fails empirically (RD threshold AUROC < 0.65): SATISFIED — AUROC=0.410, direction reversed",
    "condition_2": "Cross-domain absorption rates indistinguishable from null controls: SATISFIED — absorption_rate=0.0 for animate, proper_noun, first_letter in C2",
    "additional_falsification": "ASI (H3) AUROC=0.422 below null mean 0.497 — anti-correlated"
  },
  "reasons": [
    "H1 FALSIFIED: absorbed letter features have LOWER cos^2(theta) than non-absorbed (pos=0.052 vs neg=0.127, Cohen's d=-0.57, p=1.3e-6). The rate-distortion threshold predicts the wrong direction.",
    "H3 FALSIFIED: ASI AUROC=0.422 is below null mean 0.497. The cos^2 component (AUROC=0.206) actively hurts the composite vs freq_ratio alone (AUROC=0.612).",
    "H2 NOT SUPPORTED: All non-spelling hierarchies show absorption_rate=0.0 with concept latents reliably active (mean_max_activation=16-22). Traditional absorption not detectable in GPT-2 Small L6 for semantic hierarchies.",
    "H4a NOT SUPPORTED: Linear model fits absorption-vs-sparsity better than sigmoid (BIC diff=-3.22, LRT p=0.456). Phase transition framing is not empirically justified.",
    "cand_b PIVOT TRIGGER is met and represents a scientifically interesting pivot: the inverse geometric finding (EDA works, cos^2 anti-works) is itself a publishable discovery that inverts a naive theoretical prediction."
  ],
  "next_actions": [
    "Redesign proposal around cand_b: 'Absorption-as-Representational-Diagnostic with EDA as probe-free detector'",
    "Reframe: the key discovery is that encoder-decoder misalignment (EDA, AUROC=0.681) predicts absorption while decoder-decoder cosine similarity anti-predicts (AUROC=0.318) — this inverts the rate-distortion theory's geometric prediction",
    "Use existing 11-config B2/E1 cross-layer data to compute EDA-based absorption profile across layers",
    "Test absorption graph consistency across SAE training seeds (addresses Paulo-Belrose 2025 concern)",
    "Use existing F2 data to test whether Matryoshka reduces absorption via EDA reduction or cos^2 shift",
    "Characterize EDA as probe-free absorption detector: precision/recall at different EDA thresholds, comparison to Tian et al. feature sensitivity metric",
    "Frame paper: 'We sought to validate rate-distortion theory; data falsified the geometric prediction. Instead, encoder-decoder misalignment is the structural marker of absorption. We characterize absorption at the SAE geometric level and establish when EDA-based detection is reliable.'"
  ],
  "dropped_candidates": ["cand_a", "cand_c", "cand_d"],
  "what_works_from_cand_a": {
    "F1_theory_derivation": "Complete non-circular algebraic proof of Theorem 1 retained — publishable in revised form as motivation/background",
    "C1_probe_infrastructure": "All probes trained and validated (first_letter, noun_proper, animate_inanimate). Reusable.",
    "EDA_detector": "AUROC=0.681 is the positive finding. This is the anchor for the new framing.",
    "B2_E1_cross_layer_data": "11 SAE configs measured. Raw data available for absorption graph and cross-layer profile analysis.",
    "F2_mitigation_data": "Matryoshka vs. TopK decoder geometry comparison available. Reusable for architectural analysis."
  },
  "confidence_rationale": "0.62 reflects: (1) the pivot trigger is clearly met and the direction is unambiguous; (2) cand_b's core question (absorption graph structure, EDA interpretation) is genuinely novel and empirically tractable; (3) moderate uncertainty because the Paulo-Belrose consistency concern is non-trivial and the compressed cand_b proposal has not been tested at full scale; (4) EDA's L10 AUROC=0.337 vs. L6 AUROC=0.681 suggests the detector is not universal, which limits the scope of the contribution."
}


## 上一轮新颖性检查报告（必须针对发现的撞车问题进行修正）
# Novelty Report: SAE Feature Absorption Research Proposal

**Search date:** April 2026
**Workspace:** `/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current`
**Candidates assessed:** cand_a (front-runner), cand_b, cand_c, cand_d

---

## Summary

Overall novelty: **HIGH** — the core contributions of the front-runner (cand_a) are not anticipated by any single existing paper. Multiple pieces of prior art are relevant but none executes the unified program proposed here. Key risks are precisely characterized below.

---

## Candidate A: Rate-Distortion Theory + Cross-Domain + ASI + Phase Transition (Front-Runner)

### Core claims and search findings

**Claim 1: Formal proof that feature absorption is rate-distortion optimal (closed-form threshold lambda > sin^2(theta_{p,c}))**

*Prior art found:*
- Chanin et al. (2024) "A is for Absorption" (arXiv:2409.14507, NeurIPS 2025 Oral) — the canonical paper on feature absorption. Provides informal explanation: absorption "saves one L0 per parent-child pair." Does NOT formalize a closed-form threshold or prove optimality.
- Tilde Research Blog "The Rate Distortion Dance of Sparse Autoencoders" — discusses rate-distortion framing of SAEs, including the L1/L0 Lagrangian. Does NOT derive an absorption threshold as a function of decoder angle. The blog addresses the shrinkage and load-balancing problems, not a geometric absorption criterion.
- Tang et al. (2024) "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima" (arXiv:2512.05534) — proves that spurious partial minima exhibiting feature absorption exist in the piecewise biconvex landscape. Establishes necessary and sufficient conditions for correct feature recovery. Does NOT derive a closed-form absorption threshold as a function of decoder angle and sparsity penalty. Does NOT provide a quantitative prediction of which feature pairs are at risk.
- Chanin, Dulka, Garriga-Alonso (2025) "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders" (arXiv:2505.11756) — analyzes feature hedging (distinct from absorption) using a parameterized family of encoder/decoder solutions. Does NOT provide the rate-distortion absorption threshold.

*Verdict:* **Partial overlap only.** The informal observation that "absorption saves L0" is established. The piecewise biconvex theory (2512.05534) is the closest theoretical work but studies a different question (existence of spurious minima, not quantitative geometric threshold). No paper derives `lambda > sin^2(theta_{p,c})` as the absorption criterion or uses it for quantitative prediction. **Novelty: HIGH for the formal derivation.**

*Differentiation:* Our contribution is a falsifiable, quantitative threshold (closed-form function of measurable quantities: lambda, decoder angle, co-occurrence frequency), plus the key insight that co-occurrence frequency cancels from the absorption criterion. This makes the threshold far more powerful for prediction and hyperparameter guidance than any existing theoretical result.

---

**Claim 2: First cross-domain characterization of absorption across multiple semantic hierarchy types**

*Prior art found:*
- Chanin et al. (2024) — measures absorption exclusively on the first-letter spelling task (syntactic hierarchy). The paper itself notes this limitation.
- SAEBench (Karvonen et al., 2025, arXiv:2503.09532) — implements absorption metric using first-letter features only. The SAEBench documentation explicitly states: "SAEBench evaluates feature absorption by using features for 'word starts with X', which is not useful for evaluating domain-specific feature absorption."
- Matryoshka SAE paper (Bussmann et al., 2025, arXiv:2503.17547) — evaluates absorption on the first-letter task and Gemma Scope SAEs, but only on this single task type.
- RAVEL dataset (Huang et al., 2024) — provides entity-attribute data including city→country hierarchies. Has been integrated into SAEBench for RAVEL disentanglement tasks, but NOT for absorption measurement specifically.
- OrtSAE (Korznikov et al., 2025, arXiv:2509.22033) — evaluates absorption on qualitative decomposition experiments but not on systematic cross-domain hierarchy measurements.

*Verdict:* **No collision.** No paper has measured absorption rates across multiple semantic hierarchy types (entity-type, geographic, grammatical) and compared them. The gap is explicitly acknowledged in the SAEBench documentation. **Novelty: VERY HIGH.**

---

**Claim 3: Probe-free Absorption Susceptibility Index (ASI) from decoder geometry**

*Prior art found:*
- Chanin et al. (2024) — absorption metric requires specifying probe directions in advance. Explicitly supervision-dependent.
- SAEBench absorption metric — filters candidate absorbing latents by cosine similarity with ground-truth probe above a threshold. Requires probe. Does NOT predict absorption without probe directions.
- Tian et al. (2025) "Measuring Sparse Autoencoder Feature Sensitivity" (arXiv:2509.23717, NeurIPS 2025 Workshop Spotlight) — measures feature sensitivity (how reliably a feature fires on texts similar to activating examples) using GPT-4.1 to generate similar texts. Does not use decoder geometry and does not provide absorption prediction. A different concept (sensitivity vs. absorption susceptibility).
- Bricken et al. (2023) / ICLR 2025 paper — uses decoder cosine similarity to classify latent types (thresholding at 0.7 for GPT-2, 0.4 for GemmaScope). Reports ROC curves for using cosine threshold to predict latent behavior. This is the closest precursor to ASI but: (a) classifies latent types, not feature pairs; (b) does not multiply by frequency ratio; (c) does not frame as absorption susceptibility prediction; (d) does not compute AUROC against absorption labels.
- ATM (2510.08855) — tracks activation magnitudes, frequencies, and reconstruction contributions as importance scores during training. Post-hoc importance score (not probe-free absorption predictor), requires training run.

*Verdict:* **Partial overlap with Bricken et al. decoder cosine similarity work.** The ASI formula `cos^2(theta) * (freq_p / freq_c)` adds the frequency ratio term, frames the metric as a probe-free absorption predictor for feature pairs, and validates against absorption labels. The combination is novel; the decoder cosine similarity component is anticipated but not in this application context. **Novelty: HIGH, needs clear differentiation from Bricken et al.**

*Differentiation notes:* (1) ASI is a pairwise metric for feature pairs, not a per-feature metric. (2) The frequency ratio term `freq_p / freq_c` captures asymmetry of parent→child relationships, absent from any existing metric. (3) Validation against absorption labels (AUROC against Chanin et al. ground truth) is new. (4) The framing as probe-free, training-free absorption susceptibility prediction without knowing which features to look for is entirely new.

---

**Claim 4: Phase transition and hysteresis characterization of absorption onset**

*Prior art found:*
- No paper found on "SAE absorption phase transition hysteresis." Zero results for "sparse autoencoder absorption phase transition hysteresis fine-tuning reversibility."
- General phase transition literature in sparse learning (arXiv:2411.17180) addresses phase transitions in feature selection/support recovery for linear models, not SAE absorption specifically.
- Sparse PCA phase transitions (arXiv:2412.21038) — analogous BBP transition in PCA, not SAE absorption.
- The relationship between sparsity penalty (lambda) and resulting L0 is known to be nonlinear and model-dependent (mentioned by multiple SAEBench authors), but this has not been analyzed as a phase transition with falsifiable functional form predictions.

*Verdict:* **No collision.** Phase transition and hysteresis framing of SAE absorption is entirely absent from the literature. **Novelty: VERY HIGH.** This is the most novel sub-contribution.

---

### Cand A Overall Assessment

| Sub-contribution | Novelty Score | Collisions | Severity |
|---|---|---|---|
| Rate-distortion formal proof + closed-form threshold | 8/10 | Tilde Blog (informal), 2512.05534 (spurious minima proof) | partial_overlap |
| Cross-domain absorption characterization | 9/10 | None | None |
| Probe-free ASI | 7/10 | Bricken et al. decoder cosine similarity | partial_overlap |
| Phase transition + hysteresis | 9/10 | None | None |

**Cand A Novelty Score: 8/10**
**Recommendation: PROCEED**

Key differentiation needed:
- Clearly distinguish from 2512.05534 (biconvex theory studies existence of spurious minima; our theory provides a closed-form geometric threshold for specific feature pairs — different question, different answer)
- Clearly distinguish ASI from Bricken et al. cosine similarity classifier (different application, different formula, different validation)
- Cite Tilde Research blog as motivating observation and explicitly position formal derivation as the extension

---

## Candidate B: Absorption-as-Representational-Diagnostic

*Core claim:* The absorption graph encodes the model's internal feature hierarchy; cross-layer absorption profiles reveal computational structure.

*Prior art:*
- Chanin et al. (2024) studies absorption systematically but does not construct or analyze the absorption graph structure.
- Song et al. (2025) — features are only ~0.80 consistent across SAE runs (acknowledged in the proposal). This is a significant risk.
- OrtSAE (2509.22033) and Matryoshka (2503.17547) papers analyze absorption reductions, not absorption graph structure as a representation diagnostic.
- No paper frames absorption as a representational diagnostic or constructs an absorption graph for structural analysis.

**Cand B Novelty Score: 7/10 — Proceed only as fallback.**

*Risk:* Absorption graph inconsistency across SAE runs (Song et al.) could undermine the diagnostic interpretation. The conceptual framing is novel but relies heavily on the empirical question of whether the absorption graph is consistent.

---

## Candidate C: Systematic Mitigation Benchmark

*Core claim:* Controlled head-to-head comparison of Matryoshka, OrtSAE, ATM, masked regularization on same model/layer/metric.

*Prior art found:*
- SAEBench (2503.09532) — implements multiple metrics and compares architectures. Matryoshka and OrtSAE are both benchmarked. However, SAEBench comparison focuses on overall performance metrics, not a controlled head-to-head specifically on absorption trade-offs.
- Matryoshka SAE paper (2503.17547) — compares vs. BatchTopK, not vs. OrtSAE or ATM.
- OrtSAE paper (2509.22033) — compares vs. Matryoshka but not ATM.
- No single paper compares all four: Matryoshka, OrtSAE, ATM, masked regularization on the same absorption + downstream metrics.

**Cand C Novelty Score: 5/10 — Proceed only as supplement to cand_a.**

*Risk:* SAEBench already provides a partial comparison. Without theoretical framing from cand_a or cand_b, this is below top-venue bar.

---

## Candidate D: Feature Anchoring for Real LLM Absorption Reduction

*Core claim:* First empirical evaluation of feature anchoring on real LLMs for absorption reduction.

*Prior art:*
- Tang et al. (2512.05534) proposes feature anchoring as a novel technique and validates it on synthetic benchmarks and real neural representations. The paper explicitly claims real neural representation validation.

**COLLISION DETECTED:** Tang et al. (2512.05534) already introduces and evaluates feature anchoring. The proposal for cand_d claims "currently only validated on synthetic benchmarks" — this is INCORRECT based on our search. The paper explicitly validates "its effectiveness with extensive experiments across diverse SDL methods and settings" including real representations.

**Cand D Novelty Score: 3/10 — DROP.**

*Verdict:* exact_match on the core contribution. Cand D should be dropped. The only residual novelty would be testing anchoring on absorption specifically (Chanin metric) vs. feature recovery generally — a narrow incremental claim.

---

## Key Prior Work Citation Summary

| Paper | Relevance to Proposal | Severity |
|---|---|---|
| Chanin et al. (2024) arXiv:2409.14507 | Foundation: defines absorption, first-letter task, absorption metric | related_work |
| Tang et al. (2024) arXiv:2512.05534 | Partial: piecewise biconvex theory explains absorption as spurious minima; feature anchoring | partial_overlap |
| Tilde Research Blog (2024) | Partial: informal rate-distortion framing of SAEs | partial_overlap |
| Bussmann et al. (2025) arXiv:2503.17547 | Related: Matryoshka SAEs reduce absorption on first-letter task | related_work |
| Korznikov et al. (2025) arXiv:2509.22033 | Related: OrtSAE reduces absorption via decoder orthogonality | related_work |
| Karvonen et al. (2025) arXiv:2503.09532 | Related: SAEBench benchmarks absorption (first-letter only) | related_work |
| Chanin, Dulka et al. (2025) arXiv:2505.11756 | Related: Feature hedging — different failure mode, same domain | related_work |
| Tian et al. (2025) arXiv:2509.23717 | Related: Feature sensitivity (different concept from absorption susceptibility) | related_work |
| Bricken et al. (2023/2025) | Partial overlap: decoder cosine similarity to predict latent behavior | partial_overlap |
| ATM (arXiv:2510.08855) | Related: adaptive temporal masking reduces absorption 40% | related_work |
| MDL-SAEs Ayonrinde et al. (2024) | Related: compression/MDL language for SAEs | related_work |

---

## Recommendations by Candidate

| Candidate | Recommendation | Rationale |
|---|---|---|
| cand_a | PROCEED — front-runner | Four distinct novel contributions; partial overlaps are defensible and clearly differentiable. Overall novelty 8/10. |
| cand_b | MODIFY — backup only | Conceptually novel framing (absorption graph as diagnostic) but feature consistency risk is real. Proceed as fallback if cand_a theory fails. |
| cand_c | SUPPLEMENT — not standalone | Partial collision with SAEBench; below top-venue bar without theoretical framing. |
| cand_d | DROP | Core contribution (feature anchoring on real LLMs) already published in arXiv:2512.05534. |

---

## Anti-Patterns Avoided

- The rate-distortion framing is NOT the same as the Tilde Research blog: the blog is informal and addresses the general SAE optimization landscape; we derive a closed-form geometric threshold with falsifiable quantitative predictions.
- The ASI is NOT the same as Bricken et al. decoder cosine similarity: different metric (pairwise, with frequency ratio), different application (absorption susceptibility for unknown features), different validation (AUROC against Chanin labels).
- "Cross-domain" in our proposal refers to semantic hierarchy types (entity-type, geographic, grammatical), not cross-model or cross-architecture — different from Universal SAEs paper.
