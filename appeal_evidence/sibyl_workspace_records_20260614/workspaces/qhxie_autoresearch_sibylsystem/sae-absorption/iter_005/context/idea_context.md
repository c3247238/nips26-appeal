

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

**Research Topic**: Feature Absorption in Sparse Autoencoders (SAEs): Systematic Analysis and Quantification of Causes, Patterns, and Implications for Interpretability
**Survey Date**: 2026-04-14
**arXiv Search Keywords**: "feature absorption" "feature splitting" sparse autoencoders; "sparse autoencoder" "mechanistic interpretability" features superposition; SAEBench evaluation sparse autoencoders; Gated SAE JumpReLU architecture; Matryoshka SAE hierarchical; feature hedging correlated; "survey sparse autoencoders" LLM; theoretical sparse autoencoders identifiable; SAE dark matter reconstruction; "feature manifolds" scaling; KronSAE features correlation; unified theory sparse dictionary learning; transcoder interpretability absorption; SynthSAEBench synthetic hierarchy; adaptive temporal masking SAE absorption; CE-Bench contrastive evaluation SAE; feature sensitivity sparse autoencoder
**Web Search Keywords**: sparse autoencoder feature absorption interpretability 2025; SAE feature absorption Chanin 2024 arxiv 2409.14507; Matryoshka SAE hierarchical feature absorption; SAE feature absorption solutions mitigation OrtSAE masked regularization; SAE superposition toy model Anthropic survey; sparse autoencoder absorption sparsity penalty L1 L0 hierarchical theoretical; feature anchoring unified theory SDL absorption dead neurons; KronSAE Kronecker factorization absorption metrics; transcoders beat SAEs interpretability; DeepMind negative results SAE safety

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant unsupervised tool in mechanistic interpretability for decomposing the polysemantic, superposed activations of large language models (LLMs) into approximately monosemantic, human-interpretable feature directions. The foundational motivation comes from Anthropic's "Toy Models of Superposition" (2022) and the subsequent "Towards Monosemanticity" (2023), which established that neural networks encode far more features than they have neurons by representing features as overlapping linear directions—a phenomenon called superposition. SAEs address this by projecting activations into an overcomplete sparse basis via a sparsity-inducing objective (L1 penalty or TopK activation), with the hope of recovering the underlying "true" monosemantic features.

The field advanced rapidly through 2024–2025: Anthropic and OpenAI scaled SAEs to frontier models (Claude 3 Sonnet, GPT-4), the community developed several improved architectures (Gated SAE, JumpReLU SAE, TopK SAE, Matryoshka SAE, OrtSAE), and comprehensive benchmarks (SAEBench, CE-Bench) emerged to standardize evaluation. However, alongside these successes came a growing body of negative results that challenge whether SAEs robustly recover the mechanistically meaningful features they promise. Three failure modes have attracted particular attention: (1) **feature absorption**—where a general feature silently fails to fire because a more specific co-occurring feature encodes the same information; (2) **feature hedging**—where correlated features are merged due to insufficient SAE width; and (3) **feature inconsistency**—where independent training runs converge to different feature sets. Feature absorption is arguably the most fundamental of these, as it creates a false sense of interpretability: the SAE appears to have learned a clean monosemantic latent, but that latent has systematic "holes" in its recall.

The current state of the field (early 2026) is one of productive tension: SAEs remain the most tractable approach to large-scale mechanistic interpretability, but their theoretical guarantees are weak, and several recent papers show that even well-trained SAEs may only marginally outperform random baselines on downstream tasks. Notably, DeepMind's safety research team publicly deprioritized SAE research in 2025 after finding that dense linear probes dramatically outperform SAE probes on safety-relevant downstream tasks (e.g., harmful intent detection), with feature absorption being a key culprit. Meanwhile, Anthropic successfully used SAE-based features for circuit tracing in Claude 3.5 Haiku ("On the Biology of a Large Language Model", Lindsey et al., 2025), demonstrating that when features are reliable, they enable powerful mechanistic understanding—making absorption a critical obstacle to broader adoption.

The community is actively developing architectural solutions (Matryoshka SAEs, OrtSAE, KronSAE, masked regularization), alternative paradigms (transcoders, crosscoders), and theoretical frameworks (identifiability analysis, feature anchoring, MDL-based training) to address these shortcomings. A unified theoretical framework (arXiv:2512.05534) now casts all major sparse dictionary learning methods as a single piecewise biconvex optimization problem and provides principled explanations for absorption—though feature anchoring, its proposed remedy, has only been validated on synthetic benchmarks. Additionally, foundational assumptions of SAEs have been challenged: Leask et al. (ICLR 2025) show SAE features are neither canonical nor atomic, Engels et al. (ICLR 2025) discover irreducible multi-dimensional features that SAEs may fundamentally misrepresent, and Chanin & Garriga-Alonso (2025) demonstrate that incorrect L0 (which is common in practice) causes SAEs to learn systematically wrong features. Feature absorption specifically sits at the intersection of theory (caused by sparsity optimization under feature hierarchy) and practice (undermines causal circuit analysis and safety-relevant feature detection), making it one of the most important open problems in SAE-based interpretability.

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
| 28 | Sparse Autoencoders Do Not Find Canonical Units of Analysis | arXiv:2502.04878 (ICLR 2025) | 2025 | Shows SAE features are neither complete nor atomic using SAE stitching and meta-SAEs; novel latents in larger SAEs capture info missed by smaller SAEs; meta-SAEs decompose features into sub-features; fundamentally challenges the canonical feature assumption | Does not study absorption directly; focused on non-canonicality and compositionality |
| 29 | Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders | arXiv:2508.16560 | 2025 | Demonstrates that incorrect L0 causes SAEs to mix correlated features ("cheat"); too-low L0 triggers feature hedging, too-high L0 finds degenerate solutions; proposes proxy metric for finding correct L0; most open-source SAEs have L0 that is too low | Builds on toy model analysis; relationship to absorption not fully characterized |
| 30 | Not All Language Model Features Are One-Dimensionally Linear | arXiv:2405.14860 (ICLR 2025) | 2024 | Discovers irreducible multi-dimensional features (circular representations for days/months) in GPT-2 and Mistral 7B; challenges the one-dimensional linear representation hypothesis; shows multi-dimensional features are used for modular arithmetic computation | SAEs may miss or distort multi-dimensional features; relationship to absorption of multi-dimensional features unexplored |
| 31 | On the Biology of a Large Language Model | transformer-circuits.pub (Anthropic 2025) | 2025 | Introduces attribution graphs for tracing computational circuits in Claude 3.5 Haiku; reveals planning-ahead, hallucination default-refusal, and jailbreak mechanisms; demonstrates practical utility of feature-level interpretability for understanding model behavior | Uses SAE features but does not specifically study absorption; shows downstream value of reliable feature decomposition |
| 32 | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | arXiv:2408.05147 (BlackboxNLP 2024) | 2024 | Releases 400+ open JumpReLU SAEs on all layers/sublayers of Gemma 2 2B/9B/27B; widths 1k-1M; primary evaluation target for absorption research; Gemma Scope 2 (Dec 2024) adds Matryoshka SAEs and transcoders on Gemma 3 | JumpReLU architecture may worsen absorption; massive resource for community but trained with single architecture family |
| 33 | Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words | arXiv:2501.06254 | 2025 | Evaluates SAEs specifically on polysemous words; finds performance ranking ReLU > JumpReLU > TopK on F1; challenges assumption that better MSE-L0 frontier implies better feature quality; reveals architecture-specific weaknesses for polysemantic inputs | Narrow evaluation domain (polysemous words); does not directly measure absorption but highlights related failure mode |

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

**Gap 9: Confounding of absorption with incorrect L0 and feature hedging.**
Chanin & Garriga-Alonso (2025) show most open-source SAEs have L0 that is too low, causing feature hedging (correlated feature mixing). Feature hedging and absorption are distinct phenomena with different causes (width/L0 insufficient vs. hierarchical feature structure), but both manifest as features failing to fire where they should. No study systematically disentangles the two: measuring how much of observed "absorption" in practice is actually L0-induced hedging vs. true hierarchy-driven absorption. This distinction is critical for choosing the right mitigation strategy.

**Gap 10: Absorption of multi-dimensional features is unstudied.**
Engels et al. (ICLR 2025) show some features are irreducibly multi-dimensional (e.g., circular representations for days/months). SAEs forced to represent these as one-dimensional directions may exhibit novel absorption-like pathologies where parts of the multi-dimensional feature are absorbed by co-occurring one-dimensional features. The interaction between feature dimensionality and absorption has not been investigated.

**Gap 11: No study of absorption across SAE architectures under matched conditions on the polysemantic word task.**
Minegishi et al. (2025) show that ReLU > JumpReLU > TopK on polysemous word F1 evaluation, suggesting architecture choice affects how SAEs handle correlated/overlapping features. Whether this ranking holds specifically for absorption (vs. general polysemanticity handling) has not been tested.

---

## 5. Available Resources

### Open-Source Code
- **sae-spelling** (canonical absorption study): https://github.com/lasr-spelling/sae-spelling — Chanin et al. absorption experiments on first-letter task; absorption rate metric implementation
- **sparse-but-wrong-paper**: https://github.com/chanind/sparse-but-wrong-paper — Chanin & Garriga-Alonso L0 analysis experiments
- **feature-hedging-paper**: https://github.com/chanind/feature-hedging-paper — Chanin et al. feature hedging analysis code
- **SAELens**: https://github.com/decoderesearch/SAELens (also https://github.com/jbloomAus/SAELens) — Standard SAE training and evaluation library; supports all major architectures; deep integration with TransformerLens and HuggingFace Transformers
- **Language-Model-SAEs** (OpenMOSS): https://github.com/OpenMOSS/Language-Model-SAEs — Full distributed SAE training with head/data/model parallelism; supports vanilla SAE, CLT, CrossCoder with ReLU/JumpReLU/TopK/BatchTopK
- **Gemma Scope**: https://huggingface.co/google/gemma-scope — 400+ pre-trained JumpReLU SAEs on Gemma 2 2B / 9B / 27B, primary evaluation target; Gemma Scope 2 adds Matryoshka SAEs and transcoders on Gemma 3
- **SAEBench**: https://github.com/adamkarvonen/SAEBench + interactive at https://www.neuronpedia.org/sae-bench — 8-metric evaluation suite including absorption
- **OrtSAE**: Available from Korznikov et al. (2025) — implementation of orthogonality penalty
- **Neuronpedia**: https://www.neuronpedia.org — Interactive feature explorer, steering, circuit tracing; now open-source; supports 50M+ latents across many models
- **SAEDashboard**: https://github.com/jbloomAus/SAEDashboard — Replicates Anthropic-style sparse autoencoder feature visualizations; supports cross-layer transcoders
- **TransformerLens**: https://github.com/TransformerLensOrg/TransformerLens — Mechanistic interpretability library for 50+ models; hook-based activation caching and editing; deep integration with SAELens
- **MultiDimensionalFeatures**: https://github.com/JoshEngels/MultiDimensionalFeatures — Engels et al. code for finding multi-dimensional features in LLMs
- **awesome-SAE**: https://github.com/zepingyu0512/awesome-SAE — Curated list of SAE papers and resources

### Datasets
- **OpenWebText**: Standard SAE pre-training corpus (used in Chanin et al., Gao et al.)
- **Pile / RedPajama**: Alternative corpora for SAE training
- **Gemma Scope activation caches**: Pre-computed for efficient SAE training and evaluation
- **First-letter spelling task** (constructed): In-context learning prompts using template `{token} has the first letter: {letter}`; test set from alphabetical token lists

### Pre-trained Models and SAEs
- **Gemma Scope SAEs**: 400+ JumpReLU SAEs on Gemma 2 (2B, 9B, 27B), all layers and sublayers, widths 1k–1M — HuggingFace: https://huggingface.co/google/gemma-scope ; Gemma Scope 2 adds Matryoshka SAEs + transcoders on Gemma 3
- **GPT-2 SAEs** (EleutherAI/OpenAI): Standard evaluation target for reproducibility; available on Neuronpedia (GPT2-Small v5)
- **Llama 3.2 1B / 3B SAEs**: Available via SAELens; used in Chanin et al. absorption validation
- **Pythia SAEs**: Available via SAELens; Pythia-70M/160M/410M used in foundational SAE work (Cunningham et al., ICLR 2024)
- **Claude 3 Sonnet SAEs** (Anthropic, internal): 1M/4M/34M feature SAEs; not publicly released
- **GPT-4 SAE** (OpenAI, internal): 16M latent SAE; not publicly released

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

9. **L0 as a confound in absorption studies**: Chanin & Garriga-Alonso (2025) show most open-source SAEs have L0 that is too low, causing feature hedging/mixing. Disentangling the effects of incorrect L0 from true absorption (which occurs even at correct L0 due to feature hierarchy) would sharpen the definition and measurement of absorption.

10. **Non-canonicality and absorption**: Leask et al. (ICLR 2025) show SAE features are neither complete nor atomic. Meta-SAE decomposition of features may provide a novel lens for detecting which "atomic" features are actually absorbing parent features, connecting non-canonicality to absorption mechanistically.

11. **Multi-dimensional features and absorption**: Engels et al. (ICLR 2025) show some features are irreducibly multi-dimensional (circular representations). SAEs forced to represent these as one-dimensional may exhibit absorption-like behavior where the multi-dimensional feature is partially captured by multiple latents. This interaction between multi-dimensionality and absorption is completely unexplored.

**Cross-domain analogies with potential:**
- Hierarchical dictionary learning in signal processing / compressed sensing (multi-scale wavelet decompositions) may provide theoretical tools for analyzing feature hierarchy in SAEs
- Multi-label classification with label implication (where absorbing a specific label "absorbs" the general label) has been studied in the NLP literature and may provide relevant algorithmic solutions
- Neuroscience: Anthropic's circuit tracing work ("On the Biology of a Large Language Model") shows the practical value of reliable feature decomposition for understanding model computation; absorption directly undermines this program

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
| TransformerLens | High | MIT | Adopt | Standard mech interp library with hook-based activation access; deep integration with SAELens; 50+ model support; needed for activation extraction and intervention experiments |
| Neuronpedia | High | Open source | Adopt | Interactive feature exploration, search across 50M+ latents, steering, live testing; useful for qualitative absorption case studies and feature dashboard analysis |
| sparse-but-wrong-paper (Chanin) | Medium | check paper | Extend | L0 analysis framework and proxy metric for correct L0; relevant for controlling L0 confound in absorption experiments |
| Language-Model-SAEs (OpenMOSS) | Medium | check paper | Compose | Distributed SAE training with multiple architecture support; useful if scaling to larger models or needing architectures not in SAELens |

Highlight: The **sae-spelling + SAELens + Gemma Scope + TransformerLens** combination gives a complete pipeline from activation extraction to SAE training to absorption evaluation with minimal implementation overhead. All are MIT-licensed (or equivalent research licenses) and have active community support. The absorption metric code in sae-spelling is directly reusable; the main extension needed is adapting it to feature hierarchies beyond first-letter spelling. **SAEBench** provides the standardized multi-metric framework for fair comparison across architectures.

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

33. Leask, P., Bussmann, B., Pearce, M., Bloom, J., Tigges, C., Al Moubayed, N., Sharkey, L., & Nanda, N. (2025). Sparse Autoencoders Do Not Find Canonical Units of Analysis. *ICLR 2025*. arXiv:2502.04878. URL: https://arxiv.org/abs/2502.04878

34. Chanin, D., & Garriga-Alonso, A. (2025). Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders. arXiv:2508.16560. Code: https://github.com/chanind/sparse-but-wrong-paper

35. Engels, J., Liao, I., Michaud, E.J., Gurnee, W., & Tegmark, M. (2024). Not All Language Model Features Are One-Dimensionally Linear. *ICLR 2025*. arXiv:2405.14860. Code: https://github.com/JoshEngels/MultiDimensionalFeatures

36. Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., et al. (2025). On the Biology of a Large Language Model. *Transformer Circuits Thread*. URL: https://transformer-circuits.pub/2025/attribution-graphs/biology.html

37. Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramar, J., Dragan, A., Shah, R., & Nanda, N. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. *BlackboxNLP 2024*. arXiv:2408.05147. Weights: https://huggingface.co/google/gemma-scope

38. Minegishi, G., et al. (2025). Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words. arXiv:2501.06254.

39. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2024). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *ICLR 2024*. arXiv:2309.08600. URL: https://openreview.net/forum?id=F76bwRSLeK

40. Bloom, J., Tigges, C., Duong, A., & Chanin, D. (2024). SAELens. GitHub: https://github.com/decoderesearch/SAELens
