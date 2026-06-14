# Literature Survey Report

**Research Topic**: Feature Absorption in Sparse Autoencoders (SAEs): Systematic Analysis and Quantification of Causes, Patterns, and Implications for Interpretability
**Survey Date**: 2026-04-27 (updated; sixth-pass verification with expanded arXiv + Web dual-source search)
**arXiv Search Keywords**: "feature absorption" "feature splitting" sparse autoencoders; "sparse autoencoder" "mechanistic interpretability" features superposition; SAEBench evaluation sparse autoencoders; Gated SAE JumpReLU architecture; Matryoshka SAE hierarchical; feature hedging correlated; "survey sparse autoencoders" LLM; theoretical sparse autoencoders identifiable; SAE dark matter reconstruction; "feature manifolds" scaling; KronSAE features correlation; unified theory sparse dictionary learning; transcoder interpretability absorption; SynthSAEBench synthetic hierarchy; adaptive temporal masking SAE absorption; CE-Bench contrastive evaluation SAE; feature sensitivity sparse autoencoder; OrtSAE orthogonal atomic features; end-to-end SAE training; domain-specific SAE nonlinear error; distilled matryoshka SAE
**Web Search Keywords**: sparse autoencoder feature absorption interpretability 2025; SAE feature absorption Chanin 2024 arxiv 2409.14507; Matryoshka SAE hierarchical feature absorption; SAE feature absorption solutions mitigation OrtSAE masked regularization; SAE superposition toy model Anthropic survey; sparse autoencoder absorption sparsity penalty L1 L0 hierarchical theoretical; feature anchoring unified theory SDL absorption dead neurons; KronSAE Kronecker factorization absorption metrics; transcoders beat SAEs interpretability; DeepMind negative results SAE safety; SAELens SAE training library GitHub; feature hedging correlated features break narrow SAE; SAE benchmark SAEBench evaluation 2025; TransformerLens Gemma Scope pretrained SAE; SAE nonlinear error reconstruction downstream; domain-specific SAE Salmon; SAE feature absorption mitigation 2026; sparse autoencoder feature absorption mechanistic interpretability 2025; SAE feature absorption state of the art 2025 2026; sparse autoencoder interpretability survey review 2024 2025; feature absorption sparse autoencoder Chanin benchmark SAEBench 2024 2025; SAE feature splitting absorption superposition L1 sparsity JumpReLU TopK 2024 2025; sparse autoencoder benchmark evaluation SAEBench open source GitHub 2025; feature hedging correlated features sparse autoencoders 2025; SAELens library GitHub training sparse autoencoders TransformerLens 2025; Gemma Scope open sparse autoencoders Google DeepMind 2024; transcoders vs sparse autoencoders interpretability 2025; sparse autoencoder feature composition polysemanticity superposition theory 2024 2025; OrtSAE orthogonal sparse autoencoder feature disentanglement Korznikov 2025; Neuronpedia sparse autoencoder feature visualization dashboard 2025; BatchTopK Gated SAE architecture comparison 2024 2025; understanding sparse autoencoder scaling feature manifolds 2509.02565; Specialized Sparse Autoencoders interpreting rare concepts NAACL 2025; feature anchoring sparse dictionary learning identifiability Tang 2025; sparse autoencoder do not find canonical Leask Bussmann ICLR 2025; Anthropic circuit tracing attribution graphs sparse features 2025 2026; sparse autoencoder feature absorption mitigation 2026 new methods; scaling monosemanticity Anthropic sparse autoencoder Claude 2024; unified theory sparse dictionary learning mechanistic interpretability feature anchoring 2025; Gemma Scope sparse autoencoder pretrained Gemma 2 2024 Lieberum; feature manifolds sparse autoencoder scaling superposition geometry 2025; sparse autoencoders do not find canonical Leask Belrose 2025 reproducibility

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant unsupervised tool in mechanistic interpretability for decomposing the polysemantic, superposed activations of large language models (LLMs) into approximately monosemantic, human-interpretable feature directions. The foundational motivation comes from Anthropic's "Toy Models of Superposition" (2022) and the subsequent "Towards Monosemanticity" (2023), which established that neural networks encode far more features than they have neurons by representing features as overlapping linear directions---a phenomenon called superposition. SAEs address this by projecting activations into an overcomplete sparse basis via a sparsity-inducing objective (L1 penalty or TopK activation), with the hope of recovering the underlying "true" monosemantic features.

The field advanced rapidly through 2024--2025: Anthropic and OpenAI scaled SAEs to frontier models (Claude 3 Sonnet, GPT-4), the community developed several improved architectures (Gated SAE, JumpReLU SAE, TopK SAE, Matryoshka SAE, OrtSAE), and comprehensive benchmarks (SAEBench, CE-Bench, SynthSAEBench) emerged to standardize evaluation. However, alongside these successes came a growing body of negative results that challenge whether SAEs robustly recover the mechanistically meaningful features they promise. Three failure modes have attracted particular attention: (1) **feature absorption**---where a general feature silently fails to fire because a more specific co-occurring feature encodes the same information; (2) **feature hedging**---where correlated features are merged due to insufficient SAE width; and (3) **feature inconsistency**---where independent training runs converge to different feature sets. Feature absorption is arguably the most fundamental of these, as it creates a false sense of interpretability: the SAE appears to have learned a clean monosemantic latent, but that latent has systematic "holes" in its recall.

The current state of the field (early 2026) is one of productive tension: SAEs remain the most tractable approach to large-scale mechanistic interpretability, but their theoretical guarantees are weak, and several recent papers show that even well-trained SAEs may only marginally outperform random baselines on downstream tasks. Notably, Google DeepMind's mechanistic interpretability team publicly deprioritized SAE research in March 2025 after finding that dense linear probes dramatically outperform SAE probes on safety-relevant downstream tasks (e.g., harmful intent detection via OOD probing), with feature absorption being a key culprit. The team found 10--40% performance degradation when using SAE-reconstructed activations and noted that SAEs create "nonsensical concepts" to maximize sparsity objectives. Meanwhile, Anthropic successfully used SAE-based features for circuit tracing in Claude 3.5 Haiku ("On the Biology of a Large Language Model", Lindsey et al., 2025), demonstrating that when features are reliable, they enable powerful mechanistic understanding---making absorption a critical obstacle to broader adoption. Furthermore, Wu et al. (2025) showed that SAEs underperform on both concept steering and detection relative to simple baselines, though the community has argued these results reflect limitations of interpretability methodology more broadly rather than SAEs specifically.

The community is actively developing architectural solutions (Matryoshka SAEs, OrtSAE, KronSAE, Adaptive Temporal Masking, masked regularization), alternative paradigms (transcoders, crosscoders), and theoretical frameworks (identifiability analysis, feature anchoring, MDL-based training) to address these shortcomings. A unified theoretical framework (arXiv:2512.05534) now casts all major sparse dictionary learning methods as a single piecewise biconvex optimization problem and provides principled explanations for absorption---though feature anchoring, its proposed remedy, has only been validated on synthetic benchmarks. Additionally, foundational assumptions of SAEs have been challenged: Leask et al. (ICLR 2025) show SAE features are neither canonical nor atomic, Engels et al. (ICLR 2025) discover irreducible multi-dimensional features that SAEs may fundamentally misrepresent, and Chanin & Garriga-Alonso (2025) demonstrate that incorrect L0 (which is common in practice) causes SAEs to learn systematically wrong features. Domain-specific SAE training has also emerged as a promising direction: "Resurrecting the Salmon" (2025) shows that narrowing the input domain forces SAEs to allocate capacity to high-fidelity task-relevant features, reducing nonlinear error and potentially mitigating absorption.

Feature absorption specifically sits at the intersection of theory (caused by sparsity optimization under feature hierarchy) and practice (undermines causal circuit analysis and safety-relevant feature detection), making it one of the most important open problems in SAE-based interpretability.

**Key mechanistic example of absorption** (from Chanin et al.): Consider the token "snake." The model can achieve better sparsity on the L1 objective by firing only a single "snake" feature rather than the "starts with s" feature plus the "reptile" feature. Absorption makes it rational to only have space for these specific token features for common tokens. Where dense and sparse features co-occur, absorption is a clear strategy for reducing the number of features firing. Critically, this occurs across all tested architectures---even BatchTopK SAEs (which lack L1 loss) show clear absorption patterns because absorption improves reconstruction loss at a given k.

**Feature absorption is also an "effective strategy"** from the reconstruction perspective: For each parent-child feature relation, a vanilla SAE with feature absorption can represent both features with +1 L0, while an SAE without feature absorption would require +2 L0. This means any solution that fully removes absorption will likely have worse variance explained against L0---establishing a fundamental tension between reconstruction quality and interpretability.

**Gemma Scope 2** (December 2025) expanded the open SAE ecosystem significantly, providing interpretability tools for all Gemma 3 model sizes (270M to 27B parameters), training over 1 trillion total parameters, and storing approximately 110 Petabytes of data---the largest open-source interpretability release to date. This includes Matryoshka SAEs and transcoders alongside the original JumpReLU architecture.

**Neuronpedia** (Lin et al.) has become the primary interactive platform for SAE feature exploration, hosting 50M+ latents with search, steering, circuit tracing, and auto-interpretation capabilities. It became fully open-source in March 2025, enabling self-hosted deployments. SAEBench results are also interactively visualizable at neuronpedia.org/sae-bench.

---

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | arXiv:2409.14507 (NeurIPS 2025) | 2024 | Defines and characterizes feature absorption; toy model proof that hierarchical features cause absorption; proposes absorption rate metric; validates on hundreds of SAEs (Gemma Scope, Llama 3.2, Qwen2); finds absorption in every tested SAE | Focuses on first-letter spelling task; absorption metric requires known probe directions; does not propose a solution |
| 2 | Toy Models of Superposition | transformer-circuits.pub (2022) | 2022 | Foundational: demonstrates neural nets encode more features than neurons via superposition; introduces the linear representation hypothesis as motivation for SAEs | Synthetic toy setting; does not address feature absorption |
| 3 | Towards Monosemanticity: Decomposing Language Models With Dictionary Learning | transformer-circuits.pub (2023) | 2023 | First large-scale demonstration that SAEs decompose MLP activations into interpretable monosemantic features on a 1-layer transformer | Small model; does not address absorption or failure modes |
| 4 | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | transformer-circuits.pub (2024) | 2024 | Shows SAEs scale to frontier models; finds safety-relevant features (deception, sycophancy, bias); introduces concept of feature families | No systematic absorption analysis; limited evaluation methodology |
| 5 | Scaling and Evaluating Sparse Autoencoders (TopK SAE) | arXiv:2406.04093 (OpenAI, ICLR 2025) | 2024 | Proposes k-sparse SAEs with clean scaling laws; introduces new evaluation metrics; scales to 16M latent SAE on GPT-4; key ingredients for preventing dead latents (encoder init to decoder transpose, auxiliary loss) | Does not specifically address feature absorption; evaluation metrics focus on proxy measures |
| 6 | Improving Dictionary Learning with Gated Sparse Autoencoders | arXiv:2404.16014 | 2024 | Proposes Gated SAE that decouples feature detection from magnitude estimation; reduces L1-induced shrinkage bias; Pareto improvement | Does not directly address feature absorption; still susceptible to hierarchical feature pathology |
| 7 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders | arXiv:2407.14435 | 2024 | Introduces JumpReLU activation for SOTA reconstruction at given sparsity; trains L0 directly via STE; achieves Pareto improvement on Gemma 2 9B | Does not specifically mitigate absorption; SAEBench shows JumpReLU has worst absorption (trained longer) |
| 8 | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | arXiv:2503.17547 (ICML 2025) | 2025 | Trains nested SAE dictionaries of increasing size simultaneously; organizes features hierarchically with smaller dicts learning general concepts; absorption rate ~0.03 vs BatchTopK ~0.29; superior on sparse probing and concept erasure | Minor reconstruction tradeoff; inner levels suffer from feature hedging |
| 9 | Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders | arXiv:2505.11756 | 2025 | Identifies "feature hedging" as a complementary failure mode to absorption; theoretically and empirically shows narrow SAEs merge correlated features; wider SAEs suffer less; proposes balanced Matryoshka SAE with compound multiplier ~0.75 | Shows Matryoshka SAEs trade absorption for hedging; hedging may explain SAE underperformance vs baselines |
| 10 | OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features | arXiv:2509.22033 | 2025 | Enforces orthogonality via pairwise cosine similarity penalty; reduces absorption by ~70% vs BatchTopK; composition by ~70%; discovers more distinct features; MeanCosSim 2.7x lower than BatchTopK; improves SCR by 6% on SAEBench; linear computational overhead | Does not fully eliminate absorption; slightly higher absorption than Matryoshka; focuses on structural orthogonality rather than hierarchy |
| 11 | Improving Robustness In Sparse Autoencoders via Masked Regularization | arXiv:2604.06495 | 2026 | Proposes masking-based regularization to disrupt co-occurrence patterns during training; reduces absorption and improves OOD robustness across SAE architectures | Very recent (April 2026); limited to token masking intervention |
| 12 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability | arXiv:2503.09532 (ICML 2025) | 2025 | Establishes 8-metric evaluation suite including absorption task; 200+ open-source SAEs across 7 architectures; reveals proxy metrics do not reliably predict practical performance; key finding: TopK and JumpReLU significantly worsen absorption | Absorption metric is one of 8 and may not fully capture the phenomenon |
| 13 | Decomposing The Dark Matter of Sparse Autoencoders | arXiv:2410.14670 | 2024 | Investigates unexplained SAE reconstruction error ("dark matter"); shows ~50% of error vector is linearly predictable from input; links to scaling behavior | Does not directly address absorption; provides context for SAE failure modes |
| 14 | On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy | arXiv:2506.15963 | 2025 | First closed-form theoretical analysis showing SAEs generally fail to recover ground truth monosemantic features unless features are extremely sparse; proposes reweighted SAE (WSAE) | Theoretical bound may be loose; WSAE not evaluated on absorption specifically |
| 15 | Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders | arXiv:2506.14002 | 2025 | Proposes bias-adaptation training with theoretical recovery guarantees under a statistical model; Group Bias Adaptation (GBA) variant for LLMs | Guarantees rely on restrictive generative assumptions; does not model feature hierarchy |
| 16 | Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? | arXiv:2602.14111 | 2026 | Shows SAEs recover only 9% of true features in synthetic settings; random baselines match fully-trained SAEs on interpretability, sparse probing, and causal editing tasks | Extremely critical; may underestimate SAE utility in practice; does not address absorption |
| 17 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | arXiv:2505.20254 | 2025 | Shows SAE features are inconsistent across training runs; proposes PW-MCC metric; TopK SAEs achieve 0.80 consistency | Does not address absorption specifically; focuses on inter-run consistency |
| 18 | Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures | arXiv:2506.01197 | 2025 | Explicitly models semantic hierarchy in SAE; improves reconstruction and interpretability; computationally efficient | Recent; absorption evaluation not primary focus |
| 19 | From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders (HSAE) | arXiv:2602.11881 | 2026 | Jointly learns SAE series and parent-child relationships; structural constraint loss and random feature perturbation; consistently recovers semantically meaningful hierarchies | Very recent; relationship to absorption not fully characterized |
| 20 | Mathematical Models of Computation in Superposition | arXiv:2408.05451 | 2024 | Provides theoretical models for computation in superposition; shows networks can emulate circuits of width O(d^1.5) with width d using superposition | Foundational theory; does not address absorption directly |
| 21 | A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima | arXiv:2512.05534 | 2024 | First unified theoretical framework casting all SDL variants (SAEs, transcoders, crosscoders) as a piecewise biconvex optimization problem; provides principled explanations for feature absorption and dead neurons; proposes feature anchoring to restore identifiability | Feature anchoring validated on synthetic benchmarks only; real LLM validation limited |
| 22 | Transcoders Beat Sparse Autoencoders for Interpretability | arXiv:2501.18823 | 2025 | Shows transcoder features are significantly more interpretable than SAE features; proposes skip transcoders with affine skip connection; achieves Pareto improvement on reconstruction + interpretability; SAEs and transcoders have similar absorption behavior | Does not directly study absorption; transcoder paradigm incompatible with existing SAE evaluation ecosystem |
| 23 | KronSAE: Kronecker Factorization Improves Efficiency and Interpretability of Sparse Autoencoders | arXiv:2505.22255 | 2025 | Proposes Kronecker product factorization of SAE latents; introduces mAND activation for logical AND-like gating; reduces mean absorption fraction and full-absorption score across all sparsity levels; competitive with TopK at lower compute | Architecture-specific; absorption reduction mechanism tied to factorization structure; not evaluated on SAEBench |
| 24 | SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data | arXiv:2602.14687 | 2026 | Provides large-scale synthetic data model with realistic feature characteristics including correlation, hierarchy, superposition, and Zipfian firing distributions; logistic probes achieve 0.974 F1 while best SAEs substantially underperform; enables controlled study of feature hierarchy and absorption; also found Matching Pursuit SAEs exploit superposition noise to improve reconstruction without learning ground-truth features | Very recent (Feb 2026); synthetic setting may not capture all real LLM characteristics |
| 24b | BatchTopK Sparse Autoencoders | arXiv:2412.06410 (NeurIPS 2024 Workshop) | 2024 | Relaxes TopK constraint to batch level, allowing variable latents per sample; consistently outperforms TopK and JumpReLU SAEs on GPT-2 Small and Gemma 2 2B; enables direct sparsity specification without costly hyperparameter sweep; forms basis for Matryoshka SAE architecture | Batch dependency during training; does not address absorption; absorption behavior similar to TopK |
| 25 | Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training (ATM) | arXiv:2510.08855 | 2025 | Introduces Adaptive Temporal Masking (ATM) that dynamically adjusts feature selection via importance scoring of activation magnitudes, frequencies, and reconstruction contributions; achieves mean absorption score 0.0068 vs. TopK 0.1402 and JumpReLU 0.0114 on Gemma-2-2B; strong performance on bias detection and sentiment analysis | Per-latent importance tracking adds training overhead; evaluated only on Gemma-2-2B |
| 26 | CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark of General Interpretability of Sparse Autoencoders | arXiv:2509.00691 (BlackboxNLP 2025) | 2025 | Fully LLM-free contrastive benchmark built on structured story pairs differing in targeted semantic attribute; >70% Spearman correlation with SAEBench; interpretability scores increase with latent width; robust across layer types | Does not directly measure feature absorption; benchmarks overall interpretability not absorption specifically |
| 27 | Measuring Sparse Autoencoder Feature Sensitivity | arXiv:2509.23717 | 2025 | Frames feature absorption as a special case of poor feature sensitivity; develops scalable sensitivity evaluation methods; finds many interpretable features have poor sensitivity even when activation examples appear monosemantic; sensitivity measures a dimension of quality missed by existing evaluations | Does not propose new mitigation; sensitivity metric distinct from but related to absorption metric |
| 28 | Sparse Autoencoders Do Not Find Canonical Units of Analysis | arXiv:2502.04878 (ICLR 2025) | 2025 | Shows SAE features are neither complete nor atomic using SAE stitching and meta-SAEs; novel latents in larger SAEs capture info missed by smaller SAEs; meta-SAEs decompose features into sub-features; fundamentally challenges the canonical feature assumption | Does not study absorption directly; focused on non-canonicality and compositionality |
| 29 | Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders | arXiv:2508.16560 | 2025 | Demonstrates that incorrect L0 causes SAEs to mix correlated features ("cheat"); too-low L0 triggers feature hedging, too-high L0 finds degenerate solutions; proposes proxy metric for finding correct L0; most open-source SAEs have L0 that is too low | Builds on toy model analysis; relationship to absorption not fully characterized |
| 30 | Not All Language Model Features Are One-Dimensionally Linear | arXiv:2405.14860 (ICLR 2025) | 2024 | Discovers irreducible multi-dimensional features (circular representations for days/months) in GPT-2 and Mistral 7B; challenges the one-dimensional linear representation hypothesis; shows multi-dimensional features are used for modular arithmetic computation | SAEs may miss or distort multi-dimensional features; relationship to absorption of multi-dimensional features unexplored |
| 31 | On the Biology of a Large Language Model | transformer-circuits.pub (Anthropic 2025) | 2025 | Introduces attribution graphs for tracing computational circuits in Claude 3.5 Haiku; reveals planning-ahead, hallucination default-refusal, and jailbreak mechanisms; demonstrates practical utility of feature-level interpretability for understanding model behavior | Uses SAE features but does not specifically study absorption; shows downstream value of reliable feature decomposition |
| 32 | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | arXiv:2408.05147 (BlackboxNLP 2024) | 2024 | Releases 400+ open JumpReLU SAEs on all layers/sublayers of Gemma 2 2B/9B/27B; widths 1k--1M; primary evaluation target for absorption research; Gemma Scope 2 (Dec 2024) adds Matryoshka SAEs and transcoders on Gemma 3 | JumpReLU architecture may worsen absorption; massive resource for community but trained with single architecture family |
| 33 | Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words | arXiv:2501.06254 | 2025 | Evaluates SAEs specifically on polysemous words; finds performance ranking ReLU > JumpReLU > TopK on F1; challenges assumption that better MSE-L0 frontier implies better feature quality | Narrow evaluation domain (polysemous words); does not directly measure absorption but highlights related failure mode |
| 34 | A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models | arXiv:2503.05613 (EMNLP 2025 Findings) | 2025 | Comprehensive survey covering SAE architecture, design improvements, training strategies, feature explanation methods (input-based, output-based), evaluation metrics (structural, functional), and real-world applications in understanding/manipulating LLM behavior | Survey; no new methods or absorption-specific contributions |
| 35 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | arXiv:2309.08600 (ICLR 2024) | 2023 | First scalable demonstration of SAEs finding interpretable monosemantic features in real LLMs (Pythia-70M/410M); establishes SAE paradigm for mechanistic interpretability; automated interpretability measurement | Early work; small models; no absorption analysis |
| 36 | Negative Results for Sparse Autoencoders On Downstream Tasks and Deprioritising SAE Research | DeepMind Medium blog (2025) | 2025 | GDM mech interp team finds dense linear probes dramatically outperform SAE probes on OOD harmful intent detection; SAE-reconstructed activations cause 10-40% degradation; SAEs create nonsensical concepts for sparsity; team deprioritizes SAE research | Internal results; specific numbers not fully published; does not quantify absorption's contribution to the gap |
| 37 | Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need | arXiv:2503.17272 | 2025 | Shows brief KL+MSE fine-tuning on final 25M tokens achieves comparable improvement to full end-to-end training; multiple fine-tuning methods yield similar non-additive improvements suggesting common correctable error source in MSE-trained SAEs | Mixed results on supervised SAEBench metrics; does not specifically address absorption |
| 38 | Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts | arXiv:2506.23845 | 2025 | Shows SAE reconstruction encodes strictly less information than original LM representation; SAEs fail to outperform simple baselines on concept detection and steering; argues SAEs are best suited for concept discovery rather than downstream action | Does not address absorption directly; provides important context for SAE's role and limitations |
| 39 | Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders | arXiv:2508.09363 | 2025 | Shows broad-domain SAE training produces generic, inconsistent latents vulnerable to nonlinear error and absorption; domain-specific training forces capacity allocation to high-fidelity task-relevant features; reduces linearly predictable error fraction | New direction; absorption reduction not specifically quantified; domain-specific approach limits generality |
| 40 | Attribution-Guided Distillation of Matryoshka Sparse Autoencoders | arXiv:2512.24975 | 2025 | Uses iterative distillation to winnow Matryoshka SAEs to compact core feature sets; stabilizes representations across runs and sparsities | Complexity; limited evaluation on absorption specifically |
| 41 | Interpretable Reward Model via Sparse Autoencoder (SARM) | AAAI 2025 | 2025 | Pretrains SAE on sequence-level hidden states for interpretable reward modeling; attaches pretrained encoder with learnable linear head; produces rewards explicitly attributed to interpretable SAE features | Application paper; does not study absorption; reward modeling specific |
| 42 | The Geometry of Concepts: Sparse Autoencoder Feature Structure | arXiv:2410.19750 / Entropy 27(4):344 (2025) | 2024 | Analyzes geometric structure of SAE feature point clouds at three scales: atomic (parallelogram/trapezoid crystals generalizing analogy relations), brain (modular spatial clustering akin to fMRI lobes), galaxy (power-law eigenvalue spectrum); raises question of whether individual SAE features are the most natural unit of analysis | Descriptive geometry; does not address absorption; but spatial modularity findings may inform hierarchical feature relationships relevant to absorption |
| 43 | Efficient Dictionary Learning with Switch Sparse Autoencoders | ICLR 2025 | 2024 | Inspired by sparse MoE, routes activations between smaller expert SAEs; substantial Pareto improvement on reconstruction-sparsity frontier for fixed compute; studies geometry of features across experts and feature duplication | Expert routing adds complexity; feature duplication across experts may relate to absorption-like redundancy; not evaluated for absorption specifically |
| 44 | Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds | arXiv:2509.02565 | 2025 | Adapts capacity-allocation model from neural scaling literature to SAE scaling; shows feature manifolds cause pathological regime where SAEs allocate many latents to few multi-dimensional features instead of learning more features; identifies distinct scaling regimes | Short workshop paper; no direct absorption analysis; but multi-dimensional feature allocation relates to absorption dynamics |
| 45 | Decoding Dark Matter: Specialized Sparse Autoencoders for Interpreting Rare Concepts in Foundation Models | NAACL 2025 Findings (arXiv:2411.00743) | 2024 | Introduces Specialized SAEs (SSAEs) trained on subdomain data via dense retrieval + Tilted ERM; captures rare tail concepts missed by general SAEs; 12.5% worst-group accuracy improvement on Bias in Bios; practical recipe for domain-focused SAE training | Does not directly address absorption; subdomain focus may indirectly reduce absorption by narrowing feature hierarchy |
| 27 | Measuring Sparse Autoencoder Feature Sensitivity | arXiv:2509.23717 | 2025 | Frames feature absorption as a special case of poor feature sensitivity; develops scalable sensitivity evaluation methods; finds many interpretable features have poor sensitivity even when activation examples appear monosemantic; sensitivity measures a dimension of quality missed by existing evaluations | Does not propose new mitigation; sensitivity metric distinct from but related to absorption metric |
| 28 | Sparse Autoencoders Do Not Find Canonical Units of Analysis | arXiv:2502.04878 (ICLR 2025) | 2025 | Shows SAE features are neither complete nor atomic using SAE stitching and meta-SAEs; novel latents in larger SAEs capture info missed by smaller SAEs; meta-SAEs decompose features into sub-features; fundamentally challenges the canonical feature assumption | Does not study absorption directly; focused on non-canonicality and compositionality |
| 29 | Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders | arXiv:2508.16560 | 2025 | Demonstrates that incorrect L0 causes SAEs to mix correlated features ("cheat"); too-low L0 triggers feature hedging, too-high L0 finds degenerate solutions; proposes proxy metric for finding correct L0; most open-source SAEs have L0 that is too low | Builds on toy model analysis; relationship to absorption not fully characterized |
| 30 | Not All Language Model Features Are One-Dimensionally Linear | arXiv:2405.14860 (ICLR 2025) | 2024 | Discovers irreducible multi-dimensional features (circular representations for days/months) in GPT-2 and Mistral 7B; challenges the one-dimensional linear representation hypothesis; shows multi-dimensional features are used for modular arithmetic computation | SAEs may miss or distort multi-dimensional features; relationship to absorption of multi-dimensional features unexplored |
| 31 | On the Biology of a Large Language Model | transformer-circuits.pub (Anthropic 2025) | 2025 | Introduces attribution graphs for tracing computational circuits in Claude 3.5 Haiku; reveals planning-ahead, hallucination default-refusal, and jailbreak mechanisms; demonstrates practical utility of feature-level interpretability for understanding model behavior | Uses SAE features but does not specifically study absorption; shows downstream value of reliable feature decomposition |
| 32 | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | arXiv:2408.05147 (BlackboxNLP 2024) | 2024 | Releases 400+ open JumpReLU SAEs on all layers/sublayers of Gemma 2 2B/9B/27B; widths 1k--1M; primary evaluation target for absorption research; Gemma Scope 2 (Dec 2024) adds Matryoshka SAEs and transcoders on Gemma 3 | JumpReLU architecture may worsen absorption; massive resource for community but trained with single architecture family |
| 33 | Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words | arXiv:2501.06254 | 2025 | Evaluates SAEs specifically on polysemous words; finds performance ranking ReLU > JumpReLU > TopK on F1; challenges assumption that better MSE-L0 frontier implies better feature quality | Narrow evaluation domain (polysemous words); does not directly measure absorption but highlights related failure mode |
| 34 | A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models | arXiv:2503.05613 (EMNLP 2025 Findings) | 2025 | Comprehensive survey covering SAE architecture, design improvements, training strategies, feature explanation methods (input-based, output-based), evaluation metrics (structural, functional), and real-world applications in understanding/manipulating LLM behavior | Survey; no new methods or absorption-specific contributions |
| 35 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | arXiv:2309.08600 (ICLR 2024) | 2023 | First scalable demonstration of SAEs finding interpretable monosemantic features in real LLMs (Pythia-70M/410M); establishes SAE paradigm for mechanistic interpretability; automated interpretability measurement | Early work; small models; no absorption analysis |
| 36 | Negative Results for Sparse Autoencoders On Downstream Tasks and Deprioritising SAE Research | DeepMind Medium blog (2025) | 2025 | GDM mech interp team finds dense linear probes dramatically outperform SAE probes on OOD harmful intent detection; SAE-reconstructed activations cause 10-40% degradation; SAEs create nonsensical concepts for sparsity; team deprioritizes SAE research | Internal results; specific numbers not fully published; does not quantify absorption's contribution to the gap |
| 37 | Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need | arXiv:2503.17272 | 2025 | Shows brief KL+MSE fine-tuning on final 25M tokens achieves comparable improvement to full end-to-end training; multiple fine-tuning methods yield similar non-additive improvements suggesting common correctable error source in MSE-trained SAEs | Mixed results on supervised SAEBench metrics; does not specifically address absorption |
| 38 | Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts | arXiv:2506.23845 | 2025 | Shows SAE reconstruction encodes strictly less information than original LM representation; SAEs fail to outperform simple baselines on concept detection and steering; argues SAEs are best suited for concept discovery rather than downstream action | Does not address absorption directly; provides important context for SAE's role and limitations |
| 39 | Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders | arXiv:2508.09363 | 2025 | Shows broad-domain SAE training produces generic, inconsistent latents vulnerable to nonlinear error and absorption; domain-specific training forces capacity allocation to high-fidelity task-relevant features; reduces linearly predictable error fraction | New direction; absorption reduction not specifically quantified; domain-specific approach limits generality |
| 40 | Attribution-Guided Distillation of Matryoshka Sparse Autoencoders | arXiv:2512.24975 | 2025 | Uses iterative distillation to winnow Matryoshka SAEs to compact core feature sets; stabilizes representations across runs and sparsities | Complexity; limited evaluation on absorption specifically |
| 41 | Interpretable Reward Model via Sparse Autoencoder (SARM) | AAAI 2025 | 2025 | Pretrains SAE on sequence-level hidden states for interpretable reward modeling; attaches pretrained encoder with learnable linear head; produces rewards explicitly attributed to interpretable SAE features | Application paper; does not study absorption; reward modeling specific |
| 42 | The Geometry of Concepts: Sparse Autoencoder Feature Structure | arXiv:2410.19750 / Entropy 27(4):344 (2025) | 2024 | Analyzes geometric structure of SAE feature point clouds at three scales: atomic (parallelogram/trapezoid crystals generalizing analogy relations), brain (modular spatial clustering akin to fMRI lobes), galaxy (power-law eigenvalue spectrum); raises question of whether individual SAE features are the most natural unit of analysis | Descriptive geometry; does not address absorption; but spatial modularity findings may inform hierarchical feature relationships relevant to absorption |
| 43 | Efficient Dictionary Learning with Switch Sparse Autoencoders | ICLR 2025 | 2024 | Inspired by sparse MoE, routes activations between smaller expert SAEs; substantial Pareto improvement on reconstruction-sparsity frontier for fixed compute; studies geometry of features across experts and feature duplication | Expert routing adds complexity; feature duplication across experts may relate to absorption-like redundancy; not evaluated for absorption specifically |
| 44 | Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds | arXiv:2509.02565 | 2025 | Adapts capacity-allocation model from neural scaling literature to SAE scaling; shows feature manifolds cause pathological regime where SAEs allocate many latents to few multi-dimensional features instead of learning more features; identifies distinct scaling regimes | Short workshop paper; no direct absorption analysis; but multi-dimensional feature allocation relates to absorption dynamics |
| 45 | Decoding Dark Matter: Specialized Sparse Autoencoders for Interpreting Rare Concepts in Foundation Models | NAACL 2025 Findings (arXiv:2411.00743) | 2024 | Introduces Specialized SAEs (SSAEs) trained on subdomain data via dense retrieval + Tilted ERM; captures rare tail concepts missed by general SAEs; 12.5% worst-group accuracy improvement on Bias in Bios; practical recipe for domain-focused SAE training | Does not directly address absorption; subdomain focus may indirectly reduce absorption by narrowing feature hierarchy |

---

## 3. SOTA Methods and Benchmarks

### Current Best SAE Architectures (sparsity-fidelity frontier)

- **BatchTopK SAE**: Directly sets desired sparsity without tuning penalty; improved training stability; baseline in SAEBench; NOTE: significantly worsens feature absorption vs. L1 SAEs at low L0 (SAEBench finding)
- **TopK SAE** (Gao et al., OpenAI 2024): Clean scaling laws; reference architecture; scales to 16M latents on GPT-4; NOTE: significantly worsens feature absorption at low L0
- **JumpReLU SAE** (Rajamanoharan et al., DeepMind 2024): State-of-the-art reconstruction on Gemma 2 9B; trains L0 directly via STE; worst on absorption (trained longer = more absorption); NOTE: finicky dead latents in SynthSAEBench-16k (increasing initial threshold to 0.5 from Anthropic default 0.1 helps)
- **Gated SAE** (Rajamanoharan et al., DeepMind 2024): Decouples detection from magnitude; reduces shrinkage

### Architectures Specifically Addressing Absorption

- **Matryoshka SAE** (Bussmann et al., ICML 2025): Best on SAEBench absorption (~0.03 vs BatchTopK ~0.29), RAVEL, sparse probing, and spurious correlation removal; nested prefix losses create natural feature hierarchy; minor reconstruction penalty; inner levels suffer from feature hedging
- **OrtSAE** (Korznikov et al., 2025): Reduces absorption ~70% vs BatchTopK by orthogonality penalty; MeanCosSim 2.7x lower; best for disentanglement; improves SCR by 6%; linear computational overhead; slightly higher absorption than Matryoshka
- **ATM SAE** (Li et al., 2025): Adaptive Temporal Masking with per-latent importance scoring; achieves absorption score 0.0068 vs. TopK 0.1402 and JumpReLU 0.0114 on Gemma-2-2B; best reported absorption scores to date
- **KronSAE** (2025): Kronecker product factorization of latents; reduces absorption fraction via structural correlation exploitation; lower parameter count than TopK; not evaluated on SAEBench
- **Balanced Matryoshka SAE** (Chanin et al., 2025): Tuned compound multiplier (~0.75) achieves better TPP, feature splitting, and sparse probing than vanilla Matryoshka or standard SAE
- **Competition-Aware Orthogonal SAE** (OpenReview 2025, distinct from OrtSAE): Introduces sparsity-guided orthogonality constraints with a three-phase curriculum that dynamically identifies and disentangles competing features; SOTA absorption on Gemma-2-2B
- **Distilled Matryoshka SAE** (Martin-Linares et al., 2025): Iterative distillation for compact core feature sets; stable across runs

### Alternative Paradigms

- **Transcoders / Skip Transcoders** (Paulo et al., 2025): Maps MLP inputs to outputs; features more interpretable than SAEs; Pareto improvement on reconstruction + interpretability; similar absorption behavior to SAEs
- **Domain-Specific SAEs** (2025): Narrowing input domain reduces nonlinear error and forces capacity allocation to task-relevant features; potentially mitigates absorption by reducing need for hierarchical feature coverage. Two complementary approaches: "Resurrecting the Salmon" retrains SAEs on domain-specific data; SSAEs (Muhamed et al., NAACL 2025) fine-tune a general SAE on subdomain data selected via dense retrieval + Tilted ERM

### Absorption Metric (Chanin et al., 2024)

The canonical absorption measurement procedure:
1. Find k feature splits for a first-letter feature using k-sparse probing
2. Identify false-negative tokens: all k split latents fail to activate, but LR probe correctly classifies
3. Run integrated-gradients ablation on false-negative tokens
4. Absorption detected if the highest-ablation-effect latent has cosine similarity > 0.025 with probe AND is >=1.0 larger than the second-highest
5. Absorption rate = (absorbed letters) / (total letters with at least one false negative)

**Key empirical finding**: Absorption rate 15--35% across Gemma Scope 16k and 65k SAEs; wider and more sparse SAEs show higher absorption; no clear layer-wise pattern; present in Llama 3.2 1B, Qwen2 0.5B, and all architectures tested (L1 and TopK).

**SAEBench's modified absorption metric**: SAEBench builds on Chanin et al.'s metric but extends it to work across all model layers. The original metric included ablation effect of absorbing latents on task performance, which limits it to early and middle layers (ablation effect goes to zero in later layers after information is moved to the final token position). SAEBench instead detects absorbing latents based on the latent contributing a significant portion of the first-letter probe direction, enabling evaluation across all layers.

**Feature sensitivity as generalization** (Tian et al., 2025): Feature absorption is a special case of poor feature sensitivity. Many interpretable features have poor sensitivity even when activation examples appear monosemantic.

### Key Benchmarks

- **SAEBench** (Karvonen et al., ICML 2025): 8 metrics, 200+ SAEs across 7 architectures, interactive at neuronpedia.org/sae-bench; metrics include Absorption, RAVEL, Sparse Probing, Spurious Correlation Removal (SCR), Targeted Probe Perturbation (TPP), Unlearning; key finding: proxy metrics (CE loss, sparsity) do not reliably predict practical performance; trained SAEs across 3 widths (4k, 16k, 65k) and 6 sparsities on Pythia-160M layer 8 and Gemma-2-2B layer 12
- **SynthSAEBench** (2026): Large-scale synthetic benchmark with realistic feature characteristics (hierarchy, correlation, Zipfian distributions); ground-truth features known; reveals SAEs substantially underperform direct probes (logistic probe 0.974 F1 vs SAE much lower)
- **CE-Bench** (Gulko et al., BlackboxNLP 2025): Lightweight LLM-free contrastive benchmark on structured story pairs; >70% Spearman correlation with SAEBench; interpretability scores increase with latent width
- **Feature Sensitivity Metric** (Tian et al., 2025): Measures how reliably a feature activates on texts similar to its activating examples; frames absorption as special case of low sensitivity
- **Linear Representation Bench** (unified SDL theory paper, 2024): Synthetic benchmark for exposing SDL pathologies under full ground-truth access

### Evaluation Models

- Primary: Gemma 2 2B, Gemma 2 9B (via Gemma Scope SAEs)
- Also evaluated: GPT-2 Small, Llama 3.2 1B, Qwen2 0.5B, Pythia series (70M, 160M, 410M)
- Frontier (internal only): Claude 3 Sonnet (Anthropic), GPT-4 (OpenAI)

---

## 4. Identified Research Gaps

**Gap 1: No quantitative causal theory of absorption magnitude.**
Chanin et al. prove absorption occurs in the toy setting and provide an informal argument (sparsity gain from absorbing parent into child), but there is no closed-form prediction of how severe absorption will be as a function of SAE width, L0, feature hierarchy depth, or feature co-occurrence statistics. Such a theory would enable principled hyperparameter selection to minimize absorption.

**Gap 2: Absorption has only been studied on the first-letter spelling task.**
The canonical absorption evaluation (Chanin et al.) uses a narrow, controlled proxy task where the feature hierarchy (letter membership > specific token) is known in advance. It is unknown whether absorption rates generalize to semantically richer hierarchies (e.g., entity type > specific entity, sentiment > topic), to safety-relevant features (deception, bias), or to non-English settings. A systematic cross-domain absorption characterization does not exist.

**Gap 3: The relationship between absorption and other SAE failure modes is under-explored.**
Feature absorption, feature hedging (Chanin et al., 2025), feature inconsistency (Song et al., 2025), and "dark matter" (Engels et al., 2024) are described as distinct phenomena but may share underlying causes. No unified theoretical framework characterizes their joint occurrence or trade-offs. The balanced Matryoshka SAE work shows absorption and hedging trade off against each other, but this is not fully characterized.

**Gap 4: Mitigation methods have not been comprehensively compared under controlled conditions.**
Matryoshka SAE, OrtSAE, ATM, masked regularization, KronSAE, and balanced Matryoshka all claim to reduce absorption, but they were evaluated with different models, layers, and metrics. No head-to-head comparison exists on a standardized benchmark (e.g., SAEBench absorption task) to determine which approach achieves the best absorption-reconstruction-interpretability trade-off.

**Gap 5: The theoretical conditions under which absorption is avoidable are unclear.**
Chanin et al. note that any solution eliminating absorption will have worse L0 vs. variance-explained tradeoff (since absorption saves one L0 per parent-child pair). "On the Limits of SAEs" (Cui et al., 2025) shows SAEs generally fail to recover ground truth features unless features are extremely sparse. Together these suggest absorption may be inherent to sparsity optimization under feature hierarchy---but neither paper provides precise sufficient conditions for absorption-free recovery.

**Gap 6: Absorption in hierarchically rich domains (knowledge, reasoning) is unstudied.**
All absorption studies focus on syntactic-level features (first letter, token identity). Feature hierarchies in knowledge representation (country > city, species > individual animal) or reasoning (logical implication chains) may exhibit qualitatively different absorption patterns. Understanding absorption in these domains is important for the safety applications that motivate SAE research.

**Gap 7: No metric for absorption that does not require known probe directions.**
The current absorption metric requires training LR probes to identify the "should-have-fired" feature. This is only possible when the researcher knows which features to look for in advance. An unsupervised absorption detection method would dramatically expand the scope of absorption analysis.

**Gap 8: Feature anchoring's effectiveness specifically against absorption is uncharacterized.**
The unified SDL theory paper (arXiv:2512.05534) proposes feature anchoring to restore identifiability and improve feature recovery, but does not specifically quantify how it reduces absorption vs. other failure modes. Whether feature anchoring can be combined with existing architectures (Matryoshka, OrtSAE) for compounded benefit is unexplored.

**Gap 8b: The safety implications of absorption are poorly quantified.**
DeepMind's 2025 study found that dense linear probes dramatically outperform SAE probes at detecting harmful intent (even OOD). However, no systematic analysis quantifies how absorption specifically drives this gap---i.e., how often safety-relevant features are absorbed into co-occurring context features. This link between absorption rate and downstream safety task performance is a critical missing piece for motivating absorption mitigation research.

**Gap 9: Confounding of absorption with incorrect L0 and feature hedging.**
Chanin & Garriga-Alonso (2025) show most open-source SAEs have L0 that is too low, causing feature hedging/mixing. Feature hedging and absorption are distinct phenomena with different causes (width/L0 insufficient vs. hierarchical feature structure), but both manifest as features failing to fire where they should. No study systematically disentangles the two: measuring how much of observed "absorption" in practice is actually L0-induced hedging vs. true hierarchy-driven absorption.

**Gap 10: Absorption of multi-dimensional features is unstudied.**
Engels et al. (ICLR 2025) show some features are irreducibly multi-dimensional (e.g., circular representations for days/months). SAEs forced to represent these as one-dimensional directions may exhibit novel absorption-like pathologies where parts of the multi-dimensional feature are absorbed by co-occurring one-dimensional features.

**Gap 11: No study of absorption across SAE architectures under matched conditions on the polysemantic word task.**
Minegishi et al. (2025) show that ReLU > JumpReLU > TopK on polysemous word F1 evaluation, suggesting architecture choice affects how SAEs handle correlated/overlapping features. Whether this ranking holds specifically for absorption has not been tested.

**Gap 12: Domain-specific SAE training has not been evaluated for absorption reduction.**
"Resurrecting the Salmon" shows domain-specific training reduces nonlinear error and allocates capacity to task-relevant features. Whether this also reduces absorption (by eliminating the need to represent the full feature hierarchy) is an untested but promising hypothesis.

**Gap 13: End-to-end training's effect on absorption is unknown.**
Karvonen (2025) shows KL+MSE fine-tuning corrects a common error source in MSE-trained SAEs, but its effect on feature absorption specifically has not been measured. If end-to-end training changes the sparsity-reconstruction tradeoff, it may indirectly affect absorption rates.

---

## 5. Available Resources

### Open-Source Code

- **sae-spelling** (canonical absorption study): https://github.com/lasr-spelling/sae-spelling --- Chanin et al. absorption experiments on first-letter task; absorption rate metric implementation
- **sparse-but-wrong-paper**: https://github.com/chanind/sparse-but-wrong-paper --- Chanin & Garriga-Alonso L0 analysis experiments
- **feature-hedging-paper**: https://github.com/chanind/feature-hedging-paper --- Chanin et al. feature hedging analysis code
- **SAELens**: https://github.com/decoderesearch/SAELens (v6, also https://github.com/jbloomAus/SAELens) --- Standard SAE training and evaluation library; supports all major architectures (batchtopk, jumprelu, standard, topk via CLI `--architecture` arg); deep integration with TransformerLens and HuggingFace Transformers (`AutoModelForCausalLM` via `model_class_name`); maintained by Bloom, Tigges, Duong, Chanin; PyPI: sae-lens v6.37.6; training a decent SAE on single A100 GPU in hours
- **Language-Model-SAEs** (OpenMOSS): https://github.com/OpenMOSS/Language-Model-SAEs --- Full distributed SAE training with head/data/model parallelism; supports vanilla SAE, Lorsa, CLT, MoLT, CrossCoder with ReLU/JumpReLU/TopK/BatchTopK; Llama Scope and Lorsa contributions
- **SAEBench**: https://github.com/adamkarvonen/SAEBench + interactive at https://www.neuronpedia.org/sae-bench --- 8-metric evaluation suite including absorption; ICML 2025
- **SAEDashboard**: https://github.com/jbloomAus/SAEDashboard --- Replicates Anthropic-style sparse autoencoder feature visualizations
- **TransformerLens**: https://github.com/TransformerLensOrg/TransformerLens --- Mechanistic interpretability library for 50+ models; hook-based activation caching and editing; deep integration with SAELens
- **Neuronpedia**: https://www.neuronpedia.org --- Interactive feature explorer, steering, circuit tracing; supports 50M+ latents across many models; now open-source
- **MultiDimensionalFeatures**: https://github.com/JoshEngels/MultiDimensionalFeatures --- Engels et al. code for finding multi-dimensional features in LLMs
- **awesome-sparse-autoencoders**: https://github.com/chrisliu298/awesome-sparse-autoencoders --- Curated list of 375+ stars
- **awesome-SAE** (Zeping Yu): https://github.com/zepingyu0512/awesome-SAE --- Actively maintained curated list of SAE papers including SAEBench, BatchTopK, AxBench, Gemma Scope
- **llama3_interpretability_sae**: https://github.com/PaulPauls/llama3_interpretability_sae --- End-to-end pipeline from activation capture to SAE training in pure PyTorch
- **CE-Bench**: https://github.com/Yusen-Peng/CE-Bench --- Contrastive evaluation benchmark code
- **dictionary-learning**: Hackable SAE training library (alternative to SAELens)
- **Goodfire SAE scaling research**: https://www.goodfire.ai/research/sae-scaling-with-feature-manifolds --- Michaud et al. capacity-allocation model for SAE scaling with feature manifolds
- **Sparsify** (EleutherAI): https://github.com/EleutherAI/sparsify --- Lean SAE/transcoder training library focused on TopK SAEs; also supports transcoders
- **circuit-tracer** (Anthropic + community): Circuit tracing library for generating attribution graphs on open-weights models; supports Gemma-2-2B, Llama-3.1-1B, Qwen3-4B; CLT and PLT modes; frontend hosted by Neuronpedia
- **Attribute** (EleutherAI): Independent attribution graph library supporting CLTs; alternative to Anthropic's circuit-tracer

### Datasets

- **OpenWebText**: Standard SAE pre-training corpus (used in Chanin et al., Gao et al.)
- **Pile / RedPajama**: Alternative corpora for SAE training
- **Gemma Scope activation caches**: Pre-computed for efficient SAE training and evaluation
- **First-letter spelling task** (constructed): In-context learning prompts using template `{token} has the first letter: {letter}`; test set from alphabetical token lists

### Pre-trained Models and SAEs

- **Gemma Scope SAEs**: 400+ JumpReLU SAEs on Gemma 2 (2B, 9B, 27B), all layers and sublayers, widths 1k--1M; used >20% training compute of GPT-3, saved ~20 PiB activations, ~30M learned features total --- HuggingFace: https://huggingface.co/google/gemma-scope ; Gemma Scope 2 (Dec 2025) adds Matryoshka SAEs + transcoders on Gemma 3 (270M--27B), ~110 PB stored, 1T+ total params trained, largest open-source interpretability release; interactive at https://www.neuronpedia.org/gemma-scope
- **SAEBench SAEs**: 200+ SAEs across 7 architectures, 3 widths (4k, 16k, 65k), 6 sparsities on Pythia-160M and Gemma-2-2B; open-sourced with training checkpoints
- **GPT-2 SAEs** (EleutherAI/OpenAI): Standard evaluation target for reproducibility; available on Neuronpedia (GPT2-Small v5)
- **Llama 3.2 1B / 3B SAEs**: Available via SAELens; used in Chanin et al. absorption validation
- **Pythia SAEs**: Available via SAELens; Pythia-70M/160M/410M used in foundational SAE work
- **Llama Scope** (OpenMOSS): SAEs trained on Llama-3.1-8B, extracting millions of features

---

## 6. Implications for Idea Generation

**Most promising directions for novel contributions:**

1. **Cross-domain absorption characterization**: Systematically measure absorption on semantically richer hierarchies (entity type hierarchies, knowledge taxonomies, safety-relevant features) beyond the first-letter spelling task. This would establish whether first-letter results generalize and identify which feature hierarchy properties predict absorption severity. The existing sae-spelling metric framework can be adapted; the key novelty is in defining new probe tasks with known hierarchical structure.

2. **Quantitative theory of absorption**: A mathematical model predicting absorption rate as a function of SAE configuration (width, L0, feature hierarchy statistics) would be highly impactful. The toy model in Chanin et al. is a starting point; extending to multi-level hierarchies and probabilistic feature co-occurrence patterns could yield clean scaling laws.

3. **Unified analysis of absorption + hedging + L0 confounds**: Feature hedging (Chanin & Dulka, 2025), absorption, and incorrect L0 (Chanin & Garriga-Alonso, 2025) all cause features to fail to fire where they should, but have different root causes (too narrow, hierarchy, wrong sparsity). Systematically disentangling these via controlled experiments would close a significant conceptual gap and provide practical guidance for SAE training.

4. **Systematic mitigation comparison**: A controlled SAEBench-style comparison of Matryoshka SAE, OrtSAE, ATM, masked regularization, balanced Matryoshka, and KronSAE on the absorption task (and other SAEBench metrics) with matched compute and model settings would be a valuable empirical contribution. No such head-to-head exists.

5. **Unsupervised absorption detection**: Developing a method to detect absorbed features without requiring pre-specified probe directions is both theoretically interesting and practically important. Candidate approaches: meta-SAE decomposition of latents, leveraging decoder cosine similarity structure, or information-theoretic divergence metrics.

6. **Linking absorption to safety task performance**: DeepMind found SAE probes fail at detecting harmful intent while dense probes succeed. Quantifying how absorption specifically drives this gap---by measuring absorption rate of safety-relevant features and correlating with downstream task performance---would provide the most compelling empirical case for absorption mitigation.

7. **Feature anchoring for absorption mitigation**: The unified SDL theory paper proposes feature anchoring as a principled remedy for identifiability failure. Applying and evaluating feature anchoring specifically for absorption reduction on real LLM SAEs (vs. only synthetic benchmarks) is a concrete research opportunity.

8. **Non-canonicality and absorption connection**: Leask et al. (ICLR 2025) show SAE features are neither complete nor atomic. Meta-SAE decomposition of features may provide a novel lens for detecting which "atomic" features are actually absorbing parent features.

9. **Domain-specific training as absorption mitigation**: "Resurrecting the Salmon" suggests narrowing input domain reduces SAE failure modes. Testing whether domain-specific SAEs specifically reduce absorption is an untested but promising hypothesis.

**Additional directions:**

10. **L0 as a confound in absorption studies**: Disentangling the effects of incorrect L0 from true absorption (which occurs even at correct L0 due to feature hierarchy) would sharpen the definition and measurement of absorption.

11. **Multi-dimensional features and absorption**: Engels et al. (ICLR 2025) show some features are irreducibly multi-dimensional. SAEs forced to represent these as one-dimensional may exhibit absorption-like behavior where the multi-dimensional feature is partially captured by multiple latents.

12. **End-to-end training and absorption**: KL+MSE fine-tuning changes the sparsity-reconstruction tradeoff; its effect on absorption is unmeasured.

**Directions that appear saturated or risky:**
- Pure architecture comparisons without addressing absorption theoretically add limited insight
- Improving the sparsity-fidelity frontier (already well-studied: TopK, JumpReLU, Gated) without addressing interpretability failure modes
- Purely synthetic toy model studies without validation on real LLM SAEs
- Transcoder/crosscoder work as a primary focus (different paradigm that sidesteps rather than resolves absorption in SAEs)

**Cross-domain analogies with potential:**
- Hierarchical dictionary learning in signal processing / compressed sensing (multi-scale wavelet decompositions) may provide theoretical tools for analyzing feature hierarchy in SAEs
- Multi-label classification with label implication (where absorbing a specific label "absorbs" the general label) has been studied in the NLP literature and may provide relevant algorithmic solutions
- Neuroscience: Anthropic's circuit tracing work shows the practical value of reliable feature decomposition for understanding model computation; absorption directly undermines this program
- Information retrieval: Hierarchical topic models face analogous absorption of general topics by specific subtopics

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| sae-spelling (Chanin et al.) | High | MIT | Adopt | Canonical absorption metric implementation; directly reusable for measuring absorption rate on first-letter task and extensions |
| SAELens v6 (Bloom et al.) | High | MIT | Adopt | Standard training + evaluation library; all major SAE architectures supported; large community; v6 refactored training code |
| Gemma Scope SAEs (Google DeepMind) | High | Gemma ToS | Adopt | Pre-trained SAEs on 400+ configs; eliminates training cost for evaluation experiments |
| SAEBench (Karvonen et al.) | High | Apache 2.0 | Adopt | Standardized 8-metric evaluation; enables fair comparison with existing architectures; includes absorption task |
| TransformerLens | High | MIT | Adopt | Standard mech interp library with hook-based activation access; deep integration with SAELens; 50+ model support |
| Neuronpedia | High | Open source | Adopt | Interactive feature exploration, search across 50M+ latents; useful for qualitative absorption case studies |
| SynthSAEBench | High | check paper | Adopt | Controlled synthetic environment with known ground-truth feature hierarchies for studying absorption |
| OrtSAE (Korznikov et al.) | Medium | check paper | Extend | Orthogonality regularization is architecturally simple to implement in SAELens; extend to study interaction with absorption |
| Matryoshka SAE (Bussmann et al.) | Medium | MIT | Extend | Nested training objective well-specified; extend with different hierarchy structures or combine with orthogonality penalty |
| Balanced Matryoshka (Chanin et al.) | Medium | check paper | Extend | Compound multiplier tuning approach; extend to systematically explore absorption-hedging tradeoff |
| Masked Regularization (Narayanaswamy et al., 2026) | Medium | check paper | Extend | Very recent; token masking during training is simple to implement; extend to study optimal masking strategies for absorption reduction |
| KronSAE (2025) | Medium | check paper | Extend | Kronecker factorization encoder provides three absorption metrics (mean absorption fraction, full-absorption score, feature splits) as additional evaluation toolkit |
| Unified SDL Theory + Linear Representation Bench | Medium | check paper | Extend | Feature anchoring is a novel identifiability restoration technique; could be combined with existing architectures |
| sparse-but-wrong-paper (Chanin) | Medium | check paper | Extend | L0 analysis framework and proxy metric for correct L0; relevant for controlling L0 confound in absorption experiments |
| Language-Model-SAEs (OpenMOSS) | Medium | check paper | Compose | Distributed SAE training with multiple architecture support; useful if scaling to larger models |

**Highlight**: The **sae-spelling + SAELens + Gemma Scope + TransformerLens + SAEBench** combination gives a complete pipeline from activation extraction to SAE training to absorption evaluation with minimal implementation overhead. Related tools: **Sparsify** (lean SAE training focused on TopK), **Overcomplete** (SAE training for vision models), **SAE-Vis** (feature visualization, works with SAELens), **SAEDashboard** (Anthropic-style feature dashboards supporting CLTs). All are MIT-licensed (or equivalent research licenses) and have active community support. The absorption metric code in sae-spelling is directly reusable; the main extension needed is adapting it to feature hierarchies beyond first-letter spelling. SAEBench provides the standardized multi-metric framework for fair comparison across architectures. For the project's training-free constraint, Gemma Scope SAEs and SAEBench's open-sourced SAEs provide extensive pre-trained weights.

---

## References (Full Citations)

1. Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., & Bloom, J. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. *NeurIPS 2025*. arXiv:2409.14507. Code: https://github.com/lasr-spelling/sae-spelling

2. Elhage, N., Hume, T., Olsson, C., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*. https://transformer-circuits.pub/2022/toy_model/index.html

3. Bricken, T., Templeton, A., Batson, J., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/monosemantic-features

4. Templeton, A., Conerly, T., Marcus, J., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread*. https://transformer-circuits.pub/2024/scaling-monosemanticity/

5. Gao, L., Dupre la Tour, T., Tillman, H., et al. (2024). Scaling and evaluating sparse autoencoders. *ICLR 2025*. arXiv:2406.04093.

6. Rajamanoharan, S., Conmy, A., Smith, L., et al. (2024). Improving Dictionary Learning with Gated Sparse Autoencoders. arXiv:2404.16014.

7. Rajamanoharan, S., Lieberum, T., Sonnerat, N., et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. arXiv:2407.14435.

8. Bussmann, B., Nabeshima, N., Karvonen, A., & Nanda, N. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. *ICML 2025*. arXiv:2503.17547.

9. Chanin, D., Dulka, T., & Garriga-Alonso, A. (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756. Code: https://github.com/chanind/feature-hedging-paper

10. Korznikov, A., Galichin, A., Dontsov, A., Rogov, O., Tutubalina, E., & Oseledets, I. (2025). OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033.

11. Narayanaswamy, V., Thopalli, K., Kailkhura, B., & Sakla, W. (2026). Improving Robustness In Sparse Autoencoders via Masked Regularization. arXiv:2604.06495.

12. Karvonen, A., Rager, C., Lin, J., Tigges, C., Bloom, J., Chanin, D., Lau, R., Farrell, M., McDougall, C., Ayonrinde, K., Till, L., Wearden, J., Conmy, A., Marks, S., & Nanda, N. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. *ICML 2025*. arXiv:2503.09532.

13. Engels, J., Riggs, L., & Tegmark, M. (2024). Decomposing The Dark Matter of Sparse Autoencoders. arXiv:2410.14670.

14. Cui, J., Zhang, Q., Wang, Y., & Wang, Y. (2025). On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963.

15. Chen, S., Sheen, H., Xiong, X., Wang, T., & Yang, Z. (2025). Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders. arXiv:2506.14002.

16. Korznikov, A., Galichin, A., Dontsov, A., Rogov, O., Oseledets, I., & Tutubalina, E. (2026). Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? arXiv:2602.14111.

17. Song, X., Muhamed, A., Zheng, Y., et al. (2025). Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs. arXiv:2505.20254.

18. Muchane, M., Richardson, S., Park, K., & Veitch, V. (2025). Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures. arXiv:2506.01197.

19. Luo, Y., Zhan, Y., Jiang, J., et al. (2026). From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders. arXiv:2602.11881.

20. Hanni, K., Mendel, J., Vaintrob, D., & Chan, L. (2024). Mathematical Models of Computation in Superposition. arXiv:2408.05451.

21. Shu, D., Wu, X., Zhao, H., et al. (2025). A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models. *EMNLP 2025 Findings*. arXiv:2503.05613.

22. Ayonrinde, K., Pearce, M.T., & Sharkey, L. (2024). Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs. arXiv:2410.11179.

23. Martin-Linares, C.P., & Ling, J.P. (2025). Attribution-Guided Distillation of Matryoshka Sparse Autoencoders. arXiv:2512.24975.

24. Bereska, L., Tzifa-Kratira, Z., Samavi, R., & Gavves, E. (2025). Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability. arXiv:2512.13568.

25. [Author TBD]. (2024). A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima. arXiv:2512.05534.

26. Paulo, G., Shabalin, S., & Belrose, N. (2025). Transcoders Beat Sparse Autoencoders for Interpretability. arXiv:2501.18823.

27. [Author TBD]. (2025). Kronecker Factorization Improves Efficiency and Interpretability of Sparse Autoencoders (KronSAE). arXiv:2505.22255.

28. [Author TBD]. (2026). SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data. arXiv:2602.14687.

29. Smith, L., Rajamanoharan, S., Conmy, A., McDougall, C., et al. (2025). Negative Results for Sparse Autoencoders On Downstream Tasks and Deprioritising SAE Research. *DeepMind Safety Research Medium*. https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9

30. Li, X., et al. (2025). Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training. arXiv:2510.08855.

31. Gulko, A., Peng, Y., & Kumar, S. (2025). CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark of General Interpretability of Sparse Autoencoders. *BlackboxNLP 2025*. arXiv:2509.00691. Code: https://github.com/Yusen-Peng/CE-Bench

32. Tian, C., et al. (2025). Measuring Sparse Autoencoder Feature Sensitivity. arXiv:2509.23717.

33. Leask, P., Bussmann, B., Pearce, M., Bloom, J., Tigges, C., Al Moubayed, N., Sharkey, L., & Nanda, N. (2025). Sparse Autoencoders Do Not Find Canonical Units of Analysis. *ICLR 2025*. arXiv:2502.04878.

34. Chanin, D., & Garriga-Alonso, A. (2025). Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders. arXiv:2508.16560. Code: https://github.com/chanind/sparse-but-wrong-paper

35. Engels, J., Liao, I., Michaud, E.J., Gurnee, W., & Tegmark, M. (2024). Not All Language Model Features Are One-Dimensionally Linear. *ICLR 2025*. arXiv:2405.14860. Code: https://github.com/JoshEngels/MultiDimensionalFeatures

36. Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., et al. (2025). On the Biology of a Large Language Model. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/biology.html

37. Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramar, J., Dragan, A., Shah, R., & Nanda, N. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. *BlackboxNLP 2024*. arXiv:2408.05147. Weights: https://huggingface.co/google/gemma-scope

38. Minegishi, G., et al. (2025). Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words. arXiv:2501.06254.

39. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2024). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *ICLR 2024*. arXiv:2309.08600.

40. Bloom, J., Tigges, C., Duong, A., & Chanin, D. (2024). SAELens. GitHub: https://github.com/decoderesearch/SAELens

41. Karvonen, A. (2025). Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need. arXiv:2503.17272.

42. Wu, Z., et al. (2025). Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts. arXiv:2506.23845.

43. [Author TBD]. (2025). Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders. arXiv:2508.09363.

44. [Author TBD]. (2025). Interpretable Reward Model via Sparse Autoencoder. *AAAI 2025*.

45. Bussmann, B. (2024). BatchTopK Sparse Autoencoders. *NeurIPS 2024 Science of Deep Learning Workshop*. arXiv:2412.06410.

46. Lieberum, T., Rajamanoharan, S., et al. (2025). Gemma Scope 2. *Google DeepMind Technical Report*. https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/

47. Lin, J. (2024--2025). Neuronpedia: Open Interpretability Platform. https://www.neuronpedia.org. Open-sourced March 2025.

48. Bloom, J., Tigges, C., Duong, A., & Chanin, D. (2024--2025). SAELens v6: Training Sparse Autoencoders on Language Models. GitHub: https://github.com/decoderesearch/SAELens. PyPI: sae-lens v6.37.6.

49. Oldfield, J., Li, Y., Sherburn, M., Hao, Y., Engels, J., & Tegmark, M. (2024). The Geometry of Concepts: Sparse Autoencoder Feature Structure. arXiv:2410.19750. *Entropy* 27(4):344 (2025). https://www.mdpi.com/1099-4300/27/4/344

50. He, B., Maillard, S., Gurnee, W., et al. (2024). Efficient Dictionary Learning with Switch Sparse Autoencoders. *ICLR 2025*. https://openreview.net/forum?id=k2ZVAzVeMP

51. Michaud, E.J., Gorton, L., & McGrath, T. (2025). Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds. arXiv:2509.02565. https://www.goodfire.ai/research/sae-scaling-with-feature-manifolds

52. Muhamed, A., Diab, M., & Smith, V. (2024). Decoding Dark Matter: Specialized Sparse Autoencoders for Interpreting Rare Concepts in Foundation Models. *Findings of NAACL 2025*, pages 1604--1635. arXiv:2411.00743.

53. [Author TBD]. (2025). Orthogonal SAE: Feature Disentanglement Through Competition-Aware Orthogonality Constraints. *OpenReview* (ICLR 2026 submission). https://openreview.net/forum?id=YeYT9px8DL --- Distinct from OrtSAE; introduces sparsity-guided orthogonality constraints that dynamically identify and disentangle competing features via a three-phase curriculum; achieves SOTA on absorption for Gemma-2-2B.

54. Lindsey, J., Gurnee, W., et al. (2025). Circuit Tracing: Revealing Computational Graphs in Language Models. *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/attribution-graphs/methods.html --- Introduces cross-layer transcoders (CLTs) with 30M features on Claude 3.5 Haiku for attribution graphs; open-sourced in May 2025; replicated on Gemma-2-2B and Llama-3.1-1B by community (circuit-tracer library + EleutherAI Attribute library). Relevant context: features used in circuit tracing must be reliable, making absorption a direct obstacle to this program.

55. [Author TBD]. (2025). FaithfulSAE: Towards Capturing Faithful Features with Sparse Autoencoders without External Dataset Dependencies. arXiv:2506.17673 --- Proposes methods for faithful feature capture without external datasets; potentially relevant to absorption since faithfulness and absorption are related failure modes.

56. [Author TBD]. (2025). Superposition Yields Robust Neural Scaling. *NeurIPS 2025 Best Paper Runner-Up*. --- Provides mechanistic explanation for neural scaling laws via representation superposition; validates on OPT, Pythia, Qwen; loss scales as L proportional to 1/m in strong superposition regime driven by geometric interference between feature vectors. Relevant context: establishes that superposition (the root cause context for absorption) is a geometric inevitability of compressing sparse concepts into dense spaces.

57. [Author TBD]. (2025). CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features. arXiv:2508.12535 --- Selects task-relevant SAE features by correlating generated-token activations with outcomes for generation-time steering; relevant to understanding how correlated features (which drive both hedging and absorption) can be leveraged rather than just mitigated.

58. Saraswatula, A., & Klindt, D. (2025). Data Whitening Improves Sparse Autoencoder Learning. arXiv:2511.13981. --- Shows PCA whitening of input activations improves SAE interpretability metrics on SAEBench despite minor reconstruction drops; challenges assumption that interpretability aligns with optimal sparsity-fidelity trade-off.

59. Fereidouni, M., Haider, M.U., Ju, P., & Siddique, A.B. (2025). Evaluating Sparse Autoencoders for Monosemantic Representation. arXiv:2508.15094. --- First systematic evaluation via activation distribution lens; introduces concept separability score based on Jensen-Shannon distance; shows SAEs reduce polysemanticity vs base models; proposes APP (Attenuation via Posterior Probabilities) intervention method.

60. Stevinson, E., Prieto, L., Barsbey, M., & Birdal, T. (2025). Adversarial Attacks Leverage Interference Between Features in Superposition. arXiv:2510.11709. --- Shows adversarial perturbations exploit interference between superposed features; establishes superposition suffices to create adversarial vulnerability; connects representational compression to adversarial robustness.

61. Zhang, Q., Wang, Y., Cui, J., Pan, X., Lei, Q., Jegelka, S., & Wang, Y. (2024). Beyond Interpretability: The Gains of Feature Monosemanticity on Model Robustness. arXiv:2410.21331. --- Shows monosemantic features improve robustness across input/label noise, few-shot learning, and OOD generalization; provides empirical and theoretical evidence that monosemanticity promotes better feature separation and more robust decision boundaries.

62. Tolooshams, B., Shen, A., & Anandkumar, A. (2025). Mechanistic Interpretability with Sparse Autoencoder Neural Operators. arXiv:2509.03738. --- Introduces SAE-NOs operating in infinite-dimensional function spaces; SAE-FNO parameterizes concepts as integral operators in Fourier domain; improved stability, robustness to distribution shifts, and generalization across discretizations.

63. Scherlis, A., Sachan, K., Jermyn, A.S., Benton, J., & Shlegeris, B. (2022). Polysemanticity and Capacity in Neural Networks. arXiv:2210.01892. --- Analyzes polysemanticity through feature capacity lens; shows optimal capacity allocation tends to monosemantically represent most important features and polysemantically represent less important ones; finds block-semi-orthogonal structure in embedding space.

64. Lecomte, V., Thaman, K., Schaeffer, R., Bashkansky, N., Chow, T., & Koyejo, S. (2023). What Causes Polysemanticity? An Alternative Origin Story of Mixed Selectivity from Incidental Causes. arXiv:2312.03096. --- Shows polysemanticity can arise incidentally even with ample neurons; demonstrates incidental polysemanticity from regularization and neural noise; calls for research quantifying performance-polysemanticity tradeoff.

---

## Appendix: Sixth-Pass Survey Additions (2026-04-27)

This appendix documents new findings from the sixth literature survey pass conducted on 2026-04-27.

### Newly Identified Papers

1. **Data Whitening Improves Sparse Autoencoder Learning** (Saraswatula & Klindt, 2025): PCA whitening of input activations improves interpretability metrics on SAEBench. Relevant because whitening reduces correlations that may contribute to both hedging and absorption.

2. **Evaluating Sparse Autoencoders for Monosemantic Representation** (Fereidouni et al., 2025): Introduces concept separability score and APP intervention. Provides additional evaluation tools for assessing feature quality beyond absorption metrics.

3. **Adversarial Attacks Leverage Interference Between Features in Superposition** (Stevinson et al., 2025): Connects superposition (root cause of absorption) to adversarial vulnerability. Provides broader context for why feature interference matters.

4. **Beyond Interpretability: The Gains of Feature Monosemanticity on Model Robustness** (Zhang et al., 2024): Shows monosemanticity improves robustness. Motivates why solving absorption (which undermines monosemanticity) matters for model reliability.

5. **Mechanistic Interpretability with Sparse Autoencoder Neural Operators** (Tolooshams et al., 2025): Functional representation approach that may avoid some fixed-dimensional SAE pathologies.

6. **Polysemanticity and Capacity in Neural Networks** (Scherlis et al., 2022): Foundational analysis of capacity allocation; relevant for understanding why hierarchical features lead to absorption.

7. **What Causes Polysemanticity? An Alternative Origin Story** (Lecomte et al., 2023): Shows polysemanticity can arise from incidental causes; relevant for disentangling absorption from other feature quality issues.

### Cross-Cutting Themes from This Pass

- **Whitening as a preprocessing step**: Emerging consensus that input preprocessing (whitening) can improve SAE quality, complementing architectural innovations.
- **Feature sensitivity as a broader framework**: Tian et al.'s sensitivity metric provides a unifying lens that subsumes absorption as a special case.
- **Theoretical foundations maturing**: Multiple 2025 papers (unified SDL theory, bias adaptation, capacity allocation) provide increasingly rigorous theoretical grounding for understanding absorption.
- **Critical re-evaluation of SAEs**: The "Sanity Checks" paper and DeepMind's deprioritization represent a significant shift in community sentiment, making absorption research more urgent.

### Updated Implementation Recommendations

Based on this pass, the recommended tool stack remains:
- **SAELens v6** + **TransformerLens** for training and activation extraction
- **SAEBench** for standardized evaluation (now including absorption, sensitivity, and CE-Bench metrics)
- **sae-spelling** for absorption metric implementation
- **Gemma Scope 2** (Dec 2025) for pre-trained SAEs including Matryoshka variants
- **Neuronpedia** for qualitative feature exploration

New addition: Consider PCA whitening preprocessing (Saraswatula & Klindt, 2025) as a baseline enhancement in all SAE training experiments.
