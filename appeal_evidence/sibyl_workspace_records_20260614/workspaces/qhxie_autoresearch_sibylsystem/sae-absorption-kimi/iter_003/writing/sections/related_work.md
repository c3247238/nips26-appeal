# 2. Related Work

## 2.1 From Superposition to Sparse Autoencoders

The theoretical foundation for SAEs rests on the superposition hypothesis articulated by Elhage et al. (2022): neural networks represent more features than they have dimensions by encoding them as non-orthogonal linear combinations of activations. This overcomplete representation is necessary because the number of meaningful concepts in language far exceeds the residual-stream dimension, but it creates polysemanticity---individual neurons that respond to multiple unrelated stimuli---making direct interpretation intractable.

Bricken et al. (2023) demonstrated that SAEs could recover monosemantic features from a one-layer transformer, sparking rapid expansion of the field. Cunningham et al. (2023) scaled SAE training to larger models, and by 2024--2025 the literature spans multiple architectures (ReLU, TopK, JumpReLU, Gated, Matryoshka), comprehensive benchmarks, and applications beyond language (vision, proteins, RNA). Templeton et al. (2024) extracted millions of interpretable features from Claude 3 Sonnet, demonstrating that SAEs can operate at frontier-model scale.

The central tension in this line of work is between reconstruction fidelity and interpretability. SAEs optimize a trade-off between reconstructing the original activations and enforcing sparsity in the latent representation. The sparsity-reconstruction Pareto frontier has become the standard way to compare architectures, but Karvonen et al. (2025) showed that gains on this frontier do not reliably predict downstream interpretability outcomes. Our work extends this skepticism to a specific downstream metric: absorption.

## 2.2 Feature Absorption and Its Mitigation

Chanin et al. (2024) identified feature absorption as a structural failure mode analytically incentivized by sparsity loss. When parent and child features co-occur, the SAE allocates capacity to children at the parent's expense, creating "holes" in feature coverage. Their analytical proof applies to hierarchical features with parent-child containment structure, and their empirical validation used first-letter classification tasks.

Subsequent work has pursued two directions: mitigation and characterization. Bussmann et al. (2025) introduced Matryoshka SAEs, which learn nested dictionaries of increasing size and reduced absorption rates from 0.49 to 0.05 on the first-letter task. Korznikov et al. (2025) enforced orthogonality constraints (OrtSAE), reducing absorption by 65% with 4--11% compute overhead. Zhan et al. (2026) proposed hierarchical SAEs with explicit tree-structured constraints. Rajamanoharan et al. (2024) introduced JumpReLU and Gated SAEs, which improve reconstruction fidelity and sparsity trade-offs but do not directly target absorption.

A critical counterpoint is Chanin's (2025) finding of feature hedging---the opposite failure mode where reconstruction loss drives correlated features to share latents rather than split them. Matryoshka SAEs, while reducing absorption, increase hedging in inner dictionary levels. This reveals that absorption and hedging sit on a Pareto frontier: no known architecture simultaneously minimizes both. Our finding that the absorption metric responds to non-hierarchical correlations adds a third dimension to this trade-off space: the metric itself may be confounded by correlation structure that is neither absorption nor hedging.

## 2.3 Benchmarks and Metric Validation

SAEBench (Karvonen et al., 2025) standardized eight evaluations for SAE comparison, including absorption, sparse probing, automated interpretability, loss recovery, and feature disentanglement. The absorption evaluation adapts Chanin et al.'s protocol: 26 first-letter hierarchies, ground-truth logistic probes, k-sparse probing with feature-splitting detection, and a composite absorption score. SAEBench has become the dominant community benchmark, with 200+ SAEs evaluated and an interactive leaderboard at neuronpedia.org/sae-bench.

Despite its centrality, the absorption metric has not undergone construct-validation testing. The first-letter task is a proxy for general hierarchical absorption, but whether it generalizes to semantic, syntactic, or factual hierarchies is unknown. Chanin et al. (2024) explicitly noted this gap, listing "finding examples of feature absorption unrelated to character identification" as future work. Our study is the first to address this gap.

The broader issue of proxy metric validity has received increasing attention. Kantamneni et al. (2025) showed that SAEs do not consistently outperform strong non-SAE baselines on downstream sparse probing tasks, questioning whether SAE-based features carry genuine utility beyond interpretability appeal. Lieberum et al. (2024) found that feature interpretability ratings from automated methods correlate weakly with human judgments. These findings align with our results: metrics that appear meaningful in controlled settings may not generalize to the phenomena they claim to measure.

## 2.4 Construct Validity in Machine Learning Evaluation

Construct validity---the degree to which a measurement instrument captures the theoretical construct it claims to measure---is a cornerstone of psychometric theory (Cronbach & Meehl, 1955) but is rarely applied to ML benchmarks. In natural language processing, the validity of automatic metrics (BLEU, ROUGE, BERTScore) has been repeatedly questioned; these metrics correlate poorly with human judgments on generation tasks (Novikova et al., 2017). In computer vision, ImageNet accuracy has been criticized as a proxy for general visual understanding (Geirhos et al., 2020).

Within mechanistic interpretability specifically, the gap between proxy metrics and ground-truth causal effects is acute. Probing-based metrics measure "knowledge about" a concept, but causal interventions (activation patching, ablation) are needed to establish "use of" that concept (Belinkov, 2022). The absorption metric is probe-based: it detects whether parent-feature information is present in SAE latents, not whether the model causally depends on that information. Our Random-SAE finding---that a control with no learned structure achieves comparable semantic-hierarchy absorption---suggests the metric may be measuring geometric properties of the probe-task interaction rather than a meaningful structural property of the SAE.

## 2.5 Positioning This Work

Our study occupies the intersection of three literatures: SAE failure-mode characterization, benchmark validation, and construct-validity assessment. Unlike architecture papers that propose new SAE variants and report absorption scores (Bussmann et al., 2025; Korznikov et al., 2025), we do not introduce a new method. Unlike theoretical work that analyzes absorption as an optimization property (arXiv:2512.05534), we do not derive new bounds or guarantees. Instead, we ask a methodological question: does the dominant absorption metric measure what it claims to measure?

This question is timely because absorption has become a primary criterion for architecture comparison. Papers routinely report first-letter absorption as a key result, and SAEBench's interactive leaderboard ranks SAEs partly by this metric. If the metric lacks construct validity, these comparisons may be misleading. Our findings do not invalidate the first-letter task as a controlled experimental setting---it remains useful for studying absorption dynamics---but they caution against treating first-letter scores as a proxy for general absorption behavior.

We now describe the measurement protocol that adapts the SAEBench absorption evaluator to semantic hierarchies, non-hierarchical controls, and a randomized baseline.

<!-- FIGURES
- None
-->
