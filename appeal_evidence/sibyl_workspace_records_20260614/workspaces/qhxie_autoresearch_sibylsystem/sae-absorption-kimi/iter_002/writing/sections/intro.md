# 1 Introduction

### 1.1 The Rise of Feature Absorption as a Central Pathology

Sparse autoencoders (SAEs) have become the dominant tool for decomposing neural network activations into interpretable features \cite{chanin2024absorption}. By learning an overcomplete dictionary of latent directions, SAEs aim to reverse superposition---the phenomenon where a network represents more concepts than it has dimensions by encoding them in non-orthogonal directions \cite{cui2025monosemantic}. The ideal outcome is monosemanticity: each latent dimension encodes a single, human-interpretable concept. In practice, SAEs suffer from polysemanticity and a related failure mode known as feature absorption.

Feature absorption occurs when a parent feature's information is lost because the SAE allocates representational capacity to child features at the parent's expense \cite{chanin2024absorption}. Chanin et al. proved analytically that the sparsity loss used to train SAEs creates a structural incentive for absorption whenever features stand in a parent-child (hierarchical) relationship. The more sparse the representation, the stronger this incentive becomes. This makes absorption not merely an empirical observation but a theoretically predicted consequence of the training objective.

The SAEBench benchmark \cite{adamkarvonen2025} standardized the measurement of feature absorption, making it one of eight canonical evaluations for SAEs. The metric has since shaped architectural development: Matryoshka SAEs \cite{bussmann2025matryoshka}, OrtSAE \cite{korznikov2025ortsae}, and Hierarchical SAEs \cite{zhan2026hierarchical} all report absorption reductions as primary contributions. A benchmark that influences this many follow-up architectures demands rigorous validation.

### 1.2 The First-Letter Benchmark and Its Limitations

The current SAEBench absorption metric uses first-letter classification tasks. The canonical example is a hierarchy where "starts with S" is the parent and "short" is the child. Ground-truth logistic probes are trained on base-model residual-stream activations to classify these categories, and the absorption score measures how much parent-feature information is lost when the SAE represents the input. The advantages of this benchmark are clear: ground-truth labels are unambiguous, causal ablations are computationally tractable, and the hierarchy definition is clean.

Yet the limitations are equally clear. First-letter hierarchies are an artificial task. They differ from real semantic hierarchies in frequency structure, representational geometry, and the semantic coherence of the parent-child relationship. The question of whether absorption on "starts with S" $\supset$ "short" predicts absorption on "animal" $\supset$ "mammal" $\supset$ "dog" has never been tested. Chanin et al. \cite{chanin2024absorption} explicitly noted this gap, calling for "finding examples of feature absorption unrelated to character identification" as future work. That call has gone unanswered.

If first-letter absorption is uncorrelated with semantic-hierarchy absorption, then architectures optimized for the former may not improve behavior on the latter. The community would be optimizing the wrong target.

### 1.3 Research Questions

This paper conducts the first systematic construct-validity study of the SAEBench feature absorption metric. We test three research questions:

- **RQ1:** Do first-letter absorption scores correlate with semantic-hierarchy absorption scores across diverse SAE architectures?
- **RQ2:** Is the metric specific to hierarchical features, or does it detect absorption-like behavior in non-hierarchical correlated features?
- **RQ3:** How robust is the correlation across feature-splitting thresholds and base models?

### 1.4 Contributions

Our contributions are:

1. **First construct-validity test** of the dominant SAE absorption metric on real semantic hierarchies drawn from WordNet.
2. **Evidence that the metric lacks hierarchy specificity**: non-hierarchy correlated features show higher absorption than semantic hierarchies ($t = -4.748$, $p = 0.003$).
3. **Random-SAE control** revealing that semantic-hierarchy absorption scores are identical between trained and random SAEs (0.352), indicating the metric captures artifacts unrelated to learned structure.
4. **Open-source replication materials** (WordNet hierarchy dataset, evaluation code, per-hierarchy breakdowns).

The remainder of the paper is organized as follows. Section 2 describes the measurement protocol, including SAE selection, hierarchy construction, and the absorption formula. Section 3 presents the empirical results for all three hypotheses. Section 4 interprets the findings and discusses implications for benchmark design. Section 5 concludes with recommendations.

<!-- FIGURES
- None
-->
