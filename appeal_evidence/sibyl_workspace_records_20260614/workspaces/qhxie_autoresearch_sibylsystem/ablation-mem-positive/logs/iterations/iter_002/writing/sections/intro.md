# 1. Introduction

Sparse Autoencoders (SAEs) have become the dominant tool for mechanistic interpretability of large language models (LLMs), enabling decomposition of neural network activations into human-interpretable features. However, **feature absorption**—a phenomenon where parent features are subsumed by more specific child features during sparse optimization—undermines the reliability of circuit discovery and feature attribution. When a parent feature is absorbed, it exhibits systematic false negatives: it fails to activate on inputs containing the parent concept when a child concept is also present, creating an "interpretability illusion" that challenges the fundamental premise of SAE-based analysis.

Despite extensive study of feature absorption (Chanin et al., 2024; Bricken et al., 2023), the field lacks a quantitative theoretical framework predicting where absorption becomes severe. Prior work has documented the phenomenon but not characterized its threshold behavior or scaling properties. This gap limits our ability to design SAEs that minimize absorption and to interpret absorption metrics reliably.

A particularly puzzling observation is the **actionability paradox** reported by Basu et al. (2026): SAE-based steering achieves 98.2% AUROC for feature detection yet produces 0% measurable output change. Near-perfect feature detection does not guarantee steering utility, suggesting that absorbed features may route through pathways resistant to direct intervention. Understanding why remains an open problem.

This paper presents the first systematic application of statistical physics phase transition theory to SAE feature absorption. We demonstrate that absorption exhibits quasi-critical threshold behavior at a critical sparsity $\lambda_c \approx 5 \times 10^{-5}$, with susceptibility peaking at $\chi_{max} = 11.19$. We establish finite-size scaling with critical exponent $\nu = 3$ (R² = 0.951), representing the first quantitative measurement of this scaling law in SAE literature. Perhaps most surprisingly, we discover that absorbed features exhibit coefficient of variation (CV) approximately 733× higher than non-absorbed features (7.33 vs. 0.01)—a finding we term the **variance paradox**. This suggests absorption selectively preserves context-sensitive, high-variance information rather than uniformly degrading signal, providing a potential mechanism for the actionability paradox: high-CV absorbed features may route through specialized child channels that resist direct steering intervention.

We acknowledge several limitations: all experiments use GPT-2 Small (117M parameters), the phase transition was gradual rather than sharp (chi_ratio = 1.88 < 3.0 threshold), and cross-layer measurement at the true critical point remains future work. Nevertheless, our findings offer both a theoretical framework for understanding absorption phenomenology and practical guidance for interpreting SAE-based feature analysis.

## 1.1 Contributions

This paper makes the following contributions:

1. **First phase transition framework for SAE absorption**: We formalize feature absorption as a quasi-critical phenomenon with a measurable critical sparsity threshold, connecting SAE interpretability to established statistical physics theory.

2. **First measurement of finite-size scaling**: We demonstrate that absorption transition width scales with dictionary size as $\delta\lambda \propto N^{-1/\nu}$, establishing $\nu = 3$ as the critical exponent with R² = 0.951 scaling collapse quality.

3. **Discovery of the variance paradox**: Absorbed features exhibit CV approximately 733× higher than non-absorbed features (7.33 vs. 0.01), indicating absorption selectively preserves context-sensitive high-variance information—a genuine empirical discovery requiring new theoretical explanation.

4. **Connection to the actionability paradox**: We provide a mechanism linking the variance paradox to Basu et al.'s observation: high-CV absorbed features may route through specialized child channels that resist direct steering, explaining why near-perfect detection does not guarantee steering utility.

5. **Evidence against the "layer as temperature" narrative**: At standard sparsity (λ = 0.001), absorption rate saturates at 1.0 for all layers, contradicting the hypothesis that layer depth provides an analogous temperature axis for controlling absorption criticality.

These contributions are evaluated against six hypotheses derived from the phase transition framework (Table 1), with four supported, one supported in reversed form (variance paradox), and two not supported.

<!-- FIGURES
- None
-->