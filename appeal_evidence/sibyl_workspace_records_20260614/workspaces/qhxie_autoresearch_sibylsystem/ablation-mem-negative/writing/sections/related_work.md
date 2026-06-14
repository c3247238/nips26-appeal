# 2. Related Work

## 2.1 Sparse Autoencoders and Interpretability

Sparse autoencoders with an L1 sparsity penalty on the latent representation were introduced by Makhzani and Frey [2014] and have become central to mechanistic interpretability of language models. The standard SAE architecture learns an overcomplete dictionary $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{SAE}}}$ that reconstructs residual stream activations $x \in \mathbb{R}^{d_{\text{model}}}$ via a sparse latent code $z \in \mathbb{R}^{d_{\text{SAE}}}$. Recent architectural variants include JumpReLU [Rajamanoharan et al., 2024], which uses a learned threshold gating mechanism, and TopK SAE [Gao et al., 2024], which enforces sparsity by keeping only the $k$ highest activations. Large-scale pretrained SAE suites such as GemmaScope [Lieberum et al., 2024] have made SAEs accessible for broad research. The SAELens and TransformerLens ecosystems provide standardized tooling for training and evaluation.

## 2.2 Feature Absorption

Chanin et al. [2024] established feature absorption as a fundamental failure mode in SAEs. Their framework defines absorption as the suppression of a hierarchical parent feature when a more specific child feature co-occurs, with the parent activation merged into the child's latent representation to increase sparsity while maintaining reconstruction quality. Their detection protocol uses $k$-sparse probing with integrated gradients ablation, requiring (1) known parent concepts, (2) supervised probe training, and (3) ground-truth feature labels. This fully supervised approach limits detection to known concepts.

Absorption is connected to superposition [Elhage et al., 2022], the phenomenon where neural networks represent more features than dimensions by encoding features in non-orthogonal directions. Polysemanticity---when a single feature responds to multiple unrelated inputs [Schubert et al., 2024]---is related but distinct: polysemanticity implies unrelated concepts, while absorption implies hierarchical relationships. Hierarchical SAEs (HSAE) [Chen et al., 2025] propose an architectural mitigation by explicitly modeling feature hierarchies, but require retraining.

## 2.3 Co-Occurrence Analysis of SAE Features

Co-occurrence analysis has been used to study SAE feature structure, but not for absorption detection. "The Geometry of Concepts" (arXiv:2410.19750) applies spectral clustering to phi coefficient matrices to discover functional lobes---groups of features with correlated activation patterns---but does not address parent-child absorption relationships. Clarke et al. [2024] analyze co-occurrence for compositionality and ambiguity, again without absorption-specific detection. The gap is clear: no prior method uses co-occurrence patterns to identify absorbed parent-child pairs.

## 2.4 Absorption Mitigation Methods

Architectural solutions including Matryoshka SAE, OrtSAE, KronSAE, and ATM address absorption through modified training objectives or dictionary structures, but all require retraining the SAE. SAEBench [Karvonen et al., 2025] includes an absorption metric, but uses probe projection---still requiring supervised concept labels. No training-free, inference-time compensation method exists.

Table 3 summarizes the positioning of UAD against prior work.

| Method | Supervision | Probe Required | Ground-Truth Parent | F1 | Applicability |
|--------|-------------|----------------|---------------------|-----|---------------|
| Chanin et al. (2024) | Full | Yes | Yes | N/A (defines truth) | Known concepts only |
| SAEBench | Partial | Yes | No | N/A | Known concepts only |
| UAD (Ours) | **None** | **No** | **No** | **0.725** | **Any SAE, any corpus** |

UAD is the only method requiring zero supervision---a qualitative shift, not an incremental improvement. Unlike "Geometry of Concepts," which discovers general feature structure, UAD specifically targets parent-child absorption relationships. Unlike Matryoshka SAE or OrtSAE, our preliminary DFDA method requires no architectural changes or retraining.

<!-- FIGURES
- Table 3: inline — Comparison of UAD with prior absorption detection methods
-->
