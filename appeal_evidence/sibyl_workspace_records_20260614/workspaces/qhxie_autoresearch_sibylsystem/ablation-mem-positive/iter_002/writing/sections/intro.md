## 1. Introduction

### 1.1 Motivation and Background

Sparse autoencoders (SAEs) have become the dominant unsupervised tool for extracting interpretable features from the internal activations of language models (Cunningham et al., 2023; Elhage et al., 2022). By decomposing high-dimensional residual stream representations into sparse combinations of latent features, SAEs promise a mechanistic understanding of how transformers encode and manipulate semantic concepts. Recent scaling efforts have produced pretrained SAE suites for models such as Gemma 2 (GemmaScope, Lieberum et al., 2024) and community efforts have released SAEs for GPT-2 (Templeton et al., 2024; Bloom, 2024), enabling systematic feature discovery across network depth.

A critical pathology threatens this program: feature absorption, where a general (parent) feature is subsumed by a more specific (child) feature during sparse optimization (Chanin et al., 2024). When absorption occurs, a latent that appears monosemantic---activating strongly on a specific concept---in fact inherits responsibility for broader related concepts that have been "absorbed" into it. This creates an interpretability illusion: the analyst sees a clean feature detector but misses systematic false negatives on the absorbed parent concepts. Chanin et al. (2024) introduced the first systematic study of absorption, using first-letter spelling tasks and ablation-based metrics to quantify its prevalence in JumpReLU SAEs.

### 1.2 Problem Statement

Despite this foundational work, three critical gaps limit our understanding of feature absorption.

**Gap 1: Probe dataset scalability.** Existing absorption studies rely on small semantic probe datasets. Chanin et al. (2024) used first-letter spelling tasks with limited generalizability, and our prior pilot studies following the Chanin et al. protocol observed 3 failed probes out of 30 (AUROC $<$ 0.7) with only 5 hyponyms per category (see `iter_002/exp/results/pilot_summary.md`), raising questions about whether absorption rates are measured on a representative sample of semantic concepts.

**Gap 2: Cross-architecture validation between JumpReLU and standard ReLU SAEs is missing.** All existing absorption analyses have been conducted on JumpReLU SAEs (GemmaScope), leaving open whether findings generalize to other architectures. The training-free absorption detector $A_j = \|d_j\|^2 / (d_j^\top e_j)$ (Chanin et al., 2024) has only been validated on constrained-decoder SAEs, and its behavior on architectures with different normalization schemes remains unknown.

**Gap 3: Metric stability.** The ablation-based absorption metric---measuring accuracy differences before and after ablating the top latent---exhibits near-zero scores in practice, suggesting functional insensitivity rather than true absence of absorption (Chanin et al., 2024). A projection-based alternative measures the fraction of probe weight captured by the top latent, but its stability across the two most widely used architectural families has not been tested.

### 1.3 Contributions

This paper addresses these three gaps through systematic experiments on two architectural families: GemmaScope JumpReLU SAEs and GPT-2 Small ReLU SAEs. Our contributions are:

1. **First systematic architecture-aware absorption analysis.** We compare absorption patterns across JumpReLU and ReLU SAEs at matched relative layer depths---the two most widely used architectural families---as a first step toward broader cross-architecture validation.

2. **Scalable semantic probe pipeline.** We expand from 5 to 15 hyponyms per semantic category (10 WordNet categories), achieving 100% valid probes on GemmaScope (30/30, mean AUROC = 0.980) and GPT-2 (30/30, mean AUROC = 0.974), eliminating the class-balance concerns that plagued prior measurements.

3. **Unexpected layer-dependent $A_j$ correlation pattern.** While our original hypothesis (H2) that $A_j$ would correlate positively across all GPT-2 layers failed (mean $\rho = -0.194$), we discovered a non-monotonic correlation with projection absorption: strongly positive at mid-layers ($\rho = 0.705$ at layer 8, $p = 0.023$) but negative at shallow and deep layers ($\rho = -0.590$ at layer 5, $\rho = -0.697$ at layer 10). Sign flips between layer 8 and both adjacent layers are statistically significant ($z = 2.91$, $p = 0.004$ vs layer 5; $z = 3.25$, $p = 0.001$ vs layer 10).

4. **Evidence that decoder norm constraints persist across the two architectures tested.** Both GemmaScope JumpReLU and GPT-2 ReLU maintain decoder norms of approximately 1.0, suggesting that norm constraints may emerge from training dynamics rather than architectural design alone. Architectural effects cannot be ruled out with only two SAE families.

5. **Validation that projection-based absorption is stable across architectures.** Projection absorption differs by 7.7% between JumpReLU (98.2%) and ReLU (91.2%) SAEs. While this difference is statistically significant ($p < 0.001$), the absolute magnitude is small and both architectures show consistently high rates ($>$90%), suggesting projection-based metrics capture a robust architectural invariant.

### 1.4 Paper Organization

Section 2 reviews related work on SAE architectures, absorption pathologies, and evaluation benchmarks. Section 3 describes our methodology: the scalable semantic probe construction pipeline that addresses Gap 1, the three absorption metrics (ablation-based, projection-based, and training-free $A_j$), and our cross-architecture comparison protocol. Section 4 presents experimental results across three studies: scaled semantic probes on GemmaScope (E3v2), GPT-2 $A_j$ validation (E6v2), and cross-architecture comparison (E7). Section 5 discusses the implications of our findings for architecture stability, decoder norm constraints, and the layer-dependent detection pattern. Section 6 concludes with a summary of contributions and directions for future work.

<!-- FIGURES
- None
-->
