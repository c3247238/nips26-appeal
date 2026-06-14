# 2. Related Work

Our study addresses three open questions about SAE feature absorption: whether it can be detected without probe directions, whether corpus statistics predict it, and whether it matters for downstream performance. We organize prior work along these three axes and identify the gap each component fills.

## 2.1 Feature Absorption: Definition and Measurement

Chanin et al. (arXiv:2409.14507; NeurIPS 2025) established the canonical formalization: a parent latent $j$ (e.g., "first letter = A") is absorbed by a child latent $c$ (e.g., "the token 'April'") when $a_j = 0$ on inputs where $a_j$ should be positive, while $a_c > 0$ on those same inputs. Their measurement protocol trains linear probes to identify ground-truth feature directions, then uses integrated gradients to attribute false negatives to specific absorbing latents. On mid-layer Gemma Scope SAEs (Gemma 2 2B, widths 16k and 65k), they report absorption rates of 15--35% across the 26 first-letter classes. This rate is explicitly acknowledged as a lower bound: it captures only full, single-latent absorption (what we term Type I), ignoring partial suppression and distributed multi-child effects.

Karvonen et al. (arXiv:2503.09532) incorporated the Chanin metric as one of eight evaluation dimensions in SAEBench, a standardized benchmark covering 200+ open-source SAEs across eight architectures. SAEBench reports absorption scores alongside sparse probing, RAVEL, SCR, and unlearning, but does not analyze correlations between absorption and the other metrics---a gap our Component 3 addresses.

Tian et al. (arXiv:2509.23717) frame absorption as a special case of poor feature *sensitivity*: a feature that activates selectively on its target concept but fails to activate on similar inputs has low sensitivity. Their scalable sensitivity metric reveals that many features rated as interpretable by activation-example inspection nonetheless have poor recall, consistent with the partial absorption (Type II) our taxonomy quantifies.

A LessWrong post ("Looking for Feature Absorption Automatically") attempted cosine-similarity-based detection using ablation effect vectors rather than decoder geometry or co-activation statistics. The approach yielded negative results. Our competition coefficient $\alpha_{ij}$ differs in two respects: it operates on co-activation frequencies rather than ablation effects, and it multiplies niche overlap $\sigma_{ij}$ by the frequency ratio $f_j / f_i$ to capture the asymmetric competitive pressure from frequent children on rare parents.

**Gap.** All existing absorption measurement methods require pre-specified probe directions---the researcher must know which features to look for. This restricts absorption analysis to controlled proxy tasks (first-letter spelling) and precludes systematic study of absorption in safety-relevant or semantically rich feature hierarchies. No prior work computes a probe-free absorption predictor from activation statistics alone.

## 2.2 SAE Architectures and Absorption Reduction

Multiple SAE variants reduce absorption, but no unified mechanistic account explains why.

Rajamanoharan et al. (arXiv:2407.14435) introduced JumpReLU SAEs, which train $L_0$ directly via the straight-through estimator. JumpReLU achieves state-of-the-art reconstruction fidelity on Gemma 2 9B but, paradoxically, exhibits the highest absorption rates in SAEBench---consistent with the hypothesis that longer training amplifies the sparsity-gradient dynamics that produce absorption.

Bussmann et al. (arXiv:2503.17547; ICML 2025) proposed Matryoshka SAEs, which train nested dictionaries of increasing width simultaneously. The nested structure explicitly allocates general features to smaller inner dictionaries and specific features to larger outer ones, reducing the parent-child competition that drives absorption. Matryoshka SAEs achieve the best absorption scores in SAEBench while maintaining competitive reconstruction.

Korznikov et al. (arXiv:2509.22033) enforce pairwise orthogonality on decoder columns (OrtSAE), reducing absorption by 65% relative to standard SAEs. Orthogonality directly reduces the decoder cosine similarity between parent and child features, which in our framework corresponds to reducing the candidate pair set in the $\alpha_{ij}$ pre-filter (pairs with $\cos(\mathbf{d}_i, \mathbf{d}_j) > 0.15$).

Narayanaswamy et al. (arXiv:2604.06495) introduced masked regularization, which randomly masks high-frequency tokens during SAE training to disrupt co-occurrence patterns. This directly suppresses the co-activation statistics that enter our niche overlap term $\sigma_{ij}$, providing independent motivation for the competition coefficient framework.

Li et al. (arXiv:2510.08855) proposed Adaptive Temporal Masking (ATM SAE), which dynamically scores per-latent importance based on activation magnitude, frequency, and reconstruction contribution. ATM achieves the lowest reported absorption score (0.0068 vs. TopK 0.1402 and JumpReLU 0.0114 on Gemma-2-2B).

**Gap.** These architectures reduce absorption through diverse mechanisms (nesting, orthogonality, masking, temporal weighting), but no single quantity unifies them. Our competition coefficient $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ provides a candidate unifying lens: each architecture can be analyzed in terms of how it reduces $\sigma_{ij}$ (niche overlap), $f_j / f_i$ (frequency imbalance), or both.

## 2.3 SAE Evaluation and Downstream Impact

SAEBench (Karvonen et al., arXiv:2503.09532) provides the first multi-metric evaluation suite, revealing that proxy metrics (CE loss recovered, sparsity) do not reliably predict practical SAE performance. This finding motivates our H3 analysis: if proxy metrics are unreliable, is the absorption metric itself predictive of downstream quality?

Korznikov et al. (arXiv:2602.14111) present a provocative negative result: on synthetic benchmarks, SAEs recover only 9% of ground-truth features, and random baselines match trained SAEs on interpretability and sparse probing tasks. While this finding pertains to synthetic settings and does not directly invalidate SAEs on real models, it raises the stakes for demonstrating that absorption---a metric defined on real models---predicts real downstream performance.

DeepMind's safety research team publicly deprioritized SAE research in 2025 after finding that dense linear probes dramatically outperform 1-sparse SAE probes on harmful intent detection. The blog post identifies feature absorption as a key culprit but provides no systematic quantification of the absorption--performance link. Our Component 3 provides this missing analysis: we compute Pearson and Spearman correlations between absorption scores and four SAEBench downstream metrics across 54 Gemma Scope SAEs, with Bonferroni correction and pre-specified effect-size thresholds.

**Gap.** No prior work directly tests whether absorption scores correlate with downstream SAE quality across a large set of SAE configurations. The assumed causal chain (less absorption $\to$ better downstream interpretability) is widely invoked but empirically unvalidated.

## 2.4 Ecological Competition Models in Machine Learning

The Lotka-Volterra competitive exclusion principle---that two species with identical niches cannot coexist at equilibrium---has been applied sparingly in ML. Park et al. ("The Geometry of Concepts," 2025) invoke an ecological niche analogy in one sentence when discussing concept interference in representation spaces, but do not formalize the analogy or derive quantitative predictions.

In population ecology, the competition coefficient $\alpha_{ij}$ quantifies how much species $j$'s resource consumption reduces the per-capita growth rate of species $i$. When $\alpha_{ij} > 1$, species $i$ is excluded. We adapt this framework to SAE features: $\sigma_{ij}$ (normalized co-activation rate) maps to niche overlap, $f_j / f_i$ (frequency ratio) maps to carrying capacity imbalance, and $\alpha_{ij} > 1$ predicts competitive exclusion (absorption) of the rarer feature $i$.

**Gap.** No prior work formalizes the ecological competition analogy for SAE absorption or derives a quantitative absorption predictor from it. Our competition coefficient $\alpha_{ij}$ is the first such formulation.

## 2.5 Feature Splitting, Superposition, and Related Phenomena

Feature absorption is distinct from two related phenomena. Elhage et al. (2022) introduced the *superposition* framework, showing that neural networks encode more features than they have dimensions by representing features as non-orthogonal directions. SAEs are designed to resolve superposition, but absorption occurs even in SAEs that successfully decompose the activation space---it is a failure of the SAE's sparsity optimization, not a failure to resolve superposition.

Feature *splitting* (Chanin et al., arXiv:2409.14507) describes a related but opposite effect: a single concept is split across multiple SAE latents as dictionary width increases. Splitting creates redundancy; absorption creates omission. Chanin et al. show that wider SAEs reduce splitting but may increase absorption, producing the "width paradox" that our H4 (distributed absorption score) addresses.

Feature *hedging* (Chanin, Dulka, and Garriga-Alonso, arXiv:2505.11756) is a complementary failure mode where narrow SAEs merge correlated features. Hedging occurs in capacity-limited regimes; absorption occurs in hierarchical feature regimes regardless of capacity. The two phenomena have opposite dependencies on SAE width: wider SAEs reduce hedging but may increase distributed absorption.

A unified theoretical framework (arXiv:2512.05534) casts all sparse dictionary learning methods as piecewise biconvex optimization and identifies stable partial minima where absorbed features are trapped. This framework proposes *feature anchoring* to restore identifiability, but validates the approach only on synthetic benchmarks. Our empirical study on real models (GPT-2 Small, Gemma Scope SAEs) provides complementary evidence by testing whether the competition-coefficient formulation captures the dynamics that produce these partial minima.

<!-- FIGURES
- None
-->
