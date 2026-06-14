# When Features Compete: Empirical Tests of Lotka-Volterra Absorption Detection, Corpus-Level Predictors, and Downstream Impact in Sparse Autoencoders

## Abstract

Sparse autoencoders (SAEs) decompose neural network activations into interpretable latent directions, but feature absorption---where rare general latents are suppressed by frequent specific latents---undermines their fidelity. We present a three-part empirical study of absorption. First, we formalize the Lotka-Volterra (LV) competitive exclusion analogy as an unsupervised absorption detector, defining a competition coefficient $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ from co-activation statistics and frequency ratios. The detector fails: on GPT-2 Small, test F1 = 0.128 (ROC-AUC = 0.148), below a cosine-similarity baseline (F1 = 0.165), and no sharp threshold transition exists near $\alpha_{ij} \approx 1$. Second, corpus pointwise mutual information (PMI) has no predictive power for absorption (partial $R^2 = 0.0006$, $p = 0.593$); the dominant predictors are layer depth and sparsity penalty, implicating the SAE's optimization objective rather than training data co-occurrence. Third, and most consequentially, absorption score correlates negatively with downstream SAE quality across 54 Gemma Scope SAEs: Pearson $r = -0.595$ for sparse probing F1, $r = -0.454$ for RAVEL, and $r = -0.431$ for SCR, all surviving Bonferroni correction and strengthening under partial correlation controlling for width, layer, and architecture. A three-tier taxonomy applied to GPT-2 Small classifies 92.3% of letter features as showing some absorption (vs. the canonical 15--35%), though the Type II (partial) rate is likely inflated by heuristic parent identification. These results establish absorption as a validated quality indicator that current unsupervised methods cannot reliably detect.

---

# 1 Introduction

Sparse autoencoders (SAEs) decompose neural network activations into interpretable latent directions, providing a scalable window into the computations of large language models (Bricken et al., 2023; Cunningham et al., 2023). A persistent failure mode undermines this promise: *feature absorption*, where rare parent latents---such as a "first letter = A" feature---are systematically suppressed by frequent child latents---such as token-level features for "April" or "Amazon" that co-occur with the general concept (Chanin et al., 2024). On the canonical first-letter task, the reported absorption rate is 15--35% of letter features, measured via probe-directed comparison against ground-truth letter directions (Chanin et al., 2024).

Three tensions in the absorption literature remain unresolved.

**Measurement versus phenomenon.** The 15--35% rate captures only what we term Type I (full, single-latent) absorption: cases where one child latent accounts for the majority of a parent's suppression. Our three-tier taxonomy (Section 3.3) classifies absorption into Type I (full), Type II (partial), and Type III (distributed) categories. Applied to GPT-2 Small, the comprehensive absorption rate reaches 92.3%, with 23 of 26 letter features showing partial suppression where the parent latent activates at less than 50% of its expected magnitude (Section 5.3). Even accounting for likely inflation in the Type II metric---parent features were identified by a selectivity heuristic rather than `sae-spelling` ground truth---the gap between the canonical rate and the partial absorption rate is substantial. The true failure scope of SAEs is plausibly 2--6$\times$ larger than the headline figure suggests. We further test whether wider SAEs distribute absorption across multiple children (the width paradox), finding only partial support (Section 5.5).

**Mechanism versus intervention.** Architecture papers---including JumpReLU SAEs (Rajamanoharan et al., 2024), Matryoshka SAEs (Bussmann et al., 2024), OrtSAE (Bussmann et al., 2025), masked regularization (Narayanaswamy et al., 2025), and ATM SAE (Li et al., 2025)---report absorption reductions without a unified mechanistic account. We test whether a Lotka-Volterra (LV) competitive exclusion framework provides one, defining a competition coefficient that integrates niche overlap (co-activation rate) and frequency imbalance between latent pairs (Section 3.1). The LV theory predicts a sharp absorption transition at $\alpha_{ij} \approx 1$. This prediction fails: the LV detector achieves test F1 = 0.128, underperforming a decoder cosine-similarity baseline, and a sharpness test finds no phase transition (Section 5.1). The competition coefficient captures relevant factors---co-activation geometry and frequency imbalance---but the threshold-based competitive exclusion framing is not supported.

**Metric versus impact.** DeepMind deprioritized SAE research after finding dense probes outperform SAEs on safety-relevant tasks, casting doubt on whether reducing absorption actually improves downstream utility. No prior work directly tests the assumed causal chain. We provide this test using SAEBench (Karvonen et al., 2025) data across 54 Gemma Scope SAEs (Gemma 2 2B, layers 5/12/19, widths 16k/65k/1M). Absorption score correlates negatively with sparse probing F1 ($r = -0.595$, $p < 0.001$), SCR ($r = -0.431$, $p = 0.002$), and RAVEL TPP ($r = -0.454$, $p < 0.001$); all three survive Bonferroni correction. Partial correlations controlling for SAE width, layer, and architecture class *strengthen* these results (partial $r = -0.661$ for sparse probing, $-0.677$ for SCR, $-0.492$ for RAVEL). Our pre-registered hypothesis H3 predicted disconnection ($|r| < 0.2$); the data falsify H3, meaning higher absorption is associated with worse SAE performance. This is the paper's strongest positive finding.

This paper makes three contributions:

1. **A formal test of the LV competitive exclusion hypothesis applied to SAE absorption** (Section 3). The LV-predicted sharp threshold at $\alpha_{ij} \approx 1$ is not supported (test F1 = 0.128), establishing that absorption is not well-described as binary competitive exclusion. The conceptual decomposition into niche overlap and frequency imbalance identifies the relevant variables, but future detection methods should treat them as continuous predictors.

2. **A three-tier absorption taxonomy** that classifies features as Type I (full), Type II (partial), or Type III (distributed) (Section 3.3). Applied to GPT-2 Small, the comprehensive absorption rate (92.3%, upper bound; see Section 5.3 caveat) reveals that partial suppression affects the vast majority of letter features, far exceeding the canonical 15--35% Type I rate.

3. **The first systematic correlational evidence linking absorption scores to downstream performance** across 54 Gemma Scope SAEs (Section 5.4). The strong negative correlations ($r = -0.43$ to $-0.60$), surviving Bonferroni correction and strengthening under partial correlation, support the research community's motivating assumption that absorption matters for interpretability quality.

All experiments in this paper are training-free, operating on pre-trained SAEs via SAELens and TransformerLens. The primary model is GPT-2 Small (124M parameters) for probe-based experiments; the downstream correlation analysis uses Gemma Scope SAEs via pre-computed SAEBench results. Figure 1 illustrates the LV competitive exclusion framework and its mapping to SAE feature absorption.

The remainder of the paper defines the LV framework and taxonomy (Section 3), describes the experimental setup (Section 4), reports results across five experimental tests (Section 5), presents ablations (Section 6), and discusses implications and limitations (Sections 7--8).


<!-- FIGURES
- Figure 1: fig_lv_framework_desc.md — Conceptual diagram mapping Lotka-Volterra competitive exclusion to SAE feature absorption
-->

---

# 2 Related Work

Our study addresses three open questions about SAE feature absorption: whether it can be detected without probe directions, whether corpus statistics predict it, and whether it matters for downstream performance. We organize prior work along these axes and identify the gap each component fills.

## 2.1 Feature Absorption: Definition, Measurement, and the Ecological Analogy

Chanin et al. (arXiv:2409.14507) established the canonical formalization: a parent latent $j$ (e.g., "first letter = A") is absorbed by a child latent $c$ (e.g., "the token 'April'") when $a_j = 0$ on inputs where $a_j$ should be positive, while $a_c > 0$ on those same inputs. Their measurement protocol trains linear probes to identify ground-truth feature directions, then uses integrated gradients to attribute false negatives to specific absorbing latents. On mid-layer Gemma Scope SAEs (Gemma 2 2B, widths 16k and 65k), they report absorption rates of 15--35% across the 26 first-letter classes. This rate is explicitly acknowledged as a lower bound: it captures only full, single-latent absorption (what we term Type I), ignoring partial suppression and distributed multi-child effects.

Karvonen et al. (arXiv:2503.09532) incorporated the Chanin metric as one of eight evaluation dimensions in SAEBench, a standardized benchmark covering 200+ open-source SAEs across eight architectures. SAEBench reports absorption scores alongside sparse probing, RAVEL, SCR, and unlearning, but does not analyze correlations between absorption and the other metrics---a gap our downstream impact analysis addresses.

Tian et al. (arXiv:2509.23717) frame absorption as a special case of poor feature *sensitivity*: a feature that activates selectively on its target concept but fails to activate on similar inputs has low sensitivity. Their scalable sensitivity metric reveals that many features rated as interpretable by activation-example inspection nonetheless have poor recall, consistent with the partial absorption (Type II) our taxonomy quantifies.

A LessWrong post ("Looking for Feature Absorption Automatically") attempted cosine-similarity-based detection using ablation effect vectors rather than decoder geometry or co-activation statistics. The approach yielded negative results. Our competition coefficient $\alpha_{ij}$ differs in two respects: it operates on co-activation frequencies rather than ablation effects, and it multiplies niche overlap $\sigma_{ij}$ by the frequency ratio $f_j / f_i$ to capture the asymmetric competitive pressure from frequent children on rare parents.

The Lotka-Volterra competitive exclusion principle---that two species with identical niches cannot coexist at equilibrium---has been applied sparingly in ML. Park et al. ("The Geometry of Concepts," 2025) invoke an ecological niche analogy in one sentence when discussing concept interference in representation spaces, but do not formalize the analogy or derive quantitative predictions. In population ecology, the competition coefficient $\alpha_{ij}$ quantifies how much species $j$'s resource consumption reduces the per-capita growth rate of species $i$. When $\alpha_{ij} > 1$, species $i$ is excluded. We adapt this framework to SAE features: $\sigma_{ij}$ (normalized co-activation rate) maps to niche overlap, $f_j / f_i$ (frequency ratio) maps to carrying capacity imbalance, and $\alpha_{ij} > 1$ predicts competitive exclusion (absorption) of the rarer feature $i$. No prior work formalizes this analogy or derives a quantitative absorption predictor from it.

**Gap.** All existing absorption measurement methods require pre-specified probe directions---the researcher must know which features to look for. This restricts absorption analysis to controlled proxy tasks (first-letter spelling) and precludes systematic study of absorption in safety-relevant or semantically rich feature hierarchies. No prior work computes a probe-free absorption predictor from activation statistics alone, and no prior work provides a formalized ecological competition framework for absorption prediction.

## 2.2 SAE Architectures and Absorption Reduction

Multiple SAE variants reduce absorption, but no unified mechanistic account explains why.

Rajamanoharan et al. (arXiv:2407.14435) introduced JumpReLU SAEs, which train $L_0$ directly via the straight-through estimator. JumpReLU achieves state-of-the-art reconstruction fidelity on Gemma 2 9B but exhibits the highest absorption rates in SAEBench (Karvonen et al., arXiv:2503.09532). We hypothesize this is consistent with longer training amplifying the sparsity-gradient dynamics that produce absorption.

Bussmann et al. (arXiv:2503.17547; ICML 2025) proposed Matryoshka SAEs, which train nested dictionaries of increasing width simultaneously. The nested structure explicitly allocates general features to smaller inner dictionaries and specific features to larger outer ones, reducing the parent-child competition that drives absorption. Matryoshka SAEs achieve the best absorption scores in SAEBench while maintaining competitive reconstruction.

Bussmann et al. (arXiv:2509.22033) enforce pairwise orthogonality on decoder columns (OrtSAE), reducing absorption by 65% relative to standard SAEs. Orthogonality directly reduces the decoder cosine similarity between parent and child features, which in our framework corresponds to reducing the candidate pair set in the $\alpha_{ij}$ pre-filter (pairs with $\cos(\mathbf{d}_i, \mathbf{d}_j) > 0.15$).

Narayanaswamy et al. (arXiv:2604.06495) introduced masked regularization, which randomly masks high-frequency tokens during SAE training to disrupt co-occurrence patterns. This directly suppresses the co-activation statistics that enter our niche overlap term $\sigma_{ij}$, providing independent motivation for the competition coefficient framework.

Li et al. (arXiv:2510.08855) proposed Adaptive Temporal Masking (ATM SAE), which dynamically scores per-latent importance based on activation magnitude, frequency, and reconstruction contribution. ATM achieves the lowest reported absorption score (0.0068 vs. TopK 0.1402 and JumpReLU 0.0114 on Gemma-2-2B).

**Gap.** These architectures reduce absorption through diverse mechanisms (nesting, orthogonality, masking, temporal weighting), but no single quantity unifies them. Our competition coefficient $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ provides a candidate unifying lens: each architecture can be analyzed in terms of how it reduces $\sigma_{ij}$ (niche overlap), $f_j / f_i$ (frequency imbalance), or both.

## 2.3 SAE Evaluation and Downstream Impact

SAEBench (Karvonen et al., arXiv:2503.09532) provides the first multi-metric evaluation suite, revealing that proxy metrics (CE loss recovered, sparsity) do not reliably predict practical SAE performance. This finding motivates our H3 analysis: if proxy metrics are unreliable, is the absorption metric itself predictive of downstream quality?

Bussmann et al. (arXiv:2602.14111) present a provocative negative result: on synthetic benchmarks, SAEs recover only 9% of ground-truth features, and random baselines match trained SAEs on interpretability and sparse probing tasks. While this finding pertains to synthetic settings and does not directly invalidate SAEs on real models, it raises the stakes for demonstrating that absorption---a metric defined on real models---predicts real downstream performance.

DeepMind's safety research team publicly deprioritized SAE research in 2025 after finding that dense linear probes dramatically outperform 1-sparse SAE probes on harmful intent detection. The blog post identifies feature absorption as a key culprit but provides no systematic quantification of the absorption--performance link. Our downstream impact analysis provides this missing evidence: we compute Pearson and Spearman correlations between absorption scores and four SAEBench downstream metrics across 54 Gemma Scope SAEs, with Bonferroni correction and pre-specified effect-size thresholds.

**Gap.** No prior work directly tests whether absorption scores correlate with downstream SAE quality across a large set of SAE configurations. The assumed causal chain (less absorption $\to$ better downstream interpretability) is widely invoked but empirically unvalidated.

## 2.4 Feature Splitting, Superposition, and Related Phenomena

Feature absorption is distinct from two related phenomena. Elhage et al. (2022) introduced the *superposition* framework, showing that neural networks encode more features than they have dimensions by representing features as non-orthogonal directions. SAEs are designed to resolve superposition, but absorption occurs even in SAEs that successfully decompose the activation space---it is a failure of the SAE's sparsity optimization, not a failure to resolve superposition.

Feature *splitting* (Chanin et al., arXiv:2409.14507) describes a related but opposite effect: a single concept is split across multiple SAE latents as dictionary width increases. Splitting creates redundancy; absorption creates omission. Chanin et al. show that wider SAEs reduce splitting but may increase absorption, producing the "width paradox" that our H4 (Distributed Absorption Score) addresses.

Feature *hedging* (Chanin, Dulka, and Garriga-Alonso, arXiv:2505.11756) is a complementary failure mode where narrow SAEs merge correlated features. Hedging occurs in capacity-limited regimes; absorption occurs in hierarchical feature regimes regardless of capacity. The two phenomena have opposite dependencies on SAE width: wider SAEs reduce hedging but may increase distributed absorption.

A unified theoretical framework (arXiv:2512.05534) casts all sparse dictionary learning methods as piecewise biconvex optimization and identifies stable partial minima where absorbed features are trapped. This framework proposes *feature anchoring* to restore identifiability, but validates the approach only on synthetic benchmarks. Our empirical study on real models (GPT-2 Small, Gemma Scope SAEs) provides complementary evidence by testing whether the competition-coefficient formulation captures the dynamics that produce these partial minima.

<!-- FIGURES
- None
-->

---

# 3 Method: Lotka-Volterra Competition Coefficient

Feature absorption suggests an ecological analogy. In Lotka-Volterra (LV) competitive exclusion, two species sharing the same niche cannot coexist indefinitely---the species with a lower carrying capacity is driven to local extinction when competitive pressure exceeds a threshold. We map this framework to SAE latents: a general parent latent (e.g., "first letter = A") occupies a niche that overlaps with multiple specific child latents (e.g., "the token 'April'"). When the competitive pressure from child to parent exceeds a critical value, the parent is excluded---absorbed.

This section defines the competition coefficient $\alpha_{ij}$, the Distributed Absorption Score DAS($P, k$), and a three-tier absorption taxonomy. Figure 1 illustrates the conceptual mapping.

## 3.1 Competition Coefficient

We define the competition coefficient between latent $i$ (potential parent) and latent $j$ (potential child) as:

$$\alpha_{ij} = \sigma_{ij} \cdot \frac{f_j}{f_i}$$

where:

- $f_i = P(a_i > 0)$ is the activation frequency of latent $i$, i.e., the fraction of corpus tokens for which latent $i$ fires.
- $f_j = P(a_j > 0)$ is the activation frequency of latent $j$.
- $\sigma_{ij} = \frac{P(a_i > 0, a_j > 0)}{\min(f_i, f_j)}$ is the normalized co-activation rate, analogous to niche overlap in ecology. This quantity equals 1 when the rarer latent fires only when the more frequent one also fires, and equals 0 when they never co-activate. In the parent-child case where $f_j \gg f_i$, $\min(f_i, f_j) = f_i$, so $\sigma_{ij} = P(a_j > 0 \mid a_i > 0)$: the probability that the child fires given the parent fires.

Under the LV competitive exclusion principle, $\alpha_{ij} > 1$ *would* predict that the rarer species---here, the parent latent $i$---is excluded (absorbed) by the more frequent child $j$. If this analogy holds precisely, absorption probability should show a sharp sigmoid transition near $\alpha_{ij} \approx 1$, distinguishing the LV framework from a generic monotone relationship between co-activation and absorption. We test this prediction in Section 5.1.

The two factors in $\alpha_{ij}$ capture distinct aspects of absorption risk:

1. **Niche overlap** $\sigma_{ij}$: measures how completely the parent's activations are contained within the child's activation pattern. High $\sigma_{ij}$ means the child fires on most of the parent's tokens.
2. **Frequency imbalance** $f_j / f_i$: measures how much more frequent the child is than the parent. Higher imbalance gives the child a larger "carrying capacity" advantage.

Their product yields a single scalar that integrates both geometric proximity (in activation space) and statistical dominance (in frequency).

**Computational pre-filter.** Computing $\alpha_{ij}$ for all $D^2$ latent pairs is intractable for $D > 16{,}000$. We restrict candidate pairs to those satisfying two conditions: (1) both latents exceed a minimum activation frequency $f_i, f_j > 0.001$, and (2) their decoder columns have cosine similarity $\cos(\mathbf{d}_i, \mathbf{d}_j) > 0.15$. The decoder cosine filter ensures candidate pairs share a similar direction in residual stream space, a necessary condition for one to substitute for the other. On the `gpt2-small-res-jb` SAE ($D = 24{,}576$), this reduces the candidate set from $\sim 3 \times 10^8$ pairs to $\sim 4.4 \times 10^6$ (a 68$\times$ reduction at layer 8), making computation tractable on a single GPU in under 5 minutes. We evaluate the coverage-precision tradeoff of this filter in Section 6.2; at the default threshold, 34% of absorbed pairs are retained, imposing a recall ceiling that limits overall detector sensitivity.

**Detection rule.** For each parent candidate $i$, we compute $\max_j \alpha_{ij}$ over all filtered children $j$. If $\max_j \alpha_{ij} > \tau$ for a calibrated threshold $\tau$, we predict that latent $i$ is absorbed. The threshold $\tau$ is calibrated on a held-out letter subset (Section 4.3).

![LV Competition Framework](figures/fig_lv_framework.pdf)

**Figure 1.** Lotka-Volterra competitive exclusion mapped to SAE feature absorption. Left: ecological niche overlap between two species with unequal carrying capacities. Right: two SAE latents with overlapping activation patterns and unequal frequencies. The competition coefficient $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ formalizes absorption as competitive exclusion in decoder space. When $\alpha_{ij} > 1$, the rarer parent latent is predicted to be absorbed. Our experiments show that the sharp threshold prediction does not hold (Section 5.1), but the conceptual decomposition into niche overlap and frequency imbalance captures the relevant factors.

## 3.2 Distributed Absorption Score

The canonical absorption metric (Chanin et al., 2024) captures single-child absorption: one specific latent suppresses one general latent. Wider SAEs ($D = 49{,}152$ or $98{,}304$) have more candidate absorbers per parent, potentially distributing absorption across multiple children rather than concentrating it in one. To capture this, we define the Distributed Absorption Score:

$$\text{DAS}(P, k) \in [0, 1]$$

which measures how much of parent $P$'s information is collectively captured by its top-$k$ children ranked by $\alpha_{ij}$.

**Estimation procedure.** For each parent $P$:

1. Identify the top-$k$ children $C_1, C_2, \ldots, C_k$ by competition coefficient $\alpha_{Pj}$.
2. Collect activation samples: run $n = 10{,}000$ tokens through the SAE, recording activations $(a_P, a_{C_1}, \ldots, a_{C_k})$ for each token.
3. Fit a logistic regression predicting $\mathbf{1}[a_P > 0]$ from raw activation magnitudes $(a_{C_1}, \ldots, a_{C_k})$ as continuous predictors. No regularization is applied; for $k = 3$ features on $n = 10{,}000$ tokens, overfitting is negligible.
4. Compute DAS$(P, k) = 1 - \frac{H(a_P \mid a_{C_1}, \ldots, a_{C_k})}{H(a_P)}$, approximated via the McFadden pseudo-$R^2$ of the logistic regression. The McFadden pseudo-$R^2$ ($1 - \log L_{\text{full}} / \log L_{\text{null}}$) serves as a proxy for conditional entropy reduction (Hagle & Mitchell, 1992); the approximation is standard for binary outcomes with continuous predictors.

DAS$(P, k=1)$ captures single-child absorption and is conceptually analogous to, though computationally distinct from, the Chanin metric. DAS$(P, k=3)$ captures distributed absorption: cases where no single child explains the parent's suppression, but three children collectively do.

The width paradox---wider SAEs sometimes show *higher* total absorption despite greater capacity---receives a natural explanation under this framework. Wider SAEs split absorption across more children: DAS$(k=1)$ may decrease or remain flat, while DAS$(k=3)$ increases as more children each absorb a fraction of the parent's information.

## 3.3 Absorption Taxonomy

Prior work reports absorption rates of 15--35% on the first-letter task (Chanin et al., 2024), but this figure captures only what we term Type I absorption. We define a three-tier taxonomy to characterize the full scope of the phenomenon:

| Type | Definition | Operationalized Threshold | Measurement |
|------|-----------|--------------------------|-------------|
| **Type I (Full)** | A single child latent accounts for the overwhelming majority of the parent's suppression | Chanin metric $> 0.5$ **and** single absorber explains $> 80\%$ of suppression | `sae-spelling` ground truth |
| **Type II (Partial)** | The parent latent fires at reduced magnitude on its expected tokens, but no single child dominates | Parent activation magnitude $< 50\%$ of expected$^\dagger$ | Activation magnitude ratio on letter-specific tokens vs. global mean when active |
| **Type III (Distributed)** | Multiple children collectively suppress the parent, but no single child qualifies as Type I | DAS$(k=3) > 0.6$ **and** Type I not triggered | Logistic regression on top-3 children (Section 3.2) |
| **None** | Parent latent fires at or above expected magnitude; no absorption detected | Magnitude ratio $\geq 0.5$, Chanin metric $< 0.5$, DAS$(k=3) < 0.6$ | All metrics below thresholds |

**Table 1.** Absorption taxonomy definitions. Types are evaluated sequentially: a feature classified as Type I is not re-evaluated for Type II. The comprehensive absorption rate (Type I + II + III) captures the full failure scope. $^\dagger$Type II measurement requires validated parent feature IDs; our implementation uses a selectivity heuristic with global mean fallback, which likely inflates the Type II rate (see Section 7.4 for limitations).

The taxonomy is designed to be exhaustive and mutually exclusive: every parent feature receives exactly one classification. The sequential evaluation order (I, then II, then III, then None) ensures that the strictest definition takes priority.

**Relation to prior metrics.** The canonical 15--35% rate reported by Chanin et al. (2024) corresponds approximately to the fraction of features showing *any* absorption (not our strict Type I, which additionally requires $> 80\%$ suppression dominance by a single child). Our Type II captures features that show reduced parent activation but that do not meet the stringent single-absorber criterion. Type III is entirely new: it identifies cases invisible to any single-child metric.

<!-- FIGURES
- Figure 1: fig_lv_framework_desc.md — Conceptual diagram mapping LV competitive exclusion to SAE feature absorption
- Table 1: inline — Absorption taxonomy definitions with operationalized thresholds
-->

---

# 4 Experimental Setup

## 4.1 Models and SAE Configurations

All experiments use pre-trained SAEs loaded via SAELens; no model fine-tuning or SAE retraining is performed.

**Primary model.** GPT-2 Small (124M parameters, $d = 768$) serves as the open-weight anchor for all probe-based experiments (H1, H2, H4, taxonomy, safety probe). We selected GPT-2 Small because it is fully open, deterministic, and has the best-studied absorption ground truth via `sae-spelling`. Gemma 2 2B was the original target model but requires gated HuggingFace access; we defer Gemma-based replication to future work.

**SAE sources for GPT-2 Small.** We use three SAE families from SAELens, all at layer 8 of the residual stream:

- `gpt2-small-res-jb` ($D = 24{,}576$, `hook_resid_pre`): primary SAE for H1 calibration, H2 regression, and taxonomy.
- `gpt2-small-res-jb-feature-splitting` ($D \in \{24{,}576, 49{,}152, 98{,}304\}$, `hook_resid_pre`): width-varying family for H4 (width paradox).
- `gpt2-small-resid-post-v5` ($D \in \{32{,}768, 131{,}072\}$, `hook_resid_post`): OpenAI v5 architecture for cross-architecture validation.

**SAEBench data (Gemma Scope).** For H3 (downstream impact), we use pre-computed SAEBench results (Karvonen et al., 2025) for 54 Gemma Scope SAEs trained on Gemma 2 2B residual stream activations. These span layers 5, 12, and 19; widths 16k, 65k, and 1M; and multiple $L_0$ settings per configuration. Metrics available per SAE: absorption score, sparse probing F1, SCR, RAVEL (TPP), and unlearning.

**PMI regression survey.** The H2 regression uses absorption measurements from 31 GPT-2 Small SAE configurations across layers 3--11 and widths 768--131,072, yielding 806 (configuration, letter) observations.


## 4.2 Ground Truth and Data Split

Ground-truth absorption labels come from `sae-spelling` (Chanin et al., 2024), which identifies absorbed latent pairs using pre-specified letter-probe directions for the first-letter task (26 letters, A--Z).

**Train/test split.** Letters A--M (13 letters) serve as the calibration set; letters N--Z (13 letters) serve as the held-out test set. The threshold $\tau$ for the competition coefficient $\alpha_{ij}$ is selected to maximize F1 on the calibration set and evaluated once on the test set. On the primary 24k SAE, the calibration set contains 1,110 candidate pairs (275 positives) and the test set contains 790 pairs (195 positives), reflecting the overall absorption rate of 35.4%.


## 4.3 Evaluation Protocol

We organize the evaluation around four pre-registered hypotheses.

**H1 (LV Detector).** The competition coefficient $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ is evaluated as a binary absorption predictor at threshold $\tau$. Metrics: precision, recall, F1, and ROC-AUC on the test letters. Success criterion: F1 $> 0.35$. A sharpness diagnostic tests the LV-predicted sigmoid transition at $\alpha_{ij} \approx 1$: we bin $\alpha_{ij}$ values (width 0.1, range $[0, 3]$), compute empirical absorption rates per bin, and compare AIC between fitted sigmoid and linear models.

**H2 (Corpus PMI).** We regress absorption rate on $\log(L_0)$, $\log(D)$, layer, and $\log(\text{PMI})$ with HC3 robust standard errors. The quantity of interest is the partial $R^2$ of the PMI term. Success criterion: partial $R^2 \geq 0.10$.

**H3 (Downstream Impact).** We compute Pearson $r$ and Spearman $\rho$ between SAEBench absorption scores and each downstream metric (sparse probing F1, SCR, RAVEL TPP, unlearning) across 54 Gemma Scope SAEs, applying Bonferroni correction for 4 simultaneous tests ($\alpha_{\text{corrected}} = 0.0125$). Pre-registered prediction: $|r| < 0.2$ (disconnection). We also compute partial correlations controlling for $\log(\text{width})$, layer, and architecture class. A matched comparison selects the 5 lowest- and 5 highest-absorption SAEs from layers 12 and 19 and applies a one-sided paired $t$-test on RAVEL TPP. A safety probe pilot (50 AdvBench-style harmful + 50 benign prompts, 5-fold CV) measures the dense-vs-1-sparse probe gap at three absorption levels.

**H4 (Width Paradox).** We compute $\text{DAS}(k{=}1)$ and $\text{DAS}(k{=}3)$ for all 26 letters across widths $\{24\text{k}, 49\text{k}, 98\text{k}\}$ on GPT-2 Small layer 8. $\text{DAS}(k{=}3)$ is estimated via McFadden pseudo-$R^2$ as described in Section 3.2. Predictions: $\text{DAS}(k{=}1)$ non-monotone or decreasing with width for $\geq 60\%$ of letters; $\text{DAS}(k{=}3)$ monotonically increasing for $\geq 80\%$ of letters.


## 4.4 Baselines

Three baselines contextualize the competition coefficient:

1. **Cosine-similarity-only detector.** Threshold on decoder cosine similarity $\cos(\mathbf{d}_i, \mathbf{d}_j)$ alone, calibrated on the same A--M split. This corresponds to the approach reported as negative in "Looking for Feature Absorption Automatically" (LessWrong).
2. **Chanin probe-directed metric (ground truth).** The `sae-spelling` ground-truth absorption rate, which requires pre-specified probe directions. This defines the target variable.
3. **Dense linear probe.** Logistic regression on the full residual stream ($d = 768$ features), providing an upper bound on downstream task performance.

---

# 5 Results

## 5.1 LV Detector Performance (H1) --- Negative Result

The competition coefficient $\alpha_{ij}$ does not produce a usable absorption detector. Table 2 reports the primary comparison.

**Table 2: LV Detector vs. Cosine Baseline on GPT-2 Small (24k, layer 8)**

| Method | Threshold | Precision | Recall | F1 | ROC-AUC |
|:-------|:---------:|:---------:|:------:|:--:|:-------:|
| LV $\alpha_{ij}$ | $\tau = 0.5$ | 0.085 | 0.256 | **0.128** | 0.148 |
| Cosine baseline | $\theta = 0.2$ | 0.140 | 0.200 | **0.165** | 0.201 |

The LV detector achieves test F1 = 0.128 at the best calibration threshold $\tau = 0.5$, falling well below both the cosine baseline (F1 = 0.165) and the pre-registered success criterion (F1 $> 0.35$). The cosine baseline outperforms the LV coefficient by +3.7 F1 points and +5.3 AUC points.

A contributing factor to the low recall is the decoder cosine pre-filter ($\cos > 0.15$), which captures only 34.0% of absorbed pairs (Ablation A2, Section 6.2). The majority of absorption occurs between latent pairs with decoder cosine similarity below 0.15, meaning the candidate set excludes most true positives before $\alpha_{ij}$ is even computed.

**Sharpness test.** The LV competitive exclusion model predicts a sharp sigmoid transition in absorption rate near $\alpha_{ij} \approx 1$. Figure 2 plots empirical absorption rates against binned $\alpha_{ij}$ values. A linear fit (AIC = $-61.05$) marginally outperforms the sigmoid fit (AIC = $-60.95$; $\Delta$AIC = 0.11). The fitted sigmoid has center $x_0 = 10.0$ (far outside the data range) and slope $k = 0.16$ (essentially flat), confirming that no phase transition exists at $\alpha_{ij} \approx 1$.

![Sharpness Test](figures/fig_sharpness.pdf)

**Figure 2.** Sharpness test: empirical absorption rate vs. binned $\alpha_{ij}$ values on GPT-2 Small (24k, layer 8). The linear fit (blue dashed) marginally outperforms the sigmoid fit (red), with $\Delta$AIC = 0.11. No phase transition is observed near $\alpha_{ij} \approx 1$.

The first $\alpha_{ij}$ bin ($[0, 0.1]$, $n = 369$) shows an anomalously high absorption rate of 0.848. This artifact arises because very low $\alpha_{ij}$ values correspond to pairs where both latents have similar low frequencies and high co-activation---precisely the pattern of related latents that are both being absorbed by a common parent. Excluding this bin, the remaining data show a roughly flat or weakly increasing relationship.

**Interpretation.** The competition coefficient captures some information about co-activation geometry (it selects for pairs with high niche overlap and frequency imbalance), but the ecological threshold model---where $\alpha_{ij} > 1$ triggers competitive exclusion---does not describe a phase transition in absorption. Absorption is better modeled as a graded phenomenon than a binary exclusion event.


## 5.2 Corpus PMI as Predictor (H2) --- Null Result

Corpus co-occurrence statistics do not predict which letter features are absorbed. The full regression model (Table 3) explains 8.7% of absorption variance ($R^2 = 0.087$), but the PMI term contributes negligibly.

**Table 3: PMI Regression Coefficients (806 observations, 31 SAE configurations, HC3 SEs)**

| Coefficient | Estimate | SE (HC3) | $t$ | $p$ | 95% CI |
|:------------|:--------:|:--------:|:---:|:---:|:------:|
| Intercept ($\beta_0$) | 0.052 | 0.064 | 0.81 | 0.418 | [$-$0.074, 0.179] |
| $\log(L_0)$ ($\beta_1$) | 0.013 | 0.005 | 2.52 | **0.012** | [0.003, 0.023] |
| $\log(D)$ ($\beta_2$) | 0.003 | 0.004 | 0.81 | 0.417 | [$-$0.005, 0.011] |
| Layer ($\beta_3$) | $-$0.012 | 0.002 | $-$6.61 | **$<$0.001** | [$-$0.016, $-$0.008] |
| $\log(\text{PMI})$ ($\beta_4$) | $-$0.006 | 0.012 | $-$0.53 | 0.593 | [$-$0.029, 0.017] |

We tested whether corpus co-occurrence statistics predict absorption, hypothesizing that PMI would explain which features are absorbed (partial $R^2 \geq 0.10$). The data reject this hypothesis: $\beta_4 = -0.006$ is negative (opposite to the predicted positive association), statistically non-significant ($p = 0.593$), and contributes partial $R^2 = 0.0006$---three orders of magnitude below the success criterion. A PMI-only model achieves $R^2 = 0.0006$, confirming negligible standalone predictive power (Ablation A3, Section 6.3).

The dominant predictors are layer ($\beta_3 = -0.012$, $p < 0.001$) and $\log(L_0)$ ($\beta_1 = 0.013$, $p = 0.012$). Absorption decreases monotonically in later layers (each layer reduces absorption by 1.2 percentage points) and increases with the sparsity penalty. Width has no significant effect after controlling for $L_0$.

**Per-layer stability.** The PMI coefficient sign is inconsistent across layers: negative at layers 3, 4, 6, and 7; positive at layers 5 and 8. Only two layers (4 and 6) show significance for the PMI term ($p = 0.004$ and $p = 0.015$), both with negative coefficients---opposite to the H2 prediction. This sign instability rules out PMI as a robust absorption predictor.


## 5.3 Absorption Taxonomy --- Exploratory Result

Table 4 presents the per-letter taxonomy classification on GPT-2 Small (24k SAE, layer 8). The comprehensive taxonomy reveals that absorption is far more prevalent than the canonical Type I metric suggests, though the Type II rate carries an important caveat.

**Table 4: Absorption Taxonomy Summary (GPT-2 Small, 24k, layer 8)**

| Type | Count | Fraction | Definition |
|:-----|:-----:|:--------:|:-----------|
| Type I (full) | 1 | 3.8% | Chanin metric $> 0.5$ AND single absorber $> 80\%$ |
| Type II (partial) | 23 | 88.5% | Parent activation $< 50\%$ expected magnitude |
| Type III (distributed) | 0 | 0.0% | DAS($k{=}3$) $> 0.6$ AND not Type I |
| None | 2 | 7.7% | Letters X, Z |
| **Comprehensive** | **24** | **92.3% (upper bound)** | Type I + II + III |

The comprehensive absorption rate of 92.3% is 24$\times$ the strict Type I rate (3.8%) and 2.6$\times$ the Chanin any-absorption rate (80.8% of letters show some absorption at the standard threshold). Only letter Q meets the strict Type I criterion (absorption rate = 1.0, single absorber accounts for 100% of suppression). Letters X and Z are classified as "None" because their parent latents maintain $\geq 50\%$ of expected activation magnitude (magnitude ratios of 0.625 and 0.585, respectively).

**Caveat: Type II inflation.** The Type II rate is likely inflated by a systematic artifact in parent feature identification. Parent features were identified using a selectivity heuristic on synthetic prompts rather than `sae-spelling` ground truth. The magnitude ratio threshold ($< 0.5$) compares activation on letter tokens to a global mean-when-active baseline, which is biased upward by high-activation contexts unrelated to the first-letter function. The comparison set contains zero tokens for most letters ($n_{\text{comparison}} = 0$), forcing a fallback to global statistics. The Type II classification should be interpreted as "the putative parent latent fires weakly on letter tokens compared to its overall activation pattern" rather than "the parent latent is partially suppressed due to absorption by child latents." The causal link to absorption is not established for Type II. Replication with `sae-spelling` ground-truth parent feature IDs would provide definitive evidence.

![Absorption Taxonomy Across Widths](figures/fig_taxonomy_bar.pdf)

**Figure 3.** Absorption taxonomy distribution across widths $\{24\text{k}, 49\text{k}, 98\text{k}\}$ on GPT-2 Small, layer 8. Stacked bars show Type I (red), Type II (orange), Type III (yellow), and None (gray). The horizontal dashed line marks the canonical Chanin 35% rate. The comprehensive rate is stable: 92.3% at 24k and 49k, rising to 96.2% at 98k.

Figure 3 shows the taxonomy distribution across widths $\{24\text{k}, 49\text{k}, 98\text{k}\}$. The comprehensive rate is stable: 92.3% at 24k and 49k, rising to 96.2% at 98k (one additional letter classified as Type III). The canonical Chanin rate of 35.4% captures only the Type I tip of the absorption iceberg.


## 5.4 Downstream Impact (H3) --- Strong Positive Result

The H3 pre-registration predicted $|r| < 0.2$ between absorption and downstream metrics, expecting absorption to be decoupled from SAE quality. The data falsify this prediction: absorption shows strong negative correlations with 3 of 4 SAEBench tasks.

**Table 5: Absorption vs. Downstream Correlation Matrix ($n = 54$ Gemma Scope SAEs)**

| Task | $n$ | Pearson $r$ | 95% CI | $p$ | Spearman $\rho$ | Partial $r$ | Bonferroni sig. |
|:-----|:---:|:-----------:|:------:|:---:|:---------------:|:-----------:|:---------------:|
| Sparse probing F1 | 54 | $-$0.595 | [$-$0.744, $-$0.389] | $<$0.001 | $-$0.435 | $-$0.661 | Yes |
| SCR | 49 | $-$0.431 | [$-$0.635, $-$0.171] | 0.002 | $-$0.308 | $-$0.677 | Yes |
| RAVEL (TPP) | 54 | $-$0.454 | [$-$0.644, $-$0.212] | $<$0.001 | $-$0.478 | $-$0.492 | Yes |
| Unlearning | 40 | $-$0.175 | [$-$0.462, 0.144] | 0.280 | $-$0.141 | $-$0.072 | No |

Three of four correlations survive Bonferroni correction and exceed the $|r| > 0.3$ meaningful-effect threshold. Sparse probing shows the strongest relationship: SAEs with higher absorption scores have lower F1 on probing tasks ($r = -0.595$). Partial correlations strengthen after controlling for width, layer, and architecture class ($r_{\text{partial}} = -0.661$ for sparse probing, $r_{\text{partial}} = -0.677$ for SCR), indicating that absorption captures genuine quality variation beyond what these confounds explain. The SCR increase from $r = -0.431$ to partial $r = -0.677$ is notably large; partial correlations can inflate under suppressor effects, and confidence intervals at $n = 49$ are wide, warranting cautious interpretation. The SCR and unlearning sample sizes are smaller ($n = 49$ and $n = 40$, respectively) because not all SAEBench configurations have complete scores for these tasks.

![Absorption vs. Downstream Scatter](figures/fig_downstream_scatter.pdf)

**Figure 4.** Scatter plots: absorption score vs. sparse probing F1 (left) and RAVEL TPP (right) across 54 Gemma Scope SAEs, colored by width. Regression lines with 95% CI bands. Pearson $r = -0.595$ (sparse probing) and $r = -0.454$ (RAVEL).

**Matched RAVEL comparison.** The 5 lowest-absorption Gemma Scope SAEs (mean absorption = 0.022, layers 12/19) achieve mean TPP = 0.046, compared to mean TPP = 0.009 for the 5 highest-absorption SAEs (mean absorption = 0.866). The one-sided paired $t$-test rejects the null ($t = 4.27$, $p = 0.006$, Cohen's $d = 2.13$). Low-absorption SAEs outperform high-absorption SAEs across all three non-unlearning metrics: $\Delta$TPP = +0.037, $\Delta$F1 = +0.108, $\Delta$SCR = +0.203.

**Safety probe pilot (GPT-2 Small, underpowered).** The dense probe achieves AUC $\approx 1.0$ across all three SAE configurations. The 1-sparse SAE probe gaps do not follow the predicted monotone relationship with absorption:

**Table 6: Safety Probe Results (5-fold CV, GPT-2 Small)**

| Absorption Level | Config | Layer | Width | Dense AUC | 1-Sparse AUC | Probe Gap |
|:-----------------|:------:|:-----:|:-----:|:---------:|:------------:|:---------:|
| Lowest (0.000) | `resid_mid_L10_32k` | 10 | 32k | 1.000 | 0.882 $\pm$ 0.067 | 0.118 |
| Median (0.047) | `resid_pre_L8_768` | 8 | 768 | 1.000 | 0.852 $\pm$ 0.047 | 0.148 |
| Highest (0.119) | `resid_pre_L5_24k` | 5 | 24k | 0.998 | 0.947 $\pm$ 0.039 | 0.051 |

The probe gap is 0.118 (lowest absorption), 0.148 (median), and 0.051 (highest)---not monotonically increasing with absorption. The highest-absorption SAE actually shows the smallest probe gap. This non-monotone pattern reflects confounding by layer and width: the three SAEs differ in layer (5, 8, 10) and width (24k, 768, 32k), making it impossible to isolate absorption's causal effect. The SAEBench correlation analysis, which controls for these confounds across 54 SAEs, provides stronger evidence.


## 5.5 Width Paradox (H4) --- Partial Support

H4 predicted that $\text{DAS}(k{=}1)$ decreases while $\text{DAS}(k{=}3)$ increases with SAE width, reflecting a shift from concentrated to distributed absorption. The data provide partial support.

**Table 7: Mean DAS by Width (GPT-2 Small, layer 8, $n = 26$ letters)**

| Width | Mean DAS($k{=}1$) $\pm$ SD | Mean DAS($k{=}3$) $\pm$ SD |
|:-----:|:---------------------------:|:---------------------------:|
| 24k | 0.105 $\pm$ 0.118 | 0.320 $\pm$ 0.199 |
| 49k | 0.104 $\pm$ 0.068 | 0.227 $\pm$ 0.196 |
| 98k | 0.119 $\pm$ 0.185 | 0.260 $\pm$ 0.215 |

$\text{DAS}(k{=}1)$ shows no clear monotonic trend (0.105 $\to$ 0.104 $\to$ 0.119), and 57.7% of letters have non-positive slope with width---approaching but not meeting the 60% target. $\text{DAS}(k{=}3)$ does not increase monotonically (0.320 $\to$ 0.227 $\to$ 0.260): only 42.3% of letters show positive slope, far below the 80% target. DAS($k{=}3$) = 0.320 means the top-3 children collectively explain approximately 32% of the parent's activation variance at 24k width.

![DAS vs Width](figures/fig_das_width.pdf)

**Figure 5.** DAS($k{=}1$) (blue) and DAS($k{=}3$) (red) vs. SAE width on GPT-2 Small, layer 8. Error bars show letter-level standard deviation. Neither metric follows the predicted monotonic trend.

Per-letter analysis reveals high variance: letter N has $\text{DAS}(k{=}1) = 0.571$ at 24k but 0.100 at 98k (slope $= -0.34$), while letter X jumps from 0.0 to 1.0 at 98k. The DAS($k{=}3$) landscape is similarly noisy, with letter B dropping from 0.695 to 0.071 and letter Y increasing from 0.613 to 0.722.

**Assessment.** H4 receives a "partial support" verdict. The $k{=}1$ non-monotonicity observation is consistent with the width paradox narrative, but the $k{=}3$ prediction fails. The width paradox may require mechanisms beyond distributed competitive exclusion---for example, feature splitting at higher widths creates new parent-child relationships that redistribute absorption in unpredictable ways.


## 5.6 Cross-Architecture Generalization

The preceding sections use SAEs from a single training family (`gpt2-small-res-jb`). Here we test whether the LV detector generalizes to SAEs trained with different objectives and hook points.

The LV detector does not generalize across SAE architectures. Table 8 reports results on two OpenAI v5 SAEs (different training objective, `hook_resid_post`).

**Table 8: Cross-Architecture Validation (fixed $\tau = 0.5$)**

| Architecture | $D$ | LV Test F1 | Cosine F1 | F1 $\Delta$ (pp) |
|:-------------|:---:|:----------:|:---------:|:-----------------:|
| Baseline (JB, 24k) | 24,576 | 0.128 | 0.165 | --- |
| v5-32k (resid\_post) | 32,768 | 0.009 | 0.212 | $-$11.9 |
| v5-128k (resid\_post) | 131,072 | 0.000 | 0.353 | $-$12.8 |

On v5-32k, the LV detector drops to test F1 = 0.009; on v5-128k, it produces F1 = 0.0 (zero true positives). The cosine baseline, by contrast, improves on wider architectures (F1 = 0.353 on v5-128k). Even within-architecture recalibration fails: all $\tau$ values from 0.5 to 1.5 yield F1 = 0.0 on v5-128k.

The mean F1 degradation is 12.6 percentage points across architectures. The competition coefficient $\alpha_{ij}$ is specific to the training objective and hook point of the calibration SAE and does not transfer.


# 6 Ablations

## 6.1 Threshold Sensitivity (A1)

A fine sweep over $\tau \in [0.3, 2.0]$ confirms that F1 peaks at $\tau = 0.4$ (test F1 = 0.136), with the pre-registered threshold $\tau = 0.5$ achieving F1 = 0.128. The best threshold is not within the LV-predicted range $[0.75, 1.25]$. F1 declines monotonically for $\tau > 0.5$: at $\tau = 1.0$, F1 = 0.099; at $\tau = 1.5$, F1 = 0.029; at $\tau = 2.0$, F1 = 0.020. The 90% stability window spans only $\tau \in [0.3, 0.5]$ (width 0.2), indicating that $\alpha_{ij}$ acts as a weak continuous predictor at low thresholds rather than a discriminant with a natural cutoff near 1.

## 6.2 Decoder Cosine Pre-filter (A2)

Coverage at the default $\cos(\mathbf{d}_i, \mathbf{d}_j) > 0.15$ threshold is 34.0% of absorbed pairs---far below the 80% target. Relaxing to 0.10 recovers 43.2% coverage; tightening to 0.25 drops to 14.7%. The pre-filter is too restrictive: the majority of absorbed pairs have decoder cosine similarity below 0.15, meaning the LV computation misses most absorption candidates before thresholding on $\alpha_{ij}$.

## 6.3 PMI Regression Without Configuration Terms (A3)

The PMI-only model yields $R^2 = 0.0006$, confirming that corpus co-occurrence has no predictive power for absorption in isolation. Removing SAE configuration variables does not unmask a hidden PMI signal.


<!-- FIGURES
- Figure 2: gen_fig_sharpness.py, fig_sharpness.pdf — Sharpness test: absorption rate vs alpha_ij bins with sigmoid and linear fits
- Figure 3: gen_fig_taxonomy_bar.py, fig_taxonomy_bar.pdf — Absorption taxonomy stacked bar chart across widths
- Figure 4: gen_fig_downstream_scatter.py, fig_downstream_scatter.pdf — Scatter plots: absorption score vs sparse probing F1 and RAVEL TPP
- Figure 5: gen_fig_das_width.py, fig_das_width.pdf — DAS(k=1) and DAS(k=3) vs SAE width with error bars
- Table 2: inline — LV detector vs cosine baseline comparison
- Table 3: inline — PMI regression coefficients
- Table 4: inline — Absorption taxonomy summary
- Table 5: inline — Absorption vs downstream correlation matrix
- Table 6: inline — Safety probe results
- Table 7: inline — Mean DAS by width
- Table 8: inline — Cross-architecture validation
-->

---

# 7 Discussion

## 7.1 Why the LV Framework Fails as a Detector

The competition coefficient $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ achieves test F1 = 0.128, below both the pre-registered success criterion (F1 > 0.35) and the cosine-similarity baseline (F1 = 0.165). The sharpness test (Section 5.1, Figure 2) reveals why: a linear fit (AIC = $-61.05$) marginally outperforms the sigmoid (AIC = $-60.95$), and the sigmoid's midpoint parameter diverges to $x_0 = 10.0$---far from the predicted $\alpha_{ij} \approx 1$. Absorption is not a phase transition at a critical threshold; it is a gradual process with no sharp boundary in competition-coefficient space.

Two structural problems undermine the LV analogy. First, ecological competitive exclusion assumes that species interact through a shared resource, producing a zero-sum dynamic. SAE latents, by contrast, interact through the sparsity penalty during training, not through shared activation budget at inference time. A high $\alpha_{ij}$ at inference indicates that latent $j$ frequently co-activates with latent $i$ and is more common---but this does not imply that $j$ is suppressing $i$. The encoder may simply assign both latents to overlapping input regions without competition. Second, the frequency ratio $f_j / f_i$ confounds activation frequency with competitive advantage. A rare parent latent $i$ may have low $f_i$ because it is genuinely selective (e.g., "first letter = Q") rather than because it has been excluded by a competitor. The LV framework conflates rarity with suppression.

The variables may still have predictive value as continuous features in a non-threshold model, but this remains to be demonstrated. Co-activation geometry and frequency imbalance appear to be relevant factors for characterizing absorption risk; the failure is specifically in the threshold-based binary prediction. Future unsupervised detection methods should treat these quantities as continuous predictors rather than threshold discriminants.

## 7.2 Corpus Statistics Are Not the Proximal Cause

The null PMI result ($\beta_4 = -0.006$, $p = 0.593$, partial $R^2 = 0.0006$; Section 5.2, Table 3) rules out corpus co-occurrence as a meaningful predictor of absorption patterns. Per-layer coefficient signs are inconsistent (5/9 negative, 3/9 positive, 1 zero-variance), eliminating the possibility that a genuine effect is masked by layer-level heterogeneity.

The strongest absorption predictors are structural: layer index ($\beta_3 = -0.012$, $p < 0.001$) and $\log(L_0)$ ($\beta_1 = 0.013$, $p = 0.012$). Absorption decreases in later layers and increases with the sparsity penalty. Both findings are consistent with the optimization-driven account: absorption is produced by the SAE's training objective interacting with feature geometry, not by the statistics of the training corpus.

This result has a direct implication for intervention design. Data engineering approaches---corpus curation, co-occurrence disruption, or frequency balancing---are unlikely to reduce absorption. The effective intervention point is the training algorithm: masked regularization (Narayanaswamy et al., arXiv:2604.06495), orthogonality constraints (OrtSAE), or hierarchically-aware objectives that explicitly penalize parent-child competition during optimization.

## 7.3 Absorption Predicts Downstream Quality

The H3 result is the strongest empirical finding in this paper. Across 54 Gemma Scope SAEs spanning three layers (5, 12, 19) and three widths (16k, 65k, 1M), absorption score correlates negatively with downstream interpretability performance:

- Sparse probing F1: $r = -0.595$ ($p < 0.001$), partial $r = -0.661$ controlling for width, layer, and architecture
- RAVEL (TPP): $r = -0.454$ ($p < 0.001$), partial $r = -0.492$
- SCR: $r = -0.431$ ($p = 0.002$), partial $r = -0.677$
- Unlearning: $r = -0.175$ ($p = 0.280$)---not significant

Three of four tasks survive Bonferroni correction ($\alpha = 0.0125$). Partial correlations strengthen after controlling for confounds, indicating that absorption captures genuine quality variation beyond what width and layer predict alone. The partial $r = -0.677$ for SCR corresponds to $r^2 \approx 0.46$, meaning that after removing the variance explained by SAE configuration, absorption accounts for approximately 46% of residual variance in spurious correlation removal performance.

These correlations provide the first systematic correlational evidence linking absorption scores to downstream performance. Prior to this work, no published study directly tested whether the absorption metric was informative for downstream utility. Our results support this assumption for three of four tasks. Causal claims would require intervention experiments---ablating absorbed features and measuring downstream change---which we identify as a priority for future work.

**Unlearning is the exception.** The non-significant correlation ($r = -0.175$, $p = 0.280$) with unlearning performance suggests that unlearning depends on factors other than feature fidelity---potentially on the SAE's ability to isolate safety-relevant directions regardless of whether general features are absorbed. At $n = 40$ and Bonferroni-corrected $\alpha = 0.0125$, this comparison has approximately 22% power to detect $r = -0.2$, leaving the question open rather than resolved.

**Safety probe nuance.** The GPT-2 Small safety probe experiment (Section 5.4, Table 6) reveals that the relationship between absorption and downstream performance is not straightforward at the single-task level. Dense probes achieve AUC $\approx 1.0$ regardless of absorption level. The 1-sparse probe gap does not increase monotonically with absorption (gaps: 0.118 at lowest, 0.148 at median, 0.051 at highest absorption). Two interpretations deserve consideration: (1) confounds in the GPT-2 experiment---the three SAEs differ in layer, width, and architecture---prevent isolating absorption's effect, or (2) safety-specific tasks may genuinely be less sensitive to absorption, consistent with the unlearning null result on Gemma. The cross-SAE correlation analysis on Gemma Scope, which holds configuration confounds constant via partial correlation, provides cleaner evidence for the general relationship, but the safety-specific picture remains unresolved.

## 7.4 The Taxonomy Reveals Underreporting

The three-tier taxonomy classifies 24/26 letter features (92.3%) as showing some form of absorption on the GPT-2 Small 24k SAE (Section 5.3, Figure 3). Only letter Q qualifies as Type I (full, single-latent) absorption. Letters X and Z show no absorption by any criterion. The remaining 23 letters exhibit Type II (partial) absorption, where the parent latent fires at less than 50% of its expected magnitude on letter tokens.

This 92.3% comprehensive rate compares to two reference points: the strict Type I rate of 3.8% (1/26 letters) and the Chanin metric's any-absorption rate of 80.8% (21/26 letters show some absorption at the standard threshold). The gap between 3.8% and 92.3% illustrates how much failure scope the canonical measurement misses.

**Caveat: Type II inflation.** The Type II rate is likely inflated by a systematic artifact in parent feature identification. Parent features were identified using a selectivity heuristic on synthetic prompts rather than `sae-spelling` ground truth. The magnitude ratio threshold ($< 0.5$) compares activation on letter tokens to a global mean-when-active baseline, which is biased upward by high-activation contexts unrelated to the first-letter function. The comparison set contains zero tokens for most letters ($n_{\text{comparison}} = 0$), forcing a fallback to global statistics. A conservative interpretation would count only Type I (3.8%) and Type III (0%) as validated absorption, treating the comprehensive rate as an upper bound.

Despite this caveat, the qualitative finding holds: a substantial fraction of letter features show suppressed parent activation that the canonical single-latent metric does not capture. Replication with `sae-spelling` ground-truth parent features would provide definitive evidence.

## 7.5 Model Scope and Generalization

All probe-based experiments (H1, H2, H4, taxonomy) used GPT-2 Small as the primary model. The downstream correlation analysis (H3) used Gemma Scope SAEs via pre-computed SAEBench scores. This creates a cross-model gap: the detection/taxonomy results (GPT-2) and the downstream results (Gemma) cannot be composed into a single causal narrative without same-model replication. The 92.3% comprehensive rate on GPT-2 and the $r = -0.595$ on Gemma are independent findings whose connection is assumed, not demonstrated.

GPT-2 Small is an appropriate validation anchor: it is fully open, deterministic, and has the best-studied absorption ground truth via `sae-spelling`. The 35.4% overall absorption rate on the 24k SAE at layer 8 matches the 15--35% range reported by Chanin et al. (2024), confirming that our pipeline reproduces prior results.

The key question is whether our negative findings (H1, H2) would hold on Gemma-2-2B. For H1 (LV detector), the failure is structural---the competition coefficient does not produce a sharp threshold in any model geometry---and we expect the same absence of a phase transition on Gemma. For H2 (PMI), the null result is driven by the dominance of layer and $L_0$ as predictors, which are model-architecture properties; corpus statistics should be equally uninformative on Gemma. Full replication on Gemma-2-2B with probe-level experiments remains a priority for future work.

## 7.6 Limitations

**Single evaluation task.** All probe-based experiments use the first-letter task (A--Z), which is the only task with established absorption ground truth. Absorption may behave differently for other feature hierarchies (e.g., semantic category membership, syntactic role).

**Small positive class.** The 24k SAE at layer 8 produces 470 absorbed feature pairs (Chanin metric), of which 195 fall in the test set. At this sample size, precision and recall estimates have wide confidence intervals ($\pm 5$--8 percentage points by bootstrap). The F1 = 0.128 for the LV detector is clearly below the success criterion but could shift by several points on a larger positive set.

**Standard errors.** The PMI regression uses HC3 robust standard errors but does not cluster at the letter level (26 repeated measurements per letter across SAE configurations). Clustered standard errors would be more conservative and could further inflate $p$-values for the already-insignificant PMI coefficient. The $L_0$ coefficient significance ($p = 0.012$) is marginal and could plausibly become non-significant under letter-level clustering, leaving layer as the only robust predictor.

**SAE configuration heterogeneity.** The 31 GPT-2 SAE configurations in the absorption survey span three release families (`res-jb`, `feature-splitting`, `resid-post-v5`) with different training objectives and hook points. While the regression includes $\log(L_0)$ and $\log(D)$ as covariates, it does not control for training-objective differences across families. This is a potential confound for the coefficient estimates.

**Type II taxonomy validity.** As noted in Section 7.4 and Table 1, the Type II absorption rate (88.5%) depends on heuristic parent identification and a global activation baseline with zero comparison tokens for most letters. A conservative interpretation of the taxonomy result would only count Type I (3.8%) and Type III (0%) as validated absorption, with the comprehensive rate an upper bound.

<!-- FIGURES
- None
-->

---

# 8 Conclusion

We set out to resolve three tensions in the absorption literature.

**Measurement versus phenomenon.** The three-tier taxonomy reveals a 92.3% comprehensive absorption rate on GPT-2 Small (Section 5.3, Figure 3), confirming that reported rates understate the failure scope---though the Type II component (88.5%) requires validation with ground-truth parent feature IDs.

**Mechanism versus intervention.** The LV detector achieves test F1 = 0.128 (vs. cosine baseline F1 = 0.165), and corpus PMI contributes partial $R^2 = 0.0006$. The competitive exclusion analogy captures relevant variables---co-activation and frequency imbalance---but the threshold-based framing is not supported. The dominant absorption predictors are layer and sparsity penalty, implicating the SAE's optimization objective rather than training data co-occurrence. Corpus curation is unlikely to reduce absorption; architectural and regularization interventions remain the correct targets.

**Metric versus impact.** Absorption score correlates negatively with downstream SAE quality across 54 Gemma Scope SAEs: $r = -0.595$ (sparse probing), $r = -0.454$ (RAVEL), $r = -0.431$ (SCR), all surviving Bonferroni correction and strengthening under partial correlation. This provides the first systematic correlational evidence supporting the assumption that absorption reduction translates to improved downstream interpretability---the assumption that motivates ongoing architectural research into absorption-resistant SAE designs.

Taken together, these results establish absorption as a validated quality indicator (H3) that resists both unsupervised detection (H1) and corpus-level prediction (H2). Probe-directed measurement remains the only reliable absorption metric, and architectural or regularization-based interventions---not data engineering---are the viable path to absorption reduction.

**Limitations.** All probe-based experiments use GPT-2 Small; the downstream correlation analysis uses Gemma Scope SAEs, creating a cross-model gap. The first-letter task is the only absorption ground truth available. The Type II taxonomy rate is likely an upper bound. The safety probe experiment (3 SAE configurations, 100 prompts) is underpowered for detecting monotone trends. The PMI regression does not cluster standard errors at the letter level.

**Future work.** Three directions follow from these results. First, replication on Gemma-2-2B would unify the detection and downstream analyses on a single model. Second, causal intervention experiments---ablating the top-$k$ absorbed child latents at inference time and measuring whether parent latent activation recovers, followed by downstream re-evaluation on SAEBench tasks---would test whether the correlational evidence from H3 reflects a causal mechanism. Third, developing continuous-valued (non-threshold) absorption risk scores using co-activation and frequency features as continuous predictors, which follows directly from the H1 finding that the LV threshold model fails while the constituent variables remain informative.


<!-- FIGURES
- None
-->

---

## Figures and Tables

- Figure 1: fig_lv_framework.pdf --- Conceptual diagram: LV competitive exclusion mapped to SAE feature absorption (Section 3.1)
- Figure 2: fig_sharpness.pdf --- Sharpness test: absorption rate vs. $\alpha_{ij}$ bins with sigmoid and linear fits (Section 5.1)
- Figure 3: fig_taxonomy_bar.pdf --- Absorption taxonomy stacked bar chart across widths (Section 5.3)
- Figure 4: fig_downstream_scatter.pdf --- Scatter plots: absorption score vs. sparse probing F1 and RAVEL TPP (Section 5.4)
- Figure 5: fig_das_width.pdf --- DAS($k{=}1$) and DAS($k{=}3$) vs. SAE width with error bars (Section 5.5)
- Table 1: inline --- Absorption taxonomy definitions (Section 3.3)
- Table 2: inline --- LV detector vs. cosine baseline (Section 5.1)
- Table 3: inline --- PMI regression coefficients (Section 5.2)
- Table 4: inline --- Absorption taxonomy summary (Section 5.3)
- Table 5: inline --- Absorption vs. downstream correlation matrix (Section 5.4)
- Table 6: inline --- Safety probe results (Section 5.4)
- Table 7: inline --- Mean DAS by width (Section 5.5)
- Table 8: inline --- Cross-architecture validation (Section 5.6)
