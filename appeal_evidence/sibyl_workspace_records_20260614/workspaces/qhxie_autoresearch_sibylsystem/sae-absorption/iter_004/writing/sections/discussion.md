# 7 Discussion

## 7.1 Why the LV Framework Fails as a Detector

The competition coefficient $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ achieves test F1 = 0.128, below both the pre-registered success criterion (F1 > 0.35) and the cosine-similarity baseline (F1 = 0.165). The sharpness test (Section 5.1, Figure 2) reveals why: a linear fit (AIC = $-61.05$) marginally outperforms the sigmoid (AIC = $-60.95$), and the sigmoid's midpoint parameter diverges to $x_0 = 10.0$ --- far from the predicted $\alpha_{ij} \approx 1$. Absorption is not a phase transition at a critical threshold; it is a gradual process with no sharp boundary in competition-coefficient space.

Two structural problems undermine the LV analogy. First, ecological competitive exclusion assumes that species interact through a shared resource, producing a zero-sum dynamic. SAE latents, by contrast, interact through the sparsity penalty during training, not through shared activation budget at inference time. A high $\alpha_{ij}$ at inference indicates that latent $j$ frequently co-activates with latent $i$ and is more common --- but this does not imply that $j$ is suppressing $i$. The encoder may simply assign both latents to overlapping input regions without competition. Second, the frequency ratio $f_j / f_i$ confounds activation frequency with competitive advantage. A rare parent feature $i$ may have low $f_i$ because it is genuinely selective (e.g., "first letter = Q") rather than because it has been excluded by a competitor. The LV framework conflates rarity with suppression.

The conceptual contribution survives: co-activation geometry and frequency imbalance are the right variables for characterizing absorption risk. The failure is specifically in the threshold-based binary prediction --- the claim that $\alpha_{ij} > 1$ produces a discrete transition from non-absorbed to absorbed. Future unsupervised detection methods should treat these quantities as continuous predictors rather than threshold discriminants.

## 7.2 Corpus Statistics Are Not the Proximal Cause

The null PMI result ($\beta_4 = -0.0063$, $p = 0.593$, partial $R^2 = 0.0006$; Section 5.2) rules out corpus co-occurrence as a meaningful predictor of absorption patterns. Per-layer coefficient signs are inconsistent (5/9 negative, 3/9 positive, 1 zero-variance; Table 3), eliminating the possibility that a genuine effect is masked by layer-level heterogeneity.

The strongest absorption predictors are structural: layer index ($\beta_3 = -0.012$, $p < 0.0001$) and $\log(L_0)$ ($\beta_1 = 0.013$, $p = 0.012$). Absorption decreases in later layers and increases with the sparsity penalty. Both findings are consistent with the optimization-driven account: absorption is produced by the SAE's training objective interacting with feature geometry, not by the statistics of the training corpus.

This result has a direct implication for intervention design. Data engineering approaches --- corpus curation, co-occurrence disruption, or frequency balancing --- are unlikely to reduce absorption. The effective intervention point is the training algorithm: masked regularization (arXiv:2604.06495), orthogonality constraints (OrtSAE), or hierarchically-aware objectives that explicitly penalize parent-child competition during optimization.

## 7.3 Absorption Predicts Downstream Quality

The H3 falsification is the strongest empirical result in this paper. Across 54 Gemma Scope SAEs spanning three layers (5, 12, 19) and three widths (16k, 65k, 1M), absorption score correlates negatively with downstream interpretability performance:

- Sparse probing F1: $r = -0.595$ ($p < 0.001$), partial $r = -0.661$ controlling for width, layer, and architecture
- RAVEL (TPP): $r = -0.454$ ($p < 0.001$), partial $r = -0.492$
- SCR: $r = -0.431$ ($p = 0.002$), partial $r = -0.677$
- Unlearning: $r = -0.175$ ($p = 0.280$) --- not significant

Three of four tasks survive Bonferroni correction ($\alpha = 0.0125$). Partial correlations strengthen after controlling for confounds, indicating that absorption captures genuine quality variation beyond what width and layer predict alone. The partial $r = -0.677$ for SCR is the largest effect: after removing the variance explained by SAE configuration, absorption accounts for approximately 46% of residual variance in spurious correlation removal performance.

These correlations provide the first direct test of the assumed causal chain from absorption to downstream interpretability. Prior to this work, the research community treated absorption reduction as an intrinsic goal, without systematic evidence that lower absorption translates to better SAE utility. Our results validate this assumption for three of four tasks.

**Unlearning is the exception.** The non-significant correlation ($r = -0.175$, $p = 0.280$) with unlearning performance, combined with the smaller sample size ($n = 40$ vs. $n = 54$ for other tasks), suggests that unlearning depends on factors other than feature fidelity --- potentially on the SAE's ability to isolate safety-relevant directions regardless of whether general features are absorbed.

**Safety probe nuance.** The GPT-2 Small safety probe experiment (Section 5.4, Table 5) reveals that the relationship between absorption and downstream performance is not straightforward at the single-task level. Dense probes achieve AUC $\approx 1.0$ regardless of absorption level. The 1-sparse probe gap does not increase monotonically with absorption (gaps: 0.118 at lowest, 0.148 at median, 0.051 at highest absorption). This non-monotonicity likely reflects confounds in the GPT-2 experiment: the three SAEs differ in layer (5, 8, 10), width (768 to 32k), and architecture --- not just absorption level. The cross-SAE correlation analysis on Gemma Scope, which holds these confounds constant via partial correlation, provides cleaner evidence. The safety probe result should be interpreted as showing that absorption's downstream impact is real but mediated by task-specific and configuration-specific factors that a simple three-SAE comparison cannot disentangle.

## 7.4 The Taxonomy Reveals Underreporting

The three-tier taxonomy classifies 24/26 letter features (92.3%) as showing some form of absorption on the GPT-2 Small 24k SAE (Section 5.3, Figure 4). Only letter Q qualifies as Type I (full, single-latent) absorption. Letters X and Z show no absorption by any criterion. The remaining 23 letters exhibit Type II (partial) absorption, where the parent latent fires at less than 50% of its expected magnitude on letter tokens.

This 92.3% comprehensive rate compares to two reference points: the strict Type I rate of 3.8% (1/26 letters) and the Chanin metric's any-absorption rate of 80.8% (21/26 letters). The gap between 3.8% and 92.3% illustrates how much failure scope the canonical measurement misses.

**Caveat: Type II inflation.** The Type II rate is likely inflated by a systematic artifact in parent feature identification. Parent features were identified using a selectivity heuristic on synthetic prompts rather than `sae-spelling` ground truth. The magnitude ratio threshold ($< 0.5$) compares activation on letter tokens to a global mean-when-active baseline, which is biased upward by high-activation contexts unrelated to the first-letter function. Additionally, the comparison set contains zero tokens for most letters ($n_{\text{comparison}} = 0$), forcing a fallback to global statistics. The Type II classification should therefore be interpreted as "the putative parent feature fires weakly on letter tokens compared to its overall activation pattern" rather than "the parent feature is partially suppressed due to absorption by child features." The causal link to absorption is not established for Type II.

Despite this caveat, the qualitative finding holds: a substantial fraction of letter features show suppressed parent activation that the canonical single-latent metric does not capture. Replication with `sae-spelling` ground-truth parent features would provide definitive evidence.

## 7.5 Model Scope and Generalization

All probe-based experiments (H1, H2, H4, taxonomy) used GPT-2 Small as the primary model because Gemma-2-2B requires gated HuggingFace access that was not available during experimentation. The downstream correlation analysis (H3) used Gemma Scope SAEs via pre-computed SAEBench scores, providing a partial bridge to the larger model family.

GPT-2 Small is an appropriate validation anchor: it is fully open, deterministic, and has the best-studied absorption ground truth via `sae-spelling`. The 35.4% overall absorption rate on the 24k SAE at layer 8 matches the 15--35% range reported by Chanin et al. (2024), confirming that our pipeline reproduces prior results.

The key question is whether our negative findings (H1, H2) would hold on Gemma-2-2B. For H1 (LV detector), the failure is structural --- the competition coefficient does not produce a sharp threshold in any model geometry --- and we expect the same absence of a phase transition on Gemma. For H2 (PMI), the null result is driven by the dominance of layer and $L_0$ as predictors, which are model-architecture properties; corpus statistics should be equally uninformative on Gemma. For H3, we already have Gemma Scope evidence (54 SAEs) confirming the positive result. Full replication on Gemma-2-2B with probe-level experiments remains a priority for future work.

## 7.6 Limitations

**Single evaluation task.** All probe-based experiments use the first-letter task (A--Z), which is the only task with established absorption ground truth. Absorption may behave differently for other feature hierarchies (e.g., semantic category membership, syntactic role). The cross-hierarchy absorption experiment was not feasible in this iteration due to probe quality constraints (see Section 7.5).

**Small positive class.** The 24k SAE at layer 8 produces 470 absorbed feature pairs (Chanin metric), of which 195 fall in the test set. At this sample size, precision and recall estimates have wide confidence intervals ($\pm 5$--8 percentage points by bootstrap). The F1 = 0.128 for the LV detector is clearly below the success criterion but could shift by several points on a larger positive set.

**Standard errors.** The PMI regression uses HC3 robust standard errors but does not cluster at the letter level (26 repeated measurements per letter across SAE configurations). Clustered standard errors would be more conservative and could further inflate $p$-values for the already-insignificant PMI coefficient.

**SAE configuration heterogeneity.** The 31 GPT-2 SAE configurations in the absorption survey span three release families (`res-jb`, `feature-splitting`, `resid-post-v5`) with different training objectives and hook points. While the regression includes $\log(L_0)$ and $\log(D)$ as covariates, it does not control for training-objective differences across families. This is a potential confound for the coefficient estimates.

**Type II taxonomy validity.** As noted in Section 7.4, the Type II absorption rate (88.5%) depends on heuristic parent identification and a global activation baseline. A conservative interpretation of the taxonomy result would only count Type I (3.8%) and Type III (0%) as validated absorption, with the comprehensive rate an upper bound requiring confirmation.

<!-- FIGURES
- None
-->
