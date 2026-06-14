# Supervisor Review: Augmentation Ordering Study

**Date**: 2026-04-03
**Role**: Independent Research Supervisor (NeurIPS-calibrated)
**Scope**: Research contribution — novelty, soundness, experimental rigor, reproducibility

---

## Overall Score: 4.0 / 10

**Verdict: CONTINUE** (full-scale experiments required before any submission)

**Dimension Scores**:
| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty & Significance | 7/10 | Genuine gap; well-motivated question |
| Technical Soundness | 4/10 | Proof gap in Theorem 1; DPI logical inconsistency |
| Experimental Rigor | 2/10 | Pilot-scale only; two blocks near random chance; no statistical tests |
| Reproducibility | 5/10 | Protocol well-documented; pilot implementation matches spec |

---

## Executive Summary

This paper addresses a genuine and documented gap in the augmentation literature: no prior work isolates augmentation operation ordering as the sole independent variable in a controlled factorial experiment. Two published surveys (Cheung & Yeung 2023; Yang et al. 2023) explicitly identify this as an open question, and no concurrent work threatens the novelty window as of April 2026. The theoretical scaffolding — Wasserstein NC measure and DPI reversibility principle — is creative and provides the paper with a distinctive theoretical identity beyond a pure ablation study.

However, the current draft presents exclusively pilot-scale results:

- **Tier 1**: 10 epochs, 100-sample subsets, 1 seed per condition
- **Tier 2**: 10 epochs, 5k samples, 1 seed, 1 architecture, 1 dataset
- **Tier 3**: 3 magnitude levels, 1 seed each
- **Tier 4**: NC_2 computed on 100 samples / 100 projections (pre-registered: 10k / 1000)

With n=1 seed, no paired t-test or ANOVA is possible. The hypothesis verdicts ("H1 confirmed", "H5 falsified") are premature. Two of four architecture-dataset blocks have accuracy near the random-chance floor (ResNet-18 CIFAR-10: 10.01–10.97%; ViT-S/4 CIFAR-100: 2.64–2.89%), making ordering comparisons in those blocks uninterpretable. The theoretical contributions have proof gaps. No venue would accept this in current form.

The path to a publishable paper is clear: run full Tier 1 (20 GPU-h), fix the theoretical gaps, and reframe the narrative. The research question is worth pursuing.

---

## Dimension 1: Novelty & Significance (7/10)

**Strengths**:
- The augmentation ordering question is explicitly named as an open problem in two independent survey papers; this is unusually strong motivation.
- No prior work performs a controlled factorial experiment isolating ordering as the sole independent variable. AutoAugment searches over ordered sub-policies without ablating order; RandAugment randomizes order without comparing fixed orderings; PBA tests epoch-level schedule ordering, not per-image operation ordering.
- The NC_2 Wasserstein measure and the DPI-based reversibility principle are novel theoretical contributions, even if they are subsequently falsified as predictors. Showing that natural theoretical candidates fail is itself a scientific contribution.
- The ViT-vs-CNN architecture-dependent sensitivity, if confirmed at full scale, adds depth beyond a pure ordering study.

**Weaknesses**:
- The theoretical contributions' predictive value has already failed at pilot scale. If the full-scale experiments confirm that NC_2 and DPI-MI do not predict accuracy rankings, the paper must reframe its theoretical contribution as "explanation of why standard distributional measures fail, and what that implies about augmentation theory." This reframe is defensible but requires careful execution.
- The practical significance depends entirely on whether the ordering effects survive full-scale training. At pilot scale, the best-worst spread on convergent blocks (0.88-2.32%) is suggestive but not confirmed.

---

## Dimension 2: Technical Soundness (4/10)

### Issue 1 (Critical): Missing bubble-sort decomposition in Theorem 1 proof

The generalization bound states:
$$|gen(\sigma) - gen(\sigma')| \leq \frac{2}{\sqrt{n}} \sum_{(i,j): \text{order differs}} NC_2(t_i, t_j; \mu)$$

The proof sketch jumps from "By the Lipschitz property" to this sum without showing the key step: decomposing the permutation difference $\sigma \to \sigma'$ into a sequence of adjacent transpositions (bubble sort), applying the triangle inequality at each step, and bounding each step's $W_2$ contribution by the corresponding $NC_2$ term. This is the mathematical core of the proof and its absence leaves the theorem unverified.

**Required fix**: Show the bubble-sort decomposition explicitly, either in the proof sketch (4-5 equations) or in a formal appendix.

### Issue 2 (Major): DPI contraction coefficient is distribution-independent as defined

The contraction coefficient $\eta_i = \sup_{p \neq q} D_{KL}(t_i \# p \| t_i \# q) / D_{KL}(p \| q)$ is a worst-case quantity over all input distributions. By definition, $\eta_i$ is a property of channel $t_i$ alone, not of its position in the pipeline. The claim "placing a high-contraction transform early discards information that subsequent transforms could have preserved" requires that contraction be input-distribution-dependent. The current definition defeats this argument.

**Required fix**: Redefine $\eta_i(\nu)$ as input-distribution-dependent, conditioning the supremum on distributions $p, q \ll \nu$. The ordering-dependence of mutual information then follows from the ordering-dependence of $\nu$ (which is the output distribution of all preceding transforms).

### Issue 3 (Minor): RandomHorizontalFlip classified as "medium reversibility"

The section calls Flip "perfectly invertible" but classifies it as medium reversibility. A perfectly invertible bijection has $\eta = 1$ and zero information cost; the "medium" classification mixes information-theoretic and semantic notions of reversibility. The classification should be consistent with the formal definition.

---

## Dimension 3: Experimental Rigor (2/10)

### Fatal Problem: Two of four blocks are near random chance

| Block | Accuracy Range | Random Chance | Interpretable? |
|-------|---------------|---------------|----------------|
| ResNet-18 CIFAR-10 | 10.01–10.97% | 10% (10 classes) | No |
| ViT-S/4 CIFAR-100 | 2.64–2.89% | 1% uniform / ~2-5% without learning | Borderline |
| ResNet-18 CIFAR-100 | 45.75–46.63% | ~1% | Yes |
| ViT-S/4 CIFAR-10 | 17.38–19.70% | 10% | Yes |

The H1 "confirmation" count of "3/4 blocks" includes ResNet-18 CIFAR-10 where the model has learned essentially nothing. Excluding this degenerate block, only 2 of 3 interpretable blocks show spread > 0.5%. This is a weaker signal than claimed.

### No statistical tests executed

The pre-registered analysis plan requires: paired t-tests (same seed, different ordering), Bonferroni correction for 15 pairwise comparisons per block, Cohen's d from between-seed variance, and two-way ANOVA for the ordering × architecture interaction. With n=1 seed, none of these are possible. The paper should not use "confirmed" or "falsified" terminology for any hypothesis until these tests are executed.

### Asymmetric baseline comparison

Ordering experiments: 10 epochs, 100 samples. Baselines: 30 epochs, full dataset. The paper places these in the same table with a caveat. This framing is methodologically unsound and will immediately disqualify the paper at any rigorous venue. The solution is either to run baselines at the same pilot conditions or to exclude baselines from the results section until full-scale experiments are complete.

### NC_2 computed at 1% of pre-registered scale

The pre-registration specifies 10k samples / 1000 projections for the NC_2 SWD computation. The actual Tier 4a used 100 samples / 100 projections — a 100x/10x reduction. The resulting NC_2 values (range 0.035–0.052) may be unreliable estimates at this scale for 3072-dimensional image distributions. The H3 falsification verdict should be labeled "preliminary" until full-scale recomputation.

### M=14 identical accuracy is suspicious

In Tier 3, both orderings at M=14 achieve exactly 0.245 accuracy (to 3 decimal places). This exact equality is statistically improbable for independently trained models unless: (a) there is a genuine convergence to the same noise floor, or (b) there is a data integrity issue (shared cache, shared seed state). The paper should verify that the M=14 runs were genuinely independent and report per-epoch training curves.

---

## Dimension 4: Reproducibility (5/10)

**Strengths**:
- Hyperparameters are specified precisely (SGD momentum 0.9, weight decay 5e-4, cosine annealing; AdamW weight decay 0.05, 10-epoch warmup).
- Paired seed design is well-explained and scientifically justified.
- The four-tier experimental structure is well-documented with clear scope for each tier.
- The experimental code is functional (pilot infrastructure produces results that match reported numbers upon cross-validation).

**Weaknesses**:
- Augmentation magnitude mapping (M → specific crop_padding, cj_brightness, etc.) is not specified in the paper text. The M=14 configuration (crop_padding=8, cj_brightness=0.8, rotation=30) appears in the raw results JSON but not in the method section.
- The within-category ordering for Tier 2 (e.g., which specific geometric operations appear in which positions within the geometric block) is underspecified.
- Code availability is not mentioned anywhere in the paper.

---

## Cross-Validation: Paper Claims vs. Raw Data

The following claims were cross-validated against the raw experiment JSON files:

| Claim | Raw Data | Match? |
|-------|----------|--------|
| ViT-S/4 CIFAR-10 spread = 2.32% | tier1_analysis.json: vit_cifar10 spread = 0.0232 | Yes |
| Best ordering: Flip->CJ->Crop (ViT CIFAR-10) | tier1_analysis.json: order_3 best_acc = 0.197 | Yes |
| NC_2 Crop-CJ = 0.051 | tier4a_nc.json: swd_forward_vs_reverse = 0.05149 | Yes |
| rho_s(NC_2) = -0.20, p = 0.68 | tier4_correlation.json: rho = -0.2, p_value = 0.6831 | Yes |
| M=14 spread = 0.00% | tier3_results.json: M14 spread = 0.0 | Yes |
| Tier 2 interleaved P->G = 29.39% | final_summary.json: interleaved_pg = 0.2939 | Yes |
| H2 "confirmed": reversibility-sorted wins 2/4 | tier4_correlation.json: blocks_confirmed = 2 | Yes |

No numerical discrepancies found between paper text and raw data files. The reporting is honest and the numbers match.

---

## What Would Raise the Score

**From 4.0 to 6.5** (one major experimental action):
- Run full Tier 1: 200 epochs, 5 seeds, full CIFAR-10/100 datasets
- Execute pre-registered statistical tests: paired t-tests, Bonferroni correction, two-way ANOVA
- Achieve significance (p < 0.05 corrected) in at least 2/4 blocks with spread > 0.5%

**From 6.5 to 7.0** (fix theoretical gaps):
- Add bubble-sort decomposition proof for Theorem 1
- Fix DPI contraction coefficient to be distribution-dependent

**From 7.0 to 7.5** (Tier 2 validation):
- Confirm that category-level interleaved ordering advantage holds above 2 pp at full scale (50 epochs, 2 seeds, full datasets)
- If confirmed, this becomes the paper's headline practical finding

**From 7.5 to 8.0** (theoretical reframing):
- Reframe NC and DPI as "natural theoretical candidates that fail, and show why"
- Introduce feature-space NC_2 as alternative theoretical bridge
- Lead the paper with empirical discovery, not with theoretical predictions that are subsequently falsified

A score of 8.0+ would require: confirmed ordering effects with statistical significance across multiple blocks, the Tier 2 category-ordering result validated, at least partial theoretical account of why ordering matters (e.g., feature-space NC_2 recovers predictive power), and results at ImageNet scale or at least CIFAR-100 at full resolution.

---

## Summary of Issues by Severity

| Severity | Count | Top Issue |
|----------|-------|-----------|
| Critical | 3 | Pilot-scale-only results; degenerate training blocks; proof gap in Theorem 1 |
| Major | 5 | DPI logical gap; NC_2 at 1% scale; Tier 2 unvalidated; M=14 suspicious; H2 redefinition |
| Minor | 2 | Flip reversibility classification inconsistency; MI combined rho obscures sign flip |

---

## Recommendation

**CONTINUE** with full-scale Tier 1 experiments before any submission. The research question is novel and worth pursuing. The experimental infrastructure is validated. The pilot evidence is consistent with a real effect in interpretable blocks. The theoretical gaps are fixable. No fundamental design flaw prevents a publishable result.

The mandatory reframe: drop "theory-validated-by-experiment" as the primary framing. Lead with empirical discovery, report NC and DPI falsifications as findings, and develop the theory-practice gap as the novel theoretical contribution.
