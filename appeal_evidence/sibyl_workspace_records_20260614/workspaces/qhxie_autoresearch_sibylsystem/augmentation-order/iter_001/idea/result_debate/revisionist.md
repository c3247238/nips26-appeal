# Revisionist Analysis: Augmentation Ordering Study

**Agent**: Revisionist  
**Date**: 2026-04-02  
**Data sources**: final_summary.json, tier1_analysis.md, tier3_results.json, tier4a_nc.json, tier4b_mi.json, pilot_summary.md

---

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|---|---|---|---|
| H1 (Ordering matters) | **Confirmed** | 3/4 arch-dataset blocks show spread > 0.5%; max spread = 2.32% (ViT x CIFAR-10). Tier 0 pilot showed 2.68% spread. | **Medium** -- pilot-mode (10 epochs, 1 seed for Tier 1) prevents statistical testing. Cohen's d values (2.27-2.71) are astronomically high for single-seed data and therefore unreliable. |
| H2 (Architecture differential) | **Inconclusive** (relabeled from "confirmed") | The original H2 predicted CNN sensitivity to photometric and ViT sensitivity to geometric ordering. The final_summary.json relabeled H2 as "reversibility-sorted wins" -- a different hypothesis entirely. On the actual H2 question: ViT shows larger spread (2.32%) than ResNet (0.96%) on CIFAR-10, but the best orderings differ inconsistently across datasets, not along the predicted CNN-photometric / ViT-geometric axis. No ANOVA was possible (1 seed). | **Low** -- cannot assess interaction without multi-seed data. |
| H3 (NC predicts effect) | **Falsified** | Spearman rho = -0.20, p = 0.68 (permutation). The NC_2 proxy ranked Crop-CJ as most non-commutative, but this did not translate to accuracy sensitivity. The negative correlation means the measure actively misleads. | **Medium-High** -- small n (6 orderings, 3 pairs), but the direction is wrong, not just weak. |
| H4 (DPI/MI predicts accuracy) | **Inconclusive** | CIFAR-10 rho = +0.54 (p = 0.28), CIFAR-100 rho = -0.66 (p = 0.19). Combined rho = -0.057. The sign flips across datasets, making the MI estimate non-predictive in general. Reversibility-sorted ordering (CJ->Flip->Crop) has higher MI on CIFAR-10 but NOT on CIFAR-100, and ties on accuracy in both. | **Low** -- contradictory signals across datasets. |
| H5 (Magnitude scaling) | **Falsified** | M5 spread = 0.35%, M9 spread = 0.88%, M14 spread = 0.00%. The non-monotonic pattern (inverted U) directly contradicts the hypothesis. At M14, both orderings converge to identical accuracy (0.245). | **Medium** -- single seed, but M14 = 0.0 spread is a clear signal. |

---

## 2. Surprise Analysis

### Surprise 1: M14 spread collapse to zero (deviation: -100% from predicted)

**Expected**: H5 predicted spread would be largest at M14 (at least 1.5x the M5 spread).  
**Observed**: M14 spread = 0.000, both orderings at 24.5%.  
**Wrong assumption**: We assumed non-commutativity effects would monotonically amplify with magnitude. Instead, extreme augmentation appears to create a "destructive regime" where the training signal is so corrupted that ordering becomes irrelevant -- both orderings produce equally poor feature extractors. The information loss is so severe that the DPI chain I(y; A_sigma(x)) collapses regardless of the order in which information is destroyed. This is analogous to multiplying two very small numbers: the order does not matter because the product is approximately zero either way.

### Surprise 2: NC_2 correlation is negative (-0.20), not positive (deviation: -140% from threshold)

**Expected**: rho > 0.5 between NC_2 and accuracy sensitivity.  
**Observed**: rho = -0.20, meaning higher non-commutativity between transform pairs actually corresponds to LESS accuracy sensitivity to their ordering.  
**Wrong assumption**: The theoretical framework assumed that the Wasserstein distance between pushforward distributions (t_i o t_j # mu vs t_j o t_i # mu) in pixel space directly translates to differences in learned representations. But pixel-space distributional distance and feature-space learning dynamics are different things. A pair may produce very different pixel distributions (high NC_2) but the model may be invariant to those differences, or SGD may erase them over training. The Sliced Wasserstein Distance proxy (100 samples, 100 projections) is also a crude approximation of W_2 -- but even with better estimation, the conceptual gap between pixel-space non-commutativity and accuracy impact remains.

### Surprise 3: Best ordering flips between datasets (deviation from any consistent pattern)

**Expected**: A single "best" ordering principle (whether NC-based, DPI-based, or conventional).  
**Observed**:  
- ResNet CIFAR-10: CJ->Flip->Crop best (reversibility-sorted wins)  
- ResNet CIFAR-100: Flip->Crop->CJ best (geometric-first wins)  
- ViT CIFAR-10: Flip->CJ->Crop best (mixed)  
- ViT CIFAR-100: Flip->Crop->CJ best (geometric-first wins)  
**Wrong assumption**: We assumed ordering effects are a property of the transforms themselves (via NC_2 or reversibility). The data suggests ordering effects are a joint function of transforms, architecture, AND dataset difficulty. On CIFAR-100 (harder task, more classes), geometric-first (Flip->Crop->CJ) consistently wins across both architectures. On CIFAR-10 (easier), the pattern breaks. This suggests that when the classification task is harder, preserving spatial structure early (geometric-first) matters more -- the model needs every bit of spatial signal. When the task is easy (CIFAR-10), the model can afford to lose spatial information early and still learn.

### Surprise 4: Tier 0 vs Tier 1 ordering rankings are inconsistent

**Expected**: Ordering rankings should be stable across pilot scales.  
**Observed**:  
- Tier 0 (5k, 10 epochs): CJ->Crop->Flip best, Flip->CJ->Crop worst  
- Tier 1 ResNet CIFAR-10 (full, 10 epochs, 1 seed): CJ->Flip->Crop best, CJ->Crop->Flip worst  
The best and worst orderings changed between Tier 0 and Tier 1, and both are CJ-first orderings! The Tier 0 "finding" that CJ-first dominates does not robustly hold.  
**Wrong assumption**: We treated the Tier 0 pilot as a reliable direction indicator. But with 5k samples and 10 epochs, the noise floor may exceed the true ordering effect, making rankings unstable.

---

## 3. Mental Model Revision

We assumed that augmentation ordering effects are governed by a universal principle (either non-commutativity magnitude or information-theoretic reversibility) that could be derived from the transforms alone, independent of the downstream task. **The data suggests instead that ordering effects are an emergent property of the transform-architecture-dataset-magnitude quadruple, and no single theoretical reduction captures them.** Specifically: (a) the NC_2 measure from pixel-space distributions does not predict feature-space learning outcomes, breaking the bridge between our theoretical bound and practice; (b) the DPI reversibility principle works on CIFAR-10 but reverses on CIFAR-100, suggesting task difficulty modulates which information loss pathway hurts more; (c) extreme augmentation magnitude creates a "floor effect" that washes out ordering differences entirely, contradicting the simple monotonic amplification model.

The most honest summary: **ordering matters (H1 is real), but WHY it matters is not captured by either theoretical framework we proposed.** The effect appears to be a second-order interaction that resists simple first-principles reduction.

---

## 4. Reframing Test

**Original research question**: "Does the permutation of augmentation operations produce significant accuracy differences, and can the Wasserstein NC measure / DPI reversibility principle predict and explain these differences?"

**Should we reframe?** Yes.

The data supports the first clause (ordering matters) but falsifies the second (our theories do not predict the effect). A revised question that better matches what the data shows:

> "Augmentation ordering produces measurable accuracy differences, but these differences are dataset- and architecture-contingent. What properties of the task-architecture combination determine which orderings are better, and why does the effect exhibit a non-monotonic relationship with augmentation magnitude?"

This reframing acknowledges:
1. The ordering effect is real (strong empirical finding, publication-worthy)
2. The NC/DPI theories are insufficient (honest negative result for the theoretical contribution)
3. The magnitude non-monotonicity is the most surprising and unexplained finding
4. The dataset-dependence pattern (harder tasks favor geometric-first) is the most promising lead for a new explanatory framework

---

## 5. New Hypotheses

### NH1: Task Difficulty Modulates Optimal Ordering (Falsifiable)

**Statement**: For a fixed architecture and augmentation set, harder classification tasks (more classes, lower baseline accuracy) systematically favor geometric-first orderings because they preserve more task-discriminative spatial structure.

**Test**: Run the same 6 orderings on CIFAR-10 subsets with artificially varied difficulty (e.g., 10-class, 20-class, 50-class, 100-class subsets of CIFAR-100). If the geometric-first advantage monotonically increases with number of classes, NH1 is supported. If the pattern is U-shaped or non-monotonic, NH1 is falsified.

**Grounding**: CIFAR-100 (100 classes) shows Flip->Crop->CJ winning across both architectures. CIFAR-10 (10 classes) shows no consistent pattern. This suggests a difficulty-dependent crossover.

### NH2: Magnitude-Ordering Interaction Has an Inverted-U Shape (Falsifiable)

**Statement**: The accuracy spread across orderings peaks at intermediate augmentation magnitudes and collapses at both extremes: at low magnitude, transforms approximately commute (small perturbations); at high magnitude, all orderings destroy task-relevant information equally ("noise floor" regime).

**Test**: Run the best and worst orderings at 7 magnitude levels (M=1,3,5,7,9,11,14) on CIFAR-100 with ResNet-18, 5 seeds. Plot spread vs. magnitude. If the curve is an inverted U peaking at intermediate M, NH2 is confirmed.

**Grounding**: M5=0.35%, M9=0.88%, M14=0.00% already hints at this pattern. But with only 3 points and 1 seed, we cannot distinguish inverted-U from noise.

### NH3: Feature-Space NC_2 (Not Pixel-Space) Predicts Ordering Effects (Falsifiable)

**Statement**: The failure of pixel-space NC_2 is because SGD operates in feature space. Computing non-commutativity in the representation space of a pretrained model (e.g., measuring W_2 between feature distributions of differently-ordered augmented batches using a frozen ResNet backbone) will recover the predicted positive correlation with accuracy.

**Test**: Take a pretrained ResNet-18, freeze it, pass 10k images through all 6 orderings, compute pairwise SWD in the penultimate layer's feature space. Correlate with accuracy rankings. If Spearman rho > 0.5 in feature space (vs. -0.20 in pixel space), NH3 is supported.

**Grounding**: The gap between pixel-space NC_2 and accuracy can be explained if the model's learned invariances absorb some pixel-space non-commutativity. Measuring NC_2 after the model's feature extractor accounts for these invariances.

---

## Critical Methodological Concerns

1. **All results are pilot-mode**: 10 epochs, 1 seed for Tier 1, no paired t-tests possible. The absolute accuracy values (10-19% for Tier 1 orderings vs. 80-92% for baselines) suggest the ordering experiments may not have converged. The baselines appear to be from different runs (200 epochs?) -- if so, direct comparison is invalid.

2. **Suspicious accuracy discrepancy**: Tier 1 ordering results show 10-46% accuracy, while baselines show 80-92%. This is a >40 percentage point gap that is unexplained. If ordering runs used 10 epochs and baselines used 200, the spreads observed in ordering runs may not survive to convergence. **This is the single biggest threat to H1.**

3. **H2 was silently redefined**: The original H2 was about architecture-differential ordering preferences (CNN vs ViT). The final_summary redefined it as "reversibility-sorted wins" -- a test of the DPI hypothesis. This obscures the original architecture question, which remains unanswered.

4. **Tier 2 category ordering pilot**: The 9.01% spread between all_geo_first (20.38%) and interleaved_pg (29.39%) is large, but these are 10-epoch pilots on 5k samples. At such early training, the ordering effect may reflect different optimization trajectories that converge later.

---

## Summary for Synthesizer

**What we learned that we did not expect**: (1) The theoretical frameworks (NC_2, DPI/MI) both fail to predict ordering effects, despite the effects being real. (2) The magnitude-ordering relationship is non-monotonic, not monotonic. (3) The best ordering is dataset-dependent, not universal. (4) There is a concerning accuracy discrepancy between ordering runs and baselines that needs investigation.

**How beliefs should update**: Ordering effects exist but are empirical, not theory-reducible with our current tools. The paper's strongest contribution is the existence proof (H1) and the falsification of both theoretical frameworks (H3, H5). The DPI principle (H4) is a coin flip. The most promising new direction is exploring why task difficulty modulates the optimal ordering.
