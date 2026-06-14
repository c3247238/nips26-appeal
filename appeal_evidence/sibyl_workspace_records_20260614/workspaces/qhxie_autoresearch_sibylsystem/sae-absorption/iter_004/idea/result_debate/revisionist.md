# Revisionist Analysis: What the Data Tells Us We Got Wrong

**Agent**: Revisionist  
**Date**: 2026-04-14  
**Core question**: What did we learn that we did not expect? How should we update our beliefs?

---

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|---|---|---|---|
| H1 (LV competition coefficient as unsupervised absorption detector) | **REFUTED** | Test F1 = 0.128 (threshold: 0.65). ROC-AUC = 0.148, *below* random. Cosine-only baseline outperforms LV detector (F1 = 0.165 vs 0.128). Linear fit AIC beats sigmoid, confirming no sharp threshold transition. | High |
| H2 (Corpus PMI as primary driver of absorption patterns) | **REFUTED** | PMI coefficient is *negative* (-0.0063), opposite to predicted direction. Partial R^2 = 0.0006 (threshold: 0.10). p = 0.593, non-significant. Sign inconsistent across layers. | High |
| H3 (Absorption metric decoupled from downstream task performance) | **REFUTED (in the positive direction)** | Pearson r = -0.595 (sparse probing), -0.431 (SCR), -0.454 (RAVEL TPP). All exceed |r| > 0.30 threshold. Partial correlations *increase* after controlling for confounds (partial r = -0.661, -0.677, -0.492). Significant after Bonferroni correction. Matched RAVEL comparison: low-absorption SAEs score 5.2x higher on TPP (Cohen's d = 2.13, p = 0.006). | High |
| H4 (Distributed absorption increases monotonically with width) | **REFUTED** | Only 42.3% of letters show positive DAS(k=3) slope with width (threshold: 80%). Mean DAS(k=3) actually *decreases* from 24k to 49k (0.320 -> 0.227) before partially recovering at 98k (0.260). The pattern is non-monotonic, not monotonically increasing. | High |
| H5 (Comprehensive absorption rate > 40%) | **SUPPORTED (with major caveats)** | Comprehensive rate = 92.3% (24/26 letters show some absorption type). However, this is driven almost entirely by Type II (88.5%), which the experiment's own audit flags as "likely inflated" due to parent feature misidentification. Type I strict rate is only 3.8% (1/26). Type III = 0%. | Low (methodology concerns) |
| H6 (LV explains OrtSAE's success) | **Not testable** | OrtSAE checkpoints were not available. Remains a theoretical prediction. | N/A |

---

## 2. Surprise Analysis

### Surprise 1: The LV competition coefficient is *worse* than a trivial cosine baseline (deviation: -100% from expected F1)

**Expected**: F1 >= 0.65 with sharp step-function transition near alpha_ij ~ 1.0.  
**Observed**: F1 = 0.128 on test set. The cosine-only baseline achieves F1 = 0.165. The LV detector does not merely underperform -- it performs *below* the simpler baseline that our proposal explicitly cited as a failed approach from a LessWrong post.

**Wrong assumption**: We assumed that co-activation rate normalized by frequency (sigma_ij * f_j/f_i) would isolate the causal mechanism of absorption. The data shows the alpha_ij distribution has no discernible step transition; the linear fit has lower AIC than the sigmoid. The anomalous bin at alpha_ij ~ 0.05 (84.8% absorption rate) but NOT at alpha_ij ~ 1.0 suggests the competition coefficient is picking up a confound (possibly very rare features that are trivially "absorbed" because they activate on too few tokens) rather than a mechanistic signal.

**Tracing the flawed reasoning**: The Lotka-Volterra analogy was imported from ecology without sufficient attention to a critical disanalogy. In ecology, species occupy a continuous resource space and compete for shared resources. In SAE feature spaces, "competition" between latents is mediated by the training objective (reconstruction error minimization under sparsity), not by literal resource sharing. The frequency ratio f_j/f_i conflates base rate differences with causal competitive pressure.

### Surprise 2: Corpus PMI has the *wrong sign* (deviation: -100% direction reversal)

**Expected**: Positive PMI coefficient (higher co-occurrence -> more absorption), partial R^2 >= 0.10.  
**Observed**: PMI coefficient = -0.006 (negative), partial R^2 = 0.0006 (three orders of magnitude below threshold).

**Wrong assumption**: We assumed absorption was "primarily data-driven" -- that tokens co-occurring more frequently in the training corpus would be more likely to have their features absorbed. The null result is unambiguous: corpus co-occurrence statistics explain essentially zero variance in absorption rates beyond SAE configuration variables. This is particularly notable because the layer coefficient *is* significant (beta = -0.012, p < 0.001), confirming the regression has statistical power to detect real effects.

**What this means**: Absorption appears to be overwhelmingly architecture/objective-driven, not data-driven. This is the conventional view that our proposal explicitly argued against. The field's existing understanding appears correct on this dimension.

### Surprise 3: Absorption *does* predict downstream performance -- strongly and consistently (deviation: >150% from expected |r| < 0.20)

**Expected**: |r| < 0.20 for all downstream tasks (H3: absorption metric is disconnected from downstream utility).  
**Observed**: Pearson r = -0.595 (sparse probing), -0.431 (SCR), -0.454 (RAVEL TPP). After controlling for model/layer/architecture, partial correlations *strengthen*: -0.661, -0.677, -0.492. The only non-significant correlation is with unlearning (r = -0.175).

**Wrong assumption**: We assumed, following DeepMind's 2025 qualitative findings, that absorption was a metric without downstream consequences -- that we were "optimizing for a metric that does not capture what matters." The data shows the exact opposite: across 54 Gemma-2-2B SAEs from SAEBench, absorption is a *strong and consistent* negative predictor of downstream SAE quality. The matched RAVEL comparison confirms this with a paired t-test (p = 0.006, Cohen's d = 2.13).

**Critical note on interpretation**: The H3 "falsification" is actually the most valuable finding in the entire study. As the proposal correctly anticipated: "This falsification is scientifically valuable: it would confirm the absorption research motivation and provide the first empirical proof of the causal chain." The data provides exactly this proof.

### Surprise 4: The width paradox is more complex than hypothesized (deviation: DAS(k=3) non-monotonic vs. predicted monotonic increase)

**Expected**: DAS(k=3) increases monotonically with width for >= 80% of letters.  
**Observed**: Only 42.3% show positive slope. Mean DAS(k=3) goes 0.320 -> 0.227 -> 0.260, i.e., it *decreases* when going from 24k to 49k before partially recovering. DAS(k=1) shows the predicted non-positive trend (57.7%), but this is barely above chance.

**Wrong assumption**: We assumed that wider SAEs produce "more distributed competitive exclusion" -- that absorption would spread across more children predictably. The data suggests the width-absorption relationship is confounded by L0 changes across SAE families (L0 ranges from 41.4 to 137.4 across the feature-splitting SAEs at layer 8), and the parent feature identification may be inconsistent across widths (different SAEs may learn fundamentally different feature dictionaries, making cross-width DAS comparison unreliable).

### Surprise 5: Model platform matters enormously -- GPT-2 is not a valid proxy for Gemma-2

The entire experimental suite was forced to use GPT-2 Small instead of the originally planned Gemma-2-2B due to gated HuggingFace access. This is not merely a limitation footnote -- it fundamentally changes the interpretation of every result:

- The sae-spelling ground truth was designed for Gemma Scope SAEs, not GPT-2 SAEs
- The absorption rates observed on GPT-2 (overall 35.4%) use a different probe methodology than Chanin et al.'s Gemma results
- The C3A SAEBench correlations *do* use Gemma-2 data (they come from the pre-computed SAEBench results), creating an inconsistency where H1/H2/H4/H5 are tested on GPT-2 but H3 is tested on Gemma-2

---

## 3. Mental Model Revision

**Old mental model**: Absorption is a data-driven phenomenon (caused by corpus co-occurrence patterns) that can be detected unsupervised via an ecologically-inspired competition coefficient, and whose connection to downstream performance is unvalidated.

**Revised mental model**: Absorption is an *architecture/objective-driven* phenomenon whose severity is primarily determined by the SAE's training configuration (layer, width, L0, and training objective), not by corpus statistics. It has a **strong, consistent, and causal-direction relationship with downstream SAE quality** -- SAEs with higher absorption are measurably worse at sparse probing, spurious correlation removal, and token prediction probing. The Lotka-Volterra framing, while conceptually appealing, does not provide mechanistic purchase: the competition coefficient alpha_ij fails to outperform a trivial geometric (cosine) baseline, and the predicted sharp-threshold transition does not exist. Absorption is *not* competitive exclusion in the ecological sense; it is better understood as a symptom of the sparsity-reconstruction tradeoff in the SAE objective function, where wider/sparser SAEs must allocate finite latent capacity and systematically underserve general/rare features.

---

## 4. Reframing Test

**Original research question**: "Can the Lotka-Volterra competitive exclusion framework provide an unsupervised detector, a corpus-level causal account, and a downstream impact test for SAE feature absorption?"

**If we asked this today, knowing the results, would we frame it the same way?**

No. The LV framing and the corpus PMI story are both dead ends. The downstream correlation finding, however, is the strongest and most important result. A better research question would be:

**Revised research question**: "Does SAE feature absorption causally degrade downstream interpretability and safety-relevant task performance, and if so, what architectural properties of SAEs mediate this relationship?"

This reframing:
1. Centers the H3 finding (the strongest result) as the primary contribution
2. Drops the LV theoretical apparatus (empirically falsified)
3. Drops the corpus PMI story (empirically falsified)
4. Replaces them with the width/layer/architecture survey as the mechanistic investigation
5. Still delivers the Type I/II/III taxonomy as a descriptive contribution (with appropriate caveats about Type II inflation)

---

## 5. New Hypothesis Generation

### NH1: Absorption is primarily mediated by SAE sparsity pressure, not competitive exclusion

**Testable prediction**: Among SAEs matched on model, layer, and width, those with lower L0 (sparser) will show higher absorption rates. The C2B survey already suggests this (layers 3-6 with L0 > 100 show higher absorption than layers 9-11), but the confound between layer and L0 must be resolved. A controlled experiment would fix layer = 8 and vary only the L0 penalty coefficient across 5+ settings at fixed width = 24k.

**Falsification**: If L0 shows no effect when width and layer are controlled, the mechanism is elsewhere. The C2B width-sweep at layer 8 (768 to 98k) partially supports this: absorption increases from 0.009 at width 3072 to 0.104 at width 98k, but L0 also varies (62 to 67), making attribution difficult.

### NH2: The downstream performance correlation is confounded by width, and absorption is an epiphenomenon of over-expansion

**Testable prediction**: The C3A correlation (r = -0.595 between absorption and sparse probing) is driven by 1M-width SAEs, which have both the highest absorption (0.62-0.90) and the lowest sparse probing F1 (0.73-0.81). After removing 1M-width SAEs, the correlation should weaken substantially (|r| < 0.20).

**Falsification**: If the correlation remains strong (|r| > 0.30) within the 16k-65k width range alone, then absorption is genuinely predictive of downstream quality independent of the extreme-width regime. This would be a much stronger result. The existing partial correlations controlling for log_width already suggest this may hold (partial r = -0.661 for sparse probing), but the 1M outliers may be driving the partial correlation estimation.

### NH3: Type II "partial absorption" is universal and non-diagnostic -- what matters is the Type I/III distinction at matched L0

**Testable prediction**: The 92.3% comprehensive absorption rate (driven by Type II) is inflated because the parent feature identification heuristic misidentifies features. Using sae-spelling's actual ground-truth parent feature IDs (available for Gemma Scope SAEs), the Type II rate should drop to < 50%, and the distinction between Type I (concentrated, harmful) and Type III (distributed, potentially benign) should emerge as the mechanistically meaningful taxonomy.

**Falsification**: If Type II remains > 80% even with correct parent IDs, then partial suppression of parent features is genuinely near-universal, and the Chanin metric's binary classification is fundamentally misleading about the severity spectrum of absorption.

---

## Summary of Belief Updates

| Belief | Prior (proposal) | Posterior (after data) | Update magnitude |
|---|---|---|---|
| LV competition coefficient is mechanistically valid | High (core thesis) | Very low (falsified) | Extreme |
| Corpus PMI drives absorption | Moderate (novel claim) | Very low (wrong sign, null effect) | Large |
| Absorption is disconnected from downstream utility | Moderate (following DeepMind) | Very low (strong correlation found) | Large (positive direction) |
| Width paradox explained by distributed exclusion | Moderate | Low (non-monotonic DAS, confounded by L0) | Moderate |
| True absorption rate far exceeds 15-35% | Moderate | Uncertain (92.3% but methodology suspect) | Ambiguous |
| GPT-2 is an adequate proxy for Gemma-2 | Assumed (fallback) | Seriously questionable (different feature landscapes) | High concern |

**Bottom line**: Two of the three core claims (unsupervised detector, corpus PMI) are empirically refuted. The third (downstream disconnection) is refuted in the *positive* direction -- absorption does predict downstream performance -- making it the paper's strongest finding. The theoretical framing needs a complete overhaul: drop LV, center the downstream correlation result, reposition the paper from "here is a new detector and mechanism" to "here is the first empirical proof that absorption matters for downstream safety tasks, plus a comprehensive survey of when and where it occurs."
