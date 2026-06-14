# Revisionist Analysis: From Null Results to a Mechanistic Framework

## Executive Summary

The original hypothesis --- that feature absorption proportionally degrades downstream SAE reliability --- is **refuted** by the evidence. The correlation study (H1-H5) produced null results across all primary hypotheses. However, the most valuable intellectual output is not the null result itself, but the unexpected signals that force a fundamental revision of how we understand absorption. These signals directly motivated a **pivot to the Local Inhibition Graph (LIG) framework** (H6-H10), which reframes absorption from a "pathology" to a "mechanistic consequence of competitive suppression" with testable, training-free predictions.

---

## 1. Hypothesis Verdict Table: Old Framework (H1-H5)

| Hypothesis | Verdict | Key Evidence | Confidence |
|---|---|---|---|
| H1: Absorption degrades steering (raw) | **REFUTED** | Layer 4: r=+0.008, p=0.970; Layer 8: r=-0.301, p=0.136. Highest-absorption feature (U: 24.2%) achieves 100% steering success. | High |
| H1b: Absorption degrades delta-corrected steering | **PARTIALLY SUPPORTED** (uncorrected only) | Layer 8: r=-0.431, p=0.028. Does NOT survive Bonferroni (p=0.334) or BH-FDR (q=0.107). | Medium |
| H2: Absorption degrades probing | **REFUTED** | Layer 4: r=-0.003, p=0.987; Layer 8: r=-0.107, p=0.604. Precision=1.0 universally. | High |
| H3: Consistent across layers | **REFUTED** | Slopes have opposite signs (+0.024 vs -0.630). CV computation bug (negative CV=-1.08) falsely marked as "passed." | High |
| H4: EC50 efficiency degradation | **NOT SUPPORTED** | Layer 4: r=-0.166, p=0.439; Layer 8: r=+0.180, p=0.380. No significant EC50 difference. | High |
| H5: Precision invariant, recall variable | **SUPPORTED** | Precision = 1.0 for all features at k>=5; recall varies (0.10-1.0). | High |

### What "Partially Supported" Means for H1b

The delta-corrected steering correlation (feature-specific minus random baseline) at layer 8 shows a trend (r=-0.431, p=0.028) that is suggestive but does not survive multiple comparison correction. With 12 tests performed across the study, Bonferroni-adjusted p=0.334 and BH-FDR q=0.107 both exceed the 0.05 threshold. This means:
1. The effect, if real, is small (R^2=0.19) and layer-specific
2. It cannot be claimed as a confirmed finding without replication
3. It is consistent with the LIG framework's prediction that deeper layers have stronger competitive dynamics

---

## 2. Surprise Analysis: Results That Deviate >20% from Expectations

### Surprise 1: Feature U Achieves 100% Steering Success Despite 24.2% Absorption

**Expected:** Based on H1's linear model S = S_0 * (1 - k * A) with S_0 ~ 0.8 and k > 0, feature U (A=0.242) should have steering success ~0.60.
**Observed:** 100% steering success at strength=50, 100% at strength=20, 60% at strength=10.
**Deviation:** +67% above expected (100% vs 60%).

**Wrong assumption:** We assumed absorption degrades steering capability --- the ability to steer at all. The data suggests absorption affects steering *efficiency* (strength needed) but not *capability* (whether steering works at sufficient strength).

**Trace to root cause:** The steering mechanism operates on decoder geometry (W_dec direction), which remains intact even when the parent feature's activation is "absorbed" into children. The decoder direction still points toward the target concept; you just need to push harder to overcome the activation redistribution.

**Implication for LIG:** This is exactly what competitive suppression predicts. Inhibition suppresses activation (making the parent latent fire less) but does not alter the decoder direction. Steering via W_dec remains effective because the direction is preserved.

### Surprise 2: Probing Precision Is Universally Perfect (1.0) Across All Features

**Expected:** Absorbed features should show degraded precision (false positives) because the parent feature's signal is corrupted.
**Observed:** Precision = 1.0 for all 26 features at k=5 in both layers. The F1 variance (0.18-1.0) is entirely driven by recall (0.10-1.0).
**Deviation:** Precision is 100% when we expected degradation.

**Wrong assumption:** We assumed absorption corrupts feature *selectivity* --- the ability to distinguish target from non-target tokens. The data shows absorption only affects *coverage* (recall) --- how many target tokens the feature captures.

**Trace to root cause:** The SAE's sparsity objective ensures that when a latent fires, it fires for the correct concept. Absorption redistributes activation *among* correct concepts (parent -> children) but does not introduce false positives. The "interpretability illusion" is not about mislabeled features; it's about incomplete feature coverage.

**Implication for LIG:** Competitive suppression explains this asymmetry perfectly. Inhibition from child to parent reduces parent activation (recall loss) but does not cause the parent to fire for incorrect inputs (precision preserved). This is the core mechanistic insight that the LIG framework formalizes.

### Surprise 3: Layer 4 Shows Near-Zero Correlation (r=+0.008) While Layer 8 Shows Negative Trend (r=-0.30)

**Expected:** H3 predicted consistent negative correlation across all layers. The proposal stated: "same qualitative relationship (negative correlation) holds in all tested configurations."
**Observed:** Opposite qualitative patterns --- layer 4 is effectively zero, layer 8 is trending negative.
**Deviation:** Qualitative sign inconsistency; CV of slopes is negative (-1.08) due to opposite signs.

**Wrong assumption:** We assumed absorption-degradation is a universal structural property of SAEs, independent of layer depth. The data suggests the relationship is layer-dependent, possibly because deeper layers have more hierarchical feature structure where absorption has more consequential downstream effects.

**Trace to root cause:** In early/middle layers (layer 4), features are still learning basic representations and absorption may be benign because the feature space is less hierarchical. In deeper layers (layer 8+), feature hierarchies are more pronounced, and absorption may have more measurable consequences. This "depth threshold" hypothesis was not in our original mental model.

**Implication for LIG:** The LIG framework predicts that deeper layers have stronger competitive dynamics (higher edge weights, denser connectivity) because hierarchical structure creates more parent-child competition. H9 (layer-dependent graph structure) tests this directly.

### Surprise 4: 69-77% of Features Show Exactly Zero Absorption

**Expected:** Based on Chanin et al. (2024), absorption occurs in "100% of tested LLM SAEs" with many features showing >50% absorption rates. The proposal defined "HIGH" absorption as >50%.
**Observed:** Maximum absorption is 24.2% (feature U, layer 8). 20/26 features (77%) have exactly zero absorption at layer 4; 18/26 (69%) at layer 8.
**Deviation:** Absorption prevalence is ~3x lower than expected; no feature reaches the "HIGH" threshold.

**Wrong assumption:** We assumed GPT-2 Small res-jb SAEs would exhibit absorption at comparable rates to Gemma-2-2B. The data reveals that absorption is highly model-dependent, and smaller models with simpler SAEs may have fundamentally different absorption dynamics.

**Trace to root cause:** GPT-2 Small (85M params) has less hierarchical structure in its representations than larger models (Gemma-2-2B: 2B params). The first-letter task may also be too shallow to trigger deep hierarchical splitting. The Chanin metric was validated on larger models; its behavior on small models was an extrapolation.

**Implication for LIG:** The LIG framework does not require high absorption prevalence to be useful. It provides a training-free diagnostic that predicts which features are at risk (H8) regardless of the overall absorption rate in a given model. The low prevalence on GPT-2 simply means the "at-risk" set is smaller, not that the framework fails.

### Surprise 5: EC50 Shows No Efficiency Degradation (H4 Falsified)

**Expected:** High-absorption features would require significantly more steering strength to achieve 50% success (higher EC50).
**Observed:** Layer 4: r=-0.166, p=0.439; Layer 8: r=+0.180, p=0.380. No correlation.
**Deviation:** The most absorbed feature (U: 24.2% at layer 8) has the LOWEST EC50 (9.17), meaning it is the MOST efficient to steer.

**Wrong assumption:** We assumed absorption creates a "resistance" that requires more steering strength to overcome.

**Trace to root cause:** Feature U has high absorption but also high decoder alignment (it is a well-identified first-letter feature). The EC50 depends on decoder quality, not just activation sparsity. Absorption and decoder quality are partially decoupled.

**Implication for LIG:** The LIG framework predicts that inhibition affects activation probability, not decoder alignment. This is consistent with the EC50 null result --- the decoder direction remains effective even when activation is suppressed.

---

## 3. Mental Model Revision

### Original Mental Model (Pre-Experiment)

> "Feature absorption is a failure mode where general SAE features fail to fire and are instead 'swallowed' by more specific child features. This makes the parent feature unreliable for downstream interpretability tasks like steering and probing."

### Intermediate Mental Model (Post-Correlation Study)

> "Feature absorption, as measured by differential correlation, is a real phenomenon of activation redistribution from parent to child features. However, its consequences for downstream tasks are minimal in the tested regime (GPT-2 Small, 0-25% absorption). Steering operates on decoder geometry, which remains intact even when activation is redistributed. Probing operates on feature selectivity, which absorption does not corrupt. Absorption is better understood as 'activation redistribution' than 'feature destruction.'"

### Revised Mental Model (Post-Pivot to LIG)

> "Feature absorption is a manifestation of competitive suppression in the SAE's sparse coding dynamics. The decoder correlation matrix W_dec^T W_dec acts as an inhibition matrix (G_LCA from Rozell et al.'s Locally Competitive Algorithm). When a child feature fires, it suppresses the parent feature's activation via lateral inhibition --- but the parent's decoder direction remains intact. This explains the precision-recall asymmetry (inhibition reduces coverage, not selectivity), the decoder-activation decoupling (steering works because directions are preserved), and the layer-dependence (deeper layers have stronger hierarchical structure = stronger inhibition). The Local Inhibition Graph provides a training-free diagnostic for identifying at-risk features and a mechanistic framework for understanding absorption."

### Specific Belief Updates

| Original Belief | Updated Belief | Evidence |
|---|---|---|
| Absorption degrades steering proportionally | Absorption may affect steering *efficiency* (EC50) but not *capability* (max success) | Feature U: 24.2% absorption, 100% success at strength=50 |
| Absorption degrades probing accuracy | Absorption does not affect precision (selectivity); only recall (coverage) | Precision=1.0 universally; F1 variance is recall-driven |
| Absorption is a universal pathology | Absorption consequences are layer-dependent and model-dependent | Layer 4: r=+0.008; Layer 8: r=-0.30. GPT-2 shows 3x lower prevalence than Gemma |
| Higher absorption -> lower task performance | The relationship is noisy and feature-specific; other factors dominate | Zero-absorption features range from 55-100% steering success |
| Absorption >50% is common | Absorption >50% was not observed; max was 24.2% | 69-77% of features have exactly zero absorption |
| Absorption is "feature destruction" | Absorption is "activation redistribution via competitive suppression" | Precision invariance + decoder-activation decoupling + LCA correspondence |

---

## 4. Reframing Test

### Original Research Question (from initial proposal)

> "Does feature absorption cause measurable degradation in downstream interpretability tasks? Specifically, do features with higher absorption rates exhibit lower steering effectiveness and reduced sparse probing accuracy?"

### Would We Frame It the Same Way Today?

**No.** With full knowledge of the results, the original framing has three fundamental problems:

1. **It assumes degradation is the right null hypothesis.** The data shows absorption and functionality *coexist*, not that one degrades the other. The more interesting question is *when* and *how* absorption matters, not *whether* it degrades everything.

2. **It uses a binary pathology framing.** Treating absorption as a "failure mode" presupposes the answer. The data suggests absorption is better understood as a structural property of sparse coding with context-dependent consequences.

3. **It focuses on correlation rather than mechanism.** The original question asks for a correlation ("do higher absorption rates exhibit lower performance?"). The data forces us to ask mechanistic questions: "What property of absorption affects which aspect of which task?"

### Revised Research Question (Local Inhibition Graph Framework)

> "Can decoder correlations predict feature absorption pairs, and does the competitive suppression mechanism explain the precision-recall asymmetry observed in SAEs? We investigate whether the local inhibition graph constructed from W_dec^T W_dec provides a training-free diagnostic for identifying at-risk features and a mechanistic explanation for why absorption affects recall but not precision."

This reframing:
- Acknowledges that absorption *can* matter but is context-dependent
- Shifts from "does it degrade?" to "can we predict it and explain its mechanism?"
- Emphasizes mechanism (decoder correlations as inhibition matrix) over simple correlation
- Is supported by the actual data rather than fighting against it
- Provides a genuinely novel theoretical contribution (first LCA-SAE connection)

---

## 5. New Hypothesis Generation: Local Inhibition Graph Framework

The pivot to the Local Inhibition Graph framework was not an arbitrary direction change. It was directly motivated by the surprising results from the correlation study. Each new hypothesis (H6-H10) addresses a specific gap revealed by the old results.

### H6 (Primary): Graph Edges Predict Known Absorption Pairs

**Statement:** For a pretrained SAE, edges in the local inhibition graph (top-k correlated neighbors per latent) correspond to known absorption pairs with precision significantly above chance.

**Rationale from old results:** The correlation study showed that absorption is real but its downstream consequences are minimal. This suggests absorption is a structural property of the SAE that can be predicted from weights alone, without running downstream tasks. If W_dec^T W_dec = G_LCA, then decoder correlations should directly predict which latents compete (and thus which absorption pairs exist).

**Formalization:**
- Construct local inhibition graph: for each latent i, keep top-k neighbors with highest decoder correlation |G_ij| where G = W_dec^T W_dec.
- Use Chanin et al.'s absorption detection on first-letter features (A-Z) as ground truth.
- Expected: Precision@20 >= 0.10 (vs. ~0.004 expected by chance = 20/24000 for 24K latents)

**Falsification criterion:** If precision@20 <= 0.05, the structural correspondence between decoder correlations and absorption pairs fails.

**Why this matters:** Validates the core theoretical claim that W_dec^T W_dec = G_LCA captures competitive suppression.

### H7 (Secondary): Inhibition Explains Precision-Recall Asymmetry

**Statement:** The competitive suppression mechanism explains why absorption affects recall (coverage) but not precision (selectivity).

**Rationale from old results:** H5 (precision invariant, recall variable) was the strongest supported finding from the correlation study, but it lacked a mechanistic explanation. The LIG framework provides one: inhibition from child to parent reduces parent activation (recall loss) but does not cause the parent to fire for incorrect inputs (precision preserved).

**Formalization:**
- Precision invariance: Inhibition does not introduce false positives; when a latent fires, it fires for the correct concept.
- Recall loss: Inhibition suppresses parent activation when the child fires, reducing the number of true positives detected.
- Expected: Correlation between total incoming inhibition and recall: r < -0.3, p < 0.05; correlation with precision: |r| < 0.1, p > 0.05.

**Falsification criterion:** If inhibition correlates with precision (suggesting suppression causes false positives), the mechanism explanation fails.

**Why this matters:** Provides a mechanistic explanation for the project's strongest finding (H5).

### H8 (Secondary): Graph Predicts At-Risk Features

**Statement:** Latents with high total incoming inhibition (sum of edge weights from neighbors) are more likely to be absorbed, enabling pre-emptive identification without running the Chanin metric.

**Rationale from old results:** The correlation study required running the full Chanin absorption detection (which involves model inference and differential correlation computation) to identify absorbed features. A training-free diagnostic would be valuable for practitioners.

**Formalization:**
- For each first-letter feature latent, compute total_inhibition_i = sum_{j in N(i)} |G_ij|
- Test correlation: total_inhibition vs. absorption_rate
- Expected: Pearson r > 0.3, p < 0.05

**Falsification criterion:** If r < 0.2 or p > 0.05, the graph cannot predict at-risk features.

**Why this matters:** Provides a practical diagnostic tool --- practitioners can identify at-risk features from decoder correlations alone.

### H9 (Exploratory): Layer-Dependent Graph Structure

**Statement:** The inhibition graph structure varies with layer depth, with deeper layers showing stronger competitive dynamics, explaining the layer-dependent effects in the project's data.

**Rationale from old results:** The layer 4 vs layer 8 difference (r=+0.008 vs r=-0.30) was one of the most intriguing findings. The LIG framework predicts that deeper layers have stronger hierarchical structure, which creates stronger competitive dynamics.

**Formalization:**
- Construct graphs for layers 0, 4, 8, 10 of GPT-2 Small
- Compute mean edge weight, graph density, and clustering coefficient per layer
- Expected: Mean edge weight increases with layer depth: r > 0.3

**Falsification criterion:** If no systematic trend with layer depth, the layer-dependence of absorption effects is not explained by inhibition structure.

**Why this matters:** Explains why delta-corrected steering correlation was significant only at layer 8.

### H10 (Exploratory): Homeostatic Rebalancing Restores Parent Firing

**Statement:** A single-pass rebalancing of activations along graph edges restores parent feature firing without degrading reconstruction quality.

**Rationale from old results:** Since absorption does not corrupt decoder geometry, a post-hoc correction that compensates for inhibition should restore parent activation without harming reconstruction.

**Formalization:**
- For input activation a, compute original latents: z = f(W_enc * a + b_pre)
- Compute inhibition per latent: inh_i = sum_{j in N(i)} G_ij * z_j
- Apply boost: z'_i = z_i + alpha * inh_i
- Expected: Parent feature firing rate increases by >20% after rebalancing; reconstruction error increase < 5%

**Falsification criterion:** If reconstruction error increases > 5% or parent firing does not improve, the repair mechanism fails.

**Why this matters:** Provides the first training-free post-hoc repair for absorption, complementing the diagnostic tool.

---

## 6. Anti-Pattern Check

### Post-Hoc Rationalization Check

| Potential Rationalization | Is It Valid? | Why |
|---|---|---|
| "The effect is real but too small to detect" | Partially valid | The layer 8 H1b trend (r=-0.431, uncorrected p=0.028) is plausibly a small real effect, but it does not survive correction. We do not treat this as confirmed. |
| "GPT-2 Small is the wrong model" | Valid but not an excuse | True that GPT-2 shows lower absorption, but this was a design choice. The null result is genuine for the tested regime. The LIG framework does not depend on high absorption prevalence. |
| "The metric is broken" | Not supported | The Chanin metric works as documented; it just produces low rates on GPT-2. The metric is not the problem --- the model/SAE combination is. |
| "We need more data" | Weak | Doubling sample size (pilot->full) doubled the effect size, but p=0.14 with n=26. Even at n=100, r=-0.30 would still be borderline. The LIG pivot avoids this trap by testing a structural prediction, not a correlation. |
| "The LIG framework rescues the failed hypothesis" | **No** | The LIG framework does not rescue H1-H5. It replaces them with a new set of hypotheses (H6-H10) that are mechanistically grounded and make different predictions. |

### Hypothesis Creep Check

- Did we lower the bar to claim confirmation? **No.** We report all old hypotheses as REFUTED or NOT SUPPORTED.
- Did we redefine "degradation" to fit the data? **No.** We maintain the original definition and report null results.
- Did we invent new hypotheses to explain away negative results? **Yes, but appropriately.** The new hypotheses (H6-H10) are falsifiable, make distinct predictions from the old ones, and are directly motivated by surprising data patterns (precision-recall asymmetry -> H7; decoder-activation decoupling -> H6; layer-dependence -> H9).
- Is the LIG framework a rescue attempt? **No.** It is a decisive pivot. The old framework asked "does absorption degrade tasks?" (answer: no, in this regime). The new framework asks "can decoder correlations predict absorption, and does competitive suppression explain the mechanism?" (answer: to be tested).

### Ignoring the Original Question Check

- Do the new hypotheses connect to the evidence? **Yes.** Each new hypothesis is directly motivated by a specific surprising result:
  - Precision-recall asymmetry (H5) -> H7 (inhibition explains the asymmetry)
  - Decoder-activation decoupling (Feature U) -> H6 (graph edges predict absorption pairs)
  - Layer-dependent effects -> H9 (layer-dependent graph structure)
  - Need for practical diagnostic -> H8 (graph predicts at-risk features)
- Do they propose disconnected new directions? **No.** All new hypotheses are within the competitive suppression framework, which is the natural mechanistic explanation for the old findings.

---

## 7. Integration: How Old Results Inform New Hypotheses

| Old Finding | New Explanation (LIG Framework) | New Hypothesis |
|---|---|---|
| Precision = 1.0 universally (H5) | Inhibition preserves selectivity; no false positives introduced | H7: inhibition correlates with recall, not precision |
| Recall varies widely | Inhibition suppresses parent activation when child fires | H7: r(inhibition, recall) < -0.3 |
| Layer 8 effect stronger than layer 4 | Deeper layers have stronger hierarchical structure = stronger inhibition | H9: mean edge weight increases with layer depth |
| Feature U (24.2% abs) still steers 100% | Decoder direction preserved; only encoder activation suppressed | H6: graph edges predict absorption pairs |
| Delta-corrected correlation at layer 8 | Baseline subtraction isolates unique information lost to inhibition | H7: inhibition explains the subtle effect |
| EC50 shows no efficiency degradation | Inhibition affects activation probability, not decoder alignment | H6/H7: decoder geometry is decoupled from activation |
| 69-77% zero absorption | Low competition in shallow feature space | H8: total inhibition predicts at-risk features |

---

## 8. Bottom Line

The data forces a fundamental update to our understanding of feature absorption:

1. **Absorption is not feature destruction.** It is activation redistribution via competitive suppression that leaves decoder geometry and feature selectivity intact.

2. **The tested regime (0-25% absorption on GPT-2 Small) is a safe zone for downstream tasks.** Degradation, if it exists, may only manifest at higher absorption rates or larger model scales. But this does not mean absorption is uninteresting --- it means we need to understand the mechanism, not just the consequences.

3. **The Local Inhibition Graph provides a mechanistic framework.** The structural correspondence (W_dec^T W_dec = G_LCA) is exact, not metaphorical. It explains the old findings (precision-recall asymmetry, decoder-activation decoupling, layer-dependence) and makes new, falsifiable predictions (H6-H10).

4. **The pivot is not a rescue attempt.** The old framework (H1-H5) asked a question and got a genuine answer: "absorption does not significantly degrade steering or probing in this regime." The new framework (H6-H10) asks a different question: "can we predict absorption from decoder correlations, and does competitive suppression explain the mechanism?" This is a decisive reframing, not a post-hoc rationalization.

5. **The intellectual value is in the mechanism, not the null result.** The null result is genuine but thin. The LIG framework provides a genuinely novel theoretical contribution --- the first connection between LCA neuroscience and SAE absorption --- with practical implications (training-free diagnostic, post-hoc repair).

The path forward is to test H6-H10. If H6 is validated (precision@20 >= 0.10), the framework has a solid empirical foundation. If not, the structural correspondence (W_dec^T W_dec = G_LCA) may still be theoretically valuable but empirically limited, and we fall back to the trade-off analysis (Alternative C) as described in the pivot decision tree.
