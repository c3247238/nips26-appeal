# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

**1. Assumption**: Absorbed features can be identified via activation patching and show "genuine" causal recovery.

- **Evidence challenging it**: The 67.3% mean recovery in activation patching could reflect legitimate feature decomposition rather than absorption. When a child feature like "letter A at word start" is zeroed, the parent recovers — but this is exactly what we'd expect if the features were properly decomposed, not absorbed. The recovery metric conflates "absorption exists" with "features are related."
- **Source**: pilot_activation_patching results (iter_004)

**2. Assumption**: CV > 1.0 is a meaningful threshold for predicting steering effectiveness.

- **Evidence challenging it**: The CV threshold of 1.0 was chosen post-hoc after observing pilot data. This is p-hacking. Without prospective validation on held-out features, the predictive value of CV cannot be claimed.
- **Source**: novelty_report.json, cand_cv_actionability weaknesses

**3. Assumption**: "Robust absorbed" vs "fragile absorbed" decomposition explains steering heterogeneity.

- **Evidence challenging it**: The bypass vs. mediated regime routing hypothesis is post-hoc theorizing. No independent evidence corroborates that high-CV features route through different channels than low-CV features. The proposed mechanism (context-sensitive vs stable child channels) is speculation.
- **Source**: cand_cv_actionability mechanistic hypothesis

**4. Assumption**: High-CV steering effect (0.153) is practically significant.

- **Evidence challenging it**: 0.153 logit change is still a small effect. In absolute terms, this is marginal. Basu et al. showed 98.2% AUROC detection failed to produce ANY measurable steering — but their steering magnitude was also presumably small. The comparison isn't apples-to-apples.
- **Source**: Basu et al. (2026), pilot_steering_cv results

**5. Assumption**: The actionability paradox is domain-specific (clinical = low-CV, non-clinical = high-CV) rather than universal.

- **Evidence challenging it**: This is a convenient reframing to avoid direct collision with Basu et al. But there's no evidence that clinical features are predominantly low-CV. The "domain-specificity" claim is unfalsifiable — if steering fails in any domain, one can always claim that domain has "wrong" feature type distribution.
- **Source**: idea_context.md, actionability paradox framing

**6. Assumption**: Absorbed features with high CV preserve "context-sensitive specialized information."

- **Evidence challenging it**: An alternative explanation: high CV in absorbed features reflects noise amplification from suppression, not preservation of specialized information. When parent activation is suppressed, residual signal through child channels becomes noisy. This explains the 733x CV ratio without invoking "specialized information preservation."
- **Source**: H4 alternative mechanism in idea_context.md

---

## Landscape of Doubt

The front-runner proposal ("CV Predicts Steering Heterogeneity Within Absorbed SAE Features") rests on several assumptions that deserve scrutiny:

1. **The "genuine absorption" claim**: 67.3% activation recovery is taken as evidence of "genuine" absorption, but recovery of parent features when child features are zeroed is exactly what we'd expect from ANY hierarchical feature decomposition. The distinction between "absorption" (pathology) and "legitimate feature specialization" (expected behavior) is not operationally defined.

2. **The CV-steering correlation**: 0.153 vs 0.075 is a 2x ratio, but both are small absolute effects. The field's critical question is "does absorption measurement help predict what we can steer?" Even if CV predicts steering, the practical utility of steering 0.153 logit change is unclear.

3. **The mechanism story**: "Bypass vs mediated regime routing" is compelling narrative but lacks independent evidence. Without mechanistic validation, this is just-so story telling.

4. **The Basu et al. collision management**: Framing Basu et al. as "domain-specific" is unfalsifiable. If steering fails in any domain, claim that domain has low-CV features. This is not a scientific hypothesis — it's ad-hoc protection of the claim.

5. **The 9-word sample**: The activation patching validation uses 9 persistent core words. This is insufficient for generalization. Statistical power for detecting heterogeneity is low.

---

## Phase 2: Initial Candidates

### Candidate A: Feature Decomposition vs. Absorption — Is the Distinction Real?

- **Challenged assumption**: The 67.3% activation recovery demonstrates "genuine absorption" rather than legitimate feature decomposition.
- **Evidence against**: When we zero a child feature ("letter A at word start") and the parent recovers, this is consistent with BOTH (a) absorption pathology and (b) proper hierarchical feature specialization. The metric cannot distinguish them.
- **Contrarian hypothesis**: The features identified as "absorbed" may simply be properly decomposed hierarchical features — the "absorption" interpretation is assumed, not demonstrated.
- **Exploitation plan**: Design a control experiment: if absorption is real, features from unrelated hierarchies should NOT show recovery when a child is zeroed. If absorption is just decomposition, ALL hierarchical features will show similar recovery patterns.
- **Novelty estimate**: 6/10 (requires rethinking what "absorption" means operationally)

### Candidate B: The CV-Steering Correlation Is a Magnitude Artifact

- **Challenged assumption**: CV predicts steering effectiveness independent of activation magnitude.
- **Evidence against**: High-CV features might simply have larger activation variance, which correlates with larger decoder magnitudes, which produce larger steering effects. The Fano factor normalization may not fully account for this.
- **Contrarian hypothesis**: The 2x steering effect (0.153 vs 0.075) is a decoder magnitude artifact, not a genuine CV-based predictor.
- **Exploitation plan**: Control for decoder magnitude more strictly. If high-CV and low-CV groups have matched decoder magnitudes, does the steering difference persist?
- **Novelty estimate**: 5/10 (requires better controls)

### Candidate C: The Actionability Paradox Is Real and Universal

- **Challenged assumption**: The actionability paradox can be explained by domain-specific sampling (clinical = low-CV) and does not represent universal failure.
- **Evidence against**: This "domain-specificity" claim is unfalsifiable. It's an ad-hoc protection of the hypothesis. If steering fails in a new domain, proponents can always claim that domain has the wrong feature type distribution.
- **Contrarian hypothesis**: The Basu et al. result reflects something fundamental: internal feature detection (even 98.2% AUROC) does not guarantee we can intervene on model behavior. The 2x steering effect observed here is too small to overturn this conclusion.
- **Exploitation plan**: Test whether ANY intervention magnitude can produce reliable behavioral changes using absorbed features. If not, the actionability paradox is real regardless of CV.
- **Novelty estimate**: 7/10 (tests a fundamental question)

---

## Phase 3: Self-Critique

### Against Candidate A: Feature Decomposition vs. Absorption

- **Steelman**: The literature (Chanin et al., 2024) explicitly defines absorption as a pathological state where parent features are subsumed by child features during sparse optimization. The activation patching result (67.3% recovery) is exactly what they'd predict for absorbed features. The recovery is too high to be explained by random co-occurrence.
- **Cherry-picking check**: We selected 9 "persistent core words" that showed up consistently. If we tested random words, would we still see 67% recovery? The selection criterion (persistence across runs) biases toward features with strong hierarchical relationships.
- **Confounding check**: If the SAE trained on data where hierarchical features co-occur frequently, we'd expect hierarchical decomposition even without pathology. The recovery could reflect co-occurrence statistics, not absorption.
- **Actionability check**: Even if our interpretation is wrong (it's decomposition, not absorption), the practical implications are the same — absorbed/decomposed features can be partially recovered via activation patching.
- **Verdict**: WEAK — The interpretation (absorption vs decomposition) is ambiguous, but the practical finding (recovery is possible) still holds.

### Against Candidate B: CV-Steering Correlation as Magnitude Artifact

- **Steelman**: Fano factor (CV²/mean) explicitly controls for magnitude. The pilot experiment used Fano factor to select features. So the steering difference is NOT simply a magnitude artifact.
- **Cherry-picking check**: 30 high-CV vs 30 low-CV is a reasonable sample, but the CV calculation used the SAME data that computed steering effects. The threshold (1.0) was chosen post-hoc.
- **Confounding check**: Is there a third variable (e.g., feature frequency, decoder sparsity) that correlates with both CV and steering effectiveness?
- **Actionability check**: Even if CV is just a proxy for magnitude, having a simple statistical predictor (CV) is still useful for practitioners. The "true mechanism" doesn't matter for practical utility.
- **Verdict**: MODERATE — The Fano factor control addresses the main concern, but the post-hoc threshold selection remains problematic.

### Against Candidate C: Actionability Paradox Is Universal

- **Steelman**: Basu et al. (2026) showed 98.2% AUROC detection but ZERO steering in clinical domain. This is not a small effect — it's a complete failure. The 0.153 steering effect here is orders of magnitude smaller than what would constitute meaningful behavioral control.
- **Cherry-picking check**: We only tested 60 features (30 high-CV, 30 low-CV). The Basu et al. study was more comprehensive. Small-sample positive results are more likely to be noise.
- **Confounding check**: The clinical vs. non-clinical domain distinction is post-hoc. Basu et al. didn't claim their result was universal — they just reported it. The generalization to "actionability paradox is universal" is also a stretch.
- **Actionability check**: If actionability paradox IS universal, then SAE-based interpretability has a fundamental limitation. This would be important negative evidence.
- **Verdict**: STRONG — The evidence for universal actionability failure is strong. The 0.153 vs 0.075 difference is too small to overturn the paradox. The "domain-specificity" reframe is ad-hoc.

---

## Phase 4: Refinement

### Dropped Candidates

- **Candidate A (Feature Decomposition vs. Absorption)**: The ambiguity in interpretation doesn't change practical recommendations. Even if "absorption" is mislabeled, the phenomenon is real and the findings stand.

### Strengthened Candidates

- **Candidate B (CV as Magnitude Artifact)**: The Fano factor control addresses the main concern, but the post-hoc threshold problem remains. We can strengthen this by validating CV threshold on a held-out feature set prospectively.

- **Candidate C (Actionability Paradox Is Real)**: This gains strength from the Basu et al. result being a complete failure (0% steering) vs. our "positive" result being a small effect (0.153). The disparity in magnitudes is telling.

### Selected Front-Runner

**The contrarian front-runner is Candidate C (Actionability Paradox Is Real)** — not because it's the most positive story, but because it's the most important question and the evidence is strongest.

**The key insight**: The 2x CV-based difference in steering (0.153 vs 0.075) does NOT overturn the actionability paradox. Basu et al. showed COMPLETE failure (0% steering). Our result shows a small effect that may not generalize or be practically useful.

**What this means for the proposal**: The framing should change from "CV predicts which absorbed features can be steered" to "Even with CV-based decomposition, steering effects remain small — revisiting the actionability paradox." This is more honest and more publishable as a negative/complexifying result.

---

## Phase 5: Final Proposal

# Rethinking SAE Actionability: Small Effects and the Limits of Internal Feature Detection

### Challenged Assumption

The field assumes that if we can measure absorption (and find feature heterogeneity), we can meaningfully steer model behavior. The Basu et al. "actionability paradox" is assumed to be (a) domain-specific or (b) resolvable via better feature selection (e.g., CV-based filtering).

### Evidence

**For the assumption (mainstream view)**:
- Pilot steering: High-CV features show 2x larger steering (0.153 vs 0.075)
- Activation patching: 67.3% mean recovery when child is zeroed
- CV distinguishes absorbed features (CV=7.33) from non-absorbed (CV=0.01)

**Against the assumption (contrarian view)**:
- Basu et al. (2026): 98.2% AUROC but 0% steering in clinical domain
- The 0.153 steering effect is small in absolute terms
- The "domain-specificity" reframe is ad-hoc and unfalsifiable
- 9-word sample is too small for generalization

### Hypothesis

The actionability paradox is NOT domain-specific — it reflects a fundamental limitation: **internal feature detection (even when highly accurate) does not guarantee behavioral steerability via SAE interventions**. The CV-based decomposition identifies features with slightly larger steering effects, but the absolute magnitude remains small and may not be practically useful.

**Alternative interpretation**: The small steering effects observed in both Basu et al. (clinical) and our work (non-clinical LLM) suggest a common cause: the residual stream compensates for SAE-based interventions through parallel pathways.

### Method

1. **Magnitude comparison**: Compare absolute steering magnitudes across studies (ours: 0.153, Basu et al.: ~0). If Basu et al. measured similar magnitudes, the "domain-specificity" reframe collapses.

2. **Prospective CV validation**: Split features into train/test sets BEFORE computing CV. Validate that CV > 1.0 predicts steering on held-out features. This addresses the p-hacking concern.

3. **Behavioral relevance test**: Instead of logit change, measure actual behavioral changes (task accuracy, generation quality). If steering doesn't change behavior, logit changes are academic.

4. **Compensation detection**: Test whether the model routes around SAE steering via parallel pathways. If so, no amount of feature selection will solve the actionability paradox.

### Experimental Plan

| Experiment | Duration | Purpose |
|------------|----------|---------|
| E1: Cross-study magnitude comparison | 15 min | Compare absolute steering magnitudes across Basu et al. and this work |
| E2: Prospective CV validation | 30 min | Split features into train/test, validate CV threshold prospectively |
| E3: Behavioral steering test | 45 min | Measure actual task accuracy change from steering, not just logit change |
| E4: Compensation pathway analysis | 45 min | Test whether residual stream routes around SAE steering |

**Total**: ~135 min (may need to split across iterations)

### Baselines

- Basu et al. (2026): 98.2% AUROC, 0% steering (clinical)
- This work: CV-based decomposition, small steering effects (non-clinical LLM)
- No-steering baseline: Task accuracy without SAE intervention

### Risk Assessment

**Risk**: The actionability paradox IS universal, and CV-based selection doesn't solve it.
**Likelihood**: High — the evidence from Basu et al. is strong.
**Implication**: SAE-based interpretability has fundamental limitations. The field needs to reconsider what "actionability" means.

**Risk**: CV threshold is not predictive on held-out features.
**Likelihood**: Medium — threshold was chosen post-hoc.
**Implication**: The predictor is artifact, not signal.

**Risk**: Behavioral steering produces no measurable change even with logit effects.
**Likelihood**: Medium — small logit changes may not translate to behavior.
**Implication**: The practical utility of SAE steering is even more limited than thought.

### Novelty Claim

This work provides the **first systematic comparison of steering magnitudes across Basu et al. and non-clinical LLM domains**, establishing that the actionability paradox reflects absolute magnitude limitations, not domain-specific feature distributions. It also provides **the first prospective validation of CV as a steering predictor**, addressing the p-hacking concern in the current proposal.

**Key insight**: A 2x improvement from 0.075 to 0.153 is not meaningful if both are small. The field should focus on absolute effect sizes, not relative improvements.

### Revised Framing

Instead of "CV predicts steering heterogeneity within absorbed features" (which overclaims), use:

**"Rethinking the Actionability Paradox: Steering Magnitudes and the Limits of Internal Feature Detection in SAEs"**

This framing:
1. Accepts Basu et al. as a serious constraint, not an edge case
2. Reports the CV-based decomposition as a nuanced finding (small but real effect)
3. Focuses on the fundamental question: what can SAE steering actually accomplish?
4. Is publishable regardless of outcome (positive: small effect found, negative: no practical utility)

---

## Contrarian Summary

The current proposal's strength is that it confronts the actionability paradox directly. Its weakness is that it overclaims: a 2x improvement (0.153 vs 0.075) does not overturn a universal paradox when the baseline is near-zero. The contrarian position is that the actionability paradox is REAL and UNIVERSAL, and the CV-based decomposition is a small nuance within that constraint. This is more honest, more publishable (as a negative/complexifying result), and more useful for the field.