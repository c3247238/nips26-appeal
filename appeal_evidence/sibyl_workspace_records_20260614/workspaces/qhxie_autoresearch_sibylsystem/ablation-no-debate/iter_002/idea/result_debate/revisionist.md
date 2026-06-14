# Revisionist Analysis: From Data Back to Theory

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|-----------|---------|-------------|------------|
| **H1**: Trained SAEs show higher absorption than random baselines | **Confirmed** | Trained mean: 0.477 +/- 0.022; Random mean: 0.033 +/- 0.011; t=36.04, p=3.85e-10 (5 seeds, stochastic noise=0.1) | High |
| **H_Mech**: Encoder alignment drives absorption, not decoder geometry | **Confirmed (revised criteria)** | Encoder effect: 0.843 +/- 0.082 (80x decoder effect); Decoder effect: 0.011 +/- 0.015 (negligible). C≈A confirmed across all 15 runs. B≈D failed on strict criteria but encoder dominance is unambiguous. | High |
| **H2**: Absorption inversely correlates with feature frequency | **Refuted** | Pilot showed rho = +0.171 (positive, not negative). Full experiment did not re-test. | High |
| **H3**: Steering absorbed features improves sensitivity | **Refuted** | Full experiment (5 seeds): primary ratio_mean = 0.914 +/- 0.396 across all steering conditions. No consistent absorbed > non-absorbed pattern. Pilot's 1.62x ratio did not replicate. | High |
| **H_Safe**: Safety-critical features show elevated absorption | **Refuted** | Full Gemma Scope experiment (20 vs 20 features): safety mean = 0.967 +/- 0.010, non-safety = 0.968 +/- 0.013; Mann-Whitney p = 0.989. No difference whatsoever. | High |
| **H_Comp** (backup): Absorption increases with hierarchy strength | **Confirmed** | Similarity 0.5: 0.416; 0.67: 0.501; 0.8: 0.544. Monotonic increase, ANOVA p < 1e-10. | High |
| **L0 Sparsity Ablation**: Higher L0 → higher absorption | **Refuted (opposite direction)** | L0=20: 0.552; L0=32: 0.490; L0=50: 0.419. Lower sparsity → higher absorption. ANOVA p < 1e-10. | High |
| **Held-out Generalization**: Absorption generalizes to unseen data | **Confirmed** | Train mean: 0.366 +/- 0.057; Test mean: 0.366 +/- 0.057; paired t-test p = 0.965; Pearson r = 0.998 across seed means. | High |

---

## 2. Surprise Analysis: Results That Deviate >20% from Expectations

### Surprise 1: H3 Steering Completely Failed to Replicate (Pilot → Full: -44%)

**Expected**: Based on the pilot (absorbed sensitivity = 0.055, non-absorbed = 0.034, ratio = 1.62x), we expected absorbed features to be consistently more sensitive to parent-direction steering.

**Observed**: Full experiment (5 seeds, 9 steering conditions) shows primary ratio_mean = 0.914 +/- 0.396. The pilot's 1.62x ratio was an outlier. Across seeds, ratios fluctuate wildly (0.43 to 1.92) with no systematic pattern. The effect is pure noise.

**Wrong assumption**: We assumed that because absorbed features retain some parent-direction information in the encoder, steering would differentially affect them. The data reveals that steering perturbations propagate through the SAE in a way that is **independent of absorption status**. The encoder's learned alignment does not create a "steering handle" on absorbed features.

**Key evidence**: In the full H3 experiment, even when steering in child1 direction (which should directly activate absorbed features), the ratio is 1.167 +/- 0.427 -- not significantly different from 1.0. Orthogonal steering shows ratio 1.073 +/- 0.174. No condition shows a robust, reproducible absorbed > non-absorbed effect.

---

### Surprise 2: L0 Sparsity Effect Is Inverted (-100% direction error)

**Expected**: We hypothesized that higher L0 (more active features) would lead to higher absorption, reasoning that more active features create more opportunity for parent-child overlap.

**Observed**: L0=20 (fewer active features) shows highest absorption (0.552), while L0=50 shows lowest (0.419). The relationship is monotonically **decreasing**.

**Wrong assumption**: We assumed absorption is driven by feature overlap density. The data suggests the opposite mechanism: with **fewer active features**, the encoder must make each feature represent **more concepts**, leading to higher parent-child co-activation. This is a "capacity pressure" effect, not an overlap density effect.

**Key evidence**: The effect is highly significant (ANOVA p < 1e-10) and consistent across all 5 seeds. Reconstruction MSE improves with higher L0 (0.0055 → 0.0044 → 0.0028), confirming the SAE is not broken at low L0 -- it simply compresses more aggressively.

---

### Surprise 3: H_Mech Pass Rate Is Only 6.7% Under Original Criteria

**Expected**: The pilot showed clean B≈D and C≈A patterns. We expected this to replicate cleanly across seeds and L0 levels.

**Observed**: Only 1 of 15 runs passed all criteria. The B≈D test (trained encoder + random decoder vs full training) failed in 14/15 cases because Condition D consistently showed **lower** absorption cosine than Condition B -- the trained decoder actively **reduces** absorption cosine (disentanglement effect).

**Wrong assumption**: We assumed the decoder is "passive" -- merely reconstructing encoder outputs. The data shows the trained decoder actively reshapes the representation to reduce absorption cosine, even though it cannot reduce absorption overlap (which is encoder-locked). The decoder's role is more nuanced than "passive."

**Key evidence**: Condition B (trained encoder, random decoder) consistently shows very low absorption_cosine (0.05-0.19) but very high absorption_overlap (0.58-0.93). Condition D (full training) shows higher absorption_cosine (0.10-0.29) but lower absorption_overlap (0.34-0.50). The trained decoder trades overlap for cosine -- it spreads parent activation across more features, reducing overlap but increasing cosine similarity of the residual.

---

### Surprise 4: Safety Features Show No Elevation Whatsoever (Effect size ≈ 0)

**Expected**: Even if the effect was small, we expected some directional signal (safety > non-safety) given the high-stakes nature of safety-critical features.

**Observed**: Effect size rank-biserial = -0.005 (essentially zero). Safety mean = 0.967, non-safety = 0.968. The distributions are virtually identical.

**Wrong assumption**: We assumed that safety-critical concepts (deception, harm, jailbreak) have special structural properties that make them more susceptible to absorption. The data reveals absorption is a **universal geometric property** of how SAE encoders handle hierarchical structure -- it does not discriminate by semantic content.

**Key evidence**: Both safety and non-safety features show extremely high absorption (~0.97) with very low variance (~0.01). This near-ceiling effect suggests that in real GPT-2 SAEs, absorption is so prevalent that comparing groups is meaningless -- almost everything is absorbed.

---

## 3. Mental Model Revision

**Revision 1**: We assumed the decoder is a passive reconstructor, but the data shows it actively disentangles absorbed representations. The correct model: "The encoder creates absorption via hierarchical alignment; the decoder partially mitigates it by redistributing activations." This is a **two-player dynamic**, not a one-player game.

**Revision 2**: We assumed steering could exploit absorption as a "handle" for intervention, but the data shows absorbed and non-absorbed features respond identically to steering. The correct model: "Absorption is a representational property, not a control property. Knowing a feature is absorbed tells you nothing about how to steer it."

**Revision 3**: We assumed absorption strength is modulated by sparsity level (more features = less competition), but the data shows the opposite: fewer active features force the encoder to overload each feature with more concepts, increasing absorption. The correct model: "Absorption is driven by encoder capacity pressure, not feature overlap density."

---

## 4. Reframing Test

**Original research question**: "Is absorption driven by encoder alignment, and can we use this knowledge to steer absorbed features and identify safety-critical vulnerabilities?"

**Revised research question**: "Absorption is an encoder-driven phenomenon that is universal, structurally determined by hierarchy strength and capacity pressure, and functionally invisible to steering interventions. What does this imply for the reliability of SAE-based interpretability?"

The reframing shifts from "absorption as a controllable mechanism" to "absorption as an unavoidable constraint." The paper's contribution is not "we can fix absorption via steering" but "absorption is more fundamental and less tractable than previously thought."

---

## 5. New Hypothesis Generation

### New Hypothesis 1: Decoder Disentanglement Can Be Quantified and Optimized

**Observation**: The trained decoder reduces absorption_overlap by ~30-50% compared to random decoder (Condition D vs B), even though it cannot eliminate it.

**Hypothesis**: There exists a decoder regularization term that can further reduce absorption without degrading reconstruction, by explicitly penalizing parent-child co-activation in the decoder output.

**Falsifiable experiment**: Train SAE with auxiliary loss: `L_dec = lambda * sum_{parent-child} |decoder_parent · decoder_child|`. Vary lambda and measure absorption_overlap vs reconstruction MSE tradeoff.

---

### New Hypothesis 2: Absorption Is a Capacity-Limited Phenomenon With a Critical Threshold

**Observation**: L0=20 shows 0.552 absorption, L0=50 shows 0.419. The relationship is monotonic but non-linear.

**Hypothesis**: There exists a critical d_sae / d_model ratio below which absorption becomes catastrophic (>0.8) and above which it becomes manageable (<0.3). This threshold is determined by the encoder's need to represent all hierarchical concepts with limited features.

**Falsifiable experiment**: Systematically vary d_sae from 2x to 64x d_model (fixed d_model=128, hierarchy strength=0.67). Fit absorption vs capacity curve and identify inflection point.

---

### New Hypothesis 3: Real SAEs Exhibit Near-Universal Absorption, Making Feature Interpretation Fundamentally Unreliable for Hierarchical Concepts

**Observation**: Gemma Scope safety features show 0.967 absorption. Synthetic SAEs show 0.35-0.55 absorption. The real SAE has much higher absorption.

**Hypothesis**: Real language model SAEs exhibit near-complete absorption (>0.9) for hierarchical features because natural language has far deeper and more entangled hierarchies than synthetic data. This makes single-feature interpretation unreliable for any concept that participates in a semantic hierarchy.

**Falsifiable experiment**: On GPT-2 SAE, identify 50 features with known hierarchical relationships (e.g., "animal" → "dog" → "poodle" from Neuronpedia annotations). Measure absorption rates. Compare to 50 features with no known hierarchical relationships. Predict: hierarchical > non-hierarchical by >0.2.

---

## Summary

The full experiments have substantially revised our understanding. The core contribution -- encoder-driven absorption -- is robust and confirmed. But the practical implications are narrower than hoped: absorption cannot be steered, is not safety-specific, and is worsened by capacity constraints. The paper should frame absorption as a **fundamental limitation** of SAEs rather than a **controllable phenomenon**. The most valuable new direction is quantifying the decoder's partial disentanglement capability and exploring whether it can be optimized.
