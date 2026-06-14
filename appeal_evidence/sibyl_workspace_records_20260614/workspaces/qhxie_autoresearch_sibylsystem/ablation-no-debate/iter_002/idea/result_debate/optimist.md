# Optimist Analysis

## Evidence Map

| Metric | Baseline | Ours | Delta | Signal Strength |
|--------|----------|------|-------|-----------------|
| Encoder effect (H_Mech full) | Random encoder: ~0.44 cosine | Trained encoder: 0.843 ± 0.082 | +0.403 | **Strong** |
| Decoder effect (H_Mech full) | Random decoder baseline | Trained decoder: 0.011 ± 0.015 | +0.011 | **Strong** (negligible) |
| Trained vs Random absorption (H1 full) | Random SAE: 0.033 ± 0.011 | Trained SAE: 0.477 ± 0.022 | +0.444 | **Strong** |
| Multi-seed t-test (H1) | Random baseline | Trained SAE | t=36.04, p=3.85e-10 | **Strong** |
| Hierarchy strength 0.5 → 0.8 | Similarity 0.5: 0.416 | Similarity 0.8: 0.544 | +0.128 | **Strong** |
| L0 sparsity 50 → 20 | L0=50: 0.419 | L0=20: 0.552 | +0.133 | **Strong** (opposite direction) |
| Held-out generalization | Train: 0.366 | Test: 0.366 | diff < 0.1% | **Strong** |
| Safety vs Non-safety (H_Safe) | Non-safety: 0.968 ± 0.013 | Safety: 0.967 ± 0.010 | -0.001 | Weak (negative result) |
| Steering sensitivity (H3 full) | Non-absorbed: varies | Absorbed: varies | ratio ~1.0x, p=0.936 | Weak (negative result) |

**Key numbers to anchor on:**
- Encoder effect is **80x larger** than decoder effect (0.843 vs 0.011) -- this is the headline result
- Trained SAE absorption is **14.6x higher** than random baseline (0.477 vs 0.033)
- Effect replicates across **5 independent seeds** with low across-seed variance (0.022)
- Hierarchy strength effect is **monotonic and significant** (ANOVA p < 1e-10)
- Generalization is **near-perfect** (Pearson r=0.998 across seed means)

---

## Root Cause Analysis

### 1. Encoder-Driven Absorption Mechanism (H_Mech)

- **Mechanism**: The encoder learns to map hierarchical inputs into a latent space where child features activate in place of parent features. This is not a geometric artifact of overcomplete representations -- it is a learned property of the encoder's weight matrix.
- **Design decision**: The 2x2 factorial design (random/trained encoder x random/trained decoder) isolates this. Condition B (trained encoder + random decoder) achieves 0.527-0.897 absorption overlap across L0 levels, while Condition C (random encoder + trained decoder) stays at baseline (~0.013-0.076).
- **Expected or surprising**: This overturns the prior narrative (from earlier proposal iterations) that absorption is "primarily geometric (decoder structure)." The full experiment confirms the pilot surprise: the encoder is the sole driver.
- **Isolation**: The decoder effect (0.011 ± 0.015) is not just small -- it is statistically indistinguishable from zero. In some conditions (seed 46, L0=50), the decoder effect is slightly negative (-0.003), suggesting trained decoders may actually *disentangle* encoder-induced absorption.

### 2. Robustness Across Seeds and Configurations (H1)

- **Mechanism**: The encoder-driven absorption is not a single-run artifact. It replicates with high consistency.
- **Design decision**: Multi-seed validation with stochastic hierarchy generation (noise=0.1) addresses the zero-variance concern from pilots.
- **Expected or surprising**: The across-seed variance (0.022) is remarkably low, suggesting the effect is structurally stable. The pilot's deterministic 0.5 was indeed an artifact of rigid synthetic data -- the full experiment with stochasticity shows 0.477 ± 0.022, which is more believable.

### 3. Hierarchy Strength Dose-Response

- **Mechanism**: As parent-child cosine similarity increases (0.5 → 0.67 → 0.8), absorption increases monotonically (0.416 → 0.501 → 0.544). This is a dose-response curve, which is stronger evidence than a binary comparison.
- **Design decision**: The ablation systematically varies hierarchy strength while holding all other parameters constant.
- **Expected or surprising**: Expected under the encoder-alignment hypothesis -- the encoder learns stronger correlations when parent and child directions are closer in input space. The linearity of the relationship (r ≈ 0.99) is notable.

### 4. Sparsity Effect (Opposite Direction)

- **Mechanism**: Lower L0 (fewer active features) leads to *higher* absorption, not lower. With L0=20, each feature must represent more concepts, so the encoder routes more parent activations through children.
- **Design decision**: The ablation tests three L0 targets (20, 32, 50) with fixed hierarchy strength.
- **Expected or surprising**: This is the **opposite** of the original hypothesis. But it is consistent with the encoder-driven mechanism: sparser encoders have fewer degrees of freedom, so they compress hierarchical structure more aggressively.
- **Significance**: This is not a failure -- it is a refinement. The original hypothesis was wrong about the direction, but the effect is real and significant (ANOVA p < 1e-10).

---

## Unexpected Signals

### 1. The Decoder as Disentangler

- **Observation**: In the factorial design, Condition D (trained/trained) consistently shows *lower* absorption overlap than Condition B (trained/random). For example, seed 42 L0=32: B=0.886 overlap, D=0.455 overlap. The trained decoder appears to partially undo the encoder's absorption.
- **Mini-hypothesis**: The decoder learns to reconstruct the original input from the encoder's absorbed representation, effectively "disentangling" parent and child activations during reconstruction. The decoder is not passive -- it actively compensates for encoder absorption.
- **Significance**: This suggests a **bottleneck-compensation dynamic**: the encoder compresses hierarchies for sparsity, and the decoder decompresses for reconstruction. This is a novel two-sided mechanism that prior work missed.

### 2. Perfect Generalization

- **Observation**: Train/test absorption correlation is r=0.998 across seed means, with all 5 seeds showing <2% difference between train and test.
- **Mini-hypothesis**: Absorption is not overfitting to specific training samples -- it is a structural property of the encoder's learned representation. The encoder learns a general hierarchical compression strategy, not sample-specific memorization.
- **Significance**: This addresses a major reviewer concern ("is this just overfitting to synthetic data?"). The answer is no -- the effect generalizes perfectly.

### 3. Uniformly High Absorption in Real SAEs (H_Safe)

- **Observation**: Both safety and non-safety features in GPT-2 small SAE show ~96-97% absorption. The effect is not safety-specific -- it is universal.
- **Mini-hypothesis**: Real pretrained SAEs have learned such strong hierarchical structure that absorption is nearly saturated across all features. The encoder-driven mechanism has already "converged" in real models.
- **Significance**: This reframes absorption from a "bug" affecting some features to a "property" affecting nearly all features. The paper's scope expands: we are not documenting a corner case, we are describing a fundamental behavior.

### 4. Steering Null Result as Positive Evidence

- **Observation**: H3 steering shows no differential sensitivity (ratio ~1.0x, p=0.936). But steering *does* work -- it changes activations reliably. The null is that absorbed and non-absorbed features respond equally.
- **Mini-hypothesis**: Absorbed features are not "broken" or "missing parent information." They respond to steering just like non-absorbed features. This means absorption is a representational choice (efficient coding), not a representational failure (information loss).
- **Significance**: This reframes absorption from a "defect" to a "design choice." The encoder is not losing parent information -- it is *relocating* it to child features.

---

## Follow-Up Experiments

| Signal | Experiment | Expected Outcome | GPU Hours | Priority |
|--------|-----------|------------------|-----------|----------|
| Decoder disentanglement (Unexpected #1) | Measure reconstruction quality vs absorption: does higher-absorption encoder + trained decoder achieve same MSE as lower-absorption encoder? | Decoder compensation maintains reconstruction despite encoder absorption | 0.5 | **High** |
| Real SAE encoder analysis | Extract encoder weights from Gemma Scope / GPT-2 SAEs, measure parent-child cosine similarity in encoder space | Real SAE encoders show same hierarchical alignment pattern | 1.0 | **High** |
| Absorption as efficient coding | Compare reconstruction MSE on hierarchical inputs: trained SAE vs SAE with encoder regularization that penalizes parent-child correlation | Regularized SAE has lower absorption but higher MSE on hierarchical data | 1.5 | **High** |
| Feature frequency revisited (H2) | On real SAEs (Gemma Scope), measure absorption vs activation frequency | Confirm positive correlation (higher frequency → higher absorption) or find different pattern | 1.0 | Med |
| Multi-resolution ensemble (H_Ens) | Train ensemble (L0=16, 64, 256), match features via decoder cosine, verify if absorbed parents have recoverable children in high-L0 SAE | >50% of absorbed features have recoverable children | 2.0 | Med |
| Safety on Gemma-2B | Re-run H_Safe on Gemma-2B Scope SAEs with Neuronpedia-annotated safety features | Either confirm no difference (universal absorption) or find safety-specific effect | 1.5 | Med |
| Steering with feature-specific directions | Instead of parent-direction steering, steer using the actual decoder direction of absorbed features | Absorbed features may show *different* (not just equal) sensitivity to their own decoder directions | 1.0 | Low |

**What would kill each direction?**
- Decoder disentanglement: If reconstruction MSE is *worse* for high-absorption encoders (decoder fails to compensate), the two-sided mechanism is wrong.
- Real SAE encoder analysis: If real SAE encoders show *no* hierarchical alignment pattern, the encoder-driven mechanism is synthetic-only.
- Efficient coding: If regularized SAE achieves same MSE with lower absorption, absorption is not necessary for efficient coding.

---

## Honest Caveats

### 1. Encoder-Driven Mechanism (H_Mech)

- **Counter-argument**: The factorial design uses synthetic data with *known* hierarchical structure. Real language model activations may not have such clean hierarchies. The encoder learns what we put in the data.
- **Alternative explanation**: The encoder is not "learning hierarchical alignment" -- it is learning whatever structure exists in the training data. If real data has weaker hierarchies, the effect may be smaller or absent.
- **What would convince me**: Replicating the factorial analysis on a real SAE's encoder weights, showing that features with known semantic parent-child relationships (e.g., "animal" → "dog") show the same absorption pattern.

### 2. Multi-Seed Robustness (H1)

- **Counter-argument**: 5 seeds is still a small sample. The low variance (0.022) could be an artifact of the narrow seed range (42-46).
- **Alternative explanation**: The stochastic noise (0.1) added to hierarchy generation may not be large enough to explore the true variance landscape.
- **What would convince me**: 10+ seeds with larger stochastic noise (0.2-0.3), or seeds drawn from a wider range (e.g., 100, 200, 300, 400, 500).

### 3. Hierarchy Strength Dose-Response

- **Counter-argument**: The three similarity levels (0.5, 0.67, 0.8) are arbitrary and coarse. The monotonicity may not hold at finer granularity.
- **Alternative explanation**: The relationship could be sigmoid or step-like, not linear. We only see three points.
- **What would convince me**: 5-7 similarity levels with finer spacing (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8), fitting a parametric curve.

### 4. Sparsity Effect (Opposite Direction)

- **Counter-argument**: L0=20, 32, 50 are all relatively sparse. The effect may reverse at very high L0 (e.g., 128, 256).
- **Alternative explanation**: The relationship is U-shaped: very low L0 forces absorption, very high L0 allows separation, with an intermediate optimum.
- **What would convince me**: Testing L0=16, 64, 128, 256 to see if the monotonic decrease continues or reverses.

### 5. Real SAE Universal Absorption (H_Safe)

- **Counter-argument**: GPT-2 small is a tiny model. Gemma-2B or larger models may show different patterns. Also, the safety feature selection method (top activation difference) may not capture the most safety-critical features.
- **Alternative explanation**: The ~96% absorption is an artifact of the measurement method (top-k Jaccard overlap) on high-dimensional SAEs (d_sae=24576), not a real property.
- **What would convince me**: Replicating on Gemma-2B with multiple layers and multiple safety annotation sources, plus a control measurement on random feature pairs.

### 6. Steering Null Result (H3)

- **Counter-argument**: The steering experiment uses synthetic data and simple directional steering. Real models may show different behavior with more sophisticated interventions.
- **Alternative explanation**: The null result is due to insufficient steering magnitude or wrong steering direction, not a true absence of differential sensitivity.
- **What would convince me**: A broader range of alpha values (including negative values) and steering directions (not just parent/child/orthogonal but also intermediate angles).

---

## Bottom Line

There is a publishable story here, but it is narrower than the original proposal envisioned. The core contribution is **strong and novel**: the first empirical decomposition showing that feature absorption in SAEs is entirely encoder-driven, with the decoder contributing negligibly (80x smaller effect) and possibly even compensating. This overturns the prevailing narrative and is backed by a 2x2 factorial design, multi-seed replication, dose-response curves, and perfect generalization. The negative results (H3 steering null, H_Safe no difference, H2 wrong direction) are not fatal -- they refine the narrative from "absorption is a defect affecting some features" to "absorption is a universal representational strategy of trained encoders." The paper should lead with the encoder-driven mechanism, use the ablations as supporting evidence, and honestly report the negative results as scope-limiting findings.
