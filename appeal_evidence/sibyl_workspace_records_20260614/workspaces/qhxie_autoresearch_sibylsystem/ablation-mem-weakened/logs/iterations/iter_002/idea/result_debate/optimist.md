# Optimist Analysis: Local Inhibition Graph Framework

## Context

This analysis evaluates the **Local Inhibition Graph** proposal (Candidate F, front-runner) against the empirical evidence accumulated across iterations 1-8. The project has pivoted from a failed correlation study (cand_a, result debate score: 3/10) to a neuroscience-inspired theoretical framework. This optimist analysis asks: what evidence supports this pivot, and what signals suggest the new direction will succeed?

---

## Evidence Map: What the Data Actually Shows

### Prior Empirical Findings (Iterations 1-8)

| Metric | Value | Signal Strength | Relevance to Inhibition Graph |
|--------|-------|-----------------|------------------------------|
| H1b delta-corrected steering (L8, Pearson) | r = -0.431, p = 0.028 | **Moderate** | Supports competitive suppression at deeper layers |
| H1b delta-corrected steering (L8, Spearman) | rho = -0.502, p = 0.009 | **Moderate** | Monotonic non-linear relationship suggests threshold effect |
| Random baseline validation (L4) | Feature-specific = 0.799 vs random = 0.344, d = 1.26 | **Strong** | Validates that decoder directions capture real structure |
| Random baseline validation (L8) | Feature-specific = 0.857 vs random = 0.379, d = 1.18 | **Strong** | Same validation at layer where H1b is significant |
| H5 precision invariance (L4, k=5) | Precision std = 0.054, Recall std = 0.199 | **Strong** | Exactly what competitive suppression predicts |
| H5 precision invariance (L8, k=5) | Precision std = 0.028, Recall std = 0.192 | **Strong** | Precision even more invariant at deeper layer |
| Precision = 1.0 count (L4, k=5) | 21/26 features | **Strong** | Selectivity preserved under absorption |
| Precision = 1.0 count (L8, k=5) | 25/26 features | **Strong** | Near-universal selectivity at deeper layer |
| Feature U steering robustness | 24.2% absorption, 100% success at s=50 | **Strong** | Decoder geometry intact; only activation suppressed |
| EC50 no efficiency degradation | p = 0.23 (L4), p = 0.44 (L8) | **Moderate** | Inhibition affects activation probability, not decoder alignment |
| Layer-dependence of H1b | Significant at L8, not L4 | **Moderate** | Deeper layers = stronger hierarchical structure = stronger inhibition |
| H1 raw steering (L8) | r = -0.301, p = 0.136 | Weak | Trending negative but non-significant |
| H2 probing F1 | r = -0.107 (L8), p = 0.604 | None | Probing measures selectivity, not coverage |

### Novelty Verification (External Evidence)

| Claim | Search Result | Confidence |
|-------|--------------|------------|
| No prior LCA-SAE connection | "NO MATCHES" across 4 search queries | **High** |
| No prior inhibition graph for SAEs | "NO MATCHES" for "inhibition graph" + SAE | **High** |
| No prior training-free repair | Matryoshka/OrtSAE/ATM all require retraining | **High** |
| Rozell LCA (2008) | ~2000 citations, zero LLM SAE applications | **High** |

**Signal strength definitions:**
- **Strong**: Clear effect with p < 0.05, consistent across measures, or effect size d > 0.8
- **Moderate**: Effect reaches p < 0.05 uncorrected but not after MCP, or effect size in medium range
- **Weak**: Trending but not significant (p < 0.20)
- **None**: No detectable effect

---

## Root Cause Analysis: Why the Inhibition Graph Framework Is Supported

### 1. The Mathematical Correspondence Is Exact, Not Metaphorical

**Mechanism:** In the Locally Competitive Algorithm (Rozell et al., 2008), the dynamics are:
```
tau * du/dt = -u + (a - G*u)_+
```
where G is the inhibition matrix. For SAEs with tied encoder/decoder, the encoder is W_enc = W_dec^T, and the reconstruction is a_hat = W_dec * z. The decoder correlation matrix W_dec^T W_dec appears naturally in the encoder computation. This is not an analogy --- it is an exact structural identity: **W_dec^T W_dec = G_LCA**.

**Design decision:** The Interdisciplinary perspective identified this correspondence during the ideation round. It was not post-hoc rationalization of null results --- it was identified as a candidate direction before the result debate recommended pivoting.

**Expected or surprising:** Expected in hindsight, but genuinely novel. The LCA framework has been around for 17 years with ~2000 citations, yet no one has connected it to SAEs. This is a genuine gap, not a forced connection.

**Evidence strength:** The correspondence is mathematically derivable from first principles. No empirical validation is needed to establish the structural identity. What needs validation is whether this structural identity has explanatory power for absorption.

### 2. Precision Invariance (H5) Is Exactly What Competitive Suppression Predicts

**Mechanism:** In competitive suppression, child features inhibit parent features. When the parent feature's concept is present, the child fires instead of the parent. The child still fires *correctly* --- it does not fire for unrelated inputs. This means:
- **Precision preserved**: When a latent fires, it fires for the right reason (no false positives)
- **Recall reduced**: The parent fails to fire when the child fires instead (true positives missed)

**Design decision:** The precision-recall decomposition was added after the pilot when probing F1 showed no correlation with absorption. This was a methodological improvement, not part of the original proposal.

**Expected or surprising:** Surprising at the time, but exactly predicted by the inhibition framework. The data shows precision std = 0.028-0.054 vs recall std = 0.192-0.199 --- a 3.7-6.9x difference. This is not noise; it is a systematic pattern.

**Why this matters:** If absorption had corrupted precision, the SAE would be introducing mislabeled features --- a much more serious failure. The invariance of precision means the SAE's semantic structure remains intact. The inhibition framework explains WHY: competitive suppression does not create false positives; it only suppresses true positives.

### 3. Delta-Corrected Steering (H1b) Shows the Signal That Raw Metrics Miss

**Mechanism:** Raw steering metrics conflate two effects: (1) generic decoder-direction steering (any SAE latent direction biases the model toward letter-related tokens) and (2) feature-specific steering (the latent direction specifically activates the target feature). The random baseline subtraction removes effect (1), isolating effect (2). The residual (effect 2) degrades with absorption.

**Design decision:** The random baseline was added after the pilot showed null results. This was the single most important methodological addition.

**Expected or surprising:** Both. The existence of a significant correlation after baseline correction validates the core intuition. But the layer-dependence (significant only at L8, not L4) was unexpected.

**Inhibition explanation:** Deeper layers have stronger hierarchical feature composition. At layer 4, features are relatively flat; at layer 8, parent-child hierarchies are well-developed. Stronger hierarchies mean stronger competitive suppression, which means stronger absorption consequences. This explains the layer-dependence naturally.

**Evidence:**
- L8: r = -0.431 (Pearson), rho = -0.502 (Spearman), both p < 0.05 uncorrected
- L4: r = +0.245 (Pearson), rho = +0.231 (Spearman), both p > 0.05
- Spearman > Pearson at L8 suggests monotonic but non-linear relationship --- consistent with threshold model

### 4. Steering Robustness (Feature U: 24.2% absorption, 100% success) Is Explained by Decoder-Activation Decoupling

**Mechanism:** Steering injects activation along the decoder direction. Even if the natural encoder activation of the parent is suppressed by child features, the injected activation is strong enough to overcome this suppression. The decoder direction itself is unchanged --- only the encoder's natural activation pattern is affected.

**Inhibition explanation:** Competitive suppression operates on the encoder side (which latents fire), not the decoder side (what direction they represent). The inhibition graph predicts: decoder directions are preserved; only activation probabilities are redistributed. This is exactly what the data shows.

**Evidence:** Feature U (highest absorption: 24.2%) achieves 100% steering success at strength=50. The range of success rates for zero-absorption features at strength=50 is 55-100% --- Feature U is at the top of this range, not the bottom.

### 5. EC50 Null Result Is Consistent with Inhibition Mechanism

**Mechanism:** EC50 measures the steering strength needed to reach 50% success. If absorption affected decoder geometry, high-absorption features would need more strength. If absorption only affects encoder activation, the decoder direction is unchanged and EC50 should be similar.

**Inhibition explanation:** Competitive suppression affects activation probability, not decoder alignment. The EC50 null result (p = 0.23 at L4, p = 0.44 at L8) is not a failure --- it is a prediction of the inhibition framework.

**Evidence:** No significant EC50 difference between high and low absorption features. This is consistent with the mechanism.

---

## Unexpected Signals: The Data Reveals More Than We Asked

### 1. Precision Is Universally Near-Perfect --- A Structural Property, Not a Coincidence

**Observation:** At k=5, 21/26 features at L4 and 25/26 at L8 have precision = 1.0. The precision standard deviation (0.054 at L4, 0.028 at L8) is tiny compared to recall std (0.199 at L4, 0.192 at L8).

**Mini-hypothesis:** The SAE's L1 sparsity penalty creates a hard threshold: a latent either fires correctly or not at all. There is no mechanism in the SAE objective that would cause a latent to fire incorrectly --- false positives would require the decoder direction to align with wrong concepts, which the reconstruction loss prevents.

**Inhibition connection:** Competitive suppression does not introduce false positives because the suppression is from child to parent, not from unrelated concepts. The child fires correctly for the parent's concept; it just "steals" the activation.

**Significance:** This suggests SAEs are fundamentally *selective* even when not *comprehensive*. The field has focused on coverage (recall) metrics like explained variance, but selectivity (precision) may be the more robust property. The inhibition framework provides a mechanistic explanation for this robustness.

### 2. Random Baseline Success Is Non-Negligible --- Revealing Decoder Correlation Structure

**Observation:** Random feature steering achieves 34.4% success at L4 and 37.9% at L8.

**Mini-hypothesis:** SAE decoder directions are not orthogonal. Any random direction has some projection onto letter-related concepts because the SAE has learned a structured representation where related concepts are geometrically proximate.

**Inhibition connection:** This non-orthogonality IS the inhibition graph. The decoder correlation matrix W_dec^T W_dec captures exactly this geometric structure. The fact that random directions have non-zero task-specific projection is direct evidence that decoder correlations encode semantic relationships.

**Significance:** This is the key insight behind both H1b and the inhibition graph. The non-zero random baseline means raw steering metrics measure generic SAE direction effects + feature-specific effects. The inhibition graph provides a principled way to model and correct for these generic effects.

### 3. GPT-2 Small Shows Lower Absorption Than Expected --- Scale Threshold Hypothesis

**Observation:** Only 23-31% of features show any absorption, and the maximum rate is 24.2% --- well below the >50% "HIGH" threshold from Chanin et al.

**Mini-hypothesis:** GPT-2 Small (124M params) may be too small to exhibit strong absorption. Absorption may be a scale-dependent phenomenon that emerges more strongly in models with richer feature hierarchies.

**Inhibition connection:** Larger models have deeper, richer feature hierarchies, which means stronger competitive suppression and more absorption. The inhibition framework predicts that absorption severity scales with hierarchical complexity.

**Significance:** This means the GPT-2 Small results are a *lower bound* on the true effect. Testing on Gemma-2-2B (2B) or larger models should show stronger inhibition structure and more pronounced absorption consequences.

### 4. Cross-Model Validation (Pythia-70M) Shows Comparable Absorption but Null Correlation

**Observation:** Pythia-70M shows comparable absorption rates (mean 8.2%, max 27.2%) but no significant correlation with steering (r = -0.041, p = 0.841).

**Mini-hypothesis:** Pythia-70M (19M effective parameters) is too small for reliable steering measurement. Steering effects are weak on small models, so the signal-to-noise ratio is too low.

**Inhibition connection:** The inhibition framework predicts that the effect size depends on both absorption rate AND model scale (hierarchical complexity). Pythia-70M has comparable absorption rates but weaker hierarchical structure, so the downstream consequences are muted.

**Significance:** This supports the scale threshold hypothesis and means the GPT-2 Small H1b result may be a lower bound.

### 5. Spearman > Pearson for H1b Suggests Non-Linear Threshold Effect

**Observation:** At L8, Spearman rho = -0.502 vs Pearson r = -0.431. The Spearman correlation is stronger and more significant (p = 0.009 vs p = 0.028).

**Mini-hypothesis:** The relationship between absorption and steering degradation is monotonic but non-linear. Below some threshold, absorption has minimal effect; above the threshold, the effect accelerates.

**Inhibition connection:** In competitive networks, inhibition effects are often threshold-dependent. Weak inhibition may be overcome by input drive; strong inhibition dominates. This predicts a sigmoidal or step-function relationship.

**Significance:** A non-linear relationship is more interesting than a linear one. It suggests there may be a critical absorption rate above which consequences become severe --- a testable prediction for the inhibition framework.

---

## Follow-Up Experiments for the Inhibition Graph

| Signal | Follow-Up Experiment | Expected Outcome | GPU Hours | Priority |
|--------|---------------------|-------------------|-----------|----------|
| H6 (graph predicts absorption) | Construct inhibition graph for GPT-2 Small L8 (24K latents, k=20); validate precision@20 against Chanin absorption pairs | Precision@20 >= 0.10 (vs ~0.004 chance = 25x enrichment); Fisher exact p < 0.001 | ~0.5 hr | **Critical** |
| H7 (inhibition explains precision-recall) | Compute total incoming inhibition per feature; correlate with recall and precision | r(recall, inhibition) < -0.3, p < 0.05; r(precision, inhibition) ~ 0, p > 0.05 | ~0.5 hr | **Critical** |
| H8 (graph predicts at-risk features) | Compute total_inhibition for all 26 first-letter features; correlate with absorption rate | r > 0.3, p < 0.05; top quartile has >2x absorption rate vs bottom | ~0.5 hr | **High** |
| H9 (layer-dependent structure) | Construct graphs for L0, L4, L8, L10; compare mean edge weight, density, clustering | Mean edge weight increases with depth; r(mean_weight, layer) > 0.3; L8 has strongest structure | ~1 hr | **High** |
| H10 (homeostatic rebalancing) | Apply z'_i = z_i + alpha * inh_i; test parent firing restoration + reconstruction constraint | Parent firing +20%; recon error increase < 5%; optimal alpha in [0.5, 2.0] | ~1 hr | Medium |
| Cross-model validation | Replicate H6-H8 on Gemma-2-2B (layer 12, 16K dict) | Stronger effects (higher precision@20, stronger correlations) due to richer hierarchies | ~2 hr | **High** |
| Scale threshold | Test on Pythia-160M (layer 8) vs Pythia-70M | Significant correlation emerges where 70M showed none | ~2 hr | Medium |
| Semantic features | Test H6-H7 on WordNet hierarchy (animal->dog->poodle) | Precision invariance holds; recall correlates with hierarchy depth | ~4 hr | Medium |
| Alternative architectures | Compare res-jb vs JumpReLU vs gated SAE on same model/layer | JumpReLU shows higher absorption but same precision invariance pattern | ~3 hr | Low |

**Falsifiability criteria:**
- If H6 fails (precision@20 <= 0.05), the structural correspondence between decoder correlations and absorption pairs fails empirically. Fallback: report as finding about decoder correlation limitations; pivot to trade-off analysis (cand_c).
- If H7 fails (inhibition correlates with precision), the competitive suppression mechanism is incomplete. Fallback: more complex inhibition model needed.
- If H10 fails (recon error > 5% or no parent firing improvement), drop repair claims; diagnostic contribution stands independently.

**What each follow-up would show in the paper:**
- H6 validation -> Figure 2: "Inhibition graph edges predict absorption pairs"
- H7 validation -> Figure 3: "Competitive suppression explains precision-recall asymmetry"
- H8 validation -> Figure 4: "Graph-based at-risk feature identification"
- H9 validation -> Figure 5: "Layer-dependent inhibition structure"
- H10 validation -> Figure 6: "Homeostatic rebalancing restores parent firing"
- Cross-model -> Figure 7: "Validation across model families"

---

## Honest Caveats

### The Structural Correspondence (W_dec^T W_dec = G_LCA)

**Counter-argument:** The correspondence is mathematically exact only for tied encoder/decoder SAEs. Many modern SAEs use untied weights (separate encoder and decoder). The correspondence may not generalize.

**Alternative explanation:** Even for untied SAEs, decoder correlations may still encode semantic relationships, just not exactly the LCA inhibition matrix. The framework would need modification.

**What would convince me:** Testing H6 on both tied (res-jb) and untied (JumpReLU, gated) SAEs. If precision@20 is significantly above chance for both, the framework generalizes. If only for tied SAEs, the claim must be qualified.

### H1b Does Not Survive Multiple Comparison Correction

**Counter-argument:** The H1b result at L8 (p = 0.028 Pearson, p = 0.009 Spearman) does not survive Bonferroni (p = 0.334) or BH-FDR (q = 0.107). With 12 tests, this could be a false positive.

**Alternative explanation:** The correlation could be driven by a small number of outlier features. Feature U (24.2% absorption, 100% success) and Feature H (19.0% absorption, 55% success) have very different success rates.

**What would convince me:** Replication on an independent sample (different random seed, different prompt set) with n=26 showing r < -0.4, p < 0.05. Or replication on Gemma-2-2B with a stronger effect. The inhibition graph experiments (H6-H8) are independent tests that do not rely on H1b.

### The Inhibition Graph Has Not Been Constructed Yet

**Counter-argument:** All the evidence above is indirect. The core claim --- that decoder correlation edges predict absorption pairs --- has not been tested. The framework could fail empirically even if the math is elegant.

**Alternative explanation:** Decoder correlations may encode many types of relationships (synonymy, antonymy, co-occurrence), not just parent-child absorption. The signal-to-noise ratio may be too low.

**What would convince me:** H6 experiment showing precision@20 >= 0.10. This is a clear, falsifiable threshold. If the graph does not predict absorption pairs, the framework's central claim fails.

### Precision Invariance May Be Task-Specific

**Counter-argument:** The first-letter task may be too easy --- any latent that fires for "starts with A" is almost definitionally correct. Semantic features with more ambiguity might show precision degradation.

**Alternative explanation:** The k-sparse probe training may select latents that are inherently precise, making the probe reflect SAE structure rather than measuring anything independent.

**What would convince me:** Precision invariance replicated on semantic hierarchy features (WordNet) where child concepts are semantically similar to parents and could plausibly cause false positives.

### The Neuroscience Connection May Be Criticized as Superficial

**Counter-argument:** Reviewers may view the LCA connection as "neuroscience-washing" --- borrowing terminology without adding substance.

**Alternative explanation:** The connection is exact (W_dec^T W_dec = G_LCA), not metaphorical. But if the graph does not predict absorption, the connection is merely a mathematical curiosity.

**What would convince me:** H6 validation. If decoder correlations predict absorption, the neuroscience connection provides explanatory power, not just terminology.

---

## Bottom Line

There is a **genuinely novel and potentially publishable story** here, but it hinges on experiments that have not yet been run. The prior data provides strong *indirect* support for the inhibition framework: precision invariance, delta-corrected steering, layer-dependence, and decoder-activation decoupling all have natural explanations in competitive suppression. The novelty verification confirms no prior work has made the LCA-SAE connection. However, the **critical test** (H6: graph edges predict absorption pairs) remains unvalidated.

**The optimistic case:** If H6 validates (precision@20 >= 0.10), the paper has four firsts: (1) first LCA-SAE connection, (2) first local inhibition graph for SAE diagnostics, (3) first mechanistic explanation for precision-recall asymmetry, (4) first training-free repair. This is a strong theoretical contribution with practical utility.

**The pessimistic case:** If H6 fails (precision@20 <= 0.05), the framework's central claim collapses. Fallback: pivot to trade-off analysis (cand_c) or write a null-result cautionary tale.

**My assessment:** The indirect evidence is strong enough to justify running the H6-H8 experiments (~2 GPU hours). The risk is bounded (if it fails, we pivot), and the potential upside is significant. The mathematical exactness of the correspondence (W_dec^T W_dec = G_LCA) makes this a genuinely novel theoretical contribution if empirically validated.
