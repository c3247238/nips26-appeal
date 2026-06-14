# Skeptic Analysis: Local Inhibition Graph Proposal

## Statistical Risk Inventory

### Risk 1: The "Significant" H1b Result Fails Multiple Comparison Correction

**The number:** H1b at layer 8: Pearson p=0.028, Spearman p=0.009.

**Why it's unreliable:** With 12 statistical tests performed (4 hypotheses x 2 layers x 2 metrics), the Bonferroni-corrected alpha is 0.00417. The Spearman p=0.009 for H1b does NOT survive Bonferroni (corrected p=0.107) or Benjamini-Hochberg FDR (q=0.107). **Zero hypotheses survive proper multiple comparison correction.** The proposal correctly notes this in its "Evidence-Driven Revisions" section (line 20: "No hypothesis survives multiple comparison correction"), yet the new H1 (graph prediction) is pitched as a "clear positive claim with clear significance threshold." This is ironic --- the same statistical framework that produced the uncorrected p=0.028 (now deemed insufficient) is being replaced by another framework whose significance threshold (precision@20 > 0.10) has not been validated.

**Severity: SERIOUS CONCERN.** The pattern of overinterpreting uncorrected p-values is a recurring issue. The proposal's H1 threshold (precision@20 >= 0.10 vs 0.004 chance) is a precision metric, not a p-value, but the underlying statistical reasoning is similarly fragile.

### Risk 2: n=26 Features with Extreme Variance Imbalance

**The number:** 26 first-letter features, of which only 4-6 show >10% absorption at any layer.

**Why it's unreliable:** With n=26 and only 4-6 "high absorption" features, the effective sample size for correlation analysis is tiny. The power to detect a medium effect (r=0.5) at alpha=0.05 is approximately 60% --- and that's for the full n=26. For subgroup analyses (high vs low absorption), n drops to ~5 vs ~21, making any comparison severely underpowered. The EC50 analysis compounds this: Layer 4 has only 24/26 valid EC50 values, and the high-absorption group has n=5-6 features.

The pilot summary (line 78) states: "Features with >10% absorption: 4/5" --- but this is from the pilot (layer 8, 50 samples). The full experiment shows layer 4 has 6 medium-absorption features, layer 8 has 4. The variance is layer-dependent and unstable.

**Severity: FATAL FLAW for correlation claims.** The n=26 sample is too small to support the proposed H3 (correlation between total inhibition and absorption rate: r > 0.3, p < 0.05). With only 4-6 high-absorption features, detecting r=0.3 requires n=84 for 80% power. The proposed experiment is underpowered by design.

### Risk 3: The "Precision Invariance" Claim Rests on a Ceiling Effect

**The number:** precision_recall_analysis.json shows precision_mean at k=5: 0.975 (L4) and 0.995 (L8); n_precision_one = 21-25/26.

**Why it's unreliable:** With 21-25 of 26 features showing precision=1.0 at k>=5, there is virtually no variance in precision to correlate with anything. The claim "precision is invariant to absorption" is tautological --- when 85-96% of features have perfect precision, there's no signal to detect. The proposal reframes this as "inhibition explains precision invariance," but the simpler explanation is a ceiling effect in the probing metric.

The full_activation_probing_l4.json shows mean_full_f1 = 1.0 with ALL 26 features achieving F1=1.0 when using all 24,576 latents. The k-sparse probing (k=5) is an artificial constraint that creates variance, but this variance is driven by feature-specific probe quality, not absorption.

**Severity: SERIOUS CONCERN.** The precision-recall asymmetry may be an artifact of the probing methodology, not a real phenomenon requiring theoretical explanation.

---

## Alternative Explanations

### For H1b (Delta-Corrected Correlation at Layer 8)

**Alternative: The delta correction introduces a spurious correlation through mathematical coupling.**

The delta metric is defined as: delta = feature_steering - random_baseline_steering. If random baseline steering has higher variance for features that also happen to have higher absorption (e.g., due to feature-specific prompt sensitivity), the delta correction could create artificial correlation. The random baseline uses the SAME prompts with a random latent --- if certain prompts are inherently harder to steer (regardless of latent), and those prompts happen to correspond to high-absorption features, the delta correction amplifies a confound.

**Evidence:** Layer 4 shows delta_r = +0.245 (positive, wrong direction), while layer 8 shows delta_r = -0.431. If the inhibition mechanism is real, why does it reverse sign across layers? The proposal claims "deeper layers have stronger hierarchical structure = stronger inhibition," but this is post-hoc rationalization.

### For Precision-Recall Asymmetry

**Alternative: The asymmetry is a measurement artifact of k-sparse probing, not a mechanistic phenomenon.**

k-sparse probing selects the top-k latents by activation magnitude. Precision = (true positives) / (selected latents). With k=5 and only 1 true parent latent, precision is 1.0 if the parent is in the top-5, regardless of absorption. Recall = (true positives) / (all relevant latents). Since "all relevant latents" is defined as the parent latent only, recall is binary (parent in top-k or not). The "variance" in recall comes from whether the parent makes the top-k, which depends on probe quality, not absorption.

The full_activation_probing shows ALL features achieve F1=1.0 with full latents. This means the parent latent IS detectable --- the k-sparse constraint is what creates the "recall loss." This is a methodological choice, not a biological phenomenon.

### For the LCA-W_dec^T W_dec Correspondence

**Alternative: The correspondence is mathematically trivial and provides no explanatory power.**

The proposal claims "W_dec^T W_dec = G_LCA" is "exact, not metaphorical." But any matrix of inner products between vectors is an "inhibition matrix" in the LCA framework. The LCA update rule is: dz/dt = (1/tau)(W^T x - z - G*z). Setting G = W_dec^T W_dec is a definitional choice, not a derived result. The question is whether this choice explains anything beyond the definition.

The proposal does not derive any novel predictions from this correspondence that are not already obvious from the definition of decoder correlations. "Features with correlated decoders compete" is not a prediction --- it's the definition of the graph.

---

## Proxy Metric Audit

| Claimed Contribution | What We Actually Measure | Gap |
|---|---|---|
| "Graph edges predict absorption pairs" | Precision@k of decoder-correlation neighbors against Chanin absorption pairs | The Chanin metric itself is controversial; validating against it validates a metric, not a mechanism |
| "Inhibition explains precision-recall asymmetry" | Precision variance << Recall variance in k-sparse probing | Ceiling effect in precision; recall variance may be probe-quality artifact |
| "Graph predicts at-risk features" | Correlation between total incoming inhibition and absorption rate | Both are derived from the same decoder matrix; this is circular |
| "Training-free diagnostic" | Top-k neighbors from W_dec^T W_dec | Any SAE analysis tool computes this; the "diagnostic" value is untested |
| "Mechanistic explanation for absorption" | Mathematical analogy to LCA | No causal mechanism is demonstrated; analogy does not imply mechanism |

**Critical gap:** The proposal claims to explain "why absorption affects recall but not precision" (H2), but the experiments do not actually measure "absorption affecting recall." They measure k-sparse probing recall, which is a function of (a) probe quality, (b) k value, and (c) feature activation magnitude. None of these are directly linked to the inhibition mechanism.

---

## Severity Classification

### Fatal Flaws

1. **H3 is statistically impossible with current data.** The proposal claims "correlation between total incoming inhibition and absorption rate: r > 0.3, p < 0.05." With n=26 and 4-6 high-absorption features, power to detect r=0.3 is ~25%. The experiment is underpowered by design. Even if the effect exists, it won't be detectable.

2. **Circular reasoning in H3.** "Total incoming inhibition" is computed from decoder correlations. "Absorption rate" is computed from decoder correlations (via the Chanin metric, which uses decoder geometry). Correlating two quantities derived from the same matrix is not independent validation --- it's internal consistency checking.

### Serious Concerns

3. **The LCA correspondence is a rebranding, not a discovery.** Setting G = W_dec^T W_dec in the LCA framework is a definitional move. No new predictions follow. The proposal's "mechanistic explanation" is: "inhibition suppresses parent activation." This is restating "absorption means parent doesn't fire" in different words.

4. **H1 precision@20 threshold is arbitrary.** The threshold of 0.10 (vs 0.004 chance) is not derived from any theory. If precision@20 = 0.05, the proposal says "proceed with diagnostic-only claims." But 0.05 vs 0.10 is a 2x difference --- why is one "partial validation" and the other "failure"? The decision tree is designed to always yield a "proceed" outcome.

5. **Homeostatic rebalancing (H5) has no theoretical justification.** The update rule z'_i = z_i + alpha * inh_i adds activation proportional to neighbor activation. If child features suppress parents, adding MORE suppression (inh_i is positive when neighbors fire) would further suppress, not restore. The sign in the formula appears wrong: if G_ij > 0 (correlated decoders) and z_j > 0 (child fires), then inh_i > 0, so z'_i = z_i + alpha * inh_i INCREASES parent activation. But this is not "homeostatic rebalancing" --- it's arbitrary activation boosting. The constraint "reconstruction error < 5%" is a post-hoc fix; if the boost degrades reconstruction, the mechanism fails.

### Minor Caveats

6. **The cross-model validation (Gemma-2-2B) is listed as "High" likelihood of failure.** The proposal acknowledges this but does not adjust its claims accordingly.

7. **The "training-free" claim is technically true but misleading.** Computing W_dec^T W_dec for 24K latents requires ~2.4B operations and significant memory. It's "training-free" but not "computation-free."

---

## Concrete Remediation

### For Fatal Flaw 1 (H3 Underpowered)

**Experiment:** Expand the feature set from 26 first-letter features to 200+ features using WordNet hierarchies (animal -> dog -> poodle, vehicle -> car -> sedan). This increases n for correlation analysis and tests generalization beyond first-letter artifacts.

**Expected outcome:** If the inhibition mechanism is real, the correlation should strengthen with more features. If it disappears, the original finding was a first-letter artifact.

**Dataset:** WordNet noun hierarchies, selecting parent-child pairs at depths 3-5.

### For Fatal Flaw 2 (Circular H3)

**Experiment:** Use an independent absorption metric that does NOT rely on decoder correlations. For example, use the SAEBench ablation-based absorption metric (measure reconstruction change when parent vs child is ablated). Compare this against the inhibition graph.

**Expected outcome:** If decoder-correlation inhibition truly predicts absorption, the correlation should hold with an independent absorption measure. If not, the "prediction" is circular.

**Dataset:** Same GPT-2 Small SAE, SAEBench ablation metric.

### For Serious Concern 3 (LCA as Rebranding)

**Experiment:** Derive a quantitative prediction from the LCA framework that is NOT obvious from the definition. For example: in LCA, the steady-state activation is z* = (W^T W + G)^{-1} W^T x. If G = W_dec^T W_dec, then the steady-state should satisfy a specific relationship between parent and child activations. Test whether this relationship holds in the SAE's actual activations.

**Expected outcome:** If the LCA correspondence is more than analogy, the predicted activation relationship should match empirical activations within some tolerance. If not, the correspondence is purely formal.

**Dataset:** Activation traces from GPT-2 Small on first-letter prompts, comparing predicted vs actual parent-child activation ratios.

### For Serious Concern 4 (Arbitrary H1 Threshold)

**Experiment:** Compute the null distribution of precision@k by random permutation (not just the expected value). Report the p-value of the observed precision@k against this null. Set the threshold based on statistical significance (e.g., p < 0.001 for Bonferroni correction across k values), not an arbitrary precision value.

**Expected outcome:** A properly controlled H1 with permutation testing and multiple comparison correction. If precision@20 = 0.10 is truly significant, it will survive. If not, the claim is unsupported.

**Dataset:** Same graph construction, 10,000 random permutations of latent indices.

### For Serious Concern 5 (H5 Sign Error)

**Experiment:** Before running H5, fix the update rule. If inhibition suppresses parent activation, the rebalancing should REDUCE neighbor activation, not boost parent activation. Test both signs: z'_i = z_i - alpha * inh_i (subtractive) and z'_i = z_i + alpha * inh_i (additive, current proposal). Compare which restores parent firing without degrading reconstruction.

**Expected outcome:** If the subtractive rule works, the inhibition framework has predictive power. If neither works, the repair mechanism is infeasible.

**Dataset:** Same GPT-2 Small SAE, alpha sweep [0.01, 0.1, 1.0, 10.0].

---

## Summary Verdict

The Local Inhibition Graph proposal is intellectually appealing but statistically fragile. Its core claims rest on:

1. **An uncorrected p-value** (H1b p=0.028) that the proposal itself acknowledges does not survive multiple comparison correction.
2. **An underpowered design** (n=26, 4-6 high-absorption features) for the proposed H3 correlation.
3. **Circular reasoning** (correlating two quantities derived from the same matrix).
4. **A mathematical analogy** (LCA correspondence) that has not been shown to yield novel predictions.
5. **A likely sign error** in the homeostatic rebalancing formula.

The proposal's decision tree (lines 298-319) is designed to always yield "PROCEED" --- even H1 "not validated" leads to "retain as theoretical speculation in Discussion." This is not a falsifiable research program.

**Recommendation:** Before proceeding, run the remediation experiments above. If H1 with proper permutation testing and H3 with independent absorption metric both fail, the framework has no empirical foundation and should be abandoned in favor of the Contrarian's Alternative C (Trade-off Analysis), which at least makes descriptive claims that do not require mechanistic validation.
